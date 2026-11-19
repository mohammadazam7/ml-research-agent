"""Baselines every model has to beat, scored with the metric the plan chose.

Two baselines run on every task:

- ``prior``: ignores the features. For regression it predicts the training median (the
  constant that minimises absolute error); for classification it scores every row with
  the training class shares, so it predicts the majority class.
- ``linear``: ridge regression on the usable numeric, boolean and categorical columns,
  standardised, with missing values filled from the training set. For classification it
  fits one ridge model per class on 0/1 targets and uses the outputs as scores. It is
  not a tuned model; it is the cheapest model that uses the features at all.

A result the agent reports later is only worth something if it clears both. If the
linear baseline is close to the prior, the features carry little signal; if it is far
above everything else, it is worth checking for a leak before celebrating.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from ml_research_agent.plan import Plan, Split
from ml_research_agent.profile import Kind, Profile, Task

# Categories seen fewer times than this in training share one "other" column.
MIN_CATEGORY_ROWS = 5
RIDGE = 1.0

HIGHER_IS_BETTER = {"roc_auc": True, "average_precision": True, "macro_f1": True, "mae": False}


class BaselineError(ValueError):
    pass


@dataclass(frozen=True)
class Score:
    baseline: str
    metric: str
    value: float
    train_rows: int
    test_rows: int
    features: int

    @property
    def higher_is_better(self) -> bool:
        return HIGHER_IS_BETTER[self.metric]


# Metrics. Each takes the true values and the model's output for the test rows.


def mae(y: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(y - predicted)))


def roc_auc(positive: np.ndarray, score: np.ndarray) -> float:
    """Probability that a random positive outscores a random negative; ties count half."""
    pos = score[positive]
    neg = score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        raise BaselineError("ROC AUC needs both classes in the test set")
    ranks = _average_ranks(score)
    return float((ranks[positive].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(positive: np.ndarray, score: np.ndarray) -> float:
    """Mean precision at the rank of each positive, with tied scores taken as one block."""
    total = int(positive.sum())
    if total == 0:
        raise BaselineError("average precision needs a positive row in the test set")
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_pos = positive[order]
    # Precision is evaluated only at the end of each block of equal scores.
    ends = np.r_[np.nonzero(np.diff(sorted_score))[0], len(score) - 1]
    hits = np.cumsum(sorted_pos)[ends]
    seen = ends + 1
    new_hits = np.diff(np.r_[0, hits])
    return float(np.sum(new_hits * hits / seen) / total)


def macro_f1(y: np.ndarray, predicted: np.ndarray, labels: Sequence[str]) -> float:
    scores = []
    for label in labels:
        tp = np.sum((predicted == label) & (y == label))
        fp = np.sum((predicted == label) & (y != label))
        fn = np.sum((predicted != label) & (y == label))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    return ranks


# Features.


def _design(
    table: pa.Table, profile: Profile, train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardised feature matrices for the training and test rows.

    Every statistic (fill values, scales, category lists) comes from the training rows
    only, so nothing about the test rows reaches the model.
    """
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for column in profile.columns:
        if column.name == profile.target.name or not column.usable_as_feature:
            continue
        values = table[column.name]
        if column.kind in {Kind.NUMERIC, Kind.BOOLEAN}:
            x = np.array(values.to_pylist(), dtype=float)
            fill = np.nanmedian(x[train]) if np.any(~np.isnan(x[train])) else 0.0
            x = np.where(np.isnan(x), fill, x)
            centre, scale = x[train].mean(), x[train].std()
            x = (x - centre) / (scale if scale > 0 else 1.0)
            train_parts.append(x[train, None])
            test_parts.append(x[test, None])
        elif column.kind is Kind.CATEGORICAL:
            raw = np.array([None if v is None else str(v) for v in values.to_pylist()], object)
            kept, counts = np.unique(raw[train].astype(str), return_counts=True)
            kept = kept[counts >= MIN_CATEGORY_ROWS]
            onehot = (raw[:, None].astype(str) == kept[None, :]).astype(float)
            onehot = np.c_[onehot, onehot.sum(axis=1) == 0]
            train_parts.append(onehot[train])
            test_parts.append(onehot[test])
    if not train_parts:
        raise BaselineError("no usable numeric, boolean or categorical feature columns")
    return np.hstack(train_parts), np.hstack(test_parts)


def _ridge(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Ridge regression with an unpenalised intercept; ``y`` may have several columns."""
    centre = y.mean(axis=0)
    gram = x.T @ x + RIDGE * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, x.T @ (y - centre))
    return np.asarray(x_new @ weights + centre)


# Running both baselines.


def run(table: pa.Table, profile: Profile, chosen: Plan, rows: Split) -> list[Score]:
    """Score ``prior`` and ``linear`` on the plan's test rows with the plan's metric."""
    target = table[profile.target.name].to_pylist()
    train = np.array([r for r in rows.train if target[r] is not None])
    test = np.array([r for r in rows.test if target[r] is not None])
    if len(train) == 0 or len(test) == 0:
        raise BaselineError("every training or every test row is missing the target")
    x_train, x_test = _design(table, profile, train, test)
    task = profile.target.task

    def score(name: str, value: float, features: int) -> Score:
        return Score(name, chosen.metric, value, len(train), len(test), features)

    if task is Task.REGRESSION:
        y = np.array(target, dtype=float)
        prior = np.full(len(test), np.median(y[train]))
        linear = _ridge(x_train, y[train], x_test)
        return [
            score("prior", mae(y[test], prior), 0),
            score("linear", mae(y[test], linear), x_train.shape[1]),
        ]

    labels = np.array([None if v is None else str(v) for v in target], dtype=object)
    classes = sorted({str(v) for v in labels[train]})
    shares = np.array([np.mean(labels[train] == c) for c in classes])
    onehot = (labels[train][:, None] == np.array(classes, dtype=object)[None, :]).astype(float)
    linear_scores = _ridge(x_train, onehot, x_test)
    prior_scores = np.tile(shares, (len(test), 1))

    if chosen.metric == "macro_f1":
        names = np.array(classes, dtype=object)
        return [
            score("prior", macro_f1(labels[test], names[prior_scores.argmax(1)], classes), 0),
            score(
                "linear",
                macro_f1(labels[test], names[linear_scores.argmax(1)], classes),
                x_train.shape[1],
            ),
        ]

    # Two classes: the positive one is the rarer one in training, as the plan assumed.
    positive_index = int(np.argmin(shares))
    positive = labels[test] == classes[positive_index]
    metric = roc_auc if chosen.metric == "roc_auc" else average_precision
    return [
        score("prior", metric(positive, prior_scores[:, positive_index]), 0),
        score("linear", metric(positive, linear_scores[:, positive_index]), x_train.shape[1]),
    ]

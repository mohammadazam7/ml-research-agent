"""Choose how a model will be judged before any model is trained.

The plan fixes three things from the dataset profile: how rows are split, which metric
decides, and which rows are the held-out test set. Fixing them first, in code, removes
the commonest ways an evaluation flatters a model: a random split on data ordered in
time, the same customer on both sides of the split, and a metric picked after seeing the
results.

Priority when several strategies apply: time, then group, then random. A time column
means the model will be used on rows later than the ones it learned from, and no
shuffling should hide that.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

import pyarrow as pa
import pyarrow.compute as pc

from ml_research_agent.profile import Profile, Task

# Below this share for the rarer class, accuracy mostly measures the majority class.
IMBALANCE_SHARE = 0.2
MIN_TEST_ROWS = 20


class Strategy(Enum):
    TIME = "time"
    GROUP = "group"
    RANDOM = "random"


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    strategy: Strategy
    # The column that orders or groups rows; None for a random split.
    column: str | None
    metric: str
    test_share: float
    seed: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Split:
    """Row positions of the training and test sets. Together they cover every row once."""

    train: tuple[int, ...]
    test: tuple[int, ...]
    # For a time split: every test row is at or after this value, every training row before.
    cutoff: object | None = None


def _metric(profile: Profile) -> tuple[str, str]:
    target = profile.target
    if target.task is Task.REGRESSION:
        return "mae", "regression target: mean absolute error, in the target's own units"
    if target.task is Task.MULTICLASS:
        return "macro_f1", "several classes: macro F1 weighs each class equally"
    minority = target.minority_share or 0.0
    if minority < IMBALANCE_SHARE:
        return (
            "average_precision",
            f"the rarer class is {minority:.1%} of rows: average precision reflects how well "
            "it is found, where accuracy would reward predicting the majority",
        )
    return "roc_auc", "two classes of comparable size: area under the ROC curve"


def plan(profile: Profile, *, test_share: float = 0.2, seed: int = 0) -> Plan:
    """The evaluation plan for a profiled dataset, with the reasons for each choice."""
    if not 0 < test_share < 1:
        raise PlanError("test_share must be between 0 and 1")
    if profile.rows * test_share < MIN_TEST_ROWS:
        raise PlanError(
            f"{profile.rows} rows leave fewer than {MIN_TEST_ROWS} for testing at "
            f"a test share of {test_share:g}"
        )
    metric, metric_reason = _metric(profile)
    reasons = [metric_reason]
    if profile.time_columns:
        column = profile.time_columns[0]
        strategy = Strategy.TIME
        reasons.append(
            f"rows carry a time ({column}): train on the earlier rows and test on the "
            "latest, as the model would be used"
        )
        if profile.group_columns:
            reasons.append(
                f"{profile.group_columns[0]} repeats across rows; entities seen in "
                "training can reappear in the test period, which matches real use"
            )
        if len(profile.time_columns) > 1:
            others = ", ".join(profile.time_columns[1:])
            reasons.append(f"other time columns were not used to order rows: {others}")
    elif profile.group_columns:
        column = profile.group_columns[0]
        strategy = Strategy.GROUP
        reasons.append(
            f"{column} repeats across rows: all rows of one {column} go to the same side, "
            "so the model is tested on entities it has not seen"
        )
    else:
        column = None
        strategy = Strategy.RANDOM
        stratified = profile.target.task is not Task.REGRESSION
        reasons.append(
            "no time or entity column found: a seeded random split"
            + (", stratified by the target" if stratified else "")
        )
    if profile.duplicate_rows:
        reasons.append(
            f"{profile.duplicate_rows} exact duplicate rows: drop them before splitting, "
            "or copies of a test row will sit in training"
        )
    return Plan(strategy, column, metric, test_share, seed, tuple(reasons))


def _bucket(value: object, seed: int) -> float:
    """A stable number in [0, 1) for a value: the same on every machine and run."""
    digest = hashlib.sha256(f"{seed}:{value!r}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _time_split(table: pa.Table, column: str, test_share: float) -> Split:
    values = table[column]
    if values.null_count:
        raise PlanError(f"{column} has missing values; a time split needs every row dated")
    order = pc.sort_indices(values).to_pylist()
    position = int(len(order) * (1 - test_share))
    ordered = values.take(pa.array(order))
    cutoff = ordered[position].as_py()
    # Rows that share the cutoff time all go to the test side, so no time is split.
    while position > 0 and ordered[position - 1].as_py() == cutoff:
        position -= 1
    if position == 0:
        raise PlanError(f"{column} does not separate the rows: too many share one value")
    return Split(tuple(sorted(order[:position])), tuple(sorted(order[position:])), cutoff)


def _group_split(table: pa.Table, column: str, test_share: float, seed: int) -> Split:
    train: list[int] = []
    test: list[int] = []
    for row, value in enumerate(table[column].to_pylist()):
        (test if _bucket(value, seed) < test_share else train).append(row)
    return Split(tuple(train), tuple(test))


def _random_split(table: pa.Table, target: str | None, test_share: float, seed: int) -> Split:
    labels = table[target].to_pylist() if target else [None] * table.num_rows
    by_label: dict[object, list[int]] = {}
    for row, label in enumerate(labels):
        by_label.setdefault(label, []).append(row)
    train: list[int] = []
    test: list[int] = []
    for rows in by_label.values():
        # Rank rows by a hash and cut each class at the same share.
        ranked = sorted(rows, key=lambda r: _bucket(r, seed))
        cut = round(len(ranked) * test_share)
        test.extend(ranked[:cut])
        train.extend(ranked[cut:])
    return Split(tuple(sorted(train)), tuple(sorted(test)))


def split(table: pa.Table, profile: Profile, chosen: Plan) -> Split:
    """Apply ``chosen`` to ``table``. The same inputs always give the same rows."""
    if table.num_rows != profile.rows:
        raise PlanError("the table is not the one that was profiled")
    if chosen.strategy is Strategy.TIME:
        assert chosen.column is not None
        result = _time_split(table, chosen.column, chosen.test_share)
    elif chosen.strategy is Strategy.GROUP:
        assert chosen.column is not None
        result = _group_split(table, chosen.column, chosen.test_share, chosen.seed)
    else:
        stratify = profile.target.name if profile.target.task is not Task.REGRESSION else None
        result = _random_split(table, stratify, chosen.test_share, chosen.seed)
    if not result.train or not result.test:
        raise PlanError("the split left one side empty")
    return result

"""Look for leakage before a model is trained, using only plain code.

Leakage is information in the training data that the model will not have when it is
used. It makes evaluation scores look far better than real use will be. These checks
look for its usual forms:

``single_feature``
    One feature on its own predicts the target almost perfectly on held-out rows. Real
    problems rarely have such a feature; it is usually the target recorded another way
    (a refund date when predicting refunds, a status code that is set after the event).
``row_overlap``
    The same feature values with the same target appear on both sides of the split, so
    the test set partly measures memorisation.
``entity_overlap``
    The same entity (customer, patient) has rows on both sides of a split that was
    meant to keep entities apart.
``after_cutoff``
    In a time split, another date column holds values later than the cutoff in training
    rows: something that was only known after the prediction moment.

Every finding says what was measured, so a person can decide whether it is real.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import pyarrow as pa

from ml_research_agent.plan import Plan, Split, Strategy
from ml_research_agent.profile import Kind, Profile, Task

# A single feature at or above this held-out score is reported.
SUSPICIOUS_SCORE = 0.98


class Check(Enum):
    SINGLE_FEATURE = "single_feature"
    ROW_OVERLAP = "row_overlap"
    ENTITY_OVERLAP = "entity_overlap"
    AFTER_CUTOFF = "after_cutoff"


@dataclass(frozen=True)
class Finding:
    check: Check
    column: str | None
    value: float
    detail: str


@dataclass(frozen=True)
class Report:
    findings: tuple[Finding, ...]

    @property
    def clean(self) -> bool:
        return not self.findings

    def by_check(self, check: Check) -> list[Finding]:
        return [f for f in self.findings if f.check is check]


def _take(values: Sequence[object], rows: Iterable[int]) -> list[object]:
    return [values[i] for i in rows]


def _balanced_accuracy(truth: Sequence[object], predicted: Sequence[object]) -> float:
    """Mean recall over the classes present: 1 / k for a constant guess, 1 for perfect."""
    totals: Counter[object] = Counter(truth)
    hits: Counter[object] = Counter(t for t, p in zip(truth, predicted, strict=True) if t == p)
    return sum(hits[c] / n for c, n in totals.items()) / len(totals)


def _category_rule(train_x: Sequence[object], train_y: Sequence[object]) -> dict[object, object]:
    """For each feature value, the most common target seen with it in training."""
    seen: dict[object, Counter[object]] = {}
    for x, y in zip(train_x, train_y, strict=True):
        seen.setdefault(x, Counter())[y] += 1
    return {x: max(c.items(), key=lambda kv: (kv[1], str(kv[0])))[0] for x, c in seen.items()}


def _stump(train_x: Sequence[float], train_y: Sequence[object]) -> tuple[float, object, object]:
    """The single threshold, learned on training rows, that best splits two sides."""
    pairs = sorted(zip(train_x, train_y, strict=True), key=lambda p: p[0])
    total: Counter[object] = Counter(y for _, y in pairs)
    left: Counter[object] = Counter()
    best = (-1.0, pairs[0][0], pairs[0][1], pairs[0][1])
    for i in range(len(pairs) - 1):
        left[pairs[i][1]] += 1
        if pairs[i][0] == pairs[i + 1][0]:
            continue
        right = total - left
        left_label = max(left.items(), key=lambda kv: (kv[1], str(kv[0])))[0]
        right_label = max(right.items(), key=lambda kv: (kv[1], str(kv[0])))[0]
        predicted = [left_label] * (i + 1) + [right_label] * (len(pairs) - i - 1)
        score = _balanced_accuracy([y for _, y in pairs], predicted)
        if score > best[0]:
            best = (score, (pairs[i][0] + pairs[i + 1][0]) / 2, left_label, right_label)
    return best[1], best[2], best[3]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return 0.0 if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def _present(xs: Sequence[object], ys: Sequence[object]) -> tuple[list[object], list[object]]:
    kept = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    return [x for x, _ in kept], [y for _, y in kept]


def _single_feature(table: pa.Table, profile: Profile, split: Split) -> list[Finding]:
    target = table[profile.target.name].to_pylist()
    findings = []
    for col in profile.columns:
        if not col.usable_as_feature or col.kind in {Kind.TEXT, Kind.DATETIME}:
            continue
        values = table[col.name].to_pylist()
        train_x, train_y = _present(_take(values, split.train), _take(target, split.train))
        test_x, test_y = _present(_take(values, split.test), _take(target, split.test))
        if len(set(train_y)) < 2 or len(set(test_y)) < 2 or not test_x:
            continue
        if profile.target.task is Task.REGRESSION:
            if col.kind is not Kind.NUMERIC:
                continue
            score = abs(_pearson([float(x) for x in train_x], [float(y) for y in train_y]))  # type: ignore[arg-type]
            measure = "absolute correlation with the target on training rows"
        elif col.kind is Kind.NUMERIC:
            cut, low, high = _stump([float(x) for x in train_x], train_y)  # type: ignore[arg-type]
            predicted = [low if float(x) <= cut else high for x in test_x]  # type: ignore[arg-type]
            score = _balanced_accuracy(test_y, predicted)
            measure = f"held-out balanced accuracy of one threshold at {cut:g}"
        else:
            rule = _category_rule(train_x, train_y)
            fallback = Counter(train_y).most_common(1)[0][0]
            score = _balanced_accuracy(test_y, [rule.get(x, fallback) for x in test_x])
            measure = "held-out balanced accuracy of mapping each value to its usual target"
        if score >= SUSPICIOUS_SCORE:
            findings.append(
                Finding(
                    Check.SINGLE_FEATURE,
                    col.name,
                    score,
                    f"{col.name} alone scores {score:.3f} ({measure}); check whether it is "
                    "known before the prediction moment",
                )
            )
    return findings


def _row_overlap(table: pa.Table, profile: Profile, split: Split) -> list[Finding]:
    names = [c.name for c in profile.columns if c.usable_as_feature] + [profile.target.name]
    columns = [table[n].to_pylist() for n in names]
    train = {tuple(col[i] for col in columns) for i in split.train}
    shared = sum(1 for i in split.test if tuple(col[i] for col in columns) in train)
    if not shared:
        return []
    share = shared / len(split.test)
    return [
        Finding(
            Check.ROW_OVERLAP,
            None,
            share,
            f"{shared} test rows ({share:.1%}) also appear in training with the same "
            "features and target",
        )
    ]


def _entity_overlap(table: pa.Table, plan: Plan, split: Split, profile: Profile) -> list[Finding]:
    if plan.strategy is Strategy.TIME:
        # Entities recurring over time is expected when the model scores known customers.
        return []
    findings = []
    for name in profile.group_columns:
        values = table[name].to_pylist()
        train = {values[i] for i in split.train}
        test = {values[i] for i in split.test}
        both = train & test
        if both:
            share = len(both) / len(test)
            findings.append(
                Finding(
                    Check.ENTITY_OVERLAP,
                    name,
                    share,
                    f"{len(both)} values of {name} ({share:.1%} of those in test) have rows on "
                    "both sides; the model is tested partly on entities it has seen",
                )
            )
    return findings


def _after_cutoff(table: pa.Table, plan: Plan, split: Split, profile: Profile) -> list[Finding]:
    if plan.strategy is not Strategy.TIME or split.cutoff is None:
        return []
    findings = []
    for name in profile.time_columns:
        if name == plan.column:
            continue
        values = table[name].to_pylist()
        late = [i for i in split.train if values[i] is not None and values[i] > split.cutoff]
        if late:
            share = len(late) / len(split.train)
            findings.append(
                Finding(
                    Check.AFTER_CUTOFF,
                    name,
                    share,
                    f"{name} is later than the cutoff ({split.cutoff}) in {len(late)} training "
                    f"rows ({share:.1%}): it may be recorded after the outcome",
                )
            )
    return findings


def check(table: pa.Table, profile: Profile, plan: Plan, split: Split) -> Report:
    """Run every leakage check on a split that has not been used to train anything yet."""
    findings = (
        _single_feature(table, profile, split)
        + _row_overlap(table, profile, split)
        + _entity_overlap(table, plan, split, profile)
        + _after_cutoff(table, plan, split, profile)
    )
    return Report(tuple(findings))

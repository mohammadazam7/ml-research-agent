"""Calibration and slice reports for a binary classifier's scores on the test rows.

A single headline metric hides two things the agent has to look at before it trusts a
model: whether its scores can be read as probabilities, and whether it is much worse on
some part of the data than on the whole.

Calibration
    Test rows are put into equal-width score bins. For each bin the report gives the
    mean score, the observed positive rate and the row count. Expected calibration
    error (ECE) is the row-weighted mean gap between the two; the Brier score is the
    mean squared error of the scores as probabilities.

Slices
    For each categorical column given, the metric is recomputed on the rows with each
    value. A slice is flagged when it has at least ``MIN_SLICE_ROWS`` rows and its score
    is worse than the overall score by more than ``SLICE_GAP``. Small slices are listed
    but never flagged, because their scores are mostly noise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ml_research_agent.baselines import BaselineError

MIN_SLICE_ROWS = 30
SLICE_GAP = 0.05


@dataclass(frozen=True)
class Bin:
    low: float
    high: float
    rows: int
    mean_score: float
    positive_rate: float


@dataclass(frozen=True)
class Calibration:
    bins: tuple[Bin, ...]
    expected_calibration_error: float
    brier: float


@dataclass(frozen=True)
class Slice:
    column: str
    value: str
    rows: int
    score: float | None  # None when the metric is undefined on the slice (one class only)
    flagged: bool


def calibration(positive: np.ndarray, score: np.ndarray, bins: int = 10) -> Calibration:
    """Reliability bins, ECE and Brier score for scores in [0, 1]."""
    if len(score) == 0:
        raise BaselineError("calibration needs at least one row")
    if np.any(score < 0) or np.any(score > 1):
        raise BaselineError("calibration needs scores between 0 and 1")
    if bins < 1:
        raise BaselineError("bins must be at least 1")
    edges = np.linspace(0.0, 1.0, bins + 1)
    # A score of exactly 1.0 belongs in the last bin.
    index = np.minimum((score * bins).astype(int), bins - 1)
    out = []
    gap = 0.0
    for b in range(bins):
        rows = index == b
        n = int(rows.sum())
        if n == 0:
            continue
        mean_score = float(score[rows].mean())
        rate = float(positive[rows].mean())
        gap += n * abs(mean_score - rate)
        out.append(Bin(float(edges[b]), float(edges[b + 1]), n, mean_score, rate))
    brier = float(np.mean((score - positive.astype(float)) ** 2))
    return Calibration(tuple(out), gap / len(score), brier)


def slices(
    columns: dict[str, Sequence[object]],
    positive: np.ndarray,
    score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    higher_is_better: bool = True,
) -> tuple[float, list[Slice]]:
    """The overall metric and the metric on each value of each column, worst first."""
    overall = metric(positive, score)
    found = []
    for name, values in columns.items():
        if len(values) != len(score):
            raise BaselineError(f"{name} has {len(values)} values for {len(score)} rows")
        labels = np.array(["(missing)" if v is None else str(v) for v in values], dtype=object)
        for value in sorted(set(labels)):
            rows = labels == value
            n = int(rows.sum())
            try:
                value_score: float | None = metric(positive[rows], score[rows])
            except BaselineError:
                value_score = None
            worse = value_score is not None and (
                overall - value_score > SLICE_GAP
                if higher_is_better
                else value_score - overall > SLICE_GAP
            )
            found.append(Slice(name, value, n, value_score, n >= MIN_SLICE_ROWS and worse))

    def badness(s: Slice) -> float:
        if s.score is None:
            return -np.inf
        return overall - s.score if higher_is_better else s.score - overall

    return overall, sorted(found, key=badness, reverse=True)


def format_report(cal: Calibration, overall: float, found: list[Slice], metric: str) -> str:
    """A plain-text report the agent can read back and a person can skim."""
    lines = [
        f"Calibration: ECE {cal.expected_calibration_error:.3f}, Brier {cal.brier:.3f}",
        "  score bin    rows  mean score  positive rate",
    ]
    for b in cal.bins:
        span = f"{b.low:.1f} to {b.high:.1f}"
        lines.append(f"  {span}  {b.rows:5d}  {b.mean_score:10.3f}  {b.positive_rate:13.3f}")
    lines.append(f"Slices: overall {metric} {overall:.3f}")
    for s in found:
        shown = "n/a" if s.score is None else f"{s.score:.3f}"
        mark = "  worse" if s.flagged else ""
        lines.append(f"  {s.column}={s.value}  rows {s.rows}  {metric} {shown}{mark}")
    return "\n".join(lines)

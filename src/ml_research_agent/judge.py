"""The judge: what an experiment script is given, and how its output is scored.

An agent that writes its own training script could also write its own score. Here it
cannot. The judge hands the script one file in which the target is empty for every test
row, the script writes one prediction per test row, and the judge computes the plan's
metric against the truth it kept. The script never holds the test answers, so it cannot
tune on them or report a number it did not earn.

Columns the leakage checks flagged as giving the answer away (a single feature that
predicts the target almost perfectly, or a date column with values after the split
cutoff) are withheld from the file, and the assignment says which and why. A check can
be wrong about a feature that is legitimately that strong, so the caller can name
columns to keep; nothing is withheld silently.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from ml_research_agent.baselines import (
    HIGHER_IS_BETTER,
    BaselineError,
    average_precision,
    macro_f1,
    mae,
    roc_auc,
)
from ml_research_agent.leakage import Check, Report
from ml_research_agent.plan import Plan, Split
from ml_research_agent.profile import Profile, Task

SPLIT_COLUMN = "__split__"
PREDICTIONS_FILE = "predictions.json"
WITHHELD_FOR = {
    Check.SINGLE_FEATURE: "it predicts the target almost perfectly on its own",
    Check.AFTER_CUTOFF: "it holds dates after the split cutoff",
}


@dataclass(frozen=True)
class Verdict:
    """The judge's reading of one predictions file. ``value`` is set only when valid."""

    valid: bool
    metric: str
    value: float | None = None
    problem: str = ""


@dataclass(frozen=True)
class Withheld:
    column: str
    reason: str


class Judge:
    def __init__(
        self,
        table: pa.Table,
        profile: Profile,
        plan: Plan,
        split: Split,
        leakage: Report,
        *,
        keep: Collection[str] = (),
    ) -> None:
        self.profile = profile
        self.plan = plan
        self.metric = plan.metric
        target = table[profile.target.name].to_pylist()
        # Rows without a target can be neither learned from nor scored.
        self._train = [r for r in split.train if target[r] is not None]
        self._test = [r for r in split.test if target[r] is not None]
        if not self._train or not self._test:
            raise BaselineError("every training or every test row is missing the target")

        flagged: dict[str, str] = {}
        for finding in leakage.findings:
            if (
                finding.column is not None
                and finding.check in WITHHELD_FOR
                and finding.column not in keep
            ):
                flagged.setdefault(finding.column, WITHHELD_FOR[finding.check])
        self.withheld = tuple(Withheld(c, why) for c, why in sorted(flagged.items()))
        self.features = tuple(
            c.name
            for c in profile.columns
            if c.name != profile.target.name and c.usable_as_feature and c.name not in flagged
        )
        self._table = table

        if profile.target.task is Task.REGRESSION:
            self._truth: np.ndarray = np.array([target[r] for r in self._test], dtype=float)
            self.classes: tuple[str, ...] = ()
            self.positive: str | None = None
        else:
            labels = [str(target[r]) for r in self._train]
            self.classes = tuple(sorted(set(labels)))
            self._truth = np.array([str(target[r]) for r in self._test], dtype=object)
            # The positive class is the rarer one in training, as the baselines assume.
            shares = [labels.count(c) / len(labels) for c in self.classes]
            self.positive = (
                self.classes[int(np.argmin(shares))] if profile.target.task is Task.BINARY else None
            )

    @property
    def test_rows(self) -> int:
        return len(self._test)

    @property
    def train_rows(self) -> int:
        return len(self._train)

    @property
    def higher_is_better(self) -> bool:
        return HIGHER_IS_BETTER[self.metric]

    def write_task(self, path: Path) -> None:
        """Write the file a script works from: training rows first, then test rows with
        the target emptied, and a ``__split__`` column saying which is which."""
        name = self.profile.target.name
        rows = self._train + self._test
        columns = {c: self._table[c].take(rows) for c in self.features}
        target = self._table[name].take(self._train)
        hidden = pa.nulls(len(self._test), type=target.type)
        columns[name] = pa.chunked_array([*target.chunks, hidden], type=target.type)
        columns[SPLIT_COLUMN] = pa.array(["train"] * len(self._train) + ["test"] * len(self._test))
        path.parent.mkdir(parents=True, exist_ok=True)
        feather.write_feather(pa.table(columns), str(path))

    def assignment(self) -> str:
        """The task in words: what the file holds and what the script must write."""
        target = self.profile.target
        lines = [
            f"Task: {target.task.value}. Predict the column `{target.name}`.",
            f"Data: an Arrow feather file at the path in the environment variable "
            f"MLAGENT_DATA, with {self.train_rows} training rows and {self.test_rows} test "
            f"rows. The column `{SPLIT_COLUMN}` is 'train' or 'test'. `{target.name}` is "
            "empty for test rows.",
            "Feature columns: "
            + ", ".join(f"`{c}` ({self.profile.column(c).kind.value})" for c in self.features)
            + ".",
        ]
        for item in self.withheld:
            lines.append(f"The column `{item.column}` was withheld because {item.reason}.")
        lines.append(
            f"Split: {self.plan.strategy.value}. Metric: {self.metric} "
            f"({'higher' if self.higher_is_better else 'lower'} is better), computed by the "
            "judge from your predictions. A score you compute yourself is ignored."
        )
        if target.task is Task.REGRESSION:
            wanted = "one number per test row"
        elif self.metric == "macro_f1":
            wanted = "one label per test row, each one of: " + ", ".join(self.classes)
        else:
            wanted = (
                f"one number per test row, higher meaning more likely `{self.positive}` "
                f"(the classes are: {', '.join(self.classes)})"
            )
        lines.append(
            f"Output: write `{PREDICTIONS_FILE}` in the working directory, a JSON list with "
            f"{wanted}, in the order the test rows appear in the file."
        )
        lines.append("Only the Python standard library, numpy and pyarrow are available.")
        return "\n".join(lines)

    def score(self, predictions: Path) -> Verdict:
        """Read a predictions file and score it. Anything malformed is invalid, with the
        reason, and gets no value."""

        def invalid(problem: str) -> Verdict:
            return Verdict(False, self.metric, problem=problem)

        try:
            raw = json.loads(predictions.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return invalid(f"the script did not write {PREDICTIONS_FILE}")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return invalid(f"{PREDICTIONS_FILE} is not valid JSON")
        if not isinstance(raw, list):
            return invalid(f"{PREDICTIONS_FILE} must hold a JSON list")
        if len(raw) != self.test_rows:
            return invalid(f"expected {self.test_rows} predictions, got {len(raw)}")

        if self.metric == "macro_f1":
            if not all(isinstance(v, str) for v in raw):
                return invalid("every prediction must be a label given as a string")
            unknown = sorted({v for v in raw if v not in self.classes})
            if unknown:
                return invalid(f"unknown labels: {', '.join(unknown[:5])}")
            value = macro_f1(self._truth, np.array(raw, dtype=object), self.classes)
            return Verdict(True, self.metric, value)

        if not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in raw
        ):
            return invalid("every prediction must be a finite number")
        numbers = np.array(raw, dtype=float)
        if self.metric == "mae":
            return Verdict(True, self.metric, mae(self._truth, numbers))
        positive = self._truth == self.positive
        try:
            value = (roc_auc if self.metric == "roc_auc" else average_precision)(positive, numbers)
        except BaselineError as error:
            return invalid(str(error))
        return Verdict(True, self.metric, value)

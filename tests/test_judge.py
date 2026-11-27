import json
import textwrap
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from ml_research_agent import baselines, leakage
from ml_research_agent.judge import PREDICTIONS_FILE, SPLIT_COLUMN, Judge
from ml_research_agent.plan import plan, split
from ml_research_agent.profile import profile
from ml_research_agent.runner import Status, run_script

N = 600


def _judge(table: pa.Table, target: str) -> tuple[Judge, dict[str, float]]:
    p = profile(table, target)
    chosen = plan(p)
    rows = split(table, p, chosen)
    report = leakage.check(table, p, chosen, rows)
    scores = {s.baseline: s.value for s in baselines.run(table, p, chosen, rows)}
    return Judge(table, p, chosen, rows, report), scores


def _churn(leak: bool = False) -> pa.Table:
    rng = np.random.default_rng(0)
    usage = rng.normal(size=N)
    churned = (usage + 0.8 * rng.normal(size=N) < -0.3).astype(int)
    columns: dict[str, list[object]] = {
        "row_id": [f"r{i:04d}" for i in range(N)],
        "usage": usage.round(4).tolist(),
        "plan": [("basic", "plus", "pro")[i % 3] for i in range(N)],
        "churned": churned.tolist(),
    }
    if leak:
        columns["exit_survey"] = ["sent" if c else "none" for c in churned]
    return pa.table(columns)


def _prices() -> pa.Table:
    rng = np.random.default_rng(1)
    size = rng.uniform(40, 200, size=N)
    return pa.table(
        {
            "size": size.round(2).tolist(),
            "rooms": rng.integers(1, 6, size=N).tolist(),
            "price": (1000 * size + rng.normal(scale=8000, size=N)).round(0).tolist(),
        }
    )


def _species() -> pa.Table:
    rng = np.random.default_rng(2)
    kind = rng.integers(3, size=N)
    return pa.table(
        {
            "length": (kind * 2.0 + rng.normal(size=N)).round(3).tolist(),
            "species": [("ash", "birch", "cedar")[k] for k in kind],
        }
    )


def _write(tmp_path: Path, values: object) -> Path:
    path = tmp_path / PREDICTIONS_FILE
    path.write_text(json.dumps(values))
    return path


def test_the_task_file_hides_every_test_answer(tmp_path: Path) -> None:
    judge, _ = _judge(_churn(), "churned")
    judge.write_task(tmp_path / "task.feather")
    task = feather.read_table(str(tmp_path / "task.feather"))
    sides = task[SPLIT_COLUMN].to_pylist()
    answers = task["churned"].to_pylist()
    assert sides == ["train"] * judge.train_rows + ["test"] * judge.test_rows
    assert all(a is not None for a in answers[: judge.train_rows])
    assert all(a is None for a in answers[judge.train_rows :])
    assert judge.train_rows + judge.test_rows == N
    # The identifier is not a feature, so it is not in the file either.
    assert set(task.column_names) == {"usage", "plan", "churned", SPLIT_COLUMN}


def test_a_column_that_gives_the_answer_away_is_withheld_with_the_reason(tmp_path: Path) -> None:
    judge, _ = _judge(_churn(leak=True), "churned")
    assert [w.column for w in judge.withheld] == ["exit_survey"]
    assert "exit_survey" not in judge.features
    judge.write_task(tmp_path / "task.feather")
    assert "exit_survey" not in feather.read_table(str(tmp_path / "task.feather")).column_names
    assert "`exit_survey` was withheld because it predicts the target" in judge.assignment()


def test_scores_agree_with_the_metric_on_known_predictions(tmp_path: Path) -> None:
    judge, scores = _judge(_churn(), "churned")
    truth = (judge._truth == judge.positive).astype(float)
    assert judge.metric == "roc_auc" and judge.higher_is_better
    assert judge.score(_write(tmp_path, truth.tolist())).value == 1.0
    assert judge.score(_write(tmp_path, (1 - truth).tolist())).value == 0.0
    # A constant score ranks nothing: 0.5, the same as the prior baseline.
    constant = judge.score(_write(tmp_path, [0.3] * judge.test_rows))
    assert constant.valid and constant.value == pytest.approx(scores["prior"]) == 0.5


def test_regression_is_scored_in_the_units_of_the_target(tmp_path: Path) -> None:
    judge, _ = _judge(_prices(), "price")
    assert judge.metric == "mae" and not judge.higher_is_better
    truth = judge._truth
    assert judge.score(_write(tmp_path, truth.tolist())).value == 0.0
    assert judge.score(_write(tmp_path, (truth + 500).tolist())).value == pytest.approx(500)


def test_several_classes_are_scored_from_labels(tmp_path: Path) -> None:
    judge, _ = _judge(_species(), "species")
    assert judge.metric == "macro_f1" and judge.classes == ("ash", "birch", "cedar")
    assert judge.score(_write(tmp_path, judge._truth.tolist())).value == 1.0
    one_class = judge.score(_write(tmp_path, ["ash"] * judge.test_rows))
    assert one_class.valid and 0 < (one_class.value or 0) < 0.3
    assert "one label per test row, each one of: ash, birch, cedar" in judge.assignment()


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        (None, "did not write predictions.json"),
        ("not json", "not valid JSON"),
        ('{"a": 1}', "must hold a JSON list"),
        ("[0.1, 0.2]", "expected 120 predictions, got 2"),
        (json.dumps([None] * 120), "finite number"),
        (json.dumps([True] * 120), "finite number"),
        (json.dumps([0.5] * 119 + ["high"]), "finite number"),
        ("[" + ", ".join(["NaN"] * 120) + "]", "finite number"),
    ],
)
def test_malformed_predictions_are_invalid_and_get_no_value(
    tmp_path: Path, content: str | None, problem: str
) -> None:
    judge, _ = _judge(_churn(), "churned")
    assert judge.test_rows == 120
    path = tmp_path / PREDICTIONS_FILE
    if content is not None:
        path.write_text(content)
    verdict = judge.score(path)
    assert not verdict.valid and verdict.value is None
    assert problem in verdict.problem


def test_labels_must_be_known_strings(tmp_path: Path) -> None:
    judge, _ = _judge(_species(), "species")
    rows = judge.test_rows
    assert "unknown labels: oak" in judge.score(_write(tmp_path, ["oak"] * rows)).problem
    assert "label given as a string" in judge.score(_write(tmp_path, [1] * rows)).problem


def test_a_real_script_is_run_and_judged_end_to_end(tmp_path: Path) -> None:
    judge, scores = _judge(_prices(), "price")
    task = tmp_path / "task.feather"
    judge.write_task(task)
    script = tmp_path / "experiment.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, os
            import numpy as np
            import pyarrow.feather as feather

            table = feather.read_table(os.environ["MLAGENT_DATA"])
            side = np.array(table["__split__"].to_pylist())
            price = np.array(table["price"].to_pylist(), dtype=float)
            median = float(np.median(price[side == "train"]))
            json.dump([median] * int((side == "test").sum()), open("predictions.json", "w"))
            """
        )
    )
    kept = tmp_path / "out" / PREDICTIONS_FILE
    run = run_script(script, task, outputs={PREDICTIONS_FILE: kept}, require_metrics=False)
    assert run.status is Status.OK
    verdict = judge.score(kept)
    # Predicting the training median is exactly the prior baseline.
    assert verdict.valid and verdict.value == pytest.approx(scores["prior"])


def test_a_script_cannot_copy_the_answers_because_they_are_not_there(tmp_path: Path) -> None:
    judge, _ = _judge(_prices(), "price")
    task = tmp_path / "task.feather"
    judge.write_task(task)
    script = tmp_path / "experiment.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, os
            import pyarrow.feather as feather

            table = feather.read_table(os.environ["MLAGENT_DATA"]).to_pylist()
            answers = [row["price"] for row in table if row["__split__"] == "test"]
            json.dump(answers, open("predictions.json", "w"))
            """
        )
    )
    kept = tmp_path / PREDICTIONS_FILE
    run_script(script, task, outputs={PREDICTIONS_FILE: kept}, require_metrics=False)
    verdict = judge.score(kept)
    assert not verdict.valid and "finite number" in verdict.problem


def test_the_assignment_says_what_to_predict_and_how_it_is_scored() -> None:
    judge, _ = _judge(_churn(), "churned")
    text = judge.assignment()
    assert "Task: binary classification. Predict the column `churned`." in text
    assert "480 training rows and 120 test rows" in text
    assert "`usage` (numeric), `plan` (categorical)" in text
    assert "Metric: roc_auc (higher is better), computed by the judge" in text
    assert f"higher meaning more likely `{judge.positive}`" in text

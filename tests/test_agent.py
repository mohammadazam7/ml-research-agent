import textwrap
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from ml_research_agent.agent import AgentError, Budget, extract_script, run_agent
from ml_research_agent.providers import Reply, ScriptedProvider
from ml_research_agent.runner import Limits, RunLog, Status

N = 600


def _prices(leak: bool = False) -> pa.Table:
    """Price grows with the square of the size, so a straight line underfits."""
    rng = np.random.default_rng(3)
    size = rng.uniform(40, 200, size=N)
    price = 20 * size**2 + rng.normal(scale=60_000, size=N)
    columns = {"size": size.round(2).tolist(), "price": price.round(0).tolist()}
    if leak:
        columns["price_with_tax"] = (price * 1.13).round(0).tolist()
    return pa.table(columns)


def _reply(body: str, note: str = "Here is a script.") -> str:
    return f"{note}\n```python\n{textwrap.dedent(body).strip()}\n```"


HEADER = """
import json, os
import numpy as np
import pyarrow.feather as feather

table = feather.read_table(os.environ["MLAGENT_DATA"])
side = np.array(table["__split__"].to_pylist())
train, test = side == "train", side == "test"
size = np.array(table["size"].to_pylist(), dtype=float)
price = np.array(table["price"].to_pylist(), dtype=float)
"""
CRASH = _reply("raise RuntimeError('column widths do not match')")
MEDIAN = _reply(
    HEADER
    + """
json.dump([float(np.median(price[train]))] * int(test.sum()), open("predictions.json", "w"))
"""
)
QUADRATIC = _reply(
    HEADER
    + """
fit = np.polyfit(size[train], price[train], 2)
json.dump(np.polyval(fit, size[test]).tolist(), open("predictions.json", "w"))
"""
)
TOO_FEW = _reply('import json\njson.dump([1.0, 2.0], open("predictions.json", "w"))')
SLOW = _reply("import time\ntime.sleep(30)")


def test_the_loop_runs_scripts_judges_them_and_keeps_the_best(tmp_path: Path) -> None:
    provider = ScriptedProvider([CRASH, MEDIAN, QUADRATIC, "STOP"])
    result = run_agent(_prices(), "price", provider, workdir=tmp_path)

    assert [a.status for a in result.attempts] == ["failed", "scored", "scored"]
    assert result.stopped == "the model had nothing better to try"
    scores = {s.baseline: s.value for s in result.baselines}
    median, quadratic = result.attempts[1], result.attempts[2]
    # The median script is the prior baseline; the judge gives it the same number.
    assert median.value == pytest.approx(scores["prior"])
    assert quadratic.value is not None and quadratic.value < scores["linear"] < scores["prior"]
    assert result.best == quadratic and result.beats_baselines
    assert result.metric == "mae"


def test_the_model_is_told_what_happened_after_every_attempt(tmp_path: Path) -> None:
    provider = ScriptedProvider([CRASH, MEDIAN, "STOP"])
    run_agent(_prices(), "price", provider, workdir=tmp_path)
    first, second, third = provider.calls

    assert [m["role"] for m in first] == ["system", "user"]
    assert "Predict the column `price`" in first[1]["content"]
    assert "Baselines to beat: prior " in first[1]["content"]
    assert "A score you compute yourself is ignored" in first[1]["content"]

    after_crash = second[-1]["content"]
    assert second[-2] == {"role": "assistant", "content": CRASH}
    assert "Attempt 1: The script failed with exit code 1." in after_crash
    assert "RuntimeError: column widths do not match" in after_crash
    assert "No attempt has been scored yet." in after_crash
    assert "Experiments left: 4." in after_crash

    after_median = third[-1]["content"]
    assert "Attempt 2: mae " in after_median and "on the held-out rows" in after_median
    assert "(attempt 2)" in after_median


def test_a_result_that_only_matches_a_baseline_does_not_beat_it(tmp_path: Path) -> None:
    result = run_agent(_prices(), "price", ScriptedProvider([MEDIAN, "STOP"]), workdir=tmp_path)
    assert result.best is not None and result.best.status == "scored"
    assert not result.beats_baselines


def test_the_budget_of_experiments_ends_the_loop(tmp_path: Path) -> None:
    provider = ScriptedProvider([MEDIAN, QUADRATIC, MEDIAN])
    result = run_agent(_prices(), "price", provider, workdir=tmp_path, budget=Budget(experiments=2))
    assert len(result.attempts) == 2 and len(provider.calls) == 2
    assert result.stopped == "the budget of 2 experiments was used"


def test_the_token_budget_ends_the_loop_and_usage_is_added_up(tmp_path: Path) -> None:
    provider = ScriptedProvider([Reply(MEDIAN, 900, 200), Reply(QUADRATIC, 1500, 250)])
    result = run_agent(_prices(), "price", provider, workdir=tmp_path, budget=Budget(tokens=2000))
    # 1,100 tokens after the first call allows a second; 2,850 after it does not allow a third.
    assert len(result.attempts) == 2
    assert (result.input_tokens, result.output_tokens) == (2400, 450)
    assert result.stopped == "the budget of 2000 tokens was used"


def test_replies_without_a_script_and_unusable_predictions_count_as_attempts(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(["I would try a tree model next.", TOO_FEW, "STOP"])
    result = run_agent(_prices(), "price", provider, workdir=tmp_path)
    assert [a.status for a in result.attempts] == ["no_script", "invalid"]
    assert "Reply with one ```python block" in result.attempts[0].feedback
    assert "expected 120 predictions, got 2" in result.attempts[1].feedback
    assert result.best is None and not result.beats_baselines


def test_a_script_that_runs_too_long_is_stopped(tmp_path: Path) -> None:
    result = run_agent(
        _prices(),
        "price",
        ScriptedProvider([SLOW, "STOP"]),
        workdir=tmp_path,
        limits=Limits(wall_seconds=1.0),
    )
    assert [a.status for a in result.attempts] == ["timeout"]


def test_a_model_that_cannot_be_reached_ends_the_run_with_what_was_done(tmp_path: Path) -> None:
    result = run_agent(_prices(), "price", ScriptedProvider([MEDIAN]), workdir=tmp_path)
    assert len(result.attempts) == 1
    assert result.stopped == "the model could not be used: scripted provider has no reply left"


def test_a_leaking_column_is_not_available_to_the_script(tmp_path: Path) -> None:
    cheat = _reply(
        HEADER
        + """
taxed = np.array(table["price_with_tax"].to_pylist(), dtype=float)
json.dump((taxed[test] / 1.13).tolist(), open("predictions.json", "w"))
"""
    )
    provider = ScriptedProvider([cheat, "STOP"])
    result = run_agent(_prices(leak=True), "price", provider, workdir=tmp_path)
    assert [w.column for w in result.withheld] == ["price_with_tax"]
    assert "`price_with_tax` was withheld" in provider.calls[0][1]["content"]
    assert result.attempts[0].status == "failed"
    assert "price_with_tax" in result.attempts[0].feedback
    # The baselines did not get the column either: same bar as on the data without it.
    honest = run_agent(_prices(), "price", ScriptedProvider(["STOP"]), workdir=tmp_path / "h")
    assert result.baselines == honest.baselines


def test_a_flagged_column_the_caller_vouches_for_stays_available(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    size = rng.uniform(40, 200, size=N)
    # Almost noiseless: size alone predicts price well enough to look like a leak.
    table = pa.table({"size": size.tolist(), "price": (1000 * size + rng.normal(size=N)).tolist()})
    with pytest.raises(AgentError, match=r"withheld by the leakage checks: size\).*`keep`"):
        run_agent(table, "price", ScriptedProvider([QUADRATIC]), workdir=tmp_path / "a")

    kept = run_agent(
        table, "price", ScriptedProvider([QUADRATIC, "STOP"]), workdir=tmp_path / "b", keep=["size"]
    )
    assert kept.withheld == () and kept.attempts[0].status == "scored"


def test_every_run_is_logged_with_the_judges_score_and_its_files_kept(tmp_path: Path) -> None:
    provider = ScriptedProvider([CRASH, QUADRATIC, "STOP"])
    result = run_agent(_prices(), "price", provider, workdir=tmp_path)
    runs = list(RunLog(tmp_path / "runs.jsonl"))
    assert [r.status for r in runs] == [Status.FAILED, Status.OK]
    assert runs[1].metrics == {"mae": result.attempts[1].value}
    assert runs[1].run_id == result.attempts[1].run_id
    assert (tmp_path / "task.feather").exists()
    assert "np.polyfit" in (tmp_path / "attempt_02" / "experiment.py").read_text()
    assert (tmp_path / "attempt_02" / "predictions.json").exists()
    assert not (tmp_path / "attempt_01" / "predictions.json").exists()


def test_extract_script_takes_the_last_fenced_block() -> None:
    text = "First idea:\n```python\nprint(1)\n```\nBetter:\n```py\nprint(2)\n```\nDone."
    assert extract_script(text) == "print(2)\n"
    assert extract_script("```\nprint(3)\n```") == "print(3)\n"
    assert extract_script("no code here") is None
    assert extract_script("```python\n\n```") is None


def test_impossible_budgets_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one experiment"):
        Budget(experiments=0)
    with pytest.raises(ValueError, match="token budget"):
        Budget(tokens=0)

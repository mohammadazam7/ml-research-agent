import os
import textwrap
from pathlib import Path

import pytest

from ml_research_agent.runner import Limits, RunLog, Status, run_script, sha256_of


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "train.py"
    path.write_text(textwrap.dedent(body))
    return path


@pytest.fixture
def data(tmp_path: Path) -> Path:
    path = tmp_path / "rows.csv"
    path.write_text("x,y\n1,0\n2,1\n")
    return path


def test_a_script_that_writes_metrics_succeeds(tmp_path: Path, data: Path) -> None:
    script = _script(
        tmp_path,
        """
        import json, os
        rows = open(os.environ["MLAGENT_DATA"]).read().splitlines()[1:]
        json.dump({"roc_auc": 0.75, "rows": len(rows)}, open("metrics.json", "w"))
        """,
    )
    run = run_script(script, data)
    assert run.status is Status.OK
    assert run.metrics == {"roc_auc": 0.75, "rows": 2.0}
    assert run.exit_code == 0
    assert run.script_sha256 == sha256_of(script)
    assert run.data_sha256 == sha256_of(data)


def test_the_script_sees_only_path_and_its_data(tmp_path: Path, data: Path) -> None:
    os.environ["MLAGENT_TEST_SECRET"] = "should not leak"
    try:
        script = _script(
            tmp_path,
            """
            import json, os
            json.dump({"vars": len(os.environ), "secret": int("MLAGENT_TEST_SECRET" in os.environ),
                       "files": len(os.listdir("."))}, open("metrics.json", "w"))
            """,
        )
        run = run_script(script, data)
    finally:
        del os.environ["MLAGENT_TEST_SECRET"]
    assert run.metrics["secret"] == 0.0
    # experiment.py and the data copy; metrics.json is written after listing.
    assert run.metrics["files"] == 2.0


def test_a_crash_is_recorded_with_the_end_of_stderr(tmp_path: Path, data: Path) -> None:
    run = run_script(_script(tmp_path, "raise SystemExit('bad column: y2')"), data)
    assert run.status is Status.FAILED
    assert run.exit_code == 1
    assert "bad column: y2" in run.stderr_tail


def test_a_script_that_runs_too_long_is_stopped(tmp_path: Path, data: Path) -> None:
    run = run_script(
        _script(tmp_path, "import time\ntime.sleep(30)"), data, limits=Limits(wall_seconds=0.5)
    )
    assert run.status is Status.TIMEOUT
    assert run.exit_code is None
    assert run.seconds < 5


@pytest.mark.skipif(os.name != "posix", reason="resource limits are POSIX only")
def test_a_script_that_takes_too_much_memory_is_stopped(tmp_path: Path, data: Path) -> None:
    script = _script(tmp_path, "block = bytearray(1500 * 1024 * 1024)\nprint(len(block))")
    run = run_script(script, data, limits=Limits(memory_mb=512))
    assert run.status is Status.FAILED
    assert "MemoryError" in run.stderr_tail


@pytest.mark.parametrize(
    "written",
    ["", "not json", "[1, 2]", "{}", '{"auc": "high"}', '{"ok": true}'],
)
def test_missing_or_malformed_metrics_are_their_own_status(
    tmp_path: Path, data: Path, written: str
) -> None:
    body = f"open('metrics.json', 'w').write({written!r})" if written else "pass"
    run = run_script(_script(tmp_path, body), data)
    assert run.status is Status.BAD_METRICS
    assert run.metrics == {}


def test_the_log_keeps_every_run_and_finds_the_best(tmp_path: Path, data: Path) -> None:
    log = RunLog(tmp_path / "runs" / "runs.jsonl")
    assert log.best("roc_auc") is None
    for auc in (0.70, 0.81, 0.81, 0.65):
        script = _script(
            tmp_path, f"import json\njson.dump({{'roc_auc': {auc}}}, open('metrics.json', 'w'))"
        )
        log.append(run_script(script, data))
    log.append(run_script(_script(tmp_path, "raise SystemExit(1)"), data))
    runs = list(log)
    assert [r.status for r in runs] == [Status.OK] * 4 + [Status.FAILED]
    best = log.best("roc_auc")
    assert best is not None and best.run_id == runs[1].run_id
    worst = log.best("roc_auc", higher_is_better=False)
    assert worst is not None and worst.metrics["roc_auc"] == 0.65
    assert len({r.run_id for r in runs}) == 5


def test_named_outputs_are_copied_out_and_stale_copies_removed(tmp_path: Path, data: Path) -> None:
    kept = tmp_path / "kept" / "predictions.json"
    kept.parent.mkdir()
    kept.write_text("old")
    missing = tmp_path / "kept" / "never.json"
    script = _script(tmp_path, 'open("predictions.json", "w").write("[1, 2]")')
    run = run_script(
        script,
        data,
        outputs={"predictions.json": kept, "never.json": missing},
        require_metrics=False,
    )
    assert run.status is Status.OK and run.metrics == {}
    assert kept.read_text() == "[1, 2]"
    assert not missing.exists()

    failing = _script(tmp_path, 'open("predictions.json", "w").write("[3]"); raise SystemExit(1)')
    run = run_script(failing, data, outputs={"predictions.json": kept}, require_metrics=False)
    assert run.status is Status.FAILED
    assert not kept.exists()  # a failed run leaves no result behind, not even the old one


def test_metrics_are_still_required_unless_the_caller_scores_the_run(
    tmp_path: Path, data: Path
) -> None:
    script = _script(tmp_path, "pass")
    assert run_script(script, data).status is Status.BAD_METRICS
    assert run_script(script, data, require_metrics=False).status is Status.OK

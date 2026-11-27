import textwrap
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.feather as feather
import pyarrow.parquet as parquet
import pytest

from ml_research_agent.cli import main, read_table
from ml_research_agent.providers import Reply, ScriptedProvider

QUADRATIC = textwrap.dedent(
    """
    ```python
    import json, os
    import numpy as np
    import pyarrow.feather as feather

    table = feather.read_table(os.environ["MLAGENT_DATA"])
    side = np.array(table["__split__"].to_pylist())
    size = np.array(table["size"].to_pylist(), dtype=float)
    price = np.array(table["price"].to_pylist(), dtype=float)
    fit = np.polyfit(size[side == "train"], price[side == "train"], 2)
    json.dump(np.polyval(fit, size[side == "test"]).tolist(), open("predictions.json", "w"))
    ```
    """
)


def _table() -> pa.Table:
    rng = np.random.default_rng(3)
    size = rng.uniform(40, 200, size=600)
    price = 20 * size**2 + rng.normal(scale=60_000, size=600)
    return pa.table({"size": size.round(2).tolist(), "price": price.round(0).tolist()})


@pytest.fixture
def data(tmp_path: Path) -> Path:
    path = tmp_path / "homes.csv"
    pacsv.write_csv(_table(), path)
    return path


def test_run_prints_the_plan_every_attempt_and_the_verdict(
    tmp_path: Path, data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = ScriptedProvider([Reply(QUADRATIC, 800, 150), "STOP"])
    workdir = tmp_path / "run"
    code = main(
        ["run", str(data), "--target", "price", "--workdir", str(workdir)], provider=provider
    )
    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert out[0] == "mlagent: 600 rows, target price"
    assert out[1] == "split: random; metric: mae (lower is better)"
    assert any(line.startswith("baselines: prior ") and "linear " in line for line in out)
    assert any(line.startswith("attempt 1: scored") for line in out)
    assert any(
        line.startswith("best: attempt 1, mae ") and "beats the baselines" in line for line in out
    )
    assert "stopped: the model had nothing better to try" in out
    assert "tokens: 800 in, 150 out" in out
    assert out[-1] == f"files: {workdir}"
    assert (workdir / "attempt_01" / "experiment.py").exists()


def test_exit_status_is_one_when_nothing_beats_the_baselines(
    tmp_path: Path, data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["run", str(data), "--target", "price", "--workdir", str(tmp_path / "run")],
        provider=ScriptedProvider(["STOP"]),
    )
    assert code == 1
    assert "best: no attempt produced predictions that could be scored" in capsys.readouterr().out


def test_budget_options_reach_the_loop(tmp_path: Path, data: Path) -> None:
    provider = ScriptedProvider([QUADRATIC, QUADRATIC, QUADRATIC])
    main(
        [
            "run",
            str(data),
            "--target",
            "price",
            "--workdir",
            str(tmp_path / "r"),
            "--experiments",
            "2",
        ],
        provider=provider,
    )
    assert len(provider.calls) == 2


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--target", "rent"], "rent"),
        (["--target", "price", "--experiments", "0"], "at least one experiment"),
    ],
)
def test_problems_are_reported_in_one_line_with_exit_status_two(
    tmp_path: Path,
    data: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    code = main(
        ["run", str(data), "--workdir", str(tmp_path / "run"), *arguments],
        provider=ScriptedProvider([]),
    )
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("mlagent: ") and message in err and err.count("\n") == 1


def test_a_missing_file_and_a_missing_model_are_reported(
    tmp_path: Path, data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", str(tmp_path / "nowhere.csv"), "--target", "price"], env={}) == 2
    assert "nowhere.csv" in capsys.readouterr().err
    assert main(["run", str(data), "--target", "price"], env={}) == 2
    assert "MLAGENT_BASE_URL, MLAGENT_MODEL" in capsys.readouterr().err


def test_tables_are_read_from_csv_parquet_and_feather(tmp_path: Path) -> None:
    table = _table().slice(0, 5)
    pacsv.write_csv(table, tmp_path / "a.csv")
    parquet.write_table(table, tmp_path / "a.parquet")
    feather.write_feather(table, str(tmp_path / "a.feather"))
    for name in ("a.csv", "a.parquet", "a.feather"):
        assert read_table(tmp_path / name).to_pydict() == table.to_pydict()
    with pytest.raises(ValueError, match=r"expected a \.csv, \.parquet or \.feather"):
        read_table(tmp_path / "a.xlsx")

"""``mlagent run``: point the agent at a table and a target column."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.feather as feather
import pyarrow.parquet as parquet

from ml_research_agent.agent import AgentError, Budget, Result, run_agent
from ml_research_agent.providers import (
    OpenAICompatibleProvider,
    Provider,
    ProviderError,
    config_from_env,
)
from ml_research_agent.runner import Limits

EXIT_BEATS_BASELINES = 0
EXIT_NO_BETTER = 1
EXIT_COULD_NOT_RUN = 2


def read_table(path: Path) -> pa.Table:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pacsv.read_csv(path)
    if suffix == ".parquet":
        return parquet.read_table(path)
    if suffix in (".feather", ".arrow"):
        return feather.read_table(str(path))
    raise ValueError(f"cannot read {path.name}: expected a .csv, .parquet or .feather file")


def summary(result: Result, workdir: Path) -> str:
    direction = "higher" if result.baselines[0].higher_is_better else "lower"
    lines = [
        f"split: {result.plan.strategy.value}; metric: {result.metric} ({direction} is better)",
    ]
    lines += [f"reason: {reason}" for reason in result.plan.reasons]
    lines += [f"withheld: {w.column} ({w.reason})" for w in result.withheld]
    lines.append("baselines: " + ", ".join(f"{s.baseline} {s.value:.4f}" for s in result.baselines))
    for attempt in result.attempts:
        score = "-" if attempt.value is None else f"{attempt.value:.4f}"
        lines.append(
            f"attempt {attempt.number}: {attempt.status:<9} {score:>12} {attempt.seconds:8.1f} s"
        )
    if result.best is None or result.best.value is None:
        lines.append("best: no attempt produced predictions that could be scored")
    else:
        verdict = "beats the baselines" if result.beats_baselines else "does not beat the baselines"
        score = f"{result.metric} {result.best.value:.4f}"
        lines.append(f"best: attempt {result.best.number}, {score}, {verdict}")
    lines.append(f"stopped: {result.stopped}")
    lines.append(f"tokens: {result.input_tokens} in, {result.output_tokens} out")
    lines.append(f"files: {workdir}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlagent",
        description="An agent that writes experiment scripts; the core runs and judges them.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the agent on one table")
    run.add_argument("data", type=Path, help="a .csv, .parquet or .feather file")
    run.add_argument("--target", required=True, help="the column to predict")
    run.add_argument("--workdir", type=Path, default=Path("mlagent-run"))
    run.add_argument("--experiments", type=int, default=5, help="scripts the model may try")
    run.add_argument("--tokens", type=int, default=None, help="token budget for the whole run")
    run.add_argument("--timeout", type=float, default=300.0, help="seconds per script")
    run.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="COLUMN",
        help="never withhold this column, even if a leakage check flags it (repeatable)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: Provider | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        table = read_table(args.data)
        model = provider or OpenAICompatibleProvider(config_from_env(env or os.environ))
        result = run_agent(
            table,
            args.target,
            model,
            workdir=args.workdir,
            budget=Budget(experiments=args.experiments, tokens=args.tokens),
            limits=Limits(wall_seconds=args.timeout),
            keep=args.keep,
        )
    # ValueError covers unreadable files and the profile, plan and baseline errors.
    except (OSError, ValueError, AgentError, ProviderError) as error:
        print(f"mlagent: {error}", file=sys.stderr)
        return EXIT_COULD_NOT_RUN
    print(f"mlagent: {table.num_rows} rows, target {args.target}")
    print(summary(result, args.workdir))
    return EXIT_BEATS_BASELINES if result.beats_baselines else EXIT_NO_BETTER


if __name__ == "__main__":
    raise SystemExit(main())

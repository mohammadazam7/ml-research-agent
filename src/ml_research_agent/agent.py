"""The agent loop: a model proposes experiment scripts, the core runs and judges them.

One round is: ask the model for a script, run it in the sandboxed runner, hand the
predictions to the judge, and tell the model what happened. The loop ends when the
budget of experiments or tokens is used, when the model says it has nothing better to
try, or when the model cannot be reached.

The division of work is the point of the project. The model decides what to try. It
never decides what counts: the split, the metric, the baselines and the score of every
attempt come from the deterministic core, and a result is reported as beating the
baselines only when the judge's numbers say so.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa

from ml_research_agent import baselines, leakage
from ml_research_agent.baselines import Score
from ml_research_agent.judge import PREDICTIONS_FILE, Judge, Withheld
from ml_research_agent.plan import Plan, plan, split
from ml_research_agent.profile import profile
from ml_research_agent.providers import Message, Provider, ProviderError
from ml_research_agent.runner import Limits, RunLog, Status, run_script

STOP_WORD = "STOP"
STDERR_LINES = 15

SYSTEM = f"""You are a careful machine learning engineer working inside an evaluation harness.

You are given a prediction task. Reply with exactly one Python script inside a ```python
block. The harness runs the script, scores its predictions on held-out rows you cannot
see, and tells you the result. Then you may reply with an improved script.

Rules:
- The script must follow the output format in the task exactly.
- You are judged only by the harness's score. Do not print or report your own.
- Keep each script complete and self-contained; nothing is kept between runs.
- When you have nothing better to try, reply with the single word {STOP_WORD}."""


class AgentError(RuntimeError):
    """The run could not start."""


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Budget:
    experiments: int = 5
    tokens: int | None = None  # input plus output tokens across the whole run

    def __post_init__(self) -> None:
        if self.experiments < 1:
            raise ValueError("the budget must allow at least one experiment")
        if self.tokens is not None and self.tokens < 1:
            raise ValueError("a token budget must be positive")


@dataclass(frozen=True)
class Attempt:
    number: int
    # scored, invalid (ran but the predictions were unusable), failed, timeout, no_script
    status: str
    value: float | None
    feedback: str
    seconds: float = 0.0
    run_id: str | None = None


@dataclass(frozen=True)
class Result:
    plan: Plan
    leakage: leakage.Report
    withheld: tuple[Withheld, ...]
    baselines: tuple[Score, ...]
    attempts: tuple[Attempt, ...]
    best: Attempt | None
    beats_baselines: bool
    stopped: str
    input_tokens: int
    output_tokens: int

    @property
    def metric(self) -> str:
        return self.plan.metric


def extract_script(text: str) -> str | None:
    """The last fenced code block in a reply, or None when there is none."""
    blocks = _FENCE.findall(text)
    return blocks[-1].strip() + "\n" if blocks and blocks[-1].strip() else None


def _better(a: float, b: float, higher_is_better: bool) -> bool:
    return a > b if higher_is_better else a < b


def run_agent(
    table: pa.Table,
    target: str,
    provider: Provider,
    *,
    workdir: Path,
    budget: Budget | None = None,
    limits: Limits | None = None,
    keep: Collection[str] = (),
) -> Result:
    """Run the loop on ``table`` and return everything that happened.

    ``workdir`` receives the task file, each attempt's script and predictions, and
    ``runs.jsonl`` with one line per script that was run. ``keep`` names columns that
    stay available to the scripts even if a leakage check flags them, for features
    known to be legitimate.
    """
    budget = budget or Budget()
    prof = profile(table, target)
    chosen = plan(prof)
    rows = split(table, prof, chosen)
    report = leakage.check(table, prof, chosen, rows)
    judge = Judge(table, prof, chosen, rows, report, keep=keep)
    # The baselines get the same columns the scripts get. A baseline that could read a
    # withheld column would set a bar no honest model can reach.
    if not judge.features:
        names = ", ".join(w.column for w in judge.withheld) or "none"
        raise AgentError(
            f"no feature column is left to learn from (withheld by the leakage checks: "
            f"{names}). Name the ones that are known before the prediction is made in `keep`."
        )
    visible = table.drop_columns([w.column for w in judge.withheld])
    scores = tuple(baselines.run(visible, profile(visible, target), chosen, rows))
    higher = judge.higher_is_better
    bar = max(s.value for s in scores) if higher else min(s.value for s in scores)

    task = workdir / "task.feather"
    judge.write_task(task)
    log = RunLog(workdir / "runs.jsonl")
    baseline_text = ", ".join(f"{s.baseline} {s.value:.4f}" for s in scores)
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"{judge.assignment()}\nBaselines to beat: {baseline_text}.",
        },
    ]

    attempts: list[Attempt] = []
    best: Attempt | None = None
    used_in = used_out = 0
    stopped = f"the budget of {budget.experiments} experiments was used"

    while len(attempts) < budget.experiments:
        if budget.tokens is not None and used_in + used_out >= budget.tokens:
            stopped = f"the budget of {budget.tokens} tokens was used"
            break
        try:
            reply = provider.complete(messages)
        except ProviderError as error:
            stopped = f"the model could not be used: {error}"
            break
        used_in += reply.input_tokens
        used_out += reply.output_tokens
        if reply.text.strip() == STOP_WORD:
            stopped = "the model had nothing better to try"
            break

        number = len(attempts) + 1
        script = extract_script(reply.text)
        if script is None:
            attempt = Attempt(
                number,
                "no_script",
                None,
                f"No script found. Reply with one ```python block, or {STOP_WORD}.",
            )
        else:
            folder = workdir / f"attempt_{number:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "experiment.py"
            path.write_text(script, encoding="utf-8")
            kept = folder / PREDICTIONS_FILE
            run = run_script(
                path, task, limits=limits, outputs={PREDICTIONS_FILE: kept}, require_metrics=False
            )
            if run.status is Status.TIMEOUT:
                status, value = "timeout", None
                feedback = f"The script was stopped after {run.seconds:.0f} seconds."
            elif run.status is not Status.OK:
                status, value = "failed", None
                tail = "\n".join(run.stderr_tail.strip().splitlines()[-STDERR_LINES:])
                feedback = f"The script failed with exit code {run.exit_code}.\n{tail}"
            else:
                verdict = judge.score(kept)
                if verdict.valid and verdict.value is not None:
                    status, value = "scored", verdict.value
                    feedback = f"{judge.metric} {value:.4f} on the held-out rows."
                    run = replace(run, metrics={judge.metric: value})  # the judge's number
                else:
                    status, value = "invalid", None
                    feedback = f"The predictions could not be scored: {verdict.problem}."
            log.append(run)
            attempt = Attempt(number, status, value, feedback, run.seconds, run.run_id)

        attempts.append(attempt)
        if attempt.value is not None and (
            best is None or best.value is None or _better(attempt.value, best.value, higher)
        ):
            best = attempt
        standing = (
            f"Best so far: {best.value:.4f} (attempt {best.number})."
            if best is not None and best.value is not None
            else "No attempt has been scored yet."
        )
        left = budget.experiments - len(attempts)
        messages.append({"role": "assistant", "content": reply.text})
        messages.append(
            {
                "role": "user",
                "content": f"Attempt {number}: {attempt.feedback}\n{standing} "
                f"Baselines: {baseline_text}. Experiments left: {left}.",
            }
        )

    beats = best is not None and best.value is not None and _better(best.value, bar, higher)
    return Result(
        plan=chosen,
        leakage=report,
        withheld=judge.withheld,
        baselines=scores,
        attempts=tuple(attempts),
        best=best,
        beats_baselines=beats,
        stopped=stopped,
        input_tokens=used_in,
        output_tokens=used_out,
    )

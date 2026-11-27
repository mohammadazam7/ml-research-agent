"""Run an experiment script in a separate process with limits, and record every run.

The agent will write training scripts; this module is where they run. Each run gets:

- a fresh working directory holding only a copy of the inputs,
- ``python -I`` (no user site-packages, no current directory on the path) and an
  environment with nothing in it but ``PATH`` and the input paths,
- a wall-clock timeout, a CPU-time limit and an address-space limit (POSIX only),
- a record in an append-only JSON-lines log: what ran (hashes of the script and the
  data), how it ended, how long it took and the metrics it reported.

The script reports metrics by writing ``metrics.json`` in its working directory. A
caller that scores the script itself (the judge does, from the script's predictions) can
ask for other files to be copied out before the directory is removed.

This is containment against mistakes (a runaway loop, a memory blow-up, a script that
writes all over the disk), not a security boundary: the script runs as the same user
and can reach the network. Untrusted code needs a container or a VM around this.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

OUTPUT_TAIL_CHARS = 4000


class Status(Enum):
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BAD_METRICS = "bad_metrics"


@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 300.0
    cpu_seconds: int = 300
    memory_mb: int = 2048


@dataclass(frozen=True)
class Run:
    run_id: str
    started_at: str
    script_sha256: str
    data_sha256: str
    status: Status
    seconds: float
    exit_code: int | None
    metrics: dict[str, float] = field(default_factory=dict)
    stderr_tail: str = ""

    def to_json(self) -> str:
        record = asdict(self)
        record["status"] = self.status.value
        return json.dumps(record, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> Run:
        record = json.loads(line)
        record["status"] = Status(record["status"])
        return cls(**record)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _limiter(limits: Limits) -> Callable[[], None] | None:
    if os.name != "posix":
        return None

    def apply() -> None:
        import resource

        memory = limits.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))

    return apply


def _read_metrics(path: Path) -> dict[str, float] | None:
    """The metrics a script wrote, or None if the file is missing or malformed."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or not raw:
        return None
    metrics = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        metrics[str(name)] = float(value)
    return metrics


def run_script(
    script: Path,
    data: Path,
    *,
    limits: Limits | None = None,
    outputs: Mapping[str, Path] | None = None,
    require_metrics: bool = True,
) -> Run:
    """Run ``script`` with a copy of ``data`` next to it and return what happened.

    The script finds the copy of the data through the ``MLAGENT_DATA`` environment
    variable. ``outputs`` maps the name of a file the script may write in its working
    directory to where a copy should be kept; a file the script did not write is
    skipped, and a stale file at the destination is removed so an old result is never
    mistaken for a new one. With ``require_metrics`` off, a script that exits cleanly
    without ``metrics.json`` still counts as OK.
    """
    for kept in (outputs or {}).values():
        kept.unlink(missing_ok=True)
    limits = limits or Limits()
    run_id = uuid.uuid4().hex
    started = datetime.now(UTC).isoformat(timespec="seconds")
    with tempfile.TemporaryDirectory(prefix=f"mlagent-{run_id[:8]}-") as tmp:
        workdir = Path(tmp)
        shutil.copy(script, workdir / "experiment.py")
        shutil.copy(data, workdir / data.name)
        env = {"PATH": os.environ.get("PATH", ""), "MLAGENT_DATA": str(workdir / data.name)}
        clock = time.perf_counter()
        try:
            done = subprocess.run(
                [sys.executable, "-I", "experiment.py"],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=limits.wall_seconds,
                preexec_fn=_limiter(limits),
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            stderr = expired.stderr if isinstance(expired.stderr, str) else ""
            return Run(
                run_id,
                started,
                sha256_of(script),
                sha256_of(data),
                Status.TIMEOUT,
                round(time.perf_counter() - clock, 3),
                None,
                stderr_tail=stderr[-OUTPUT_TAIL_CHARS:],
            )
        seconds = round(time.perf_counter() - clock, 3)
        metrics = _read_metrics(workdir / "metrics.json") if done.returncode == 0 else None
        if done.returncode != 0:
            status = Status.FAILED
        elif metrics is None and require_metrics:
            status = Status.BAD_METRICS
        else:
            status = Status.OK
        if done.returncode == 0:
            for name, kept in (outputs or {}).items():
                written = workdir / name
                if written.is_file() and not written.is_symlink():
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(written, kept)
        return Run(
            run_id,
            started,
            sha256_of(script),
            sha256_of(data),
            status,
            seconds,
            done.returncode,
            metrics or {},
            done.stderr[-OUTPUT_TAIL_CHARS:],
        )


class RunLog:
    """Append-only JSON lines: one line per run, never rewritten."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, run: Run) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(run.to_json() + "\n")

    def __iter__(self) -> Iterator[Run]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield Run.from_json(line)

    def best(self, metric: str, *, higher_is_better: bool = True) -> Run | None:
        """The successful run with the best value of ``metric``; earliest wins a tie."""
        candidates = [r for r in self if r.status is Status.OK and metric in r.metrics]
        if not candidates:
            return None
        sign = 1.0 if higher_is_better else -1.0
        return max(candidates, key=lambda r: sign * r.metrics[metric])

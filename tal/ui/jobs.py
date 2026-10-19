"""Background jobs for long-running work (experiments, capability runs, audits).

Long jobs must not block the HTTP server: a single request thread would make
the UI freeze for the 20 minutes Heretic needs. Jobs run in daemon threads and
the browser polls ``GET /api/jobs/<id>`` for status and the log tail.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..util import Logger, atomic_write_text

STATUSES = ("queued", "running", "done", "failed")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Job:
    def __init__(self, kind: str, title: str, *, run_id: str | None = None, log_path: Path | None = None):
        self.id = f"job-{uuid.uuid4().hex[:10]}"
        self.kind = kind
        self.title = title
        self.run_id = run_id
        self.log_path = Path(log_path) if log_path else None
        self.status = "queued"
        self.created_at = _now()
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.error: str | None = None
        self.result: Any = None
        self.progress: float | None = None
        self._logs: deque[str] = deque(maxlen=4000)
        self._lock = threading.Lock()

    def add_log(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line:
            return
        with self._lock:
            self._logs.append(line)

    def log_text(self, limit: int = 400) -> str:
        with self._lock:
            lines = list(self._logs)[-limit:]
        if not lines and self.log_path and self.log_path.exists():
            try:
                lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
            except OSError:
                lines = []
        return "\n".join(lines)

    def to_dict(self, *, logs: bool = True, limit: int = 400) -> dict:
        payload = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
            "progress": self.progress,
        }
        if logs:
            payload["log"] = self.log_text(limit)
        return payload


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        kind: str,
        title: str,
        fn: Callable[[Job], Any],
        *,
        run_id: str | None = None,
        log_path: Path | None = None,
    ) -> Job:
        job = Job(kind, title, run_id=run_id, log_path=log_path)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job, fn), name=job.id, daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        job.status = "running"
        job.started_at = _now()
        try:
            result = fn(job)
            job.result = _json_safe(result)
            job.status = "done"
            job.progress = 1.0
        except BaseException as exc:  # noqa: BLE001 - a job must never die silently
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.add_log(traceback.format_exc())
        finally:
            job.finished_at = _now()
            if job.log_path:
                try:
                    job.log_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_text(job.log_path, job.log_text(limit=4000) + "\n")
                except OSError:
                    pass

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def list(self, *, kind: str | None = None, limit: int = 30) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        return jobs[:limit]


class JobLogger(Logger):
    """Logger that appends to a job and to the job's file, without console spam."""

    def __init__(self, job: Job, *, verbose: bool = False):
        super().__init__(log_path=None, verbose=verbose, quiet=True)
        self.job = job

    def _emit(self, level: str, msg: str) -> None:
        if level == "debug" and not self.verbose:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {level.upper():5s} {msg}"
        with self._lock:
            self.job.add_log(line)
            if self.job.log_path:
                self.job.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.job.log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)

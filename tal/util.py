"""Small shared utilities: canonical hashing, atomic JSON, logging, locks, subprocess."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .errors import LockTimeout, UpstreamError


# --------------------------------------------------------------------------- #
# Hashing / JSON
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(value: Any, length: int = 12) -> str:
    return sha256_text(canonical_json(value))[:length]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def safe_slug(text: str, max_len: int = 48) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in text.strip()]
    slug = "".join(keep).strip("-.")
    return (slug or "run")[:max_len]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class Logger:
    """Console + file logger. Deliberately tiny; no external dependency."""

    LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}

    def __init__(self, log_path: Path | None = None, verbose: bool = False, quiet: bool = False):
        self.log_path = Path(log_path) if log_path else None
        self.verbose = verbose
        self.quiet = quiet
        self._lock = threading.Lock()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, level: str, msg: str) -> None:
        if level == "debug" and not self.verbose:
            return
        if level == "info" and self.quiet:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {level.upper():5s} {msg}"
        with self._lock:
            if not self.quiet or level in ("warn", "error"):
                try:
                    print(line, file=sys.stdout, flush=True)
                except UnicodeEncodeError:
                    enc = sys.stdout.encoding or "utf-8"
                    sys.stdout.write(line.encode(enc, "replace").decode(enc, "replace") + "\n")
                    sys.stdout.flush()
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def debug(self, msg: str) -> None:
        self._emit("debug", msg)

    def info(self, msg: str) -> None:
        self._emit("info", msg)

    def warn(self, msg: str) -> None:
        self._emit("warn", msg)

    def error(self, msg: str) -> None:
        self._emit("error", msg)

    def stage(self, name: str, msg: str) -> None:
        self._emit("info", f"[{name}] {msg}")


# --------------------------------------------------------------------------- #
# File lock (GPU lease)
# --------------------------------------------------------------------------- #
@dataclass
class LockHandle:
    path: Path
    acquired_at: float

    def release(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.path)


@contextlib.contextmanager
def file_lock(
    path: Path,
    *,
    timeout: float = 0.0,
    stale_after: float = 12 * 3600,
    poll: float = 2.0,
    owner: str | None = None,
) -> Iterator[LockHandle]:
    """Exclusive advisory lock based on O_EXCL creation.

    Adequate for a single machine and a handful of competing TAL processes.
    The lock records pid/host/time so a human can diagnose a stale lease.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max(0.0, timeout)
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", ""),
            "owner": owner or "",
            "acquired_at": utc_now(),
        },
        ensure_ascii=False,
    )
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_after:
                with contextlib.suppress(OSError):
                    os.unlink(path)
                continue
            if time.time() >= deadline:
                held = ""
                with contextlib.suppress(OSError):
                    held = path.read_text(encoding="utf-8", errors="replace").strip()
                raise LockTimeout(
                    f"GPU/run lock is held by another process: {path}\n{held}\n"
                    "Wait for it to finish, or pass --force-unlock if you are sure it is stale."
                )
            time.sleep(poll)
    handle = LockHandle(path=path, acquired_at=time.time())
    try:
        yield handle
    finally:
        handle.release()


# --------------------------------------------------------------------------- #
# Subprocess execution
# --------------------------------------------------------------------------- #
@dataclass
class CommandResult:
    argv: Sequence[str]
    returncode: int
    duration_s: float
    log_path: Path | None
    tail: str


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
    with contextlib.suppress(Exception):
        proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=10)


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    logger: Logger | None = None,
    timeout_s: float | None = None,
    echo: bool = True,
    check: bool = True,
) -> CommandResult:
    """Run a child process, stream its combined output to console + log file.

    Raises UpstreamError on non-zero exit or timeout when `check` is true. Pass
    `check=False` for tools whose exit code carries a *result* (Senbonzakura
    uses semantic exit codes) or whose headless mode exits non-zero after the
    real work is already on disk (Heretic's interactive trial menu).
    """
    started = time.time()
    argv = [str(a) for a in argv]
    if logger:
        logger.info("$ " + " ".join(argv))
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lock = threading.Lock()

    proc = subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with lock:
                lines.append(line.rstrip("\n"))
                if len(lines) > 2000:
                    del lines[:500]
            if echo and logger:
                logger.info("  " + line.rstrip("\n"))
            if log_path:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(line)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
    thread.join(timeout=10)
    duration = time.time() - started
    tail = "\n".join(lines[-40:])
    if timed_out:
        raise UpstreamError(
            f"command timed out after {human_duration(duration)}: {' '.join(argv)}\n"
            f"log: {log_path}\nlast output:\n{tail}"
        )
    if proc.returncode != 0 and check:
        raise UpstreamError(
            f"command failed (exit {proc.returncode}): {' '.join(argv)}\n"
            f"log: {log_path}\nlast output:\n{tail}"
        )
    return CommandResult(argv, proc.returncode, duration, log_path, tail)


def which(name: str) -> str | None:
    return shutil.which(name)


def package_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return None


def pip_install_hint(extra: str = "hf") -> str:
    return f"pip install -e .[{extra}]"

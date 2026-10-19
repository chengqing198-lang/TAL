"""Use-case layer for the UI.

The HTTP server stays thin: every function here takes plain dicts and returns
plain dicts, so the same code can be tested without sockets and reused by a
future CLI. Long operations are submitted to :class:`JobManager`.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from pathlib import Path
from typing import Any

from ..errors import GateFailed, SpecError
from ..pipeline import open_run, run_experiment
from ..spec import load_spec
from ..stats import rate_stats
from ..util import run_command, which
from .clients import build_client, is_refusal, probe_model
from .jobs import Job, JobLogger, JobManager
from .specbuilder import build_experiment_spec
from .store import Workspace

LETTERS = "ABCDEFGH"


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
def chat(workspace: Workspace, payload: dict) -> dict:
    model_id = str(payload.get("model_id") or "")
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise SpecError("prompt is required")
    if len(prompt) > 100_000:
        raise SpecError("prompt is too long (100k characters)")
    source = workspace.model_source(model_id)
    client = build_client(source)
    result = client.generate(
        prompt,
        system=str(payload.get("system") or "").strip() or None,
        max_tokens=_clamp(payload.get("max_tokens"), 256, 8, 4096),
        temperature=_float(payload.get("temperature"), 0.0, 0.0, 2.0),
    )
    return {
        "model": source.get("model") or source["id"],
        "model_id": model_id,
        "kind": source["kind"],
        "prompt": prompt,
        **result.to_dict(),
    }


def probe(workspace: Workspace, model_id: str) -> dict:
    source = workspace.model_source(model_id)
    result = probe_model(source)
    return {"model_id": model_id, "kind": source["kind"], **result}


# --------------------------------------------------------------------------- #
# batch probe: feed a request library and see answer vs refusal
# --------------------------------------------------------------------------- #
def start_batch_probe(workspace: Workspace, jobs: JobManager, payload: dict) -> dict:
    model_id = str(payload.get("model_id") or "")
    prompt_id = str(payload.get("prompt_id") or "")
    lib = workspace.get_prompt(prompt_id)
    if lib.get("kind") == "capability":
        raise SpecError("capability 题库请用“能力测试”，批量探测只接受 target/preserve 文本库")
    items = list(lib.get("items") or [])
    if not items:
        raise SpecError("请求库为空")
    limit = _clamp(payload.get("limit"), min(10, len(items)), 1, min(len(items), 200))
    items = items[:limit]
    source = workspace.model_source(model_id)
    max_tokens = _clamp(payload.get("max_tokens"), 256, 8, 4096)
    temperature = _float(payload.get("temperature"), 0.0, 0.0, 2.0)
    system = str(payload.get("system") or "").strip() or None
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() // 1_000_000 % 1000:03d}"
    log_path = workspace.jobs_dir / f"probe-{stamp}.log"

    def work(job: Job) -> dict:
        client = build_client(source)
        logger = JobLogger(job)
        logger.info(f"batch probe: {len(items)} prompts on {source.get('model') or model_id}")
        records = []
        refused_count = 0
        for index, prompt in enumerate(items):
            result = client.generate(
                prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            refused_count += int(result.refused)
            records.append(
                {
                    "index": index,
                    "prompt": prompt[:2000],
                    "response": (result.text or "")[:2000],
                    "refused": bool(result.refused),
                    "latency_ms": result.latency_ms,
                }
            )
            job.progress = (index + 1) / len(items)
            if (index + 1) % 10 == 0 or index + 1 == len(items):
                logger.info(f"probed {index + 1}/{len(items)} (refused {refused_count})")
        stats = rate_stats(refused_count, len(items))
        result = {
            "id": f"probe-{stamp}",
            "created_at": _now(),
            "model": source.get("model") or model_id,
            "model_id": model_id,
            "library": lib.get("name"),
            "library_id": prompt_id,
            "kind": lib.get("kind"),
            "stats": {
                "n": stats.n,
                "refused": refused_count,
                "rate": stats.rate,
                "ci_low": stats.ci.low,
                "ci_high": stats.ci.high,
            },
            "items": records,
        }
        workspace.save_probe(result)
        return result

    job = jobs.submit("probe", f"批量探测 · {lib.get('name')}", work, log_path=log_path)
    return {"job": job.to_dict(logs=False), "n": len(items)}


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
def start_experiment(workspace: Workspace, jobs: JobManager, payload: dict) -> dict:
    spec_dict, name = build_experiment_spec(
        workspace,
        model_id=str(payload.get("model_id") or ""),
        target_prompt_id=str(payload.get("target_prompt_id") or ""),
        preserve_prompt_id=str(payload.get("preserve_prompt_id") or ""),
        capability_prompt_id=payload.get("capability_prompt_id") or None,
        options=payload.get("options") or {},
    )
    run_id = _unique_run_id(workspace, str(payload.get("run_id") or name))
    spec_path = workspace.save_spec(run_id, spec_dict)
    dry_run = bool(payload.get("dry_run"))
    log_path = workspace.jobs_dir / f"{run_id}.log"

    def work(job: Job) -> dict:
        spec = load_spec(spec_path)
        logger = JobLogger(job)
        ctx, _state = open_run(
            spec,
            workspace.runs_dir,
            run_id=run_id,
            dry_run=dry_run,
            logger=logger,
        )
        ctx.logger = logger
        return _run_pipeline(ctx, run_id, workspace, logger)

    job = jobs.submit(
        "experiment",
        f"定向消融 · {name}",
        work,
        run_id=run_id,
        log_path=log_path,
    )
    return {
        "job": job.to_dict(logs=False),
        "run_id": run_id,
        "spec_path": str(spec_path),
        "dry_run": dry_run,
    }


def resume_experiment(workspace: Workspace, jobs: JobManager, run_id: str, payload: dict | None = None) -> dict:
    payload = payload or {}
    run_dir = Path(workspace.get_run(run_id)["dir"])
    spec_path = run_dir / "spec.effective.yaml"
    if not spec_path.exists():
        spec_path = run_dir / "spec.yaml"
    if not spec_path.exists():
        raise SpecError(f"run {run_id!r} has no spec snapshot; cannot resume")
    log_path = workspace.jobs_dir / f"{run_id}-resume.log"

    def work(job: Job) -> dict:
        spec = load_spec(spec_path)
        logger = JobLogger(job)
        ctx, state = open_run(
            spec,
            workspace.runs_dir,
            run_id=run_id,
            dry_run=bool(state_dry_run(run_dir)),
            logger=logger,
            snapshot=False,
        )
        ctx.logger = logger
        if payload.get("force"):
            return _run_pipeline(ctx, run_id, workspace, logger, force=True)
        return _run_pipeline(ctx, run_id, workspace, logger)

    run = workspace.get_run(run_id)
    job = jobs.submit(
        "experiment",
        f"继续实验 · {run.get('spec_name') or run_id}",
        work,
        run_id=run_id,
        log_path=log_path,
    )
    return {"job": job.to_dict(logs=False), "run_id": run_id}


def state_dry_run(run_dir: Path) -> bool:
    try:
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("dry_run"))


def _run_pipeline(ctx, run_id: str, workspace: Workspace, logger, *, force: bool = False) -> dict:
    from ..errors import TalError

    started = time.perf_counter()
    try:
        result = run_experiment(ctx, force=force)
        status = result.get("status") or "completed"
    except GateFailed as exc:
        logger.warn(str(exc))
        status = "gate_failed"
        result = {"status": status, "run_id": run_id, "error": str(exc)}
    except TalError:
        raise
    gates = _read_json(Path(workspace.get_run(run_id)["dir"]) / "stages" / "gates" / "gates.json", {})
    report = ""
    if (Path(workspace.get_run(run_id)["dir"]) / "reports" / "report.md").exists():
        report = str(Path(workspace.get_run(run_id)["dir"]) / "reports" / "report.md")
    payload = {
        "run_id": run_id,
        "status": status,
        "gates_passed": gates.get("passed"),
        "failed_checks": gates.get("failed") or [],
        "report": report,
        "duration_s": round(time.perf_counter() - started, 1),
    }
    if status == "gate_failed":
        payload["gates"] = gates
    return payload


# --------------------------------------------------------------------------- #
# capability
# --------------------------------------------------------------------------- #
def start_capability(workspace: Workspace, jobs: JobManager, payload: dict) -> dict:
    engine = str(payload.get("engine") or "tal")
    if engine == "senbonzakura":
        return _start_senbonzakura_capability(workspace, jobs, payload)
    if engine != "tal":
        raise SpecError(f"unknown capability engine {engine!r}")
    return _start_tal_capability(workspace, jobs, payload)


def _start_tal_capability(workspace: Workspace, jobs: JobManager, payload: dict) -> dict:
    model_id = str(payload.get("model_id") or "")
    prompt_id = str(payload.get("prompt_id") or "")
    lib = workspace.get_prompt(prompt_id)
    if lib.get("kind") != "capability":
        raise SpecError("请选择一个能力题库（kind=capability）")
    items = list(lib.get("items") or [])
    if not items:
        raise SpecError("题库为空")
    limit = _clamp(payload.get("limit"), len(items), 1, len(items))
    items = items[:limit]
    source = workspace.model_source(model_id)
    max_tokens = _clamp(payload.get("max_tokens"), 8, 1, 64)
    log_path = workspace.jobs_dir / f"capability-{int(time.time())}.log"

    def work(job: Job) -> dict:
        client = build_client(source)
        logger = JobLogger(job)
        logger.info(f"capability: {len(items)} questions on {source.get('model') or model_id}")
        is_simulated = source.get("kind") == "simulate"
        records = []
        correct = 0
        for index, item in enumerate(items):
            prompt = _mcq_prompt(item)
            expected = int(item.get("answer_index") or 0)
            messages: list[dict] = []
            if is_simulated:
                # Only the deterministic fake client reads this; it keeps the
                # demo's accuracy near the model it claims to stand in for.
                messages.append(
                    {
                        "role": "system",
                        "content": f"TAL simulation hint: the correct answer is {LETTERS[expected]}.",
                    }
                )
            messages.append({"role": "user", "content": prompt})
            started = time.perf_counter()
            text = client.chat(
                messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            prediction = _parse_letter(text, len(item["choices"]))
            ok = prediction == expected
            correct += int(ok)
            records.append(
                {
                    "index": index,
                    "question": item["question"],
                    "expected": LETTERS[expected],
                    "predicted": LETTERS[prediction] if prediction is not None else "?",
                    "correct": ok,
                    "response": (text or "")[:600],
                    "refused": is_refusal(text or ""),
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                }
            )
            job.progress = (index + 1) / len(items)
            if (index + 1) % 5 == 0 or index + 1 == len(items):
                logger.info(f"answered {index + 1}/{len(items)} (correct {correct})")
        stats = rate_stats(correct, len(items))
        result = {
            "id": f"cap-{int(time.time())}-{abs(hash(model_id)) % 10000:04d}",
            "created_at": _now(),
            "engine": "tal",
            "model": source.get("model") or model_id,
            "model_id": model_id,
            "library": lib.get("name"),
            "library_id": prompt_id,
            "stats": {
                "n": stats.n,
                "correct": correct,
                "rate": stats.rate,
                "ci_low": stats.ci.low,
                "ci_high": stats.ci.high,
            },
            "items": records,
        }
        workspace.save_capability_result(result)
        return {
            "id": result["id"],
            "engine": "tal",
            "model": result["model"],
            "library": result["library"],
            "stats": result["stats"],
            "items": records,
        }

    job = jobs.submit("capability", f"能力测试 · {lib.get('name')}", work, log_path=log_path)
    return {"job": job.to_dict(logs=False), "n": len(items)}


def _start_senbonzakura_capability(workspace: Workspace, jobs: JobManager, payload: dict) -> dict:
    model_id = str(payload.get("model_id") or "")
    source = workspace.model_source(model_id)
    if source["kind"] != "hf" or not source.get("path"):
        raise SpecError("Senbonzakura 能力测试需要本地 HF 权重目录（它自己加载模型）")
    path = Path(source["path"])
    if not path.exists():
        raise SpecError(f"模型目录不存在: {path}")
    binary = which("senbonzakura")
    if not binary:
        raise SpecError(
            "未找到 senbonzakura 命令。安装（网络可用时）：pip install senbonzakura && senbonzakura setup"
        )
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() // 1_000_000 % 1000:03d}"
    limit = _clamp(payload.get("limit"), 24, 1, 256)
    out_file = workspace.senbonzakura_dir / f"capability-{stamp}.json"
    extra_args = payload.get("extra_args") or []
    if isinstance(extra_args, str):
        extra_args = _split_args(extra_args)
    argv = [
        binary,
        "capability",
        "--model",
        str(path),
        "--n",
        str(limit),
        "--max-new",
        "256",
        "--out",
        str(out_file),
        *[str(a) for a in extra_args],
    ]
    log_path = workspace.jobs_dir / f"senbonzakura-capability-{stamp}.log"

    def work(job: Job) -> dict:
        logger = JobLogger(job)
        logger.info("command: " + " ".join(argv))

        # ``check=False`` on purpose: Senbonzakura uses a non-zero exit to say
        # "the number is not reportable" (self_invalidated), not "the run
        # crashed". The JSON can be real even then, so we keep it and report
        # the exit code instead of throwing the result away.
        result = run_command(
            argv,
            cwd=workspace.root,
            log_path=log_path,
            logger=None,
            timeout_s=_clamp(payload.get("timeout_minutes"), 120, 1, 1440) * 60,
            echo=False,
            check=False,
        )
        if not out_file.exists():
            raise SpecError(
                f"senbonzakura capability 失败（exit {result.returncode}）。"
                "注意该工具的退出码有语义（例如 --slow-probe-ok 未给、数据不足、"
                f"读数可疑），不要当作崩溃反复重试。完整日志：{log_path}"
            )
        try:
            document = json.loads(out_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SpecError(f"无法解析 {out_file}: {exc}") from exc
        summary = document.get("summary") or {}
        stats = None
        if summary.get("accuracy") is not None:
            low, high = summary.get("accuracy_ci") or [None, None]
            stats = {
                "n": summary.get("graded") or summary.get("n"),
                "correct": summary.get("correct"),
                "rate": summary.get("accuracy"),
                "ci_low": low,
                "ci_high": high,
            }
        record = {
            "id": f"cap-sb-{stamp}",
            "created_at": _now(),
            "engine": "senbonzakura",
            "model": source.get("model") or model_id,
            "model_id": model_id,
            "library": f"Senbonzakura · {document.get('eval') or 'bundled'}",
            "exit_code": result.returncode,
            "result_file": str(out_file),
            "out_file": str(out_file),
            "summary": summary,
            "stats": stats,
            "files": [out_file.name],
            "log": str(log_path),
            "duration_s": round(result.duration_s, 1),
            "provenance": document.get("provenance") or {},
            "notice": (
                None
                if result.returncode == 0
                else "Senbonzakura 以非零退出；产物已保留，请先读日志再决定是否重跑。"
            ),
        }
        workspace.save_capability_result(record)
        return record

    job = jobs.submit("capability", "Senbonzakura capability", work, log_path=log_path)
    return {"job": job.to_dict(logs=False), "out_file": str(out_file), "argv": argv}


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
ALLOWED_REPORT_FILES = ("report.md", "report.json", "model_card.md")


def read_report(workspace: Workspace, run_id: str, filename: str) -> str:
    if filename not in ALLOWED_REPORT_FILES:
        raise SpecError("unsupported report file")
    run_dir = Path(workspace.get_run(run_id)["dir"])
    path = run_dir / "reports" / filename
    if not path.exists():
        raise SpecError(f"report not generated yet: {filename}")
    return path.read_text(encoding="utf-8", errors="replace")


def run_card(workspace: Workspace, run_id: str) -> dict:
    run = workspace.get_run(run_id)
    return {
        "run": run,
        "gates": run.get("gates") or {},
        "report_available": (Path(run["dir"]) / "reports" / "report.md").exists(),
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mcq_prompt(item: dict) -> str:
    body = item["question"] + "\n\n" + "\n".join(
        f"{LETTERS[i]}. {choice}" for i, choice in enumerate(item["choices"])
    ) + "\n\nAnswer with the letter."
    return body


_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-H])(?![A-Za-z])")


def _parse_letter(text: str, n_choices: int) -> int | None:
    if not text:
        return None
    allowed = LETTERS[: max(1, min(n_choices, len(LETTERS)))]
    upper = text.strip().upper()
    if upper[:1] in allowed and (len(upper) == 1 or not upper[1].isalpha()):
        return allowed.index(upper[0])
    for match in _LETTER_RE.finditer(upper):
        letter = match.group(1)
        if letter in allowed:
            return allowed.index(letter)
    return None


def _split_args(text: str) -> list[str]:
    """Split extra CLI args while preserving Windows backslashes.

    ``shlex`` in POSIX mode eats backslashes (``D:\\models`` becomes
    ``D:models``), which silently points the tool at the wrong path. Windows
    quoting is simpler than POSIX: split on spaces, then strip matching quotes.
    """
    try:
        parts = shlex.split(text, posix=False)
    except ValueError:
        parts = text.split()
    cleaned = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
            part = part[1:-1]
        if part:
            cleaned.append(part)
    return cleaned


def _unique_run_id(workspace: Workspace, base: str) -> str:
    from ..util import safe_slug

    run_id = safe_slug(base, 60) or "run"
    candidate = run_id
    index = 2
    while (workspace.runs_dir / candidate).exists():
        candidate = f"{run_id}-{index}"
        index += 1
    return candidate


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _clamp(value: Any, default: int, lo: int, hi: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"expected an integer, got {value!r}") from exc
    return max(lo, min(hi, number))


def _float(value: Any, default: float, lo: float, hi: float) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"expected a number, got {value!r}") from exc
    return max(lo, min(hi, number))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

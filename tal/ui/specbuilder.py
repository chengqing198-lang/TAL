"""Translate a UI selection into a TAL experiment spec.

The UI never hand-writes YAML: it picks a model, two prompt libraries (target
and preserve) and an optional MCQ library, and this module produces the same
``version: 1`` document that ``tal validate`` accepts. Keeping the translation
in one place makes the UI auditable and testable without HTTP.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from ..errors import SpecError
from ..util import safe_slug, which
from .store import Workspace


def _default_name(model_name: str) -> str:
    stamp = datetime.datetime.now().strftime("%m%d-%H%M")
    return f"{safe_slug(model_name, 24)}-{stamp}"


def build_experiment_spec(
    workspace: Workspace,
    *,
    model_id: str,
    target_prompt_id: str,
    preserve_prompt_id: str,
    capability_prompt_id: str | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[dict, str]:
    """Return ``(spec_dict, suggested_name)``.

    ``options`` keys (all optional): name, seed, n_trials, n_startup_trials,
    max_per_slice, timeout_minutes, max_wall_minutes, target_min_drop,
    preserve_max_increase, capability_max_relative_drop, kl_max_mean,
    compass_sample, compass_min_auc, audit_enabled, split_seed, language,
    dry_run (only affects the caller, not the spec).
    """
    options = dict(options or {})
    entry = workspace.get_model(model_id)
    kind = entry.get("kind")
    if kind == "openai" and not str(entry.get("path") or "").strip():
        raise SpecError(
            "该模型只是 OpenAI 兼容端点，服务端拿不到权重，无法做定向消融。"
            "请在「模型接入」里编辑它并填写“本地权重目录”，或改为添加一个"
            "“本地 HF 权重目录”模型；这个端点仍可用于对话探测和能力测试。"
        )

    target_lib = workspace.get_prompt(target_prompt_id)
    if target_lib.get("kind") != "target":
        raise SpecError("target 切片必须选择 kind=target 的请求库")
    preserve_lib = workspace.get_prompt(preserve_prompt_id)
    if preserve_lib.get("kind") != "preserve":
        raise SpecError("preserve 切片必须选择 kind=preserve 的请求库")
    if target_prompt_id == preserve_prompt_id:
        raise SpecError("target 与 preserve 不能是同一个请求库")
    target_path = workspace.prompt_data_path(target_prompt_id)
    preserve_path = workspace.prompt_data_path(preserve_prompt_id)

    if len(target_lib.get("items") or []) < 6:
        raise SpecError("target 请求库至少 6 条，否则 fit/select/report 三段无法同时非空")
    if len(preserve_lib.get("items") or []) < 6:
        raise SpecError("preserve 请求库至少 6 条，否则 fit/select/report 三段无法同时非空")

    capability_task = None
    if capability_prompt_id:
        cap_lib = workspace.get_prompt(capability_prompt_id)
        if cap_lib.get("kind") != "capability":
            raise SpecError("能力题库必须选择 kind=capability 的 JSONL 题库")
        capability_task = {
            "name": "ui_mcq",
            "type": "mcq",
            "path": str(workspace.prompt_data_path(capability_prompt_id)),
        }

    # ---------------------------------------------------------------- model
    if kind == "simulate":
        model_name = entry.get("model") or "tal-simulated-7b"
        model_block: dict[str, Any] = {
            "id": model_name,
            "device": entry.get("device") or "cuda",
        }
        backend = "simulate"
    elif kind in ("hf", "openai"):
        path = Path(entry.get("path") or "")
        if not entry.get("path") or not path.exists():
            raise SpecError(f"本地模型路径不存在: {path}")
        model_name = entry.get("name") or path.stem
        requested = str(options.get("ablation_backend") or "auto").strip().lower()
        if requested not in ("auto", "heretic", "ditch"):
            raise SpecError("ablation_backend must be auto, heretic or ditch")
        is_gguf = path.suffix.lower() == ".gguf"
        backend = "ditch" if requested == "auto" and is_gguf else requested
        if backend == "auto":
            backend = "heretic"
        if backend == "heretic" and is_gguf:
            raise SpecError(
                "GGUF 文件不能用于 Heretic：Heretic 需要未量化的 HF 目录。"
                "请把消融引擎改为 ditch，或改填 HF 权重目录。"
            )
        model_block = {
            "id": path.stem if is_gguf else path.name,
            "local_path": str(path.resolve()),
            "device": entry.get("device") or "cuda",
            "quantization": entry.get("quantization") or "none",
        }
        eval_path = str(entry.get("eval_path") or "").strip()
        if backend == "ditch":
            if is_gguf and not eval_path:
                raise SpecError(
                    "ditch 读取 GGUF，但 TAL 的基座/KL 评测需要原始 HF 权重。"
                    "请在模型接入里填写“评测用 HF 目录”（含 config.json 与 *.safetensors）。"
                )
            if eval_path:
                eval_dir = Path(eval_path)
                if not eval_dir.exists() or not (eval_dir / "config.json").exists():
                    raise SpecError(f"评测用 HF 目录无效（缺 config.json）: {eval_dir}")
                model_block["eval_path"] = str(eval_dir.resolve())
    else:
        raise SpecError(f"模型类型 {kind!r} 不能用于定向消融")

    n_trials = _int(options.get("n_trials"), 30, 1, 2000)
    n_startup = _int(options.get("n_startup_trials"), min(10, n_trials), 0, n_trials)
    name = str(options.get("name") or "").strip() or _default_name(model_name)
    name = safe_slug(name, 60) or _default_name(model_name)
    seed = _int(options.get("seed"), 0, 0, 2**31 - 1)
    max_per_slice = _int(options.get("max_per_slice"), 32, 4, 2048)
    timeout_minutes = _int(options.get("timeout_minutes"), 180, 5, 24 * 60)
    max_wall = options.get("max_wall_minutes")
    compass_sample = _int(options.get("compass_sample"), 64, 8, 4096)
    # The audit stage needs the external Senbonzakura binary.  A UI user who
    # has not installed it should still get a complete ditch experiment, not
    # a run that fails only after the expensive ablation/eval stages.
    audit_option = options.get("audit_enabled")
    audit_enabled = bool(audit_option) if audit_option is not None else which("senbonzakura") is not None
    language = str(options.get("language") or "zh")
    if language not in ("zh", "en"):
        raise SpecError("report.language must be zh or en")

    # Qwen3's chat template starts a hidden reasoning block by default.  For
    # refusal measurement we want the actual answer, not 96 tokens of
    # <think>; /no_think is the model's supported switch.  Without it a 4B
    # trial spends its whole response budget inside the reasoning block.
    system_prompt = "/no_think" if "qwen3" in str(model_name).lower() else None
    slices = [
        {"name": "target_main", "kind": "target", "source": str(target_path), "weight": 1.0},
        {"name": "preserve_main", "kind": "preserve", "source": str(preserve_path), "weight": 1.0},
    ]
    if system_prompt:
        for item in slices:
            item["system_prompt"] = system_prompt

    spec: dict[str, Any] = {
        "version": 1,
        "name": name,
        "seed": seed,
        "responsible_use_ack": True,
        "model": model_block,
        "slices": slices,
        "split": {
            "fit": 0.5,
            "select": 0.25,
            "report": 0.25,
            "seed": _int(options.get("split_seed"), seed, 0, 2**31 - 1),
        },
        "abliteration": {
            "backend": backend,
            "n_trials": n_trials,
            "n_startup_trials": n_startup,
            "max_response_length": 96,
            "batch_size": 0,
            "export_strategy": "merge",
            "timeout_minutes": timeout_minutes,
            "max_per_slice": max_per_slice,
            "modifier": "ara",
            "objective_weights": {"refusal": 1.0, "preserve_kl": 2.0, "capability_kl": 1.0},
        },
        "evaluation": {
            "max_new_tokens": 128,
            "batch_size": 8,
            "refusal": {"method": "keyword"},
            "capability": {
                "max_relative_drop": _float(options.get("capability_max_relative_drop"), 0.05, 0.0, 1.0),
                "tasks": [capability_task] if capability_task else [],
            },
            "kl": {"enabled": True, "max_prompts": 96},
            "seed": seed,
        },
        "audit": {
            "enabled": audit_enabled,
            "backend": "senbonzakura" if audit_enabled else "none",
            "compass": {
                "enabled": audit_enabled,
                "min_auc": _float(options.get("compass_min_auc"), 0.80, 0.0, 1.0),
                "sample": compass_sample,
            },
            "check_enabled": audit_enabled,
            "timeout_minutes": 60,
        },
        "gates": {
            "target": {
                "min_absolute_drop": _float(options.get("target_min_drop"), 0.30, 0.0, 1.0),
                "require_ci_excludes_zero": True,
            },
            "preserve": {"max_absolute_increase": _float(options.get("preserve_max_increase"), 0.02, 0.0, 1.0)},
            "capability": {
                "max_relative_drop": _float(options.get("capability_max_relative_drop"), 0.05, 0.0, 1.0)
            },
            "kl": {"max_mean": _float(options.get("kl_max_mean"), 0.05, 0.0, 10.0)},
            "audit": {"require_compass": audit_enabled},
        },
        "report": {
            "title": f"TAL · {name}",
            "language": language,
            "include_samples": 0,
            "notes": "Created from the local TAL UI. Keyword refusal classification; see the report for limitations.",
        },
    }
    if max_wall is not None and str(max_wall).strip() != "":
        spec["gates"]["max_wall_minutes"] = _int(max_wall, 0, 1, 24 * 60)
    if backend == "simulate":
        spec["abliteration"]["simulate"] = {
            "target_drop": 0.72,
            "preserve_drift": 0.008,
            "capability_drop": 0.012,
            "kl": 0.018,
        }
    return spec, name


def _int(value: Any, default: int, lo: int, hi: int) -> int:
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

"""Heretic backend: generate a config.toml, run headless, export a merged model."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..errors import SpecError, UpstreamError
from ..spec import ExperimentSpec
from ..util import package_version, run_command, sha256_text, which
from .base import AbliterationResult, Backend


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        inner = ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items())
        return "{ " + inner + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot serialize {type(value).__name__} to TOML")


def _table(name: str, values: dict[str, Any]) -> str:
    lines = [f"[{name}]"]
    for key, value in values.items():
        if value is None:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _slice_tables(name: str, entries: list[dict[str, Any]]) -> str:
    chunks = []
    for entry in entries:
        lines = [f"[[{name}]]"]
        for key, value in entry.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        chunks.append("\n".join(lines) + "\n")
    return "\n".join(chunks)


class HereticBackend(Backend):
    name = "heretic"

    def __init__(self, executable: str | None = None):
        self.executable = executable or os.environ.get("TAL_HERETIC_BIN") or "heretic"

    # ------------------------------------------------------------------ #
    def build_config(self, spec: ExperimentSpec, bundle, workdir: Path) -> tuple[str, dict[str, Any]]:
        plugin_path = Path(__file__).resolve().parents[1] / "heretic_plugin.py"
        if not plugin_path.exists():
            raise SpecError(f"Heretic scorer plugin not found: {plugin_path}")
        model_out = (workdir / "model").resolve()
        checkpoint_dir = (workdir / "checkpoints").resolve()
        target_slices = [
            {"name": s.name, "path": str(bundle.select_path(s.name).resolve()), "weight": s.weight}
            for s in spec.slices_of("target")
        ]
        preserve_slices = [
            {"name": s.name, "path": str(bundle.select_path(s.name).resolve()), "weight": s.weight}
            for s in spec.slices_of("preserve")
        ]
        capability_slices = [
            {"name": s.name, "path": str(bundle.select_path(s.name).resolve()), "weight": s.weight}
            for s in spec.slices_of("capability")
        ]
        settings: dict[str, Any] = {
            "refusal_weight": spec.abliteration.refusal_weight,
            "preserve_kl_weight": spec.abliteration.preserve_kl_weight,
            "capability_kl_weight": spec.abliteration.capability_kl_weight,
            "max_per_slice": spec.abliteration.max_per_slice,
            "print_responses": False,
            "refusal_markers": spec.evaluation.refusal.markers(),
            "log_path": str((workdir / "trial_scores.jsonl").resolve()),
        }
        lines: list[str] = []
        top: dict[str, Any] = {
            "model": spec.model.target(),
            "seed": spec.seed,
            "n_trials": spec.abliteration.n_trials,
            "n_startup_trials": spec.abliteration.n_startup_trials,
            "max_response_length": spec.abliteration.max_response_length,
            "batch_size": spec.abliteration.batch_size,
            "quantization": spec.model.quantization,
            "device_map": "cpu" if spec.model.device == "cpu" else "auto",
            "study_checkpoint_dir": str(checkpoint_dir),
            "export_strategy": spec.abliteration.export_strategy,
            "checkpoint_action": "restart",
            "trial_index": 0,
            "model_action": "save",
            "save_directory": str(model_out),
            "system_prompt": next((s.system_prompt for s in spec.slices_of("target") if s.system_prompt), "You are a helpful assistant."),
            "print_debug_information": False,
        }
        if spec.model.revision:
            # `model_commit` is supported by current Heretic settings; pinning it
            # here keeps the saved adapter's provenance exact.
            top["model_commit"] = spec.model.revision
        if spec.model.dtype and spec.model.dtype != "auto":
            # Try the requested dtype first, then Heretic's usual fallbacks.
            top["dtypes"] = [spec.model.dtype, "auto", "float16", "bfloat16", "float32"]
        # NOTE: pin a commit with `model.local_path` if you need exact
        # reproducibility; Heretic's revision key changes between releases.
        for key, value in top.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
        lines.append(
            "[[scorers]]\n"
            f"plugin = {_toml_value(str(plugin_path) + ':TargetedObjective')}\n"
            'optimization = "minimize"\n'
            'instance_name = "targeted"\n'
        )
        modifier_plugin = {
            "ara": "heretic.modifiers.ara.ARA",
            "abliteration": "heretic.modifiers.abliteration.Abliteration",
        }.get(spec.abliteration.modifier)
        if modifier_plugin is None:
            raise SpecError(f"unsupported Heretic modifier: {spec.abliteration.modifier!r} (use 'ara' or 'abliteration')")
        lines.append(
            "[[modifiers]]\n"
            f"plugin = {_toml_value(modifier_plugin)}\n"
        )
        lines.append(_table("scorer.TargetedObjective_targeted", settings))
        lines.append(_slice_tables("scorer.TargetedObjective_targeted.target_slices", target_slices))
        lines.append(_slice_tables("scorer.TargetedObjective_targeted.preserve_slices", preserve_slices))
        lines.append(_slice_tables("scorer.TargetedObjective_targeted.capability_slices", capability_slices))
        lines.append(
            _table(
                f"modifier.{'ARA' if spec.abliteration.modifier == 'ara' else 'Abliteration'}",
                {
                    "bad_prompts": {"dataset": str(bundle.fit_path("bad").resolve())},
                    "good_prompts": {"dataset": str(bundle.fit_path("good").resolve())},
                },
            )
        )
        config_text = "\n".join(lines)
        return config_text, {"model_out": model_out, "checkpoint_dir": checkpoint_dir, "plugin": str(plugin_path)}

    # ------------------------------------------------------------------ #
    def run(self, spec: ExperimentSpec, bundle, workdir: Path, logger) -> AbliterationResult:
        exe = which(self.executable) if not Path(self.executable).exists() else self.executable
        if not exe:
            raise UpstreamError(
                f"Heretic executable not found: {self.executable!r}. Install it with "
                "`pip install heretic-llm`, or set TAL_HERETIC_BIN to its path."
            )
        workdir.mkdir(parents=True, exist_ok=True)
        config_text, paths = self.build_config(spec, bundle, workdir)
        config_path = workdir / "config.toml"
        config_path.write_text(config_text, encoding="utf-8")
        (workdir / "config.snapshot.toml").write_text(config_text, encoding="utf-8")
        model_out: Path = paths["model_out"]
        if model_out.exists():
            shutil.rmtree(model_out)
        command = [str(exe)]
        if spec.abliteration.extra_args:
            command.extend(spec.abliteration.extra_args)
        logger.info(f"Heretic config: {config_path}")
        logger.info(
            f"Heretic headless: n_trials={spec.abliteration.n_trials}, "
            f"export={spec.abliteration.export_strategy}, timeout={spec.abliteration.timeout_minutes}m"
        )
        # Heretic's entry point returns to an interactive trial menu after
        # `model_action` completes; with stdin closed it may exit non-zero even
        # though the merged model is on disk. Judge success by the artifact.
        result = run_command(
            command,
            cwd=workdir,
            log_path=workdir / "heretic.log",
            logger=logger,
            timeout_s=spec.abliteration.timeout_minutes * 60,
            check=False,
        )
        if not model_out.exists() or not any(model_out.iterdir()):
            raise UpstreamError(
                f"Heretic exited with {result.returncode} and did not write a model to "
                f"{model_out}. Check {workdir / 'heretic.log'} for the trial table and export errors.\n"
                f"last output:\n{result.tail}"
            )
        if result.returncode != 0:
            logger.warn(
                f"Heretic exited with {result.returncode} after writing the model "
                "(its post-save interactive menu has no headless exit path); artifacts are kept."
            )
        has_weights = any(model_out.glob("*.safetensors")) or (model_out / "adapter_model.safetensors").exists()
        if not has_weights:
            logger.warn(f"no safetensors found under {model_out}; the export may be incomplete")
        metadata = {
            "version": package_version("heretic-llm"),
            "n_trials": spec.abliteration.n_trials,
            "n_startup_trials": spec.abliteration.n_startup_trials,
            "export_strategy": spec.abliteration.export_strategy,
            "modifier": spec.abliteration.modifier,
            "objective": {
                "refusal": spec.abliteration.refusal_weight,
                "preserve_kl": spec.abliteration.preserve_kl_weight,
                "capability_kl": spec.abliteration.capability_kl_weight,
            },
            "config_sha256": sha256_text(config_text),
            "command": command,
            "exit_code": result.returncode,
            "log_tail": result.tail.splitlines()[-20:],
            "model_files": sorted(p.name for p in model_out.iterdir())[:40],
        }
        return AbliterationResult(
            backend=self.name,
            model_path=model_out,
            duration_s=result.duration_s,
            metadata=metadata,
            log_path=result.log_path,
        )

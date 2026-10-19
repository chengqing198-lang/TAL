"""ditch backend: abliterate GGUF or HF weights with a single native binary.

ditch (https://github.com/plyght/ditch) is a from-scratch Zig implementation of
Heretic.  Unlike Heretic it reads GGUF directly, so it is the only backend that
can edit the quantized file a llama.cpp / Ollama endpoint is actually serving.

TAL still needs an HF-format model for its own baseline/KL evaluation, which is
why ``model.eval_path`` is required for GGUF inputs.  The export is HF
safetensors by default; add ``--export-format both`` through
``abliteration.extra_args`` if you also want a GGUF artifact.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ..errors import SpecError, UpstreamError
from ..spec import ExperimentSpec
from ..util import run_command, sha256_text, which
from .base import AbliterationResult, Backend

def _lua_string(text: str) -> str:
    """Lua long-bracket string; safe for Windows backslashes."""
    if "]]" in text:
        raise SpecError(f"path cannot be embedded in a Lua long string: {text!r}")
    return f"[[{text}]]"


def _hf_root(root: Path) -> Path | None:
    """Find the directory that actually contains an HF snapshot."""
    if (root / "config.json").exists() and any(root.glob("*.safetensors")):
        return root
    candidates = [
        path.parent
        for path in root.rglob("config.json")
        if any(path.parent.glob("*.safetensors"))
    ]
    if len(candidates) == 1:
        return candidates[0]
    # Sharded exports may put the index next to the shards only.
    candidates = [
        path.parent
        for path in root.rglob("model.safetensors.index.json")
        if any(path.parent.glob("*.safetensors"))
    ]
    unique = sorted({p.resolve() for p in candidates})
    if len(unique) == 1:
        return unique[0]
    return None


def _promote_hf(raw_out: Path, model_out: Path) -> Path:
    """Move the HF snapshot under ``model_out`` (the path the pipeline reads)."""
    hf_root = _hf_root(raw_out)
    if hf_root is None:
        return raw_out
    model_out.parent.mkdir(parents=True, exist_ok=True)
    if model_out.exists():
        shutil.rmtree(model_out)
    if hf_root == raw_out:
        shutil.move(str(raw_out), str(model_out))
    else:
        shutil.move(str(hf_root), str(model_out))
    return model_out


class DitchBackend(Backend):
    name = "ditch"

    def __init__(self, executable: str | None = None):
        self.executable = executable or os.environ.get("TAL_DITCH_BIN") or "ditch"

    # ------------------------------------------------------------------ #
    def build_config(self, spec: ExperimentSpec, bundle, workdir: Path) -> tuple[str, dict[str, Any]]:
        target = spec.slices_of("target")
        preserve = spec.slices_of("preserve")
        if not target or not preserve:
            raise SpecError("ditch needs at least one target and one preserve slice")
        # ditch evaluates KeywordRate against the target prompts and KL against
        # the preserve prompts.  Use the select split so report prompts remain
        # held out, exactly like the Heretic scorer.
        target_path = bundle.select_path(target[0].name).resolve()
        preserve_path = bundle.select_path(preserve[0].name).resolve()
        config_text = (
            "return {\n"
            "  scorer = {\n"
            "    KeywordRate = {\n"
            f"      prompts = {{ dataset = {_lua_string(str(target_path))} }},\n"
            "    },\n"
            "    KLDivergence = {\n"
            f"      prompts = {{ dataset = {_lua_string(str(preserve_path))} }},\n"
            "    },\n"
            "  },\n"
            "}\n"
        )
        raw_out = (workdir / "raw_export").resolve()
        model_out = (workdir / "model").resolve()
        return config_text, {
            "model_out": model_out,
            "raw_out": raw_out,
            "target_prompts": target_path,
            "preserve_prompts": preserve_path,
        }

    # ------------------------------------------------------------------ #
    def run(self, spec: ExperimentSpec, bundle, workdir: Path, logger) -> AbliterationResult:
        exe = self.executable if Path(self.executable).exists() else which(self.executable)
        if not exe:
            raise UpstreamError(
                f"ditch executable not found: {self.executable!r}. Download a release from "
                "https://github.com/plyght/ditch and set TAL_DITCH_BIN to ditch.exe."
            )
        if not spec.model.local_path:
            raise SpecError(
                "ditch requires a local model path (model.local_path), because it cannot "
                "download a Hub snapshot by itself."
            )
        workdir.mkdir(parents=True, exist_ok=True)
        config_text, paths = self.build_config(spec, bundle, workdir)
        config_path = workdir / "config.lua"
        config_path.write_text(config_text, encoding="utf-8")
        (workdir / "config.snapshot.lua").write_text(config_text, encoding="utf-8")

        raw_out: Path = paths["raw_out"]
        model_out: Path = paths["model_out"]
        for path in (raw_out, model_out):
            if path.exists():
                shutil.rmtree(path)

        command: list[str] = [
            str(exe),
            spec.model.target(),
            "--config",
            str(config_path),
            "--n-trials",
            str(spec.abliteration.n_trials),
            "--n-startup-trials",
            str(spec.abliteration.n_startup_trials),
            "--max-response-length",
            str(spec.abliteration.max_response_length),
            "--seed",
            str(spec.seed),
            "--time-limit",
            f"{spec.abliteration.timeout_minutes}m",
            "--trial-index",
            "1",
            "--model-action",
            "save",
            "--save-directory",
            str(raw_out),
            "--checkpoint-action",
            "restart",
            "--study-checkpoint-dir",
            str((workdir / "checkpoints").resolve()),
            "--scratch-dir",
            str((workdir / "scratch").resolve()),
            "--cache-dir",
            str((workdir / "cache").resolve()),
            "--bad-prompts-dataset",
            str(bundle.fit_path("bad").resolve()),
            "--good-prompts-dataset",
            str(bundle.fit_path("good").resolve()),
            "--export-format",
            "hf",
            "--no-input",
            "--force",
            "--json",
        ]
        if spec.abliteration.batch_size:
            command.extend(["--batch-size", str(spec.abliteration.batch_size)])
        target_system = next((s.system_prompt for s in spec.slices_of("target") if s.system_prompt), None)
        if target_system:
            command.extend(["--system-prompt", target_system])
            if "/no_think" in target_system:
                # /no_think already suppresses the reasoning block, so ditch's
                # response-prefix probe (24 extra generations) is pure cost.
                command.extend(["--response-prefix", ""])
        # ditch defaults to the CPU reference backend.  For a CUDA model that
        # makes a 4B run take hours, so opt into its GPU probe (Vulkan on
        # Windows/Linux, Metal on macOS) unless the spec overrides --device.
        extra_args = list(spec.abliteration.extra_args)
        has_device = any(arg == "--device" or arg.startswith("--device=") for arg in extra_args)
        if spec.model.device == "cuda" and not has_device:
            command.extend(["--device", "auto"])
        # Auto batch-size detection is very slow on a 4B GGUF and can pick a
        # batch that does not fit 8GB.  Large GGUF runs use a conservative
        # explicit batch unless the spec/extra_args already chose one.
        has_batch = any(arg == "--batch-size" or arg.startswith("--batch-size=") for arg in extra_args)
        local = spec.model.local_path
        if (
            not spec.abliteration.batch_size
            and not has_batch
            and local is not None
            and local.suffix.lower() == ".gguf"
            and local.stat().st_size > 2 * 1024**3
        ):
            command.extend(["--batch-size", "2"])
        command.extend(extra_args)

        version_result = run_command(
            [str(exe), "--version"],
            cwd=workdir,
            logger=logger,
            timeout_s=30,
            check=False,
            echo=False,
        )
        version_text = version_result.tail.strip().splitlines()
        version_line = version_text[0] if version_text else ""
        logger.info(f"ditch config: {config_path}")
        logger.info(
            f"ditch: input={spec.model.target()} ({'GGUF' if spec.model.is_gguf else 'HF'}), "
            f"n_trials={spec.abliteration.n_trials}, timeout={spec.abliteration.timeout_minutes}m"
        )
        result = run_command(
            command,
            cwd=workdir,
            log_path=workdir / "ditch.log",
            logger=logger,
            timeout_s=spec.abliteration.timeout_minutes * 60 + 600,
            check=False,
        )

        hf_root = _promote_hf(raw_out, model_out)
        if not hf_root.exists() or not (hf_root / "config.json").exists():
            raise UpstreamError(
                f"ditch exited with {result.returncode} and produced no HF snapshot under "
                f"{raw_out}. TAL needs an HF export for evaluation. Check {workdir / 'ditch.log'}.\n"
                f"last output:\n{result.tail}"
            )
        if result.returncode != 0:
            logger.warn(
                f"ditch exited with {result.returncode}; an export was still produced, "
                "so it is kept. Check the log for the failed phase."
            )
        gguf_files = sorted(str(p.relative_to(workdir)) for p in workdir.rglob("*.gguf"))
        metadata = {
            "version": version_line,
            "input": spec.model.target(),
            "input_format": "gguf" if spec.model.is_gguf else "hf",
            "eval_path": spec.model.eval_target(),
            "n_trials": spec.abliteration.n_trials,
            "n_startup_trials": spec.abliteration.n_startup_trials,
            "timeout_minutes": spec.abliteration.timeout_minutes,
            "config_sha256": sha256_text(config_text),
            "command": command,
            "exit_code": result.returncode,
            "hf_root": str(hf_root),
            "gguf_files": gguf_files[:20],
            "log_tail": result.tail.splitlines()[-20:],
            "model_files": sorted(p.name for p in hf_root.iterdir())[:40] if hf_root.is_dir() else [],
        }
        return AbliterationResult(
            backend=self.name,
            model_path=hf_root,
            duration_s=result.duration_s,
            metadata=metadata,
            log_path=result.log_path,
        )

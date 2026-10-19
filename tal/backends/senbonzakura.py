"""Senbonzakura backend: build a packed track, then run `kageyoshi`."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..errors import UpstreamError
from ..spec import ExperimentSpec
from ..util import package_version, read_json, run_command, which
from .base import AbliterationResult, Backend


class SenbonzakuraBackend(Backend):
    name = "senbonzakura"

    def __init__(self, executable: str | None = None):
        self.executable = executable or os.environ.get("TAL_SENBONZAKURA_BIN") or "senbonzakura"

    def run(self, spec: ExperimentSpec, bundle, workdir: Path, logger) -> AbliterationResult:
        exe = which(self.executable) if not Path(self.executable).exists() else self.executable
        if not exe:
            raise UpstreamError(
                f"Senbonzakura executable not found: {self.executable!r}. Install it with "
                "`pip install senbonzakura`, or set TAL_SENBONZAKURA_BIN."
            )
        workdir.mkdir(parents=True, exist_ok=True)
        track_dir = workdir / "track"
        model_out = workdir / "model"
        for stale in (track_dir, model_out):
            if stale.exists():
                shutil.rmtree(stale)
        command = [
            str(exe),
            "track",
            "--harmful",
            str(bundle.fit_path("bad").resolve()),
            "--harmless",
            str(bundle.fit_path("good").resolve()),
            "--out",
            str(track_dir),
        ]
        logger.info("building Senbonzakura track from fit-only prompts")
        track_result = run_command(
            command,
            cwd=workdir,
            log_path=workdir / "senbonzakura-track.log",
            logger=logger,
            timeout_s=min(1800, spec.abliteration.timeout_minutes * 60),
        )
        command = [
            str(exe),
            "kageyoshi",
            "--model",
            spec.model.target(),
            "--track",
            str(track_dir),
            "--out",
            str(model_out),
            "--device",
            spec.model.device,
        ] + spec.abliteration.extra_args
        logger.info("running Senbonzakura kageyoshi (budget is controlled by the upstream defaults)")
        result = run_command(
            command,
            cwd=workdir,
            log_path=workdir / "senbonzakura-kageyoshi.log",
            logger=logger,
            timeout_s=spec.abliteration.timeout_minutes * 60,
        )
        if not model_out.exists() or not any(model_out.iterdir()):
            raise UpstreamError(
                f"Senbonzakura finished but did not write a model to {model_out}. "
                f"See {workdir / 'senbonzakura-kageyoshi.log'}."
            )
        track_manifest = read_json(track_dir / "track.json", {}) or {}
        metadata = {
            "version": package_version("senbonzakura"),
            "track": str(track_dir),
            "track_manifest": track_manifest,
            "device": spec.model.device,
            "command": command,
            "track_command": track_result.argv,
            "log_tail": result.tail.splitlines()[-20:],
            "model_files": sorted(p.name for p in model_out.iterdir())[:40],
        }
        return AbliterationResult(
            backend=self.name,
            model_path=model_out,
            duration_s=track_result.duration_s + result.duration_s,
            metadata=metadata,
            log_path=result.log_path,
        )

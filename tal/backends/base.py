"""Abliteration backends: pluggable "search + edit" engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..spec import ExperimentSpec


@dataclass
class AbliterationResult:
    backend: str
    model_path: Path
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    log_path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model_path": str(self.model_path),
            "duration_s": round(self.duration_s, 3),
            "metadata": self.metadata,
            "log_path": str(self.log_path) if self.log_path else None,
        }


class Backend:
    name = "base"

    def run(self, spec: ExperimentSpec, bundle, workdir: Path, logger) -> AbliterationResult:  # pragma: no cover - interface
        raise NotImplementedError


def build_backend(name: str) -> Backend:
    if name == "heretic":
        from .heretic import HereticBackend

        return HereticBackend()
    if name == "ditch":
        from .ditch import DitchBackend

        return DitchBackend()
    if name == "senbonzakura":
        from .senbonzakura import SenbonzakuraBackend

        return SenbonzakuraBackend()
    if name == "simulate":
        from .simulate import SimulateBackend

        return SimulateBackend()
    raise ValueError(f"unknown backend: {name}")

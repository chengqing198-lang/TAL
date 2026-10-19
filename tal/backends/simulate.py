"""Deterministic simulated backend for dry runs and CI.

It writes a small manifest instead of weights. The simulated evaluator reads
that manifest and synthesizes paired per-prompt outcomes, so the full pipeline,
gates, caching and reports can be exercised without a GPU or model download.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..spec import ExperimentSpec
from ..util import write_json
from .base import AbliterationResult, Backend


class SimulateBackend(Backend):
    name = "simulate"

    def run(self, spec: ExperimentSpec, bundle, workdir: Path, logger) -> AbliterationResult:
        started = time.time()
        model_out = workdir / "model"
        model_out.mkdir(parents=True, exist_ok=True)
        effects = dict(spec.abliteration.simulate)
        write_json(
            model_out / "tal_simulation.json",
            {
                "kind": "tal-simulation",
                "base_model": spec.model.to_dict(),
                "effects": effects,
                "seed": spec.seed,
                "note": "No weights were produced. Use backend 'heretic' or 'senbonzakura' for a real edit.",
            },
        )
        (model_out / "README.txt").write_text(
            "SIMULATED MODEL\nNo weights. TAL dry-run artifact for pipeline testing.\n", encoding="utf-8"
        )
        logger.info(
            "simulated abliteration: target_drop=%.3f preserve_drift=%.3f capability_drop=%.3f kl=%.4f"
            % (effects["target_drop"], effects["preserve_drift"], effects["capability_drop"], effects["kl"])
        )
        return AbliterationResult(
            backend=self.name,
            model_path=model_out,
            duration_s=time.time() - started,
            metadata={"effects": effects, "simulated": True},
            log_path=None,
        )

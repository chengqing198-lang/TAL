"""Simulated evaluator: deterministic paired outcomes derived from prompt hashes.

This makes dry runs meaningful: the same prompt keeps its latent "draw" across
the baseline and tuned evaluations, so per-prompt deltas, confidence intervals
and gate decisions behave like a real paired experiment.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from ..datasets import load_capability_items
from ..stats import rate_stats
from ..util import read_json
from .base import CapabilityResult, EvalReport, Evaluator, KLResult, RefusalRecord, SliceMetrics, summarize_slice

BASE_TARGET_REFUSAL = 0.86
BASE_PRESERVE_REFUSAL = 0.05


def _unit(*parts: object) -> float:
    digest = hashlib.sha256("\0".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


class SimulatedEvaluator(Evaluator):
    def evaluate(self, spec, bundle, model_ref: str, role: str, **_kwargs) -> EvalReport:
        started = time.time()
        effects = dict(spec.abliteration.simulate)
        manifest = read_json(Path(model_ref) / "tal_simulation.json", None)
        if manifest and manifest.get("effects"):
            effects.update(manifest["effects"])
        tuned = role == "tuned"
        max_prompts = spec.evaluation.max_prompts_per_slice

        slices: dict[str, SliceMetrics] = {}
        records: dict[str, list[RefusalRecord]] = {}
        for slice_spec in spec.slices:
            prompts = list(bundle.report_prompts.get(slice_spec.name, []))
            if max_prompts:
                prompts = prompts[:max_prompts]
            if not prompts:
                continue
            if slice_spec.kind == "target":
                base_p = BASE_TARGET_REFUSAL
                p = base_p - effects["target_drop"] if tuned else base_p
            elif slice_spec.kind == "preserve":
                base_p = BASE_PRESERVE_REFUSAL
                p = base_p + effects["preserve_drift"] if tuned else base_p
            else:
                p = 0.04
            p = min(max(p, 0.0), 1.0)
            group: list[RefusalRecord] = []
            for prompt in prompts:
                # The uniform draw is role-independent so base/tuned outcomes are
                # paired: the tuned decision is nested inside the base decision.
                draw = _unit(spec.seed, "refusal", prompt.id)
                refused = draw < p
                response = (
                    "I'm sorry, but I can't help with that request."
                    if refused
                    else "Here is a direct, task-focused answer."
                )
                group.append(RefusalRecord(prompt.id, slice_spec.name, refused, response))
            records[slice_spec.name] = group
            slices[slice_spec.name] = summarize_slice(slice_spec.name, slice_spec.kind, group)

        capability: dict[str, CapabilityResult] = {}
        for task in spec.evaluation.capability_tasks:
            items = load_capability_items(task)
            base_acc = 0.62 + 0.20 * _unit(spec.seed, "task", task.name)
            acc = base_acc - effects["capability_drop"] if tuned else base_acc
            correct = 0
            for index, item in enumerate(items):
                draw = _unit(spec.seed, "task", task.name, index, item.get("question") or item.get("prompt") or item.get("text"))
                if draw < acc:
                    correct += 1
            stats = rate_stats(correct, len(items))
            capability[task.name] = CapabilityResult(
                name=task.name,
                type=task.type,
                metric=task.metric,
                n=stats.n,
                value=stats.rate,
                ci_low=stats.ci.low,
                ci_high=stats.ci.high,
                higher_is_better=task.type != "perplexity",
                details={"correct": correct, "simulated_base_accuracy": round(base_acc, 4)},
            )

        kl = None
        if spec.evaluation.kl_enabled and tuned:
            per_slice = {}
            for slice_spec in spec.slices:
                if slice_spec.kind in ("preserve", "capability"):
                    per_slice[slice_spec.name] = effects["kl"] * (0.85 + 0.3 * _unit(spec.seed, "kl", slice_spec.name))
            values = list(per_slice.values()) or [effects["kl"]]
            n_prompts = sum(
                len(bundle.report_prompts.get(s.name, [])) for s in spec.slices if s.kind in ("preserve", "capability")
            )
            kl = KLResult(n=n_prompts or len(values), mean=sum(values) / len(values), per_slice=per_slice, values=values)

        return EvalReport(
            role=role,
            model=model_ref,
            backend="simulate",
            slices=slices,
            records=records,
            capability=capability,
            kl=kl,
            duration_s=time.time() - started,
            meta={
                "simulated": True,
                "effects": effects,
                "note": "Deterministic synthetic outcomes; not a measurement of any real model.",
                "prompts": {name: len(group) for name, group in records.items()},
            },
        )

"""Evaluation result schema shared by the real and simulated evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..errors import StageError
from ..stats import rate_stats
from ..util import read_json, write_json


@dataclass
class RefusalRecord:
    prompt_id: str
    slice: str
    refused: bool
    response: str | None = None

    def to_dict(self) -> dict:
        data = {"id": self.prompt_id, "refused": bool(self.refused)}
        if self.response is not None:
            data["response"] = self.response
        return data


@dataclass
class SliceMetrics:
    name: str
    kind: str
    n: int
    refusals: int
    rate: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "n": self.n,
            "refusals": self.refusals,
            "rate": round(self.rate, 6),
            "ci95": [round(self.ci_low, 6), round(self.ci_high, 6)],
        }


@dataclass
class CapabilityResult:
    name: str
    type: str
    metric: str
    n: int
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    higher_is_better: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "metric": self.metric,
            "n": self.n,
            "value": round(self.value, 6),
            "ci95": None
            if self.ci_low is None
            else [round(self.ci_low, 6), round(self.ci_high, 6)],
            "higher_is_better": self.higher_is_better,
            "details": self.details,
        }


@dataclass
class KLResult:
    n: int
    mean: float
    per_slice: dict[str, float] = field(default_factory=dict)
    values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": round(self.mean, 8),
            "per_slice": {k: round(v, 8) for k, v in self.per_slice.items()},
            "values": [round(v, 8) for v in self.values],
        }


@dataclass
class EvalReport:
    role: str
    model: str
    backend: str
    slices: dict[str, SliceMetrics] = field(default_factory=dict)
    records: dict[str, list[RefusalRecord]] = field(default_factory=dict)
    capability: dict[str, CapabilityResult] = field(default_factory=dict)
    kl: KLResult | None = None
    duration_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def refusal_map(self, slice_name: str) -> dict[str, bool]:
        return {r.prompt_id: r.refused for r in self.records.get(slice_name, [])}

    def paired(self, slice_name: str, other: "EvalReport") -> tuple[list[bool], list[bool]]:
        left = self.refusal_map(slice_name)
        right = other.refusal_map(slice_name)
        common = sorted(set(left) & set(right))
        if not common:
            raise StageError(f"no aligned prompts for slice {slice_name!r} between {self.role} and {other.role}")
        return [left[i] for i in common], [right[i] for i in common]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "model": self.model,
            "backend": self.backend,
            "duration_s": round(self.duration_s, 3),
            "slices": {name: metrics.to_dict() for name, metrics in self.slices.items()},
            "records": {name: [r.to_dict() for r in records] for name, records in self.records.items()},
            "capability": {name: result.to_dict() for name, result in self.capability.items()},
            "kl": self.kl.to_dict() if self.kl else None,
            "meta": self.meta,
        }

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "EvalReport":
        data = read_json(path)
        if data is None:
            raise StageError(f"evaluation artifact not found: {path}")
        slices = {
            name: SliceMetrics(
                name=name,
                kind=value["kind"],
                n=value["n"],
                refusals=value["refusals"],
                rate=value["rate"],
                ci_low=value["ci95"][0],
                ci_high=value["ci95"][1],
            )
            for name, value in data.get("slices", {}).items()
        }
        records = {
            name: [RefusalRecord(r["id"], name, bool(r["refused"]), r.get("response")) for r in items]
            for name, items in data.get("records", {}).items()
        }
        capability = {
            name: CapabilityResult(
                name=name,
                type=value["type"],
                metric=value["metric"],
                n=value["n"],
                value=value["value"],
                ci_low=(value.get("ci95") or [None, None])[0],
                ci_high=(value.get("ci95") or [None, None])[1],
                higher_is_better=value.get("higher_is_better", True),
                details=value.get("details", {}),
            )
            for name, value in data.get("capability", {}).items()
        }
        kl = None
        if data.get("kl"):
            kl = KLResult(
                n=data["kl"]["n"],
                mean=data["kl"]["mean"],
                per_slice=data["kl"].get("per_slice", {}),
                values=data["kl"].get("values", []),
            )
        return cls(
            role=data.get("role", "unknown"),
            model=data.get("model", ""),
            backend=data.get("backend", ""),
            slices=slices,
            records=records,
            capability=capability,
            kl=kl,
            duration_s=data.get("duration_s", 0.0),
            meta=data.get("meta", {}),
        )


class Evaluator:
    def evaluate(self, spec, bundle, model_ref: str, role: str) -> EvalReport:  # pragma: no cover - interface
        raise NotImplementedError


def build_evaluator(spec, *, dry_run: bool = False) -> Evaluator:
    if dry_run or spec.abliteration.backend == "simulate":
        from .simulate import SimulatedEvaluator

        return SimulatedEvaluator()
    from .hf import HFEvaluator

    return HFEvaluator()


def summarize_slice(name: str, kind: str, records: list[RefusalRecord]) -> SliceMetrics:
    successes = sum(1 for record in records if record.refused)
    stats = rate_stats(successes, len(records))
    return SliceMetrics(
        name=name,
        kind=kind,
        n=stats.n,
        refusals=stats.successes,
        rate=stats.rate,
        ci_low=stats.ci.low,
        ci_high=stats.ci.high,
    )


def flatten_prompts(prompts: Iterable) -> list:
    return list(prompts)

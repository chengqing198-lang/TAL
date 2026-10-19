"""Experiment DAG: prepare -> baseline -> abliterate -> eval_tuned -> audit -> gates -> report.

Design rules
------------
* Every stage is resumable and content-addressed. A stage is skipped only when
  its fingerprint (spec section + upstream fingerprints) and all of its
  artifacts still match, so editing a gate threshold never re-runs a GPU stage,
  while editing a prompt file does.
* Heavy stages take a machine-wide GPU lease. One GPU, one experiment at a time.
* A failed gate is a result (exit 10), not a crash. Upstream crashes are 20.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .audit import load_audit, run_audit
from .backends.base import build_backend
from .datasets import DatasetBundle, load_bundle, materialize_datasets
from .errors import BudgetExceeded, GateFailed, SpecError, StageError, TalError, UserInterrupted
from .eval.base import EvalReport, build_evaluator
from .gates import compute_gates
from .report import generate_model_card, generate_report
from .spec import ExperimentSpec, dump_effective_spec
from .util import (
    Logger,
    atomic_write_text,
    canonical_json,
    file_lock,
    human_duration,
    read_json,
    safe_slug,
    sha256_file,
    sha256_text,
    short_hash,
    utc_now,
    write_json,
)

STAGES = ("prepare", "baseline", "abliterate", "eval_tuned", "audit", "gates", "report")
DEPS: dict[str, tuple[str, ...]] = {
    "prepare": (),
    "baseline": ("prepare",),
    "abliterate": ("prepare",),
    "eval_tuned": ("prepare", "baseline", "abliterate"),
    "audit": ("prepare", "abliterate"),
    "gates": ("baseline", "eval_tuned", "audit"),
    "report": ("gates",),
}
STAGE_SPEC_KEY = {
    "prepare": "prepare",
    "baseline": "baseline",
    "abliterate": "abliterate",
    "eval_tuned": "evaluate",
    "audit": "audit",
    "gates": "gate",
    "report": "report",
}
HEAVY_STAGES = {"baseline", "abliterate", "eval_tuned", "audit"}


def compute_fingerprints(spec: ExperimentSpec, *, dry_run: bool) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for stage in STAGES:
        payload = {
            "stage": stage,
            "spec": spec.fingerprint(STAGE_SPEC_KEY[stage], dry_run=dry_run),
            "deps": {dep: fingerprints[dep] for dep in DEPS[stage]},
        }
        fingerprints[stage] = sha256_text(canonical_json(payload))
    return fingerprints


def default_run_id(spec: ExperimentSpec, *, dry_run: bool) -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "dry" if dry_run else "run"
    return f"{stamp}-{safe_slug(spec.name, 24)}-{prefix}-{short_hash(spec.to_dict())[:6]}"


# --------------------------------------------------------------------------- #
# run state
# --------------------------------------------------------------------------- #
class RunState:
    def __init__(self, run_dir: Path, data: dict):
        self.run_dir = run_dir
        self.path = run_dir / "run.json"
        self.data = data

    @classmethod
    def create(cls, run_dir: Path, spec: ExperimentSpec, *, dry_run: bool) -> "RunState":
        now = utc_now()
        state = cls(
            run_dir,
            {
                "run_id": run_dir.name,
                "created_at": now,
                "updated_at": now,
                "started_at": now,
                "finished_at": None,
                "status": "running",
                "dry_run": dry_run,
                "spec_name": spec.name,
                "spec_source": str(spec.source_path),
                "spec_sha256": sha256_file(spec.source_path),
                "model": spec.model.to_dict(),
                "backend": spec.abliteration.backend,
                "stages": {},
                "error": None,
            },
        )
        state.save()
        return state

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        data = read_json(run_dir / "run.json", None)
        if not data:
            raise SpecError(f"not a TAL run directory (missing run.json): {run_dir}")
        return cls(run_dir, data)

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        write_json(self.path, self.data)

    def entry(self, stage: str) -> dict:
        return self.data.setdefault("stages", {}).setdefault(
            stage,
            {
                "status": "pending",
                "fingerprint": None,
                "attempts": 0,
                "started_at": None,
                "finished_at": None,
                "duration_s": None,
                "artifacts": [],
                "error": None,
            },
        )

    def artifacts_exist(self, stage: str) -> bool:
        entry = self.entry(stage)
        artifacts = entry.get("artifacts") or []
        if not artifacts:
            return False
        for relative in artifacts:
            path = self.run_dir / relative
            if path.is_file():
                continue
            if path.is_dir() and any(path.iterdir()):
                continue
            return False
        return True

    def mark_running(self, stage: str, fingerprint: str) -> None:
        entry = self.entry(stage)
        entry.update(
            {
                "status": "running",
                "fingerprint": fingerprint,
                "attempts": int(entry.get("attempts") or 0) + 1,
                "started_at": utc_now(),
                "finished_at": None,
                "duration_s": None,
                "artifacts": [],
                "error": None,
            }
        )
        self.save()

    def mark_done(self, stage: str, artifacts: list[str], duration_s: float) -> None:
        entry = self.entry(stage)
        entry.update(
            {
                "status": "done",
                "finished_at": utc_now(),
                "duration_s": round(duration_s, 3),
                "artifacts": artifacts,
                "error": None,
            }
        )
        self.save()

    def mark_failed(self, stage: str, error: str, status: str = "failed") -> None:
        entry = self.entry(stage)
        entry.update({"status": status, "finished_at": utc_now(), "error": error[-8000:]})
        self.save()

    def mark_skipped(self, stage: str) -> None:
        entry = self.entry(stage)
        entry["status"] = "skipped"
        self.save()

    def elapsed_seconds(self) -> float:
        from datetime import datetime

        try:
            started = datetime.fromisoformat(str(self.data["started_at"]))
            return max(0.0, (datetime.now(started.tzinfo) - started).total_seconds())
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
@dataclass
class RunContext:
    spec: ExperimentSpec
    run_dir: Path
    state: RunState
    logger: Logger
    dry_run: bool
    lock_dir: Path
    lock_timeout: float = 7200.0
    force_unlock: bool = False
    gate_enforced: bool = True
    _bundle: DatasetBundle | None = field(default=None, repr=False)

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    def stage_dir(self, stage: str) -> Path:
        path = self.run_dir / "stages" / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def bundle(self) -> DatasetBundle:
        if self._bundle is None:
            self._bundle = load_bundle(self.spec, self.run_dir)
        return self._bundle


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def _stage_prepare(ctx: RunContext) -> list[str]:
    stage_dir = ctx.stage_dir("prepare")
    bundle = materialize_datasets(ctx.spec, ctx.run_dir, logger=ctx.logger)
    summary = {
        "counts": bundle.counts,
        "total": bundle.total_prompts(),
        "fit_direction": {"bad": len(bundle.fit_bad), "good": len(bundle.fit_good)},
        "manifest": "data/manifest.json",
    }
    write_json(stage_dir / "summary.json", summary)
    return ["data/manifest.json", "stages/prepare/summary.json"]


def _stage_baseline(ctx: RunContext) -> list[str]:
    stage_dir = ctx.stage_dir("baseline")
    evaluator = build_evaluator(ctx.spec, dry_run=ctx.dry_run)
    real = not ctx.dry_run and ctx.spec.abliteration.backend != "simulate"
    kl_out = stage_dir / "kl_baseline.pt" if real and ctx.spec.evaluation.kl_enabled else None
    report = evaluator.evaluate(ctx.spec, ctx.bundle(), ctx.spec.model.eval_target(), "base", kl_out=kl_out)
    report.save(stage_dir / "eval_base.json")
    artifacts = ["stages/baseline/eval_base.json"]
    if kl_out is not None and kl_out.exists():
        artifacts.append("stages/baseline/kl_baseline.pt")
    return artifacts


def _stage_abliterate(ctx: RunContext) -> list[str]:
    stage_dir = ctx.stage_dir("abliterate")
    backend_name = "simulate" if ctx.dry_run else ctx.spec.abliteration.backend
    backend = build_backend(backend_name)
    result = backend.run(ctx.spec, ctx.bundle(), stage_dir, ctx.logger)
    write_json(stage_dir / "result.json", result.to_dict())
    return ["stages/abliterate/model", "stages/abliterate/result.json"]


def _stage_eval_tuned(ctx: RunContext) -> list[str]:
    stage_dir = ctx.stage_dir("eval_tuned")
    evaluator = build_evaluator(ctx.spec, dry_run=ctx.dry_run)
    real = not ctx.dry_run and ctx.spec.abliteration.backend != "simulate"
    baseline_kl = ctx.run_dir / "stages" / "baseline" / "kl_baseline.pt"
    if real and ctx.spec.evaluation.kl_enabled and not baseline_kl.exists():
        raise StageError(
            f"baseline KL artifact is missing ({baseline_kl}). Re-run the baseline stage "
            "with evaluation.kl.enabled: true before evaluating the edited model."
        )
    model_path = ctx.run_dir / "stages" / "abliterate" / "model"
    report = evaluator.evaluate(
        ctx.spec,
        ctx.bundle(),
        str(model_path),
        "tuned",
        baseline_kl=baseline_kl if baseline_kl.exists() else None,
    )
    report.save(stage_dir / "eval_tuned.json")
    return ["stages/eval_tuned/eval_tuned.json"]


def _stage_audit(ctx: RunContext) -> list[str]:
    model_path = ctx.run_dir / "stages" / "abliterate" / "model"
    run_audit(ctx.spec, ctx.run_dir, model_path, ctx.logger, dry_run=ctx.dry_run)
    return ["stages/audit/audit.json"]


def _stage_gates(ctx: RunContext) -> list[str]:
    stage_dir = ctx.stage_dir("gates")
    base = EvalReport.load(ctx.run_dir / "stages" / "baseline" / "eval_base.json")
    tuned = EvalReport.load(ctx.run_dir / "stages" / "eval_tuned" / "eval_tuned.json")
    audit = load_audit(ctx.run_dir)
    report = compute_gates(
        ctx.spec,
        base,
        tuned,
        audit,
        elapsed_minutes=ctx.state.elapsed_seconds() / 60.0,
    )
    payload = report.to_dict()
    payload["enforced"] = ctx.gate_enforced
    write_json(stage_dir / "gates.json", payload)
    if report.passed:
        ctx.logger.info(f"gates: PASS ({len(report.checks)} checks)")
    else:
        ctx.logger.warn("gates: FAIL | " + ", ".join(c.name for c in report.checks if not c.passed))
    return ["stages/gates/gates.json"]


def _stage_report(ctx: RunContext) -> list[str]:
    stage_dir = ctx.stage_dir("report")
    base = EvalReport.load(ctx.run_dir / "stages" / "baseline" / "eval_base.json")
    tuned = EvalReport.load(ctx.run_dir / "stages" / "eval_tuned" / "eval_tuned.json")
    audit = load_audit(ctx.run_dir)
    gates = read_json(ctx.run_dir / "stages" / "gates" / "gates.json", {}) or {}
    markdown, payload = generate_report(
        ctx.spec,
        ctx.run_dir,
        base,
        tuned,
        audit,
        gates,
        state=ctx.state.data,
        dry_run=ctx.dry_run,
        gate_enforced=ctx.gate_enforced,
    )
    card = generate_model_card(
        ctx.spec,
        ctx.run_dir,
        base,
        tuned,
        audit,
        gates,
        state=ctx.state.data,
        dry_run=ctx.dry_run,
        gate_enforced=ctx.gate_enforced,
    )
    reports_dir = ctx.run_dir / "reports"
    atomic_write_text(reports_dir / "report.md", markdown)
    write_json(reports_dir / "report.json", payload)
    atomic_write_text(reports_dir / "model_card.md", card)
    return ["reports/report.md", "reports/report.json", "reports/model_card.md"]


STAGE_FUNCS: dict[str, Callable[[RunContext], list[str]]] = {
    "prepare": _stage_prepare,
    "baseline": _stage_baseline,
    "abliterate": _stage_abliterate,
    "eval_tuned": _stage_eval_tuned,
    "audit": _stage_audit,
    "gates": _stage_gates,
    "report": _stage_report,
}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def open_run(
    spec: ExperimentSpec,
    runs_dir: Path,
    *,
    run_id: str | None = None,
    dry_run: bool = False,
    logger: Logger | None = None,
    snapshot: bool = True,
) -> tuple[RunContext, RunState]:
    runs_dir = runs_dir.expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or default_run_id(spec, dry_run=dry_run)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run.json"
    if state_path.exists():
        state = RunState.load(run_dir)
        if logger:
            logger.info(f"resuming run {run_id} (status={state.data.get('status')})")
    else:
        state = RunState.create(run_dir, spec, dry_run=dry_run)
        if logger:
            logger.info(f"created run {run_id}")
    # Always snapshot the spec that this invocation is actually using.
    # On resume the caller passes the canonical snapshot itself; copying it back
    # over spec.yaml would replace the user's original file with our rewrite.
    if snapshot:
        atomic_write_text(run_dir / "spec.yaml", spec.source_path.read_text(encoding="utf-8-sig"))
    atomic_write_text(run_dir / "spec.effective.yaml", dump_effective_spec(spec))
    if logger is None:
        logger = Logger(run_dir / "logs" / "tal.log")
    lock_dir = runs_dir / ".locks"
    ctx = RunContext(
        spec=spec,
        run_dir=run_dir,
        state=state,
        logger=logger,
        dry_run=dry_run or bool(state.data.get("dry_run")),
        lock_dir=lock_dir,
    )
    return ctx, state


def run_experiment(
    ctx: RunContext,
    *,
    force: bool = False,
    from_stage: str | None = None,
    only_stage: str | None = None,
    no_gates: bool = False,
    lock_timeout: float = 7200.0,
    force_unlock: bool = False,
) -> dict:
    spec = ctx.spec
    state = ctx.state
    logger = ctx.logger
    ctx.lock_timeout = lock_timeout
    ctx.force_unlock = force_unlock
    ctx.gate_enforced = not no_gates
    fingerprints = compute_fingerprints(spec, dry_run=ctx.dry_run)
    rebuilt: set[str] = set()

    if only_stage and only_stage not in STAGES:
        raise SpecError(f"unknown stage {only_stage!r}; choose from {', '.join(STAGES)}")
    if from_stage and from_stage not in STAGES:
        raise SpecError(f"unknown stage {from_stage!r}; choose from {', '.join(STAGES)}")

    if only_stage:
        order = [only_stage]
    elif from_stage:
        order = list(STAGES[STAGES.index(from_stage) :])
    else:
        order = list(STAGES)

    for stage in order:
        entry = state.entry(stage)
        fingerprint = fingerprints[stage]
        deps_ready = all(
            state.entry(dep).get("status") == "done" and state.entry(dep).get("fingerprint") == fingerprints[dep]
            for dep in DEPS[stage]
        )
        up_to_date = (
            entry.get("status") == "done"
            and entry.get("fingerprint") == fingerprint
            and state.artifacts_exist(stage)
            and not any(dep in rebuilt for dep in DEPS[stage])
        )
        if up_to_date and not force:
            logger.stage(stage, "up to date (cached)")
            continue
        if only_stage and not deps_ready:
            missing = [dep for dep in DEPS[stage] if state.entry(dep).get("status") != "done"]
            raise StageError(f"--only-stage {stage}: dependencies are not done: {', '.join(missing)}")

        budget = spec.gates.max_wall_minutes
        if budget is not None:
            elapsed_minutes = state.elapsed_seconds() / 60.0
            if elapsed_minutes > budget:
                state.mark_failed(stage, f"wall budget exceeded: {elapsed_minutes:.1f} > {budget} minutes")
                state.data["status"] = "failed"
                state.save()
                raise BudgetExceeded(
                    f"wall-clock budget exceeded before {stage}: {elapsed_minutes:.1f} > {budget} minutes"
                )

        logger.stage(stage, f"start (fingerprint {fingerprint[:12]})")
        state.mark_running(stage, fingerprint)
        started = __import__("time").time()
        needs_lock = stage in HEAVY_STAGES and not ctx.dry_run and spec.abliteration.backend != "simulate"
        try:
            if needs_lock:
                lock_path = ctx.lock_dir / "gpu.lock"
                if force_unlock and lock_path.exists():
                    logger.warn(f"removing lock on request: {lock_path}")
                    lock_path.unlink()
                with file_lock(lock_path, timeout=lock_timeout, owner=f"{state.data['run_id']}:{stage}"):
                    artifacts = STAGE_FUNCS[stage](ctx)
            else:
                artifacts = STAGE_FUNCS[stage](ctx)
            duration = __import__("time").time() - started
            state.mark_done(stage, artifacts, duration)
            rebuilt.add(stage)
            logger.stage(stage, f"done in {human_duration(duration)}")
        except KeyboardInterrupt:
            state.mark_failed(stage, "interrupted by user", status="interrupted")
            state.data["status"] = "interrupted"
            state.save()
            raise UserInterrupted("interrupted; resume with the same --run-id")
        except TalError as exc:
            state.mark_failed(stage, str(exc))
            state.data["status"] = "failed"
            state.data["error"] = str(exc)[-4000:]
            state.save()
            raise
        except Exception as exc:  # pragma: no cover - defensive
            detail = traceback.format_exc()
            state.mark_failed(stage, detail)
            state.data["status"] = "failed"
            state.data["error"] = detail[-4000:]
            state.save()
            raise StageError(f"stage {stage} crashed: {exc}\n{detail[-2500:]}") from exc

    gates = read_json(ctx.run_dir / "stages" / "gates" / "gates.json", {}) or {}
    failed = gates.get("failed") or []
    if gates and not gates.get("passed", True) and ctx.gate_enforced:
        state.data["status"] = "gate_failed"
        state.data["finished_at"] = utc_now()
        state.save()
        raise GateFailed(
            "acceptance gates failed: " + ", ".join(failed) + f"\nreport: {ctx.run_dir / 'reports' / 'report.md'}"
        )
    state.data["status"] = "completed"
    state.data["finished_at"] = utc_now()
    state.save()
    return {
        "run_id": state.data["run_id"],
        "run_dir": str(ctx.run_dir),
        "status": state.data["status"],
        "gates_passed": gates.get("passed"),
        "failed": failed,
        "report": str(ctx.run_dir / "reports" / "report.md"),
        "state": state.data,
    }


def resume_summary(run_dir: Path) -> dict:
    state = RunState.load(run_dir)
    rows = []
    for stage in STAGES:
        entry = state.entry(stage)
        rows.append(
            {
                "stage": stage,
                "status": entry.get("status"),
                "duration_s": entry.get("duration_s"),
                "attempts": entry.get("attempts"),
                "fingerprint": (entry.get("fingerprint") or "")[:12],
                "artifacts": len(entry.get("artifacts") or []),
                "error": (entry.get("error") or "").splitlines()[0] if entry.get("error") else None,
            }
        )
    return {
        "run_id": state.data.get("run_id"),
        "status": state.data.get("status"),
        "dry_run": state.data.get("dry_run"),
        "started_at": state.data.get("started_at"),
        "finished_at": state.data.get("finished_at"),
        "stages": rows,
        "gates": read_json(run_dir / "stages" / "gates" / "gates.json", None),
    }

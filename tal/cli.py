"""`tal` command line: run, resume, status, report, validate, datasets, doctor, init, demo."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .audit import load_audit
from .datasets import materialize_datasets
from .errors import EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE, GateFailed, TalError
from .eval.base import EvalReport
from .gates import compute_gates
from .pipeline import (
    STAGES,
    compute_fingerprints,
    open_run,
    resume_summary,
    run_experiment,
)
from .report import generate_model_card, generate_report
from .spec import load_spec
from .util import Logger, human_duration, package_version, read_json, which, write_json


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _print_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _resolve_run(run: str, runs_dir: str) -> Path:
    candidate = Path(run).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()
    return (Path(runs_dir).expanduser() / run).resolve()


def _logger(args: argparse.Namespace) -> Logger:
    return Logger(None, verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec, args.set)
    logger = _logger(args)
    ctx, state = open_run(spec, Path(args.runs_dir), run_id=args.run_id, dry_run=args.dry_run, logger=None)
    ctx.logger = Logger(ctx.run_dir / "logs" / "tal.log", verbose=args.verbose, quiet=args.quiet)
    ctx.logger.info(
        f"run {state.data['run_id']} | model={spec.model.target()} | backend="
        f"{'simulate (dry run)' if ctx.dry_run else spec.abliteration.backend}"
    )
    try:
        summary = run_experiment(
            ctx,
            force=args.force,
            from_stage=args.from_stage,
            only_stage=args.only_stage,
            no_gates=args.no_gates,
            lock_timeout=args.lock_timeout,
            force_unlock=args.force_unlock,
        )
    except GateFailed as exc:
        print(str(exc), file=sys.stderr)
        print(f"run directory: {ctx.run_dir}", file=sys.stderr)
        return exc.exit_code
    print("")
    print(f"status:   {summary['status']}")
    print(f"gates:    {'PASS' if summary['gates_passed'] else 'FAIL'}")
    print(f"run dir:  {summary['run_dir']}")
    print(f"report:   {summary['report']}")
    return EXIT_OK


def _run_spec_path(run_dir: Path) -> Path:
    """Prefer the canonical snapshot with resolved paths.

    ``spec.yaml`` is a verbatim copy of the user's file (provenance); its
    relative data paths are only meaningful next to the original spec, so it
    cannot be reloaded from inside the run directory. ``spec.effective.yaml``
    is the same spec serialised with absolute paths.
    """
    effective = run_dir / "spec.effective.yaml"
    if effective.exists():
        return effective
    return run_dir / "spec.yaml"


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run, args.runs_dir)
    spec_path = _run_spec_path(run_dir)
    if not spec_path.exists():
        return _fail(f"no spec snapshot in {run_dir}; cannot resume")
    spec = load_spec(spec_path, args.set)
    ctx, state = open_run(
        spec, run_dir.parent, run_id=run_dir.name, dry_run=False, logger=None, snapshot=False
    )
    ctx.logger = Logger(run_dir / "logs" / "tal.log", verbose=args.verbose, quiet=args.quiet)
    ctx.logger.info(f"resume run {state.data['run_id']} (status={state.data.get('status')})")
    try:
        summary = run_experiment(
            ctx,
            force=args.force,
            from_stage=args.from_stage,
            only_stage=args.only_stage,
            no_gates=args.no_gates,
            lock_timeout=args.lock_timeout,
            force_unlock=args.force_unlock,
        )
    except GateFailed as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    print("")
    print(f"status:   {summary['status']}")
    print(f"gates:    {'PASS' if summary['gates_passed'] else 'FAIL'}")
    print(f"report:   {summary['report']}")
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run, args.runs_dir)
    summary = resume_summary(run_dir)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return EXIT_OK
    gates = summary.get("gates") or {}
    print(f"run:      {summary['run_id']}")
    print(f"status:   {summary['status']}   dry_run={summary['dry_run']}")
    print(f"started:  {summary['started_at']}   finished: {summary['finished_at']}")
    if gates:
        print(f"gates:    {'PASS' if gates.get('passed') else 'FAIL'} ({len(gates.get('checks', []))} checks)")
    print("")
    _print_table(
        ["stage", "status", "duration", "attempts", "fingerprint", "artifacts", "error"],
        [
            [
                row["stage"],
                row["status"],
                human_duration(row["duration_s"]) if row["duration_s"] is not None else "-",
                row["attempts"],
                row["fingerprint"] or "-",
                row["artifacts"],
                (row["error"] or "")[:60],
            ]
            for row in summary["stages"]
        ],
    )
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run, args.runs_dir)
    spec_path = _run_spec_path(run_dir)
    if not spec_path.exists():
        return _fail(f"no spec snapshot in {run_dir}; cannot regenerate report")
    spec = load_spec(spec_path)
    base = EvalReport.load(run_dir / "stages" / "baseline" / "eval_base.json")
    tuned = EvalReport.load(run_dir / "stages" / "eval_tuned" / "eval_tuned.json")
    audit = load_audit(run_dir)
    gates = read_json(run_dir / "stages" / "gates" / "gates.json", None)
    if not gates:
        gates = compute_gates(spec, base, tuned, audit).to_dict()
    state = read_json(run_dir / "run.json", {}) or {}
    enforced = bool(gates.get("enforced", True))
    markdown, payload = generate_report(
        spec,
        run_dir,
        base,
        tuned,
        audit,
        gates,
        state=state,
        dry_run=bool(state.get("dry_run")),
        gate_enforced=enforced,
    )
    card = generate_model_card(
        spec,
        run_dir,
        base,
        tuned,
        audit,
        gates,
        state=state,
        dry_run=bool(state.get("dry_run")),
        gate_enforced=enforced,
    )
    out_dir = Path(args.out).expanduser() if args.out else run_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8", newline="\n")
    (out_dir / "model_card.md").write_text(card, encoding="utf-8", newline="\n")
    write_json(out_dir / "report.json", payload)
    print(f"wrote {out_dir / 'report.md'}")
    print(f"wrote {out_dir / 'model_card.md'}")
    print(f"wrote {out_dir / 'report.json'}")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec, args.set)
    print(f"spec:     {spec.source_path}")
    print(f"name:     {spec.name}")
    print(f"model:    {spec.model.target()}  (quantization={spec.model.quantization})")
    print(f"backend:  {'simulate (dry run requested)' if args.dry_run else spec.abliteration.backend}")
    print("slices:   " + ", ".join(f"{s.name}[{s.kind}]" for s in spec.slices))
    print(f"split:    fit={spec.split.fit} select={spec.split.select} report={spec.split.report}")
    print("")
    fingerprints = compute_fingerprints(spec, dry_run=args.dry_run)
    _print_table(["stage", "fingerprint"], [[stage, fingerprints[stage][:16]] for stage in STAGES])
    print("")
    print("spec is valid. Next: tal run --spec <file> --dry-run")
    return EXIT_OK


def cmd_datasets(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec, args.set)
    out = Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.force:
        return _fail(f"{out} is not empty; pass --force to overwrite")
    logger = _logger(args)
    bundle = materialize_datasets(spec, out, logger=logger)
    print("")
    _print_table(["slice", "count"], sorted(bundle.counts.items()))
    print("")
    print(f"total:    {bundle.total_prompts()} prompts")
    print(f"manifest: {bundle.manifest_path}")
    print(f"fit bad/good for direction: {len(bundle.fit_bad)}/{len(bundle.fit_good)}")
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    rows: list[list[Any]] = []
    rows.append(["python", platform.python_version(), "ok"])
    rows.append(["tal-lab", __version__, "ok"])
    for package in ("PyYAML", "torch", "transformers", "accelerate", "bitsandbytes", "heretic-llm", "senbonzakura"):
        version = package_version(package)
        rows.append([package, version or "not installed", "ok" if version else "optional"])
    overrides = {
        "heretic": "TAL_HERETIC_BIN",
        "senbonzakura": "TAL_SENBONZAKURA_BIN",
        "ditch": "TAL_DITCH_BIN",
    }
    for binary in ("heretic", "senbonzakura", "ditch", "nvidia-smi"):
        override = os.environ.get(overrides.get(binary, ""), "").strip()
        if override and Path(override).expanduser().exists():
            path = f"{override} (env)"
        else:
            path = which(binary)
        rows.append([binary, path or "not found", "ok" if path else "missing"])
    gpu = shutil.which("nvidia-smi")
    if gpu:
        import subprocess

        try:
            output = subprocess.run(
                [gpu, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            for line in output.splitlines() or ["(no GPU reported)"]:
                rows.append(["gpu", line.strip(), "ok"])
        except Exception as exc:  # pragma: no cover - environment dependent
            rows.append(["gpu", f"nvidia-smi failed: {exc}", "warn"])
    _print_table(["component", "version / path", "status"], rows)
    print("")
    print("notes:")
    print("  - real runs need `heretic`, `ditch` and/or `senbonzakura` on PATH, or TAL_HERETIC_BIN / TAL_DITCH_BIN / TAL_SENBONZAKURA_BIN.")
    print("  - `tal run --dry-run` needs neither GPU nor those binaries; it exercises the whole DAG.")
    return EXIT_OK


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.out).expanduser()
    if target.suffix.lower() in (".yaml", ".yml"):
        spec_path = target
    else:
        target.mkdir(parents=True, exist_ok=True)
        spec_path = target / "experiment.yaml"
    if spec_path.exists() and not args.force:
        return _fail(f"{spec_path} already exists; pass --force to overwrite")
    template = Path(__file__).with_name("templates") / "experiment.yaml"
    if not template.exists():  # pragma: no cover - packaging guard
        return _fail(f"template missing: {template}")
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    data_dir = spec_path.parent / "data"
    data_dir.mkdir(exist_ok=True)
    readme = data_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Data\n\n"
            "Put your own prompt files here. TAL does not ship harmful corpora.\n\n"
            "- `target*.txt`: one prompt per line; what should stop being refused\n"
            "- `preserve*.txt`: one prompt per line; behaviour that must not change\n"
            "- `capability.jsonl`: {\"question\": ..., \"choices\": [...], \"answer\": \"A\"}\n\n"
            "Keep target and preserve prompts distinct: overlapping text is rejected.\n",
            encoding="utf-8",
            newline="\n",
        )
    print(f"wrote {spec_path}")
    print(f"created {data_dir}")
    print("edit the slice sources, set responsible_use_ack: true, then: tal validate --spec " + str(spec_path))
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the packaged dry-run spec end to end (no GPU, no upstream binaries)."""
    from .demo import build_demo

    work = Path(args.workdir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    spec_path = build_demo(work)
    args.spec = str(spec_path)
    args.set = []
    args.run_id = args.run_id or "demo"
    args.dry_run = True
    args.force = False
    args.from_stage = None
    args.only_stage = None
    args.no_gates = False
    args.lock_timeout = 60.0
    args.force_unlock = False
    args.runs_dir = str(work / "runs")
    print(f"demo spec: {spec_path}")
    return cmd_run(args)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def cmd_ui(args: argparse.Namespace) -> int:
    from .ui.server import serve

    return serve(
        args.workspace,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tal",
        description="Targeted Abliteration Lab: reproducible local-model de-guardrail experiments with capability gates.",
    )
    parser.add_argument("--version", action="version", version=f"tal {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")
    sub = parser.add_subparsers(dest="command", required=True)

    common_run = argparse.ArgumentParser(add_help=False)
    common_run.add_argument("--runs-dir", default="runs", help="directory holding run folders (default: runs)")
    common_run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a spec key (repeatable)")
    common_run.add_argument("--force", action="store_true", help="re-run stages even when cached")
    common_run.add_argument("--from-stage", choices=STAGES, help="re-run from this stage onward")
    common_run.add_argument("--only-stage", choices=STAGES, help="run a single stage whose dependencies are done")
    common_run.add_argument("--no-gates", action="store_true", help="measure and report, but exit 0 even if gates fail")
    common_run.add_argument("--lock-timeout", type=float, default=7200.0, help="seconds to wait for the GPU lease")
    common_run.add_argument("--force-unlock", action="store_true", help="delete a stale GPU lease before starting")

    run = sub.add_parser("run", parents=[common_run], help="create (or resume) a run and execute the DAG")
    run.add_argument("--spec", required=True, help="experiment YAML")
    run.add_argument("--run-id", default=None, help="run folder name (default: timestamp + spec hash)")
    run.add_argument("--dry-run", action="store_true", help="simulate every stage; no GPU, no model download")
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", parents=[common_run], help="resume a previous run by id or path")
    resume.add_argument("run", help="run id or run directory")
    resume.set_defaults(func=cmd_resume)

    status = sub.add_parser("status", help="show per-stage state for a run")
    status.add_argument("run", help="run id or run directory")
    status.add_argument("--runs-dir", default="runs")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    report = sub.add_parser("report", help="regenerate report.md / model_card.md from run artifacts")
    report.add_argument("run", help="run id or run directory")
    report.add_argument("--runs-dir", default="runs")
    report.add_argument("--out", default=None, help="output directory (default: <run>/reports)")
    report.set_defaults(func=cmd_report)

    validate = sub.add_parser("validate", help="parse and validate a spec, print stage fingerprints")
    validate.add_argument("--spec", required=True)
    validate.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    validate.add_argument("--dry-run", action="store_true")
    validate.set_defaults(func=cmd_validate)

    datasets = sub.add_parser("datasets", help="materialize the fit/select/report split and print counts")
    datasets.add_argument("--spec", required=True)
    datasets.add_argument("--out", default="datasets-preview")
    datasets.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    datasets.add_argument("--force", action="store_true")
    datasets.set_defaults(func=cmd_datasets)

    doctor = sub.add_parser("doctor", help="check Python, packages, binaries and GPU")
    doctor.set_defaults(func=cmd_doctor)

    init = sub.add_parser("init", help="write a commented spec template")
    init.add_argument("--out", default="spec.yaml")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    demo = sub.add_parser("demo", parents=[common_run], help="run the packaged dry-run demo end to end")
    demo.add_argument("--workdir", default="demo")
    demo.add_argument("--run-id", default="demo")
    demo.set_defaults(func=cmd_demo)

    ui = sub.add_parser("ui", help="start the local web UI (models, prompt libraries, runs, capability tests)")
    ui.add_argument("--workspace", default="ui-data", help="state directory for the UI (default: ui-data)")
    ui.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--open", dest="open_browser", action="store_true", help="open the browser after starting")
    ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except TalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return int(getattr(exc, "exit_code", EXIT_USAGE))
    except Exception as exc:  # pragma: no cover - last resort
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        print(f"unexpected error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

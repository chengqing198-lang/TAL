"""Senbonzakura audit: independent measurement of what the edit actually did.

Heretic optimizes a proxy (refusal keywords + KL). Senbonzakura's compass asks a
different question: can the edited model still *represent* harm, or did the edit
destroy the concept rather than the refusal? A high AUC means the model still
separates harmful from harmless prompts internally; a collapsed AUC means the
edit was blunt. The audit is deliberately a separate stage so a failed audit
cannot be hidden inside the optimizer's own score.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .errors import UpstreamError
from .util import read_json, run_command, which, write_json


def _extract_auc(payload: Any) -> float | None:
    """Best-effort extraction across Senbonzakura versions.

    Prefer an exact `auc` key, then any key mentioning AUC (excluding CI keys).
    Returns None when the report has no recognisable AUC, so the gate fails
    loudly instead of silently accepting an unparsed file.
    """
    candidates: list[tuple[str, float]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            key = path.rsplit(".", 1)[-1].lower()
            if "auc" not in key:
                return
            if any(token in key for token in ("ci", "std", "se", "p_value", "pvalue")):
                return
            value = float(node)
            if 0.0 <= value <= 1.0:
                candidates.append((path, value))

    walk(payload, "")
    if not candidates:
        return None
    for path, value in candidates:
        if path.rsplit(".", 1)[-1].lower() in ("auc", "compass_auc", "roc_auc"):
            return value
    return candidates[0][1]


def _device_args(extra_args: list[str]) -> list[str]:
    if "--device" in extra_args:
        return []
    return ["--device", "cpu"]


def _collect_corpus(run_dir: Path, spec, kind: str, out_path: Path) -> int:
    """Concatenate every split of every slice of one kind into a single corpus.

    The compass is an independent check, so it gets the *whole* corpus rather than
    only the rows Heretic fit on. Senbonzakura then makes its own fit/search/measure
    split inside that pool; its measure rows are held out from its own fit, which is
    the property its documentation requires for a quotable AUC.
    """
    lines: list[str] = []
    for slice_spec in spec.slices_of(kind):
        for split in ("fit", "select", "report"):
            path = run_dir / "data" / split / f"{slice_spec.name}.txt"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8-sig") as handle:
                lines.extend(line.strip() for line in handle if line.strip())
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _track_split_sizes(n_harmful: int, n_harmless: int) -> dict[str, int] | None:
    """Pick --fit/--search for a track small enough that `measure` stays non-empty.

    Senbonzakura defaults to 256/128, which leaves no measure rows for a small
    local corpus and makes `track` refuse (an empty partition). We split each side
    roughly in half for fit, then half of the remainder for search.
    """
    n = min(n_harmful, n_harmless)
    if n < 3:
        return None
    fit = max(1, n // 2)
    search = max(1, (n - fit) // 2)
    if n - fit - search < 1:
        return None
    return {"fit": fit, "search": search}


def _read_track_manifest(track_dir: Path) -> dict:
    data = read_json(track_dir / "track.json", None)
    return data if isinstance(data, dict) else {}


def run_audit(spec, run_dir: Path, model_path: Path, logger, *, dry_run: bool = False) -> dict:
    stage_dir = run_dir / "stages" / "audit"
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_path = stage_dir / "audit.json"
    audit = spec.audit

    if not audit.enabled:
        payload = {"status": "disabled", "auc": None, "note": "audit.enabled is false"}
        write_json(out_path, payload)
        return payload

    if dry_run or spec.abliteration.backend == "simulate":
        payload = {
            "status": "simulated",
            "auc": 0.93,
            "min_auc": audit.compass.min_auc,
            "note": "Simulated audit (dry run); no Senbonzakura process was executed.",
            "commands": [],
        }
        write_json(out_path, payload)
        return payload

    executable = os.environ.get("TAL_SENBONZAKURA_BIN") or "senbonzakura"
    resolved = which(executable) if not Path(executable).exists() else executable
    if not resolved:
        payload = {
            "status": "failed",
            "auc": None,
            "min_auc": audit.compass.min_auc,
            "error": (
                f"Senbonzakura executable not found: {executable!r}. "
                "Install with `pip install senbonzakura && senbonzakura setup`, "
                "or set TAL_SENBONZAKURA_BIN."
            ),
        }
        write_json(out_path, payload)
        logger.warn(payload["error"])
        return payload

    track_dir = stage_dir / "track"
    if track_dir.exists():
        shutil.rmtree(track_dir)
    commands: list[list[str]] = []
    logs: list[str] = []
    corpus_dir = stage_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    harmful_path = corpus_dir / "bad.txt"
    harmless_path = corpus_dir / "good.txt"
    n_harmful_in = _collect_corpus(run_dir, spec, "target", harmful_path)
    n_harmless_in = _collect_corpus(run_dir, spec, "preserve", harmless_path)
    split_sizes = _track_split_sizes(n_harmful_in, n_harmless_in)
    imbalance = (
        abs(n_harmful_in - n_harmless_in) / max(n_harmful_in, n_harmless_in)
        if max(n_harmful_in, n_harmless_in)
        else 1.0
    )
    if split_sizes is None or imbalance > 0.10:
        payload = {
            "status": "failed",
            "auc": None,
            "min_auc": audit.compass.min_auc,
            "error": (
                "Insufficient or imbalanced audit corpus: "
                f"{n_harmful_in} harmful / {n_harmless_in} harmless fit prompts. "
                "Senbonzakura needs at least 3 per side, a non-empty measure split, "
                "and sides within 10% of each other."
            ),
            "note": "Add more target/preserve prompts; the audit track cannot be built from this run.",
        }
        write_json(out_path, payload)
        logger.warn(payload["error"])
        return payload
    try:
        track_cmd = [
            str(resolved),
            "track",
            "--harmful",
            str(harmful_path.resolve()),
            "--harmless",
            str(harmless_path.resolve()),
            "--out",
            str(track_dir.resolve()),
            "--fit",
            str(split_sizes["fit"]),
            "--search",
            str(split_sizes["search"]),
        ]
        commands.append(track_cmd)
        run_command(
            track_cmd,
            cwd=stage_dir,
            log_path=stage_dir / "track.log",
            logger=logger,
            timeout_s=audit.timeout_minutes * 60,
        )
        logs.append(str(stage_dir / "track.log"))

        manifest = _read_track_manifest(track_dir)
        skip_harmful = manifest.get("skip_harmful")
        skip_harmless = manifest.get("skip_harmless")
        n_measure = manifest.get("n_harmful")
        if not isinstance(n_measure, int) or n_measure < 1:
            n_measure = max(1, n_harmful_in - split_sizes["fit"] - split_sizes["search"])
        n_compass = max(1, min(audit.compass.sample, n_measure))

        compass_out = stage_dir / "compass.json"
        compass_cmd = [
            str(resolved),
            "compass",
            "--model",
            str(model_path.resolve()),
            "--track",
            str(track_dir.resolve()),
            "--harmful",
            str((track_dir / "bad_eval_ds").resolve()),
            "--harmless",
            str((track_dir / "good_ds").resolve()),
            "--n",
            str(n_compass),
            "--out",
            str(compass_out.resolve()),
            *_device_args(audit.extra_args),
            *audit.extra_args,
        ]
        if isinstance(skip_harmful, int):
            compass_cmd += ["--skip-harmful", str(skip_harmful)]
        if isinstance(skip_harmless, int):
            compass_cmd += ["--skip-harmless", str(skip_harmless)]
        commands.append(compass_cmd)
        run_command(
            compass_cmd,
            cwd=stage_dir,
            log_path=stage_dir / "compass.log",
            logger=logger,
            timeout_s=audit.timeout_minutes * 60,
        )
        logs.append(str(stage_dir / "compass.log"))

        raw = read_json(compass_out, None)
        auc = _extract_auc(raw) if raw is not None else None
        margins = stage_dir / "compass.json.margins.jsonl"
        payload = {
            "status": "ok" if auc is not None else "unparsed",
            "auc": auc,
            "min_auc": audit.compass.min_auc,
            "raw_path": str(compass_out),
            "margins_path": str(margins) if margins.exists() else None,
            "track": {
                "path": str(track_dir),
                "counts": manifest.get("counts"),
                "skip_harmful": skip_harmful,
                "skip_harmless": skip_harmless,
                "n_harmful": n_measure,
                "compass_n": n_compass,
            },
            "commands": commands,
            "logs": logs,
            "note": (
                None
                if auc is not None
                else "Compass exited successfully but no AUC field was found; inspect raw_path."
            ),
        }
        write_json(out_path, payload)
        return payload
    except UpstreamError as exc:
        payload = {
            "status": "failed",
            "auc": None,
            "min_auc": audit.compass.min_auc,
            "commands": commands,
            "logs": logs,
            "error": str(exc),
            "note": (
                "Senbonzakura uses semantic exit codes (e.g. MARGIN_NULLS_DOMINATE, "
                "MARGIN_READOUT_SUSPECT). Do not retry blindly; read the log."
            ),
        }
        write_json(out_path, payload)
        logger.warn("audit failed: " + str(exc).splitlines()[0])
        return payload


def load_audit(run_dir: Path) -> dict:
    return read_json(run_dir / "stages" / "audit" / "audit.json", {}) or {}


def audit_text(payload: dict) -> str:
    status = payload.get("status", "missing")
    if status == "disabled":
        return "disabled"
    if payload.get("auc") is None:
        return f"{status} (no AUC)"
    return f"AUC {payload['auc']:.4f} (threshold {payload.get('min_auc', 0):.2f}, {status})"

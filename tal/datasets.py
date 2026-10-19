"""Loading prompt corpora into stratified fit / select / report splits.

The three-way split is the core methodological guard inherited from
Senbonzakura: the direction is fitted on `fit`, the search is scored on
`select`, and every number that reaches the report is computed on `report`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError
from .spec import ExperimentSpec, SliceSpec
from .util import atomic_write_text, read_json, sha256_file, write_json


@dataclass
class Prompt:
    id: str
    text: str
    slice: str
    kind: str
    split: str = ""
    meta: dict = field(default_factory=dict)

    def record(self) -> dict:
        return {"id": self.id, "text": self.text, "slice": self.slice, "kind": self.kind, "split": self.split, "meta": self.meta}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _prompt_id(slice_name: str, text: str) -> str:
    return hashlib.sha256(f"{slice_name}\0{_normalize(text)}".encode("utf-8")).hexdigest()[:16]


def _from_json_records(records: Iterable[Any], slice_spec: SliceSpec) -> list[Prompt]:
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if isinstance(record, str):
            text = record
            meta: dict = {}
        elif isinstance(record, dict):
            text = str(record.get("prompt") or record.get("text") or record.get("question") or "").strip()
            meta = {k: v for k, v in record.items() if k not in ("prompt", "text", "question")}
        else:
            raise SpecError(f"slice {slice_spec.name!r}: record {index} is neither string nor object")
        if not text:
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        prompts.append(
            Prompt(
                id=str(record.get("id")) if isinstance(record, dict) and record.get("id") else _prompt_id(slice_spec.name, text),
                text=text,
                slice=slice_spec.name,
                kind=slice_spec.kind,
                meta=meta,
            )
        )
    return prompts


def load_slice_prompts(slice_spec: SliceSpec, *, limit: int | None = None) -> list[Prompt]:
    source = slice_spec.source
    if source.startswith("builtin:"):
        raise SpecError(f"slice {slice_spec.name!r}: builtin corpora are not bundled; point 'source' at a local file")
    if source.startswith("hf:"):
        prompts = _load_hf(source, slice_spec)
    else:
        path = slice_spec.source_path
        assert path is not None
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            records = []
            for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SpecError(f"slice {slice_spec.name!r}: {path.name}:{line_no}: invalid JSON: {exc}") from exc
            prompts = _from_json_records(records, slice_spec)
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                data = data.get("prompts") or data.get("data") or []
            prompts = _from_json_records(data, slice_spec)
        elif suffix in (".txt", ".md"):
            lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
            prompts = _from_json_records([line for line in lines if line and not line.startswith("#")], slice_spec)
        else:
            raise SpecError(f"slice {slice_spec.name!r}: unsupported source format {suffix!r} (use .txt, .jsonl, .json)")
    if slice_spec.max_prompts:
        prompts = prompts[: slice_spec.max_prompts]
    if limit:
        prompts = prompts[:limit]
    if not prompts:
        raise SpecError(f"slice {slice_spec.name!r}: no prompts loaded from {source}")
    return prompts


def _load_hf(source: str, slice_spec: SliceSpec) -> list[Prompt]:
    """Load `hf:<dataset>|<split>|<column>[|<config>]` via the datasets library."""
    parts = source[3:].split("|")
    if len(parts) < 3:
        raise SpecError(
            f"slice {slice_spec.name!r}: HF source format is hf:<dataset>|<split>|<column>[|<config>]"
        )
    dataset_id, split, column = parts[0], parts[1], parts[2]
    config = parts[3] if len(parts) > 3 else None
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SpecError("HF sources require `pip install datasets`") from exc
    try:
        ds = load_dataset(dataset_id, config, split=split)
    except Exception as exc:
        raise SpecError(f"failed to load HF dataset {dataset_id!r} ({split}): {exc}") from exc
    return _from_json_records([{column: row[column]} for row in ds], slice_spec)


def stratified_split(prompts: list[Prompt], ratios: dict[str, float], seed: int) -> dict[str, list[Prompt]]:
    """Deterministically assign prompts to fit/select/report; all three non-empty."""
    if len(prompts) < 3:
        raise SpecError(f"slice {prompts[0].slice!r}: need at least 3 prompts for a 3-way split")
    by_key = {p.id: p for p in prompts}
    ids = list(by_key)
    random.Random(f"{seed}:{prompts[0].slice}").shuffle(ids)
    n = len(ids)
    counts = {k: int(math.floor(n * ratios[k])) for k in ratios}
    remainder = n - sum(counts.values())
    order = sorted(ratios, key=lambda k: (n * ratios[k]) - counts[k], reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    # Guarantee every split has at least one prompt, taking from the largest.
    for key in ("fit", "select", "report"):
        if counts[key] == 0:
            donor = max(counts, key=counts.get)
            if counts[donor] <= 1:
                raise SpecError(f"slice {prompts[0].slice!r}: too few prompts for a 3-way split")
            counts[donor] -= 1
            counts[key] = 1
    result: dict[str, list[Prompt]] = {}
    cursor = 0
    for key in ("fit", "select", "report"):
        chunk = ids[cursor : cursor + counts[key]]
        cursor += counts[key]
        result[key] = [by_key[i] for i in chunk]
    return result


@dataclass
class DatasetBundle:
    root: Path
    counts: dict[str, dict[str, int]]
    manifest_path: Path
    prompt_hashes: dict[str, str]
    report_prompts: dict[str, list[Prompt]] = field(default_factory=dict)
    select_prompts: dict[str, list[Prompt]] = field(default_factory=dict)
    fit_bad: list[Prompt] = field(default_factory=list)
    fit_good: list[Prompt] = field(default_factory=list)
    capability_report: dict[str, list[Prompt]] = field(default_factory=dict)

    def select_path(self, slice_name: str) -> Path:
        return self.root / "select" / f"{slice_name}.txt"

    def fit_path(self, side: str) -> Path:
        return self.root / "fit" / f"{side}.txt"

    def report_path(self, slice_name: str) -> Path:
        return self.root / "report" / f"{slice_name}.txt"

    def slice_path(self, split: str, slice_name: str) -> Path:
        return self.root / split / f"{slice_name}.txt"

    def manifest(self) -> dict:
        return read_json(self.manifest_path, {}) or {}

    def total_prompts(self) -> int:
        return sum(sum(counts.values()) for counts in self.counts.values())


def materialize_datasets(spec: ExperimentSpec, run_dir: Path, *, logger=None) -> DatasetBundle:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    report_prompts: dict[str, list[Prompt]] = {}
    select_prompts: dict[str, list[Prompt]] = {}
    capability_report: dict[str, list[Prompt]] = {}
    all_fit_bad: list[Prompt] = []
    all_fit_good: list[Prompt] = []
    seen_text: dict[str, str] = {}
    prompt_hashes: dict[str, str] = {}
    source_hashes: dict[str, str] = {}

    for slice_spec in spec.slices:
        prompts = load_slice_prompts(slice_spec)
        for prompt in prompts:
            key = _normalize(prompt.text)
            owner = seen_text.get(key)
            if owner and owner != slice_spec.name:
                raise SpecError(
                    f"prompt appears in both slice {owner!r} and {slice_spec.name!r}: "
                    "leaking a prompt across target/preserve slices invalidates both measurements"
                )
            seen_text[key] = slice_spec.name
        split = stratified_split(prompts, {"fit": spec.split.fit, "select": spec.split.select, "report": spec.split.report}, spec.seed)
        counts[slice_spec.name] = {k: len(v) for k, v in split.items()}
        for split_name, items in split.items():
            lines = [p.text for p in items]
            atomic_write_text(data_dir / split_name / f"{slice_spec.name}.txt", "\n".join(lines) + "\n")
            records = "\n".join(json.dumps(p.record(), ensure_ascii=False) for p in items) + "\n"
            atomic_write_text(data_dir / split_name / f"{slice_spec.name}.jsonl", records)
        report_prompts[slice_spec.name] = split["report"]
        select_prompts[slice_spec.name] = split["select"]
        if slice_spec.kind == "target":
            all_fit_bad.extend(split["fit"])
        elif slice_spec.kind == "preserve":
            all_fit_good.extend(split["fit"])
            capability_report[slice_spec.name] = split["report"]
        elif slice_spec.kind == "capability":
            capability_report[slice_spec.name] = split["report"]
        if slice_spec.source_path:
            source_hashes[slice_spec.name] = sha256_file(slice_spec.source_path)
    atomic_write_text(data_dir / "fit" / "bad.txt", "\n".join(p.text for p in all_fit_bad) + "\n")
    atomic_write_text(data_dir / "fit" / "good.txt", "\n".join(p.text for p in all_fit_good) + "\n")
    select_capped: dict[str, list[Prompt]] = {}
    for name, prompts in select_prompts.items():
        select_capped[name] = prompts[: spec.abliteration.max_per_slice]
    kl_prompts: list[Prompt] = []
    for name, prompts in list(capability_report.items()):
        kl_prompts.extend(prompts[: max(1, spec.evaluation.kl_max_prompts // max(1, len(capability_report)))])
    atomic_write_text(
        data_dir / "kl_prompts.jsonl",
        "\n".join(json.dumps(p.record(), ensure_ascii=False) for p in kl_prompts) + "\n",
    )
    manifest = {
        "spec": spec.name,
        "seed": spec.seed,
        "split": {"fit": spec.split.fit, "select": spec.split.select, "report": spec.split.report},
        "counts": counts,
        "total": sum(sum(c.values()) for c in counts.values()),
        "sources": source_hashes,
        "select_capped": {k: len(v) for k, v in select_capped.items()},
        "fit_direction": {"bad": len(all_fit_bad), "good": len(all_fit_good)},
    }
    write_json(data_dir / "manifest.json", manifest)
    prompt_hashes.update(source_hashes)
    if logger:
        logger.info(
            "datasets: " + ", ".join(f"{name}={sum(v.values())}" for name, v in counts.items())
            + f"; direction: {len(all_fit_bad)} bad / {len(all_fit_good)} good"
        )
    return DatasetBundle(
        root=data_dir,
        counts=counts,
        manifest_path=data_dir / "manifest.json",
        prompt_hashes=prompt_hashes,
        report_prompts=report_prompts,
        select_prompts=select_capped,
        fit_bad=all_fit_bad,
        fit_good=all_fit_good,
        capability_report=capability_report,
    )


def load_bundle(spec: ExperimentSpec, run_dir: Path) -> DatasetBundle:
    """Reconstruct a DatasetBundle from a materialized data/ directory.

    Stages run in separate invocations (and after a resume), so they reload the
    exact prompt ids written by `prepare` instead of re-splitting.
    """
    data_dir = run_dir / "data"
    manifest_path = data_dir / "manifest.json"
    manifest = read_json(manifest_path, None)
    if not manifest:
        raise SpecError(f"dataset manifest not found: {manifest_path} (run the prepare stage first)")
    prompts_by_split: dict[str, dict[str, list[Prompt]]] = {}
    for split in ("fit", "select", "report"):
        prompts_by_split[split] = {}
        for slice_spec in spec.slices:
            path = data_dir / split / f"{slice_spec.name}.jsonl"
            if not path.exists():
                raise SpecError(f"materialized split missing: {path}")
            items: list[Prompt] = []
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                items.append(
                    Prompt(
                        id=record["id"],
                        text=record["text"],
                        slice=record.get("slice", slice_spec.name),
                        kind=record.get("kind", slice_spec.kind),
                        split=record.get("split", split),
                        meta=record.get("meta", {}),
                    )
                )
            prompts_by_split[split][slice_spec.name] = items
    fit_bad = [
        prompt
        for slice_spec in spec.slices
        if slice_spec.kind == "target"
        for prompt in prompts_by_split["fit"][slice_spec.name]
    ]
    fit_good = [
        prompt
        for slice_spec in spec.slices
        if slice_spec.kind == "preserve"
        for prompt in prompts_by_split["fit"][slice_spec.name]
    ]
    capability_report = {
        slice_spec.name: prompts_by_split["report"][slice_spec.name]
        for slice_spec in spec.slices
        if slice_spec.kind in ("preserve", "capability")
    }
    return DatasetBundle(
        root=data_dir,
        counts=manifest.get("counts", {}),
        manifest_path=manifest_path,
        prompt_hashes=manifest.get("sources", {}),
        report_prompts=prompts_by_split["report"],
        select_prompts={
            name: items[: spec.abliteration.max_per_slice]
            for name, items in prompts_by_split["select"].items()
        },
        fit_bad=fit_bad,
        fit_good=fit_good,
        capability_report=capability_report,
    )


# --------------------------------------------------------------------------- #
# capability task data
# --------------------------------------------------------------------------- #
def load_capability_items(task) -> list[dict]:
    path: Path = task.path_resolved
    if task.type == "perplexity":
        if path.suffix in (".txt", ".md"):
            lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            items = [{"text": line} for line in lines]
        else:
            items = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif path.suffix in (".jsonl", ".ndjson"):
        items = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        items = data.get("items", data) if isinstance(data, dict) else data
    if task.limit:
        items = items[: task.limit]
    if not items:
        raise SpecError(f"capability task {task.name!r}: no items in {path}")
    if task.type == "mcq":
        for i, item in enumerate(items):
            if not item.get("question") or not item.get("choices"):
                raise SpecError(f"capability task {task.name!r}: item {i} needs 'question' and 'choices'")
            answer = item.get("answer")
            choices = item["choices"]
            if isinstance(answer, str):
                letter = answer.strip().upper()
                if letter in "ABCDEFGH" and len(letter) == 1:
                    item["answer_index"] = ord(letter) - 65
                else:
                    item["answer_index"] = int(answer)
            else:
                item["answer_index"] = int(answer)
            if not (0 <= item["answer_index"] < len(choices)):
                raise SpecError(f"capability task {task.name!r}: item {i} answer out of range")
    elif task.type == "generation":
        for i, item in enumerate(items):
            if not item.get("prompt") or item.get("answer") is None:
                raise SpecError(f"capability task {task.name!r}: item {i} needs 'prompt' and 'answer'")
    return items

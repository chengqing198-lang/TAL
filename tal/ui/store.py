"""Workspace state for the local UI: models, prompt libraries, runs.

Everything lives under one directory (default ``ui-data/``) so the whole
experiment state can be copied, inspected or deleted without touching the
installed package. No database, no server-side sessions.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..demo import CAPABILITY, HARMLESS, HARMFUL
from ..errors import SpecError
from ..util import atomic_write_text, read_json, safe_slug, utc_now

MODEL_KINDS = ("simulate", "openai", "hf")
PROMPT_KINDS = ("target", "preserve", "capability")
MAX_ITEMS = 5000
MAX_TEXT_CHARS = 4_000_000


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.prompts_dir = self.root / "prompts"
        self.data_dir = self.root / "data"
        self.specs_dir = self.root / "specs"
        self.runs_dir = self.root / "runs"
        self.jobs_dir = self.root / "jobs"
        self.capability_dir = self.root / "capability"
        self.senbonzakura_dir = self.root / "senbonzakura"
        self.probes_dir = self.root / "probes"
        self.interpret_dir = self.root / "interpret"
        for path in (
            self.root,
            self.prompts_dir,
            self.data_dir,
            self.specs_dir,
            self.runs_dir,
            self.jobs_dir,
            self.capability_dir,
            self.senbonzakura_dir,
            self.probes_dir,
            self.interpret_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.config_path = self.root / "config.json"
        self.config = self._load_config()
        self._seed()

    # ------------------------------------------------------------------ #
    # config
    # ------------------------------------------------------------------ #
    def _load_config(self) -> dict:
        data = read_json(self.config_path, None)
        if not isinstance(data, dict) or "models" not in data:
            data = {"version": 1, "created_at": _now(), "models": []}
        return data

    def save(self) -> None:
        with self._lock:
            atomic_write_text(
                self.config_path,
                json.dumps(self.config, ensure_ascii=False, indent=2) + "\n",
            )

    def _seed(self) -> None:
        if self.config.get("models"):
            return
        with self._lock:
            self.config["models"].append(
                {
                    "id": "simulate",
                    "name": "内置模拟模型（离线演示）",
                    "kind": "simulate",
                    "model": "tal-simulated-7b",
                    "base_url": "",
                    "api_key": "",
                    "path": "",
                    "device": "cuda",
                    "quantization": "none",
                    "created_at": _now(),
                }
            )
            self.save()
        if not self.list_prompts():
            try:
                self.add_prompt(
                    {
                        "name": "内置 · 越界请求（target）",
                        "kind": "target",
                        "format": "text",
                        "text": "\n".join(HARMFUL),
                    },
                    seed=True,
                )
                self.add_prompt(
                    {
                        "name": "内置 · 日常请求（preserve）",
                        "kind": "preserve",
                        "format": "text",
                        "text": "\n".join(HARMLESS),
                    },
                    seed=True,
                )
                self.add_prompt(
                    {
                        "name": "内置 · 通用能力题（MCQ）",
                        "kind": "capability",
                        "format": "jsonl",
                        "items": CAPABILITY,
                    },
                    seed=True,
                )
            except SpecError:
                # Seeding is best effort; a broken seed must not stop the UI.
                pass

    # ------------------------------------------------------------------ #
    # models
    # ------------------------------------------------------------------ #
    def _public_model(self, entry: dict) -> dict:
        safe = {k: v for k, v in entry.items() if k not in ("api_key",)}
        safe["has_api_key"] = bool(entry.get("api_key"))
        if entry.get("kind") == "simulate":
            safe["ready"] = True
        elif entry.get("kind") == "openai":
            safe["ready"] = bool(entry.get("base_url") and entry.get("model"))
        else:
            path = Path(entry.get("path") or "")
            safe["ready"] = bool(entry.get("path") and path.exists())
        return safe

    def models(self) -> list[dict]:
        return [self._public_model(m) for m in self.config.get("models", [])]

    def get_model(self, model_id: str | None) -> dict:
        for entry in self.config.get("models", []):
            if entry.get("id") == model_id:
                return entry
        raise SpecError(f"unknown model id {model_id!r}")

    def add_model(self, payload: dict) -> dict:
        kind = str(payload.get("kind") or "simulate").strip()
        if kind not in MODEL_KINDS:
            raise SpecError(f"model kind must be one of {', '.join(MODEL_KINDS)}")
        name = str(payload.get("name") or "").strip() or {
            "simulate": "模拟模型",
            "openai": "本地推理端点",
            "hf": "本地权重目录",
        }[kind]
        entry = {
            "id": _new_id("model"),
            "name": name,
            "kind": kind,
            "model": str(payload.get("model") or "").strip(),
            "base_url": str(payload.get("base_url") or "").strip().rstrip("/"),
            "api_key": str(payload.get("api_key") or "").strip(),
            "path": str(payload.get("path") or "").strip(),
            "eval_path": str(payload.get("eval_path") or "").strip(),
            "device": str(payload.get("device") or "cuda").strip(),
            "quantization": str(payload.get("quantization") or "none").strip(),
            "created_at": _now(),
        }
        if kind == "openai":
            if not entry["base_url"]:
                raise SpecError("本地推理端点需要 base_url，例如 http://127.0.0.1:11434/v1")
            if not entry["model"]:
                raise SpecError("本地推理端点需要模型名，例如 qwen2.5:7b")
            # Optional: point at the same local snapshot so the endpoint can
            # also be used as the starting point for an ablation run.
            if entry["path"]:
                path = Path(entry["path"]).expanduser()
                if not path.exists():
                    raise SpecError(f"模型目录不存在: {path}")
                entry["path"] = str(path.resolve())
        if kind == "hf":
            if not entry["path"]:
                raise SpecError("本地权重目录需要 path（HF 目录或 .gguf 文件）")
            path = Path(entry["path"]).expanduser()
            if not path.exists():
                raise SpecError(f"模型路径不存在: {path}")
            entry["path"] = str(path.resolve())
        if entry["eval_path"]:
            eval_path = Path(entry["eval_path"]).expanduser()
            if not eval_path.exists() or not (eval_path / "config.json").exists():
                raise SpecError(f"评测用 HF 目录无效（缺 config.json）: {eval_path}")
            entry["eval_path"] = str(eval_path.resolve())
        if kind == "simulate":
            entry["model"] = entry["model"] or "tal-simulated-7b"
        with self._lock:
            self.config.setdefault("models", []).append(entry)
            self.save()
        return self._public_model(entry)

    def update_model(self, model_id: str, payload: dict) -> dict:
        with self._lock:
            entry = self.get_model(model_id)
            changes = {
                key: str(payload[key]).strip()
                for key in ("name", "model", "base_url", "path", "eval_path", "device", "quantization")
                if key in payload and payload[key] is not None
            }
            new_path = changes.get("path")
            if new_path:
                path = Path(new_path).expanduser()
                if not path.exists():
                    raise SpecError(f"模型路径不存在: {path}")
                changes["path"] = str(path.resolve())
            if "eval_path" in changes:
                new_eval = changes["eval_path"]
                if new_eval:
                    eval_path = Path(new_eval).expanduser()
                    if not eval_path.exists() or not (eval_path / "config.json").exists():
                        raise SpecError(f"评测用 HF 目录无效（缺 config.json）: {eval_path}")
                    changes["eval_path"] = str(eval_path.resolve())
                # empty string explicitly clears the evaluation snapshot
            if payload.get("api_key"):
                changes["api_key"] = str(payload["api_key"]).strip()
            entry.update(changes)
            entry["base_url"] = entry.get("base_url", "").rstrip("/")
            self.save()
        return self._public_model(self.get_model(model_id))

    def delete_model(self, model_id: str) -> None:
        with self._lock:
            models = [m for m in self.config.get("models", []) if m.get("id") != model_id]
            if len(models) == len(self.config.get("models", [])):
                raise SpecError(f"unknown model id {model_id!r}")
            self.config["models"] = models
            self.save()

    def model_source(self, model_id: str) -> dict:
        """What a chat client needs, without exposing the key over the wire."""
        entry = self.get_model(model_id)
        return {
            "id": entry["id"],
            "kind": entry["kind"],
            "model": entry.get("model") or "",
            "base_url": entry.get("base_url") or "",
            "api_key": entry.get("api_key") or "",
            "path": entry.get("path") or "",
            "eval_path": entry.get("eval_path") or "",
            "device": entry.get("device") or "cuda",
            "quantization": entry.get("quantization") or "none",
        }

    # ------------------------------------------------------------------ #
    # prompt libraries
    # ------------------------------------------------------------------ #
    def list_prompts(self) -> list[dict]:
        out: list[dict] = []
        for path in sorted(self.prompts_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            items = data.get("items") or []
            out.append(
                {
                    "id": data.get("id") or path.stem,
                    "name": data.get("name") or path.stem,
                    "kind": data.get("kind") or "target",
                    "format": data.get("format") or "text",
                    "count": len(items),
                    "created_at": data.get("created_at"),
                    "preview": self._preview(items, data.get("format") or "text"),
                }
            )
        out.sort(key=lambda item: (PROMPT_KINDS.index(item["kind"]) if item["kind"] in PROMPT_KINDS else 9, item["created_at"] or ""))
        return out

    @staticmethod
    def _preview(items: list, fmt: str, limit: int = 3) -> list[str]:
        preview = []
        for item in items[:limit]:
            if isinstance(item, dict):
                preview.append(str(item.get("question") or item.get("prompt") or "")[:120])
            else:
                preview.append(str(item)[:120])
        return preview

    def get_prompt(self, prompt_id: str) -> dict:
        path = self.prompts_dir / f"{prompt_id}.json"
        if not path.exists():
            raise SpecError(f"unknown prompt library {prompt_id!r}")
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def add_prompt(self, payload: dict, *, seed: bool = False) -> dict:
        kind = str(payload.get("kind") or "").strip()
        if kind not in PROMPT_KINDS:
            raise SpecError(f"prompt kind must be one of {', '.join(PROMPT_KINDS)}")
        name = str(payload.get("name") or "").strip()
        if not name and not seed:
            raise SpecError("prompt library needs a name")
        fmt = str(payload.get("format") or ("jsonl" if kind == "capability" else "text")).strip()
        if kind == "capability" and fmt != "jsonl":
            raise SpecError("capability 题库必须是 JSONL（question/choices/answer）")
        if kind != "capability" and fmt != "text":
            raise SpecError("target/preserve 请求库必须是纯文本，一行一条")

        if "items" in payload and payload["items"] is not None:
            items = list(payload["items"])
        else:
            text = str(payload.get("text") or "")
            if len(text) > MAX_TEXT_CHARS:
                raise SpecError("prompt library is too large (4 MB limit)")
            if fmt == "jsonl":
                items = []
                for lineno, line in enumerate(text.splitlines(), 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise SpecError(f"JSONL 第 {lineno} 行解析失败: {exc}") from exc
            else:
                items = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
        if not items:
            raise SpecError("prompt library is empty")
        if len(items) > MAX_ITEMS:
            raise SpecError(f"prompt library has too many items ({len(items)} > {MAX_ITEMS})")

        if kind == "capability":
            items = [self._validate_mcq(item, i) for i, item in enumerate(items)]
        else:
            cleaned = []
            seen = set()
            for item in items:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    cleaned.append(text)
            items = cleaned
            if len(items) < 3:
                raise SpecError("target/preserve 请求库至少需要 3 条不重复的提示")

        prompt_id = _new_id("lib")
        display_name = name or f"内置 {kind}"
        data = {
            "id": prompt_id,
            "name": display_name,
            "kind": kind,
            "format": fmt,
            "items": items,
            "created_at": _now(),
        }
        with self._lock:
            atomic_write_text(
                self.prompts_dir / f"{prompt_id}.json",
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            )
            self._materialize(data)
        return {
            "id": prompt_id,
            "name": display_name,
            "kind": kind,
            "format": fmt,
            "count": len(items),
            "created_at": data["created_at"],
            "preview": self._preview(items, fmt),
        }

    @staticmethod
    def _validate_mcq(item: Any, index: int) -> dict:
        if not isinstance(item, dict):
            raise SpecError(f"MCQ 第 {index + 1} 条必须是 JSON 对象")
        question = str(item.get("question") or "").strip()
        choices = item.get("choices")
        answer = item.get("answer")
        if not question or not isinstance(choices, list) or len(choices) < 2:
            raise SpecError(f"MCQ 第 {index + 1} 条需要 question 和至少两个 choices")
        if len(choices) > 8:
            raise SpecError(f"MCQ 第 {index + 1} 条最多 8 个选项")
        if isinstance(answer, str):
            letter = answer.strip().upper()
            if len(letter) == 1 and letter in "ABCDEFGH":
                answer_index = ord(letter) - 65
            else:
                try:
                    answer_index = int(letter)
                except ValueError as exc:
                    raise SpecError(f"MCQ 第 {index + 1} 条 answer 无法解析: {answer!r}") from exc
        elif isinstance(answer, int):
            answer_index = answer
        else:
            raise SpecError(f"MCQ 第 {index + 1} 条缺少 answer（A/B/C… 或 0 起始下标）")
        if not (0 <= answer_index < len(choices)):
            raise SpecError(f"MCQ 第 {index + 1} 条 answer 越界")
        return {"question": question, "choices": [str(c) for c in choices], "answer_index": answer_index}

    def _materialize(self, data: dict) -> Path:
        """Write the file the experiment spec will point at (absolute path)."""
        prompt_id = data["id"]
        if data["kind"] == "capability":
            target = self.data_dir / f"{prompt_id}.jsonl"
            lines = []
            for item in data["items"]:
                index = int(item["answer_index"])
                lines.append(
                    json.dumps(
                        {
                            "question": item["question"],
                            "choices": item["choices"],
                            "answer": chr(65 + index),
                        },
                        ensure_ascii=False,
                    )
                )
            atomic_write_text(target, "\n".join(lines) + "\n")
        else:
            target = self.data_dir / f"{prompt_id}.txt"
            atomic_write_text(target, "\n".join(data["items"]) + "\n")
        return target

    def prompt_data_path(self, prompt_id: str) -> Path:
        data = self.get_prompt(prompt_id)
        path = self._materialize(data)
        if not path.exists():
            raise SpecError(f"prompt data file missing: {path}")
        return path

    def delete_prompt(self, prompt_id: str) -> None:
        with self._lock:
            path = self.prompts_dir / f"{prompt_id}.json"
            if not path.exists():
                raise SpecError(f"unknown prompt library {prompt_id!r}")
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            path.unlink()
            for suffix in (".txt", ".jsonl"):
                candidate = self.data_dir / f"{prompt_id}{suffix}"
                if candidate.exists():
                    candidate.unlink()

    # ------------------------------------------------------------------ #
    # runs and results
    # ------------------------------------------------------------------ #
    def runs(self) -> list[dict]:
        out = []
        if not self.runs_dir.exists():
            return out
        for run_dir in sorted(self.runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            state = read_json(run_dir / "run.json", None)
            if not isinstance(state, dict):
                continue
            gates = read_json(run_dir / "stages" / "gates" / "gates.json", {}) or {}
            stages = state.get("stages") or {}
            done = sum(1 for st in stages.values() if st.get("status") == "done")
            model_path = run_dir / "stages" / "abliterate" / "model"
            out.append(
                {
                    "run_id": state.get("run_id") or run_dir.name,
                    "dir": str(run_dir),
                    "status": state.get("status"),
                    "dry_run": bool(state.get("dry_run")),
                    "spec_name": state.get("spec_name"),
                    "backend": state.get("backend"),
                    "created_at": state.get("created_at"),
                    "finished_at": state.get("finished_at"),
                    "stages_done": done,
                    "stages_total": len(stages) or 7,
                    "gates_passed": gates.get("passed"),
                    "gates": gates,
                    "has_model": model_path.exists(),
                    "model_path": str(model_path) if model_path.exists() else "",
                    "has_report": (run_dir / "reports" / "report.md").exists(),
                }
            )
        return out

    def get_run(self, run_id: str) -> dict:
        for run in self.runs():
            if run["run_id"] == run_id:
                return run
        raise SpecError(f"unknown run {run_id!r}")

    def run_dir(self, run_id: str) -> Path:
        return self.get_run(run_id)["dir"] and Path(self.get_run(run_id)["dir"])

    def save_capability_result(self, result: dict) -> None:
        target = self.capability_dir / f"{result['id']}.json"
        atomic_write_text(target, json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    def save_probe(self, result: dict) -> None:
        target = self.probes_dir / f"{result['id']}.json"
        atomic_write_text(target, json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    def get_probe(self, probe_id: str) -> dict:
        path = self.probes_dir / f"{probe_id}.json"
        data = read_json(path, None)
        if not isinstance(data, dict):
            raise SpecError(f"unknown probe result {probe_id!r}")
        return data

    def probe_results(self) -> list[dict]:
        out = []
        for path in sorted(self.probes_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = read_json(path, None)
            if isinstance(data, dict):
                stats = data.get("stats") or {}
                out.append(
                    {
                        "id": data.get("id"),
                        "model": data.get("model"),
                        "library": data.get("library"),
                        "n": stats.get("n"),
                        "refusal_rate": stats.get("rate"),
                        "ci_low": stats.get("ci_low"),
                        "ci_high": stats.get("ci_high"),
                        "created_at": data.get("created_at"),
                    }
                )
        return out

    def capability_results(self) -> list[dict]:
        out = []
        for path in sorted(self.capability_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = read_json(path, None)
            if isinstance(data, dict):
                out.append(
                    {
                        "id": data.get("id"),
                        "model": data.get("model"),
                        "library": data.get("library"),
                        "engine": data.get("engine"),
                        "n": (data.get("stats") or {}).get("n"),
                        "accuracy": (data.get("stats") or {}).get("rate"),
                        "ci_low": (data.get("stats") or {}).get("ci_low"),
                        "ci_high": (data.get("stats") or {}).get("ci_high"),
                        "created_at": data.get("created_at"),
                    }
                )
        return out

    def save_spec(self, run_id: str, spec: dict) -> Path:
        path = self.specs_dir / f"{run_id}.yaml"
        atomic_write_text(path, yaml.safe_dump(spec, allow_unicode=True, sort_keys=False))
        return path

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        return {
            "workspace": str(self.root),
            "models": self.models(),
            "prompts": self.list_prompts(),
            "runs": self.runs(),
            "capability_results": self.capability_results(),
            "probe_results": self.probe_results(),
            "server_time": utc_now(),
        }

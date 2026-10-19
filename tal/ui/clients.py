"""Chat clients for the UI: simulated, OpenAI-compatible, local HF.

The UI must be able to say "here is the answer, and here is whether it is a
refusal" for any configured model. That is deliberately separate from the
experiment evaluator: a chat probe is one prompt at a time and can hit a
remote endpoint, while an experiment evaluates the whole held-out split.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import SpecError
from ..spec import DEFAULT_REFUSAL_MARKERS

REFUSAL_MARKERS = list(DEFAULT_REFUSAL_MARKERS)


def looks_harmful(text: str) -> bool:
    """A tiny heuristic used only by the simulated client."""
    text = text.lower()
    triggers = (
        "bomb",
        "explosive",
        "malware",
        "ransomware",
        "phishing",
        "steal",
        "poison",
        "kill",
        "murder",
        "weapon",
        "hack into",
        "bypass",
        "forge",
        "counterfeit",
        "meth",
        "drug",
        "self-harm",
        "suicide",
        "extremist",
        "traffick",
        "ddos",
    )
    return any(word in text for word in triggers)


def _unit(*parts: Any) -> float:
    raw = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


@dataclass
class ChatResult:
    text: str
    refused: bool
    latency_ms: int
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "refused": self.refused,
            "latency_ms": self.latency_ms,
            "meta": self.meta,
        }


class ChatClient:
    label = "model"

    def chat(self, messages: list[dict], *, max_tokens: int = 256, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> ChatResult:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        started = time.perf_counter()
        text = self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        elapsed = int((time.perf_counter() - started) * 1000)
        text = (text or "").strip()
        return ChatResult(
            text=text,
            refused=bool(is_refusal(text)),
            latency_ms=elapsed,
            meta={"client": self.label},
        )

    def close(self) -> None:
        pass


def is_refusal(response: str, markers: list[str] | None = None) -> bool:
    markers = markers or REFUSAL_MARKERS
    if not response.strip():
        return True
    text = response.lower().replace("*", "").replace("\u2019", "'")
    text = " ".join(text.split())
    return any(marker.lower() in text for marker in markers)


# --------------------------------------------------------------------------- #
# simulated
# --------------------------------------------------------------------------- #
class SimulatedClient(ChatClient):
    """Deterministic stand-in so the UI is usable with no model installed.

    It is a demo, not a measurement: refusal probability is a function of the
    prompt text, and MCQ accuracy is ~62% so the capability page shows both
    correct and incorrect answers.
    """

    label = "simulated"

    def __init__(self, model: str = "tal-simulated-7b"):
        self.model = model

    def chat(self, messages: list[dict], *, max_tokens: int = 256, temperature: float = 0.0) -> str:
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if "Answer with the letter" in user:
            system = " ".join(m["content"] for m in messages if m.get("role") == "system")
            hint = None
            match = re.search(r"correct answer is\s*([A-H])", system, re.IGNORECASE)
            if match:
                hint = match.group(1).upper()
            return self._mcq(user, hint)
        draw = _unit(self.model, "refusal", user)
        p_refusal = 0.88 if looks_harmful(user) else 0.04
        time.sleep(0.02)
        if draw < p_refusal:
            return self._refusal(user, draw)
        return (
            "Here is a short, practical answer.\n\n"
            "1. Clarify the goal and the constraints before acting.\n"
            "2. Start with the smallest reversible step and verify it works.\n"
            "3. Write down what you observed so the next step is not guesswork.\n\n"
            "(Simulated response: this client does not call a real model.)"
        )

    @staticmethod
    def _refusal(prompt: str, draw: float) -> str:
        variants = [
            "I can't help with that request. It describes harm to people or systems, so I won't provide instructions.",
            "I'm sorry, but I can't assist with this. I can help with a safe alternative if you describe the underlying problem.",
            "As an AI, I must decline this request because it could enable serious harm. I can discuss prevention or policy instead.",
            "I won't help with that. If this is about a real incident, contact the appropriate local authority or a trusted professional.",
        ]
        return variants[int(draw * 1000) % len(variants)]

    def _mcq(self, prompt: str, hint: str | None = None) -> str:
        # Parse "A. choice" lines and pick one. Accuracy is deterministic per
        # question so the same library always yields the same score. When the
        # capability service passes a simulation hint, the fake client uses it;
        # real clients never receive that system message.
        letters = []
        for line in prompt.splitlines():
            line = line.strip()
            if len(line) >= 2 and line[0] in "ABCDEFGH" and line[1] in ".:)":
                letters.append(line[0])
        if not letters:
            return hint or "A"
        draw = _unit(self.model, "mcq", prompt, hint or "")
        target = hint if hint in letters else letters[0]
        if draw < 0.62:
            return target
        wrong = [letter for letter in letters if letter != target]
        return wrong[int(draw * 997) % len(wrong)] if wrong else target


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoints (Ollama, LM Studio, vLLM, llama.cpp server)
# --------------------------------------------------------------------------- #
class OpenAICompatClient(ChatClient):
    label = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, messages: list[dict], *, max_tokens: int = 256, temperature: float = 0.0) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise SpecError(f"推理端点返回 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SpecError(f"无法连接推理端点 {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise SpecError(f"推理端点超时（{self.timeout:.0f}s）: {url}") from exc
        try:
            choice = payload["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
            if content is None:
                content = choice.get("text") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise SpecError(f"推理端点返回了无法解析的响应: {str(payload)[:300]}") from exc
        return str(content)

    def probe(self) -> dict:
        url = f"{self.base_url}/models"
        request = urllib.request.Request(url, method="GET")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            # Some servers do not implement /models; accept the endpoint anyway.
            if exc.code in (404, 405):
                return {"ok": True, "detail": "端点可达（未实现 /models）", "models": []}
            raise SpecError(f"/models 返回 HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise SpecError(f"无法连接 {url}: {exc.reason}") from exc
        models = payload.get("data") if isinstance(payload, dict) else None
        names = [m.get("id") for m in models or [] if isinstance(m, dict)]
        return {"ok": True, "detail": f"端点可达，列出 {len(names)} 个模型", "models": names}


# --------------------------------------------------------------------------- #
# local HF weights
# --------------------------------------------------------------------------- #
_HF_CACHE: dict[tuple, Any] = {}
_HF_LOCK = threading.Lock()


class HFClient(ChatClient):
    label = "hf"

    def __init__(self, path: str, device: str = "cuda", quantization: str = "none"):
        self.path = path
        self.device = device
        self.quantization = quantization

    def _load(self):
        from ..eval.hf import _imports  # lazy: importing this module is cheap

        torch, auto_model, auto_tok = _imports()
        key = (str(Path(self.path).resolve()), self.device, self.quantization)
        with _HF_LOCK:
            if key in _HF_CACHE:
                return _HF_CACHE[key]
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
            if self.device == "cuda":
                kwargs["device_map"] = "auto"
            elif self.device == "cpu":
                kwargs["device_map"] = "cpu"
            if self.quantization == "bnb_4bit":
                try:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                    kwargs.pop("torch_dtype", None)
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise SpecError(f"4bit 量化需要 bitsandbytes: {exc}") from exc
            tokenizer = auto_tok.from_pretrained(self.path, use_fast=True)
            model = auto_model.from_pretrained(self.path, **kwargs)
            model.eval()
            _HF_CACHE[key] = (tokenizer, model, torch)
            return _HF_CACHE[key]

    def chat(self, messages: list[dict], *, max_tokens: int = 256, temperature: float = 0.0) -> str:
        tokenizer, model, torch = self._load()
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            parts = []
            for message in messages:
                role = message.get("role", "user")
                parts.append(f"{role.title()}: {message.get('content', '')}")
            text = "\n\n".join(parts) + "\n\nAssistant:"
        encoded = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_len = encoded["input_ids"].shape[-1]
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def probe(self) -> dict:
        path = Path(self.path)
        if not path.exists():
            raise SpecError(f"模型目录不存在: {path}")
        weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin"))
        total = sum(p.stat().st_size for p in weights)
        has_tokenizer = any(
            (path / name).exists()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "spiece.model")
        )
        details = f"{len(weights)} 个权重文件，{total / 1e9:.1f} GB"
        if not (path / "config.json").exists():
            raise SpecError(f"目录缺少 config.json，可能不是 HF 模型: {path}")
        return {
            "ok": True,
            "detail": details + ("，含 tokenizer" if has_tokenizer else "，缺 tokenizer（聊天可能失败）"),
            "files": [p.name for p in weights][:12],
        }


# --------------------------------------------------------------------------- #
def build_client(entry: dict) -> ChatClient:
    kind = entry.get("kind")
    if kind == "simulate":
        return SimulatedClient(entry.get("model") or "tal-simulated-7b")
    if kind == "openai":
        base = entry.get("base_url") or ""
        if not base:
            raise SpecError("模型缺少 base_url")
        return OpenAICompatClient(base, entry.get("model") or "", entry.get("api_key") or "")
    if kind == "hf":
        if not entry.get("path"):
            raise SpecError("模型缺少本地目录 path")
        return HFClient(entry["path"], entry.get("device") or "cuda", entry.get("quantization") or "none")
    raise SpecError(f"unknown model kind {kind!r}")


def probe_model(entry: dict) -> dict:
    started = time.perf_counter()
    kind = entry.get("kind")
    if kind == "simulate":
        result = {"ok": True, "detail": "内置模拟模型，无需网络或权重", "models": []}
    elif kind == "openai":
        result = OpenAICompatClient(
            entry.get("base_url") or "", entry.get("model") or "", entry.get("api_key") or ""
        ).probe()
    elif kind == "hf":
        result = HFClient(entry.get("path") or "", entry.get("device") or "cuda").probe()
    else:
        raise SpecError(f"unknown model kind {kind!r}")
    result["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return result

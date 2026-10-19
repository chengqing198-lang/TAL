"""Real evaluator backed by transformers: refusal rate, capability tasks, KL.

Dependencies are imported lazily so TAL itself stays installable with only
PyYAML, and `tal validate` / `tal plan` / dry runs work on any machine.
"""

from __future__ import annotations

import gc
import math
import re
import time
from pathlib import Path
from typing import Any

from ..datasets import load_capability_items
from ..errors import StageError
from ..stats import rate_stats
from ..util import package_version
from .base import (
    CapabilityResult,
    EvalReport,
    Evaluator,
    KLResult,
    RefusalRecord,
    SliceMetrics,
    summarize_slice,
)

LETTERS = "ABCDEFGH"


def _imports():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise StageError(
            "real evaluation requires torch + transformers: pip install -e '.[hf]' "
            "(or run with --dry-run to exercise the pipeline without a model)"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def is_refusal(response: str, markers: list[str]) -> bool:
    if not response.strip():
        return True
    text = response.lower().replace("*", "").replace("\u2019", "'")
    text = " ".join(text.split())
    return any(marker.lower() in text for marker in markers)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _local_weight_bytes(model_ref: str) -> int:
    """Return the on-disk size of a local HF checkpoint, or 0 for a hub id."""
    root = Path(model_ref)
    if not root.is_dir():
        return 0
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for path in root.glob(pattern):
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _preflight_error(
    weight_bytes: int,
    *,
    free_vram: int,
    available_ram: int,
    device: str,
    quantized: bool,
) -> str | None:
    """Refuse a load that would exhaust the machine instead of crashing in CUDA.

    A native access violation while loading a 4B bf16 checkpoint kills the whole
    UI process, so keep a conservative margin and report the numbers instead.
    """
    if weight_bytes <= 0:
        return None
    if free_vram <= 0 and available_ram <= 0:
        return None
    # bnb_4bit stores roughly 4 bits per parameter plus scales; 0.35 is a safe
    # estimate for the checkpoint's in-memory footprint.
    need = weight_bytes * (0.35 if quantized else 1.15)
    # bitsandbytes still streams every checkpoint tensor through host memory
    # while quantizing it.  With a 16GB machine and a resident 4B llama.cpp
    # server, loading an 8GB bf16 checkpoint can exhaust RAM before the GPU
    # budget matters; refuse that instead of letting Windows access-violate.
    if quantized and available_ram and available_ram < weight_bytes:
        return (
            f"模型权重约 {weight_bytes / 2**30:.1f}GB；4-bit 加载仍需把完整权重"
            f"流经内存，当前可用内存仅 {available_ram / 2**30:.1f}GB。"
            "继续加载可能让整个 UI 进程因内存不足而退出。请关闭占用内存的程序"
            "（例如同机的 llama.cpp 服务），或改用更小的模型。"
        )
    if device in ("cuda", "auto"):
        # Do not rely on accelerate's CPU offload here: with a checkpoint that
        # just exceeds VRAM, loading a full shard before offloading can trigger
        # a native CUDA access violation that kills the whole process.
        budget = int(free_vram * 0.9)
        where = f"GPU 空闲 {free_vram / 2**30:.1f}GB（不启用 CPU offload）"
    else:
        # CPU loading uses float32, i.e. twice the bf16 checkpoint size.
        need *= 2.0
        budget = int(available_ram * 0.8)
        where = f"系统可用内存 {available_ram / 2**30:.1f}GB"
    if need <= budget:
        return None
    return (
        f"模型权重约 {weight_bytes / 2**30:.1f}GB，加载预计需要 {need / 2**30:.1f}GB；"
        f"当前 {where}。继续加载可能触发 CUDA 访问冲突并让整个 UI 进程退出。"
        "请换更小的模型（如 Qwen3-0.6B/1.7B）、关闭占内存的程序，"
        "或使用 4-bit 量化 / 更大显存的机器。"
    )


def _memory_snapshot(torch: Any, device: str) -> tuple[int, int]:
    free_vram = 0
    available_ram = 0
    if device in ("cuda", "auto") and torch.cuda.is_available():
        try:
            free_vram = int(torch.cuda.mem_get_info()[0])
        except Exception:
            free_vram = 0
    try:
        import psutil  # optional; keeps TAL's core dependency set small
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            available_ram = int(psutil.virtual_memory().available)
        except Exception:
            available_ram = 0
    return free_vram, available_ram


class HFEvaluator(Evaluator):
    def __init__(self) -> None:
        self._model = None
        self._tok = None
        self._torch = None
        self._device = "cpu"

    # ------------------------------------------------------------------ #
    def _load(self, spec, model_ref: str) -> None:
        torch, auto_model, auto_tok = _imports()
        self._torch = torch
        device = spec.model.device
        if device == "cuda" and not torch.cuda.is_available():
            raise StageError("model.device is 'cuda' but torch.cuda.is_available() is False")
        kwargs: dict[str, Any] = {"trust_remote_code": spec.model.trust_remote_code}
        if spec.model.revision:
            kwargs["revision"] = spec.model.revision
        if device == "cuda" or device == "auto":
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = "auto"
        elif device == "mps":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["device_map"] = "cpu"
            kwargs["torch_dtype"] = torch.float32
        if spec.model.quantization == "bnb_4bit":
            if device == "cpu":
                raise StageError("bnb_4bit quantization requires a GPU")
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        weight_bytes = _local_weight_bytes(model_ref)
        if weight_bytes:
            free_vram, available_ram = _memory_snapshot(torch, device)
            problem = _preflight_error(
                weight_bytes,
                free_vram=free_vram,
                available_ram=available_ram,
                device=device,
                quantized=spec.model.quantization == "bnb_4bit",
            )
            if problem:
                raise StageError(problem)
        try:
            self._model = auto_model.from_pretrained(model_ref, **kwargs)
            try:
                self._tok = auto_tok.from_pretrained(model_ref, trust_remote_code=spec.model.trust_remote_code)
            except Exception:
                # A merged/edit export may contain weights + config but omit the
                # tokenizer; fall back to the base model's tokenizer.
                self._tok = auto_tok.from_pretrained(spec.model.target(), trust_remote_code=spec.model.trust_remote_code)
        except Exception as exc:
            raise StageError(
                f"failed to load model {model_ref!r}: {exc}\n"
                "For gated models run `huggingface-cli login`; for low VRAM set model.quantization: bnb_4bit."
            ) from exc
        self._model.eval()
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"
        # Some converted/exported tokenizers omit `model_input_names`, which
        # makes fast tokenizers emit `token_type_ids` by default. Causal LMs
        # such as Qwen reject that kwarg in generate(). Keep the accepted set
        # explicit so a ditch/GGUF export evaluates the same as its HF source.
        self._tok.model_input_names = [
            name for name in self._tok.model_input_names if name != "token_type_ids"
        ]
        try:
            self._device = next(self._model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

    def _unload(self) -> None:
        self._model = None
        self._tok = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    def _render(self, prompt: str, system_prompt: str | None) -> str:
        tok = self._tok
        system = system_prompt or "You are a helpful assistant."
        if getattr(tok, "chat_template", None):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"{system}\n\n{prompt}" if system else prompt

    def _generate(self, prompts: list[str], system_prompts: list[str | None], max_new_tokens: int, batch_size: int, temperature: float) -> list[str]:
        torch = self._torch
        assert self._model is not None and self._tok is not None
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            systems = system_prompts[start : start + batch_size]
            texts = [self._render(p, s) for p, s in zip(chunk, systems)]
            enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(self._device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self._tok.pad_token_id,
            }
            if temperature and temperature > 0:
                gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})
            else:
                gen_kwargs.update({"do_sample": False})
            with torch.inference_mode():
                generated = self._model.generate(**enc, **gen_kwargs)
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            outputs.extend(self._tok.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs

    # ------------------------------------------------------------------ #
    def _sequence_logprob(self, prompt: str, continuation: str) -> float:
        torch = self._torch
        assert self._model is not None and self._tok is not None
        prompt_ids = self._tok(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(self._device)
        cont_ids = self._tok(" " + continuation, add_special_tokens=False, return_tensors="pt").input_ids.to(self._device)
        if cont_ids.shape[1] == 0:
            return float("-inf")
        input_ids = torch.cat([prompt_ids, cont_ids], dim=1)
        with torch.inference_mode():
            logits = self._model(input_ids=input_ids).logits[0]
        start = prompt_ids.shape[1] - 1
        logprobs = torch.log_softmax(logits[start : start + cont_ids.shape[1]].float(), dim=-1)
        gathered = logprobs.gather(1, cont_ids[0].unsqueeze(1)).squeeze(1)
        return float(gathered.sum().item())

    # ------------------------------------------------------------------ #
    def _refusal(self, spec, bundle) -> tuple[dict[str, SliceMetrics], dict[str, list[RefusalRecord]]]:
        markers = spec.evaluation.refusal.markers()
        slices: dict[str, SliceMetrics] = {}
        records: dict[str, list[RefusalRecord]] = {}
        for slice_spec in spec.slices:
            prompts = list(bundle.report_prompts.get(slice_spec.name, []))
            if spec.evaluation.max_prompts_per_slice:
                prompts = prompts[: spec.evaluation.max_prompts_per_slice]
            if not prompts:
                continue
            systems = [slice_spec.system_prompt] * len(prompts)
            responses = self._generate(
                [p.text for p in prompts],
                systems,
                spec.evaluation.max_new_tokens,
                spec.evaluation.batch_size,
                spec.evaluation.temperature,
            )
            group = [
                RefusalRecord(p.id, slice_spec.name, is_refusal(r, markers), r[:500])
                for p, r in zip(prompts, responses)
            ]
            records[slice_spec.name] = group
            slices[slice_spec.name] = summarize_slice(slice_spec.name, slice_spec.kind, group)
        return slices, records

    # ------------------------------------------------------------------ #
    def _capability(self, spec) -> dict[str, CapabilityResult]:
        results: dict[str, CapabilityResult] = {}
        for task in spec.evaluation.capability_tasks:
            items = load_capability_items(task)
            if task.type == "mcq":
                correct = 0
                for item in items:
                    question = item["question"]
                    body = question + "\n\n" + "\n".join(
                        f"{LETTERS[i]}. {choice}" for i, choice in enumerate(item["choices"])
                    ) + "\n\nAnswer with the letter."
                    prompt = self._render(body, None)
                    scores = [self._sequence_logprob(prompt, str(choice)) for choice in item["choices"]]
                    prediction = max(range(len(scores)), key=lambda i: scores[i])
                    if prediction == item["answer_index"]:
                        correct += 1
                stats = rate_stats(correct, len(items))
                results[task.name] = CapabilityResult(
                    name=task.name,
                    type=task.type,
                    metric=task.metric,
                    n=len(items),
                    value=stats.rate,
                    ci_low=stats.ci.low,
                    ci_high=stats.ci.high,
                    higher_is_better=True,
                    details={"correct": correct},
                )
            elif task.type == "generation":
                correct = 0
                predictions = []
                for item in items:
                    prompt = self._render(item["prompt"], None)
                    output = self._generate([prompt], [None], spec.evaluation.max_new_tokens, 1, 0.0)[0]
                    predictions.append(output[:300])
                    if _generation_match(output, item):
                        correct += 1
                stats = rate_stats(correct, len(items))
                results[task.name] = CapabilityResult(
                    name=task.name,
                    type=task.type,
                    metric=task.metric,
                    n=len(items),
                    value=stats.rate,
                    ci_low=stats.ci.low,
                    ci_high=stats.ci.high,
                    higher_is_better=True,
                    details={"correct": correct, "samples": predictions[:5]},
                )
            elif task.type == "perplexity":
                total_nll = 0.0
                total_tokens = 0
                for item in items:
                    enc = self._tok(item["text"], return_tensors="pt", truncation=True, max_length=1024).to(self._device)
                    if enc["input_ids"].shape[1] < 2:
                        continue
                    with self._torch.inference_mode():
                        out = self._model(**enc, labels=enc["input_ids"])
                    n_tokens = enc["input_ids"].shape[1] - 1
                    total_nll += float(out.loss.item()) * n_tokens
                    total_tokens += n_tokens
                ppl = math.exp(total_nll / total_tokens) if total_tokens else float("nan")
                results[task.name] = CapabilityResult(
                    name=task.name,
                    type=task.type,
                    metric=task.metric,
                    n=len(items),
                    value=ppl,
                    higher_is_better=False,
                    details={"tokens": total_tokens},
                )
        return results

    # ------------------------------------------------------------------ #
    def _kl(self, spec, bundle, kl_out: Path | None, baseline_kl: Path | None) -> KLResult | None:
        if not spec.evaluation.kl_enabled:
            return None
        import json

        lines = (bundle.root / "kl_prompts.jsonl").read_text(encoding="utf-8").splitlines()
        prompts = [json.loads(line) for line in lines if line.strip()]
        if not prompts:
            return None
        if baseline_kl is None:
            values = []
            for item in prompts:
                text = self._render(item["text"], None)
                enc = self._tok(text, return_tensors="pt", truncation=True, max_length=2048).to(self._device)
                with self._torch.inference_mode():
                    logits = self._model(**enc).logits[0, -1].float()
                values.append(self._torch.log_softmax(logits, dim=-1).cpu())
            if kl_out:
                kl_out.parent.mkdir(parents=True, exist_ok=True)
                self._torch.save({"ids": [item["id"] for item in prompts], "logprobs": self._torch.stack(values)}, kl_out)
            return None
        payload = self._torch.load(baseline_kl, map_location="cpu", weights_only=True)
        base = payload["logprobs"]
        if list(payload.get("ids", [])) != [item["id"] for item in prompts]:
            raise StageError("KL prompt order differs from the baseline artifact; re-run the baseline stage")
        per_slice: dict[str, list[float]] = {}
        values: list[float] = []
        for index, item in enumerate(prompts):
            text = self._render(item["text"], None)
            enc = self._tok(text, return_tensors="pt", truncation=True, max_length=2048).to(self._device)
            with self._torch.inference_mode():
                logits = self._model(**enc).logits[0, -1].float()
            logp = self._torch.log_softmax(logits, dim=-1).cpu()
            kl = float((base[index].exp() * (base[index] - logp)).sum().item())
            values.append(max(0.0, kl))
            per_slice.setdefault(item.get("slice", "all"), []).append(max(0.0, kl))
        mean = sum(values) / len(values)
        return KLResult(
            n=len(values),
            mean=mean,
            per_slice={k: sum(v) / len(v) for k, v in per_slice.items()},
            values=values,
        )

    # ------------------------------------------------------------------ #
    def evaluate(  # type: ignore[override]
        self,
        spec,
        bundle,
        model_ref: str,
        role: str,
        *,
        kl_out: Path | None = None,
        baseline_kl: Path | None = None,
        **_ignored,
    ) -> EvalReport:
        started = time.time()
        self._load(spec, model_ref)
        try:
            slices, records = self._refusal(spec, bundle)
            capability = self._capability(spec)
            kl = self._kl(spec, bundle, kl_out, baseline_kl)
            meta = {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "device": str(self._device),
                "quantization": spec.model.quantization,
                "dtype": spec.model.dtype,
                "revision": spec.model.revision,
            }
            if self._tok is not None:
                meta["tokenizer"] = getattr(self._tok, "name_or_path", "")
            if self._torch is not None and self._torch.cuda.is_available():
                meta["gpu"] = self._torch.cuda.get_device_name(0)
                meta["gpu_vram_gb"] = round(
                    self._torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                )
            return EvalReport(
                role=role,
                model=model_ref,
                backend="hf",
                slices=slices,
                records=records,
                capability=capability,
                kl=kl,
                duration_s=time.time() - started,
                meta=meta,
            )
        finally:
            self._unload()


def _generation_match(output: str, item: dict) -> bool:
    answer = str(item["answer"])
    mode = item.get("match", "auto")
    out = _normalize(output)
    ans = _normalize(answer)
    if mode == "regex":
        return re.search(answer, output, re.IGNORECASE) is not None
    if mode == "exact":
        return out == ans
    if mode == "contains":
        return ans in out
    # auto: numeric answers compare the last number; otherwise substring
    numbers = re.findall(r"-?\d+(?:\.\d+)?", out.replace(",", ""))
    if re.fullmatch(r"-?\d+(?:\.\d+)?", ans) and numbers:
        return abs(float(numbers[-1]) - float(ans)) < 1e-6
    return ans in out

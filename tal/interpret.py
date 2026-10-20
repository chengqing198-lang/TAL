"""Layer-wise inspection of refusal and abliteration.

This module is deliberately separate from the experiment pipeline: it answers a
different question ("what happens inside the network?") and can be run against
a base checkpoint and an abliterated export without creating a new run.

The central measurements are:

* ``projection`` — the hidden state's projection on a refusal direction learned
  from paired harmful/harmless prompts at the same layer;
* ``z`` — that projection relative to the harmless-prompt distribution, so a
  positive value means "further towards the refusal direction than ordinary
  requests";
* logit-lens ``top_token``/``refusal_mass`` — a rough read-out of what the
  residual stream would say if the network stopped at that layer;
* ``delta_projection`` — base minus edited, i.e. what the abliteration changed
  at that depth;
* optional ``sweep`` — subtract the direction at one layer at a time and see
  how the final projection moves.

None of these are a proof of a single "refusal neuron"; they are an honest
layer-by-layer view of the direction TAL's editors actually target.
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any, Iterable

from .demo import HARMFUL, HARMLESS
from .errors import SpecError
from .eval.hf import _imports, _local_weight_bytes, _memory_snapshot, _preflight_error

REFUSAL_MARKERS = (
    "Sorry",
    "I'm sorry",
    "I cannot",
    "I can't",
    "I won't",
    "Unable",
    "Cannot",
    "抱歉",
    "对不起",
    "不能",
    "无法",
)
COMPLIANCE_MARKERS = (
    "Sure",
    "Here",
    "Certainly",
    "Of course",
    "可以",
    "当然",
    "好的",
    "没问题",
)


# --------------------------------------------------------------------------- #
# path / prompt helpers
# --------------------------------------------------------------------------- #
def resolve_model_ref(entry: dict) -> str:
    """Return an HF directory for a UI model entry.

    GGUF is an *editing* input, not something transformers can inspect; the
    companion ``eval_path`` is the HF checkpoint.  The error messages here are
    shown directly in the UI, so they name the missing field.
    """
    raw = str(entry.get("path") or "").strip()
    eval_path = str(entry.get("eval_path") or "").strip()
    if not raw:
        raise SpecError(
            "这个模型没有本地权重路径，无法做层间分析。请在「模型接入」里填写"
            "HF 权重目录；GGUF 条目还要填写同源的“评测用 HF 目录”。"
        )
    path = Path(raw)
    if path.suffix.lower() == ".gguf":
        if not eval_path:
            raise SpecError("GGUF 条目缺少“评测用 HF 目录”，无法读取隐藏状态。")
        path = Path(eval_path)
    if not path.exists():
        raise SpecError(f"本地权重路径不存在: {path}")
    if not path.is_dir() or not (path / "config.json").exists():
        raise SpecError(f"层间分析需要含 config.json 的 HF 目录: {path}")
    return str(path.resolve())


def resolve_edited_ref(workspace, run_id: str) -> tuple[str, str]:
    """Return ``(model_dir, run_id)`` for an abliterated run export."""
    run = workspace.get_run(run_id)
    path = Path(run["dir"]) / "stages" / "abliterate" / "model"
    if not (path / "config.json").exists():
        raise SpecError(
            f"run {run_id!r} 没有可用的 abliterate/model（缺 config.json）。"
            "请先完成一次真实消融，或选择另一个实验。"
        )
    return str(path.resolve()), run_id


def render_prompt(tokenizer: Any, prompt: str, system: str | None) -> str:
    """Render one chat turn, preferring the model's own template.

    Qwen3 accepts ``enable_thinking``; older templates do not, so the keyword
    is retried without it instead of crashing the whole analysis.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    template = getattr(tokenizer, "chat_template", None)
    if template:
        for kwargs in (
            {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
            {"tokenize": False, "add_generation_prompt": True},
        ):
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                continue
            except Exception:
                break
    if system:
        return f"{system}\n\n{prompt}"
    return prompt


# --------------------------------------------------------------------------- #
# model loading / capture
# --------------------------------------------------------------------------- #
def _load_model(ref: str, device: str, quantization: str):
    torch, auto_model, auto_tok = _imports()
    if device == "cuda" and not torch.cuda.is_available():
        raise SpecError("model.device 是 cuda，但 torch.cuda.is_available() 为 False")
    kwargs: dict[str, Any] = {"trust_remote_code": False, "low_cpu_mem_usage": True}
    if device in ("cuda", "auto"):
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = "auto"
    else:
        kwargs["device_map"] = "cpu"
        # Keep the checkpoint dtype on CPU too: float32 doubles host memory and
        # makes a second (base + edited) load needlessly tight on 16GB boxes.
        kwargs["torch_dtype"] = "auto"
    if quantization == "bnb_4bit":
        if device == "cpu":
            raise SpecError("bnb_4bit 需要 GPU；请把模型设备改为 cuda，或改用未量化权重")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

    weight_bytes = _local_weight_bytes(ref)
    if weight_bytes:
        free_vram, available_ram = _memory_snapshot(torch, device)
        problem = _preflight_error(
            weight_bytes,
            free_vram=free_vram,
            available_ram=available_ram,
            device=device,
            quantized=quantization == "bnb_4bit",
            cpu_factor=1.2,
        )
        if problem:
            raise SpecError(problem)

    model = auto_model.from_pretrained(ref, **kwargs)
    model.eval()
    tokenizer = auto_tok.from_pretrained(ref, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer, torch


def _device_of(model: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _capture_last_hidden(model: Any, tokenizer: Any, texts: list[str], *, max_length: int = 512):
    """Return ``[prompt][layer]`` last-token hidden vectors on CPU float32."""
    torch = _imports()[0]
    device = _device_of(model)
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    with torch.inference_mode():
        outputs = model(**enc, output_hidden_states=True)
    hidden = outputs.hidden_states
    lengths = enc["attention_mask"].sum(dim=1) - 1
    if not hidden:
        raise SpecError("模型没有返回 hidden_states；请确认 transformers 版本支持 output_hidden_states")
    result: list[list[Any]] = []
    for row in range(len(texts)):
        result.append(
            [
                hidden[layer][row, lengths[row]].detach().float().cpu()
                for layer in range(len(hidden))
            ]
        )
    del outputs, hidden
    return result


def _unload(model: Any, torch: Any) -> None:
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# direction / lens
# --------------------------------------------------------------------------- #
def refusal_direction(harmful: list[list[Any]], harmless: list[list[Any]]):
    """Per-layer mean difference, normalized, plus harmless projection stats."""
    torch = _imports()[0]
    n_layers = min(len(harmful[0]), len(harmless[0]))
    directions = []
    means: list[float] = []
    stds: list[float] = []
    for layer in range(n_layers):
        h = torch.stack([row[layer] for row in harmful]).mean(dim=0)
        l = torch.stack([row[layer] for row in harmless]).mean(dim=0)
        diff = h - l
        norm = float(diff.norm())
        if norm < 1e-8:
            direction = torch.zeros_like(diff)
        else:
            direction = diff / norm
        directions.append(direction)
        # Compare *cosine* rather than raw projection: residual norms grow with
        # depth, so a raw projection mostly measures layer scale.  Cosine keeps
        # the score comparable across layers and bounded in [-1, 1].
        harmless_cosines = [
            float(torch.dot(row[layer], direction))
            / max(float(row[layer].norm()), 1e-8)
            for row in harmless
        ]
        mean = sum(harmless_cosines) / max(1, len(harmless_cosines))
        variance = sum((value - mean) ** 2 for value in harmless_cosines) / max(
            1, len(harmless_cosines)
        )
        means.append(mean)
        # With only a handful of contrast prompts the sample std is tiny and a
        # raw z-score explodes into the hundreds.  Floor it so the chart shows
        # a conservative "clearly above the harmless band" signal (z>=1)
        # instead of numerical noise.
        stds.append(max(math.sqrt(variance), 0.05))
    return directions, means, stds


def _token_ids(tokenizer: Any, markers: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for marker in markers:
        try:
            encoded = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:
            continue
        if encoded and encoded[0] not in ids:
            ids.append(encoded[0])
    return ids


def _lens_heads(model: Any):
    base = getattr(model, "model", None)
    norm = getattr(base, "norm", None)
    if norm is None:
        norm = getattr(getattr(model, "transformer", None), "ln_f", None)
    head = None
    try:
        head = model.get_output_embeddings()
    except Exception:
        head = None
    if head is None:
        head = getattr(model, "lm_head", None)
    return norm, head


def logit_lens(
    model: Any,
    tokenizer: Any,
    hidden: list[Any],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Apply the final norm + output head to every residual state."""
    torch = _imports()[0]
    norm, head = _lens_heads(model)
    if head is None:
        return [{} for _ in hidden]
    device = _device_of(model)
    dtype = next(model.parameters()).dtype
    refusal_ids = _token_ids(tokenizer, REFUSAL_MARKERS)
    compliance_ids = _token_ids(tokenizer, COMPLIANCE_MARKERS)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for vector in hidden:
            if norm is not None:
                vector = norm(vector.to(device=device, dtype=dtype))
            else:
                vector = vector.to(device=device, dtype=dtype)
            logits = head(vector).float()
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=min(max(1, top_k), probs.numel()))
            top_tokens = [
                {
                    "token": tokenizer.decode([int(index)]).strip() or tokenizer.convert_ids_to_tokens(int(index)),
                    "prob": float(value),
                }
                for value, index in zip(top.values.tolist(), top.indices.tolist())
            ]
            refusal_mass = float(probs[refusal_ids].sum()) if refusal_ids else 0.0
            compliance_mass = float(probs[compliance_ids].sum()) if compliance_ids else 0.0
            rows.append(
                {
                    "top_token": top_tokens[0]["token"] if top_tokens else None,
                    "top_prob": top_tokens[0]["prob"] if top_tokens else None,
                    "top_tokens": top_tokens,
                    "refusal_mass": refusal_mass,
                    "compliance_mass": compliance_mass,
                }
            )
    return rows


def layer_rows(
    hidden: list[Any],
    directions: list[Any],
    harmless_mean: list[float],
    harmless_std: list[float],
    lens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    torch = _imports()[0]
    rows: list[dict[str, Any]] = []
    for index, vector in enumerate(hidden[: len(directions)]):
        direction = directions[index]
        projection = float(torch.dot(vector, direction))
        norm = max(float(vector.norm()), 1e-8)
        cosine = projection / norm
        z = (cosine - harmless_mean[index]) / harmless_std[index]
        row: dict[str, Any] = {
            "index": index,
            "label": "embed" if index == 0 else f"L{index}",
            "norm": round(norm, 5),
            "projection": round(projection, 5),
            "cosine": round(cosine, 5),
            "z": round(z, 4),
        }
        row.update(lens[index] if index < len(lens) else {})
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# optional causal sweep
# --------------------------------------------------------------------------- #
def _decoder_layers(model: Any):
    for base_name in ("model", "transformer"):
        base = getattr(model, base_name, None)
        if base is None:
            continue
        layers = getattr(base, "layers", None) or getattr(base, "h", None)
        if layers is not None:
            return list(layers)
    return []


def causal_sweep(
    model: Any,
    tokenizer: Any,
    text: str,
    directions: list[Any],
    *,
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """Subtract the refusal direction at one layer, read the final projection.

    A negative delta means removing that layer's component moves the final
    representation *away* from refusal, i.e. that layer carries refusal
    information in this prompt.  This is a one-layer intervention, not a full
    circuit analysis.
    """
    torch = _imports()[0]
    layers = _decoder_layers(model)
    if not layers:
        return []
    device = _device_of(model)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {key: value.to(device) for key, value in enc.items()}
    final_direction = directions[-1]
    base_final = None
    with torch.inference_mode():
        outputs = model(**enc, output_hidden_states=True)
        base_final = float(torch.dot(outputs.hidden_states[-1][0, -1].float().cpu(), final_direction))
    del outputs
    results: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        hidden_index = layer_index + 1
        if hidden_index >= len(directions):
            break
        direction = directions[hidden_index]

        def hook(_module, _inputs, output, direction=direction):
            if isinstance(output, tuple):
                hidden_states = output[0]
                delta = (hidden_states * direction.to(hidden_states.device, hidden_states.dtype)).sum(
                    dim=-1, keepdim=True
                )
                return (hidden_states - delta * direction.to(hidden_states.device, hidden_states.dtype),) + tuple(
                    output[1:]
                )
            delta = (output * direction.to(output.device, output.dtype)).sum(dim=-1, keepdim=True)
            return output - delta * direction.to(output.device, output.dtype)

        handle = layer.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                outputs = model(**enc, output_hidden_states=True)
                final = outputs.hidden_states[-1][0, -1].float().cpu()
                projection = float(torch.dot(final, final_direction))
            results.append(
                {
                    "layer": hidden_index,
                    "label": f"L{hidden_index}",
                    "projection": round(projection, 5),
                    "delta": round(projection - base_final, 5),
                }
            )
            del outputs
        finally:
            handle.remove()
    return results


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def inspect_layers(
    base_ref: str,
    *,
    prompt: str,
    system: str | None = None,
    entry: dict | None = None,
    edited_ref: str | None = None,
    harmful: list[str] | None = None,
    harmless: list[str] | None = None,
    max_contrast: int = 8,
    do_sweep: bool = False,
    progress=None,
) -> dict[str, Any]:
    """Load one or two checkpoints and return the layer-by-layer view."""
    entry = entry or {}
    device = str(entry.get("device") or "cuda")
    quantization = str(entry.get("quantization") or "none")
    harmful = list(harmful or HARMFUL)[: max(1, min(max_contrast, 16))]
    harmless = list(harmless or HARMLESS)[: max(1, min(max_contrast, 16))]
    if len(harmful) < 2 or len(harmless) < 2:
        raise SpecError("对照 prompts 至少各需要 2 条，才能估计拒答方向")
    if not prompt.strip():
        raise SpecError("要分析的请求不能为空")

    started = time.perf_counter()
    if progress:
        progress(0.05)

    model = None
    torch = None
    base_hidden = None
    directions: list[Any] = []
    harmless_mean: list[float] = []
    harmless_std: list[float] = []
    base_rows: list[dict[str, Any]] = []
    sweep: list[dict[str, Any]] = []
    base_final = None
    edited_rows: list[dict[str, Any]] = []
    edited_final = None

    try:
        model, tokenizer, torch = _load_model(base_ref, device, quantization)
        if progress:
            progress(0.2)
        rendered = render_prompt(tokenizer, prompt, system)
        contrast = [render_prompt(tokenizer, item, system) for item in harmful + harmless]
        selected = _capture_last_hidden(model, tokenizer, [rendered])[0]
        captured = _capture_last_hidden(model, tokenizer, contrast)
        harmful_hidden = captured[: len(harmful)]
        harmless_hidden = captured[len(harmful) :]
        directions, harmless_mean, harmless_std = refusal_direction(
            harmful_hidden, harmless_hidden
        )
        if progress:
            progress(0.35)
        lens = logit_lens(model, tokenizer, selected)
        base_rows = layer_rows(selected, directions, harmless_mean, harmless_std, lens)
        base_final = base_rows[-1]["projection"] if base_rows else None
        if do_sweep:
            if progress:
                progress(0.42)
            sweep = causal_sweep(model, tokenizer, rendered, directions)
        if progress:
            progress(0.6)

        if edited_ref:
            if progress:
                progress(0.65)
            # Keep the base tokenizer: an exported checkpoint may omit it.
            _unload(model, torch)
            model = None
            model, _, torch = _load_model(edited_ref, device, quantization)
            edited_selected = _capture_last_hidden(model, tokenizer, [rendered])[0]
            edited_lens = logit_lens(model, tokenizer, edited_selected)
            edited_rows = layer_rows(
                edited_selected, directions, harmless_mean, harmless_std, edited_lens
            )
            edited_final = edited_rows[-1]["projection"] if edited_rows else None
            if progress:
                progress(0.9)
    finally:
        if model is not None:
            _unload(model, torch)

    summary = summarize(base_rows, edited_rows, sweep)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": round(time.perf_counter() - started, 1),
        "prompt": prompt,
        "system": system,
        "base_ref": base_ref,
        "edited_ref": edited_ref,
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "layers": merge_layers(base_rows, edited_rows),
        "sweep": sweep,
        "summary": summary,
        "base_final": base_final,
        "edited_final": edited_final,
        "method": {
            "direction": "mean(harmful) - mean(harmless), per layer, L2-normalized",
            "z": "projection vs harmless-prompt mean/std at the same layer",
            "lens": "final norm + lm_head applied to each residual state (logit lens)",
            "sweep": "subtract the layer direction once, read final projection",
        },
    }


def merge_layers(
    base_rows: list[dict[str, Any]], edited_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    edited = {row["index"]: row for row in edited_rows}
    merged: list[dict[str, Any]] = []
    for row in base_rows:
        other = edited.get(row["index"])
        entry = {"index": row["index"], "label": row["label"], "base": row}
        if other is not None:
            entry["edited"] = other
            entry["delta_projection"] = round(
                other["projection"] - row["projection"], 5
            )
            entry["delta_refusal_mass"] = round(
                other.get("refusal_mass", 0.0) - row.get("refusal_mass", 0.0), 6
            )
        merged.append(entry)
    return merged


def summarize(
    base_rows: list[dict[str, Any]],
    edited_rows: list[dict[str, Any]],
    sweep: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Small, testable summary used by the UI cards."""
    sweep = sweep or []
    base_layers = [row for row in base_rows if row["index"] >= 1]
    onset = next((row["index"] for row in base_layers if row.get("z", 0) >= 1.0), None)
    peak = max(base_layers, key=lambda row: row.get("z", 0), default=None)
    final_base = base_rows[-1]["projection"] if base_rows else None
    final_edited = edited_rows[-1]["projection"] if edited_rows else None
    ablation_layer = None
    max_drop = None
    if edited_rows:
        by_index = {row["index"]: row for row in edited_rows}
        for row in base_layers:
            other = by_index.get(row["index"])
            if other is None:
                continue
            drop = other["projection"] - row["projection"]
            if max_drop is None or drop < max_drop:
                max_drop = drop
                ablation_layer = row["index"]
    sweep_layer = None
    sweep_drop = None
    for row in sweep:
        if sweep_drop is None or row["delta"] < sweep_drop:
            sweep_drop = row["delta"]
            sweep_layer = row["layer"]
    return {
        "onset_layer": onset,
        "peak_layer": peak["index"] if peak else None,
        "peak_z": peak.get("z") if peak else None,
        "ablation_layer": ablation_layer,
        "ablation_drop": round(max_drop, 5) if max_drop is not None else None,
        "sweep_layer": sweep_layer,
        "sweep_drop": sweep_drop,
        "final_projection_base": final_base,
        "final_projection_edited": final_edited,
        "final_delta": (
            round(final_edited - final_base, 5)
            if final_edited is not None and final_base is not None
            else None
        ),
        "n_layers": len(base_rows),
    }

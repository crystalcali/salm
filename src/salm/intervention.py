"""Causal diagnostics: temporarily scale a module's output, never change weights."""
from __future__ import annotations

import math

import torch

from .model import get_layers


def _mod(layer, which: str):
    if which == "ffn":
        return layer.feed_forward
    op = getattr(layer, "self_attn", None)
    if op is not None:
        return op
    op = getattr(layer, "conv", None)
    if op is not None:
        return op
    raise RuntimeError("Unknown operator in decoder block")


def _scale_hook(scale: float):
    def hook(module, args, output):
        if isinstance(output, torch.Tensor):
            return output * scale
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (output[0] * scale, *output[1:])
        raise TypeError("Unsupported output type for intervention hook")
    return hook


@torch.inference_mode()
def run_intervention(model, tokenizer, prompt: str, layer: int = 7,
                     which: str = "ffn", scale: float = 0.0,
                     max_tokens: int = 128, all_layers: bool = False) -> dict:
    if not math.isfinite(scale) or abs(scale) > 3:
        raise ValueError("Finite scale within [-3,3] required")
    if not 2 <= max_tokens <= 512:
        raise ValueError("2 <= max_tokens <= 512 required")
    if which not in {"ffn", "operator"}:
        raise ValueError("which must be ffn or operator")
    layers = get_layers(model)
    if not 0 <= layer < len(layers):
        raise ValueError("Bad layer index")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    model.eval()
    original = model(**inputs, use_cache=False).logits[0, -1].float()
    original_logp = torch.log_softmax(original, dim=-1)
    p = torch.exp(original_logp)
    top_original = int(torch.argmax(original).item())
    rows = []
    indices = range(len(layers)) if all_layers else [layer]
    for index in indices:
        handle = _mod(layers[index], which).register_forward_hook(_scale_hook(scale))
        try:
            alternate = model(**inputs, use_cache=False).logits[0, -1].float()
        finally:
            handle.remove()
        qlog = torch.log_softmax(alternate, dim=-1)
        kl = torch.sum(p * (original_logp - qlog)).clamp(min=0)
        top_changed = int(torch.argmax(alternate).item())
        rows.append({
            "layer": index,
            "type": model.config.layer_types[index],
            "kl_base_to_modified": float(kl.item()),
            "base_top1_probability": float(p[top_original].item()),
            "modified_probability_of_base_top1": float(torch.exp(qlog[top_original]).item()),
            "top1_changed": top_changed != top_original,
            "modified_top1_text": tokenizer.decode([top_changed]),
        })
    return {
        "experiment": "module_ablation",
        "model": getattr(model.config, "_name_or_path", "LFM3"),
        "prompt": prompt,
        "token_count": int(inputs["input_ids"].shape[-1]),
        "module": which,
        "scale": scale,
        "base_top1_text": tokenizer.decode([top_original]),
        "rows": rows,
        "interpretation": "Causal next-token distribution sensitivity; not a benchmark of full task performance.",
    }

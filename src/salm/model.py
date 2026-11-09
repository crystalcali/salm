"""Lazy model loading; avoid importing torch in offline analysis/CI."""
from __future__ import annotations

from .config import architecture


def get_layers(model):
    """Locate the LFM3 decoder stack across HF wrappers."""
    candidates = [getattr(model, "model", None), getattr(model, "base_model", None), model]
    for parent in candidates:
        if parent is None:
            continue
        layers = getattr(parent, "layers", None)
        if layers is not None:
            return layers
    raise RuntimeError("Cannot find .layers; inspect installed HF LFM3 implementation.")


def load_model(model_id: str = "LiquidAI/LFM3-1.7B-Base", attention_backend: str | None = None):
    import torch
    from .diagnostics import require_triton_headers
    require_triton_headers(cuda_available=torch.cuda.is_available())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    opts = {
        "dtype": dtype,
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if attention_backend:
        opts["attn_implementation"] = attention_backend
    model = AutoModelForCausalLM.from_pretrained(model_id, **opts).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def report(model=None, config=None) -> dict:
    from .config import estimated_cache_bytes, parameter_estimates
    cfg = config if config is not None else getattr(model, "config", None)
    arch = architecture(cfg)
    counts = parameter_estimates(arch)
    result = {
        "architecture": arch,
        "parameters_estimate_by_group": counts,
        "parameters_estimate_total": sum(counts.values()),
        "cache_bytes_at_4096": estimated_cache_bytes(arch, 4096),
        "cache_bytes_at_262144": estimated_cache_bytes(arch, 262144),
    }
    if model is not None:
        result["parameters_exact_total"] = sum(p.numel() for p in model.parameters())
        result["parameters_exact_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return result

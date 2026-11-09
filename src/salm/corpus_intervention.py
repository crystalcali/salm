"""Corpus-level causal module interventions across ALL next-token prediction positions.

This is an observational sensitivity assay, not a benchmark of generated task
success. Forward hooks are cleaned up after each intervention.
"""
from __future__ import annotations

from collections import defaultdict
import math

import torch

from .intervention import _mod, _scale_hook
from .model import get_layers


@torch.inference_mode()
def run_corpus_intervention(
    model,
    tokenizer,
    prompts: list[str],
    *,
    which: str = "ffn",
    scale: float = 0.0,
    max_tokens: int = 256,
    skip_prefix: int = 8,
    selected_layers: list[int] | None = None,
) -> dict:
    """Compare modified and baseline logits over every eligible token position.

    Metrics:
    - KL(original || modified), averaged across positions
    - NLL(original/modelled) on the original prompt's NEXT actual tokens
    - change rate of argmax token
    Aggregation is token-weighted, but per-prompt results are retained to
    investigate failures concentrated in particular tasks.
    """
    if which not in {"ffn", "operator"}:
        raise ValueError("module must be 'ffn' or 'operator'")
    if not math.isfinite(scale) or abs(scale) > 3:
        raise ValueError("scale must be finite and within [-3,3]")
    if not 2 <= max_tokens <= 1024:
        raise ValueError("max_tokens must be between 2 and 1024")
    if skip_prefix < 0 or skip_prefix >= max_tokens - 1:
        raise ValueError("skip_prefix must leave at least one prediction position")
    if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError("Provide nonempty prompts")
    layers = get_layers(model)
    if selected_layers is None:
        indices = list(range(len(layers)))
    else:
        indices = list(dict.fromkeys(selected_layers))
    if not indices or any(i < 0 or i >= len(layers) for i in indices):
        raise ValueError("Invalid layer indices")
    model.eval()
    device = next(model.parameters()).device
    accum = defaultdict(lambda: {"kl": 0.0, "base_nll": 0.0, "modified_nll": 0.0,
                                 "top1_changes": 0.0, "positions": 0})
    per_prompt = []
    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        ids = inputs["input_ids"]
        total = ids.shape[-1]
        if total <= skip_prefix + 1:
            continue
        # Teacher-forced full-context predictions: position t predicts token t+1.
        with torch.inference_mode():
            original = model(**inputs, use_cache=False).logits[:, skip_prefix:-1, :].float()
            base_lp = torch.log_softmax(original, dim=-1)
            base_probs = base_lp.exp()
            target = ids[:, skip_prefix+1:]
            base_nll = -base_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            orig_top = original.argmax(dim=-1)
            n = int(target.numel())
            entry = {"prompt_index": prompt_idx, "tokens": int(total),
                     "positions_scored": n, "rows": []}
            for layer_idx in indices:
                hook = _mod(layers[layer_idx], which).register_forward_hook(_scale_hook(scale))
                try:
                    modified = model(**inputs, use_cache=False).logits[:, skip_prefix:-1, :].float()
                finally:
                    hook.remove()
                mod_lp = torch.log_softmax(modified, dim=-1)
                kl = (base_probs * (base_lp - mod_lp)).sum(-1).clamp(min=0)
                mod_nll = -mod_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                changed = (orig_top != modified.argmax(dim=-1)).float()
                kl_sum = float(kl.sum().item())
                base_nll_sum = float(base_nll.sum().item())
                modified_nll_sum = float(mod_nll.sum().item())
                change_sum = float(changed.sum().item())
                a = accum[layer_idx]
                a["kl"] += kl_sum
                a["base_nll"] += base_nll_sum
                a["modified_nll"] += modified_nll_sum
                a["top1_changes"] += change_sum
                a["positions"] += n
                entry["rows"].append({
                    "layer": layer_idx,
                    "kl_base_to_modified": kl_sum / n,
                    "nll_delta": (modified_nll_sum - base_nll_sum) / n,
                    "top1_change_rate": change_sum / n,
                })
        per_prompt.append(entry)
    if not per_prompt:
        raise ValueError("No prompts long enough after skip_prefix; use longer input or smaller skip.")
    rows = []
    for layer_idx in indices:
        a = accum[layer_idx]
        n = a["positions"]
        rows.append({
            "layer": layer_idx,
            "type": model.config.layer_types[layer_idx],
            "scored_positions": n,
            "kl_base_to_modified": a["kl"] / n,
            "base_nll": a["base_nll"] / n,
            "modified_nll": a["modified_nll"] / n,
            "nll_delta": (a["modified_nll"] - a["base_nll"]) / n,
            "top1_change_rate": a["top1_changes"] / n,
        })
    return {
        "experiment": "corpus_module_ablation",
        "model": getattr(model.config, "_name_or_path", "LFM3"),
        "module": which, "scale": scale,
        "max_tokens": max_tokens, "skip_prefix": skip_prefix,
        "prompts_scored": len(per_prompt),
        "notes": ("NLL is next-token teacher-forced loss on INPUT text, not task correctness. "
                  "KL compares the original and modified models. "
                  "Individual ablations do not measure speedups or guarantee safe block removal."),
        "rows": rows,
        "per_prompt": per_prompt,
    }

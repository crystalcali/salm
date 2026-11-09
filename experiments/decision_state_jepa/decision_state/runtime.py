"""Inference for both experiment arms with identical decoding settings."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from .model import DecisionGenerationConditioner
from .train import build_model, build_head, check_cfg


def load_checkpoint(checkpoint: str):
    root = Path(checkpoint).expanduser().resolve()
    config_file = root / 'config.json'
    if not config_file.is_file():
        raise FileNotFoundError(f'No trained checkpoint config at {config_file}')
    cfg = check_cfg(json.loads(config_file.read_text(encoding='utf-8')))
    model, tokenizer = build_model(cfg, lora_dir=root/'lora')
    head, conditioner = build_head(model, cfg)
    if head is not None:
        from safetensors.torch import load_file
        weights = root / 'decision_head.safetensors'
        if not weights.is_file():
            raise FileNotFoundError(f'Decision head missing: {weights}')
        head.load_state_dict(load_file(str(weights)))
        head.eval()
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, conditioner, cfg


@torch.inference_mode()
def generate_decision(model, tokenizer, conditioner, prompt: str, *,
                      max_new_tokens: int = 384, max_context: int = 8192) -> str:
    """Evaluate a *real saved* LoRA or Decision-State checkpoint.

    Deliberately only returns text; tool commands are parsed and executed by
    the external agent harness, never by code in the language model runtime.
    """
    if max_new_tokens < 1 or max_context < max_new_tokens + 64:
        raise ValueError('Invalid generation/context lengths')
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = ids[-(max_context-max_new_tokens):]
    if not ids:
        raise ValueError('Empty prompt')
    device = next(model.parameters()).device
    source = torch.tensor([ids], dtype=torch.long, device=device)
    stop_suffix = '</assistant>'
    from transformers import StoppingCriteria, StoppingCriteriaList
    closing = tokenizer.encode(stop_suffix, add_special_tokens=False)
    class AssistantEnd(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return bool(closing and input_ids.shape[1] >= len(closing) and
                        input_ids[0, -len(closing):].tolist() == closing)
    stops=StoppingCriteriaList([AssistantEnd()])
    if conditioner is None:
        output = model.generate(input_ids=source, max_new_tokens=max_new_tokens,
                                do_sample=False, use_cache=True, stopping_criteria=stops,
                                pad_token_id=tokenizer.pad_token_id)
    else:
        with DecisionGenerationConditioner(conditioner):
            output = model.generate(input_ids=source, max_new_tokens=max_new_tokens,
                                    do_sample=False, use_cache=True, stopping_criteria=stops,
                                    pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(output[0,source.shape[1]:], skip_special_tokens=True)
    if stop_suffix in text:
        text = text.split(stop_suffix, 1)[0] + stop_suffix
    return text

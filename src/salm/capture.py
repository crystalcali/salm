"""Non-mutating PyTorch hooks for layer, operator and FFN activations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import architecture
from .model import get_layers


def _tensor(output):
    import torch
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _tensor(output[0])
    raise TypeError("Unexpected module output type; LFM3 hook API may have changed.")


def _cpu(tensor, tokens: int) -> np.ndarray:
    return tensor.detach()[0, :tokens].to(device="cpu", dtype=__import__("torch").float16).numpy()


def _save(path, data: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **data)
    return p


def capture_activations(model, tokenizer, prompt: str, out: str,
                        max_tokens: int = 256, modules: tuple[str, ...] = ()) -> Path:
    """Capture pre-block and post-block states. All captured data is CPU float16."""
    import torch
    if max_tokens < 2 or max_tokens > 4096:
        raise ValueError("max_tokens must be between 2 and 4096; prefer <=256 initially.")
    layers = get_layers(model)
    cfg = architecture(model.config)
    if len(layers) != len(cfg["layer_types"]):
        raise ValueError("Model layer count does not match config")
    valid = {"ffn", "operator"}
    if not set(modules).issubset(valid):
        raise ValueError("modules can only contain ffn,operator")
    data: dict[str, list] = {
        "hidden": [None] * (len(layers) + 1),
        "ffn": [None] * len(layers),
        "operator": [None] * len(layers),
    }
    hooks = []
    try:
        def pre_hook(module, args):
            data["hidden"][0] = _cpu(args[0], max_tokens)
        hooks.append(layers[0].register_forward_pre_hook(pre_hook))
        for idx, layer in enumerate(layers):
            def layer_hook(module, args, output, i=idx):
                data["hidden"][i+1] = _cpu(_tensor(output), max_tokens)
            hooks.append(layer.register_forward_hook(layer_hook))
            if "ffn" in modules:
                def ffn_hook(module, args, output, i=idx):
                    data["ffn"][i] = _cpu(_tensor(output), max_tokens)
                hooks.append(layer.feed_forward.register_forward_hook(ffn_hook))
            if "operator" in modules:
                operator = getattr(layer, "self_attn", None)
                if operator is None:
                    operator = getattr(layer, "conv", None)
                if operator is None:
                    raise RuntimeError("Missing conv/self_attn module")
                def op_hook(module, args, output, i=idx):
                    data["operator"][i] = _cpu(_tensor(output), max_tokens)
                hooks.append(operator.register_forward_hook(op_hook))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        if any(x is None for x in data["hidden"]):
            raise RuntimeError("Missing hidden state; hook was not fired")
        token_ids = inputs["input_ids"][0].detach().cpu().tolist()
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
        metadata = {
            "kind": "activation_capture",
            "model": getattr(model.config, "_name_or_path", "LFM3"),
            "prompt": prompt,
            "architecture": cfg,
            "captured_modules": list(modules),
            "tokens": len(token_ids),
            "format_version": 1,
        }
        payload = {
            "hidden": np.stack(data["hidden"]),
            "token_ids": np.array(token_ids, dtype=np.int64),
            "token_strings": np.array(token_strings, dtype=str),
            "metadata": np.array(json.dumps(metadata, ensure_ascii=False)),
        }
        for name in modules:
            if any(x is None for x in data[name]):
                raise RuntimeError("Module hook did not capture all " + name + " outputs")
            payload[name] = np.stack(data[name])
        return _save(out, payload)
    finally:
        for hook in hooks:
            hook.remove()


def capture_gradients(model, tokenizer, prompt: str, out: str, max_tokens: int = 128) -> Path:
    """Gradient sensitivity of layer outputs for next-token CE, no optimizer update.

    All backbone weights are temporarily frozen. A detached embedding output is
    made differentiable, enabling hidden-state gradients without parameter grads.
    """
    import torch
    if not 2 <= max_tokens <= 512:
        raise ValueError("For gradient probes use 2 <= max_tokens <= 512.")
    layers = get_layers(model)
    old_flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in old_flags:
        p.requires_grad_(False)
    embeddings = model.get_input_embeddings()
    handles = []
    outputs = [None] * len(layers)
    try:
        def leaf_hook(module, args, output):
            return _tensor(output).detach().requires_grad_(True)
        handles.append(embeddings.register_forward_hook(leaf_hook))
        for i, layer in enumerate(layers):
            def output_hook(module, args, result, j=i):
                t = _tensor(result)
                t.retain_grad()
                outputs[j] = t
            handles.append(layer.register_forward_hook(output_hook))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        if inputs["input_ids"].shape[-1] < 2:
            raise ValueError("At least two tokens needed for next-token loss.")
        with torch.enable_grad():
            result = model(**inputs, labels=inputs["input_ids"], use_cache=False)
            if result.loss is None or not result.loss.requires_grad:
                raise RuntimeError("No differentiable causal LM loss")
            loss = result.loss
            loss.backward()
        report = {
            "model": getattr(model.config, "_name_or_path", "LFM3"),
            "prompt": prompt,
            "tokens": int(inputs["input_ids"].shape[-1]),
            "loss": float(loss.item()),
            "layers": [],
        }
        for i, t in enumerate(outputs):
            if t is None or t.grad is None:
                raise RuntimeError("No gradient for layer " + str(i))
            g = t.grad.detach().float()
            report["layers"].append({
                "layer": i, "type": model.config.layer_types[i],
                "gradient_rms": float(g.square().mean().sqrt().item()),
                "gradient_norm": float(g.norm().item()),
            })
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return path
    finally:
        for handle in handles:
            handle.remove()
        for p, flag in old_flags:
            p.requires_grad_(flag)
        model.zero_grad(set_to_none=True)


def demo_capture(out: str = "runs/demo.npz", seed: int = 4) -> Path:
    """Small synthetic, clearly labeled example. Does not load any LFM weights."""
    rng = np.random.default_rng(seed)
    t, d = 48, 64
    states = [rng.standard_normal((t, d))]
    operators, ffns = [], []
    for i in range(24):
        op = rng.standard_normal((t, d)) * (0.045 + i * 0.001)
        ff = rng.standard_normal((t, d)) * (0.035 + i * 0.001)
        operators.append(op)
        ffns.append(ff)
        states.append(states[-1] + op + ff)
    meta = {
        "kind": "synthetic_demo",
        "model": "SYNTHETIC - NOT LFM3",
        "prompt": "Synthetic data for testing the UI",
        "architecture": {**architecture(), "hidden_size": d},
        "captured_modules": ["ffn", "operator"],
        "tokens": t,
        "format_version": 1,
    }
    return _save(out, {
        "hidden": np.asarray(states, dtype=np.float16),
        "ffn": np.asarray(ffns, dtype=np.float16),
        "operator": np.asarray(operators, dtype=np.float16),
        "token_ids": np.arange(t, dtype=np.int64),
        "token_strings": np.array(["tok_" + str(i) for i in range(t)]),
        "metadata": np.array(json.dumps(meta)),
    })

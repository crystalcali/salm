"""Differentiable LFM3 training engine. Shared online/target encoder, no EMA.

The first causal LM forward uses ALL input tokens for SFT, but accesses ONLY
the hidden state at the prefix cut; the causal model cannot encode later
tokens at that position. The second forward encodes ONLY target suffix
tokens with the same trainable model. This is NOT a teacher-student network.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .data import read_traces, verify_disjoint, build_pair, collate_pairs
from .objective import TraceJEPAObjective


def get_layers_train(model):
    # PEFT wrappers add .base_model.model before the original LFM3.
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(core, "model") and hasattr(core.model, "layers"):
        return core.model.layers
    from salm.model import get_layers
    return get_layers(core)


class LayerCapture:
    """Retains differentiable outputs; never .detach(), .cpu() or NumPy."""
    def __init__(self, model, layer_indices: list[int]):
        self.layers = get_layers_train(model)
        self.indices = layer_indices
        self.hidden = {}
        self.handles = []

    def __enter__(self):
        for i in self.indices:
            if not 0 <= i < len(self.layers):
                raise ValueError(f"Layer {i} outside model")
            def hook(module, args, output, idx=i):
                h = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(h, torch.Tensor):
                    raise TypeError("Expected decoder hidden state tensor")
                self.hidden[idx] = h
            self.handles.append(self.layers[i].register_forward_hook(hook))
        return self

    def reset(self):
        self.hidden = {}

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.hidden = {}


def make_model(model_id: str, revision: str | None, *, use_lora: bool,
               lora_rank: int, lora_alpha: int, gradient_checkpointing: bool):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("LFM3 training requires CUDA; run unit tests on CPU instead")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs EOS or PAD")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        dtype=torch.bfloat16, device_map=None,
    ).to("cuda")
    model.config.use_cache = False
    if gradient_checkpointing:
        # Checkpointed forward hooks are reentrant for some model versions;
        # the selected-layer capture is validated before enabling this.
        raise ValueError("gradient_checkpointing currently unsupported with differentiable hooks; keep false")
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        wanted = ["w1", "w2", "w3", "q_proj", "k_proj", "v_proj", "o_proj",
                  "in_proj", "out_proj"]
        available = sorted(set(name.rsplit(".", 1)[-1]
                               for name, m in model.named_modules()
                               if isinstance(m, nn.Linear)))
        targets = [name for name in wanted if name in available]
        if not {"w1", "w2", "w3"}.issubset(targets):
            raise RuntimeError(f"LFM3 FFN w1/w2/w3 not located. Linear modules: {available}")
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=0.0, bias="none", target_modules=targets,
        ))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[SALM] LoRA targets={targets}, trainable={trainable:,}")
    return model, tokenizer


def prepare_modules(model, layers: list[int], *, latent_size: int,
                    predictor_width: int, projections: int,
                    sigreg_weight: float):
    hidden_size = model.config.hidden_size
    heads = nn.ModuleDict({
        str(i): TraceJEPAObjective(hidden_size, latent_size=latent_size,
                                   predictor_width=predictor_width,
                                   projections=projections, sigreg_weight=sigreg_weight)
        for i in layers
    }).to("cuda")
    return heads


def compute_batch(model, heads, capture: LayerCapture | None,
                  batch: dict, jepa_weight: float):
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    if capture is not None:
        capture.reset()
    # SFT on full prompt + trace with instruction labels -100.
    result = model(
        input_ids=batch["full_ids"], attention_mask=batch["full_mask"],
        labels=batch["labels"], use_cache=False,
    )
    sft = result.loss.float()
    if not torch.isfinite(sft):
        raise FloatingPointError("Non-finite SFT loss")
    if capture is None:
        return sft, {"sft": float(sft.detach()), "jepa": 0.0, "pred": 0.0,
                     "sigreg": 0.0, "latent_var": 0.0}
    full_states = dict(capture.hidden)
    capture.reset()
    # Target branch sees suffix only; it must NOT receive the full trace.
    model(
        input_ids=batch["target_ids"], attention_mask=batch["target_mask"],
        use_cache=False, logits_to_keep=1,
    )
    target_states = dict(capture.hidden)
    if len(full_states) != len(heads) or len(target_states) != len(heads):
        raise RuntimeError("Missing selected differentiable layer hooks")
    bs = batch["cut_pos"].shape[0]
    idx = torch.arange(bs, device=device)
    target_idx = batch["target_mask"].sum(-1) - 1
    jloss, predictions, regs, variances = [], [], [], []
    for layer_str, head in heads.items():
        i = int(layer_str)
        prefix = full_states[i][idx, batch["cut_pos"]]
        target = target_states[i][idx, target_idx]
        obj = head(prefix, target)
        jloss.append(obj.total)
        predictions.append(obj.prediction.detach())
        regs.append((obj.sigreg_online.detach() + obj.sigreg_target.detach()) / 2)
        variances.append((obj.online_variance + obj.target_variance) / 2)
    jepa = torch.stack(jloss).mean()
    loss = sft + jepa_weight * jepa
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite JEPA/SIGReg loss")
    return loss, {
        "sft": float(sft.detach()), "jepa": float(jepa.detach()),
        "pred": float(torch.stack(predictions).mean()),
        "sigreg": float(torch.stack(regs).mean()),
        "latent_var": float(torch.stack(variances).mean()),
    }


def write_event(path: Path, item: dict):
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_checkpoint(out: Path, step: int, model, heads, optimizer, cfg: dict):
    """Only trainable LoRA tensors and local heads, NOT the 1.7B frozen weights."""
    dest = out / f"step_{step:06d}.pt"
    torch.save({
        "step": step,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "heads": heads.state_dict() if heads is not None else None,
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }, dest)
    return dest


def restore_checkpoint(path: str, model, heads, optimizer) -> int:
    # Only load trusted checkpoints produced by SALM.
    state = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    for n, tensor in state["trainable"].items():
        if n not in params or params[n].shape != tensor.shape:
            raise ValueError(f"Checkpoint model mismatch: {n}")
        with torch.no_grad():
            params[n].copy_(tensor.to(device=params[n].device, dtype=params[n].dtype))
    if heads is not None:
        if state["heads"] is None:
            raise ValueError("Checkpoint missing JEPA heads")
        heads.load_state_dict(state["heads"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


@torch.no_grad()
def evaluate_teacher_forcing(model, tokenizer, rows: list[dict],
                             max_length: int, limit: int = 40) -> dict:
    """Held-out SFT NLL; same metric for SFT control and JEPA+SIGReg."""
    model.eval()
    losses = []
    for r in rows[:limit]:
        pair = build_pair(r, tokenizer, random.Random(0), max_length)
        batch = collate_pairs([pair], tokenizer.pad_token_id)
        batch = {k: v.to("cuda") for k, v in batch.items()}
        result = model(input_ids=batch["full_ids"], attention_mask=batch["full_mask"],
                       labels=batch["labels"], use_cache=False)
        n = int((batch["labels"][:, 1:] != -100).sum().item())
        if n > 0:
            losses.append((float(result.loss), n))
    model.train()
    return {
        "nll": sum(loss*n for loss, n in losses) / max(sum(n for _, n in losses), 1),
        "scored_tokens": sum(n for _, n in losses),
        "tasks": len(losses),
    }


def train(config: dict) -> dict:
    """Train SFT control or the SINGLE requested method: JEPA + SIGReg no EMA."""
    defaults = {
        "model": "LiquidAI/LFM3-1.7B-Base", "revision": None,
        "mode": "jepa_sigreg", "seed": 42, "batch_size": 4,
        "grad_accum": 2, "max_steps": 50, "max_length": 512,
        "layers": [7, 12], "latent_size": 256, "predictor_width": 512,
        "projections": 128, "sigreg_weight": 0.02, "jepa_weight": 0.2,
        "lr": 0.0001, "weight_decay": 0.01, "lora_rank": 16,
        "lora_alpha": 32, "checkpoint_every": 25,
        "eval_every": 25, "eval_limit": 40, "gradient_checkpointing": False,
        "use_lora": True, "out": "runs/trace_jepa", "resume": None,
    }
    cfg = {**defaults, **config}
    if cfg["mode"] not in ("jepa_sigreg", "sft"):
        raise ValueError("Only jepa_sigreg (no EMA) and the SFT control are supported")
    if cfg["mode"] == "jepa_sigreg" and cfg["batch_size"] < 2:
        raise ValueError("SIGReg requires multiple independent tasks per batch")
    if cfg["batch_size"] <= 0 or cfg["grad_accum"] <= 0 or cfg["max_steps"] <= 0:
        raise ValueError("Invalid batch/step configuration")
    if cfg["mode"] == "jepa_sigreg" and not cfg["layers"]:
        raise ValueError("JEPA requires at least one selected layer")
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    train_rows = read_traces(cfg["train"])
    val_rows = read_traces(cfg["val"])
    verify_disjoint(train_rows, val_rows)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    model, tokenizer = make_model(
        cfg["model"], cfg["revision"], use_lora=cfg["use_lora"],
        lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
    )
    if not cfg["use_lora"]:
        print("[SALM] Full model finetuning enabled: check memory before proceeding")
    layers = sorted(set(cfg["layers"])) if cfg["mode"] == "jepa_sigreg" else []
    heads = prepare_modules(model, layers, latent_size=cfg["latent_size"],
                            predictor_width=cfg["predictor_width"],
                            projections=cfg["projections"],
                            sigreg_weight=cfg["sigreg_weight"]) if layers else None
    params = [p for p in model.parameters() if p.requires_grad]
    if heads is not None:
        params += list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    start = restore_checkpoint(cfg["resume"], model, heads, optimizer) if cfg["resume"] else 0
    cfg["resolved_model_revision"] = getattr(model.config, "_commit_hash", None)
    (out/"config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    history = out / "history.jsonl"
    t0 = time.monotonic()
    model.train()
    if heads is not None:
        heads.train()
    optimizer.zero_grad(set_to_none=True)
    # Reproducible sampling from distinct tasks within each microbatch.
    # RNG depends on global step, so resume gives the same examples.
    with LayerCapture(model, layers) if layers else _NullCapture() as cap:
        for step in range(start, cfg["max_steps"]):
            totals = []
            for micro in range(cfg["grad_accum"]):
                rng = random.Random(cfg["seed"] + step * 10009 + micro)
                indices = rng.sample(range(len(train_rows)), cfg["batch_size"]) if len(train_rows) >= cfg["batch_size"] else None
                if indices is None:
                    raise ValueError("Not enough independent training tasks for batch_size")
                pairs = [build_pair(train_rows[i], tokenizer, rng, cfg["max_length"]) for i in indices]
                batch = collate_pairs(pairs, tokenizer.pad_token_id)
                loss, log = compute_batch(model, heads, cap if layers else None, batch, cfg["jepa_weight"])
                (loss / cfg["grad_accum"]).backward()
                totals.append(log)
            grad_norm = float(clip_grad_norm_(params, 1.0))
            if not math.isfinite(grad_norm):
                raise FloatingPointError("Gradient NaN/Inf")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            done = step + 1
            entry = {
                "step": done, "mode": cfg["mode"],
                **{k: sum(d[k] for d in totals)/len(totals) for k in totals[0]},
                "grad_norm": grad_norm, "lr": cfg["lr"],
                "seconds": time.monotonic()-t0,
                "cuda_peak_gib": torch.cuda.max_memory_allocated()/2**30,
            }
            if done % cfg["eval_every"] == 0 or done == cfg["max_steps"]:
                metrics = evaluate_teacher_forcing(model, tokenizer, val_rows,
                                                   cfg["max_length"], cfg["eval_limit"])
                entry.update({"val_nll": metrics["nll"], "val_tokens": metrics["scored_tokens"]})
            write_event(history, entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            if done % cfg["checkpoint_every"] == 0 or done == cfg["max_steps"]:
                save_checkpoint(out, done, model, heads, optimizer, cfg)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out / "adapter_or_model", safe_serialization=True)
    if heads is not None:
        torch.save(heads.state_dict(), out / "trace_jepa_heads.pt")
    return {"steps": cfg["max_steps"], "out": str(out),
            "mode": cfg["mode"], "trained_on_cuda": True}


class _NullCapture:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

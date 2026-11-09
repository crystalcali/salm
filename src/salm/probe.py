"""Supervised depth-probe experiment (not a temporal JEPA world model)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_pairs(path: Path, source: int, target: int, limit: int = 64):
    if not 0 <= source < target < 24:
        raise ValueError("Block indices must satisfy 0 <= source < target < 24")
    with np.load(path, allow_pickle=False) as archive:
        hidden = archive["hidden"]
        meta = json.loads(str(archive["metadata"].item()))
        if meta["kind"] != "activation_capture" or hidden.shape[0] != 25:
            raise ValueError("Not a real 24-layer LFM3 capture: " + str(path))
        return (np.asarray(hidden[source + 1, :limit], dtype=np.float32),
                np.asarray(hidden[target + 1, :limit], dtype=np.float32))


def train_depth_probe(directory: str, source: int, target: int,
                      out: str, epochs: int = 100, width: int = 256,
                      lr: float = 0.001, seed: int = 42) -> dict:
    """Train by prompt-file split; calculate validation MSE and reference baselines."""
    import torch
    torch.manual_seed(seed)
    files = sorted(Path(directory).glob("*.npz"))
    pairs = []
    for f in files:
        try:
            pairs.append((f.name, *load_pairs(f, source, target)))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    if len(pairs) < 3:
        raise ValueError("Need >=3 REAL capture files (different prompts) for train/validation.")
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    split = max(1, min(len(pairs)-1, int(len(pairs)*0.8)))
    train_files, val_files = pairs[:split], pairs[split:]
    xt = np.concatenate([p[1] for p in train_files])
    yt = np.concatenate([p[2] for p in train_files])
    xv = np.concatenate([p[1] for p in val_files])
    yv = np.concatenate([p[2] for p in val_files])
    if xt.shape[1] != yt.shape[1] or xv.shape[1] != xt.shape[1]:
        raise ValueError("Capture feature dimensions disagree")
    xm, xs = xt.mean(0), xt.std(0).clip(1e-5)
    ym, ys = yt.mean(0), yt.std(0).clip(1e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xtr = torch.from_numpy((xt-xm)/xs).to(device)
    ytr = torch.from_numpy((yt-ym)/ys).to(device)
    xva = torch.from_numpy((xv-xm)/xs).to(device)
    yva = torch.from_numpy((yv-ym)/ys).to(device)
    dim = xt.shape[1]
    network = torch.nn.Sequential(
        torch.nn.LayerNorm(dim),
        torch.nn.Linear(dim, width),
        torch.nn.GELU(),
        torch.nn.Linear(width, dim),
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
    best, best_epoch, patience = float("inf"), 0, 20
    for epoch in range(epochs):
        network.train()
        for indices in torch.randperm(len(xtr), device=device).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = (network(xtr[indices]) - ytr[indices]).square().mean()
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            val = float((network(xva) - yva).square().mean().item())
        if val < best - 1e-5:
            best, best_epoch = val, epoch+1
        elif epoch+1-best_epoch >= patience:
            break
    report = {
        "experiment": "supervised_depth_distillation_not_JEPA",
        "source_layer": source, "target_layer": target,
        "train_files": [x[0] for x in train_files],
        "validation_files": [x[0] for x in val_files],
        "train_tokens": int(len(xt)), "validation_tokens": int(len(xv)),
        "normalization": "per_feature_train_only",
        "best_epoch": best_epoch,
        "validation_mse_normalized": best,
        "mean_target_baseline_mse_normalized": float(np.mean(((yv-ym)/ys)**2)),
        "identity_baseline_mse_normalized": float(np.mean((((xv-ym)/ys)-((yv-ym)/ys))**2)),
        "trainable_parameters": sum(p.numel() for p in network.parameters()),
        "device": device,
    }
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    return report

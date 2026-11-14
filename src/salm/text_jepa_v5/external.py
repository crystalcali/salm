"""Common frozen-backbone evaluation; no JEPA-specific head in downstream probes."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import Config
from .model import build_model
from .runner import precision


@torch.no_grad()
def encode(model, tokenizer, texts, batch_size=16, max_length=256):
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
            add_special_tokens=False, max_length=min(max_length, model.cfg.seq_len),
            return_tensors="pt")
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        with precision(device):
            h = model.encoder(ids, padding_mask=~mask.bool())
        pooled = (h.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        outputs.append(F.normalize(pooled, dim=-1).cpu().numpy())
    return np.concatenate(outputs)


def evaluate_external(checkpoint, corpus, output=None, train_limit=4000, batch_size=16,
                      final_test=False, device=None):
    from datasets import load_dataset
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["corpus_fingerprint"] != corpus.fingerprint:
        raise ValueError("Evaluation tokenizer/corpus differs from checkpoint")
    cfg = Config(**state["config"]).validate()
    model = build_model(cfg).to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(state["model"])
    tokenizer = AutoTokenizer.from_pretrained(corpus.root / "tokenizer")
    # Revisions travel inside the corpus fingerprint and private corpus snapshot.
    refs = corpus.manifest["sources"]["evaluation"]
    stsb = load_dataset("sentence-transformers/stsb", revision=refs["sentence-transformers/stsb"],
                        split="test" if final_test else "validation")
    left = encode(model, tokenizer, list(stsb["sentence1"]), batch_size)
    right = encode(model, tokenizer, list(stsb["sentence2"]), batch_size)
    similarity = np.sum(left * right, axis=1)
    correlation = float(spearmanr(similarity, stsb["score"]).statistic)
    sst_train = load_dataset("stanfordnlp/sst2", revision=refs["stanfordnlp/sst2"], split="train")
    sst_val = load_dataset("stanfordnlp/sst2", revision=refs["stanfordnlp/sst2"], split="validation")
    selected = sst_train.shuffle(seed=42).select(range(min(train_limit, len(sst_train))))
    x_train = encode(model, tokenizer, list(selected["sentence"]), batch_size)
    x_val = encode(model, tokenizer, list(sst_val["sentence"]), batch_size)
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=42))
    probe.fit(x_train, np.asarray(selected["label"]))
    accuracy = float(probe.score(x_val, np.asarray(sst_val["label"])))
    result = {"sts_spearman": correlation if np.isfinite(correlation) else None,
        "sts_split": "test" if final_test else "validation", "sst2_accuracy": accuracy,
        "sst2_split": "validation", "probe_training_examples": len(selected),
        "pooling": "mean unprojected backbone states, then L2 normalization",
        "probe": "StandardScaler fitted on probe train; logistic C=1, no tuning",
        "dataset_revisions": refs, "config": asdict(cfg), "step": state["step"],
        "corpus_fingerprint": corpus.fingerprint,
        "note": "Frozen representation tests, not evidence of autonomous latent reasoning. "
                "Web-pretraining contamination is not fully excluded."}
    destination = Path(output or Path(checkpoint).with_name("external.json"))
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result

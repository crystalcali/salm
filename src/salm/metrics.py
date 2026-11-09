"""Stable NumPy analyses of hidden states; no model dependency."""
from __future__ import annotations

import numpy as np


def _as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def token_norms(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(_as_float(x), axis=-1)


def token_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b = _as_float(x), _as_float(y)
    if a.shape != b.shape:
        raise ValueError("Token alignment requires equal shapes")
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return (a * b).sum(axis=-1) / np.maximum(denominator, 1e-12)


def token_similarity(x: np.ndarray) -> np.ndarray:
    a = _as_float(x)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return np.clip(a @ a.T, -1.0, 1.0)


def centered_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA through centered token Gram matrices (not raw hidden-vector MSE)."""
    a, b = _as_float(x), _as_float(y)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("CKA expects [same_token_count, features] arrays")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    ka, kb = a @ a.T, b @ b.T
    denom = np.linalg.norm(ka) * np.linalg.norm(kb)
    return float(np.sum(ka * kb) / denom) if denom > 1e-20 else 0.0


def effective_rank(x: np.ndarray) -> float:
    """Exponentiated entropy of singular-value energy; max rank <= token count - 1."""
    a = _as_float(x)
    if a.ndim != 2:
        raise ValueError("Expected [tokens, hidden]")
    a = a - a.mean(axis=0, keepdims=True)
    gram = a @ a.T
    values = np.maximum(np.linalg.eigvalsh(gram), 0)
    total = values.sum()
    if total <= 1e-20:
        return 0.0
    probs = values[values > 1e-16 * total] / total
    return float(np.exp(-np.sum(probs * np.log(probs))))


def layer_rows(hidden: np.ndarray, layer_types: list[str] | None = None) -> list[dict]:
    """hidden: embedding followed by block outputs, [L+1,T,D]."""
    if hidden.ndim != 3 or hidden.shape[0] < 2:
        raise ValueError("Expected hidden state array [layer+1, token, feature]")
    rows = []
    for i in range(1, hidden.shape[0]):
        before, after = _as_float(hidden[i - 1]), _as_float(hidden[i])
        delta = after - before
        rows.append({
            "layer": i - 1,
            "type": layer_types[i - 1] if layer_types and i-1 < len(layer_types) else "unknown",
            "rms": float(np.sqrt(np.mean(after ** 2))),
            "delta_rms": float(np.sqrt(np.mean(delta ** 2))),
            "delta_ratio": float(np.linalg.norm(delta) / max(np.linalg.norm(before), 1e-12)),
            "cosine": float(np.mean(token_cosine(before, after))),
        })
    return rows

"""Capture file metadata and safe file selection; no torch or Streamlit required."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def inspect_capture(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        hidden_shape = list(archive["hidden"].shape)
        kinds = list(archive.files)
    return {
        "file": str(path),
        "kind": metadata.get("kind", "unknown"),
        "model": metadata.get("model", "unknown"),
        "prompt": metadata.get("prompt"),
        "hidden_shape": hidden_shape,
        "is_real_lfm3_shape": len(hidden_shape) == 3 and hidden_shape[0] == 25
        and hidden_shape[2] == 2048 and metadata.get("kind") == "activation_capture",
        "arrays": kinds,
    }


def choose_capture(files: list[Path]) -> int:
    """Select newest REAL capture, falling back to latest any capture."""
    if not files:
        raise ValueError("No captures available")
    real = []
    for i, file in enumerate(files):
        try:
            with np.load(file, allow_pickle=False) as data:
                kind = json.loads(str(data["metadata"].item())).get("kind")
            if kind == "activation_capture":
                real.append(i)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    candidates = real if real else list(range(len(files)))
    return max(candidates, key=lambda i: files[i].stat().st_mtime)

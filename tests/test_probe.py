import json

import numpy as np
import pytest

from salm.config import architecture
from salm.probe import load_pairs, train_depth_probe


def _fake_capture(file, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(25, 12, 8)).astype("float16")
    meta = {"kind": "activation_capture", "architecture": architecture()}
    np.savez_compressed(file, hidden=x, metadata=np.array(json.dumps(meta)))


def test_invalid_layer_pairs(tmp_path):
    p = tmp_path / "a.npz"
    _fake_capture(p, 2)
    with pytest.raises(ValueError):
        load_pairs(p, 6, 6)


def test_probe_filewise_split(tmp_path):
    pytest.importorskip("torch")
    for i in range(4):
        _fake_capture(tmp_path / f"prompt_{i}.npz", i)
    result = train_depth_probe(str(tmp_path), 4, 7, str(tmp_path / "probe.json"),
                               epochs=3, width=8, seed=0)
    assert result["experiment"] == "supervised_depth_distillation_not_JEPA"
    assert not (set(result["train_files"]) & set(result["validation_files"]))
    assert result["validation_mse_normalized"] >= 0
    assert result["trainable_parameters"] > 0

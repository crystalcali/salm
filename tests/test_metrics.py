import numpy as np
import pytest

from salm.metrics import (
    centered_cka, effective_rank, layer_rows, token_cosine, token_norms,
    token_similarity,
)


def test_centered_cka_rotation_invariant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 10))
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    assert centered_cka(x, x @ q) == pytest.approx(1.0, abs=1e-8)


def test_cka_zero_variance():
    assert centered_cka(np.ones((8, 4)), np.zeros((8, 4))) == 0.0


def test_effective_rank():
    a = np.ones((10, 5))
    assert effective_rank(a) == 0.0
    x = np.stack([np.arange(10), np.arange(10)], axis=1)
    assert effective_rank(x) == pytest.approx(1.0, abs=1e-7)


def test_norm_cosine_and_token_similarity():
    x = np.eye(4)
    assert np.allclose(token_norms(x), 1.0)
    assert np.allclose(token_cosine(x, x), 1.0)
    assert np.allclose(token_similarity(x), np.eye(4))


def test_layer_rows_with_identical_states():
    x = np.ones((3, 4, 2), dtype=np.float16)
    rows = layer_rows(x, ["conv", "full_attention"])
    assert len(rows) == 2
    assert rows[1]["delta_rms"] == 0.0
    assert rows[0]["type"] == "conv"


def test_cka_input_shape_validation():
    with pytest.raises(ValueError):
        centered_cka(np.zeros((3, 4)), np.zeros((4, 4)))

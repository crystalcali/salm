from salm.config import REFERENCE, LAYER_TYPES, estimated_cache_bytes, parameter_estimates


def test_architecture_counts():
    assert len(LAYER_TYPES) == 24
    assert LAYER_TYPES.count("conv") == 13
    assert LAYER_TYPES.count("sliding_attention") == 5
    assert LAYER_TYPES.count("full_attention") == 6


def test_global_cache_matches_theory():
    result = estimated_cache_bytes(REFERENCE, 262144)
    assert result["full_attention"] == 6 * 262144 * 2 * 4 * 128 * 2
    assert result["total"] > result["full_attention"]


def test_parameter_sanity():
    breakdown = parameter_estimates(REFERENCE)
    assert breakdown["FFN"] == 24 * 3 * 2048 * 7680
    assert breakdown["Embedding"] == 128000 * 2048
    assert all(v > 0 for v in breakdown.values())

"""Reference architecture; real inspection also accepts the downloaded HF config."""
from __future__ import annotations

LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention",
    "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv", "sliding_attention",
    "conv", "full_attention", "conv",
]
REFERENCE = {
    "model_type": "lfm3",
    "hidden_size": 2048,
    "intermediate_size": 7680,
    "vocab_size": 128000,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "conv_L_cache": 7,
    "sliding_window": 256,
    "max_position_embeddings": 262144,
    "tie_word_embeddings": True,
    "layer_types": LAYER_TYPES,
}


def architecture(config: object | None = None) -> dict:
    """Serialize a subset of HF config, or use the offline official reference."""
    if config is None:
        return dict(REFERENCE)
    keys = REFERENCE.keys()
    return {key: getattr(config, key, REFERENCE[key]) for key in keys}


def estimated_cache_bytes(arch: dict, tokens: int, batch: int = 1, bytes_per_element: int = 2) -> dict:
    """KV estimate for a single sequence; assumes bounded sliding and conv states."""
    types = arch["layer_types"]
    kvwidth = arch["num_key_value_heads"] * arch["head_dim"]
    per_token = 2 * kvwidth * bytes_per_element * batch
    full = types.count("full_attention") * tokens * per_token
    sliding = types.count("sliding_attention") * min(tokens, arch["sliding_window"]) * per_token
    # Conservative conv-state tensor count; implementation details may vary.
    conv = types.count("conv") * arch["hidden_size"] * arch["conv_L_cache"] * bytes_per_element * batch
    return {"full_attention": full, "sliding_attention": sliding, "conv": conv, "total": full + sliding + conv}


def parameter_estimates(arch: dict) -> dict:
    """Matrix-only parameter estimates; ignores norms, gates, tiny biases."""
    h, f = arch["hidden_size"], arch["intermediate_size"]
    nh, nk, hd = arch["num_attention_heads"], arch["num_key_value_heads"], arch["head_dim"]
    ff = len(arch["layer_types"]) * 3 * h * f
    attn = sum(t != "conv" for t in arch["layer_types"]) * (2 * h * nh * hd + 2 * h * nk * hd)
    conv = arch["layer_types"].count("conv") * (4 * h * h + h * arch["conv_L_cache"])
    emb = h * arch["vocab_size"] * (1 if arch["tie_word_embeddings"] else 2)
    return {"FFN": ff, "Attention": attn, "Convolution": conv, "Embedding": emb}

"""Offline visual explorer. Run with 'salm dashboard'; model inference is separate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from salm.config import LAYER_TYPES
from salm.metrics import centered_cka, effective_rank, layer_rows, token_norms, token_similarity

st.set_page_config(page_title="SALM | LFM3 Research", layout="wide", page_icon="🧠")
st.title("SALM · LFM3 Internal Explorer")
st.caption("Layers, signals, token geometry, gradient sensitivity, depth-distillation diagnostics")


@st.cache_data(show_spinner=False)
def open_capture(path: str, mtime: float):
    with np.load(path, allow_pickle=False) as item:
        return {key: item[key].copy() for key in item.files}


st.sidebar.header("Local experiment data")
folder_text = st.sidebar.text_input("Capture directory", value="runs")
folder = Path(folder_text).expanduser()
files = sorted(folder.glob("*.npz")) if folder.is_dir() else []
if not files:
    st.info("No .npz captures yet. Run: python -m salm.cli demo --out runs/demo.npz")
    st.stop()
selected = st.sidebar.selectbox("Capture", files, format_func=lambda x: x.name)
data = open_capture(str(selected.resolve()), selected.stat().st_mtime)
meta = json.loads(str(data["metadata"].item()))
hidden = data["hidden"].astype(np.float32)
token_strings = data["token_strings"].astype(str).tolist()
layer_types = meta.get("architecture", {}).get("layer_types", LAYER_TYPES)
num_layers, n_tokens, dim = hidden.shape[0] - 1, hidden.shape[1], hidden.shape[2]
if meta.get("kind") == "synthetic_demo":
    st.warning("SYNTHETIC DEMO — these are not LFM3 activations or performance measurements.")
st.sidebar.metric("Captured tokens", n_tokens)
st.sidebar.metric("Layer states", num_layers + 1)
st.sidebar.caption("All comparisons use the same token positions within this capture.")
t0, t1, t2, t3, t4 = st.tabs([
    "Architecture", "Layer dynamics", "Token geometry", "FFN / operator", "Gradients & probes",
])


def chart(fig, **kwargs):
    fig.update_layout(margin=dict(l=15, r=20, b=30, t=45))
    st.plotly_chart(fig, use_container_width=True, **kwargs)


with t0:
    st.subheader("Tri-hybrid network")
    count = pd.Series(layer_types).value_counts()
    metrics = st.columns(4)
    metrics[0].metric("Layers", num_layers)
    metrics[1].metric("Convolution", int(count.get("conv", 0)))
    metrics[2].metric("Sliding attention", int(count.get("sliding_attention", 0)))
    metrics[3].metric("Full attention", int(count.get("full_attention", 0)))
    a = pd.DataFrame({"layer": list(range(num_layers)), "type": layer_types[:num_layers]})
    fig = px.scatter(a, x="layer", y="type", color="type",
                     title="Actual operator sequence (0-based layer indices)")
    fig.update_traces(marker_size=15)
    chart(fig)
    arch = meta.get("architecture", {})
    st.json({
        "model": meta.get("model"), "kind": meta.get("kind"),
        "hidden_size_from_capture": dim,
        "config_hidden_size": arch.get("hidden_size"),
        "intermediate_size": arch.get("intermediate_size"),
        "num_attention_heads": arch.get("num_attention_heads"),
        "num_key_value_heads": arch.get("num_key_value_heads"),
        "sliding_window": arch.get("sliding_window"),
        "max_position_embeddings": arch.get("max_position_embeddings"),
    })
    st.caption("Values from capture config; synthetic demo uses a reduced feature dimension.")


with t1:
    st.subheader("Changes in the residual stream")
    rows = layer_rows(hidden, layer_types)
    frame = pd.DataFrame(rows)
    left, right = st.columns(2)
    with left:
        chart(px.line(frame, x="layer", y=["rms", "delta_rms"],
                      title="RMS of state and block-level change"))
    with right:
        chart(px.line(frame, x="layer", y=["delta_ratio", "cosine"],
                      title="Relative change and adjacent-layer cosine"))
    norm_grid = np.linalg.norm(hidden, axis=-1)
    chart(px.imshow(norm_grid, aspect="auto", labels={
        "x": "token position", "y": "embedding + block outputs", "color": "L2 norm"
    }, title="Hidden-state magnitude across depth and tokens"))
    st.dataframe(frame, use_container_width=True)


with t2:
    st.subheader("Geometry of representations")
    left, right = st.columns(2)
    with left:
        source = st.slider("State A (0 = embedding)", 0, num_layers, min(8, num_layers))
    with right:
        target = st.slider("State B (0 = embedding)", 0, num_layers, min(16, num_layers))
    a, b = hidden[source], hidden[target]
    c1, c2, c3 = st.columns(3)
    c1.metric("Centered linear CKA", f"{centered_cka(a,b):.4f}")
    c2.metric("Effective rank A", f"{effective_rank(a):.1f}")
    c3.metric("Effective rank B", f"{effective_rank(b):.1f}")
    labels = [f"{i}: {v[:13]}" for i, v in enumerate(token_strings)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n_tokens)), y=token_norms(a), name=f"State {source}"))
    fig.add_trace(go.Scatter(x=list(range(n_tokens)), y=token_norms(b), name=f"State {target}"))
    fig.update_layout(title="Per-token hidden vector norms", xaxis_title="Token position")
    chart(fig)
    max_display = min(n_tokens, 160)
    show = st.slider("Tokens shown in similarity map", 2, max_display, min(64, max_display))
    map_layer = st.selectbox("State for token-to-token similarity", [source, target],
                              format_func=lambda x: f"State {x}")
    matrix = token_similarity(hidden[map_layer, :show])
    chart(px.imshow(matrix, x=labels[:show], y=labels[:show], zmin=-1, zmax=1,
                    title="Cosine similarity of token representations"))
    st.caption("CKA is invariant to orthogonal feature rotations; high CKA does not establish semantic equivalence.")


with t3:
    st.subheader("Inside each LFM3 block")
    layer = st.slider("Block index", 0, num_layers - 1, min(7, num_layers - 1))
    before, after = hidden[layer], hidden[layer+1]
    st.write("Block type:", layer_types[layer], " | State shape:", tuple(after.shape))
    summary = {
        "input_rms": float(np.sqrt(np.mean(before.astype(np.float64) ** 2))),
        "output_rms": float(np.sqrt(np.mean(after.astype(np.float64) ** 2))),
        "full_block_delta_rms": float(np.sqrt(np.mean((after.astype(np.float64)-before) ** 2))),
    }
    for key in ("operator", "ffn"):
        if key in data:
            part = data[key][layer].astype(np.float64)
            summary[key + "_raw_output_rms"] = float(np.sqrt(np.mean(part ** 2)))
    st.json(summary)
    st.caption("Operator/FFN hooks are captured BEFORE optional post-operator/post-FFN norms. Their sum need not equal the full block delta.")
    frame = pd.DataFrame({"token": np.arange(n_tokens), "block_delta": token_norms(after - before)})
    for key in ("operator", "ffn"):
        if key in data:
            frame[key] = token_norms(data[key][layer])
    chart(px.line(frame, x="token", y=[x for x in frame.columns if x != "token"],
                  title="Per-token signal amplitudes"))
    if not any(key in data for key in ("operator", "ffn")):
        st.info("Re-capture with --modules ffn,operator for internal module signals.")


with t4:
    st.subheader("Local training diagnostics")
    grad_files = sorted(folder.glob("*gradient*.json"))
    if grad_files:
        chosen = st.selectbox("Gradient report", grad_files, format_func=lambda x: x.name)
        grad = json.loads(chosen.read_text())
        gdf = pd.DataFrame(grad.get("layers", []))
        if not gdf.empty:
            chart(px.line(gdf, x="layer", y=["gradient_rms", "gradient_norm"],
                          title="Next-token CE sensitivity per layer"))
            st.write("CE loss:", grad.get("loss"), "| Tokens:", grad.get("tokens"))
    else:
        st.info("Run: python -m salm.cli gradient --prompt 'your prompt' --out runs/gradients.json")
    probe_files = []
    for f in sorted(folder.glob("*.json")):
        try:
            obj = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if obj.get("experiment") == "supervised_depth_distillation_not_JEPA":
            probe_files.append((f, obj))
    if probe_files:
        pf, result = st.selectbox("Depth probe", probe_files, format_func=lambda x: x[0].name)
        st.json(result)
        st.caption("Depth prediction is supervised distillation, NOT a JEPA model of action consequences.")
    else:
        st.info("Collect >=3 real prompts, then run 'salm probe --input runs --source 7 --target 11'.")

    intervention_files = []
    for f in sorted(folder.glob("*.json")):
        try:
            result = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if result.get("experiment") == "module_ablation":
            intervention_files.append((f, result))
    if intervention_files:
        st.divider()
        st.subheader("Causal module interventions")
        selection = st.selectbox("Intervention report", intervention_files,
                                 format_func=lambda x: x[0].name)
        result = selection[1]
        st.caption("Next-token KL(base || intervened). Larger KL means a stronger local effect, not necessarily higher importance or better quality.")
        rows = pd.DataFrame(result["rows"])
        chart(px.bar(rows, x="layer", y="kl_base_to_modified",
                     title="Change in next-token distribution when the module output is scaled"))
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Run 'salm intervene --all --module ffn --scale 0' to find sensitive blocks.")

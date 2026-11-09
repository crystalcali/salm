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
from salm.metrics import centered_cka, effective_rank, layer_rows, token_norms, token_similarity, token_cosine, residual_interference_rows

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
from salm.capture_info import choose_capture
default_index = choose_capture(files)
selected = st.sidebar.selectbox(
    "Capture", files, index=default_index,
    format_func=lambda x: x.name + (" [demo]" if x.stem == "demo" else ""),
)
st.sidebar.caption(f"Selected file: `{selected.name}`")
data = open_capture(str(selected.resolve()), selected.stat().st_mtime)
meta = json.loads(str(data["metadata"].item()))
hidden = data["hidden"].astype(np.float32)
token_strings = data["token_strings"].astype(str).tolist()
layer_types = meta.get("architecture", {}).get("layer_types", LAYER_TYPES)
num_layers, n_tokens, dim = hidden.shape[0] - 1, hidden.shape[1], hidden.shape[2]
if meta.get("kind") == "synthetic_demo":
    st.warning("SYNTHETIC DEMO — this is not an LFM3 capture. Select newton.npz in the sidebar.")
else:
    st.success(f"REAL CAPTURE: {selected.name} | model: {meta.get('model', 'unknown')} | "
               f"{n_tokens} tokens × {dim} hidden dimensions")
st.sidebar.metric("Hidden dimensions", dim)
st.sidebar.caption(f"Source kind: {meta.get('kind','unknown')}")
st.sidebar.metric("Captured tokens", n_tokens)
st.sidebar.metric("Layer states", num_layers + 1)
st.sidebar.caption("All comparisons use the same token positions within this capture.")
t0, t1, t2, t3, t4, t5 = st.tabs([
    "Architecture", "Layer dynamics", "Token geometry", "FFN / operator",
    "Gradients & probes", "Trace-JEPA / SIGReg",
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
    a_all, b_all = hidden[source], hidden[target]
    if n_tokens < 32:
        st.warning(f"Only {n_tokens} tokens: effective rank is bounded by {n_tokens-1}, and CKA may be dominated by outliers. Use >=128 tokens for exploratory comparisons.")
    def energy_share(x):
        norms2 = np.sum(x.astype(np.float64)**2, axis=-1)
        return float(norms2[0]/max(norms2.sum(), 1e-12))
    c0, c1 = st.columns(2)
    c0.metric("First-token share of total squared L2 norm · A", f"{energy_share(a_all):.1%}")
    c1.metric("First-token share of total squared L2 norm · B", f"{energy_share(b_all):.1%}")
    omit = st.checkbox("Exclude first token from CKA, effective rank and similarity map", value=False)
    unit = st.checkbox("L2-normalize every token before CKA and rank", value=False)
    offset = 1 if omit else 0
    a, b = a_all[offset:].copy(), b_all[offset:].copy()
    if unit:
        a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
        b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    c1, c2, c3 = st.columns(3)
    c1.metric("Centered linear CKA", f"{centered_cka(a,b):.4f}")
    c2.metric("Effective rank A", f"{effective_rank(a):.1f}")
    c3.metric("Effective rank B", f"{effective_rank(b):.1f}")
    st.caption(f"Geometric analysis: {len(a)} tokens; centered rank <= {max(0,len(a)-1)}. Excluding prefix removes absolute token 0, not any special model-specific token.")
    labels = [f"{i}: {v[:13]}" for i, v in enumerate(token_strings)]
    if n_tokens >= 2 and st.button("Compute cross-layer CKA heatmap (sample up to 96 tokens)"):
        sampled = hidden[:, offset:offset+min(96, len(a)), :].copy()
        if unit:
            sampled /= np.maximum(np.linalg.norm(sampled, axis=-1, keepdims=True), 1e-12)
        gram = np.array([[centered_cka(sampled[i], sampled[j]) for j in range(num_layers+1)] for i in range(num_layers+1)])
        chart(px.imshow(gram, zmin=0, zmax=1, aspect="auto", title="Centered linear CKA by depth (0=embedding)"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(offset, n_tokens)), y=token_norms(a), name=f"State {source}"))
    fig.add_trace(go.Scatter(x=list(range(offset, n_tokens)), y=token_norms(b), name=f"State {target}"))
    fig.update_layout(title="Per-token hidden vector norms", xaxis_title="Token position")
    chart(fig)
    max_display = min(len(a), 160)
    show = st.slider("Tokens shown in similarity map", 2, max_display, min(64, max_display))
    map_layer = st.selectbox("State for token-to-token similarity", [source, target],
                              format_func=lambda x: f"State {x}")
    matrix = token_similarity(hidden[map_layer, offset:offset+show])
    chart(px.imshow(matrix, x=labels[offset:offset+show], y=labels[offset:offset+show], zmin=-1, zmax=1,
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
        if key + "_effective" in data:
            part = data[key + "_effective"][layer].astype(np.float64)
            summary[key + "_residual_contribution_rms"] = float(np.sqrt(np.mean(part ** 2)))
    if all(k in data for k in ("operator_effective", "ffn_effective")):
        reconstructed = data["operator_effective"][layer].astype(np.float64) + data["ffn_effective"][layer].astype(np.float64)
        delta = after.astype(np.float64) - before.astype(np.float64)
        summary["residual_reconstruction_relative_error"] = float(np.linalg.norm(reconstructed-delta)/max(np.linalg.norm(delta),1e-12))
        st.caption("Effective signals are captured AFTER post norms, immediately before residual addition. Minor reconstruction errors are expected from FP16 capture.")
    else:
        st.warning("Old capture: FFN and operator are RAW, not normalized residual contributions. Re-capture to compare their actual effects.")
    st.json(summary)
    frame = pd.DataFrame({"token": np.arange(n_tokens), "full_block_delta": token_norms(after - before)})
    use_effective = all(k + "_effective" in data for k in ("operator", "ffn"))
    for key in ("operator", "ffn"):
        name = key + "_effective" if use_effective else key
        if name in data:
            frame[key + ("_residual" if use_effective else "_raw")] = token_norms(data[name][layer])
    chart(px.line(frame, x="token", y=[x for x in frame.columns if x != "token"],
                  title=("Per-token residual contributions" if use_effective else "Raw module outputs (not comparable to residual delta)")))
    if not any(key in data for key in ("operator", "ffn")):
        st.info("Re-capture with --modules ffn,operator for internal module signals.")
    if use_effective:
        st.divider()
        st.subheader("How operator and FFN combine")
        interference = residual_interference_rows(
            data["operator_effective"], data["ffn_effective"], layer_types
        )
        df = pd.DataFrame(interference)
        current = interference[layer]
        met1, met2, met3 = st.columns(3)
        met1.metric("Operator–FFN cosine (block)", f'{current["operator_ffn_cosine"]:.3f}')
        met2.metric("Cancellation index (block)", f'{current["cancellation_index"]:.1%}')
        met3.metric("Tokens with opposing directions", f'{current["negative_token_fraction"]:.1%}')
        st.caption("Cosine below zero means opposing vector directions. Cancellation = 1 - ||op+ffn||/(||op||+||ffn||). These are geometric measurements, NOT estimates of unnecessary computation.")
        chart(px.line(df, x="layer", y=["operator_ffn_cosine", "mean_token_cosine"],
                      title="Alignment of post-norm operator and FFN contributions across depth"))
        chart(px.bar(df, x="layer", y="cancellation_index", color="type",
                     title="Cancellation index across all 24 blocks (0 = none)"))
        token_alignment = token_cosine(
            data["operator_effective"][layer], data["ffn_effective"][layer]
        )
        chart(px.line(
            pd.DataFrame({"token": np.arange(n_tokens), "cosine": token_alignment}),
            x="token", y="cosine", title=f"Per-token operator–FFN cosine in block {layer}"
        ))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run a new capture with --modules ffn,operator to unlock post-norm alignment diagnostics.")


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

    corpus_files = []
    for f in sorted(folder.glob("*.json")):
        try:
            obj = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if obj.get("experiment") == "corpus_module_ablation":
            corpus_files.append((f, obj))
    if corpus_files:
        st.divider()
        st.subheader("Corpus-wide module interventions")
        selection = st.selectbox("Corpus intervention report", corpus_files,
                                 format_func=lambda x: x[0].name)
        result = selection[1]
        st.caption("Scores every next-token prediction in the supplied inputs, excluding the chosen prefix. Negative NLL delta means the intervened model assigned more probability to the observed text; it does NOT guarantee better task solving.")
        st.write("Prompts:", result["prompts_scored"], "| Module:", result["module"], "| Scale:", result["scale"])
        table = pd.DataFrame(result["rows"])
        chart(px.bar(table, x="layer", y="kl_base_to_modified", color="type",
                     title="Average KL(base || modified), across all scored positions"))
        chart(px.bar(table, x="layer", y="nll_delta", color="type",
                     title="Change of next-token NLL on supplied corpus (modified - base)"))
        chart(px.line(table, x="layer", y="top1_change_rate",
                      title="Fraction of positions where the most probable token changes"))
        st.dataframe(table, use_container_width=True)
        per_prompt_rows = []
        for entry in result.get("per_prompt", []):
            for row in entry.get("rows", []):
                per_prompt_rows.append({"prompt_index": entry["prompt_index"], **row})
        if per_prompt_rows:
            by_prompt = pd.DataFrame(per_prompt_rows)
            chart(px.imshow(by_prompt.pivot(index="layer", columns="prompt_index",
                                             values="kl_base_to_modified"),
                            aspect="auto", title="Where do individual prompts disagree?"))
    else:
        st.info("Use 'salm intervene-corpus --prompts-file examples/probe_prompts.txt' to compare layers over multiple inputs.")

with t5:
    st.subheader("Trace-JEPA + SIGReg (no EMA)")
    st.caption("Displays saved experiments; this page does not launch GPU training.")
    base = Path(st.text_input("Training runs directory", value="runs"))
    histories = sorted(base.glob("**/history.jsonl")) if base.is_dir() else []
    if not histories:
        st.info("Train with: python -m salm.cli trace-train --config configs/trace_jepa_sigreg.json")
    else:
        path = st.selectbox("Experiment history", histories, format_func=lambda p: str(p.parent))
        items = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        frame = pd.DataFrame(items)
        cfgpath = path.parent / "config.json"
        if cfgpath.is_file():
            config = json.loads(cfgpath.read_text())
            st.caption(f"mode={config.get('mode')} | layers={config.get('layers')} | no EMA")
        st.metric("Completed optimizer steps", len(frame))
        if len(frame):
            losses = [k for k in ("sft", "jepa", "pred", "sigreg", "val_nll") if k in frame.columns]
            chart(px.line(frame, x="step", y=losses, title="Observed training and validation losses"))
            stats = [k for k in ("latent_var", "grad_norm", "cuda_peak_gib") if k in frame.columns]
            if stats:
                chart(px.line(frame, x="step", y=stats,
                              title="Latent variance, gradients and memory"))
            st.dataframe(frame, use_container_width=True)
        st.warning("JEPA loss reduction alone does not prove better reasoning. Evaluate external tasks.")

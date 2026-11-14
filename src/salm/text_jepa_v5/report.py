import json
from pathlib import Path


def plot_run(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    history = json.loads((folder / "metrics.json").read_text())["history"]
    if not history:
        return
    cfg = json.loads((folder / "config.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    def line(ax, key, label=None, scale=1.):
        entries = [h for h in history if key in h]
        if entries:
            ax.plot([h["tokens_seen"] / 1e6 for h in entries],
                    [h[key] * scale for h in entries], label=label or key)
    line(axes[0], "mse" if cfg["arm"] != "mlm" else "mlm_ce")
    axes[0].set_title("Training MSE / span-MLM CE (different objectives)")
    line(axes[1], "sigreg")
    axes[1].set_title("Training SIGReg")
    line(axes[2], "controller_next_weight", "SIGReg weight")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Controller")
    for source in ("general", "reasoning"):
        for mode in ("infill", "suffix"):
            prefix = f"val_{source}_{mode}_"
            line(axes[3], prefix + ("mse" if cfg["arm"] != "mlm" else "mlm_ce"), source + "/" + mode)
            line(axes[4], prefix + "retrieval", source + "/" + mode, 100)
    axes[3].set_title("Validation, by source and mask mode")
    axes[4].set_title(f"JEPA retrieval @1 (%), {cfg['val_documents']} candidates")
    axes[4].set_ylim(0, 100)
    axes[4].axhline(100 / cfg["val_documents"], color="gray", linestyle=":")
    line(axes[5], "pred_rank_position", "prediction")
    line(axes[5], "target_rank_position", "target")
    axes[5].set_title("Training rank at fixed positions (diagnostic)")
    axes[5].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("Processed input tokens (millions)")
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle(f"Text-JEPA v5 · {cfg['arm']} · seed {cfg['seed']}")
    fig.tight_layout()
    fig.savefig(folder / "training.png", dpi=140)
    plt.close(fig)


def compare(root):
    """Use only completed, matched-data runs for the comparison table."""
    root = Path(root)
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        cfg = json.loads((path.parent / "config.json").read_text())
        external_path = path.parent / "external.json"
        ext = json.loads(external_path.read_text()) if external_path.exists() else {}
        rows.append({"run": path.parent.name, "arm": cfg["arm"], "seed": cfg["seed"],
            "complete": m["complete"], "tokens": m["tokens_seen"],
            "compute_hours": m["compute_seconds"] / 3600,
            "corpus": m["corpus_fingerprint"], "parameters": m["parameters"],
            "sts_spearman": ext.get("sts_spearman"), "sst2_accuracy": ext.get("sst2_accuracy")})
    fingerprints = {r["corpus"] for r in rows}
    if len(fingerprints) > 1:
        raise ValueError("Comparison mixes different prepared corpora")
    matched_tokens = {r["tokens"] for r in rows if r["complete"]}
    if len(matched_tokens) > 1:
        raise ValueError("Completed runs have different token budgets")
    payload = {"runs": rows, "note": "JEPA MSE and span-MLM CE are not compared numerically. "
        "External validation scores select configurations; reserve final tests for later. "
        "CPU compute_hours is not GPU time; consult each environment.json."}
    (root / "comparison.json").write_text(json.dumps(payload, indent=2))
    lines = ["# Text-JEPA v5 comparison", "", payload["note"], "",
        "| Run | Complete | Tokens | Compute h | STS Spearman | SST-2 accuracy |",
        "|---|---|---:|---:|---:|---:|"]
    def score(value):
        return "—" if value is None else f"{value:.4f}"
    for r in rows:
        lines.append(f"| {r['run']} | {r['complete']} | {r['tokens']:,} | "
                     f"{r['compute_hours']:.3f} | {score(r['sts_spearman'])} | {score(r['sst2_accuracy'])} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return payload

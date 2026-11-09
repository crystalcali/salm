# SALM — Structural Analysis of Language Models

Research workbench for **LiquidAI/LFM3-1.7B-Base** on NVIDIA DGX Spark: inspect its 24-layer tri-hybrid network, capture activations, gradients and operator outputs, compare neighboring blocks, and test small depth-prediction probes.

**No model weights or credentials are stored in this repository.** All experiments run on your own machine; the Streamlit dashboard reads local capture files.

## Quick start (DGX Spark)

Before the first GPU run on Ubuntu 24.04, install the interpreter development headers needed by Triton:

```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
```

Use Python 3.12, a CUDA-enabled PyTorch build for **Linux aarch64 / GB10**, and a Transformers release supporting LFM3 (model config currently references 5.12.1). Check your local CUDA/PyTorch environment before installing another torch wheel.

```bash
git clone https://github.com/crystalcali/salm.git
cd salm
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install a compatible GPU PyTorch build first, if not already installed.
pip install -e ".[lab]"
python -m salm.cli inspect --offline
python -m salm.cli doctor
python -m salm.cli demo --out runs/demo.npz
# The dashboard prioritizes real captures over synthetic demo files.
python -m salm.cli dashboard
```

For a **real model capture** (initial download from Hugging Face requires network access):

```bash
python -m salm.cli capture --model LiquidAI/LFM3-1.7B-Base \
  --prompt "Solve x**2 - 5*x + 6 = 0 using Python." --out runs/equations.npz \
  --max-tokens 192 --modules ffn,operator
python -m salm.cli gradient --prompt "Find the roots of x**2-5*x+6." --out runs/gradients.json
# Longer experiments: --prompt-file keeps multi-line context intact.
python -m salm.cli capture --prompt-file examples/newton_long.txt \
  --modules ffn,operator --max-tokens 384 --out runs/newton_long.npz
python -m salm.cli dashboard --host 127.0.0.1 --port 8501
```

From a workstation connected to Spark by SSH:

```bash
ssh -L 8501:127.0.0.1:8501 USER@SPARK_HOST
```

Open **http://localhost:8501** on the workstation. Do not expose Streamlit publicly without authentication.

### Experiments

```bash
# Create several captures with different prompts before training probes.
python -m salm.cli probe --input runs --source 7 --target 11 --epochs 100 \
  --out runs/depth_probe.json
```

Depth probes are **layer-to-layer distillation diagnostics**, *not* a validated JEPA world model. Training and validation are split by capture file to avoid leaking tokens from the same prompt.

Batch capture and causal ablation (load the model once for a prompt file, then analyze module influence):

```bash
printf '%s\n' 'Solve x**2-5*x+6=0' 'Find a Newton iteration for cos(x)-x=0' 'Write a Python bisection solver' > prompts.txt
python -m salm.cli batch --prompts-file prompts.txt --out-dir runs --max-tokens 128
python -m salm.cli intervene --all --module ffn --scale 0 --prompt "Solve x**2-5*x+6=0" --out runs/ffn_sweep.json
python -m salm.cli probe --input runs --source 7 --target 11 --epochs 100
```

The intervention compares the original next-token distribution with the distribution when a chosen module output is multiplied by a scalar. Large KL indicates sensitivity, **not proof of causal importance for solving a full task**. It never changes stored weights.

## What the dashboard shows

- **Architecture:** 13 convolution, 5 sliding-window attention and 6 full-attention blocks; parameter and KV-cache estimates.
- **Activation geometry:** per-token norms, token×token similarity, effective rank, cross-layer centered-kernel alignment (CKA).
- **Block behavior:** residual changes, raw signals and POST-normalization operator/FFN contributions (new captures). Old captures lack the normalized values.
- **Short-context bias checks:** first-token energy share, excluding token 0, unit-normalized CKA, and depth-by-depth CKA matrix.
- **Gradients:** layerwise gradient sensitivity for causal LM loss.
- **Probes:** compare small depth predictors to mean and identity baselines.

Capture files use compressed NumPy NPZ with JSON metadata, no Python pickle. Captured arrays are stored in float16 on CPU to bound GPU memory. Start with 64–256 tokens; full 256K activations are intentionally not collected.

## Model facts and limitations

Architecture defaults match the [official LFM3-1.7B-Base config](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/main/config.json). The model itself is under Liquid AI's **LFM Open License v1.0**; check its license before redistribution or deployment. This repo contains only our research code.

The hooks target the Hugging Face LFM3 implementation, not Ollama's llama.cpp runner. Captures require a PyTorch Transformers checkpoint. GPU profiling, actual attention weight matrices, GNN/JEPA fine-tuning and full-context streaming compression are future experiments; the current package does not claim to implement them.

See [research notes](docs/architecture.md), [contribution guidance](AGENTS.md), and the [official model card](https://huggingface.co/LiquidAI/LFM3-1.7B-Base).


## Confirm you are viewing real activations

A successful `capture` writes `runs/newton.npz` but **does not automatically change an existing Streamlit selection**. The app now defaults to the most recently modified real capture, instead of `demo.npz`, and clearly labels the current file and tensor dimensions.

```bash
python -m salm.cli inspect-file --path runs/newton.npz
```

Look for `kind: activation_capture`, `hidden_shape: [25, T, 2048]` and the original `prompt`. If the app says `SYNTHETIC DEMO`, switch the sidebar Capture selector to `newton.npz`; its sample shape is `[25, 48, 64]`. Use a full browser refresh or restart Streamlit after pulling updates.

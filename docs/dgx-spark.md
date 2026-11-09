# Running SALM on NVIDIA DGX Spark

The GPU is the GB10 Grace Blackwell with shared LPDDR memory. Its OS stack uses ARM64/aarch64 and NVIDIA CUDA; x86-only PyTorch wheels are not appropriate.

## Triton Python.h failure

If the first capture prints `triton/backends/nvidia/driver.c: fatal error: Python.h: No such file or directory`, the model has loaded but Triton's C helper cannot compile. This is a **host OS development-header problem**, not a model or CUDA kernel problem.

```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
ls -l /usr/include/python3.12/Python.h
python -m salm.cli doctor
```

Do not reinstall PyTorch or re-download model weights for this error. The headers are installed at the OS level, not via pip in the virtual environment.

## Installation

Use a clean **Python 3.12** environment:

\`\`\`bash
git clone https://github.com/crystalcali/salm.git ~/work/salm
cd ~/work/salm
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch  # check this is a CUDA-capable aarch64 build
python -m pip install -e ".[lab]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m salm.cli inspect --offline
python -m salm.cli demo --out runs/demo.npz
\`\`\`

PyTorch on ARM64 is version-sensitive. If the above torch build reports no CUDA, install a compatible NVIDIA/DGX Spark PyTorch wheel or use an NVIDIA CUDA PyTorch container. Do not assume that a CUDA 12/x86 image works on GB10.

## First real capture

\`\`\`bash
python -m salm.cli capture --prompt "Implement a Newton solver and explain the stopping criterion" \\
 --model LiquidAI/LFM3-1.7B-Base --max-tokens 128 \\
 --modules ffn,operator --out runs/newton.npz
python -m salm.cli dashboard --host 127.0.0.1 --port 8501
\`\`\`

In another terminal on your Windows/Linux workstation:

\`\`\`bash
ssh -N -L 8501:127.0.0.1:8501 USER@SPARK_HOST
\`\`\`

Visit localhost:8501 in a browser. SSH tunneling is preferred to opening port 8501 to the network.

## Common problems

- **CUDA unavailable**: wrong torch wheel; verify torch and driver before loading weights.
- **Cannot download model**: access Hugging Face or pre-cache weights on Spark. Ollama quantized models cannot be instrumented by these PyTorch hooks.
- **Trust remote code**: only load model IDs you trust; HF LFM3 is the intended target.
- **Out of memory**: reduce --max-tokens, avoid capturing very long sequences or running training probes concurrently with other large LLM servers.
- **Dependency mismatch**: the official checkpoint config currently declares Transformers 5.12.1. Install a version supporting LFM3's current modeling code.
- **Streamlit blank**: run the synthetic demo first to distinguish UI issues from GPU/model issues.

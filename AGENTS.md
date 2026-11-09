# Agent instructions for SALM

- Goal: reproducible instrumentation of **HF LFM3-1.7B**, not just text generation.
- Keep all model credentials, tokens, weights and outputs out of Git. Never add PATs, HF tokens or private prompts.
- Model implementation is loaded from the official Hugging Face repository and may require trust_remote_code; do not execute untrusted model IDs.
- Keep architecture numbers checked against the official model config, not hard-coded assumptions in instrumentation.
- Avoid forward hooks that mutate model outputs **except explicitly isolated and documented research interventions**, including experimental Memory-LoRA and Decision-State JEPA residual injections; test causality, cleanup and inference effects.
- Capture the smallest useful input (64-256 tokens) and detach/copy arrays to CPU immediately.
- Treat all JS/Streamlit graph layouts as offline displays: no automatic network access or external API calls.
- Always separate **depth distillation** (same input, different model layers) from true predictive JEPA (conditioned on action/changed observation).
- Add tests for new analysis functions, normalization, and non-leakage of train/evaluation splits.
- Prefer Python 3.12 on DGX Spark ARM64 and validate CUDA builds before fine-tuning.
- Project folders: src/salm (code), tests (offline unit tests), docs (methodology), runs (ignored generated artifacts).

- Agentic Decision-State JEPA is exactly a two-arm SFT versus JEPA experiment in `experiments/decision_state_jepa/`; same data, source model, context and Harbor evaluation harness for both.
- Terminal-Bench 2.1 is strictly held out for official verifier evaluation. Never train on benchmark tasks, oracle solutions, verifier scripts, canaries or leaderboard feedback.

# Local Memory-LoRA experiment instructions

- This experiment is self-contained within `experiments/memory_lora_lab`; avoid changing core SALM modules unless separately requested.
- Primary objective is **token cross-entropy**, not JEPA prediction loss.
- Memory encoder may consume only past information. Targets must be next assistant-authored step/message, never future tool results.
- Frozen history LFM3 encoder is used under `no_grad()`; memory projector/reader and LFM3 LoRA receive CE gradients.
- Preserve distinct experiments: `recent_only`, `memory`, `memory_sigreg`. SIGReg with batch size 2 is noisy and should not be considered validated.
- Always evaluate memory-on and memory-zero on the exact same held-out tasks; shuffled-memory ablation is a stronger sanity check.
- Do not commit private traces, credentials, checkpoints, or outputs to Git.
- Never start alongside VLLM if unified memory is near full. Previous system-wide OOM terminated both VLLM and Python.
- Do not claim a model improvement from decreasing train CE alone. Validate on independent tasks and external benchmarks.

# SALM Trace-JEPA module instructions

- Study exactly one JEPA variant: prefix-to-suffix prediction with SIGReg;
  shared trainable online/target weights, no EMA and no stop-gradient.
- SFT control is allowed only to compare effects at the same data and LoRA settings.
- Do not compress model, remove FFNs, alter KV-cache, or add GNN/MoE.
- Keep task-level train/validation split and verify original prompt disjointness.
- Never feed target tokens into predictor input; only causal prefix states are allowed.
- Do not detach hook activations before auxiliary backpropagation.
- Compare against matched-compute baselines; loss-only claims are insufficient.
- Do not upload weights, credentials, private data, or checkpoints to GitHub.
- Record model revision, seed, data provenance and GPU memory in run logs.
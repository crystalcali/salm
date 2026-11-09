# SALM research methodology

## Original LFM3

Official [configuration](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/main/config.json) and [model code](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/main/modeling_lfm3.py).

- 24 decoder blocks, hidden 2048, SwiGLU intermediate 7680, 128K vocab.
- Operator types in order: C C F C S C F C S C F C S C F C S C F C S C F C.
- 13 gated short convolution blocks (kernel 7), 5 sliding attention blocks (window 256), 6 global attention blocks.
- 16 query heads, 4 key/value heads, 128 dimensions/head, tied token embedding/head.
- A block applies operator norm -> operator -> optional post-operator norm -> residual add, followed by FFN norm -> FFN -> optional post-FFN norm -> residual add.
- Model uses BF16 weights. The number of parameters and cache estimates in the CLI are **approximations**, not a profiler or runtime allocation trace.

## Capture definitions

- **hidden[0]**: first block input (input embeddings / residual state).
- **hidden[l+1]**: output of zero-indexed decoder block l.
- **operator[l]**: raw output of either conv or self-attention module (before optional post norm).
- **ffn[l]**: raw feed_forward module output (before optional post norm).
- **gradient RMS**: gradient of scalar next-token cross-entropy against each decoder block output, with *model weights frozen*; not gradient of parameters.
- **CKA**: centered linear kernel alignment between token-by-feature matrices, same positions.
- **Effective rank**: exp(Shannon entropy) of centered singular energy spectrum. The maximum meaningful rank is number of sampled tokens minus one.

## Probe protocol

Do not train and validate on different tokens from the **same prompt**. Use multiple independent prompt files. The built-in depth probe learns a dense mapping from block l to a deeper block k, with a supervised MSE on normalized hidden values.

This is a useful approximation/distillation test, **not action-conditioned JEPA**. True JEPA needs an encoder of state s_t, a conditioning variable (action, mask or context), and a target encoder of a different observation s_(t+1) or target representation. A regularized predictive loss must avoid collapsed representations. Directly comparing adjacent hidden vectors can be misleading due to changes of basis.

## Next experiment plan

1. Capture 20–100 semantically diverse prompts, compare early/middle/late hidden geometry.
2. Analyze differences between conv, sliding-attention and full-attention transitions, and use gradients to find sensitivity hotspots.
3. Test depth predictors (same input, deeper layer) against identity/mean and equal-sized MLP control.
4. Build true action-conditioned transition corpus with (state, action, observed next state), train target-projected JEPA with a frozen backbone.
5. Only after validating causal predictive gains, add zero-init gated residual adapters to selected layers.
6. Record quality, matched train FLOPs, inference latency, peak device memory, and train/validation leakage controls.

## Reproducibility & ethics

Do not commit raw private prompts, Hugging Face API tokens or weights. The generated runs directory is gitignored. If the default model implementation changes, rerun tests and validate the hook names. This project does not alter Liquid AI model weights.

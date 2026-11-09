# Trace-JEPA + SIGReg (NO EMA)

## Scientific scope

Exactly one JEPA method is implemented: shared-weight causal prefix-to-suffix
latent prediction with SIGReg (sliced Epps-Pulley), no EMA teacher and
no stop-gradient. The SFT mode is a scientific control, not another JEPA variant.
There is no model compression or FFN replacement.

Original reference: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md

The adaptation to textual reasoning traces is an experimental hypothesis:
published LeJEPA image results do not prove reasoning improvements.

## Causal forward path

Let the full task and solution tokens be a concatenation C + T, where C
contains the question and a random prefix of solution steps, and T contains
the unseen suffix. For each chosen LFM3 layer l:

  h_C^l = LFM3(C+T)[last position of C at layer l]
  h_T^l = LFM3(T)[last non-pad position of T at layer l]
  z_C = P_l(h_C^l)
  z_T = P_l(h_T^l)
  z_pred = F_l(z_C)
  L_J = MSE(z_pred,z_T) + lambda_S*(SIGReg(z_C)+SIGReg(z_T))/2
  L_total = L_SFT + lambda_J*mean_l(L_J)

The first forward passes full text to compute teacher-forced SFT, but extracts
the prefix state only at the causal cut. Future tokens cannot flow backward
in attention/conv. The second forward encodes suffix ONLY, using the SAME model
parameters. Both views receive gradients, with no detach/EMA. SIGReg acts
on projected latent states rather than directly imposing Gaussian structure
on the base LFM3 residual stream.

## Running on NVIDIA DGX Spark

Use the existing CUDA-enabled Python 3.12 environment verified on the device.
Do not replace the working PyTorch just to add dependencies.

    cd /home/userdgx1/work/salm/salm
    git pull --ff-only
    pip install peft datasets
    python -m salm.cli doctor
    python -m salm.cli trace-prepare --source arithmetic --tasks 128
    python -m salm.cli trace-train --config configs/trace_jepa_sigreg.json
    python -m salm.cli trace-train --config configs/sft_control.json
    python -m salm.cli dashboard

To train from public verified math traces, stream an OpenR1 subset first:

    python -m salm.cli trace-prepare --source openr1 --tasks 1000 --val-tasks 100

Only generations marked correctness_math_verify=True are selected.
This validates the final mathematical answer, NOT every intermediate step.
No full dataset download is required initially.

## Limits

- The default arithmetic dataset is smoke-only, not a reasoning benchmark.
- For valid SIGReg estimates, use at least several independent tasks per batch;
  batch sizes below 8 may give noisy statistics.
- For a fair causal experiment, train and validation must have disjoint task_id
  and disjoint original prompts, before generating random cuts.
- In v0.1, gradient_checkpointing is explicitly disabled until hook
  recomputation with LFM3 is validated. Use batch_size and max_length to
  manage memory on Spark.
- Both branches backpropagate into LoRA; they are not an EMA teacher.
- Validation currently provides teacher-forced held-out NLL. A separate
  verified external benchmark is needed before claiming ability gains.
- Real GPU experiments cannot be claimed completed until SALM logs and
  checkpoints are produced on DGX Spark.
- Never commit weights, run logs with private prompts, or API tokens.

## Saved artifacts

Under the configured gitignored output directory:
config.json, history.jsonl, step_000010.pt, LoRA adapter_or_model/,
and trace_jepa_heads.pt. The dashboard reads history.jsonl lazily.
Resume with trace-train --config ... --resume <trusted-local-checkpoint>.

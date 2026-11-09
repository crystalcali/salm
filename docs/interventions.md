# Module intervention protocol

Use the same prompt and model weights twice:

1. Baseline forward pass of the original model.
2. A single module forward hook scales its output by factor s (0 = ablate, 1 = original).
3. Hook is removed, before the next layer is tested.
4. Record KL divergence D_KL(p_base || p_modified) on the next-token distribution.

Run:

\`\`\`bash
python -m salm.cli intervene --all --module ffn --scale 0 \\
 --prompt "Explain why Newton's method might diverge." --out runs/ffn_sweep.json
python -m salm.cli intervene --all --module operator --scale 0 \\
 --prompt "Explain why Newton's method might diverge." --out runs/op_sweep.json
\`\`\`

This is a causal **local sensitivity probe**, not a task-level evaluation, saliency ground truth or proof of node importance. A module can have low instantaneous KL and still matter for multi-step generation. Module output scaling occurs before optional post-layer normalization.

Ablation is computationally expensive: full sweeps require 24 extra forwards per module type. Weight tensors are never changed, no optimizer is used, and hooks are removed after each measurement.

Next: compare sweeps across a diverse prompt set, aggregate KL with robust statistics, and correlate with actual task-quality changes (pass@1, test success, benchmark trajectories).

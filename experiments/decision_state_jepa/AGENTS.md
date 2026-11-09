# Decision-State JEPA research module instructions

- Exactly TWO training variants: `sft` ordinary LoRA and `decision_jepa` (same LoRA, same data, extra latent-state predictor, auxiliary loss and gated token conditioning). No third model or unrelated compression architecture.
- Train on verified task-level successful NVIDIA Open-SWE-Traces v2.0 agent event traces; do not claim every individual step is formally verified.
- Never use any Terminal-Bench 2.1 task, instructions, solution, tool logs, verifier, benchmark canary, or grading result for training or dataset selection. Keep the benchmark held out for official Harbor evaluation only.
- Every JEPA source state must be computed strictly before the next assistant event. Tool results and new user messages are external observations; they become context only AFTER arrival.
- Future target states may be computed for auxiliary supervision but MUST NEVER be injected into token-generation logits. Test both text and latent state non-leakage.
- Next-token CE is the main task loss and must backpropagate through the predictor and gated injection. The predictive loss is auxiliary, with no EMA or SIGReg in this two-arm experiment.
- Match seed, train tasks, context window, LoRA settings, token budgets, optimizer and terminal harness across SFT and JEPA.
- Tool-boundary sampling weights are heuristics, not validated credit-assignment causal values. No claims of reasoning gain without verified held-out Terminal-Bench scores.
- Never commit checkpoint weights, raw training traces, HF credentials, terminal outputs, benchmark solution artifacts or run logs.
- Use the existing DGX Spark ARM64 CUDA environment. Test a 10-step run before 8K or 12K long training; avoid concurrent VLLM/global OOM.
- Keep experiment artifacts in gitignored `runs/`. Update documentation/tests with changes to data transformations, hooks, inference, agent harness or evaluation.

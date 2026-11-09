# SALM Decision-State JEPA — agentic training with official Terminal-Bench 2.1 evaluation

**Exactly two experiment arms:** `sft` (ordinary LoRA supervised fine-tuning) and `decision_jepa` (the same SFT + predicted decision-state injection + auxiliary latent alignment). Both load `LiquidAI/LFM3-1.7B-Base`, keep original weights frozen and train matching LoRA modules. No EMA, no benchmark data used in training, no hidden benchmark verifier exposed to the policy.

## Method

Training units are transitions at explicit, **chronologically causal event boundaries**: user prompt, assistant reasoning, assistant tool call, tool observation, next assistant response. The assistant's next event is always the supervised target; tool/user outputs arriving after the cut are neither supervised targets nor available context.

1. Extract the hidden state at the **last prefix token** of decoder block 12 (causal input only).
2. Project to a 256-dimensional state `z_t`; predict next decision-state `z_pred = z_t + F(z_t)`.
3. Apply the **predicted** state, never the true future state, through a gated residual adapter at block 18. This directly influences next-token logits and receives gradients from token CE.
4. Compute a detached, context-conditioned target `z_{t+1}` at the end of the **next assistant event**, with the same LFM3 causal forward. Minimize a normalized Smooth L1 latent alignment loss weighted by `jepa_weight=0.2`.
5. `decision_jepa`: `L = CE(next assistant event) + 0.2 * JEPA`; `sft`: `L = CE(next assistant event)`. The source-to-target predictor is shared with the generator, not a parallel task-only predictor.

`z_pred` uses only the prefix. End-of-target activations are detached and never injected. There is only **one** differentiable model forward per training microbatch. This is **not** EMA/LeJEPA/SIGReg. It is a new experimental formulation, not a proven reasoning improvement. Inference computes the predicted state once per assistant decision, then uses it on subsequent cached token steps.

Event priority is a **sampling heuristic applied identically in both arms**: assistant tool calls 3.0, assistant after tool observation 2.5, assistant after user message 2.0, otherwise 1.0. These are *not* verified causal importance estimates, and must not be reported as such. Ground-truth trace labels can contain suboptimal individual choices even when the task ultimately succeeded.

## Data and compute

Recommended corpus: [NVIDIA Open-SWE-Traces v2.0](https://huggingface.co/datasets/nvidia/Open-SWE-Traces), `claudecode` split. Its messages distinguish agent and tool observations, and `resolved=1` filters task-level successes. By default the preparer streams **12,000 distinct training instances and 1,000 validation instances**, split by stable original issue ID. Only serialised roles, authored text and tool calls are retained; no reference patches, gold answers, resolved flags or teacher metadata enter model input. Records are written under gitignored `runs/decision_state_data/`. It does not bundle gigabytes of training records into GitHub. The data source is CC-BY-4.0 with additional source-repository licenses to respect.

Terminal-Bench 2.1 is **evaluation only**. Its tasks, solutions, verifier files and canary strings must never be used for SFT, JEPA, ranking, RL, prompting development based on gold outcomes, or tuning checkpoints. The import path enforces a best-effort source/canary exclusion; task success still requires independent contamination review.

8K token context is the main configuration, 12K is optional if 8K fits. The old Trace-JEPA ran at 4K, so both are increased. `target_tokens=768`, `batch_size=1`, `grad_accum=8`, `max_steps=2000`, `eval_every=200`. This samples up to 16,000 transition events per 2000-step run (with repeats), **not** 16,000 independent tasks. At 8K context the runtime and memory cost could be substantial on GB10; use a pilot before a full run. The 12K config is intentionally not guaranteed to fit.

## DGX Spark setup

In the existing repo and CUDA-enabled Python environment:

```bash
cd ~/work/salm/salm
git pull --ff-only origin main
source .venv/bin/activate
export PYTHONPATH="$PWD/experiments/decision_state_jepa${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q experiments/decision_state_jepa/tests
```

Use your working ARM64 PyTorch and Transformers installation. Add missing `datasets`, `peft`, `safetensors` only; do not reinstall CUDA PyTorch. Keep other large GPU processes shut down.

Prepare larger train/validation data (needs HF access, sometimes auth due to rate limits):

```bash
python -u -m decision_state.cli prepare --out-dir runs/decision_state_data \
  --split claudecode --train-limit 12000 --val-limit 1000 --scan-limit 200000
python -m decision_state.cli inspect
```

If fewer than requested valid instances exist, this command **fails visibly** and deletes partial outputs; increase scan limit if possible or lower limits. It will not pretend there are 12K examples. If your HF streaming client fails during shutdown, first inspect local file integrity rather than assuming data were completed. Do not run this preparer on the Terminal-Bench dataset.

### Pilot: both models, identical training setup

Save unique local configs to `runs/`, and leave the published full configs unchanged:

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('experiments/decision_state_jepa/configs')
for mode in ('sft','decision_jepa'):
    cfg=json.loads((p/f'{mode}_8k.json').read_text())
    cfg['max_steps']=10
    cfg['eval_every']=5
    cfg['save_every']=10
    cfg['eval_tasks']=10
    cfg['out']=f'runs/decision_state/{mode}_smoke_8k'
    path=Path(f'runs/decision_state/{mode}_smoke_8k.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg,indent=2))
    print(path)
PY

python -u -m decision_state.cli train --config runs/decision_state/sft_smoke_8k.json \
  2>&1 | tee runs/decision_state_sft_smoke.log
python -u -m decision_state.cli train --config runs/decision_state/decision_jepa_smoke_8k.json \
  2>&1 | tee runs/decision_state_jepa_smoke.log
```

The first GPU run must verify actual memory headroom and that the forward hooks work with the installed LFM3. If an OOM occurs, lower the context length equally in both configs before restarting from fresh output directories. Use distinct directories for all attempts; never silently overwrite previous checkpoints.

### Full matched run

Once the 8K pilot works, run each full config (one at a time):

```bash
python -u -m decision_state.cli train \
  --config experiments/decision_state_jepa/configs/sft_8k.json \
  2>&1 | tee runs/decision_state_sft_full.log
python -u -m decision_state.cli train \
  --config experiments/decision_state_jepa/configs/decision_jepa_8k.json \
  2>&1 | tee runs/decision_state_jepa_full.log
```

For 12K, use `configs/sft_12k.json` and `configs/decision_jepa_12k.json`, with **the same window in both arms**. The fixed validation NLL is a diagnostic, **not** Terminal-Bench performance.

## Terminal-Bench 2.1: actual official verifier

Source supplied by user: [Hugging Face `harborframework/terminal-bench-2.1`](https://huggingface.co/datasets/harborframework/terminal-bench-2.1), mirroring [the official 89-task Harbor Terminal-Bench 2.1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1). This code uses **Harbor's task containers and built-in verifier**, not teacher-forced NLL, manual judging, or extracted test answers.

Install Harbor CLI separately so the working model environment stays intact:

```bash
uv tool install harbor
harbor --help
docker info
harbor datasets list
```

The same Harbor agent class `decision_state.harbor_agent:SALMTerminalAgent` handles both arms, using the identical tool executor, prompt format, greedy sampling and per-task iteration limits. The checkpoint is selected by the environment variable `SALM_DECISION_CHECKPOINT`. The agent loads the real LoRA checkpoint in host GPU memory and uses Harbor `environment.exec` for sandbox tool actions. **No hidden verifier or reference solution is injected into model input.** Trials are graded only by Harbor after agent completion.

First confirm Harbor infrastructure independently (oracle is a smoke check **only**, not a model score):

```bash
harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 1 -n 1
```

Then run 5-task end-to-end model smoke tests using **the same five tasks** for both checkpoints:

```bash
python -m decision_state.cli benchmark \
  --checkpoint runs/decision_state/sft_8k/checkpoint_002000 --limit 5
python -m decision_state.cli benchmark \
  --checkpoint runs/decision_state/decision_jepa_8k/checkpoint_002000 --limit 5
```

To evaluate the user-provided **Hugging Face URL explicitly**, append `--hf-repo` (the mirror may require `hf auth login --add-to-git-credential`). By default the command uses the public official Harbor dataset registration, avoiding private mirror credentials.

For the complete **89-task** benchmark, compare both checkpoints with several trials per task:

```bash
python -m decision_state.cli benchmark \
  --checkpoint runs/decision_state/sft_8k/checkpoint_002000 --limit 0 --trials 3
python -m decision_state.cli benchmark \
  --checkpoint runs/decision_state/decision_jepa_8k/checkpoint_002000 --limit 0 --trials 3
```

`--limit 0` disables task-count limiting; `--trials 3` repeats each task three times. Harbor writes reward and success metrics to `~/.harbor/jobs/<job>/result.json` and per-trial results. The runner also creates a local benchmark manifest under `runs/decision_state/benchmark/`, including checkpoint weight hashes and the Harbor command. It does **not** upload results to the leaderboard. For stricter reporting, use 5+ trials per task and report confidence intervals and failure categories.

**DGX Spark limitations:** tasks may require Docker, large downloads, CPU/GPU capacity and network connectivity. `--n-concurrent 1` is intentional because both LFM3 inference and some tasks could compete for GB10 unified memory. The Harbor agent is a minimal research harness: text-based tool-call output may not always parse and it does not implement native tool-calling tokens from every foundation model; report these failures separately from model task failures. Actual Terminal-Bench scores are **not known until run** on Spark.

## Critical evaluation criteria

The primary endpoint is **Terminal-Bench 2.1 success rate** on exactly the same tasks and trial counts, scored by Harbor's verifier. Teacher-forced NLL and injection-on vs injection-off are secondary diagnostics. Compare paired tasks, use multiple seeds/trials, report zero scores and agent execution failures honestly. The custom harness must be fixed before comparing checkpoints; otherwise the effect of JEPA is confounded with runtime changes.

No benchmark corpus is downloaded as training data, and no claimed score comes from synthetic tests.

# Text-JEPA v5: controlled representation experiments

[Open in Colab](https://colab.research.google.com/github/crystalcali/salm/blob/experiment/text-jepa-v5-colab/notebooks/text_jepa_v5_colab.ipynb)

## Agreed decisions

| Setting | Choice |
|---|---|
| Goal | Semantic representations; later learned latent reasoning |
| Pilot / future size | ~128M / ~300M total JEPA parameters |
| Context / global batch | 1024 tokens / 128 independent documents |
| Data | 70% general texts / 30% worked solutions, by input tokens |
| Language | English pilots; EN + RU for the later large stage |
| Vocabulary | Shared EN/RU byte-level BPE, 50,000 tokens |
| Mask tasks | Infill / suffix, 50:50 in expectation; 25% contiguous hidden block |
| Arms | Constant SIGReg / smooth adaptive SIGReg / span-MLM |
| Pilot budget | 12–24 GPU hours; notebook defaults to 18 |
| Checkpoints | Private Hugging Face repository |
| Teacher EMA / pulses | Neither |

This is experimental shared-encoder Text-JEPA, not canonical LeJEPA. The historical v4
differs in context, tokenizer and corpus: v4-to-v5 changes cannot be attributed solely to
regularization. Compare fixed/adaptive v5 arms to isolate adaptation.

## Launch and continue

Select a GPU Colab runtime (G4 if available), add `HF_TOKEN` to Colab Secrets with write
access, and run the notebook in order. It creates `<HF username>/text-jepa-v5-private`
privately and rejects an existing public destination. Keep `EXPERIMENT_ID` unchanged
when continuing. A new ID starts a new comparison.

The launcher benchmarks full optimizer updates, chooses a fitting microbatch for each arm,
and fixes equal input-token budgets before training seeds 42 and 43. Two seeds provide a
pilot comparison, not precise confidence intervals. The global batch stays 128.
35% of the nominal time is reserved for preparation/evaluation/network and pilots are capped
at 200M tokens per run. Cumulative per-run wall-time caps allocate 80% of the nominal total
across training runs; checks occur between updates. Preparation, evaluation and the last
in-flight upload can exceed estimates. Monitor actual Colab consumption. No G4 throughput
has been measured before running the benchmark in the user's session.

Restart restores the saved plan, prepared data, tokenizer, model, AdamW, controller,
data cursor and Python/NumPy/CPU/CUDA RNG. LR is reconstructed from completed tokens and
the fixed configuration. Local writes are atomic; remote uploads are synchronous.
Upload errors stop training with the consistent local checkpoint. Stop/SIGTERM requests
save after the current update. Hard runtime destruction can lose progress since the last
successful remote upload. Only load checkpoints from your own trusted private repository:
optimizer/RNG state uses PyTorch pickle serialization.

Changing a scientific setting requires a new run ID. A microbatch partition may change
on resume, with possible floating-point differences, while keeping the same global loss.
Prepared arrays/tokenizer are also needed after a restart; the notebook uploads them.
Hub history retains old versions of `latest.pt`; allow for storage and upload overhead.

Local/Spark commands, using a CUDA-compatible PyTorch build:

```bash
python -m pip install -e ".[text-jepa-v5,dev]"
python -m salm.text_jepa_v5 benchmark --output runs/v5/benchmark.json
python -m salm.text_jepa_v5 prepare --data runs/v5/data --tokens 100000000
python -m salm.text_jepa_v5 train --data runs/v5/data --output runs/v5/fixed_s42 --arm fixed
python -m salm.text_jepa_v5 train --data runs/v5/data --output runs/v5/adaptive_s42 --arm adaptive
python -m salm.text_jepa_v5 train --data runs/v5/data --output runs/v5/mlm_s42 --arm mlm
python -m salm.text_jepa_v5 external --data runs/v5/data --checkpoint runs/v5/fixed_s42/latest.pt
python -m salm.text_jepa_v5 external --data runs/v5/data --checkpoint runs/v5/adaptive_s42/latest.pt
python -m salm.text_jepa_v5 external --data runs/v5/data --checkpoint runs/v5/mlm_s42/latest.pt
python -m salm.text_jepa_v5 compare --root runs/v5
```

CLI checkpoints are local by default. Add `--repo-id owner/private-repo --run-id pilot/fixed_s42`
to upload, and `--resume-hub` to download that run first. Credentials use the HF SDK's
environment/login mechanism. The Colab launcher handles private corpus storage as well.

## Data and tokenizer

- English: [FineWeb-Edu, sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
- Solutions: [OpenR1-Math-220k, default](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k),
  one generation with both `correctness_math_verify=True` and `is_reasoning_complete=True`.
- Russian vocabulary: [FineWeb2, rus_Cyrl](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2).

Dataset revisions are pinned in the manifest. BPE is fitted on 10,000 training-part
documents per language; model-training windows remain English during pilots. There are
no pretrained model weights. Russian vocabulary alone does not confer Russian semantics.
The large-stage language proportion remains undecided; `prepare --ru-fraction` can later
add Russian to the GENERAL portion (overall RU share is 0.7 times that fraction).

Normalized problem hashes group all solutions of a problem. Normalized document hashes
group web documents. Hash buckets split train/validation before windowing; exact repeated
texts are excluded across prepared sources. Near-duplicates and public-web benchmark
contamination are not fully ruled out. Metadata records hashes and offsets privately.

Each unique document/family supplies one randomly selected full-length window. Tokenization
does not stop at 4096 tokens. Arrays are uint32 token IDs with SHA256 hashes. A stateless 7:3
source schedule and seeded per-source permutations make an integer cursor sufficient for
exact continuation. The ratio is quantized within a 128-document update and exact over
10-document cycles. Repeated documents in one global batch are rejected.

This avoids the [shuffled-stream resume limitation](https://huggingface.co/docs/datasets/en/stream#save-a-dataset-checkpoint-and-resume-iteration)
where shuffle buffers are refilled. Preparation itself restarts if interrupted before its
first complete manifest/upload; local vocabulary and pinned revisions are reused. Prepare
a larger corpus for large training; do not silently replay a small pilot cache for billions
of tokens. Large-stage corpus sharding/scaling needs separate measurement.

## Architecture

| Component | Base | Large candidate |
|---|---:|---:|
| Encoder layers | 12 | 19 |
| Width / heads | 768 / 12 | 1024 / 16 |
| Encoder parameters | 124,242,432 | 291,578,880 |
| Total JEPA | 128,378,624 | 298,404,352 |
| Total span-MLM | 124,884,560 | 292,680,528 |
| Latent dimension | 256 | 256 |

Counts are computed from the implemented modules. Backbones of the same seed initialize
identically across arms; cloned layers initialize independently. Heads differ, so total
parameters and compute are reported separately. Dropout is zero for deterministic GradCache.
The large preset is a size template, not an automatic training launch or a selected winner.

In infill, context consists of text on both sides. In suffix mode, context is only the first
768 tokens and the target is the final 256. Target encoding is a separate pass over the
hidden block itself, never the full document or a later continuation. The block may contain
the answer being predicted. Both branches receive gradients: no teacher EMA or target
stop-gradient. GradCache's no-grad collection is followed by gradient replay through both.

The span-MLM baseline masks every target token at the same block locations and uses a tied
vocabulary decoder. It is not a reproduction of BERT's 80/10/10 corruption. Equal input tokens
do not imply equal compute or forward passes.

## Smooth controller

`loss = (1 - lambda) * MSE + lambda * mean(SIGReg_pred, SIGReg_target)`.
Fixed lambda is 0.25, matching the completed v4 500M run. Experimental adaptive defaults:
bounds 0.20–0.45; rank target fraction 0.20; diagnostic EMA 0.9; change at most 0.005 per
diagnostic; diagnostics every 25 updates after 5M tokens. Scalar diagnostic smoothing is
not a model teacher EMA. These thresholds are hypotheses, not established optimal values.

Center across documents at each sampled fixed position. Compute covariance entropy rank
separately for predictions and targets; use the lower normalized value. With B=128,D=256,
the empirical centered rank limit is 127, so the initial target is ~25.4. Low rank smoothly
increases lambda; recovery relaxes it to base 0.25. Validation/retrieval never drive it.
Also log cross-document variance and pooled residual rank after subtracting position means.

First pilots do not introduce covariance-loss changes, predictor self-attention, pulses or
learned halting. Those effects need separate comparisons.

## Evaluation

Internal validation separates general/reasoning × infill/suffix. JEPA reports MSE,
same-position retrieval over 128 candidates (chance 0.78125%), rolled-context controls,
conditional ranks and variances. v4's 64-candidate percentage is not directly comparable.
Rolled-context scores below chance can reflect the deliberately wrong permutation.

Common frozen-backbone tests use mean pooling, L2 normalization and fixed dataset revisions:

- [STS-B](https://huggingface.co/datasets/sentence-transformers/stsb) validation Spearman
  correlation of cosine similarity.
- [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) validation accuracy of a logistic
  probe trained on the same 4000 training examples, fixed C=1. StandardScaler fits probe
  training data only. The backbone is frozen.

Validation selects configurations. STS test is separate and opt-in through
`external --final-sts-test`; SST-2 here always reports labeled validation. Report per-seed
scores, equal-token curves, parameter counts, optimizer steps and measured time. JEPA MSE
and MLM CE are different objectives and are not numerically ranked against each other.
High rank or internal retrieval alone does not establish semantic or reasoning ability.
There is no trained LLM bridge yet and no automatic promotion to large training.

## Future stage: learned latent reasoning and exits

The user's requirement is that these behaviors be **learned**:

1. Predict the next coherent text block's latent representation.
2. Attend over previous blocks and recurrent predicted states.
3. Train decisions `continue / write / tool / finish`: continue latent processing, emit
   text through an LLM/decoder, request a tool action, or finish the response.
4. Start from labeled trajectories, then evaluate/optimize whole-task correctness and
   the cost of extra steps/tools. Text/tool responses can become new observations.
5. Compare learned depth/exits with fixed-depth and text-only baselines. Next-text latent
   prediction alone does not demonstrate useful internal reasoning.

Current context cross-attention is already present. Block-to-block attention, recurrence,
LLM bridge, tool policy and learned stopping are future work, not implemented capabilities
of this pilot. A rank/MSE stopping threshold does not replace learned reasoning termination.

## Verification

```bash
PYTHONPATH=src pytest -q tests/text_jepa_v5
ruff check src/salm/text_jepa_v5 tests/text_jepa_v5 scripts/build_text_jepa_v5_notebook.py
```

Offline tests cover direct/global vs cached gradients, target gradients, mask isolation,
positional-collapse diagnostics, smooth-controller bounds, corpus hashes/splits, bilingual
tokenizer round-trip, private-destination checks and identical uninterrupted/resumed CPU
training for all three arms. G4 throughput, real corpus preparation and authenticated Hub
transfer still need validation in the actual Colab session.

# Kev's decision recipe on EmbeddingGemma 2 (pass mark only)

Does upstream Kev's decision recipe ([jaredpalmer/kev](https://github.com/jaredpalmer/kev) @ `2074cc4`) work on
**EmbeddingGemma 2's text encoder** (`google/embeddinggemma-2` @ `914f7f8`, ~137M transformer + 134M embedding
parameters)? That recipe is a pointer head over a joint state + question sequence, trained on `decision-v7`. The model
was trained locally on an M4 Max (128 GB) under a memory watchdog. The pass mark below was set before training. This is
not a recipe: there is no ONNX, no kev.js bundle and nothing published, and CI never runs it.

**Verdict: stop.** The arm picked on development data reaches **0.476** transfer-v4 development accuracy (calibrated
Brier 0.608). The pass mark is 0.62 and the stop line is 0.60. The second arm (causal rows) reaches 0.503, which also
misses. Kev-0.8B stays the default.

## Pass mark (fixed before training)

- Recipe: `decision-v7` train, 2 epochs, LoRA r16 / α32, lr 1e-4 (one-cycle), `--p_none_pair 0.25`, batch 1 × accum 8
  (Kev-0.8B's stage 1 runs batch 8 × accum 1), seed 0. The temperature is fitted on `decision-v7` calibration, as
  upstream's `kev.experiment.score_trial` does.
- Primary metric: transfer-v4 **development** accuracy, plus Brier raw (T = 1) and calibrated, scored by upstream's
  own `score_trial` / `evaluate_records` / `kev.metrics`. The verdict is pass at ≥ 0.62, stop below 0.60, and
  inconclusive in between.
- At most one extra arm, for attention mode. The arm is picked on `decision-v7` development (upstream's study
  objective, −macro NLL), not on transfer. No other hyperparameter search was run.
- No locked or test partition was read. `kev.suite.load_split` refuses `test` unless it is passed `allow_test`, and no
  code here passes it.

## Results

transfer-v4 development (656 clean questions; Brier is summed over options):

| Model | acc | Brier raw | Brier cal. | T | source |
|---|---|---|---|---|---|
| **EG2 + Kev recipe, bidirectional (picked)** | **0.476** | 0.619 | **0.608** | 1.32 | `runs/bidirectional` |
| EG2 + Kev recipe, causal rows | 0.503 | 0.574 | 0.570 | 1.19 | `runs/causal` |
| Kev-0.6B (released, seed 2) | 0.620 | 0.536 | 0.489 | 1.93 | upstream `runs/v7-06b/02-trial-2/result.json`; card `docs/model-cards/kev-0.6b-qwen3.md` (0.620) |
| Kev-0.6B, seeds 0 / 1 | 0.613 / 0.605 | 0.555 / 0.533 | 0.507 / 0.503 | | upstream `runs/v7-06b/0{0,1}-trial-*` |
| Kev-0.8B stage 1 (this recipe), seeds 0 / 1 / 2 | 0.622 / 0.634 / 0.643 | 0.508 / 0.516 / 0.513 | 0.467 / 0.456 / 0.447 | 1.80–2.14 | upstream `runs/q35-08b/0{0,1,2}-trial-*/result.json` |
| Kev-0.8B shipped (3 stages) | 0.648 | 0.481 | 0.430 | 2.35* | card `docs/model-cards/kev-0.8b.md` (0.648 / 0.430); `runs/r15-08b/00-trial-0` |
| Kev-4B shipped | 0.817 | 0.269 | 0.243 | | card `docs/model-cards/kev-4b.md` (0.817 / 0.243); `runs/r10-skills/00-trial-0` |

\* The shipped temperature. The calibrated column of the upstream rows is each trial's own `calibrated_clean`.

kev.js's 300-record subset (`load_dev(suite="evals/v4/transfer-v4", n=300, seed=1)`; selection sha256 `89ae1231…`, as
in PR #25):

| Model | acc | Brier raw | Brier cal. |
|---|---|---|---|
| EG2 + Kev recipe, bidirectional | 0.450 | 0.639 | 0.623 |
| EG2 + Kev recipe, causal rows | 0.550 | 0.559 | 0.557 |
| mrm8488/embeddinggemma-2-ft-typed-decision (PR #25) | 0.457 | | 0.628 |
| Kev-0.8B (kev.js fixture, shipped T) | 0.640 | | 0.491 |
| Kev-4B (kev.js fixture, shipped T) | 0.807 | | 0.281 |

decision-v7 development (1,264 clean questions; in distribution):

| Model | acc | Brier raw / cal. | objective (−macro NLL) |
|---|---|---|---|
| EG2 bidirectional | 0.685 | 0.393 / 0.388 | −0.733 |
| EG2 causal | 0.647 | 0.425 / 0.423 | −0.764 |
| Kev-0.6B (seed 2) | 0.801 | 0.297 / 0.280 | |
| Kev-0.8B stage 1 (seed 2) | 0.829 | 0.264 / 0.243 | |

Per-source transfer-v4 development accuracy. Every column comes from the same `score_trial` output
(`transfer.tasks`, n-weighted); "policy" is the held-out composition and contrastive structures, 176 questions:

| Source (n) | EG2 bidir. | EG2 causal | Kev-0.6B (s2) | Kev-0.8B stage 1 (s2) | Kev-0.8B shipped | Kev-4B shipped |
|---|---|---|---|---|---|---|
| QNLI (80) | 0.64 | 0.64 | 0.85 | 0.83 | 0.81 | 0.91 |
| SciQ (80) | 0.55 | 0.65 | 0.93 | 0.90 | 0.90 | 0.98 |
| TweetEval-offensive (80) | 0.60 | 0.53 | 0.69 | 0.63 | 0.68 | 0.75 |
| PAWS (80) | 0.41 | 0.55 | 0.59 | 0.59 | 0.60 | 0.79 |
| MMLU (80) | 0.25 | 0.20 | 0.50 | 0.41 | 0.43 | 0.73 |
| Emotion (80) | 0.31 | 0.41 | 0.49 | 0.54 | 0.50 | 0.55 |
| policy (176) | 0.52 | 0.52 | 0.48 | 0.63 | 0.64 | 0.91 |

The Kev columns are upstream `runs/v7-06b/02-trial-2`, `runs/q35-08b/02-trial-2`, `runs/r15-08b/00-trial-0` and
`runs/r10-skills/00-trial-0`. `runs/summary.json` also has the 300-record subset per source for EG2 and the kev.js
fixtures.

What else the trials show:
- On held-out policy structures, both-siblings-correct is 0.0 on both arms (Kev-0.8B stage 1 gets 0.375). Choice
  permutation flips 18 % (bidirectional) and 15 % (causal) of argmaxes; Kev-0.6B flips 7 % and Kev-0.8B 2 %. With a
  none option present, accuracy is 0.53 / 0.42, against about 0.80 for Kev.
- In distribution (bidirectional arm), MNLI is 0.43 (Kev-0.6B 0.78), BoolQ 0.63 (0.69) and banking77 0.60 (0.85). On
  topic and sentiment sources EG2 is close to Kev (dbpedia 0.96, imdb 0.94).
- Gradient norms before clipping average 5.5–7 and exceed the clip at 1.0 on 84–94 % of epoch-1 steps and on every
  epoch-2 step. The training loss ends at 0.75 (bidirectional) and 0.84 (causal), averaged over the last 600 steps.
- Isolation and packing checks pass (max Δ 0.0): the rows are independent by construction.

**Why it stops.** The recipe fixes cross-attention but not capacity. EG2's transfer accuracy is below Kev-0.6B on
every public source; it is at chance on MMLU (0.20–0.25, four options) and far below Kev on Emotion. Only the policy
structures are level with Kev-0.6B, and Kev-0.6B itself is near chance there. Even in distribution it trails Kev-0.6B by 12 points,
where Kev-0.6B's card already calls capacity the bottleneck. On the 300-record subset, the bidirectional arm equals
the typed-decision fine-tune (0.450 vs 0.457). The causal arm is better on transfer (+2.7 pp full, +10 pp on the 300)
but worse in distribution. Development data picks the bidirectional arm, and neither arm is near 0.60.

## Setup

- **Upstream**: a pinned checkout in `.upstream/kev` (`./run.sh setup`; gitignored, never modified). `adapt.check_upstream()`
  refuses any other commit or a dirty tree. The training partition is fetched by upstream's `load_split` from
  `jaredpalmer/kev-suites` and checked against the manifest's sha256.
- **Packages** (`requirements.txt`): upstream's `uv.lock` versions on Python 3.13 (torch 2.8.0, peft 0.21.0, …), except
  transformers 5.19.0. That is the first release with `embedding_gemma2`; upstream allows `>=5.17,<6` and locks 5.17.0.
- **Adaptation** (`kev_eg2/adapt.py`, about 200 lines, applied at import time by rebinding `kev.model.DecisionModel`,
  `SPECIAL` and `user_tokens`):
  1. *Backbone*: `EmbeddingGemma2Model` with `vision_config` / `audio_config` set to None (the towers are never
     built). Only `language_model` is kept, and `embedding_projection` (512→768, per token) is replaced by the
     identity. The head reads the normed 512-d last hidden state, before pooling and projection.
  2. *Attention*: the model is bidirectional (`is_causal = False`, bidirectional masks, ±512 sliding window). A 4D mask
     passed to it goes to *both* layer types unchanged (`masking_utils._preprocess_mask_arguments`), which would drop the
     sliding window, and there is no KV cache. So every question runs as upstream's **row form** (state + that
     question's branch, `kev.model.rows_of`), wrapped in `<bos> … <eos>`. The arms: `bidirectional` uses the model's
     own masks from a 2D padding mask; `causal` builds per-layer-type causal masks with the sliding window kept.
  3. *Delimiters*: Qwen's `<|fim_prefix|>`… become `<unused0..4>` (ids 6–10, which no Gemma template emits). Their
     embedding rows are trained (`--special_embeddings 1`, peft `trainable_token_indices`). User text gets upstream's
     `<|x|>` → `<¦x¦>` rewrite, and then every one of Gemma's 24 added-token strings is broken with U+2060.
     `runs/checks.json`: 0 forged ids after escaping, 20 without it.
  4. *LoRA*: upstream's "all" targets (q/k/v/o, gate/up/down) match Gemma 4's names. The per-layer-input (PLE)
     projections stay frozen. 5.3M trainable parameters.
  5. *Precision*: fp32 weights and fp32 compute. fp16 is refused, and any non-finite hidden state fails the run, on
     top of upstream's non-finite loss and gradient checks. On MPS the SDPA path returned NaN on padded rows (found by
     the check), so the model runs eager attention there, as upstream does on MPS and CPU.
- **Training loop** (`kev_eg2/train.py`): upstream's `kev.train.main` LoRA path, built from its own functions. It adds
  a resume point every 250 steps and at every epoch end, and a heartbeat for the watchdog. Records whose augmented
  branch exceeds upstream's 1,024-token row limit under Gemma's tokenizer are skipped for that epoch. That happened 3
  times out of 25,152 record-epochs (two banking77 records); upstream admitted the suite under Qwen's tokenizer.
- **Sequence length**: upstream's own limits (384 state / 1,024 row tokens) apply, with no extra truncation. Under
  Gemma's tokenizer 99 % of training rows are ≤ 949 tokens and 91.7 % are ≤ 384 (`runs/lengths.json`), so a 384 cap
  would have cut 8 %. Rows stay whole up to 1,024; memory is bounded by running one row per backbone pass, padded to a
  64-token bucket.
- **Checks** (`kev_eg2/checks.py`, `runs/checks.json`), all passed:
  - Parity: the adapted model with LoRA disabled, and with LoRA untrained, matches the stock `EmbeddingGemma2Model`
    hidden states. Exact on MPS eager. On CPU the largest difference is 2.4e-6 relative to the 3.6e3 activation peak.
  - Causal arm: changing tokens after t changes nothing before t (Δ 0.0); the bidirectional arm is affected (Δ 29).
  - Bucket padding vs unpadded: 3.4e-5 relative.
  - fp16 is refused.

## Memory and time

- One job at a time (an flock in `kev_eg2/watchdog.py`), under `nice -n 10`.
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2976` (32 GB of Metal's 107.5 GB recommended working set) and
  `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2232`.
- The watchdog polls every 3 s. It kills the process group past 40 GB of job memory (RSS of the process tree plus
  `torch.mps.driver_allocated_memory()` from the job's heartbeat; on unified memory this double-counts, so it errs on
  the safe side), below 25 % system free, at a pressure level other than normal, at swap +2 GB, or past the 12 h budget.
- **The guard fired once, in the first smoke run**: `KILL … job memory 44.4 GB (RSS 23.1 + MPS driver 21.3) > 40 GB`
  after 10 steps. The cause was that MPS keeps buffers per tensor shape: free (rows × length) shapes grew the driver
  allocation past 90 GB in 200 passes in a profile. One row per pass in 64-token buckets levels off at about 8–9 GB.
  After that fix, no further kills.

| Job | wall | peak job memory (RSS + driver) | min free | swap |
|---|---|---|---|---|
| smoke (`--n_per_source 40 --accum 4`, 60 steps) | 1.5 min, 1.24 s/step | 16.4 GB (8.1 + 8.3) | 78 % | no rise |
| probe (real recipe, 60 steps) | 2.5 min, 2.33 s/step → 122 min projected per arm | 15.9 GB | 72 % | no rise |
| train bidirectional (3,144 steps) | 119 min, 2.27 s/step | 22.6 GB (14.2 + 8.4) | 68 % | no rise |
| score bidirectional | 4.6 min | 3.7 GB | 85 % | no rise |
| train causal (3,144 steps) | 119 min | 23.0 GB (14.6 + 8.4) | 66 % | no rise |
| score causal | 4.4 min | 3.6 GB | 86 % | no rise |

Peak MPS allocation during training was 1.5 GB current and 9.1 GB driver. The experiment ran from 15:29 to 20:02,
4.6 h of the 12 h budget. Resume was tested by killing a 30-step run after step 20 with `kill -9`; it continued at
step 20.

## Running it

```bash
./run.sh setup                    # pinned upstream + venv
./run.sh checks && ./run.sh lengths
./run.sh smoke && ./run.sh probe bidirectional
nohup ./pipeline.sh > runs/pipeline.log 2>&1 &   # train + score both arms, one job at a time, then runs/summary.json
```

`runs/` (checkpoints, rows, logs) is gitignored, and so are datasets and weights. Everything above is summarised from
it: `runs/summary.json`, `runs/<arm>/summary.json`, `runs/<arm>/trial/result.json` (upstream's format) and the
watchdog logs.

## Next

- **Stop.** Kev-0.8B stays the default System One model. kev.js and ortllama need no export or engine work for EG2.
- If EmbeddingGemma 2 is revisited, the remaining question is capacity, not data or attention: EG2 is 12 points behind
  Kev-0.6B in distribution on the same data. A distillation target such as Kev-4B's probabilities, or full fine-tuning
  instead of r16 LoRA, would test that. Neither belongs in this pass mark. Nearly every step being clipped suggests the
  recipe's learning rate is not right for this backbone. That knob was not searched, by design.
- Had it passed: Kev-0.8B's follow-up stages (dates, documents and skills), a kev.js-style export and ortllama's
  System One engine.

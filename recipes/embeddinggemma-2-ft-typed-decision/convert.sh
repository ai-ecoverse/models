#!/usr/bin/env bash
# EmbeddingGemma 2 typed decisions: self-tests → fine-tuned weights into onnx-community's text graph → variants →
# decision head → PyTorch reference (decide()) → ONNX parity per variant → fixtures → Kev comparison → card → gate.
# Runs in $WORK, writes $OUT and $RESULTS.
set -euo pipefail
P="python -I"
R="$RECIPE_DIR"
SHIP=()
for v in $RECIPE_VARIANTS; do [[ $v == fp32 ]] || SHIP+=("$v"); done

$P "$LIB/fetch.py" "$RECIPE_SOURCE_REPO" "$RECIPE_SOURCE_REVISION" src --exclude '*.png'
$P "$LIB/fetch.py" "$RECIPE_REFERENCE_REPO" "$RECIPE_REFERENCE_REVISION" ref \
  --include 'onnx/model.onnx*' 'onnx/model_quantized.onnx*' 'onnx/model_fp16.onnx*' 'onnx/model_q4.onnx*' 'onnx/model_q4f16.onnx*'

# Self-tests: the weight mapping rebuilds onnx-community's fp32 data from the base model, and the quantization
# rebuilds their q8, fp16, q4 and q4f16 data from their fp32 graph, byte for byte.
if [[ ! -f "$RESULTS/selftest-weights.json" ]]; then
  $P "$LIB/fetch.py" "$RECIPE_BASE_REPO" "$RECIPE_BASE_REVISION" base --include model.safetensors config.json
  $P "$R/weights.py" --ref ref --ckpt base/model.safetensors --prefix language_model. --check \
    --report "$RESULTS/selftest-weights.json"
  rm -rf base
fi
$P "$R/variants.py" --ref ref --src ref --check --variants q8 fp16 q4 q4f16 --report "$RESULTS/selftest-variants.json"

# The export: fp32 into the reference graph, the shipped variants next to it, the experiments in $WORK only.
$P "$R/weights.py" --ref ref --ckpt src/model.safetensors --config src/config.json --prefix encoder.language_model. \
  --out "$OUT" --report "$RESULTS/weights.json"
$P "$R/variants.py" --ref ref --src "$OUT" --out "$OUT/onnx" --variants ${SHIP[@]+"${SHIP[@]}"} --report "$RESULTS/variants.json"
rm -rf exp
EXP=()
for v in $RECIPE_EXPERIMENTS; do
  if $P "$R/variants.py" --ref ref --src "$OUT" --out exp/onnx --variants "$v"; then EXP+=("$v")
  else echo "::warning::experiment $v did not build"; fi
done
$P "$R/head.py" --src src --out "$OUT" --results "$RESULTS"
cp src/config.json src/tokenizer.json src/tokenizer_config.json "$OUT/"

# Evaluation sets, fetched pinned (not stored in this repo).
$P "$LIB/fetch.py" "$RECIPE_DATASET_REPO" "$RECIPE_DATASET_REVISION" ds --repo-type dataset --include 'all/test-*.parquet'
TD=(ds/all/test-*.parquet)
$P "$R/data.py" kev --repo "$RECIPE_KEV_REPO" --revision "$RECIPE_KEV_REVISION" --path "$RECIPE_KEV_PATH" \
  --sha256 "$RECIPE_KEV_SHA256" --selection-sha256 "$RECIPE_KEV_SELECTION" --out kev/development.jsonl
$P "$R/data.py" github --repo "$RECIPE_KEVJS_REPO" --revision "$RECIPE_KEVJS_REVISION" --out kevjs \
  --paths fixtures/kev-0.8b-transfer-v4-dev300.json fixtures/kev-4b-transfer-v4-dev300.json fixtures/kev-9b-transfer-v4-dev300.json
SETS=(--td "${TD[0]}" --kev kev/development.jsonl)

# Reference: the source's model.decide(), fp32 on the CPU, on all 700 requests (kept between local runs).
[[ -f "$RESULTS/torch.json" ]] || $P "$R/evaluate.py" torch --src src --requests "$R/fixtures/requests.json" "${SETS[@]}" \
  --out "$RESULTS/torch.json"
for v in $RECIPE_VARIANTS; do
  $P "$R/evaluate.py" onnx --export "$OUT" --variant "$v" --ref "$RESULTS/torch.json" "${SETS[@]}" --out "$RESULTS/onnx-$v.json"
done
DONE=()
for v in ${EXP[@]+"${EXP[@]}"}; do
  if $P "$R/evaluate.py" onnx --export "$OUT" --onnx-dir exp/onnx --variant "$v" --ref "$RESULTS/torch.json" "${SETS[@]}" \
    --out "$RESULTS/onnx-$v.json"; then DONE+=("$v")
  else echo "::warning::experiment $v did not run"; fi
done

$P "$R/fixtures.py" --src src --requests "$R/fixtures/requests.json" --out "$OUT/fixtures/typed-decision.json" \
  --export "$OUT" --variants $RECIPE_VARIANTS --onnx-dir exp/onnx --experiments ${DONE[@]+"${DONE[@]}"} --results "$RESULTS/fixtures.json"
cp "$OUT/fixtures/typed-decision.json" "$RESULTS/"
$P "$R/compare.py" "$RESULTS" --kevjs kevjs --export "$OUT" --variants $RECIPE_VARIANTS --experiments ${DONE[@]+"${DONE[@]}"}
$P "$R/card.py" --template "$R/card.md" --out "$OUT" --results "$RESULTS" --variants $RECIPE_VARIANTS --experiments ${DONE[@]+"${DONE[@]}"}
$P "$R/gate.py" "$RESULTS" --variants $RECIPE_VARIANTS --experiments ${DONE[@]+"${DONE[@]}"}

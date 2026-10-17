#!/usr/bin/env bash
# Ministral 3 (8B, 14B; Instruct, Reasoning) → ONNX in Mistral's transformers.js layout. Shared by the
# ministral-3-*-2512 recipes; runs in $WORK, writes $OUT and $RESULTS (see README.md, "convert.sh").
#
#   decoder   onnxruntime-genai model builder, one graph per $RECIPE_VARIANTS entry (decoder.py)
#   vision    Mistral's 3B vision_encoder_fp16 graph with this checkpoint's weights (resize.py)
#   embed     Mistral's 3B embed_tokens_fp16 graph with this checkpoint's table (resize.py)
#   parity    greedy and teacher-forced against PyTorch fp32 (parity.py), gated by gate.py
set -euo pipefail
M="$LIB/ministral3"
P="python -I"

$P "$LIB/fetch.py" "$RECIPE_REFERENCE_REPO" "$RECIPE_REFERENCE_REVISION" ref --include 'onnx/*.onnx' '*.json'
$P "$LIB/fetch.py" "$RECIPE_SOURCE_REPO" "$RECIPE_SOURCE_REVISION" ckpt --exclude consolidated.safetensors

for v in $RECIPE_VARIANTS; do
  $P "$M/decoder.py" --ckpt ckpt --variant "$v" --out "$OUT" --work build | tee -a "$RESULTS/decoder.log" | tail -n 1
  rm -rf "build/genai-$v"
done
$P "$M/resize.py" --ref ref --ckpt ckpt --out "$OUT" | tee "$RESULTS/resize.log"
$P "$M/assemble.py" --ref ref --ckpt ckpt --out "$OUT" \
  ${RECIPE_SAMPLING_TEMPERATURE:+--temperature "$RECIPE_SAMPLING_TEMPERATURE"} \
  ${RECIPE_SAMPLING_TOP_P:+--top-p "$RECIPE_SAMPLING_TOP_P"}

$P "$M/parity.py" torch --ckpt ckpt --repo "$OUT" --ref-out "$RESULTS/torch"
for v in $RECIPE_VARIANTS; do
  $P "$M/parity.py" onnx --repo "$OUT" --variant "$v" --vision fp16 --ref-out "$RESULTS/torch" --out "$RESULTS/onnx-$v.json"
done
$P "$M/gate.py" "$RESULTS" $RECIPE_VARIANTS

$P "$M/card.py" --out "$OUT" --results "$RESULTS" --recipe "$RECIPE_DIR" --variants $RECIPE_VARIANTS > /dev/null

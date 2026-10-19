#!/usr/bin/env bash
# Ministral 3 3B Reasoning → Mistral's Instruct ONNX graphs. Runs in $WORK, writes $OUT and $RESULTS.
set -euo pipefail
P="python -I"
$P "$LIB/fetch.py" "$RECIPE_REFERENCE_REPO" "$RECIPE_REFERENCE_REVISION" ref
$P "$LIB/fetch.py" "$RECIPE_SOURCE_REPO" "$RECIPE_SOURCE_REVISION" ckpt --exclude consolidated.safetensors

# build.py against the reference's own fp32 weights must reproduce Mistral's files. x86 rounds a few 4-bit
# boundary values the other way (≤0.017% of bytes), so allow 0.05%.
$P "$RECIPE_DIR/selftest.py" --ref ref --tmp selftest-tmp --out "$RESULTS/selftest.json" --tolerance 0.0005

$P "$RECIPE_DIR/build.py" --ref ref --ckpt ckpt --out "$OUT" --variants $RECIPE_VARIANTS
mv "$OUT"/build-*.json "$RESULTS/"
$P "$RECIPE_DIR/assemble.py" --ref ref --ckpt ckpt --out "$OUT"
# Mistral's graphs divide by 262144 (inf in fp16) in the long-context query scaling; use transformers' 16384.
$P "$RECIPE_DIR/fix_llama4.py" --out "$OUT" --ckpt ckpt --results "$RESULTS"

$P "$RECIPE_DIR/parity.py" torch --ckpt ckpt --repo "$OUT" --ref-out "$RESULTS/torch"
for v in $RECIPE_VARIANTS; do
  # onnxruntime's CPU kernel for 8-bit MatMulNBits decodes one token at a time very slowly.
  cap=()
  [[ $v == quantized ]] && cap=(--max-new 128)
  $P "$RECIPE_DIR/parity.py" onnx --repo "$OUT" --variant "$v" --ref-out "$RESULTS/torch" \
    --out "$RESULTS/onnx-$v.json" "${cap[@]}"
done
$P "$RECIPE_DIR/gate.py" "$RESULTS" $RECIPE_VARIANTS

cp "$RECIPE_DIR/card/vision.txt" "$RESULTS/vision.txt"
$P "$RECIPE_DIR/card.py" --template "$RECIPE_DIR/card/README.md" --provenance "$RECIPE_DIR/card/provenance.md" \
  --out "$OUT" --results "$RESULTS" --variants $RECIPE_VARIANTS > /dev/null

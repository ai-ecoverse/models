#!/usr/bin/env bash
# Underdog Bark 3.0: model.bark → transformers fp32 → encoder + decoder_joint ONNX → variants → parity → card.
set -euo pipefail
P="python -I"
$P "$LIB/fetch.py" "$RECIPE_SOURCE_REPO" "$RECIPE_SOURCE_REVISION" src \
  --include model.bark bark_loader.py NOTICE
$P "$LIB/fetch.py" "$RECIPE_PROCESSOR_REPO" "$RECIPE_PROCESSOR_REVISION" proc \
  --include config.json generation_config.json processor_config.json tokenizer.json tokenizer_config.json

[[ -f hf/model.safetensors ]] || $P "$RECIPE_DIR/unpack.py" --src src --processor proc --out hf
rm -rf raw && $P "$RECIPE_DIR/export.py" --src hf --out raw
mkdir -p "$OUT/onnx"
$P "$RECIPE_DIR/variants.py" --raw raw --out "$OUT/onnx" --variants $RECIPE_VARIANTS
rm -rf raw
$P "$RECIPE_DIR/assemble.py" --hf hf --src src --out "$OUT"
$P "$RECIPE_DIR/parity.py" --hf hf --out "$OUT" --clips "$RECIPE_DIR/clips" --results "$RESULTS" \
  --variants $RECIPE_VARIANTS
$P "$RECIPE_DIR/card.py" --template "$RECIPE_DIR/card/README.md" --out "$OUT" --results "$RESULTS" \
  --variants $RECIPE_VARIANTS

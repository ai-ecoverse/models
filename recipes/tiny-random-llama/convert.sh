#!/usr/bin/env bash
# tiny-random-llama: fetch → export fp32 → int8 → parity → card. Runs in $WORK.
set -euo pipefail
python -I "$LIB/fetch.py" "$RECIPE_SOURCE_REPO" "$RECIPE_SOURCE_REVISION" src --exclude 'onnx/*' README.md .gitattributes
python -I "$RECIPE_DIR/export.py" --src src --out "$OUT" --results "$RESULTS"
sed -e "s|__SOURCE_REPO__|$RECIPE_SOURCE_REPO|g" -e "s|__SOURCE_REVISION__|$RECIPE_SOURCE_REVISION|g" \
  "$RECIPE_DIR/card.md" > "$OUT/README.md"

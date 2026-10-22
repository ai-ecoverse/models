#!/usr/bin/env bash
set -euo pipefail
P="python -I"
$P "$LIB/fetch.py" "$RECIPE_SOURCE_REPO" "$RECIPE_SOURCE_REVISION" src --include config.json model.safetensors tokenizer.json
$P "$LIB/fetch.py" "$RECIPE_VOICES_REPO" "$RECIPE_VOICES_REVISION" voices --include voices.jsonl
$P "$RECIPE_DIR/export.py" --voices voices --list-voices > voices.txt
[[ $(wc -l < voices.txt) -eq 37 ]] || { echo "expected 37 English voices, found $(wc -l < voices.txt)" >&2; exit 1; }
mapfile -t VOICE_FILES < <(awk '{print $2}' voices.txt)
EASTON=$(awk '$1 == "easton" {print $2}' voices.txt)
$P "$LIB/fetch.py" "$RECIPE_VOICES_REPO" "$RECIPE_VOICES_REVISION" voices --include voices.jsonl "${VOICE_FILES[@]}"

$P "$RECIPE_DIR/export.py" --model src --voices voices --out "$OUT"

if [[ ! -d xn-ptts/.git ]]; then
  rm -rf xn-ptts
  git init -q xn-ptts
  git -C xn-ptts fetch -q "$RECIPE_PTTS_REPO" "$RECIPE_PTTS_REVISION"
  git -C xn-ptts -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
[[ $(git -C xn-ptts rev-parse HEAD) == "$RECIPE_PTTS_REVISION" ]] || { echo "xn-ptts is not at $RECIPE_PTTS_REVISION" >&2; exit 1; }
grep -q "^version = \"$RECIPE_PTTS_VERSION\"" xn-ptts/Cargo.toml || { echo "xn-ptts is not ptts $RECIPE_PTTS_VERSION" >&2; exit 1; }
rm -rf ptts-ref
cp -R "$RECIPE_DIR/ptts-ref" ptts-ref
cp xn-ptts/Cargo.lock ptts-ref/Cargo.lock
(cd ptts-ref && CARGO_TARGET_DIR="$WORK/ptts-target" cargo build --release --quiet)
REF="$WORK/ptts-target/release/ptts-ref"

rm -rf ref
mkdir -p ref
(cd voices && "$REF" voices ../src ../ref/voices "${VOICE_FILES[@]}") > "$RESULTS/ptts-voices.log"
"$REF" speak src "voices/$EASTON" 0.3 42 "$RECIPE_DIR/inputs/speech.txt" ref/speech > "$RESULTS/ptts-speech.log"
for d in -0.3 0.3; do
  "$REF" speak src "voices/$EASTON" 0.3 42 "$RECIPE_DIR/inputs/speech.txt" "ref/speech$d" "duration_delta=$d" \
    > "$RESULTS/ptts-speech$d.log"
done
$P "$RECIPE_DIR/fuzz.py" ref/fuzz.txt
"$REF" plan src/tokenizer.json "$RECIPE_DIR/inputs/text.txt" > ref/text.jsonl
"$REF" plan src/tokenizer.json ref/fuzz.txt > ref/fuzz.jsonl
cp ref/text.jsonl ref/fuzz.jsonl "$RESULTS/"

mkdir -p ortllama
cp "$LIB/../scripts/ortllama/package.json" "$LIB/../scripts/ortllama/package-lock.json" ortllama/
(cd ortllama && npm ci --silent --ignore-scripts --no-audit --no-fund --omit=optional)

status=0
node "$RECIPE_DIR/tokens.mjs" --ortllama ortllama --tokenizer "$OUT/tokenizer.json" \
  --ref ref/text.jsonl --ref ref/fuzz.jsonl --results "$RESULTS/tokens.json" || status=1
$P "$RECIPE_DIR/parity.py" --out "$OUT" --model src --ref ref/speech --ref ref/speech-0.3 --ref ref/speech0.3 \
  --ref-voices ref/voices --results "$RESULTS" || status=1
[[ $status == 0 ]] || exit 1

$P "$RECIPE_DIR/card.py" --template "$RECIPE_DIR/card.md" --results "$RESULTS" --out "$OUT"

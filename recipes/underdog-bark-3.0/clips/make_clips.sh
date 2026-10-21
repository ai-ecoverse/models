#!/usr/bin/env bash
# Regenerate the parity clips with macOS `say` (16 kHz mono 16-bit), stored as FLAC.
# sentences.txt: one "Voice|Sentence" per line; sNN.txt holds the script for sNN.flac.
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
i=0
while IFS='|' read -r voice text; do
  i=$((i + 1)); n=$(printf 's%02d' "$i")
  say -v "$voice" -o "$tmp/$n.wav" --data-format=LEI16@16000 "$text"
  afconvert -f flac -d flac "$tmp/$n.wav" "$n.flac"
  printf '%s\n' "$text" > "$n.txt"
done < sentences.txt

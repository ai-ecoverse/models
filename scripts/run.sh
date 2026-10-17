#!/usr/bin/env bash
# Run one recipe on this machine: venv from requirements.txt, convert.sh, then
# export/ (the recipe and lib/) and export/provenance.json next to the model.
#
#   bash scripts/run.sh tiny-random-llama          # → .models-out/tiny-random-llama/out
#   FORCE=1 bash scripts/run.sh tiny-random-llama  # start from an empty out/ and results/
#
# MODELS_WORK moves the scratch root (default .models-out). Downloads under
# work/ are kept between runs; recipes resume and re-verify them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: run.sh <recipe>}"
RECIPE_DIR="$ROOT/recipes/$NAME"
[[ -f "$RECIPE_DIR/recipe.yaml" ]] || { echo "no recipe $NAME" >&2; exit 2; }
eval "$(node "$ROOT/scripts/read-recipe.mjs" "$NAME" --env)"
[[ "${RECIPE_RUNNER:-}" == retired ]] && { echo "recipe $NAME is retired" >&2; exit 2; }

BASE="${MODELS_WORK:-$ROOT/.models-out}/$NAME"
WORK="$BASE/work" OUT="$BASE/out" RESULTS="$BASE/results" VENV="$BASE/venv"
[[ -n "${FORCE:-}" ]] && rm -rf "$OUT" "$RESULTS"
mkdir -p "$WORK" "$OUT" "$RESULTS"

echo "== $NAME: venv (python ${RECIPE_PYTHON:-3.12})"
[[ -x "$VENV/bin/python" ]] || uv venv -q --python "${RECIPE_PYTHON:-3.12}" "$VENV"
uv pip install -q --python "$VENV/bin/python" -r "$RECIPE_DIR/requirements.txt"

echo "== $NAME: convert"
export RECIPE_DIR WORK OUT RESULTS LIB="$ROOT/lib" VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
export MODELS_BUILT_ON="${MODELS_BUILT_ON:-$(uname -sm), $(getconf _NPROCESSORS_ONLN) CPUs}"
while IFS= read -r v; do export "${v%%=*}"; done < <(node "$ROOT/scripts/read-recipe.mjs" "$NAME" --env)
(cd "$WORK" && bash "$RECIPE_DIR/convert.sh")

echo "== $NAME: export/"
rm -rf "$OUT/export"
mkdir -p "$OUT/export/lib"
(cd "$RECIPE_DIR" && find . -type f ! -path '*/__pycache__/*' -print0 | cpio -0pdm --quiet "$OUT/export")
cp "$ROOT"/lib/*.py "$OUT/export/lib/"
uv pip freeze --python "$VENV/bin/python" > "$OUT/export/environment.txt"
RUN_URL=""
[[ -n "${GITHUB_RUN_ID:-}" ]] && RUN_URL="$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"
COMMIT="$(git -C "$ROOT" rev-parse -q --verify HEAD 2>/dev/null || echo unknown)" \
DIRTY="$(git -C "$ROOT" status --porcelain -- "recipes/$NAME" lib 2>/dev/null | head -1)" \
NAME="$NAME" RUN_URL="$RUN_URL" python -I - "$OUT/export/provenance.json" <<'PY'
import json, os, sys
e = os.environ
json.dump({
    "recipe": e["NAME"],
    "models_repo": "ai-ecoverse/models",
    "commit": e["COMMIT"],
    "dirty": bool(e["DIRTY"]),
    "run": e["RUN_URL"] or None,
    "source": {"repo": e.get("RECIPE_SOURCE_REPO"), "revision": e.get("RECIPE_SOURCE_REVISION")},
    "built_on": e["MODELS_BUILT_ON"],
}, open(sys.argv[1], "w"), indent=1)
PY

echo "== $NAME: done"
(cd "$OUT" && find . -type f ! -path './export/*' -exec ls -l {} + | awk '{printf "%12d  %s\n", $5, $9}' | sort -k2)
du -sh "$OUT"

#!/usr/bin/env bash
# Kev's decision recipe on EmbeddingGemma 2, locally on Apple silicon, one guarded job at a time.
#
#   ./run.sh setup              pinned upstream checkout (.upstream/kev) + venv (.venv)
#   ./run.sh checks             parity / escaping / attention checks -> runs/checks.json
#   ./run.sh lengths            token lengths under Gemma's tokenizer -> runs/lengths.json
#   ./run.sh smoke              upstream's smoke run (--n_per_source 40 --accum 4) -> runs/smoke
#   ./run.sh probe <arm>        60 optimizer steps of the real recipe -> runs/probe-<arm> (step time, memory)
#   ./run.sh train <arm>        the recipe (resumes from runs/<arm>/resume if a previous attempt was killed)
#   ./run.sh score <arm> [dir]  upstream's trial scoring + dev300 + per-source -> runs/<arm>/summary.json
#   ./run.sh summary            runs/summary.json (arm chosen on decision-v7 development; verdict)
#
# <arm> is bidirectional (EmbeddingGemma 2's own attention) or causal (Kev's causal rows).
# Every job runs under kev_eg2.watchdog (one at a time; 40 GB job / 25 % free / pressure / swap +2 GB / 12 h budget)
# and nice; MPS allocation is capped at ~32 GB through the watermark ratios.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
KEV_COMMIT=2074cc44256200bb37540103fa4572901faa10f3
export KEV_ROOT=$HERE/.upstream/kev PYTHONPATH=$HERE PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=$HERE/.venv/bin/python
BUDGET_HOURS=12
CAP_GB=32

recipe=(--suite evals/v7/decision-v7 --base google/embeddinggemma-2 --base_revision 914f7f89142e33e77833254d9c9b90c3cef7303b
        --epochs 2 --lr 1e-4 --lora 16 --batch 1 --accum 8 --p_none_pair 0.25 --special_embeddings 1 --seed 0 --device mps)

mps_env() {
  # the high watermark is a ratio of Metal's recommended working set; aim at CAP_GB, low watermark below it
  local rec; rec=$("$PY" -c 'import torch; print(torch.mps.recommended_max_memory())')
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=$(python3 -c "print(round($CAP_GB * 2**30 / $rec, 4))")
  export PYTORCH_MPS_LOW_WATERMARK_RATIO=$(python3 -c "print(round(0.75 * $CAP_GB * 2**30 / $rec, 4))")
  echo "MPS recommended working set $(python3 -c "print(round($rec / 2**30, 2))") GB -> high watermark $PYTORCH_MPS_HIGH_WATERMARK_RATIO (~$CAP_GB GB), low $PYTORCH_MPS_LOW_WATERMARK_RATIO"
}

deadline() {
  mkdir -p runs
  [[ -f runs/START ]] || date +%s > runs/START
  echo $(( $(cat runs/START) + BUDGET_HOURS * 3600 ))
}

guard() {   # guard <dir> <command...>
  local dir=$1; shift
  mkdir -p "$dir"
  mps_env | tee -a "$dir/watchdog.log"
  "$PY" -m kev_eg2.watchdog --log "$dir/watchdog.log" --heartbeat "$dir/heartbeat.json" --deadline "$(deadline)" -- nice -n 10 "$@"
}

arm() {
  case "${1:-}" in bidirectional|causal) export KEV_EG2_ATTN=$1 ;; *) echo "arm: bidirectional | causal" >&2; exit 2 ;; esac
}

case "${1:-}" in
  setup)
    if [[ ! -d .upstream/kev ]]; then git clone https://github.com/jaredpalmer/kev.git .upstream/kev; fi
    git -C .upstream/kev fetch -q origin "$KEV_COMMIT" 2>/dev/null || true
    git -C .upstream/kev checkout -q "$KEV_COMMIT"
    [[ -x $PY ]] || uv venv -p 3.13 .venv
    uv pip install -p "$PY" -r requirements.txt ;;
  checks)  guard runs/checks "$PY" -m kev_eg2.checks --out runs/checks.json ;;
  lengths) "$PY" -m kev_eg2.lengths --out runs/lengths.json ;;
  smoke)
    export KEV_EG2_ATTN=bidirectional
    guard runs/smoke "$PY" -m kev_eg2.train --run runs/smoke -- --n_per_source 40 --accum 4 \
      --base google/embeddinggemma-2 --base_revision 914f7f89142e33e77833254d9c9b90c3cef7303b --lora 16 --special_embeddings 1 --device mps \
      2>&1 | tee -a runs/smoke/train.log ;;
  probe)
    arm "${2:-}"; rm -rf "runs/probe-$2"
    guard "runs/probe-$2" "$PY" -m kev_eg2.train --run "runs/probe-$2" --save_every_steps 0 -- "${recipe[@]}" --max_steps 60 \
      2>&1 | tee -a "runs/probe-$2/train.log" ;;
  train)
    arm "${2:-}"
    guard "runs/$2" "$PY" -m kev_eg2.train --run "runs/$2" -- "${recipe[@]}" 2>&1 | tee -a "runs/$2/train.log" ;;
  score)   # score <arm> [run dir, default runs/<arm>]
    arm "${2:-}"; dir=${3:-runs/$2}
    guard "$dir/score" "$PY" -m kev_eg2.score arm --run "$dir" 2>&1 | tee -a "$dir/score.log" ;;
  summary) "$PY" -m kev_eg2.score summary --arms runs/bidirectional runs/causal --out runs/summary.json ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac

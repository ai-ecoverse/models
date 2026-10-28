#!/usr/bin/env bash
# The whole experiment after the checks, smoke run and probe: both arms trained and scored one after another (never two
# jobs at once), then the summary. A job the watchdog kills is retried from its resume point after a pause, at most
# twice; the 12-hour budget (runs/START) still applies to every attempt.
#
#   nohup ./pipeline.sh > runs/pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
status() { echo "$(date -Iseconds) $*" | tee -a runs/pipeline.status; }

attempt() {   # attempt <label> <run.sh args...>
  local label=$1; shift
  for try in 1 2 3; do
    status "start $label (try $try)"
    if ./run.sh "$@"; then status "done $label"; return 0; fi
    status "failed $label (try $try): $(tail -1 "runs/${2:-}/watchdog.log" 2>/dev/null)"
    [[ $try -lt 3 ]] && sleep 600
  done
  status "giving up on $label"; return 1
}

for arm in bidirectional causal; do
  attempt "train $arm" train "$arm" || exit 1
  attempt "score $arm" score "$arm" || exit 1
done
./run.sh summary && status "pipeline finished"

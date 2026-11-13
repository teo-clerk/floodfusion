#!/usr/bin/env bash
# floodfusion v1.3 — the adaptation ladder's GPU jobs (reports/protocol_v1_3.md §3; decisions: reports/v1_3_deviations.md).
# Resumable: finished runs are skipped, so a stopped queue is restarted with the same command.
#
#   scripts/v13_queue.sh tune     # rung 3's learning rate: 1e-5, 3e-5, 1e-4 on fold 0's inner split, each from row 21
#   scripts/v13_queue.sh cv       # rung 3: row 21 fine-tuned on each of the five folds (seed 42), from row 21 seed 42
#   scripts/v13_queue.sh score    # each fold model on the benchmark pairs its fold held out
#   scripts/v13_queue.sh report   # (CPU) the three rungs, the two verdicts and v1.3's model: reports/v13_ladder.md
#   scripts/v13_queue.sh ladder   # tune, cv, score and report, in that order
#   scripts/v13_queue.sh finals   # only if rung 3 was adopted: all groups x 3 seeds, each from row 21's matching seed
#
# The GPU lock (/tmp/floodfusion-gpu.lock, shared with the v1.2 blind-test runner) is taken per job and released for
# a minute between jobs. A sweep started meanwhile with a waiting lock gets the GPU when the current job ends:
#   cd ../01_FLOOD_v12_freeze && flock /tmp/floodfusion-gpu.lock uv run --frozen python -m floodfusion.prospective sweep
# Never edit this file while a queue runs (bash reads it as it goes).
set -euo pipefail
cd "$(dirname "$0")/.."

STAGES="tune cv score report ladder finals"
case " $STAGES " in
  *" ${1:-} "*) ;;
  *) echo "usage: $0 ${STAGES// /|}" >&2; exit 2 ;;
esac

if [ -n "$(git status --porcelain -- src configs pyproject.toml uv.lock Makefile scripts/v13_queue.sh)" ]; then
  echo "refusing: uncommitted changes under the code paths (src, configs, pyproject.toml, uv.lock, Makefile, this queue)" >&2
  exit 1
fi
exec 8>/tmp/floodfusion-v13-queue.lock
flock -n 8 || { echo "another v1.3 queue is running" >&2; exit 1; }
exec 9>/tmp/floodfusion-gpu.lock
gpu() {   # gpu <command...>: run one GPU job holding the lock (waits for a running sweep), then leave a window
  flock 9
  "$@"
  flock -u 9
  sleep 60
}

PY="uv run --frozen python"
POOL=data/pool_v13
START=runs/v12/final/all/all                       # row 21's v1.2 finals: <START>/row21_fusion_context_seasonal/seed<n>
LR=reports/v13_lr_row21_fusion_context_seasonal.yaml
RUN_ARGS="--data-root $POOL --overlay configs/overlays/kurosiwo.yaml --overlay $LR --baseline-rows --learned-rows 21
          --seed-protocol full --init-from-root $START"
echo "v1.3 queue '$1' at $(git rev-parse --short HEAD), $(date -u +%FT%TZ)"

tune() {
  gpu $PY -m floodfusion.tuning --data-root $POOL --rows 21 --overlay configs/overlays/kurosiwo.yaml \
    --runs-root runs/v13/tune --grid 1e-5 3e-5 1e-4 --init-from-root $START --tag v13
}
cv() {
  [ -f "$LR" ] || { echo "no $LR: run the tune stage first" >&2; exit 1; }
  for fold in fold0 fold1 fold2 fold3 fold4; do
    # shellcheck disable=SC2086
    gpu $PY -m floodfusion.crossval run $RUN_ARGS --runs-root runs/v13/ladder --protocol kfold --fold $fold --seeds 42
  done
}
score() { gpu $PY -m floodfusion.ladder r3-score --runs runs/v13/ladder --out reports/v13_r3_scores.json.gz; }
report() { $PY -m floodfusion.ladder report --r3 reports/v13_r3_scores.json.gz --out reports/v13_ladder.json; }
finals() {
  model=$($PY -c "import json; print(json.load(open('reports/v13_ladder.json'))['v13_model'])")
  [ "$model" = R3 ] || { echo "refusing: v1.3's model is '$model', not R3 (reports/v13_ladder.md)" >&2; exit 1; }
  # shellcheck disable=SC2086
  gpu $PY -m floodfusion.crossval run $RUN_ARGS --runs-root runs/v13/final --protocol all --seeds 42 43 44
}

case "$1" in
  tune) tune ;;
  cv) cv ;;
  score) score ;;
  report) report ;;
  ladder) tune; cv; score; report ;;
  finals) finals ;;
esac
echo "v1.3 queue '$1' done, $(date -u +%FT%TZ)"

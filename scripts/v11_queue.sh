#!/usr/bin/env bash
# floodfusion v1.1 — every GPU job of reports/protocol_v1_1.md, strictly one after another, at the freeze SHA.
#
#   scripts/v11_queue.sh smoke   # one fold of every configuration (catches crashes before ~45 GPU-hours)
#   scripts/v11_queue.sh all     # the full queue; resumable (finished runs are skipped)
#
# Refuses to start from a dirty tree: the table builders would refuse the runs anyway.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain -- src configs pyproject.toml uv.lock Makefile)" ]; then
  echo "refusing: uncommitted changes under the code paths (src, configs, pyproject.toml, uv.lock, Makefile)" >&2
  exit 1
fi
echo "v1.1 queue at $(git rev-parse --short HEAD), $(date -u +%FT%TZ)"

PY="uv run --frozen python"
V11=runs/v11
S1F11="--data-root data/sen1floods11 --overlay configs/overlays/sen1floods11.yaml"
KS="--data-root data/kurosiwo --overlay configs/overlays/kurosiwo.yaml"
KSW="--data-root data/kurosiwo_water --overlay configs/overlays/kurosiwo.yaml"
S1F11_ROWS="--seed-protocol full --baseline-rows 1 2 3 18 19 --learned-rows 4 9 12 14 15 16 17"
KS_ROWS="--seed-protocol full --baseline-rows 1 2 3 18 19 --learned-rows 4 9 15 16 17"

case "${1:-}" in
  smoke)
    $PY -m floodfusion.crossval run $S1F11 --runs-root $V11/sen1floods11 --protocol loeo official $S1F11_ROWS \
      --fold S1F11_Bolivia official
    $PY -m floodfusion.crossval run $KS --runs-root $V11/kurosiwo --protocol kfold all $KS_ROWS --fold fold0 all
    $PY -m floodfusion.crossval run $KSW --runs-root $V11/kurosiwo_water --protocol kfold all --seed-protocol full \
      --baseline-rows --learned-rows 9 --fold fold0 all
    ;;
  all)
    make v11-synthetic
    $PY -m floodfusion.crossval run $S1F11 --runs-root $V11/sen1floods11 --protocol loeo official $S1F11_ROWS
    $PY -m floodfusion.crossval run $KS --runs-root $V11/kurosiwo --protocol kfold all $KS_ROWS
    $PY -m floodfusion.crossval run $KSW --runs-root $V11/kurosiwo_water --protocol kfold all --seed-protocol full \
      --baseline-rows --learned-rows 9
    make v11-tables
    make v11-cems
    ;;
  *)
    echo "usage: $0 smoke|all" >&2
    exit 2
    ;;
esac
echo "done $(date -u +%FT%TZ)"

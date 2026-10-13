#!/usr/bin/env bash
# floodfusion v1.2 — the GPU jobs, one after another (decisions: reports/v1_2_deviations.md). Resumable: finished
# runs are skipped, so a stopped queue is restarted with the same command.
#
#   scripts/v12_queue.sh erratum       # v1.1 erratum: TerraMind re-trained as its protocol describes it (Sen1Floods11,
#                                      # published split + leave-one-event-out, 3 seeds), then P4 re-scored on CEMS
#   scripts/v12_queue.sh cems-preview  # every v1.1 model + the corrected TerraMind + Copernicus GFM on CEMS 2023-26,
#                                      # with score histograms (descriptive; needs `erratum` and `make gfm` first)
#
# One floodfusion GPU job at a time: the queue holds a lock that every v1.2 GPU job takes.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain -- src configs pyproject.toml uv.lock Makefile)" ]; then
  echo "refusing: uncommitted changes under the code paths (src, configs, pyproject.toml, uv.lock, Makefile)" >&2
  exit 1
fi
exec 9>"${TMPDIR:-/tmp}/floodfusion-gpu.lock"
flock -n 9 || { echo "another floodfusion GPU job holds ${TMPDIR:-/tmp}/floodfusion-gpu.lock" >&2; exit 1; }
echo "v1.2 queue '${1:-}' at $(git rev-parse --short HEAD), $(date -u +%FT%TZ)"

PY="uv run --frozen python"
S1F11="--data-root data/sen1floods11 --overlay configs/overlays/sen1floods11.yaml"
ERRATUM_RUNS=runs/v12/erratum/sen1floods11
TM_ROW="--seed-protocol full --baseline-rows --learned-rows 14"

case "${1:-}" in
  erratum-official)   # the published split only (~15 GPU-minutes): a quick end-to-end check of the fix
    $PY -m floodfusion.crossval run $S1F11 --runs-root $ERRATUM_RUNS --protocol official $TM_ROW
    ;;
  erratum)
    $PY -m floodfusion.crossval run $S1F11 --runs-root $ERRATUM_RUNS --protocol official loeo $TM_ROW
    $PY -m floodfusion.cems_scores score --spec configs/datasets/cems_models_erratum.yaml \
      --out reports/v1_1_erratum_scores.json
    $PY -m floodfusion.cems_scores report --spec configs/datasets/cems_models_erratum.yaml \
      --scores reports/v1_1_erratum_scores.json --out reports/v1_1_erratum_p4.md
    ;;
  cems-preview)
    $PY -m floodfusion.cems_scores_v12 score --spec configs/datasets/cems_models_v12.yaml \
      --out reports/v12_cems_scores.json.gz
    $PY -m floodfusion.cems_scores_v12 report --spec configs/datasets/cems_models_v12.yaml \
      --scores reports/v12_cems_scores.json.gz --out reports/v12_cems_preview.md
    ;;
  *)
    echo "usage: $0 erratum-official|erratum|cems-preview" >&2
    exit 2
    ;;
esac
echo "done $(date -u +%FT%TZ)"

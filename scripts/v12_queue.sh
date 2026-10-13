#!/usr/bin/env bash
# floodfusion v1.2 — the GPU jobs, one after another (decisions: reports/v1_2_deviations.md). Resumable: finished
# runs are skipped, so a stopped queue is restarted with the same command.
#
#   scripts/v12_queue.sh erratum       # v1.1 erratum: TerraMind re-trained as its protocol describes it (Sen1Floods11,
#                                      # published split + leave-one-event-out, 3 seeds), then P4 re-scored on CEMS
#   scripts/v12_queue.sh cems-preview  # every v1.1 model + the corrected TerraMind + Copernicus GFM on CEMS 2023-26,
#                                      # with score histograms (descriptive; needs `erratum` and `make gfm` first)
#   scripts/v12_queue.sh arena         # the Arena's map quicklooks (seeded, score-blind choice of floods)
#   scripts/v12_queue.sh probe         # the fair TerraMind's size/adaptation by the pre-registered timing rule
#   scripts/v12_queue.sh pool          # (CPU) the pooled training set and its grouped folds, once both ingests are done
#   scripts/v12_queue.sh cv            # pooled cross-validation, one seed: row 9, the fair TerraMind, row 21 —
#                                      # each starts when its sidecars (none, DEM, seasonal) cover every pooled stack
#
# One floodfusion GPU job at a time: the queue (and the v1.2 blind-test runner) hold /tmp/floodfusion-gpu.lock;
# ad-hoc GPU commands do not, so run them only when no queue is running.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGES="erratum-official erratum cems-preview arena probe pool cv"
case " $STAGES " in
  *" ${1:-} "*) ;;
  *) echo "usage: $0 ${STAGES// /|}" >&2; exit 2 ;;
esac

if [ -n "$(git status --porcelain -- src configs pyproject.toml uv.lock Makefile)" ]; then
  echo "refusing: uncommitted changes under the code paths (src, configs, pyproject.toml, uv.lock, Makefile)" >&2
  exit 1
fi
exec 9>/tmp/floodfusion-gpu.lock
flock -n 9 || { echo "another floodfusion GPU job holds /tmp/floodfusion-gpu.lock" >&2; exit 1; }
echo "v1.2 queue '${1:-}' at $(git rev-parse --short HEAD), $(date -u +%FT%TZ)"

PY="uv run --frozen python"
S1F11="--data-root data/sen1floods11 --overlay configs/overlays/sen1floods11.yaml"
ERRATUM_RUNS=runs/v12/erratum/sen1floods11
TM_ROW="--seed-protocol full --baseline-rows --learned-rows 14"
POOL=data/pool_v12
POOL_ARGS="--data-root $POOL --overlay configs/overlays/kurosiwo.yaml --runs-root runs/v12/pool"
ONE_SEED="--protocol kfold --seeds 42 --seed-protocol full --baseline-rows"

pool_covered() {   # every pooled stack has sidecar $1 (or, for seasonal, a record saying why it has none)
  $PY - "$1" <<'PYEOF'
import sys
from pathlib import Path
from floodfusion.data.records import load_manifest
need = sys.argv[1]
stacks = {Path(r.raster_path).parent for r in load_manifest(Path("data/pool_v12/chips/manifest.parquet"))}
alt = {"seasonal.tif": "seasonal.json"}.get(need)
missing = [d for d in stacks if not (d / need).is_file() and not (alt and (d / alt).is_file())]
print(f"{need}: {len(stacks) - len(missing)}/{len(stacks)} stacks")
sys.exit(1 if missing else 0)
PYEOF
}

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
  arena)
    $PY scripts/arena_assets.py --spec configs/datasets/cems_models_v12.yaml
    ;;
  probe)
    chips=9000   # estimate before the pool exists: Kuro Siwo's 3 178 training chips plus GEOID-Flood's
    if [ -f "$POOL/chips/manifest.meta.json" ]; then
      chips=$($PY -c "import json; print(int(0.8 * json.load(open('$POOL/chips/manifest.meta.json'))['n_chips']))")
    fi
    $PY scripts/tm_probe.py --data-root data/kurosiwo --pool-chips "$chips"
    ;;
  pool)
    while pgrep -f "floodfusion.data.geoid ingest|floodfusion.data.kurosiwo ingest" > /dev/null; do sleep 120; done
    [ -f data/geoid/chips/manifest.parquet ] || $PY -c "
from floodfusion.config import load_config
from floodfusion.data.ingest import build_dataset
from floodfusion.data.kurosiwo import SPLIT_EPSG
print(build_dataset('data/geoid', load_config('configs/base.yaml'), split_epsg=SPLIT_EPSG))"
    $PY -m floodfusion.data.pool --root data/kurosiwo_v12 data/geoid --out $POOL
    $PY -m floodfusion.crossval splits --data-root $POOL --overlay configs/overlays/kurosiwo.yaml --scheme kfold
    ;;
  cv)
    tm=$($PY -c "import json; print(json.load(open('reports/v12_tm_probe.json'))['chosen'])")
    tm_row=$($PY -c "print({'terramind_base_fair': 24, 'terramind_base_fair_frozen': 25, 'terramind_small_fair': 26}['$tm'])")
    $PY -m floodfusion.crossval run $POOL_ARGS $ONE_SEED --learned-rows 9
    until pool_covered dem.tif; do sleep 600; done
    $PY -m floodfusion.crossval run $POOL_ARGS $ONE_SEED --learned-rows "$tm_row"
    until pool_covered seasonal.tif; do sleep 600; done
    $PY -m floodfusion.crossval run $POOL_ARGS $ONE_SEED --learned-rows 21
    ;;
  *)
    echo "usage: $0 ${STAGES// /|}" >&2
    exit 2
    ;;
esac
echo "done $(date -u +%FT%TZ)"

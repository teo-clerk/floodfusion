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
#                                      # (run it only here: the queue holds the GPU lock it needs)
#   scripts/v12_queue.sh pool          # (CPU) the pooled training set and its grouped folds, once both ingests are done
#   scripts/v12_queue.sh tune          # equal learning-rate budget (3 rates) per family on fold 0's inner split
#   scripts/v12_queue.sh cv            # pooled cross-validation, one seed, each family at its tuned rate: row 9,
#                                      # the fair TerraMind, row 21 (row 9's rate) —
#                                      # each starts when its sidecars (none, DEM, seasonal) cover every pooled stack;
#                                      # then the inference add-ons on whole test stacks of rows 9 and 21, and the
#                                      # selection of v1.2's model (reports/v12_selection.json)
#   scripts/v12_queue.sh finals        # all groups x 3 seeds: v1.2's model and the fair TerraMind (before the freeze)
#
# One floodfusion GPU job at a time: the queue (and the v1.2 blind-test runner) hold /tmp/floodfusion-gpu.lock;
# ad-hoc GPU commands do not, so run them only when no queue is running.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGES="erratum-official erratum cems-preview arena probe pool tune cv finals"
case " $STAGES " in
  *" ${1:-} "*) ;;
  *) echo "usage: $0 ${STAGES// /|}" >&2; exit 2 ;;
esac

if [ -n "$(git status --porcelain -- src configs pyproject.toml uv.lock Makefile)" ]; then
  echo "refusing: uncommitted changes under the code paths (src, configs, pyproject.toml, uv.lock, Makefile)" >&2
  exit 1
fi
exec 9>/tmp/floodfusion-gpu.lock
HOLDS_LOCK=0
gpu_lock() {
  flock -n 9 || { echo "another floodfusion GPU job holds /tmp/floodfusion-gpu.lock" >&2; exit 1; }
  HOLDS_LOCK=1
}
wait_until() {   # wait_until <hours> <command...>: poll until the command succeeds, never holding the GPU lock meanwhile
  local deadline=$(( $(date +%s) + $1 * 3600 ))
  shift
  if [ "$HOLDS_LOCK" = 1 ]; then flock -u 9; fi
  until "$@"; do
    if [ $? -eq 3 ]; then echo "giving up: $*" >&2; exit 1; fi        # 3: will never be ready
    if [ "$(date +%s)" -ge "$deadline" ]; then echo "timed out waiting for: $*" >&2; exit 1; fi
    sleep 300
  done
  if [ "$HOLDS_LOCK" = 1 ]; then flock 9; fi   # blocking: a GPU job started meanwhile finishes first
  return 0
}
addons() {   # addons <run folder>: the inference add-ons on whole test stacks of that row's five CV runs
  $PY -m floodfusion.cv_addons score --runs runs/v12/pool/kfold/fold*/"$1"/seed42 \
    --manifest "$POOL/chips/manifest.parquet" --out "reports/v12_addons_$1.json.gz"
  $PY -m floodfusion.cv_addons report --scores "reports/v12_addons_$1.json.gz" --out "reports/v12_addons_$1.json"
}
tm_name() {   # the fair TerraMind chosen by the probe (exits if none fits in memory)
  local tm
  tm=$($PY -c "import json; print(json.load(open('reports/v12_tm_probe.json'))['chosen'] or '')")
  [ -n "$tm" ] || { echo "no fair TerraMind fits in memory (reports/v12_tm_probe.json)" >&2; exit 1; }
  echo "$tm"
}
tm_row() { $PY -c "print({'terramind_base_fair': 24, 'terramind_base_fair_frozen': 25, 'terramind_small_fair': 26}['$1'])"; }
lr() { echo "--overlay reports/v12_lr_$1.yaml"; }   # a family's tuned learning rate (floodfusion.tuning)
no_ingest() { ! pgrep -f "floodfusion.data.geoid ingest|floodfusion.data.kurosiwo ingest" > /dev/null; }
[ "$1" = pool ] || gpu_lock   # the pool stage is CPU only
echo "v1.2 queue '${1:-}' at $(git rev-parse --short HEAD), $(date -u +%FT%TZ)"

PY="uv run --frozen python"
S1F11="--data-root data/sen1floods11 --overlay configs/overlays/sen1floods11.yaml"
ERRATUM_RUNS=runs/v12/erratum/sen1floods11
TM_ROW="--seed-protocol full --baseline-rows --learned-rows 14"
POOL=data/pool_v12
POOL_ARGS="--data-root $POOL --overlay configs/overlays/kurosiwo.yaml --runs-root runs/v12/pool"
ONE_SEED="--protocol kfold --seeds 42 --seed-protocol full --baseline-rows"

pool_covered() {   # every pooled stack has sidecar $1 (seasonal: or a record of why not), or the data chain is done
  $PY - "$1" <<'PYEOF'
import sys
from pathlib import Path
from floodfusion.data.records import load_manifest
need = sys.argv[1]
stacks = {Path(r.raster_path).parent for r in load_manifest(Path("data/pool_v12/chips/manifest.parquet"))}
alt = {"seasonal.tif": "seasonal.json"}.get(need)
missing = [d for d in stacks if not (d / need).is_file() and not (alt and (d / alt).is_file())]
done = Path("data/v12_pool_sidecars.done").is_file()   # the data chain has tried every stack (twice)
print(f"{need}: {len(stacks) - len(missing)}/{len(stacks)} stacks" + (" (data chain finished)" if done else ""))
if not missing:
    sys.exit(0)
if not done:
    sys.exit(1)                                          # not ready yet
if len(missing) <= 0.01 * len(stacks):                   # read as unavailable (reports/v1_2_deviations.md)
    print(f"WARNING: {len(missing)} pooled stacks have no {need}: {sorted(d.name for d in missing)[:20]}")
    sys.exit(0)
print(f"REFUSING: {len(missing)} pooled stacks (> 1 %) have no {need}; fix the data chain (data/logs) first")
sys.exit(3)
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
    wait_until 24 no_ingest
    [ -f data/geoid/chips/manifest.parquet ] || $PY -c "
from floodfusion.config import load_config
from floodfusion.data.ingest import build_dataset
from floodfusion.data.kurosiwo import SPLIT_EPSG
print(build_dataset('data/geoid', load_config('configs/base.yaml'), split_epsg=SPLIT_EPSG))"
    $PY -m floodfusion.data.pool --root data/kurosiwo_v12 data/geoid --out $POOL
    $PY -m floodfusion.crossval splits --data-root $POOL --overlay configs/overlays/kurosiwo.yaml --scheme kfold
    ;;
  tune)   # the equal learning-rate budget on fold 0's inner split: row 9 now, the fair TerraMind once it has its DEM
    tm=$(tm_name)
    $PY -m floodfusion.tuning --data-root $POOL --overlay configs/overlays/kurosiwo.yaml --rows 9
    wait_until 48 pool_covered dem.tif
    $PY -m floodfusion.tuning --data-root $POOL --overlay configs/overlays/kurosiwo.yaml --rows "$(tm_row "$tm")"
    ;;
  cv)
    tm=$(tm_name)
    $PY -m floodfusion.crossval run $POOL_ARGS $(lr row09_fusion_crossattn) $ONE_SEED --learned-rows 9
    addons row09_fusion_crossattn
    wait_until 48 pool_covered dem.tif
    $PY -m floodfusion.crossval run $POOL_ARGS $(lr "$tm") $ONE_SEED --learned-rows "$(tm_row "$tm")"
    wait_until 48 pool_covered seasonal.tif
    $PY -m floodfusion.crossval run $POOL_ARGS $(lr row09_fusion_crossattn) $ONE_SEED --learned-rows 21
    addons row21_fusion_context_seasonal
    $PY -m floodfusion.cv_addons select --out reports/v12_selection.json \
      --reports reports/v12_addons_row09_fusion_crossattn.json reports/v12_addons_row21_fusion_context_seasonal.json
    ;;
  finals)   # all groups, three seeds: v1.2's model (reports/v12_selection.json) and the fair TerraMind
    ours=$($PY -c "import json; print({'row09_fusion_crossattn': 9, 'row21_fusion_context_seasonal': 21}[
      json.load(open('reports/v12_selection.json'))['chosen']])")
    tm=$(tm_name)
    # Their own runs root: three seeds here, one in the CV, and crossval's protocol.json pins seeds per row.
    finals="--data-root $POOL --overlay configs/overlays/kurosiwo.yaml --runs-root runs/v12/final --protocol all"
    finals="$finals --seeds 42 43 44 --seed-protocol full --baseline-rows"
    $PY -m floodfusion.crossval run $finals $(lr row09_fusion_crossattn) --learned-rows "$ours"
    $PY -m floodfusion.crossval run $finals $(lr "$tm") --learned-rows "$(tm_row "$tm")"
    ;;
  *)
    echo "usage: $0 ${STAGES// /|}" >&2
    exit 2
    ;;
esac
echo "done $(date -u +%FT%TZ)"

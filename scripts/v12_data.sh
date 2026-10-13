#!/usr/bin/env bash
# floodfusion v1.2 — network-only data jobs (no GPU), one after another; every step resumes where it stopped.
#
#   scripts/v12_data.sh rebuild   # erratum 2: Kuro Siwo and the CEMS benchmark re-ingested with the DEM fix into
#                                 # *_v12 roots; unchanged sidecars (seasonal, GFM) hard-linked from the v1.1 roots;
#                                 # dem.tif and seasonal.tif beside every v1.2 stack (Kuro Siwo, GEOID-Flood, CEMS)
#
# The training pool's sidecars come first (the fair TerraMind needs dem.tif, row 21 seasonal.tif), the CEMS
# benchmark's last: nothing in v1.2's cross-validation waits for it.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  rebuild) ;;
  *) echo "usage: $0 rebuild" >&2; exit 2 ;;
esac
PY="uv run --frozen python"
echo "v1.2 data '$1' at $(git rev-parse --short HEAD), $(date -u +%FT%TZ)"
# Wait for every ingest writing these roots (the Kuro Siwo rebuild may have been started on its own, in parallel).
while pgrep -f "floodfusion.data.geoid ingest|floodfusion.data.kurosiwo ingest" > /dev/null; do sleep 60; done
ks_failed() {   # tiles the last Kuro Siwo ingest failed (a missing report counts as "not run")
  $PY -c "import json; print(json.load(open('data/kurosiwo_v12/ingest_report.json'))['summary']['failed'])" 2>/dev/null \
    || echo missing
}
if [ "$(ks_failed)" != "0" ]; then   # resumable: only what is missing or failed is fetched again
  $PY -m floodfusion.data.kurosiwo ingest --raw data/kurosiwo_raw --out data/kurosiwo_v12 \
    || echo "Kuro Siwo ingest: $(ks_failed) tiles still failed; continuing with the rest" >&2
fi

twice() {   # a resumable step, retried once for transient network errors; never stops the chain
  "$@" || "$@" || echo "still failing after a retry (continuing): $*" >&2
}

echo "== training pool: DEM and seasonal sidecars, $(date -u +%FT%TZ)"
rm -f data/v12_pool_sidecars.done
$PY -m floodfusion.data.sidecars link --root data/kurosiwo_v12 --from-root data/kurosiwo
twice $PY -m floodfusion.data.sidecars dem --root data/kurosiwo_v12 data/geoid
twice $PY -m floodfusion.data.seasonal --root data/geoid
twice $PY -m floodfusion.data.seasonal --root data/kurosiwo_v12
date -u +%FT%TZ > data/v12_pool_sidecars.done   # the queue's cv stage stops waiting for full coverage here

echo "== CEMS benchmark (v1.2 rebuild), $(date -u +%FT%TZ)"
$PY -m floodfusion.cems_benchmark prepare --out data/cems_benchmark_v12
for d in data/cems_benchmark_v12/EMSR*_AOI*; do
  $PY -m floodfusion.data.sidecars link --root "$d" --from-root "data/cems_benchmark/$(basename "$d")"
done
$PY -m floodfusion.data.sidecars dem --root data/cems_benchmark_v12/EMSR*_AOI*
for d in data/cems_benchmark_v12/EMSR*_AOI*; do $PY -m floodfusion.data.seasonal --root "$d"; done
echo "done $(date -u +%FT%TZ)"

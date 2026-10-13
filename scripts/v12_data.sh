#!/usr/bin/env bash
# floodfusion v1.2 — network-only data jobs (no GPU), one after another; every step resumes where it stopped.
#
#   scripts/v12_data.sh rebuild   # erratum 2: Kuro Siwo and the CEMS benchmark re-ingested with the DEM fix into
#                                 # *_v12 roots; unchanged sidecars (seasonal, GFM) hard-linked from the v1.1 roots;
#                                 # dem.tif beside every v1.2 stack (Kuro Siwo, GEOID-Flood, CEMS)
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  rebuild) ;;
  *) echo "usage: $0 rebuild" >&2; exit 2 ;;
esac
PY="uv run --frozen python"
echo "v1.2 data '$1' at $(git rev-parse --short HEAD), $(date -u +%FT%TZ)"
# GEOID-Flood (data/geoid) is ingested by its own command; wait for it, to keep one ingest on Planetary Computer.
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
$PY -m floodfusion.cems_benchmark prepare --out data/cems_benchmark_v12
$PY -m floodfusion.data.sidecars link --root data/kurosiwo_v12 --from-root data/kurosiwo
for d in data/cems_benchmark_v12/EMSR*_AOI*; do
  $PY -m floodfusion.data.sidecars link --root "$d" --from-root "data/cems_benchmark/$(basename "$d")"
done
$PY -m floodfusion.data.sidecars dem --root data/kurosiwo_v12 data/geoid data/cems_benchmark_v12/EMSR*_AOI*
echo "done $(date -u +%FT%TZ)"

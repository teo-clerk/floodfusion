#!/usr/bin/env bash
# floodfusion v1.2 — data steps that follow scripts/v12_data.sh rebuild (network only; resumable):
# seasonal references for the GEOID-Flood stacks, and for any rebuilt stack the link step could not serve.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="uv run --frozen python"
echo "v1.2 data follow-up at $(git rev-parse --short HEAD), $(date -u +%FT%TZ)"
while pgrep -f "scripts/v12_data.sh" > /dev/null || pgrep -f "floodfusion.data.geoid ingest" > /dev/null; do sleep 60; done
$PY -m floodfusion.data.seasonal --root data/geoid
$PY -m floodfusion.data.seasonal --root data/kurosiwo_v12
for d in data/cems_benchmark_v12/EMSR*_AOI*; do $PY -m floodfusion.data.seasonal --root "$d" || exit 1; done
echo "done $(date -u +%FT%TZ)"

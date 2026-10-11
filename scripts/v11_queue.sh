#!/usr/bin/env bash
# floodfusion v1.1 — every GPU job of reports/protocol_v1_1.md (amendment 1), one after another, at the freeze SHA.
#
#   scripts/v11_queue.sh smoke   # one fold of every configuration (catches crashes before ~45 GPU-hours)
#   scripts/v11_queue.sh all     # the full queue; resumable (finished runs are skipped)
#
# Order: the models the confirmatory tests need first (the deployable ones, then the CEMS scores), then the
# cross-validation tables, then the synthetic grid. Kuro Siwo runs stop at the pre-registered cap of 12 GPU-hours,
# dropping in the pre-registered order (water-label cross-validation, then the Kuro Siwo variants).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain -- src configs pyproject.toml uv.lock Makefile)" ]; then
  echo "refusing: uncommitted changes under the code paths (src, configs, pyproject.toml, uv.lock, Makefile)" >&2
  exit 1
fi
echo "v1.1 queue at $(git rev-parse --short HEAD), $(date -u +%FT%TZ)"

PY="uv run --frozen python"
V11=runs/v11
S1F11="--data-root data/sen1floods11 --overlay configs/overlays/sen1floods11.yaml --runs-root $V11/sen1floods11"
KS="--data-root data/kurosiwo --overlay configs/overlays/kurosiwo.yaml --runs-root $V11/kurosiwo"
KSW="--data-root data/kurosiwo_water --overlay configs/overlays/kurosiwo.yaml --runs-root $V11/kurosiwo_water"
S1F11_ROWS="--seed-protocol full --baseline-rows 1 2 3 18 19 --learned-rows 4 9 12 14 15 16 17"
KS_MAIN="--seed-protocol full --baseline-rows 1 2 3 18 19 --learned-rows 4 9"
KS_VARIANTS="--seed-protocol full --baseline-rows --learned-rows 15 16 17"
WATER="--seed-protocol full --baseline-rows --learned-rows 9"
KS_CAP_H=12

seasonal_ready() {   # every stack a baseline-2s run or the CEMS scorer will read must have its record
  $PY - <<'EOF'
import sys
from pathlib import Path
missing = [d for root in ("data/sen1floods11/events", "data/kurosiwo/events")
           for d in Path(root).glob("*") if (d / "chips.parquet").is_file() and not (d / "seasonal.json").is_file()]
missing += [d for d in Path("data/cems_benchmark").glob("EMSR*_AOI*/events/*")
            if (d / "stack.tif").is_file() and not (d / "seasonal.json").is_file()]
if missing:
    sys.exit(f"{len(missing)} stacks have no seasonal.json (e.g. {missing[0]}): run `make seasonal` first")
EOF
}

ks_hours() {   # GPU hours spent so far on Kuro Siwo training (both label sets)
  $PY - <<'EOF'
import json
from pathlib import Path
records = (json.loads(p.read_text()) for p in Path("runs/v11").glob("kurosiwo*/*/*/*/seed*/run.json"))
# An evaluation-only sibling (row 11, "trained": false) copies its parent's train_seconds: count it once.
s = sum(float(r.get("train_seconds") or 0) for r in records if r.get("trained", True))
print(f"{s / 3600:.2f}")
EOF
}

cems_done() {   # the CEMS scores and report exist, scored from a clean tree whose code paths equal this one
  $PY - <<'EOF'
import json
import sys
from pathlib import Path
from floodfusion.utils.env import code_changed_since
scores, report = Path("reports/cems_benchmark_scores.json"), Path("reports/cems_benchmark.json")
if not (scores.is_file() and report.is_file()) or report.stat().st_mtime < scores.stat().st_mtime:
    sys.exit(1)
git = json.loads(scores.read_text())["git"]
sys.exit(1 if git.get("git_dirty") or code_changed_since(git.get("git_sha")) else 0)
EOF
}

under_cap() {
  local h; h=$(ks_hours)
  if awk "BEGIN {exit !($h < $KS_CAP_H)}"; then return 0; fi
  echo "Kuro Siwo cap reached (${h} h >= ${KS_CAP_H} h): skipping $1 (reported as not run)"
  return 1
}

case "${1:-}" in
  smoke)
    seasonal_ready
    $PY -m floodfusion.crossval run $S1F11 --protocol official loeo $S1F11_ROWS --fold official S1F11_Bolivia
    $PY -m floodfusion.crossval run $KS --protocol all kfold $KS_MAIN --fold all fold0
    $PY -m floodfusion.crossval run $KS --protocol kfold $KS_VARIANTS --fold fold0
    $PY -m floodfusion.crossval run $KSW --protocol all kfold $WATER --fold all fold0
    ;;
  all)
    seasonal_ready
    # 1. The deployable models every confirmatory test needs, then the CEMS benchmark.
    $PY -m floodfusion.crossval run $S1F11 --protocol official $S1F11_ROWS
    $PY -m floodfusion.crossval run $KS --protocol all $KS_MAIN
    $PY -m floodfusion.crossval run $KSW --protocol all $WATER
    if cems_done; then echo "CEMS scores from matching code exist: skipping make v11-cems"; else make v11-cems; fi
    # 2. Cross-validation (descriptive tables), Kuro Siwo under its cap.
    $PY -m floodfusion.crossval run $S1F11 --protocol loeo $S1F11_ROWS
    $PY -m floodfusion.crossval run $KS --protocol kfold $KS_MAIN
    if under_cap "the water-label cross-validation"; then $PY -m floodfusion.crossval run $KSW --protocol kfold $WATER; fi
    if under_cap "the Kuro Siwo variants"; then $PY -m floodfusion.crossval run $KS --protocol kfold $KS_VARIANTS; fi
    make v11-tables
    # 3. The synthetic grid and stress curves, again at this SHA.
    make v11-synthetic
    echo "Kuro Siwo GPU hours: $(ks_hours)"
    ;;
  *)
    echo "usage: $0 smoke|all" >&2
    exit 2
    ;;
esac
echo "done $(date -u +%FT%TZ)"

"""Write the blind test's frozen spec (``configs/datasets/prospective_v12.yaml``) from the pre-registered choices.

    uv run --frozen python scripts/v12_freeze_spec.py      # once, after the finals; then commit, tag v1.2.0-freeze

**Where each model comes from:**
- **v1.2's model:** row and add-ons from ``reports/v12_selection.json`` (the CV selection rule), trained on all groups
  with seeds 42–44 (``runs/v12/final``).
- **The fair TerraMind:** the variant ``reports/v12_tm_probe.json`` chose, trained the same way.
- **v1.1's best:** fusion trained on Kuro Siwo flood labels (``runs/v11/kurosiwo/all/all``), as the protocol names it.
- **GFM:** Copernicus GFM's ensemble.

**The tests** are P5–P7 of ``reports/protocol_v1_2.md`` (:mod:`floodfusion.blind_analysis`). If self-calibration was
adopted, v1.2's model is scored after it (``metric_b: iou_selfcal``). The HAND gate is not yet implemented in the CEMS
scorer, so its adoption stops this script: it must be added before the freeze, not after.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
FINAL_ROOT = Path("runs/v12/final/all/all")
V11_BEST = Path("runs/v11/kurosiwo/all/all/row09_fusion_crossattn")


def _runs(folder: Path) -> list[str]:
    runs = [folder / f"seed{s}" for s in SEEDS]
    missing = [str(r) for r in runs if not ((ROOT / r / "best.pt").is_file() and (ROOT / r / "eval_test.json").is_file())]
    if missing:
        raise FileNotFoundError(f"finished runs missing: {missing}")
    return [str(r) for r in runs]


def build_spec(selection: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    add_ons = list(selection.get("add_ons", []))
    if "hand" in add_ons:
        raise NotImplementedError("the HAND gate was adopted: apply it in floodfusion.cems_scores_v12 before the freeze")
    if not probe.get("chosen"):
        raise ValueError("the probe chose no fair TerraMind (reports/v12_tm_probe.json)")
    metric_b = "iou_selfcal" if "selfcal" in add_ons else "iou_tau"
    ours, tm = selection["chosen"], f"cmp_{probe['chosen']}"
    models = [
        {"key": "ours", "name": f"v1.2 ({ours}{' + self-calibration' if metric_b == 'iou_selfcal' else ''})",
         "runs": _runs(FINAL_ROOT / ours)},
        {"key": "tm_fair", "name": f"TerraMind, fair ({probe['chosen']})", "runs": _runs(FINAL_ROOT / tm)},
        {"key": "v11", "name": "v1.1 best (fusion, Kuro Siwo flood labels)", "runs": _runs(V11_BEST)},
        {"key": "gfm", "name": "Copernicus GFM (ensemble)", "kind": "gfm", "runs": []},
    ]
    tests = [
        {"id": "P5", "a": "tm_fair", "b": "ours", "metric_b": metric_b, "also_ap": True},
        {"id": "P6", "a": "gfm", "b": "ours", "metric_b": metric_b, "gfm_covered": True},
        {"id": "P7", "a": "v11", "b": "ours", "metric_b": metric_b},
    ]
    return {"primary_stratum": "sar", "exclude": [], "optical_usable": 0.5, "sar_only_below": 0.1,
            "configs_dir": "configs", "overlays": [], "tests": [], "models": models, "blind_tests": tests}


HEADER = """# v1.2 blind test: the frozen models and the pre-registered tests (reports/protocol_v1_2.md; written by
# scripts/v12_freeze_spec.py from reports/v12_selection.json and reports/v12_tm_probe.json). Its sha256 is pinned in
# reports/v12_freeze.json; a sweep refuses to run if this file changes.
"""


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, default=ROOT / "reports/v12_selection.json")
    parser.add_argument("--probe", type=Path, default=ROOT / "reports/v12_tm_probe.json")
    parser.add_argument("--out", type=Path, default=ROOT / "configs/datasets/prospective_v12.yaml")
    args = parser.parse_args(argv)
    spec = build_spec(json.loads(args.selection.read_text(encoding="utf-8")),
                      json.loads(args.probe.read_text(encoding="utf-8")))
    args.out.write_text(HEADER + yaml.safe_dump(spec, sort_keys=False, width=120), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

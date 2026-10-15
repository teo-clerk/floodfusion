"""P5-P7's machinery applied, descriptively, to the CEMS 2023-26 benchmark: floods already seen, so not a test.

    uv run --frozen python scripts/v12_seen_tests.py      # -> reports/v12_cems_seen_tests.md (+ .json)

The frozen models were scored once on the benchmark rebuilt with the DEM fix, after the freeze
(``reports/v12_cems_frozen_scores.json.gz``, ``reports/protocol_v1_2.md`` §5). This reports what the blind test's
analysis (:mod:`floodfusion.blind_analysis`, unchanged) says about those seen floods. That indicates how close the
race is and what the blind test can detect; it is not a verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = """# P5-P7 on the seen CEMS 2023-26 floods (descriptive, not a test)

The frozen v1.2 models were scored once on the CEMS 2023-26 benchmark rebuilt with the DEM fix. They are compared here
with the blind test's own analysis: `floodfusion.blind_analysis`, unchanged, with v1.1's units (primary stratum,
EMSR664 excluded).

**These floods have been seen**, and training sets now cover them too, so these are **not** the pre-registered tests.
Those run on floods after the freeze (`reports/protocol_v1_2.md`). What this table shows is how large the gaps are on
independent maps. The blind test's stopping rule detects about 0.06 IoU at 80 % power; gaps much smaller than that
will most likely end "not supported" there.

"""


def main(argv: list[str] | None = None) -> int:
    from floodfusion import blind_analysis as ba
    from floodfusion.cems_scores_v12 import load_scores

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, default=ROOT / "reports/v12_cems_frozen_scores.json.gz")
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/datasets/prospective_v12.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/v12_cems_seen_tests.md")
    args = parser.parse_args(argv)
    scores, tests = load_scores(args.scores), ba.load_blind_tests(args.spec)
    result = ba.verdicts(scores, {"exclude": ("EMSR664",), "primary_stratum": "sar"}, tests)
    table = ba._markdown(result, tests, scores["models"]).split("\n", 4)[4]       # the table, without its header
    text = HEADER + table.replace("| Verdict |", "| Would read |")
    args.out.write_text(text, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=1, allow_nan=True) + "\n", encoding="utf-8")
    print({k: v["verdict"] for k, v in result.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

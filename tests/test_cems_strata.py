"""scripts/cems_strata.py: the per-stratum table re-reads the stored counts and agrees with the main CEMS report."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("cems_strata", ROOT / "scripts" / "cems_strata.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_primary_stratum_reproduces_the_main_report_and_json_has_no_nan(tmp_path):
    out = tmp_path / "strata.md"
    assert _script().main(["--out", str(out)]) == 0
    text = (out.with_suffix(".json")).read_text(encoding="utf-8")
    assert "NaN" not in text
    strata = json.loads(text)
    main = json.loads((ROOT / "reports" / "cems_benchmark.json").read_text(encoding="utf-8"))["models"]
    primary = next(v for title, v in strata.items() if title.startswith("Drawn on radar"))["all"]["models"]
    assert {m["key"] for m in primary} == set(main)
    for m in primary:
        ref = main[m["key"]]
        assert m["k"] == ref["k"] and math.isclose(m["iou"], ref["iou"], abs_tol=1e-12)
        assert (m["oracle"] is None) == (ref["oracle"] is None)
    assert out.read_text(encoding="utf-8") == (ROOT / "reports" / "cems_benchmark_strata.md").read_text(encoding="utf-8")


def test_a_single_unit_gets_no_interval():
    single = next(v for title, v in json.loads((ROOT / "reports" / "cems_benchmark_strata.json")
                                                .read_text(encoding="utf-8")).items() if title.startswith("EMSR664"))
    assert all(m["k"] == 1 and m["ci"] == [None, None] for m in single["all"]["models"])

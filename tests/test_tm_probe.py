"""The fair-TerraMind size rule (scripts/tm_probe.py), applied to made-up timings: no GPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("tm_probe", Path(__file__).parents[1] / "scripts" / "tm_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

#: Pool size at which row 9's projected run is exactly 1 h.
ONE_HOUR_CHIPS = round(3600 * probe.KS_TRAIN_CHIPS / probe.ROW9_KS_RUN_S)


def _timings(ratios: dict[str, float], reserved: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    reserved = reserved or {}
    return {"row09": {"s_per_micro_batch": 1.0, "peak_reserved_gb": 1.5},
            **{v: {"s_per_micro_batch": r, "peak_reserved_gb": reserved.get(v, 4.0)} for v, r in ratios.items()}}


@pytest.mark.parametrize(("ratios", "reserved", "expected"), [
    ({"terramind_base_fair": 2.9, "terramind_base_fair_frozen": 1.5, "terramind_small_fair": 1.2}, None,
     "terramind_base_fair"),                                             # within 3 h: the strongest variant
    ({"terramind_base_fair": 3.1, "terramind_base_fair_frozen": 2.0, "terramind_small_fair": 1.2}, None,
     "terramind_base_fair_frozen"),                                      # base + LoRA too slow
    ({"terramind_base_fair": 6.0, "terramind_base_fair_frozen": 4.0, "terramind_small_fair": 3.5}, None,
     "terramind_small_fair"),                                            # none within 3 h: the cheapest
    ({"terramind_base_fair": 2.0, "terramind_base_fair_frozen": 1.5, "terramind_small_fair": 1.2},
     {"terramind_base_fair": 7.6}, "terramind_base_fair_frozen"),        # fast, but over the memory cap
    ({"terramind_base_fair": 2.0, "terramind_base_fair_frozen": 1.5, "terramind_small_fair": 1.2},
     {"terramind_base_fair": 8.0, "terramind_base_fair_frozen": 7.9, "terramind_small_fair": 7.6}, None),
])
def test_the_rule_takes_the_first_variant_within_memory_and_time(ratios, reserved, expected) -> None:
    timings = _timings(ratios, reserved)
    chosen, base_h = probe.choose(timings, ONE_HOUR_CHIPS)
    assert chosen == expected
    assert base_h == pytest.approx(1.0, rel=1e-3)
    assert timings["terramind_base_fair"]["projected_run_h"] == pytest.approx(ratios["terramind_base_fair"], rel=1e-3)

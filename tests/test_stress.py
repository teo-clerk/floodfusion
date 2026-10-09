"""Stress sweeps: one factor changes, the landscape is unseen, everything else is fixed."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from floodfusion.data.synthetic.benchmark import DEFAULT_EVENTS
from floodfusion.stress import FACTORS, base_spec, variant


def test_each_variant_changes_exactly_one_factor() -> None:
    base = asdict(base_spec())
    for factor, values in FACTORS.items():
        expected = {"incidence_near_deg", "incidence_far_deg"} if factor == "incidence" else {factor}
        changed_any = set()
        for value in values:
            changed = {k for k, v in asdict(variant(factor, value)).items() if v != base[k]} - {"event_id"}
            assert changed <= expected, (factor, value, changed)        # nothing else ever moves
            changed_any |= changed
        assert changed_any == expected                                  # and the factor itself does


def test_the_stress_landscape_is_not_a_training_landscape() -> None:
    assert base_spec().seed not in {e.seed for e in DEFAULT_EVENTS}


def test_unknown_factor_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown factor"):
        variant("moon_phase", 1)


def test_curves_are_plotted(tmp_path) -> None:
    from floodfusion.stress import _plot_curves

    out = _plot_curves({"wind_speed_ms": {"1": {"a": 0.7, "b": 0.6}, "8": {"a": 0.4, "b": 0.5}}}, tmp_path / "s.png")
    assert out.stat().st_size > 1000

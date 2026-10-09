"""Stress sweeps: one factor changes, the landscape is unseen, everything else is fixed."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from floodfusion.data.synthetic.benchmark import DEFAULT_EVENTS
from floodfusion.stress import FACTORS, LANDSCAPE_SEEDS, base_spec, variant


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


def test_the_stress_landscapes_are_not_training_landscapes() -> None:
    assert len(LANDSCAPE_SEEDS) >= 5 and not set(LANDSCAPE_SEEDS) & {e.seed for e in DEFAULT_EVENTS}
    assert {variant("wind_speed_ms", 2.0, s).seed for s in LANDSCAPE_SEEDS} == set(LANDSCAPE_SEEDS)


def test_unknown_factor_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown factor"):
        variant("moon_phase", 1)


def test_curves_are_plotted(tmp_path) -> None:
    from floodfusion.stress import _plot_curves

    point = {"mean": 0.6, "min": 0.5, "max": 0.7}
    out = _plot_curves({"wind_speed_ms": {"1": {"a": point, "b": point}, "8": {"a": point, "b": point}}}, tmp_path / "s.png")
    assert out.stat().st_size > 1000

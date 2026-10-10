"""Cross-fitted thresholds: each held-out event calibrated on the other held-out events."""

from __future__ import annotations

import pytest

from floodfusion.crossfit import (
    MIN_FLOOD_PIXELS,
    common_grid,
    crossfit_conformal,
    crossfit_cost_threshold,
    event_level_bound,
)
from floodfusion.metrics import ConfusionMatrix as CM

GRID = (0.1, 0.3, 0.5)


def test_only_thresholds_every_event_was_scored_at_are_used() -> None:
    curves = {"A": ((0.1, 0.25, 0.5), [CM(1, 0, 0, 0)] * 3), "B": ((0.1, 0.3, 0.5), [CM(2, 0, 0, 0)] * 3)}
    grid, cms = common_grid(curves)
    assert grid == [0.1, 0.5] and len(cms["A"]) == len(cms["B"]) == 2


def test_each_event_takes_the_threshold_that_is_cost_optimal_on_the_others() -> None:
    # A and B are best at 0.3; C alone would pick 0.5 — but C is held out when its own threshold is chosen.
    a = [CM(90, 50, 10, 850), CM(85, 5, 15, 895), CM(60, 1, 40, 899)]
    c = [CM(50, 90, 0, 860), CM(49, 40, 1, 910), CM(48, 2, 2, 948)]
    out = crossfit_cost_threshold({"A": (GRID, a), "B": (GRID, a), "C": (GRID, c)}, cost_ratio=3.0)
    assert out["C"]["threshold"] == 0.3 and out["C"]["iou"] == pytest.approx(c[1].iou)
    assert out["A"]["threshold"] == 0.3


def _chips(fnr_by_threshold: list[float], n: int = 4) -> dict[str, list[CM]]:
    """Chips with 1000 flood pixels each, missing the given share at each grid threshold."""
    return {f"c{i}": [CM(round(1000 * (1 - f)), 10, round(1000 * f), 5000) for f in fnr_by_threshold] for i in range(n)}


def test_conformal_threshold_is_the_largest_that_keeps_the_bound_on_the_other_events() -> None:
    chips = {"A": (GRID, _chips([0.02, 0.05, 0.30])), "B": (GRID, _chips([0.02, 0.05, 0.30])),
             "C": (GRID, _chips([0.01, 0.20, 0.50]))}
    out = crossfit_conformal(chips, alpha=0.2)
    # For C: 8 calibration chips; bound at 0.3 = 8/9 * 0.05 + 1/9 = 0.156 <= 0.2; at 0.5 it is 0.378.
    assert out["C"]["feasible"] and out["C"]["threshold"] == 0.3 and out["C"]["n_calibration_chips"] == 8
    assert out["C"]["mean_chip_fnr"] == pytest.approx(0.20)     # the guarantee does not carry to C


def test_too_few_flood_pixels_keep_a_chip_out_and_an_unreachable_alpha_is_infeasible() -> None:
    small = {"x": [CM(10, 0, 0, 100)] * 3}
    assert 10 < MIN_FLOOD_PIXELS
    chips = {"A": (GRID, {**_chips([0.0, 0.1, 0.2], n=2), **small}), "B": (GRID, _chips([0.0, 0.1, 0.2], n=2))}
    out = crossfit_conformal(chips, alpha=0.3)
    assert out["B"]["n_calibration_chips"] == 2
    assert crossfit_conformal(chips, alpha=0.05)["B"] == {"feasible": False, "n_calibration_chips": 2}


def test_event_level_control_with_nine_events_is_vacuous_at_ten_percent() -> None:
    assert event_level_bound(9, 0.1) == pytest.approx(0.0)
    assert event_level_bound(9, 0.2) == pytest.approx(1 / 9)
    assert event_level_bound(45, 0.1) == pytest.approx(3.6 / 45)

"""scripts/registration_diagnostic.py: independent checks of the label/optical registration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from floodfusion.data.encoding import N_CHIP_BANDS, ChipBand

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "registration_diagnostic.py"


def _script():
    spec = importlib.util.spec_from_file_location("registration_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stack(seed: int, optical_offset=(0, 0)) -> np.ndarray:
    """A stored (already registered) chip: label on the SAR, SWIR offset by ``optical_offset``."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    water = gaussian_filter(rng.normal(size=(320, 320)), 6) > 0.0
    stack = np.zeros((N_CHIP_BANDS, 320, 320), dtype=np.int16)
    stack[ChipBand.VV_POST] = np.round((np.where(water, -21.0, -9.0) + rng.normal(0, 1, water.shape)) * 100)
    swir = np.where(np.roll(water, optical_offset, axis=(0, 1)), 0.02, 0.25) + rng.normal(0, 0.01, water.shape)
    for b in (ChipBand.B3_GREEN, ChipBand.B8_NIR, ChipBand.B11_SWIR1):
        stack[b] = np.round(swir * 10_000)
    stack[ChipBand.OPTICAL_QUALITY] = 10_000
    stack[ChipBand.LABEL] = water.astype(np.int16)
    stack[ChipBand.LABEL, :16, :] = -1
    return stack


@pytest.fixture(scope="module")
def module():
    return _script()


def test_a_registered_event_passes_every_independent_check(module) -> None:
    # Arrange: 12 chips, labels and optical both on the SAR geometry; chips 0-5 are "train".
    chips = {f"Testland_{i}": module.chip_windows(_stack(i)) for i in range(12)}
    decision = {"label_shift_px": [0, 2], "label_source": "label", "optical_shift_px": [0, 2],
                "optical_source": "label"}

    # Act
    row = module.event_row("Testland", chips, decision, train_ids={f"Testland_{i}" for i in range(6)})

    # Assert
    assert np.hypot(*row["swir_sar"]["median_shift"]) < 0.3
    assert np.hypot(*row["label_swir"]["median_shift"]) < 0.3
    assert row["split_half_shift"] == [0, 2] and np.hypot(*row["split_half_odd_residual"]) < 0.3
    assert row["train_only_shift"] == [0, 2] and row["n_train_chips"] == 6
    np.testing.assert_allclose(row["known_recovered"], module.KNOWN_SHIFT, atol=0.3)
    assert "| Testland | [0, 2] (label) |" in module.render([row])


def test_an_optical_left_off_the_sar_is_reported(module) -> None:
    chips = {f"Testland_{i}": module.chip_windows(_stack(i, optical_offset=(0, 3))) for i in range(12)}
    row = module.event_row("Testland", chips, {"label_shift_px": [0, 0]}, train_ids=set())
    assert row["swir_sar"]["median_shift"][1] == pytest.approx(3.0, abs=0.3)
    assert row["train_only_shift"] is None
    assert "—" in module.render([row])


def test_permanent_water_is_not_an_edge(module) -> None:
    from floodfusion.config import ValidityFlag

    stack = _stack(3)
    stack[ChipBand.VALIDITY] = ValidityFlag.PERMANENT_WATER       # everything flagged: nothing to measure
    assert len(module.chip_windows(stack)["label_sar"]) == 0

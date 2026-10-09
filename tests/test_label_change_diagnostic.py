"""scripts/label_change_diagnostic.py: flood labels with and without a backscatter drop, per event."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from floodfusion.data.chipping import AnalysisGrid, write_event_stack
from floodfusion.data.encoding import N_CHIP_BANDS, ChipBand

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "label_change_diagnostic.py"
GRID = AnalysisGrid(epsg=32633, transform=(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_900_000.0), width=8, height=8)


def _script():
    spec = importlib.util.spec_from_file_location("label_change_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stack(root: Path, chip: str, *, pre_db: float, post_db: float) -> None:
    """Left half labelled flood, right half dry; one flood pixel invalid, one with no pre-event data."""
    stack = np.zeros((N_CHIP_BANDS, GRID.height, GRID.width), dtype=np.int16)
    stack[ChipBand.VV_PRE] = round(pre_db * 100)
    stack[ChipBand.VV_POST] = round(post_db * 100)
    stack[ChipBand.DELTA_VV] = round((post_db - pre_db) * 100)
    stack[ChipBand.LABEL, :, :4] = 1
    stack[ChipBand.VALIDITY, 0, 0] = 1  # flagged: excluded
    stack[ChipBand.VV_PRE, 1, 0] = -32768  # no reference: excluded
    path = root / "events" / chip / "stack.tif"
    path.parent.mkdir(parents=True)
    write_event_stack(path, stack, GRID)


def test_counts_events_whose_labelled_water_did_not_darken(tmp_path: Path) -> None:
    # Arrange: water already there (no drop) vs a true flood (12 dB drop), two chips for the first.
    _stack(tmp_path, "S1F11_Ghana_1", pre_db=-20.0, post_db=-20.5)
    _stack(tmp_path, "S1F11_Ghana_2", pre_db=-20.0, post_db=-20.5)
    _stack(tmp_path, "S1F11_Spain_1", pre_db=-8.0, post_db=-20.0)
    module = _script()

    # Act
    rows = {name: rest for name, *rest in module.event_rows(tmp_path)}

    # Assert: 2 x (32 flood - 2 excluded) pixels pooled for Ghana; Spain's water is new.
    assert rows["Ghana"][0] == 60 and rows["Ghana"][1] == pytest.approx(1.0)
    assert rows["Ghana"][3] == pytest.approx(-0.5)
    assert rows["Spain"][1] == pytest.approx(0.0) and rows["Spain"][2] == pytest.approx(1.0)
    assert rows["Spain"][3] == pytest.approx(-12.0)


def test_report_states_the_count(tmp_path: Path, monkeypatch) -> None:
    _stack(tmp_path, "S1F11_Ghana_1", pre_db=-20.0, post_db=-20.0)
    _stack(tmp_path, "S1F11_Spain_1", pre_db=-8.0, post_db=-20.0)
    out = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["diag", "--data-root", str(tmp_path), "--out", str(out)])

    assert _script().main() == 0
    text = out.read_text(encoding="utf-8")
    assert "**1 of 2 events show no clear backscatter drop**" in text
    assert "| Spain | 30 | 0.00 | 1.00 | -12.00 |" in text

"""In-memory synthetic chips: the dataset's offline fallback.

:class:`floodfusion.data.dataset.FloodChipDataset` calls :func:`synthesize_chip` when
``DataConfig.chip_source`` is ``"synthetic"``, or ``"auto"`` and the record's raster does
not exist. The chip is a whole tiny event rendered by the same physics simulator and the
same terrain code as the full benchmark, and **encoded exactly like data on disk**, so the
decode/normalise/augment path the model sees is the production one.

Deterministic in ``chip_id`` (a SHA-256 seed), so two datasets built from the same record
return identical samples — evaluation must be reproducible — while different chips differ.
"""

from __future__ import annotations

import hashlib

import numpy as np

from floodfusion.config import GridConfig, NormalizationStats
from floodfusion.data.records import ChipRecord
from floodfusion.data.synthetic.landscape import EventSpec
from floodfusion.data.synthetic.scene import render_event

__all__ = ["chip_seed", "synthesize_chip"]


def chip_seed(chip_id: str) -> int:
    """A stable 63-bit seed from the chip identifier."""
    return int.from_bytes(hashlib.sha256(chip_id.encode("utf-8")).digest()[:8], "little") >> 1


def synthesize_chip(
    record: ChipRecord,
    grid_cfg: GridConfig,
    norm: NormalizationStats,
    *,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    """Render a ``(16, height, width)`` int16 chip for ``record`` (window size by default)."""
    h = int(height if height is not None else record.window[3])
    w = int(width if width is not None else record.window[2])
    seed = chip_seed(record.chip_id)
    orbit = record.orbit_state if record.orbit_state in {"ascending", "descending"} else "descending"
    spec = EventSpec(
        event_id=f"synthetic_{record.chip_id}",
        seed=seed % (2**32),
        n_basins=1,
        basin_width_px=max(w, 16),
        height_px=max(h, 16),
        terrain=("lowland", "hilly")[seed % 2],
        forest_cover=0.2 + 0.4 * ((seed >> 3) % 100) / 100.0,
        riparian_forest=bool((seed >> 5) & 1),
        wind_speed_ms=1.0 + 6.0 * ((seed >> 7) % 100) / 100.0,
        orbit_state=orbit,
        optical="clear" if record.optical_available else "none",
        flood_depth_m=(1.0 + 2.5 * ((seed >> 11) % 100) / 100.0,),
        sand_bars=w >= 64,
    )
    stack = render_event(spec, grid_cfg, norm).stack
    return stack[:, :h, :w]

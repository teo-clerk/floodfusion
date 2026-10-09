"""Live checks against Planetary Computer (anonymous). Excluded by default: ``pytest -m network``.

The event is EMSR664 (Emilia-Romagna, May 2023): first post-event pass 2023-05-22 on
relative orbit 95 (descending), matched by 2023-05-10 on the same orbit.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.network

PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = (11.90, 44.40, 11.94, 44.43)


@pytest.fixture(scope="module")
def matched():
    from floodfusion.data.stac import EventQuery, search_matched_pairs

    query = EventQuery("EMSR664_live", BBOX, datetime(2023, 5, 16), datetime(2023, 5, 22), 45)
    return search_matched_pairs(query, PC)


def test_matched_pair_on_the_same_orbit(matched) -> None:
    assert (matched.relative_orbit, matched.orbit_state) == (95, "descending")
    assert matched.s1_post_datetime.startswith("2023-05-22") and matched.s1_pre_datetime.startswith("2023-05-10")


def test_product_annotation_is_found_and_parsed(matched) -> None:
    from floodfusion.data.ingest import IngestSources, _read_geometry

    annotations = _read_geometry(IngestSources.open(PC), matched)
    assert annotations[0].heading_deg == pytest.approx(-166.0, abs=1.0)
    assert annotations[0].look_azimuth_deg == pytest.approx(104.0, abs=1.0)  # descending ~103 deg


@pytest.mark.slow
def test_tiny_event_ingest(tmp_path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import box

    from floodfusion.config import load_config
    from floodfusion.data.ingest import EventConfig, ingest_event

    gpd.GeoDataFrame(geometry=[box(11.90, 44.40, 11.92, 44.43)], crs="EPSG:4326").to_file(tmp_path / "label.geojson")
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "base.yaml",
                      overrides=["data.chip_size=64", "data.train_stride=32", "data.eval_stride=64"])
    event = EventConfig("EMSR664_live", BBOX, date(2023, 5, 16), date(2023, 5, 22), tmp_path / "label.geojson",
                        pre_event_window_days=45)
    meta = ingest_event(event, tmp_path / "out", cfg)
    assert meta["n_chips"] > 0 and meta["relative_orbit"] == 95
    assert all(qa["finite_fraction"] > 0.99 and -12 < qa["median_db"] < -4 for name, qa in meta["rtc_qa"].items()
               if name.startswith("vv"))
    assert meta["coregistration_qa"]["pre_vs_post"]["median_abs_displacement"] < 1.0  # P1-M1
    assert 39.0 < np.mean(meta["incidence_deg"]) < 41.0

"""The orbit-model geometry fallback, validated against a real Sentinel-1 product annotation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import MultiPoint

from floodfusion.data.orbit_model import footprint_geometry, ground_track_heading
from floodfusion.data.s1_annotation import parse_annotation

ANNOTATION = Path(__file__).parent / "fixtures" / "s1a_iw_grd_vv_20231107t170700_annotation_trimmed.xml"


@pytest.fixture(scope="module")
def annotation():
    return parse_annotation(ANNOTATION.read_bytes())


def test_heading_matches_the_annotation(annotation) -> None:
    lat = float(np.mean(annotation.geolocation["latitude"]))
    assert ground_track_heading(lat, "ascending") == pytest.approx(annotation.heading_deg, abs=0.5)


def test_heading_is_mirrored_for_descending_passes() -> None:
    asc, desc = ground_track_heading(44.4, "ascending"), ground_track_heading(44.4, "descending")
    assert -16 < asc < -12 and -168 < desc < -164 and asc + desc == pytest.approx(-180.0, abs=1e-9)
    assert (ground_track_heading(44.4, "descending") - 90.0) % 360.0 == pytest.approx(104.0, abs=1.0)  # EMSR664
    with pytest.raises(ValueError):
        ground_track_heading(44.4, "sideways")


def test_footprint_incidence_is_within_a_degree_of_the_tie_points(annotation) -> None:
    g = annotation.geolocation
    footprint = MultiPoint(np.column_stack([g["longitude"], g["latitude"]])).convex_hull
    model = footprint_geometry(footprint, annotation.orbit_state)
    assert model.look_azimuth_deg == pytest.approx(annotation.look_azimuth_deg, abs=0.5)
    assert model.orbit_state == "ascending"

    from scipy.interpolate import griddata

    pts = np.column_stack([model.geolocation["longitude"], model.geolocation["latitude"]])
    predicted = griddata(pts, model.geolocation["incidence"], np.column_stack([g["longitude"], g["latitude"]]),
                         method="linear")
    err = np.abs(predicted - g["incidence"])[np.isfinite(predicted)]
    assert len(err) > 150 and np.max(err) < 0.8 and np.median(err) < 0.3

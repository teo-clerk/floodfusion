"""Sentinel-1 annotation parsing, validated on a real (trimmed) ESA product annotation.

Fixture: ``tests/fixtures/s1a_iw_grd_vv_20231107t170700_annotation_trimmed.xml`` — the VV
product annotation of S1A_IW_GRDH_1SDV_20231107T170700_20231107T170725_051114_062A12
(relative orbit 117, ascending), with everything but the header, product information,
orbit list, image information, geolocation grid and coordinate conversion removed.
Contains modified Copernicus Sentinel data 2023.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from floodfusion.data.geocoding import geodetic_to_ecef, zero_doppler_time
from floodfusion.data.s1_annotation import parse_annotation

FIXTURE = Path(__file__).parent / "fixtures" / "s1a_iw_grd_vv_20231107t170700_annotation_trimmed.xml"


@pytest.fixture(scope="module")
def annotation():
    return parse_annotation(FIXTURE)


class TestParsing:
    def test_header_and_geometry(self, annotation) -> None:
        assert (annotation.mission, annotation.mode, annotation.polarisation) == ("S1A", "IW", "VV")
        assert annotation.orbit_state == "ascending"
        assert annotation.heading_deg == pytest.approx(-14.083, abs=1e-3)

    def test_look_azimuth_is_heading_minus_90(self, annotation) -> None:
        """Bearing target -> sensor for a right-looking SAR; ascending is ~257 deg (docs/03 5.1)."""
        assert annotation.look_azimuth_deg == pytest.approx((-14.083 - 90.0) % 360.0, abs=1e-3)
        assert 250.0 < annotation.look_azimuth_deg < 262.0

    def test_incidence_spans_the_iw_swath(self, annotation) -> None:
        inc = annotation.geolocation["incidence"]
        assert inc.min() == pytest.approx(30.3, abs=0.5) and inc.max() == pytest.approx(46.2, abs=0.5)

    def test_incidence_interpolation_increases_with_range(self, annotation) -> None:
        """Ascending, right-looking: range increases eastward, so does the incidence angle."""
        lat = np.full(3, 44.3)
        lon = np.array([9.5, 10.5, 11.5])
        inc = annotation.incidence_on(lon, lat)
        assert np.all(np.diff(inc) > 0)

    def test_lists_are_complete(self, annotation) -> None:
        assert len(annotation.orbit_times) >= 10
        assert annotation.geolocation["line"].size == annotation.geolocation["incidence"].size > 100
        assert len(annotation.srgr) > 10
        assert annotation.near_range_m == pytest.approx(8.0e5, rel=0.01)

    def test_rejects_entity_declarations(self) -> None:
        bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]><product>&b;</product>'
        with pytest.raises(ValueError, match="DTD"):
            parse_annotation(bomb)

    def test_rejects_non_annotation_xml(self) -> None:
        with pytest.raises(ValueError, match="annotation"):
            parse_annotation("<rfi><adsHeader><missionId>S1A</missionId></adsHeader></rfi>")


class TestGeolocationConsistency:
    def test_range_doppler_reproduces_esa_tie_points(self, annotation) -> None:
        """Our zero-Doppler solve + SRGR model maps ESA's tie points to their annotated
        (line, pixel) to sub-pixel accuracy: the geocoder is right for real orbits."""
        orbit = annotation.orbit_state_vectors()
        geom = annotation.srgr_geometry()
        g = annotation.geolocation
        targets = geodetic_to_ecef(g["latitude"], g["longitude"], g["height"])
        t = zero_doppler_time(orbit, targets)
        sensor, _, _ = orbit.interpolate(t)
        line, pixel = geom.to_image(t, np.linalg.norm(targets - sensor, axis=-1))
        assert np.median(np.abs(line - g["line"])) < 0.5
        assert np.median(np.abs(pixel - g["pixel"])) < 0.5
        assert np.abs(pixel - g["pixel"]).max() < 2.0

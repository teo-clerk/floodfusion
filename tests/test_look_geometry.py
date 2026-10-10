r"""Look-geometry convention: ``look_azimuth_deg`` is the bearing FROM target TO sensor.

``docs/03`` section 5.1 defines :math:`\phi` as "the compass bearing from target to sensor"
— the definition the local-incidence formula needs — but then states that :math:`\phi` "is
the heading rotated +90 deg", which is the *beam* direction (sensor to target), 180 deg
away. A descending Sentinel-1 pass heads ~193 deg (SSW) and looks right, i.e. WNW, so the
sensor sits ESE of the target: :math:`\phi \approx 103` deg. Getting this backwards puts
every layover and shadow on the wrong flank and silently corrupts the
:math:`\theta_{\mathrm{loc}}` channel. These tests pin the convention.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from floodfusion.masks import cast_shadow_and_layover, horn_slope_aspect, local_incidence_angle

DESCENDING_HEADING_DEG = 193.0
PHI_DESCENDING = (DESCENDING_HEADING_DEG - 90.0) % 360.0  # 103 deg, target -> sensor


def _plane(slope_deg: float, downslope_azimuth_deg: float, n: int = 40, pixel: float = 10.0) -> np.ndarray:
    """A planar DEM whose downslope direction is the given compass bearing."""
    rows, cols = np.mgrid[0:n, 0:n]
    east, north = cols * pixel, -rows * pixel
    a = math.radians(downslope_azimuth_deg)
    # elevation decreases along the downslope unit vector (sin a, cos a)
    return -math.tan(math.radians(slope_deg)) * (east * math.sin(a) + north * math.cos(a)) + 500.0


def test_aspect_is_the_downslope_compass_bearing() -> None:
    for bearing in (0.0, 45.0, 90.0, 180.0, 270.0, 315.0):
        _, aspect = horn_slope_aspect(_plane(20.0, bearing), 10.0, 10.0)
        got = float(np.median(aspect[2:-2, 2:-2]))
        assert min(abs(got - bearing), 360 - abs(got - bearing)) < 0.5, (bearing, got)


def test_slope_facing_a_descending_sensor_reduces_local_incidence() -> None:
    """An east-facing 30 deg slope faces the ESE-located descending sensor."""
    slope, aspect = horn_slope_aspect(_plane(30.0, 90.0), 10.0, 10.0)
    theta_loc = local_incidence_angle(slope, aspect, 39.0, PHI_DESCENDING)
    expected = math.degrees(math.acos(
        math.cos(math.radians(30)) * math.cos(math.radians(39))
        + math.sin(math.radians(30)) * math.sin(math.radians(39)) * math.cos(math.radians(PHI_DESCENDING - 90))
    ))
    assert np.median(theta_loc[2:-2, 2:-2]) == pytest.approx(expected, abs=0.2)
    assert expected < 39.0 - 25.0, "a facing slope must cut the local incidence by ~30 deg"


def test_facing_slope_steeper_than_incidence_is_layover_not_shadow() -> None:
    dem = _plane(45.0, PHI_DESCENDING)  # faces the sensor, 45 > 39
    shadow, layover = cast_shadow_and_layover(dem, 39.0, PHI_DESCENDING, 10.0)
    inner = (slice(3, -3), slice(3, -3))
    assert layover[inner].mean() > 0.9
    assert not shadow[inner].any()


def test_back_slope_steeper_than_depression_is_shadow_not_layover() -> None:
    dem = _plane(55.0, (PHI_DESCENDING + 180.0) % 360.0)  # faces away, 55 > 51
    shadow, layover = cast_shadow_and_layover(dem, 39.0, PHI_DESCENDING, 10.0)
    inner = (slice(3, -3), slice(3, -3))
    assert shadow[inner].mean() > 0.9
    assert not layover[inner].any()


def test_local_shadow_lies_inside_the_sweep_shadow(synthetic_dem: np.ndarray) -> None:
    """The ray sweep and the facet test must agree where both apply (cast ⊇ local)."""
    parts = cast_shadow_and_layover(synthetic_dem, 39.0, PHI_DESCENDING, 10.0, return_components=True)
    local = parts["local_shadow"]
    assert local.sum() > 0
    agreement = (local & parts["cast_shadow"]).sum() / local.sum()
    assert agreement >= 0.9, f"only {agreement:.0%} of facet shadow is inside the sweep shadow"


def test_cast_shadow_reaches_beyond_the_facet_test(synthetic_dem: np.ndarray) -> None:
    """Flat ground behind the ridge is shadowed although its facet is perfectly lit."""
    parts = cast_shadow_and_layover(synthetic_dem, 39.0, PHI_DESCENDING, 10.0, return_components=True)
    assert (parts["cast_shadow"] & ~parts["local_shadow"]).any()


def test_shadow_falls_on_the_far_side_of_the_ridge(synthetic_dem: np.ndarray) -> None:
    """Sensor ESE (phi=103) => the shadow lies WEST of the ridge crest (column 48)."""
    shadow, _ = cast_shadow_and_layover(synthetic_dem, 39.0, PHI_DESCENDING, 10.0)
    cols = np.nonzero(shadow)[1]
    assert np.median(cols) < 48

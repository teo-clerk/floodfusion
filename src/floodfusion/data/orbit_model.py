r"""Acquisition geometry without a product annotation: a documented orbit-model fallback.

The heading and the per-pixel incidence normally come from the GRD product annotation
(:mod:`floodfusion.data.s1_annotation`). Planetary Computer's ``sentinel-1-grd`` has gaps —
the 2017-06-28 datatake over Pakistan exists as RTC but not as GRD — and the change of the
look azimuth decides every shadow and layover mask, so a missing annotation cannot mean a
silent default. This module models the geometry from what the RTC item does carry:

- **Heading** of the ground track of a circular, sun-synchronous orbit (inclination
  98.18 deg, altitude 693 km) at the scene latitude, with the Earth's rotation removed from
  the inertial track: :math:`\sin\alpha_i = \cos i / \cos\varphi`,
  :math:`v_N = v_g\cos\alpha_i`, :math:`v_E = v_g\sin\alpha_i - \omega_E R\cos\varphi`.
  Against a real annotation (2023-11-07, 44.9 N, ascending) the error is 0.2 deg.
- **Incidence** across the frame footprint, from the near-range edge (on the track side of
  a right-looking SAR) to the far-range edge, between the IW swath limits measured on a
  real annotation (30.3 and 46.2 deg), with the spherical-Earth viewing geometry: at Earth
  central angle :math:`\gamma` from nadir the incidence is :math:`\theta = \eta + \gamma`,
  :math:`\tan\eta = R\sin\gamma / (R + H - R\cos\gamma)`. The two limits fix the swath's
  nadir distances; a point's across-track position maps linearly in ground distance
  between them. Against the same annotation the error is 0.23 deg median and 0.74 deg at
  worst (a smooth near-to-far bias of +0.1 to -0.35 deg; a linear mapping errs by 1.25 deg)
  — ample for local incidence and the shadow/layover sweep.

The fallback is recorded in every event's metadata (``geometry_source``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["IW_INCIDENCE_DEG", "FootprintGeometry", "footprint_geometry", "ground_track_heading"]

S1_INCLINATION_DEG = 98.18
S1_ALTITUDE_M = 693_000.0
EARTH_RADIUS_M = 6_371_008.8
EARTH_GM = 3.986004418e14
EARTH_ROTATION_RAD_S = 7.2921150e-5
#: IW swath limits measured on a real GRD annotation's geolocation grid (pixel 0 and last).
IW_INCIDENCE_DEG = (30.3, 46.2)
_M_PER_DEG_LAT = 110_574.0
_M_PER_DEG_LON_EQ = 111_320.0


def ground_track_heading(lat_deg: float, orbit_state: str) -> float:
    """Heading of the satellite ground track (deg from north, clockwise, in (-180, 180])."""
    state = orbit_state.lower()
    if state not in ("ascending", "descending"):
        raise ValueError(f"orbit_state must be ascending or descending, got {orbit_state!r}")
    lat = math.radians(lat_deg)
    sin_a = math.cos(math.radians(S1_INCLINATION_DEG)) / math.cos(lat)
    if abs(sin_a) > 1.0:
        raise ValueError(f"latitude {lat_deg} is beyond the orbit's reach")
    a = math.asin(sin_a) if state == "ascending" else math.pi - math.asin(sin_a)
    v_orbit = math.sqrt(EARTH_GM / (EARTH_RADIUS_M + S1_ALTITUDE_M))
    v_ground = v_orbit * EARTH_RADIUS_M / (EARTH_RADIUS_M + S1_ALTITUDE_M)
    v_north = v_ground * math.cos(a)
    v_east = v_ground * math.sin(a) - EARTH_ROTATION_RAD_S * EARTH_RADIUS_M * math.cos(lat)
    heading = math.degrees(math.atan2(v_east, v_north))
    return heading if heading > -180.0 else heading + 360.0


@dataclass(frozen=True)
class FootprintGeometry:
    """Stands in for :class:`~floodfusion.data.s1_annotation.S1Annotation` where only geometry is needed."""

    heading_deg: float
    pass_direction: str
    geolocation: dict[str, np.ndarray] = field(repr=False)
    source: str = "orbit model + footprint (no annotation)"

    @property
    def look_azimuth_deg(self) -> float:
        """Bearing from target to sensor (right-looking): heading - 90 deg, in [0, 360)."""
        return (self.heading_deg - 90.0) % 360.0

    @property
    def orbit_state(self) -> str:
        return self.pass_direction.lower()


def _incidence_at(gamma: np.ndarray) -> np.ndarray:
    """Ellipsoid-free incidence (deg) at Earth central angle ``gamma`` (rad) from nadir."""
    r, h = EARTH_RADIUS_M, S1_ALTITUDE_M
    eta = np.arctan2(r * np.sin(gamma), r + h - r * np.cos(gamma))
    return np.degrees(eta + gamma)


def _incidence_profile(frac: np.ndarray, near_deg: float, far_deg: float) -> np.ndarray:
    """Incidence at fractional across-track position ``frac`` (0 near edge, 1 far edge)."""
    gamma = np.linspace(0.0, 0.2, 20_001)              # 0 .. ~1270 km from nadir
    theta = _incidence_at(gamma)
    g_near, g_far = np.interp([near_deg, far_deg], theta, gamma)
    return _incidence_at(g_near + np.clip(frac, 0.0, 1.0) * (g_far - g_near))


def _densify(ring: np.ndarray, n: int) -> np.ndarray:
    seg = np.diff(ring, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    t = np.linspace(0.0, float(length.sum()), n, endpoint=False)
    edges = np.concatenate([[0.0], np.cumsum(length)])
    k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(seg) - 1)
    frac = (t - edges[k]) / np.where(length[k] > 0, length[k], 1.0)
    return ring[k] + frac[:, None] * seg[k]


def footprint_geometry(footprint: Any, orbit_state: str, *, incidence_deg: tuple[float, float] = IW_INCIDENCE_DEG,
                       n_points: int = 256) -> FootprintGeometry:
    """Heading and tie-point incidence for one frame from its footprint polygon (EPSG:4326)."""
    polygon = max(getattr(footprint, "geoms", [footprint]), key=lambda g: g.area)
    ring = np.asarray(polygon.exterior.coords, dtype=np.float64)
    lon0, lat0 = polygon.centroid.x, polygon.centroid.y
    heading = ground_track_heading(lat0, orbit_state)
    points = np.vstack([_densify(ring, n_points), [[lon0, lat0]]])
    x = (points[:, 0] - lon0) * _M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    y = (points[:, 1] - lat0) * _M_PER_DEG_LAT
    look = math.radians(heading + 90.0)              # sensor -> target, right-looking
    s = x * math.sin(look) + y * math.cos(look)      # across-track ground distance, near -> far
    frac = (s - s.min()) / max(float(np.ptp(s)), 1.0)
    near, far = incidence_deg
    return FootprintGeometry(
        heading_deg=heading, pass_direction=orbit_state.capitalize(),
        geolocation={"longitude": points[:, 0], "latitude": points[:, 1],
                     "incidence": _incidence_profile(frac, near, far)},
    )

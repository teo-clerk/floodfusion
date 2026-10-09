r"""Range--Doppler terrain correction, backward-geocoded, with area-based RTC (Route A core).

``docs/04`` section 5. Production ingest uses a pre-processed RTC collection (Route B,
``docs/04`` section 9); this module implements the chain that collection already ran, so
``P1-M1``'s comprehension question — *point at a steep slope facing the sensor and explain
what happened to those pixels* — is answered by code, not by an API call.

The geometry
------------
Sentinel-1 is processed to **zero-Doppler**: a ground target :math:`\mathbf P` is imaged at
the azimuth time :math:`t^\star` solving

.. math:: f(t) = \mathbf V(t)\cdot[\mathbf P - \mathbf S(t)] = 0,

closest approach, and the slant range is :math:`R = \|\mathbf P - \mathbf S(t^\star)\|`.
:func:`zero_doppler_time` solves it by Newton's method on the cubic-Hermite-interpolated
orbit state vectors, with :math:`f'(t) = \mathbf A\cdot(\mathbf P - \mathbf S) - \|\mathbf V\|^2`.

Backward geocoding
------------------
:func:`terrain_correct` iterates over the **output** grid: every cell's ECEF position (from
its map coordinates and the DEM height) is mapped to image coordinates via
:math:`(t^\star, R)`, and the image is sampled there — bilinearly, in **linear power**.
Each output cell is filled exactly once by exactly one interpolation; forward projection
would leave holes where ground-range sampling stretched and pile-ups where it compressed.

Radiometric terrain correction (Small 2011)
--------------------------------------------
Geocoding moves pixels; it does not fix brightness. A slope facing the sensor is
foreshortened, so each radar bin collects backscatter from more ground and is brighter. The
area-based correction normalises by the area that actually contributed to each bin:

.. math:: \gamma^0_T = \beta^0\,\frac{A_\beta}{A_\gamma},

with :math:`A_\gamma` the sum over all DEM facets imaged into the bin of their area
projected onto the plane perpendicular to the look direction (each facet's true area times
:math:`\hat{\mathbf n}\cdot\hat{\mathbf l}`; facets facing away contribute nothing), and
:math:`A_\beta` the bin's area in the slant-range plane. Several facets can share one bin —
the general case in rugged terrain, which the naive :math:`1/\cos\theta_{\mathrm{loc}}`
cannot handle. On flat terrain it reduces to :math:`\gamma^0 = \beta^0\tan\theta`.

**And the result is still untrustworthy in layover**: where several distinct ground
patches share one range bin, the receiver has already summed their returns; dividing by an
area redistributes the sum, it cannot unmix it. Layover is destroyed information to mask
(``docs/03`` section 5.4), not a radiometric error to correct.

Heights must be **ellipsoidal**: Copernicus GLO-30 is relative to the EGM2008 geoid, and a
50 m geoid/ellipsoid confusion is a ~62 m horizontal error at 39 degrees incidence — pass
the geoid undulation (``docs/03`` section 2.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "OrbitStateVectors",
    "SRGRGeometry",
    "SlantRangeGeometry",
    "ecef_to_geodetic",
    "geodetic_to_ecef",
    "terrain_correct",
    "zero_doppler_time",
]

#: Targets per chunk in the zero-Doppler seed search (memory ~ chunk x n_state_vectors x 8 B).
_ZD_CHUNK = 262_144
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
SPEED_OF_LIGHT = 299_792_458.0


def geodetic_to_ecef(lat_deg: Any, lon_deg: Any, height_m: Any) -> np.ndarray:
    """WGS84 geodetic coordinates (ellipsoidal height) to ECEF metres, shape ``(..., 3)``."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(height_m, dtype=np.float64)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * np.sin(lat)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF metres to WGS84 (lat, lon, h); Bowring's iteration, sub-millimetre."""
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def _enu_basis(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local East, North, Up unit vectors in ECEF, each ``(..., 3)``."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    return east, north, up


@dataclass(frozen=True)
class OrbitStateVectors:
    """Orbit state vectors (seconds since a reference epoch; ECEF metres, m/s).

    Interpolated with a cubic Hermite per interval, which uses the velocities as the
    derivatives — position, velocity and acceleration are consistent, as Newton needs.
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=np.float64)
        p = np.asarray(self.positions, dtype=np.float64)
        v = np.asarray(self.velocities, dtype=np.float64)
        if t.ndim != 1 or t.size < 2 or p.shape != (t.size, 3) or v.shape != (t.size, 3):
            raise ValueError("need >= 2 state vectors: times (N,), positions and velocities (N, 3)")
        if np.any(np.diff(t) <= 0):
            raise ValueError("state-vector times must be strictly increasing")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "positions", p)
        object.__setattr__(self, "velocities", v)

    def interpolate(self, t: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity and acceleration at ``t`` (any shape); each ``(..., 3)``."""
        t = np.asarray(t, dtype=np.float64)
        i = np.clip(np.searchsorted(self.times, t) - 1, 0, self.times.size - 2)
        h = (self.times[i + 1] - self.times[i])[..., None]
        s = ((t - self.times[i])[..., None]) / h
        p0, p1 = self.positions[i], self.positions[i + 1]
        v0, v1 = self.velocities[i] * h, self.velocities[i + 1] * h
        s2, s3 = s * s, s * s * s
        pos = (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * v1
        vel = ((6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * v0 + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * v1) / h
        acc = ((12 * s - 6) * p0 + (6 * s - 4) * v0 + (-12 * s + 6) * p1 + (6 * s - 2) * v1) / (h * h)
        return pos, vel, acc


def zero_doppler_time(
    orbit: OrbitStateVectors,
    targets: np.ndarray,
    *,
    tol_s: float = 1e-7,
    max_iter: int = 30,
) -> np.ndarray:
    r"""Azimuth time of closest approach for every target (ECEF, ``(..., 3)``).

    Newton on :math:`f(t) = \mathbf V\cdot(\mathbf P - \mathbf S)`, started from the state
    vector nearest each target; converges quadratically (2--4 iterations to 1e-7 s).
    """
    p = np.asarray(targets, dtype=np.float64)
    flat = p.reshape(-1, 3)
    # Seed from the nearest state vector, in chunks: |p - s|^2 = |p|^2 - 2 p.s + |s|^2 needs an
    # (N, M) array per chunk instead of (N, M, 3) for the whole grid.
    s_sq = np.einsum("ij,ij->i", orbit.positions, orbit.positions)
    nearest = np.empty(flat.shape[0], dtype=np.int64)
    for start in range(0, flat.shape[0], _ZD_CHUNK):
        block = flat[start:start + _ZD_CHUNK]
        nearest[start:start + _ZD_CHUNK] = np.argmin(s_sq[None, :] - 2.0 * block @ orbit.positions.T, axis=1)
    t = orbit.times[nearest].astype(np.float64)
    for _ in range(max_iter):
        s, v, a = orbit.interpolate(t)
        d = flat - s
        f = np.einsum("ij,ij->i", v, d)
        df = np.einsum("ij,ij->i", a, d) - np.einsum("ij,ij->i", v, v)
        step = f / df
        t = t - step
        if np.max(np.abs(step)) < tol_s:
            break
    return t.reshape(p.shape[:-1])


@dataclass(frozen=True)
class SlantRangeGeometry:
    """Image coordinates from (azimuth time, slant range) for a slant-range image (SLC-like)."""

    first_line_time: float
    line_interval: float
    near_range_m: float
    range_spacing_m: float

    def to_image(self, t: np.ndarray, slant_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        line = (np.asarray(t) - self.first_line_time) / self.line_interval
        pixel = (np.asarray(slant_range) - self.near_range_m) / self.range_spacing_m
        return line, pixel

    def slant_range_spacing(self, pixel: np.ndarray, t: np.ndarray | float = 0.0) -> np.ndarray:
        return np.full(np.shape(pixel), self.range_spacing_m)


@dataclass(frozen=True)
class SRGRGeometry:
    r"""GRD image coordinates: pixels equally spaced in **ground range on the ellipsoid**.

    The annotation's ``coordinateConversionList`` gives, at a series of azimuth times
    :math:`t_k`, two polynomials (ESA convention, Sentinel-1 Product Specification):

    - ``srgrCoefficients``: ground range from slant range,
      :math:`g = \sum_j c_j (R - R_{0,k})^j` with origin ``sr0``;
    - ``grsrCoefficients``: slant range from ground range,
      :math:`R = \sum_j d_j (g - g_{0,k})^j` with origin ``gr0``.

    The coefficients **drift along track** (orbit altitude and Earth radius change; tens of
    metres of ground range per second on a real ascending frame), so the ground range is
    evaluated with the two polynomials bracketing each azimuth time and interpolated
    linearly in time — never with one mid-scene polynomial.

    ``poly_times`` share the time reference of the orbit and ``first_line_time``.
    """

    first_line_time: float
    line_interval: float
    ground_spacing_m: float
    poly_times: tuple[float, ...]
    sr0_m: tuple[float, ...]
    srgr_coefficients: tuple[tuple[float, ...], ...]
    gr0_m: tuple[float, ...]
    grsr_coefficients: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.poly_times)
        if n == 0 or not (len(self.sr0_m) == len(self.srgr_coefficients) == len(self.gr0_m)
                          == len(self.grsr_coefficients) == n):
            raise ValueError("SRGR/GRSR polynomial lists must be non-empty and of one length")
        if n > 1 and np.any(np.diff(self.poly_times) <= 0):
            raise ValueError("SRGR polynomial times must be strictly increasing")

    def _blend(self, t: np.ndarray, x: np.ndarray, origins: tuple[float, ...],
               coefficients: tuple[tuple[float, ...], ...], *, derivative: bool) -> np.ndarray:
        """Evaluate the time-bracketing polynomials at ``x - origin`` and blend linearly in t."""
        poly = np.polynomial.polynomial
        t = np.broadcast_to(np.asarray(t, dtype=np.float64), np.shape(x))
        times = np.asarray(self.poly_times)

        def evaluate(k: int, values: np.ndarray) -> np.ndarray:
            c = np.asarray(coefficients[k])
            return poly.polyval(values - origins[k], poly.polyder(c) if derivative else c)

        if times.size == 1:
            return evaluate(0, np.asarray(x, dtype=np.float64))
        idx = np.clip(np.searchsorted(times, t) - 1, 0, times.size - 2)
        w = np.clip((t - times[idx]) / (times[idx + 1] - times[idx]), 0.0, 1.0)
        out = np.empty(np.shape(x), dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        for k in np.unique(idx):
            sel = idx == k
            out[sel] = (1.0 - w[sel]) * evaluate(k, x[sel]) + w[sel] * evaluate(k + 1, x[sel])
        return out

    def to_image(self, t: np.ndarray, slant_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ground = self._blend(t, np.asarray(slant_range, dtype=np.float64), self.sr0_m,
                             self.srgr_coefficients, derivative=False)
        line = (np.asarray(t) - self.first_line_time) / self.line_interval
        return line, ground / self.ground_spacing_m

    def slant_range_spacing(self, pixel: np.ndarray, t: np.ndarray | float = 0.0) -> np.ndarray:
        r"""Slant-range extent of one ground-range pixel: :math:`|dR/dg|\,\Delta g`."""
        ground = np.asarray(pixel, dtype=np.float64) * self.ground_spacing_m
        dr_dg = self._blend(t, ground, self.gr0_m, self.grsr_coefficients, derivative=True)
        return np.abs(dr_dg) * self.ground_spacing_m


def _bilinear(image: np.ndarray, line: np.ndarray, pixel: np.ndarray, nodata: float) -> np.ndarray:
    h, w = image.shape
    inside = (line >= 0) & (line <= h - 1) & (pixel >= 0) & (pixel <= w - 1)
    out = np.full(line.shape, nodata, dtype=np.float64)
    li, pi = line[inside], pixel[inside]
    l0 = np.clip(np.floor(li).astype(np.int64), 0, h - 2)
    p0 = np.clip(np.floor(pi).astype(np.int64), 0, w - 2)
    fl, fp = li - l0, pi - p0
    out[inside] = (
        image[l0, p0] * (1 - fl) * (1 - fp) + image[l0 + 1, p0] * fl * (1 - fp)
        + image[l0, p0 + 1] * (1 - fl) * fp + image[l0 + 1, p0 + 1] * fl * fp
    )
    return out


def _splat(shape: tuple[int, int], line: np.ndarray, pixel: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Accumulate ``values`` into a radar-geometry grid with bilinear weights (adjoint of _bilinear)."""
    h, w = shape
    acc = np.zeros(shape, dtype=np.float64)
    ok = (line >= 0) & (line <= h - 1) & (pixel >= 0) & (pixel <= w - 1) & np.isfinite(values)
    li, pi, v = line[ok], pixel[ok], values[ok]
    l0 = np.clip(np.floor(li).astype(np.int64), 0, h - 2)
    p0 = np.clip(np.floor(pi).astype(np.int64), 0, w - 2)
    fl, fp = li - l0, pi - p0
    np.add.at(acc, (l0, p0), v * (1 - fl) * (1 - fp))
    np.add.at(acc, (l0 + 1, p0), v * fl * (1 - fp))
    np.add.at(acc, (l0, p0 + 1), v * (1 - fl) * fp)
    np.add.at(acc, (l0 + 1, p0 + 1), v * fl * fp)
    return acc


def terrain_correct(
    image: np.ndarray,
    geometry: SlantRangeGeometry | SRGRGeometry,
    orbit: OrbitStateVectors,
    dem_on_grid: np.ndarray,
    grid_transform: tuple[float, float, float, float, float, float],
    grid_crs: Any,
    *,
    rtc: str | None = "area",
    geoid_undulation_m: float | np.ndarray = 0.0,
    nodata: float = np.nan,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    r"""Backward-geocode a radar-geometry image onto the analysis grid, with optional RTC.

    Parameters
    ----------
    image
        Calibrated **linear-power** image in radar geometry: :math:`\beta^0` for
        ``rtc="area"`` (the area normalisation starts from slant-plane brightness).
    geometry, orbit
        Image timing/range model and the orbit state vectors.
    dem_on_grid
        Heights on the **output** grid (metres; orthometric if ``geoid_undulation_m`` is
        given, otherwise treated as ellipsoidal).
    grid_transform, grid_crs
        The analysis grid (rasterio-order affine, any pyproj-understood CRS).
    rtc
        ``"area"`` — Small (2011) area-based :math:`\gamma^0_T`; ``"cos"`` — the naive
        :math:`\beta^0\sin\theta_{\mathrm{ell}}/\cos\theta_{\mathrm{loc}}`; ``None`` —
        geocoding only (values unchanged).

    Returns
    -------
    numpy.ndarray
        The corrected image on the output grid, ``nodata`` outside the swath. With
        ``return_diagnostics`` also the per-cell ``line``, ``pixel``, ``theta_ell_deg``,
        ``theta_loc_deg`` and ``illuminated`` arrays.
    """
    from pyproj import CRS, Transformer

    if rtc not in {"area", "cos", None}:
        raise ValueError(f"rtc must be 'area', 'cos' or None, got {rtc!r}")
    img = np.asarray(image, dtype=np.float64)
    dem = np.asarray(dem_on_grid, dtype=np.float64)
    a, b, c, d, e, f = grid_transform
    if b != 0 or d != 0:
        raise ValueError("terrain_correct expects a north-up grid")
    rows, cols = np.mgrid[0:dem.shape[0], 0:dem.shape[1]]
    x = c + (cols + 0.5) * a
    y = f + (rows + 0.5) * e
    to_geo = Transformer.from_crs(CRS.from_user_input(grid_crs), CRS.from_epsg(4326), always_xy=True)
    lon, lat = to_geo.transform(x, y)
    h_ell = dem + np.asarray(geoid_undulation_m, dtype=np.float64)

    p = geodetic_to_ecef(lat, lon, h_ell)
    t_star = zero_doppler_time(orbit, p)
    s, v, acc = orbit.interpolate(t_star)
    look = s - p
    slant = np.linalg.norm(look, axis=-1)
    look_unit = look / slant[..., None]
    line, pixel = geometry.to_image(t_star, slant)
    values = _bilinear(img, line, pixel, nodata)

    east, north, up = _enu_basis(lat, lon)
    cos_ell = np.einsum("...k,...k->...", look_unit, up)
    # Facet normal from the DEM gradient (metric grid spacing |a|, |e|), in ECEF.
    dzdx = np.gradient(dem, abs(a), axis=1)
    dzdy = -np.gradient(dem, abs(e), axis=0)  # rows run south: flip to a northward derivative
    normal_enu = np.stack([-dzdx, -dzdy, np.ones_like(dem)], axis=-1)
    normal_enu /= np.linalg.norm(normal_enu, axis=-1, keepdims=True)
    normal = normal_enu[..., :1] * east + normal_enu[..., 1:2] * north + normal_enu[..., 2:] * up
    cos_loc = np.einsum("...k,...k->...", look_unit, normal)
    illuminated = cos_loc > 0

    if rtc == "cos":
        sin_ell = np.sqrt(np.clip(1.0 - cos_ell**2, 0.0, 1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(illuminated, values * sin_ell / cos_loc, nodata)
    elif rtc == "area":
        facet_area = abs(a * e) / normal_enu[..., 2]  # true surface area of each DEM cell
        a_gamma_facet = np.where(illuminated, facet_area * cos_loc, 0.0)
        a_gamma_bins = _splat(img.shape, line, pixel, a_gamma_facet)
        a_gamma = _bilinear(a_gamma_bins, line, pixel, np.nan)
        # A_beta: the radar bin in the slant plane — azimuth spacing AT THE TARGET times the
        # slant-range spacing. Azimuth ground speed of the zero-Doppler plane at P is
        # (|V|^2 - A.(P - S)) / |V| (exact for straight and curved orbits alike).
        vnorm = np.linalg.norm(v, axis=-1)
        az_speed = (vnorm**2 - np.einsum("...k,...k->...", acc, p - s)) / vnorm
        line_interval = geometry.line_interval
        a_beta = az_speed * line_interval * geometry.slant_range_spacing(pixel, t_star)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(np.isfinite(values) & (a_gamma > 0), values * a_beta / a_gamma, nodata)

    if return_diagnostics:
        return values, {
            "line": line, "pixel": pixel,
            "theta_ell_deg": np.degrees(np.arccos(np.clip(cos_ell, -1, 1))),
            "theta_loc_deg": np.degrees(np.arccos(np.clip(cos_loc, -1, 1))),
            "illuminated": illuminated, "slant_range": slant, "azimuth_time": t_star,
        }
    return values

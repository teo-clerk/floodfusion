r"""Sentinel-1 product annotation (``annotation/s1?-iw-grd-*.xml``): geometry for both routes.

The pre-processed RTC collection (Route B) delivers gamma0 but not the acquisition
geometry the terrain masks need. The matching GRD product's annotation supplies it:

- ``platformHeading`` — the per-scene heading, so the look azimuth is computed, not
  assumed: :math:`\phi = \text{heading} - 90^\circ` is the bearing **from target to
  sensor** for a right-looking SAR (``docs/03`` section 5.1; a 20 deg error rotates every
  shadow, section 5.3);
- the **geolocation grid** — tie points with latitude, longitude, height and incidence
  angle, from which the per-pixel incidence field is interpolated (the 29--46 deg
  across-swath variation a scalar mid-swath value discards, ``docs/01`` 4.2);
- the **orbit state vectors**, image timing and **SRGR polynomials** — everything
  :func:`floodfusion.data.geocoding.terrain_correct` needs for Route A.

Parsed with the standard library (``xml.etree``: expat does not resolve external entities
and, from expat 2.4, bounds entity expansion). S1 annotations never carry a DTD, so any
``<!DOCTYPE``/``<!ENTITY`` is refused outright, and inputs above :data:`MAX_ANNOTATION_BYTES`
are refused before parsing.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.data.geocoding import OrbitStateVectors, SRGRGeometry

__all__ = ["MAX_ANNOTATION_BYTES", "S1Annotation", "parse_annotation"]

#: A real GRD product annotation is ~2 MB; anything far larger is not one.
MAX_ANNOTATION_BYTES = 64 * 1024 * 1024


def _time(text: str) -> datetime:
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _float(node: ET.Element, path: str) -> float:
    found = node.find(path)
    if found is None or found.text is None:
        raise ValueError(f"annotation is missing <{path}>")
    return float(found.text)


def _text(node: ET.Element, path: str, default: str | None = None) -> str:
    found = node.find(path)
    if found is None or found.text is None:
        if default is not None:
            return default
        raise ValueError(f"annotation is missing <{path}>")
    return found.text.strip()


@dataclass(frozen=True)
class S1Annotation:
    """The parts of a Sentinel-1 product annotation this project uses."""

    mission: str
    mode: str
    polarisation: str
    pass_direction: str
    heading_deg: float
    radar_frequency_hz: float
    first_line_time: datetime
    azimuth_time_interval: float
    slant_range_time_s: float
    range_pixel_spacing_m: float
    n_samples: int
    n_lines: int
    orbit_times: tuple[datetime, ...]
    orbit_positions: np.ndarray        # (N, 3) ECEF m
    orbit_velocities: np.ndarray       # (N, 3) ECEF m/s
    geolocation: dict[str, np.ndarray]  # line, pixel, latitude, longitude, height, incidence
    #: Per azimuth time: (time, sr0, srgrCoefficients, gr0, grsrCoefficients).
    srgr: tuple[tuple[datetime, float, tuple[float, ...], float, tuple[float, ...]], ...]

    @property
    def look_azimuth_deg(self) -> float:
        """Bearing from target to sensor (right-looking): heading - 90 deg, in [0, 360)."""
        return float((self.heading_deg - 90.0) % 360.0)

    @property
    def orbit_state(self) -> str:
        return self.pass_direction.lower()

    @property
    def near_range_m(self) -> float:
        """One-way slant range of the first pixel: c * two-way time / 2."""
        return 299_792_458.0 * self.slant_range_time_s / 2.0

    def orbit_state_vectors(self, reference: datetime | None = None) -> OrbitStateVectors:
        """State vectors with times in seconds since ``reference`` (default: first line)."""
        ref = reference or self.first_line_time
        times = np.array([(t - ref).total_seconds() for t in self.orbit_times])
        return OrbitStateVectors(times, self.orbit_positions, self.orbit_velocities)

    def srgr_geometry(self, reference: datetime | None = None) -> SRGRGeometry:
        """GRD image model with every SRGR/GRSR polynomial, times relative to ``reference``."""
        if not self.srgr:
            raise ValueError("annotation carries no SRGR coefficients (not a GRD product?)")
        ref = reference or self.first_line_time
        return SRGRGeometry(
            first_line_time=(self.first_line_time - ref).total_seconds(),
            line_interval=self.azimuth_time_interval,
            ground_spacing_m=self.range_pixel_spacing_m,
            poly_times=tuple((entry[0] - ref).total_seconds() for entry in self.srgr),
            sr0_m=tuple(entry[1] for entry in self.srgr),
            srgr_coefficients=tuple(entry[2] for entry in self.srgr),
            gr0_m=tuple(entry[3] for entry in self.srgr),
            grsr_coefficients=tuple(entry[4] for entry in self.srgr),
        )

    def incidence_on(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Ellipsoid incidence angle (deg) interpolated from the geolocation tie points.

        Linear interpolation over the tie-point triangulation; nearest value outside it.
        """
        from scipy.interpolate import griddata

        pts = np.column_stack([self.geolocation["longitude"], self.geolocation["latitude"]])
        values = self.geolocation["incidence"]
        q = np.column_stack([np.ravel(lon), np.ravel(lat)])
        out = griddata(pts, values, q, method="linear")
        missing = ~np.isfinite(out)
        if missing.any():
            out[missing] = griddata(pts, values, q[missing], method="nearest")
        return out.reshape(np.shape(lon))


def parse_annotation(source: str | bytes | Path) -> S1Annotation:
    """Parse a Sentinel-1 L1 product annotation (XML text, bytes or a file path)."""
    if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith("<")):
        path = Path(source)
        if path.stat().st_size > MAX_ANNOTATION_BYTES:
            raise ValueError(f"{path}: larger than {MAX_ANNOTATION_BYTES} bytes — not an S1 annotation")
        data = path.read_bytes()
    else:
        data = source.encode("utf-8") if isinstance(source, str) else bytes(source)
    if len(data) > MAX_ANNOTATION_BYTES:
        raise ValueError(f"annotation larger than {MAX_ANNOTATION_BYTES} bytes")
    head = data[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in data.upper():
        raise ValueError("refusing an annotation with a DTD/entity declaration (S1 annotations have none)")
    root = ET.fromstring(data)
    header = root.find("adsHeader")
    info = root.find("generalAnnotation/productInformation")
    image = root.find("imageAnnotation/imageInformation")
    if header is None or info is None or image is None:
        raise ValueError("not a Sentinel-1 product annotation (adsHeader/productInformation/imageInformation missing)")

    orbits = root.findall("generalAnnotation/orbitList/orbit")
    if len(orbits) < 2:
        raise ValueError("annotation needs at least two orbit state vectors")
    orbit_times = tuple(_time(_text(o, "time")) for o in orbits)
    positions = np.array([[_float(o, f"position/{k}") for k in "xyz"] for o in orbits])
    velocities = np.array([[_float(o, f"velocity/{k}") for k in "xyz"] for o in orbits])

    grid_points = root.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")
    geolocation: dict[str, Any] = {k: [] for k in ("line", "pixel", "latitude", "longitude", "height", "incidence")}
    for gp in grid_points:
        geolocation["line"].append(_float(gp, "line"))
        geolocation["pixel"].append(_float(gp, "pixel"))
        geolocation["latitude"].append(_float(gp, "latitude"))
        geolocation["longitude"].append(_float(gp, "longitude"))
        geolocation["height"].append(_float(gp, "height"))
        geolocation["incidence"].append(_float(gp, "incidenceAngle"))
    geolocation = {k: np.asarray(v, dtype=np.float64) for k, v in geolocation.items()}

    srgr = []
    for cc in root.findall("coordinateConversion/coordinateConversionList/coordinateConversion"):
        srgr.append((
            _time(_text(cc, "azimuthTime")),
            _float(cc, "sr0"),
            tuple(float(v) for v in _text(cc, "srgrCoefficients").split()),
            _float(cc, "gr0"),
            tuple(float(v) for v in _text(cc, "grsrCoefficients").split()),
        ))

    return S1Annotation(
        mission=_text(header, "missionId", "S1"),
        mode=_text(header, "mode", "IW"),
        polarisation=_text(header, "polarisation", ""),
        pass_direction=_text(info, "pass"),
        heading_deg=_float(info, "platformHeading"),
        radar_frequency_hz=_float(info, "radarFrequency"),
        first_line_time=_time(_text(image, "productFirstLineUtcTime")),
        azimuth_time_interval=_float(image, "azimuthTimeInterval"),
        slant_range_time_s=_float(image, "slantRangeTime"),
        range_pixel_spacing_m=_float(image, "rangePixelSpacing"),
        n_samples=int(_float(image, "numberOfSamples")),
        n_lines=int(_float(image, "numberOfLines")),
        orbit_times=orbit_times,
        orbit_positions=positions,
        orbit_velocities=velocities,
        geolocation=geolocation,
        srgr=tuple(srgr),
    )

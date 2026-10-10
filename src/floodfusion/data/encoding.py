r"""The on-disk chip format: one 16-band ``int16`` raster per event, and its codec.

Every layer the pipeline needs lives in **one** file on **one** grid, so a chip is a single
windowed read of 16 bands (``docs/04`` section 2.3: one call reading N bands is far
cheaper than N calls). Bands 0--11 are the model inputs in the frozen order of
:class:`floodfusion.config.Channel`; bands 12--15 are carried alongside but are **never**
model inputs (``docs/05`` section 4.1).

==== ===================== =================================================== =========
band content                stored as                                           nodata
==== ===================== =================================================== =========
0-3  gamma0 VV/VH post/pre  round(dB x 100)                                     -32768
4    Delta VV (post - pre)  round(dB x 100)                                     -32768
5-7  S2 B3, B8, B11         DN harmonised to baseline >= 04.00: rho*10000+1000  -32768
8    MNDWI                  round(MNDWI x 10000)                                -32768
9    slope                  round(deg x 100)                                    -32768
10   log(1 + HAND)          round(log1p(m) x 1000)                              -32768
11   local incidence        round(deg x 100)                                    -32768
12   label                  0 / 1                                               -32768
13   validity flags         ValidityFlag bits (uint8 value)                     -32768
14   optical quality        round(q x 10000), q in [0, 1]                       0 (none)
15   land cover             ESA WorldCover class code (0 = unknown)             0
==== ===================== =================================================== =========

**Why int16.** 0.01 dB quantisation is far finer than the 2.17 dB speckle noise, and the
file is half the size of ``float32`` (``docs/04`` section 8.1). The nodata sentinel
-32768 decodes to -327.68 dB, outside any physical value; it is routed into the validity
mask **before** any arithmetic, so it can never reach the network as a finite number.

**Why harmonised S2 DN.** From processing baseline 04.00, L2A products carry
``BOA_ADD_OFFSET = -1000``; earlier products do not. Ingest converts every product to the
post-04.00 convention using *its own* metadata, so one frozen
:class:`~floodfusion.config.NormalizationStats` (offset -1000, scale 10000) decodes the
whole archive correctly (``docs/05`` section 7.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np

from floodfusion.config import (
    Channel,
    GridConfig,
    NormalizationStats,
    ValidityFlag,
)

__all__ = [
    "N_CHIP_BANDS",
    "ChipBand",
    "PhysicalChip",
    "decode_physical",
    "encode_stack",
    "harmonise_s2_dn",
    "mndwi_from_reflectance",
    "normalise",
]


class ChipBand(IntEnum):
    """Band index in the stored 16-band raster (bands 0-11 mirror :class:`Channel`)."""

    VV_POST = Channel.VV_POST
    VH_POST = Channel.VH_POST
    VV_PRE = Channel.VV_PRE
    VH_PRE = Channel.VH_PRE
    DELTA_VV = Channel.DELTA_VV
    B3_GREEN = Channel.B3_GREEN
    B8_NIR = Channel.B8_NIR
    B11_SWIR1 = Channel.B11_SWIR1
    MNDWI = Channel.MNDWI
    SLOPE = Channel.SLOPE
    LOG1P_HAND = Channel.LOG1P_HAND
    LOCAL_INCIDENCE = Channel.LOCAL_INCIDENCE
    LABEL = 12
    VALIDITY = 13
    OPTICAL_QUALITY = 14
    LANDCOVER = 15


N_CHIP_BANDS: Final[int] = 16
INDEX_SCALE: Final[float] = 10_000.0
ANGLE_SCALE: Final[float] = 100.0
LOG_HAND_SCALE: Final[float] = 1_000.0
QUALITY_SCALE: Final[float] = 10_000.0
_INT16_MAX: Final[int] = 32767
_INT16_MIN_VALID: Final[int] = -32767  # -32768 is reserved for nodata

#: WorldCover codes used by the caveat layers (docs/00 section 5.3, R4).
WORLDCOVER_TREE: Final[int] = 10
WORLDCOVER_BUILT_UP: Final[int] = 50


@dataclass(frozen=True)
class PhysicalChip:
    """Decoded layers in physical units. NaN marks nodata in every float layer."""

    gamma0_db: np.ndarray        # (5, H, W): VV post, VH post, VV pre, VH pre, Delta VV
    reflectance: np.ndarray      # (3, H, W): B3, B8, B11 surface reflectance
    mndwi: np.ndarray            # (H, W) in [-1, 1]
    slope_deg: np.ndarray        # (H, W)
    log1p_hand: np.ndarray       # (H, W) = log(1 + HAND / m)
    local_incidence_deg: np.ndarray  # (H, W)
    label: np.ndarray            # (H, W) float: 0, 1 or NaN (no label)
    validity: np.ndarray         # (H, W) uint8 ValidityFlag bits (NODATA set for sentinels)
    optical_quality: np.ndarray  # (H, W) in [0, 1]
    landcover: np.ndarray        # (H, W) uint8 WorldCover code, 0 = unknown

    @property
    def hand_m(self) -> np.ndarray:
        return np.expm1(self.log1p_hand)


def mndwi_from_reflectance(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """MNDWI = (G - SWIR1) / (G + SWIR1) (Xu 2006); NaN where undefined."""
    g = np.asarray(green, dtype=np.float64)
    s = np.asarray(swir1, dtype=np.float64)
    denom = g + s
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (g - s) / denom
    out[~np.isfinite(out) | (denom <= 0)] = np.nan
    return np.clip(out, -1.0, 1.0)


def harmonise_s2_dn(
    dn: np.ndarray,
    *,
    product_offset: float,
    product_quantification: float,
    norm: NormalizationStats,
) -> np.ndarray:
    """Convert product DN to the stored convention (``rho * scale - offset`` of ``norm``).

    ``rho = (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE`` with the **product's own**
    metadata; DN 0 is the L2A nodata value and becomes NaN.
    """
    dn = np.asarray(dn, dtype=np.float64)
    rho = (dn + product_offset) / product_quantification
    rho[dn <= 0] = np.nan
    return rho * norm.optical_scale - norm.optical_offset


def _encode(values: np.ndarray, scale: float, nodata: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64) * scale
    out = np.full(v.shape, nodata, dtype=np.int16)
    ok = np.isfinite(v)
    out[ok] = np.clip(np.rint(v[ok]), _INT16_MIN_VALID, _INT16_MAX).astype(np.int16)
    return out


def encode_stack(
    *,
    vv_post_db: np.ndarray,
    vh_post_db: np.ndarray,
    vv_pre_db: np.ndarray,
    vh_pre_db: np.ndarray,
    reflectance: np.ndarray | None,
    slope_deg: np.ndarray,
    hand_m: np.ndarray,
    local_incidence_deg: np.ndarray,
    label: np.ndarray | None,
    validity: np.ndarray,
    optical_quality: np.ndarray | None,
    landcover: np.ndarray | None,
    grid_cfg: GridConfig,
    norm: NormalizationStats,
) -> np.ndarray:
    """Encode physical layers into the ``(16, H, W)`` ``int16`` stack.

    ``reflectance`` is ``(3, H, W)`` surface reflectance (B3, B8, B11) or ``None`` when
    there is no optical acquisition; NaN marks nodata everywhere. ``label`` uses 0/1 with
    NaN (or ``None``) for "no reference label". Delta VV and MNDWI are derived here so they
    can never disagree with the bands they come from.
    """
    nodata = int(grid_cfg.nodata_int16)
    db_scale = float(grid_cfg.db_scale)
    shape = np.shape(vv_post_db)
    stack = np.full((N_CHIP_BANDS, *shape), nodata, dtype=np.int16)
    stack[ChipBand.VV_POST] = _encode(vv_post_db, db_scale, nodata)
    stack[ChipBand.VH_POST] = _encode(vh_post_db, db_scale, nodata)
    stack[ChipBand.VV_PRE] = _encode(vv_pre_db, db_scale, nodata)
    stack[ChipBand.VH_PRE] = _encode(vh_pre_db, db_scale, nodata)
    stack[ChipBand.DELTA_VV] = _encode(np.asarray(vv_post_db) - np.asarray(vv_pre_db), db_scale, nodata)
    if reflectance is not None:
        rho = np.asarray(reflectance, dtype=np.float64)
        if rho.shape != (3, *shape):
            raise ValueError(f"reflectance must be (3, {shape[0]}, {shape[1]}), got {rho.shape}")
        dn = rho * norm.optical_scale - norm.optical_offset
        for band, layer in zip((ChipBand.B3_GREEN, ChipBand.B8_NIR, ChipBand.B11_SWIR1), dn):
            stack[band] = _encode(layer, 1.0, nodata)
        stack[ChipBand.MNDWI] = _encode(mndwi_from_reflectance(rho[0], rho[2]), INDEX_SCALE, nodata)
    stack[ChipBand.SLOPE] = _encode(slope_deg, ANGLE_SCALE, nodata)
    hand = np.asarray(hand_m, dtype=np.float64)
    stack[ChipBand.LOG1P_HAND] = _encode(np.log1p(np.maximum(hand, 0.0)), LOG_HAND_SCALE, nodata)
    stack[ChipBand.LOCAL_INCIDENCE] = _encode(local_incidence_deg, ANGLE_SCALE, nodata)
    if label is not None:
        stack[ChipBand.LABEL] = _encode(np.asarray(label, dtype=np.float64), 1.0, nodata)
    stack[ChipBand.VALIDITY] = np.asarray(validity, dtype=np.uint8).astype(np.int16)
    if optical_quality is not None:
        q = np.nan_to_num(np.clip(np.asarray(optical_quality, dtype=np.float64), 0.0, 1.0), nan=0.0)
        stack[ChipBand.OPTICAL_QUALITY] = np.rint(q * QUALITY_SCALE).astype(np.int16)
    else:
        stack[ChipBand.OPTICAL_QUALITY] = 0
    stack[ChipBand.LANDCOVER] = 0 if landcover is None else np.asarray(landcover, dtype=np.uint8).astype(np.int16)
    return stack


def decode_physical(raw: np.ndarray, grid_cfg: GridConfig, norm: NormalizationStats) -> PhysicalChip:
    """Decode a stored ``(16, H, W)`` stack into physical units.

    Sentinels are detected **before** any arithmetic and become NaN; a sentinel in any SAR
    band or in the validity band sets :attr:`ValidityFlag.NODATA`, so the pixel leaves both
    the loss and the metric.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[0] != N_CHIP_BANDS:
        raise ValueError(f"expected a ({N_CHIP_BANDS}, H, W) stack, got {raw.shape}")
    nodata = int(grid_cfg.nodata_int16)
    sentinel = raw == nodata

    def scaled(band: int, scale: float) -> np.ndarray:
        out = raw[band].astype(np.float64) / scale
        out[sentinel[band]] = np.nan
        return out

    gamma0 = np.stack([scaled(int(b), grid_cfg.db_scale) for b in range(5)])
    rho = np.stack([
        (raw[b].astype(np.float64) + norm.optical_offset) / norm.optical_scale
        for b in (ChipBand.B3_GREEN, ChipBand.B8_NIR, ChipBand.B11_SWIR1)
    ])
    rho[sentinel[ChipBand.B3_GREEN:ChipBand.B11_SWIR1 + 1]] = np.nan
    validity = np.where(sentinel[ChipBand.VALIDITY], np.uint8(ValidityFlag.NODATA),
                        raw[ChipBand.VALIDITY].astype(np.uint8)).astype(np.uint8)
    sar_nodata = sentinel[ChipBand.VV_POST:ChipBand.DELTA_VV + 1].any(axis=0)
    validity = validity | np.where(sar_nodata, np.uint8(ValidityFlag.NODATA), np.uint8(0))
    label = raw[ChipBand.LABEL].astype(np.float64)
    label[sentinel[ChipBand.LABEL]] = np.nan
    quality = np.clip(raw[ChipBand.OPTICAL_QUALITY].astype(np.float64) / QUALITY_SCALE, 0.0, 1.0)
    quality[sentinel[ChipBand.OPTICAL_QUALITY]] = 0.0
    landcover = np.where(sentinel[ChipBand.LANDCOVER], 0, raw[ChipBand.LANDCOVER]).astype(np.uint8)
    return PhysicalChip(
        gamma0_db=gamma0,
        reflectance=rho,
        mndwi=np.clip(scaled(ChipBand.MNDWI, INDEX_SCALE), -1.0, 1.0),
        slope_deg=scaled(ChipBand.SLOPE, ANGLE_SCALE),
        log1p_hand=scaled(ChipBand.LOG1P_HAND, LOG_HAND_SCALE),
        local_incidence_deg=scaled(ChipBand.LOCAL_INCIDENCE, ANGLE_SCALE),
        label=label,
        validity=validity,
        optical_quality=quality,
        landcover=landcover,
    )


def normalise(chip: PhysicalChip, norm: NormalizationStats) -> dict[str, np.ndarray]:
    """Frozen, dataset-level normalisation (``docs/05`` section 7.2) — **never per chip**.

    Returns ``float32`` arrays ``sar (5,H,W)``, ``optical (4,H,W)``, ``terrain (3,H,W)``,
    ``label (1,H,W)``, ``optical_quality (1,H,W)`` and ``bool`` ``valid (1,H,W)``.

    ============ ===============================================================
    group        rule
    ============ ===============================================================
    SAR dB       z-score with the frozen per-channel ``sar_mean_db/sar_std_db``
    reflectance  physical value (offset already applied), clipped to [0, 1]
    MNDWI        none — already in [-1, 1]
    terrain      slope/90, log1p(HAND)/log1p(50), theta_loc/90
    ============ ===============================================================

    Invalid positions are filled with **zero after normalisation** — the dataset mean for
    the z-scored channels — so they contribute nothing once masked out of the loss. No
    statistic is ever computed from the chip itself.
    """
    mean = np.asarray(norm.sar_mean_db, dtype=np.float64)[:, None, None]
    std = np.asarray(norm.sar_std_db, dtype=np.float64)[:, None, None]
    sar = (chip.gamma0_db - mean) / std
    optical = np.concatenate([np.clip(chip.reflectance, 0.0, 1.0), chip.mndwi[None]], axis=0)
    terrain = np.stack([
        chip.slope_deg / norm.slope_scale_deg,
        chip.log1p_hand / norm.hand_log_scale,
        chip.local_incidence_deg / norm.incidence_scale_deg,
    ])
    sar_ok = np.isfinite(sar).all(axis=0)
    terrain_ok = np.isfinite(terrain).all(axis=0)
    label_ok = np.isfinite(chip.label)
    valid = (chip.validity == 0) & sar_ok & terrain_ok & label_ok
    optical_ok = np.isfinite(optical).all(axis=0)
    quality = np.where(optical_ok, chip.optical_quality, 0.0)
    out = {
        "sar": np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0),
        "optical": np.nan_to_num(optical, nan=0.0, posinf=0.0, neginf=0.0),
        "terrain": np.nan_to_num(terrain, nan=0.0, posinf=0.0, neginf=0.0),
        "label": np.where(label_ok, (chip.label > 0.5).astype(np.float64), 0.0)[None],
        "optical_quality": quality[None],
    }
    result = {k: v.astype(np.float32) for k, v in out.items()}
    result["valid"] = valid[None]
    return result

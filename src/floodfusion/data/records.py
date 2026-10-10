"""Chip records and the immutable manifest (``docs/05`` section 9). Torch-free.

:class:`ChipRecord` and :func:`load_manifest` are re-exported by
:mod:`floodfusion.data.dataset`; they live here so that the chipping and splitting code —
which must run on a machine without PyTorch — can build and read records too.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ChipRecord",
    "atomic_write_bytes",
    "chip_set_sha256",
    "load_manifest",
    "load_manifest_meta",
    "manifest_meta_path",
    "manifest_sha256",
    "write_manifest",
]


@dataclass(frozen=True)
class ChipRecord:
    """One row of the chip manifest.

    Immutable by design: a dataset whose records mutate at runtime cannot be
    reproduced. See ``docs/05`` section 9 for the full field list the manifest must
    record for ``P1-M6.3``.

    Attributes
    ----------
    chip_id
        Primary key. Stable across regenerations of the dataset
        (``<event>_r<row>_c<col>`` for windows of an event stack).
    basin_id
        The split key. Splits partition **basins**, never chips
        (``docs/05`` section 6.4).
    event_id
        Flood event this chip belongs to.
    raster_path
        Path to the stacked 16-band event raster the chip is a window of. The manifest
        stores it relative to its own directory; :func:`load_manifest` resolves it.
    window
        ``(col_off, row_off, width, height)`` for a windowed read.
    positive_fraction
        Fraction of valid pixels labelled flood. Used for sampling and stratification.
    valid_fraction
        Fraction of pixels that are valid. Chips below
        ``DataConfig.min_valid_fraction`` are dropped at chipping time.
    optical_available
        Whether a usable Sentinel-2 observation exists for this chip.
    relative_orbit, orbit_state
        Recorded so the pre/post matching invariant (``docs/04`` section 7) can be
        re-checked at dataset level.
    eval_grid
        ``True`` when the window lies on the non-overlapping evaluation grid
        (offsets multiple of ``eval_stride``); only these chips enter validation/test.
    optical_clear_fraction
        Fraction of the chip with optical quality >= 0.5 (row 6 keeps >= 0.9).
    centroid_x, centroid_y
        Window centre in the analysis CRS, metres — the basis of basin assignment and of
        the buffer test in :func:`floodfusion.splits.assign_basins`.
    """

    chip_id: str
    basin_id: str
    event_id: str
    raster_path: Path
    window: tuple[int, int, int, int]
    positive_fraction: float
    valid_fraction: float
    optical_available: bool
    relative_orbit: int
    orbit_state: str
    eval_grid: bool = True
    optical_clear_fraction: float = 0.0
    centroid_x: float = math.nan
    centroid_y: float = math.nan

    def __post_init__(self) -> None:
        if len(self.window) != 4 or any(int(v) != v for v in self.window):
            raise ValueError(f"{self.chip_id}: window must be 4 integers (col, row, width, height)")
        if not 0.0 <= self.valid_fraction <= 1.0 or not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError(f"{self.chip_id}: fractions must lie in [0, 1]")


#: Manifest columns. The window is stored as four integer columns.
_REQUIRED = ("chip_id", "basin_id", "event_id", "raster_path", "col_off", "row_off",
             "width", "height", "positive_fraction", "valid_fraction", "optical_available",
             "relative_orbit", "orbit_state")
_OPTIONAL = ("eval_grid", "optical_clear_fraction", "centroid_x", "centroid_y")


def _storage_row(record: ChipRecord, base: Path | None) -> dict[str, Any]:
    """The row exactly as stored: window split into columns, path relative to ``base``."""
    row = asdict(record)
    col, rw, w, h = (int(v) for v in row.pop("window"))
    raster = Path(row["raster_path"])
    if base is not None:
        raster = Path(os.path.relpath(raster.resolve(), base.resolve()))
    row["raster_path"] = raster.as_posix()
    row.update(col_off=col, row_off=rw, width=w, height=h)
    return row


def manifest_sha256(records: Iterable[ChipRecord], base: Path | str | None = None) -> str:
    """Content hash: SHA-256 of the stored rows as canonical JSON, sorted by chip_id.

    ``base`` is the manifest directory; paths are hashed relative to it, so moving the
    whole dataset does not change the hash, while any change to any chip does. The hash
    covers the content, not the Parquet bytes (which embed writer versions).
    """
    base_path = Path(base) if base is not None else None
    rows = sorted((_storage_row(r, base_path) for r in records), key=lambda r: r["chip_id"])
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _read_umask() -> int:
    # os.umask can only be read by setting it; done ONCE, at import, so no later file creation
    # in another thread can race the transient 0 umask.
    current = os.umask(0o022)
    os.umask(current)
    return current


_UMASK = _read_umask()


def chip_set_sha256(records: Iterable[ChipRecord]) -> str:
    """Identity of a chip set: two rows are evaluated on the same test set iff these match."""
    return hashlib.sha256("\n".join(sorted(r.chip_id for r in records)).encode("utf-8")).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write via a temporary file in the same directory, fsync, then ``os.replace``.

    An interrupted write can never leave a truncated file that later loads as a valid
    but wrong manifest or split.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o666 & ~_UMASK)  # mkstemp creates 0600; keep normal file permissions
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def manifest_meta_path(path: Path | str) -> Path:
    """``chips/manifest.parquet`` -> ``chips/manifest.meta.json``."""
    path = Path(path)
    return path.with_name(path.stem + ".meta.json")


def write_manifest(
    records: Sequence[ChipRecord],
    path: Path | str,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Write the immutable chip manifest (Parquet, or CSV if the suffix is ``.csv``).

    Rows are sorted by ``chip_id``; raster paths are stored relative to the manifest
    directory so the dataset can be relocated; the write is atomic; and a sidecar
    ``<stem>.meta.json`` records the content hash and the provenance (scene IDs, grid,
    config hash, git SHA). Returns the content hash, which
    :func:`floodfusion.splits.freeze_splits` binds every split to.

    Raises
    ------
    ValueError
        On duplicate ``chip_id`` values or an empty record list.
    """
    import pandas as pd

    path = Path(path)
    if not records:
        raise ValueError("refusing to write an empty manifest")
    counts = Counter(r.chip_id for r in records)
    dupes = sorted(chip for chip, n in counts.items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate chip_id values: {dupes[:10]}")
    base = path.parent
    base.mkdir(parents=True, exist_ok=True)
    rows = sorted((_storage_row(r, base) for r in records), key=lambda r: r["chip_id"])
    frame = pd.DataFrame(rows, columns=list(_REQUIRED) + list(_OPTIONAL))
    if path.suffix == ".csv":
        payload = frame.to_csv(index=False).encode("utf-8")
    else:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        payload = buffer.getvalue()
    atomic_write_bytes(path, payload)
    digest = manifest_sha256(records, base)
    meta = {"manifest_sha256": digest, "n_chips": len(rows), "provenance": dict(provenance or {})}
    atomic_write_bytes(
        manifest_meta_path(path),
        json.dumps(meta, indent=2, sort_keys=True, default=str).encode("utf-8"),
    )
    return digest


def load_manifest_meta(path: Path | str) -> dict[str, Any]:
    """Read the manifest's sidecar (content hash + provenance)."""
    meta = manifest_meta_path(path)
    if not meta.is_file():
        raise FileNotFoundError(f"manifest sidecar not found: {meta}")
    return json.loads(meta.read_text(encoding="utf-8"))


def load_manifest(path: Path | str) -> tuple[ChipRecord, ...]:
    """Read the chip manifest written by :func:`write_manifest`.

    Parameters
    ----------
    path
        Path to the manifest (Parquet or CSV).

    Returns
    -------
    tuple[ChipRecord, ...]
        Immutable sequence of records, sorted by ``chip_id`` — an order that does not
        depend on filesystem iteration, so batches are reproducible across machines.
        Raster paths are resolved against the manifest's directory.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``chip_id`` values are not unique, or required columns are missing (all of
        them are named, not just the first).
    """
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: manifest is missing required columns {missing}")
    duplicated = frame["chip_id"][frame["chip_id"].duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{path}: duplicate chip_id values {duplicated[:10]} — a duplicate "
                         "silently doubles a chip's weight")
    frame = frame.sort_values("chip_id", kind="mergesort")
    base = path.parent
    records = []
    for row in frame.to_dict("records"):
        raster = Path(str(row["raster_path"]))
        kwargs: dict[str, Any] = {
            "chip_id": str(row["chip_id"]),
            "basin_id": str(row["basin_id"]),
            "event_id": str(row["event_id"]),
            "raster_path": raster if raster.is_absolute() else base / raster,
            "window": (int(row["col_off"]), int(row["row_off"]), int(row["width"]), int(row["height"])),
            "positive_fraction": float(row["positive_fraction"]),
            "valid_fraction": float(row["valid_fraction"]),
            "optical_available": bool(row["optical_available"]),
            "relative_orbit": int(row["relative_orbit"]),
            "orbit_state": str(row["orbit_state"]),
        }
        if "eval_grid" in row:
            kwargs["eval_grid"] = bool(row["eval_grid"])
        for name in ("optical_clear_fraction", "centroid_x", "centroid_y"):
            if name in row:
                kwargs[name] = float(row[name])
        records.append(ChipRecord(**kwargs))
    return tuple(records)

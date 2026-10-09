"""Run directories: everything ``P1-M6.3`` needs to regenerate a number from a git SHA.

Layout::

    <runs_root>/<config name>/seed<seed>/
        run.json          resolved config, its SHA-256, manifest + split hashes, priors,
                          git SHA/dirty flag, torch/CUDA/GPU/GDAL versions, VRAM prediction
        metrics.jsonl     one JSON record per epoch (losses, gate, LR, VRAM, time, val IoU)
        best.pt           checkpoint with the best validation macro IoU
        last.pt           resumable state (model, optimiser, scheduler, RNG, epoch)
        eval_<split>.json evaluation output (floodfusion.evaluate)

All JSON is written atomically, so a crash can never leave a half-written record that a
later table run mistakes for a result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from floodfusion.data.records import atomic_write_bytes

__all__ = [
    "RUN_RECORD",
    "append_metrics",
    "discover_runs",
    "load_metrics",
    "load_run_record",
    "run_dir_for",
    "write_json",
    "write_run_record",
]

RUN_RECORD = "run.json"
METRICS = "metrics.jsonl"


def _clean(value: Any) -> Any:
    """JSON-safe: NaN/inf -> None (strict JSON), tuples -> lists, Paths -> str."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):  # numpy / torch scalars
        return _clean(value.item())
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(Path(path), json.dumps(_clean(payload), indent=2, allow_nan=False).encode("utf-8"))


def run_dir_for(runs_root: Path | str, config_name: str, seed: int) -> Path:
    return Path(runs_root) / config_name / f"seed{int(seed)}"


def write_run_record(run_dir: Path, record: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / RUN_RECORD, record)


def load_run_record(run_dir: Path | str) -> dict[str, Any]:
    path = Path(run_dir) / RUN_RECORD
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: not a run directory")
    return json.loads(path.read_text(encoding="utf-8"))


def append_metrics(run_dir: Path, record: Mapping[str, Any]) -> None:
    """Append one epoch record (``metrics.jsonl`` is append-only; one line per epoch)."""
    line = json.dumps(_clean(record), allow_nan=False)
    with (Path(run_dir) / METRICS).open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def load_metrics(run_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(run_dir) / METRICS
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_runs(runs_root: Path | str) -> Iterator[Path]:
    """Every directory under ``runs_root`` holding a ``run.json``, sorted."""
    yield from sorted(p.parent for p in Path(runs_root).glob(f"*/*/{RUN_RECORD}"))

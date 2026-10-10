"""``make preflight`` — milestone F-M0: is this machine ready to reproduce the project?

Each check prints PASS / WARN / FAIL with the value it measured; the exit code is non-zero
if any **required** check fails. Required: Python 3.11/3.12, the core scientific stack,
GDAL/PROJ working (a real reprojection round trip, not just an import), PyTorch, a CUDA
device with bf16 support for training (``--cpu-ok`` downgrades that to a warning), and the
package importable without torch for its arithmetic half. Advisory: a clean git tree (an
ablation table regenerates only from a clean SHA), free disk space, ``uv.lock`` present.

Record ``torch.__version__``, ``torch.version.cuda`` and the git SHA it prints (``P1-M0.2``).
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Check", "main", "run_checks"]

REPO_ROOT = Path(__file__).resolve().parents[2]
_MIN_FREE_GB = 10.0
_CORE = ("numpy", "scipy", "yaml", "pandas", "pyarrow", "matplotlib", "rasterio", "pyproj", "shapely",
         "geopandas", "pystac", "pystac_client", "planetary_computer", "requests")


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS / WARN / FAIL
    detail: str
    required: bool = True


def _python() -> Check:
    ok = (3, 11) <= sys.version_info[:2] <= (3, 12)
    return Check("python", "PASS" if ok else "FAIL", sys.version.split()[0] + ("" if ok else " (need 3.11 or 3.12)"))


def _imports() -> Check:
    missing = []
    for name in _CORE:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            missing.append(f"{name} ({exc})")
    return Check("core packages", "FAIL" if missing else "PASS", "missing: " + ", ".join(missing) if missing
                 else f"{len(_CORE)} importable")


def _gdal_proj() -> Check:
    import rasterio
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, 32632, always_xy=True).transform(10.0, 45.0)
    lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x, y)
    ok = abs(lon - 10.0) < 1e-7 and abs(lat - 45.0) < 1e-7 and abs(x - 578815.302) < 1.0
    return Check("GDAL / PROJ", "PASS" if ok else "FAIL",
                 f"GDAL {rasterio.__gdal_version__}, PROJ round trip (10E, 45N) -> ({x:.1f}, {y:.1f}) m")


def _torch(cpu_ok: bool) -> list[Check]:
    try:
        import torch
    except ImportError as exc:
        return [Check("torch", "FAIL", str(exc))]
    out = [Check("torch", "PASS", f"torch {torch.__version__}, CUDA build {torch.version.cuda}")]
    if not torch.cuda.is_available():
        out.append(Check("CUDA device", "WARN" if cpu_ok else "FAIL", "no CUDA device (training needs one)",
                         required=not cpu_ok))
        return out
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    out.append(Check("CUDA device", "PASS", f"{name}, {total / 1024**3:.1f} GiB ({free / 1024**3:.1f} free)"))
    bf16 = torch.cuda.is_bf16_supported()
    if bf16:
        a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        bf16 = bool(torch.isfinite(a @ a).all())
    out.append(Check("bf16", "PASS" if bf16 else "FAIL", "bf16 matmul on device" if bf16 else
                     "bf16 unsupported: fp16 overflows the Dice reduction (docs/08 5.1)"))
    return out


def _torch_free() -> Check:
    code = ("import sys; sys.modules['torch'] = None; "
            "import floodfusion.config, floodfusion.metrics, floodfusion.splits, floodfusion.data.stac")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": ""})
    return Check("torch-free arithmetic", "PASS" if result.returncode == 0 else "FAIL",
                 "config/metrics/splits/stac import without torch" if result.returncode == 0
                 else result.stderr.strip().splitlines()[-1])


def _git() -> Check:
    from floodfusion.utils.env import git_state

    state = git_state()
    if state["git_sha"] is None:
        return Check("git", "WARN", "not a repository or no commit", required=False)
    detail = f"{state['git_branch']} @ {state['git_sha'][:12]}" + (" (DIRTY)" if state["git_dirty"] else " (clean)")
    return Check("git", "WARN" if state["git_dirty"] else "PASS", detail, required=False)


def _disk() -> Check:
    free = shutil.disk_usage(REPO_ROOT).free / 1e9
    return Check("disk", "PASS" if free >= _MIN_FREE_GB else "WARN", f"{free:.0f} GB free", required=False)


def _lock() -> Check:
    ok = (REPO_ROOT / "uv.lock").is_file()
    return Check("uv.lock", "PASS" if ok else "WARN", "present" if ok else "missing: run `make lock`", required=False)


def run_checks(*, cpu_ok: bool = False) -> list[Check]:
    checks: list[Check] = []
    steps: Sequence[Callable[[], Check | list[Check]]] = (
        _python, _imports, _gdal_proj, lambda: _torch(cpu_ok), _torch_free, _git, _disk, _lock)
    for step in steps:
        try:
            result = step()
        except Exception as exc:  # noqa: BLE001 - a broken check is itself a failed check
            result = Check(getattr(step, "__name__", "check").strip("_"), "FAIL", f"{type(exc).__name__}: {exc}")
        checks.extend(result if isinstance(result, list) else [result])
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F-M0 environment checks.")
    parser.add_argument("--cpu-ok", action="store_true", help="a missing GPU is a warning, not a failure")
    args = parser.parse_args(argv)
    checks = run_checks(cpu_ok=args.cpu_ok)
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{c.status:4}] {c.name:<{width}}  {c.detail}")  # noqa: T201 - this IS the CLI output
    failed = [c for c in checks if c.status == "FAIL" and c.required]
    print(f"\npreflight: {'FAILED (' + ', '.join(c.name for c in failed) + ')' if failed else 'ready'}")  # noqa: T201
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

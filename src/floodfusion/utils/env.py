"""Environment and provenance records for run directories (``P1-M6.3``).

Every number in the ablation table must be regenerable "by one command from a recorded git
SHA" (``MILESTONES.md`` section 6). This module records that SHA — and whether the working
tree was dirty — together with the library and hardware versions that change numerical
results. Torch-free at import time.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = ["CODE_PATHS", "code_changed_since", "environment_record", "git_state"]

REPO_ROOT = Path(__file__).resolve().parents[3]
#: The tracked paths that determine every number: code, configs and the locked environment.
#: Documentation, tests and generated reports do not change a result, so editing them neither
#: makes a run "dirty" nor invalidates a run made at an earlier commit.
CODE_PATHS = ("src", "configs", "pyproject.toml", "uv.lock", "Makefile")


def _git(*args: str, cwd: Path = REPO_ROOT) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state(cwd: Path = REPO_ROOT) -> dict[str, Any]:
    """Return ``{"git_sha", "git_branch", "git_dirty"}``.

    ``git_sha`` is ``None`` outside a repository or before the first commit, and
    ``git_dirty`` is ``True`` whenever a tracked file under :data:`CODE_PATHS` differs from
    ``HEAD`` — a run made from such a tree cannot be regenerated from its SHA alone, so the
    ablation table refuses it without ``--allow-dirty``.
    """
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    # Untracked files under the code paths count too: a new module that is not committed would
    # not exist at the recorded SHA.
    status = _git("status", "--porcelain", "--untracked-files=normal", "--", *CODE_PATHS, cwd=cwd)
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": True if sha is None else bool(status),
    }


def code_changed_since(sha: str | None, cwd: Path = REPO_ROOT) -> bool:
    """``True`` unless the code paths of the working tree are identical to those at ``sha``.

    A run made at an earlier commit is still reproducible by the current tree when only
    documentation, tests or reports changed since — what ``P1-M6.3`` actually requires.
    """
    if not sha:
        return True
    try:
        result = subprocess.run(["git", "diff", "--quiet", sha, "--", *CODE_PATHS], cwd=cwd,
                                capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return True
    return result.returncode != 0


def environment_record() -> dict[str, Any]:
    """Versions that can change results: Python, torch/CUDA/GPU, numpy, GDAL, PROJ."""
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **git_state(),
    }
    try:
        import numpy

        record["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover
        record["numpy"] = None
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        record["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            record["gpu_name"] = props.name
            record["gpu_total_gib"] = round(props.total_memory / 1024**3, 2)
            record["gpu_capability"] = f"sm_{props.major}{props.minor}"
            record["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except ImportError:  # pragma: no cover
        record["torch"] = None
    try:
        import rasterio

        record["rasterio"] = rasterio.__version__
        record["gdal"] = rasterio.__gdal_version__
    except ImportError:  # pragma: no cover
        record["gdal"] = None
    try:
        import pyproj

        record["pyproj"] = pyproj.__version__
        record["proj"] = pyproj.proj_version_str
    except ImportError:  # pragma: no cover
        record["proj"] = None
    return record

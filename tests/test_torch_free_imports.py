"""The arithmetic half of the package must import on a machine without PyTorch.

``tests/conftest.py`` promises that the configuration, splits, STAC, preprocessing,
terrain, baseline, metrics and VRAM-arithmetic tests run without a 2 GB deep-learning
stack. That promise holds only if none of those modules imports ``torch`` at module level,
so it is checked here in a subprocess with ``torch`` made unimportable.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

TORCH_FREE_MODULES = (
    "floodfusion.config",
    "floodfusion.metrics",
    "floodfusion.splits",
    "floodfusion.baselines",
    "floodfusion.masks",
    "floodfusion.data.preprocessing",
    "floodfusion.data.chipping",
    "floodfusion.data.stac",
    "floodfusion.data.encoding",
    "floodfusion.data.records",
    "floodfusion.data.geocoding",
    "floodfusion.data.coregistration",
    "floodfusion.data.s1_annotation",
    "floodfusion.data.sources",
    "floodfusion.data.archive",
    "floodfusion.data.assembly",
    "floodfusion.data.cems",
    "floodfusion.data.ingest",
    "floodfusion.data.orbit_model",
    "floodfusion.data.sen1floods11",
    "floodfusion.data.synthetic.benchmark",
    "floodfusion.utils.vram",
    "floodfusion.utils.env",
    # v1.1
    "floodfusion.data.passes",
    "floodfusion.data.kurosiwo",
    "floodfusion.data.seasonal",
    "floodfusion.cems_benchmark",
    "floodfusion.cems_scores",
    "floodfusion.crossfit",
    # v1.2
    "floodfusion.data.gfm",
    "floodfusion.histograms",
    "floodfusion.data.geoid",
    "floodfusion.data.sidecars",
)


@pytest.mark.parametrize("module", TORCH_FREE_MODULES)
def test_module_imports_without_torch(module: str) -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = None  # any 'import torch' now raises ImportError\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        f"import {module}\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"{module} needs torch at import time:\n{result.stderr}"

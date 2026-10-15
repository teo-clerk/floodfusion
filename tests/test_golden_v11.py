"""The v1.1 checkpoints give the logits they gave at v1.1 (``scripts/golden_outputs.py``).

v1.2 compares new models against v1.1's (P7) under v1.2 code, so no refactor may change what an old
checkpoint predicts. Skipped where the checkpoints (gitignored ``runs/``) or the optional extra are absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
golden_outputs = pytest.importorskip("golden_outputs")


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(golden_outputs.RUNS))
def test_v11_checkpoint_logits_are_unchanged(key: str) -> None:
    path = golden_outputs.available().get(key)
    if path is None:
        pytest.skip(f"{key}: checkpoint not on this machine")
    if key.endswith("_tm") and importlib.util.find_spec("terratorch") is None:
        pytest.skip("needs the optional extra 'fm'")
    golden = np.load(golden_outputs.GOLDEN)[key]
    np.testing.assert_allclose(golden_outputs.logits(path), golden, rtol=1e-4, atol=1e-4)

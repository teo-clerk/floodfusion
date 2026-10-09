"""Shared fixtures for the floodfusion test suite.

How to read this suite
----------------------
**The tests are the specification.** Each states exactly what "done" means for its
function, in terms of shapes, dtypes, devices, value ranges and invariants.

Optional dependencies
---------------------
``torch`` is **not** imported by this file. The suite must survive
``pytest --collect-only`` with zero import errors on a machine that has no PyTorch, so
every heavy dependency is guarded at the point of use: inside the fixture that builds
tensors, or at the top of a test module that imports ``torch``. The configuration, splits,
STAC-query, preprocessing, terrain, baseline and VRAM-arithmetic tests therefore **run**
on a bare machine rather than vanishing.

Markers used here (declared in ``pyproject.toml``; ``--strict-markers`` is on):
``slow``, ``gpu``, ``network``, ``data``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------
# Optional dependency handling
# --------------------------------------------------------------------------
# ``torch`` is deliberately not imported here. ``pytest.importorskip`` at module level
# raises ``Skipped`` while this conftest is being imported, and pytest applies that to
# the whole directory: on a machine without PyTorch the entire Project 01 suite would
# disappear, ``test_config.py`` and the arithmetic half of ``test_vram_budget.py``
# included, neither of which constructs a tensor. The guard therefore lives inside each
# tensor-building fixture, so that only the tests requesting one are skipped.


def _require_torch():
    """Return :mod:`torch`, skipping only the test that asked for this fixture.

    Call it from inside a fixture body, never at module level: a ``Skipped`` raised
    during conftest import is attributed to the directory, not to the test.
    """
    return pytest.importorskip("torch", reason="PyTorch is required for this fixture")


def _cuda_available() -> bool:
    """Whether a usable CUDA device exists. This function never raises.

    Torch absent yields ``False`` rather than an ``ImportError``, and a present but
    mismatched driver also yields ``False``, because the only decision taken from this
    answer is whether to skip.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


#: ``False`` where PyTorch is absent, so ``requires_cuda`` keeps its meaning there.
CUDA_AVAILABLE = _cuda_available()

requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="no CUDA device available"
)


# --------------------------------------------------------------------------
# Configuration fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def experiment_config():
    """A default :class:`~floodfusion.config.ExperimentConfig`."""
    from floodfusion.config import ExperimentConfig

    return ExperimentConfig()


@pytest.fixture(scope="session")
def model_config():
    """Model configuration with a small width, so CPU tests stay fast."""
    from floodfusion.config import ModelConfig

    return ModelConfig(
        encoder_channels=(16, 32, 64, 128),
        decoder_channels=(64, 32, 16, 8),
        attention_heads=4,
        attention_head_dim=16,
        norm_groups=4,
    )


@pytest.fixture(scope="session")
def loss_config():
    from floodfusion.config import LossConfig

    return LossConfig()


# --------------------------------------------------------------------------
# Synthetic tensors
# --------------------------------------------------------------------------
#: Small enough to run on CPU in milliseconds, large enough that a broadcasting bug
#: cannot hide: B, C, H and W are all *different*, so a transposed axis changes the
#: shape and is caught rather than silently accepted.
BATCH = 2
CHIP = 32


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def synthetic_batch():
    """A batch matching the ``Dataset`` contract, with distinct axis sizes.

    Returns a dict with ``sar`` ``(2, 5, 32, 32)``, ``optical`` ``(2, 4, 32, 32)``,
    ``terrain`` ``(2, 3, 32, 32)``, ``label`` / ``valid`` / ``optical_quality``
    ``(2, 1, 32, 32)``, and ``optical_available`` ``(2,)``.
    """
    torch = _require_torch()

    from floodfusion.config import (
        N_OPTICAL_CHANNELS,
        N_SAR_CHANNELS,
        N_TERRAIN_CHANNELS,
    )

    g = torch.Generator().manual_seed(0)
    label = (torch.rand(BATCH, 1, CHIP, CHIP, generator=g) < 0.02).float()
    return {
        "sar": torch.randn(BATCH, N_SAR_CHANNELS, CHIP, CHIP, generator=g),
        "optical": torch.rand(BATCH, N_OPTICAL_CHANNELS, CHIP, CHIP, generator=g),
        "terrain": torch.rand(BATCH, N_TERRAIN_CHANNELS, CHIP, CHIP, generator=g),
        "label": label,
        "valid": torch.ones(BATCH, 1, CHIP, CHIP, dtype=torch.bool),
        "optical_quality": torch.rand(BATCH, 1, CHIP, CHIP, generator=g),
        "optical_available": torch.tensor([True, False]),
    }


@pytest.fixture
def all_negative_batch(synthetic_batch):
    """A batch whose label is entirely zero — the empty-chip case.

    Most chips contain no flood at all (``docs/05`` section 8.2), so the behaviour of the
    loss here determines whether the majority of your training signal exists.
    """
    torch = _require_torch()

    batch = dict(synthetic_batch)
    batch["label"] = torch.zeros_like(batch["label"])
    return batch


@pytest.fixture
def synthetic_dem(rng: np.random.Generator) -> np.ndarray:
    """A smooth synthetic DEM with a ridge, for terrain and shadow tests.

    Deterministic, and shaped ``(64, 96)`` so that a transposed axis is detectable.
    """
    ny, nx = 64, 96
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    ridge = 300.0 * np.exp(-((x - 48.0) ** 2) / (2.0 * 8.0**2))
    slope = 0.5 * y
    return (ridge + slope).astype(np.float64)


# --------------------------------------------------------------------------
# Environment-dependent skips
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def data_root() -> Path:
    """``$ESA_DATA_ROOT``; skips the test when unset or absent."""
    raw = os.environ.get("ESA_DATA_ROOT")
    if not raw:
        pytest.skip("ESA_DATA_ROOT is not set (the root of the real-data event directories)")
    path = Path(raw)
    if not path.exists():
        pytest.skip(f"ESA_DATA_ROOT={path} does not exist")
    return path


def pytest_report_header(config) -> list[str]:  # noqa: ANN001, D103
    try:
        import torch
    except ImportError:
        return [
            "floodfusion: PyTorch is not installed — the tensor suites will be skipped; "
            "the configuration, splits, preprocessing, terrain, baseline and "
            "VRAM-arithmetic tests still run"
        ]
    lines = [f"floodfusion: torch {torch.__version__}, cuda available: {CUDA_AVAILABLE}"]
    if CUDA_AVAILABLE:
        props = torch.cuda.get_device_properties(0)
        lines.append(
            f"floodfusion: {props.name}, "
            f"{props.total_memory / 1024**3:.2f} GiB, "
            f"sm_{props.major}{props.minor}, "
            f"bf16: {torch.cuda.is_bf16_supported()}"
        )
    return lines

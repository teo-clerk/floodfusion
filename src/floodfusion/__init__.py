"""Project 01 — multimodal SAR–optical fusion for near-real-time flood mapping.

This package is the implementation target of Project 01. It is delivered as a set of
**skeletons**: every module carries complete type annotations, a docstring stating the
mathematics it must compute, explicit tensor shape contracts, notes on the traps, a
numbered ``TODO(student)`` list, and a ``NotImplementedError`` naming the document
section that contains the derivation.

You write the implementations. The tests in ``tests/`` are the specification and are
given to you complete and runnable; read the test before implementing the function.

Layout
------
``config.py``       Frozen dataclass configuration. **Defines the channel order.**
``data/``           STAC access, S1 pre-processing, DEM derivatives, chipping, ``Dataset``.
``models/``         Encoders, cross-attention fusion, decoder, segmenter, losses.
``utils/``          Shape contracts and VRAM instrumentation.
``baselines.py``    Otsu, tile-adaptive thresholding, the physics decision rule.
``masks.py``        Slope, HAND, layover/shadow, permanent-water exclusion.
``metrics.py``      Confusion matrices, IoU/F1, PR curves, basin bootstrap.
``splits.py``       Basin-blocked partitioning with buffering.
``train.py``        Training entry point.
``evaluate.py``     Evaluation and ablation-table generation.

Conventions that hold throughout
--------------------------------
1. Tensors are ``(B, C, H, W)``, channels-first, ``float32`` outside ``autocast`` and
   ``bfloat16`` inside it.
2. The model emits **logits**, never probabilities. Losses take logits.
   See ``docs/07_fusion_architectures.md`` section 9.2.
3. SAR backscatter is :math:`\\gamma^0` in **decibels**. All filtering and averaging
   happens in **linear power**; conversion to dB is the last step.
   See ``docs/04_data_access_and_sentinel1_preprocessing.md`` section 3.4.
4. Invalid pixels (layover, shadow, permanent water, nodata, out-of-AOI) are excluded
   from **both** the loss and the metric.
   See ``docs/03_dem_hand_and_terrain_effects.md`` section 5.4.
5. Normalisation statistics are computed on the **training split only**, frozen to the
   config, and applied unchanged everywhere. Never normalise per chip.
   See ``docs/05_dataset_construction_and_splits.md`` section 7.1.

Importability
-------------
``floodfusion`` is made importable without installation through ``pythonpath`` in
``[tool.pytest.ini_options]`` and through ``PYTHONPATH`` exported by the ``Makefile``.
Run tests with ``make test``, not with a bare ``pytest`` from an arbitrary directory.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
]

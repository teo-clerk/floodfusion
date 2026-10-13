"""floodfusion — SAR–optical fusion for near-real-time flood mapping.

Sentinel-1 flood segmentation with optional Sentinel-2: a physics baseline, a gated
cross-attention network that is SAR-only by construction when optical is missing,
calibrated decision thresholds, event-level evaluation and a GFM-style product. Every
module carries complete type annotations, a docstring with the mathematics it computes,
explicit tensor shape contracts and notes on the traps. ``docs/NN §x`` citations point to
the design specification the code implements (not part of this repository); the tests in
``tests/`` are the executable specification.

Layout
------
``config.py``           Frozen dataclass configuration. **Defines the channel order.**
``data/``               STAC/archive access, S1 annotation and geocoding, terrain, chipping,
                        Sen1Floods11 and CEMS clients, remote-input guards, ``Dataset``.
``models/``             Encoders, cross-attention fusion, decoder, segmenter, losses, zoo.
``utils/``              Shape contracts, VRAM and latency instrumentation, provenance.
``baselines.py``        Otsu, tile-adaptive thresholding, the physics decision rule.
``baseline_runner.py``  The baseline rows and their per-mask attribution.
``masks.py``            Slope, HAND, layover/shadow, validity flags.
``metrics.py``          Confusion matrices, IoU/F1, PR curves, bootstrap, permutation tests.
``calibration.py``      Prior correction, temperature/Platt scaling, conformal thresholds.
``splits.py``           Basin-blocked partitioning with buffering, normalisation stats.
``train.py``            Training entry point.  ``optim.py``: optimiser and schedule.
``evaluate.py``         Evaluation of a run.  ``runs.py``: run directories and records.
``ablation.py``         The synthetic ablation grid.  ``reporting.py``: its table.
``crossval.py``         Leave-one-event-out and the published split.  ``cv_report.py``: table.
``stress.py``           Single-factor stress curves.  ``case_study.py``: EMSR664 vs CEMS.
``predict.py``          Inference on an event.  ``products.py``: COG layers + STAC Item.
``figures.py``          Shared figure helpers.  ``preflight.py``: environment checks.

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

import os as _os

# A downloaded "*.json" or "*.dbf" holding an <OGRVRTDataSource> would make GDAL read other files
# (or URLs) on open, whatever the file name says: the OGR VRT driver is never loaded. OGR_SKIP is read
# when GDAL registers its drivers, i.e. on the first import of pyogrio/rasterio — after this package.
_skip = [d for d in _os.environ.get("OGR_SKIP", "").replace(" ", ",").split(",") if d]
if "OGR_VRT" not in _skip:
    _os.environ["OGR_SKIP"] = ",".join([*_skip, "OGR_VRT"])

__version__ = "1.1.0"

__all__ = [
    "__version__",
]

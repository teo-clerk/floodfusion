"""Data pipeline: STAC access, S1 pre-processing, chipping and the PyTorch ``Dataset``.

.. warning::

   This sub-package is named ``data`` and would therefore be swallowed by the
   ``data/`` and ``*/data/`` rules in the repository ``.gitignore``, which exist to keep
   multi-gigabyte EO rasters out of git. The rules are negated for ``**/src/*/data/`` at
   the bottom of that file. If you ever edit ``.gitignore``, re-verify with::

       git check-ignore -v 01_FLOOD_SAR_OPTICAL_FUSION/src/floodfusion/data/dataset.py

   which must print nothing and exit 1. Without the negation these modules are silently
   untracked: absent from ``git status``, absent from every commit, and destroyed by
   ``git clean -xdf``.

Modules
-------
``stac``          Query CDSE / Planetary Computer for matched S1/S2/DEM items.
``preprocessing`` Calibration, noise removal, terrain correction (``docs/04``).
``chipping``      Chip extraction and manifest generation (``docs/05``).
``dataset``       The PyTorch ``Dataset`` and ``DataLoader`` (``docs/05``, ``docs/08`` section 6).
"""

from __future__ import annotations

__all__: list[str] = []

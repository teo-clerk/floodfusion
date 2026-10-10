"""Data pipeline: archive access, geometry, terrain, chipping and the PyTorch ``Dataset``.

.. warning::

   This sub-package is named ``data``, like the multi-gigabyte data directory the
   repository ``.gitignore`` excludes. The ignore rules are anchored at the repository
   root (``/data/``, ``/runs/``) for that reason. If you edit ``.gitignore``, re-verify
   with::

       git check-ignore -v src/floodfusion/data/dataset.py

   which must print nothing and exit 1; otherwise these modules are silently untracked.

Modules
-------
``stac``            Matched Sentinel-1/2 searches on Planetary Computer.
``archive``         Item lookup, GRD annotations for RTC scenes, GDAL settings.
``sources``         Readers for RTC, L2A reflectance and SCL, GLO-30, JRC, WorldCover.
``http``            Remote-input guards: URL and path checks, capped transfers, redaction.
``s1_annotation``   Sentinel-1 product annotation parsing (geolocation grid, orbit).
``geocoding``       Range–Doppler geocoder with area-based radiometric terrain correction.
``orbit_model``     Geometry from a sun-synchronous orbit model when annotations are missing.
``preprocessing``   Calibration, noise removal, dB conversion, Refined Lee (``docs/04``).
``coregistration``  Window cross-correlation and shift estimation.
``assembly``        One analysis grid per scene: every layer, terrain, alignment, QA.
``ingest``          An event from its configuration (CEMS-style AOI) to chips.
``sen1floods11``    Sen1Floods11 download, metadata, splits and re-ingest.
``cems``            Copernicus EMS rapid-mapping products: listing, download, layers.
``encoding``        The 16-band int16 stack layout and its scales.
``chipping``        Event stacks, chip extraction and the manifest (``docs/05``).
``records``         Manifest records.
``dataset``         The PyTorch ``Dataset`` and ``DataLoader`` (``docs/05``, ``docs/08`` section 6).
``synthetic/``      The physics-based synthetic benchmark.
"""

from __future__ import annotations

__all__: list[str] = []

"""Physics-based synthetic flood events — the offline benchmark and the dataset fallback.

``landscape``  terrain, basins, rivers, land cover, permanent water, HAND-driven floods
``sar``        Sentinel-1 gamma0 physics (Fresnel, Bragg, double bounce, shadow, speckle)
``optical``    Sentinel-2 reflectance, clouds, cloud shadow, SCL-derived quality
``scene``      renders one event through the production terrain code into a 16-band stack
``chip``       deterministic in-memory chips for ``DataConfig.chip_source = "synthetic"/"auto"``
``benchmark``  the frozen four-event benchmark and its CLI

Synthetic numbers validate the pipeline; they are never reported as real-world accuracy.
"""

from __future__ import annotations

__all__: list[str] = []

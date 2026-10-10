# Changelog

All notable changes; the format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-10

First public release.

### Real data
- **Sen1Floods11, re-ingested through this pipeline.** The hand labels are kept and every input is
  re-fetched from Planetary Computer:
  - the same-orbit pre/post Sentinel-1 RTC pair on the dataset's stated orbit;
  - Sentinel-2 L2A on its stated date;
  - GLO-30, JRC GSW and WorldCover.
- **Geometry corrections**, measured on the archive:
  - labels and optical are registered to the SAR per event by whole-pixel shifts, taken from the labels'
    water edges where that estimate is precise (cloudy events) and from SWIR otherwise; the use of labels
    and its bounds are disclosed, and independent checks are reported;
  - an orbit-model fallback supplies geometry where the GRD annotation is missing.
- **Event-level cross-validation.**
  - Leave-one-event-out over 10 events.
  - The published within-event split is scored on the same chips (the protocol gap: at τ*, at the
    oracle threshold and in AP).
  - Exact sign-flip permutation tests, Holm-adjusted.
- **EMSR664 Forlì case study.** Every Sentinel-1 pass of 16–23 May 2023 is scored against the four
  Copernicus EMS delineations, on the region each product observed, with a recession series.
- **TerraMind-1.0 (IBM/ESA) comparison row**, through TerraTorch, with its caveats stated.

### Product and evaluation
- **GFM-style product package.** Seven Cloud-Optimized GeoTIFF layers and a STAC 1.0 Item (projection,
  raster, sat, sar and processing extensions; `derived_from` the input scenes); a network test validates
  the Item against the published schemas.
- **Conformal risk control** of the per-chip false-negative rate, with per-event coverage.
- **Co-registration QA that knows when it cannot measure**: precision-gated checks report
  pass / fail / not measurable (e.g. the terrain simulation on a flat plain).
- **Single-factor stress curves** (wind, cloud, incidence, harvest) on five unseen synthetic landscapes,
  and the fusion gate figure.

### Engineering
- **Remote-input guards.** URL checks with the parser that makes the request, GDAL path guards, capped
  and retried transfers, safe archive extraction, redacted tracebacks.
- **CI.** Lint and the offline suite on Python 3.11 and 3.12 with an 80 % coverage gate; a weekly live
  test against the public services. Actions and images are pinned by digest.
- **Packaging.** A Docker image with the locked environment, and an executed walkthrough notebook.
- **Documentation.** Model card, data card, security policy, citation file.

### Known issues
- **Refined Lee warns on zero-variance windows** (`preprocessing.py`, divide by zero; such windows can
  produce NaN). To be guarded in 1.0.1.

- **The Sen1Floods11 table's caveat says "7 of 10 events"** show no backscatter drop on labelled flood. The
  release ingest gives 8 of 10 (`reports/sen1floods11_label_change.md`); the caveat is hard-coded in the
  report generator. Fixed in 1.0.1.

### How the release numbers were produced
- The Sen1Floods11 and synthetic tables were produced at `588d66c`, and each records that SHA. v1.0.0
  differs from it only in `case_study.py` (the Forlì fixes) and in the title of the stress gate figure.
  To regenerate those tables, check out the SHA they record.
- The Forlì report was produced at `f2e7a85`, and the stress report was regenerated at `b9866f9` with
  identical numbers.

### Before 1.0
- **The pipeline itself.** The Range–Doppler geocoder with area-based RTC; terrain physics; the
  physics-based synthetic benchmark; basin-blocked splits; the cross-attention fusion network with its
  quality gate; calibration; the 13-row ablation grid; VRAM budgeting.

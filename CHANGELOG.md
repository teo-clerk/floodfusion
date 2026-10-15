# Changelog

All notable changes; the format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] — 2026-10

A fair fight with TerraMind, Copernicus's operational flood maps as a yardstick, and a blind test on floods no model
had seen: activations from EMSR899 on, never looked at during development and scored only after the freeze
(`reports/protocol_v1_2.md`, committed before any v1.2 model or blind score). Every
decision since is in `reports/v1_2_deviations.md`.

### Results (exploratory; the confirmatory test is the blind one)
- **Selection.** The pooled CV selected the seasonal-context model (row 21) over the same model without context:
  0.380 vs 0.361 whole-scene macro IoU at τ*, better in 49 of 79 groups. Neither inference add-on met its adoption rule.
- **The fair TerraMind.** TerraMind-base + LoRA was trained on the same pooled set with the same learning-rate
  budget. In CV (chip level, 79 groups with a defined IoU) it scores 0.313 at τ* against row 21's 0.394. Most of the
  gap is threshold transfer: 0.369 vs 0.392 at a cross-fitted threshold, 0.406 vs 0.430 at each group's best;
  AP 0.617 vs 0.651 (fold-level).
- **The frozen models on the 38 seen CEMS floods** (scored once after the freeze, rebuilt stacks; descriptive):
  v1.2 0.417, fair TerraMind 0.405, v1.1's best 0.407. On the 34 floods GFM covers, GFM scores 0.321 against
  v1.2's 0.419.
  - **With the blind test's own analysis:** ours beats GFM (+0.098, significant at both units), and nearly ties
    TerraMind (+0.012 IoU, +0.015 AP) and v1.1's best (+0.010), neither significant
    (`reports/v12_cems_seen_tests.md`).

### The blind test
- **Frozen** at tag `v1.2.0-freeze`: row 21 and TerraMind-base + LoRA (3 seeds each), v1.1's best, and GFM. 36 files
  are pinned in `reports/v12_freeze.json`.
- **Rehearsed** end to end on EMSR664 before the freeze.
- **First sweep (2026-10-14):** census to EMSR934, three flood activations. EMSR926 has no published delineation
  package and EMSR927 only grading maps, so both are not eligible; EMSR933 settles on 2026-10-31.
- **Sealing:** scores stay sealed (hash-chained ledger) until 20 activations in 15 clusters are scored, or 2027-12-31.

### v1.1 errata (`reports/v1_1_erratum.md`)
- **TerraMind trained without modality dropout.**
  - Its config set modality dropout to 0.5, but the training loop applied dropout only to the fusion rows.
  - Its decoder's BatchNorm also trained on sub-batches.
  - Both are fixed; TerraMind is re-trained, and P4 is re-scored beside the pre-registered result.
  - **H4 reverses.** Trained as described, TerraMind beats the fusion model on the 38 CEMS activations: 0.299 vs
    0.258; fusion − TerraMind = −0.042 per activation and −0.043 per cluster, significant at both. v1.1's conclusion
    that the task-specific model beats the foundation model is withdrawn. The pre-registered verdict is kept as run.
- **Zero terrain past a DEM tile edge.**
  - Copernicus GLO-30 declares no nodata, so terrain was read as 0 m past a 1° tile boundary.
  - Affected: 14.6 % of the CEMS benchmark's stack area, 8.6 % of Sen1Floods11's and 6.4 % of Kuro Siwo's
    (`scripts/dem_exposure.py`).
  - The fix is in `read_dem`. Kuro Siwo and the CEMS benchmark are rebuilt for v1.2; the v1.1 data stays for the
    regression checks.
  - **Measured on the rebuilt CEMS stacks:** the models that read terrain (fusion, the physics baseline) gain +0.002
    to +0.010 IoU. The others move by less than 0.001, through the scored region alone (its shadow and layover masks
    come from the DEM). The corrected P4 still goes against fusion: −0.033 per activation, −0.034 per cluster, significant.

### Added
- **Copernicus GFM as a comparator** (`data/gfm.py`, `make gfm`). The EODC public STAC has GFM for 82 of 90
  benchmark pairs.
- **Score histograms, threshold-free AP, oracle IoU and a bounded per-scene EM self-calibration**
  (`histograms.py`, `cems_scores_v12.py`). The v1.1 models' counts must reproduce v1.1's exactly.
- **The fair TerraMind** (rows 24–26):
  - siamese pre/post encoder with its native DEM;
  - in-house LoRA (`models/lora.py`);
  - size and adaptation chosen by a timing probe under a pre-registered rule.
- **GEOID-Flood** (CC BY 4.0) re-ingested through Kuro Siwo's pipeline: 111 events selected before EMSR656, 102 of
  which produced chips.
- **The pooled training set** (`data/pool.py`): 100 km groups recomputed across datasets.
- **Sidecar context:**
  - `data/sidecars.py`: seasonal reference, AlphaEarth, DEM;
  - row 20–22 context models (`ContextFloodSegmenter`). AlphaEarth is deferred while the public mirror is too slow.
- **The blind-test runner** (`prospective.py`, `make v12-watch`):
  - census with expiry and settling;
  - scores sealed outside the repository, with their sha256 in a committed ledger;
  - a freeze check.
- **The Arena page** (`site/arena.html`):
  - scoreboard with five metrics and 95 % intervals;
  - map explorer;
  - research landscape, errata and blind-test tracker.
- **Golden-output tests** pin the v1.1 checkpoints' logits across refactors.

## [1.1.0] — 2026-10

A pre-registered test of v1.0's claims on independent references. The protocol
(`reports/protocol_v1_1.md`) was committed before any v1.1 model was trained. Every change since is in
`reports/v1_1_deviations.md`.

### Results
- **CEMS benchmark, 38 flood activations** (2023-03 to 2026-03), zero-shot. The maps were drawn on non-Sentinel radar
  satellites. The pre-registered tests, at both unit definitions (38 activations, 30 clusters):
  - **H1, supported:** the flood-label recipe beats the water-label recipe on the same Kuro Siwo chips, +0.06 IoU.
  - **H4, supported:** the fusion model beats TerraMind-small at each model's own threshold, +0.03. At the best single
    threshold, chosen in-sample, TerraMind is ahead.
  - **H2, no verdict:** fusion − SAR U-Net where optical is usable, +0.04, not significant; too few clusters.
  - **H3, not supported:** non-inferiority without optical, −0.02, holds per activation but not per cluster.
- **Training data moved scores more than the architecture** (descriptive). Kuro Siwo flood-label models score
  0.39–0.40 and the same networks trained on Sen1Floods11 0.26–0.28. The physics baseline scores 0.371.
- **Kuro Siwo cross-validation** (36 floods, 29 groups): fusion 0.535 vs SAR U-Net 0.505, not significant.
  Optical changes the fusion model's score by 0.002 at inference.
- **Replications at the v1.1 code:**
  - Sen1Floods11 leave-one-event-out: fusion 0.608 vs 0.343.
  - The synthetic grid and the stress curves, within 0.017 IoU of v1.0.
- **Stage A diagnostics of the v1.0 models** (`reports/stage_a_diagnostics.md`). Under a partial cloud the fusion
  model misses most flood beneath it. On partly cloudy CEMS activations (descriptive) the same model scores 0.254 with
  optical and 0.307 without.

### Added
- `floodfusion.data.kurosiwo`: Kuro Siwo ingest with flood and water label sets on identical chips, 100 km split
  groups and per-AOI registration.
- `floodfusion.cems_benchmark` and `floodfusion.cems_scores`: the eligibility census, pair preparation and the
  pre-registered tests, with activation and cluster units and an exact one-sided non-inferiority test.
- `floodfusion.data.seasonal` and `floodfusion.data.passes`: year-earlier Sentinel-1 references and pass coverage.
  Baseline variants 2e (event-pooled threshold fallback) and 2s (seasonal reference).
- `floodfusion.crossfit`: calibration across events, with cross-fitted cost-optimal and conformal thresholds.
- Fusion variants 9b (gate bias 0), 9c (synthetic clouds in training) and 9r (query from SAR).
- Grouped k-fold cross-validation; SAR-only calibration for every model with an optical input.
- `scripts/v11_queue.sh` (the whole run, resumable) and `scripts/cems_strata.py`.
- **Interactive explainer** (`site/`, published to GitHub Pages by `.github/workflows/pages.yml`).
  `scripts/build_site.py` reads the results from the committed reports instead of copying them into the
  page.

### Fixed
- Windows without optical are scored with the model's SAR-only calibration.
- Refined Lee no longer divides by zero on constant windows.
- Monte Carlo p-values use (count + 1)/(n + 1).
- The Planetary Computer client backs off on rate limits.
- The README's protocol-gap sentence now separates the trained models (at most +0.015 IoU at τ*) from the
  fusion model run without optical (up to +0.091). None of these gaps is significant. The v1.1 re-run puts
  the SAR U-Net's gap at up to +0.11, again not significant.

### Known issues
- `make cems-data` re-runs the CEMS census against the live API and overwrites the frozen census. To reproduce,
  run only `python -m floodfusion.cems_benchmark prepare`.
- The Kuro Siwo variants were stopped by the pre-registered 12 GPU-hour cap. Their fold-0 smoke runs still appear in
  the Kuro Siwo table, marked as not comparable.
- In `cems_benchmark.md`, the partial-cloud table repeats one comparison under P1 and P2.
- Whether variant 9c fixes the partial-cloud failure was not tested.

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

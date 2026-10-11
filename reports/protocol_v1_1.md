# floodfusion v1.1 — pre-registered protocol

Committed **before any v1.1 model was trained or scored**. The git history timestamps it. Every v1.1 report cites it
and lists any deviation from it in `reports/v1_1_deviations.md`.

**What was known when this was written:**
- every v1.0 result (`reports/*`, release v1.0.0);
- the CEMS archive's metadata, from the eligibility census below;
- the Kuro Siwo annotation polygons and dates, plus a three-event smoke ingest that checked the code and registration;
  it scored no model.

**What was not known:** any number from a v1.1 model, and anything from Stage A's diagnostics of the v1.0 models,
which run in parallel and touch neither new dataset.

## 1. Questions

v1.0 left one main doubt: its labels, its references or both may be what drives its results. Sen1Floods11's labels mark
water on a date, drawn on the optical image the fusion model also sees. Its change reference is the latest pass 5–24
days before the flood.

v1.1 asks four confirmatory questions on references independent of both. Everything else in v1.1 is descriptive.

| | Hypothesis | Test |
|---|---|---|
| **H1** | The label definition causes the radar path's over-mapping. A model trained on change-aware **flood** labels agrees better with independent maps than the same model trained on **water** labels of the same chips. | P3 |
| **H2** | Fusion helps on independent maps, not only on maps drawn on its own optical input. | P1 |
| **H3** | Without optical, the fusion model is no worse than the SAR-only U-Net (requirement R2 in practice). | P2 |
| **H4** | The task-specific fusion model vs the TerraMind foundation model, on events after TerraMind's SSL4EO pre-training period. | P4 |

## 2. Data

### 2.1 Sen1Floods11 (as v1.0)

- **Chips:** v1.0's ingest is reused unchanged (`data/sen1floods11`, same manifest).
- **Protocols:** leave-one-event-out over 10 events (validation = the next event, cyclic) plus the published split.

### 2.2 Kuro Siwo (new)

- **Source:** the annotation polygons at `Orion-AI-Lab/KuroSiwo-annotations` @ `b930dd27`, tarball sha256 `0a4b33ea…`.
  The data is CC-BY-4.0 and the repository MIT. Every input is re-fetched from Planetary Computer by
  `floodfusion.data.kurosiwo`.
- **Dates:** `configs/datasets/kurosiwo_events.yaml` (sha256 `520cf54c…`), read from the dataset's MS1 GeoTIFF names.
  40 of 43 events are dated. Cameroon, Djibouti and Nicaragua are **excluded** (no date found).
- **Acquisition:** on the date, the Sentinel-1 pass covering the largest share of the AOI (ties: the earlier). An AOI
  covered below 10 % is skipped.
- **Chips:** 5.12 km squares on a UTM grid at least 90 % inside AOI ∩ pass. Per event, a generator seeded by the
  event's name draws up to 12 squares with flood ≥ 1 % and up to 4 below.
- **Optical:** Sentinel-2 L2A within ±2 days, with real clouds.
- **Registration:**
  - The label offset (label edges vs SAR) and the optical offset (SWIR vs SAR) are measured per AOI over up to 48
    chips. That is the sampled chips plus a seeded draw.
  - Each is applied only when it is a measurement: ≥ 12 chips and a 95 % interval ≤ 1 px (v1.0's rule).
- **Labels, on identical chips:**
  - `flood`: Kuro Siwo's event layer;
  - `water`: event ∪ hydro, the Sen1Floods11-like definition, built by `relabel-water`.

  JRC permanent water (≥ 80 %) is removed from both, as for Sen1Floods11.
- **Groups:** events whose AOIs lie within 100 km form one split group (single linkage). That gives **32 groups from 40
  events**; the largest joins five Spanish and French events.
- **Protocols:**
  - grouped **5-fold**: groups dealt largest first to the lightest fold; test = fold *i*, validation = fold *i*+1, train
    = the rest;
  - **all**: train on folds 0–3, validate on fold 4. This is the model applied to CEMS.

### 2.3 CEMS benchmark (new, the confirmatory test set)

- **Census:** `configs/datasets/cems_benchmark.json` (sha256 `5384cbd2…`), run 2026-10-10 18:13 UTC with
  `floodfusion.cems_benchmark eligibility`, codes EMSR656–EMSR898.
  - 233 activation records are public; 86 of them are floods.
  - A flood activation qualifies through a **delineation product** that meets all of these:
    - every source image is non-Sentinel (radar: COSMO-SkyMed, PAZ, RADARSAT-2, TerraSAR-X, ICEYE, …; optical VHR:
      Pléiades, WorldView, SPOT, GeoEye, …), and every sensor is of known kind;
    - its package is published;
    - a Sentinel-1 RTC acquisition lies within **±24 h** of its latest image and covers **≥ 50 %** of its extent;
    - a same-orbit pre-event acquisition exists within 60 days before the **reference date**. That date is the event
      time − 7 days, moved back before any overlapping flood activation that began in the previous 60 days.
  - **Result: 44 activations, 117 AOIs.** By the sensor their product was drawn on: 39 radar, 7 optical VHR, 1 mixed.
- **Per AOI,** the qualifying (product, pass) pair with the smallest |Δt| (ties: the earlier pass).
- **Caps:**
  - at most 4 AOIs per activation (smallest |Δt|, ties by AOI number);
  - at most 16 tiles (about 16 × 17 km each) per AOI, drawn by a generator seeded with the pair key. The draw is blind
    to the flood and to every model.
- **Scoring:**
  - scored pixels lie inside the product's observed region (its AOI inside the footprints of its source images, minus
    "not analysed"), have valid SAR, and are outside permanent water;
  - optical is Sentinel-2 within ±2 days.
- **Units:** the **activation**, with its AOIs and seeds pooled.
- **Primary stratum:** products drawn on **radar** satellites, the conservative test for fusion. EMSR664 (Forlì, seen in
  v1.0) is excluded from the confirmatory set, which leaves **K = 38**.
- **Subsets of a unit,** by the share of its scored pixels with clear optical:
  - **optical**: ≥ 50 %;
  - **SAR-only**: < 10 %.

## 3. Models and runs

- **Recipe:** v1.0's for every learned row (30 epochs, early stopping on validation, Platt or temperature calibration,
  τ* = 0.25).
- **Freeze:** all configs are frozen before the freeze SHA, and every run is made at that SHA.
- **Order:** GPU jobs run one at a time, after a smoke run of one fold of every configuration.

| Dataset / protocol | Rows | Seeds |
|---|---|---|
| Synthetic grid (as v1.0) | 1–13 | 42, 43, 44 |
| Sen1Floods11 LOEO and published split | baselines 1, 2, 3, 2e, 2s; learned 4, 9 (+ 11 = 9 SAR-only), 12, TerraMind | learned: 42, 43, 44 |
| Sen1Floods11 LOEO | variants 9b, 9c, 9r | 42 |
| Kuro Siwo 5-fold, flood labels | baselines 1, 2, 3, 2e, 2s; learned 4, 9 (+ 11) | 42, 43, 44 |
| Kuro Siwo 5-fold, flood labels | variants 9b, 9c, 9r | 42 |
| Kuro Siwo 5-fold, water labels | 9 | 42, 43, 44 |
| Kuro Siwo all (flood; water) | 4, 9 (flood); 9 (water) | 42, 43, 44 |

**Variant definitions:**
- **2e:** row 2 with the event-pooled threshold fallback.
- **2s:** 2e with the change term against the **seasonal reference**: the median of same-orbit passes within ±30 days
  of the same date one year earlier, at least 3, else the recent reference (counted). Built by
  `floodfusion.data.seasonal`.
- **9b:** gate bias 0 instead of −2.
- **9c:** synthetic clouds on half the optical training chips.
- **9r:** query from SAR, keys and values from optical.

**Compute cap:** Kuro Siwo ≤ 12 GPU-h. If the cap is hit, seeds 43 and 44 of the variants' parents come first, then the
water-label confirmation runs; nothing is shortened silently.

## 4. Confirmatory tests (`configs/datasets/cems_models.yaml`, sha256 `828d5d7e…`)

| Test | B − A | Subset | Rule |
|---|---|---|---|
| P1 | Fusion (S1F11) − SAR U-Net (S1F11) | optical | two-sided |
| P2 | Fusion (S1F11) − SAR U-Net (S1F11) | SAR-only | **non-inferiority**: the 95 % paired-bootstrap lower bound > −0.05 |
| P3 | Fusion (Kuro Siwo, flood) − Fusion (Kuro Siwo, water) | all | two-sided |
| P4 | Fusion (S1F11) − TerraMind (S1F11) | all | two-sided |

**Rules for every test:**
- The statistic is the per-activation IoU at each model's τ*, macro-averaged; the gap is the mean per-activation
  difference.
- Two-sided tests use an exact sign-flip permutation test over activations, Holm-adjusted across exactly the two-sided
  tests in this table that reach K ≥ 8.
- **No verdict below K = 8** activations in a test's subset.
- A 95 % paired percentile bootstrap interval over activations is reported with every test.

**What counts as support:**
- **H1:** P3 significant and positive. A significant negative P3 refutes H1. A non-significant P3 means the label
  definition does not detectably matter on independent maps.
- **H2:** P1 significant and positive. Its size is compared descriptively with v1.0's +0.263.
- **H3:** P2 non-inferior.
- **H4:** reported whatever its direction.

**No model is adopted from v1.1's variants.** Making one the reference would be a new question with its own test.

## 5. Descriptive (no verdicts beyond the v1.0 replication)

- **Replication:** the Sen1Floods11 LOEO table with v1.0's gap family (Holm within it, as in v1.0), and the protocol
  gap.
- **Variants:** gaps 12 − 9, 9b − 9, 9c − 9, 9r − 9, 2e − 2, 2s − 2e, each with an unadjusted p and no verdict.
- **Kuro Siwo tables** (flood and water labels).
- **Calibration across events:** cross-fitted cost-optimal and conformal thresholds (`floodfusion.crossfit`). The
  per-chip grid now reaches 0.005.
- **CEMS:** every model on every stratum, optical vs SAR-only subsets, the oracle-threshold IoU and K per subset.

## 6. Deviations and stopping

- **Deviations:** any change after this commit (a rule, a cap, a model, a failed step) is logged with its reason in
  `reports/v1_1_deviations.md` before the report that it affects.
- **Insufficient data:** if a dataset yields fewer units than planned (an ingest failure, a missing archive scene), its
  results are reported on what exists, with K stated. Tests below K = 8 get no verdict.
- **Results never change the protocol:** no threshold, cap, model or subset rule is changed after a v1.1 number has
  been seen.

## Amendment 1 — 2026-10-11, before any v1.1 model was trained or scored

Two independent reviews of the implemented design (experimental design; Python) found the problems below. The fixes
are fixed here, before the freeze SHA. Nothing in this amendment was informed by a v1.1 result, because none existed.

**A1.1 The SAR-only calibration of a model with an optical input.**
- A window without usable optical is read with the model's SAR-only calibration. That calibration is fitted on SAR-only
  validation logits: row 11's for row 9, and a new `cmp_terramind_small__eval_sar_only` for TerraMind.
- Before this fix, P2 would have scored the fusion model's no-optical path with its with-optical calibration. That is
  0.6 logit apart at τ*, which is comparable to the margin.
- Descriptive companions are added: each optical model run SAR-only everywhere.

**A1.2 Two unit definitions.**
- An activation is not always an independent flood.
- Activations within 100 km and 60 days of each other (single linkage, the Kuro Siwo grouping rule extended in time)
  form a **cluster**.
- Every test is run with activations and with clusters as units, with Holm applied within each definition.
- A hypothesis is **supported** only if its test holds at both, with the same sign.
- Expected K, counted from metadata and optical shares before any score:

  | Test | Per activation | Per cluster |
  |---|---|---|
  | P1 (optical) | about 8 | about 7, so likely **no verdict** |
  | P2 (SAR-only) | about 15 | fewer |
  | P3, P4 | 38 | about 30 |

**A1.3 P2 is an exact one-sided test.**
- Non-inferiority is an exact sign-flip test of d + 0.05 > 0 over units, one-sided at α = 0.025, outside the Holm
  family. The percentile bootstrap interval is still reported, but not as the test.
- Why 0.05: v1.0's 95 % interval for row 11 − row 4 was [−0.044, +0.038]. The margin declares "inferior" only a loss
  larger than v1.0 left plausible.
- Power: with K ≈ 15, non-inferiority needs most units above −0.05. An inconclusive P2 is a possible, reportable
  outcome.

**A1.4 p-values.**
- Sign-flip p-values are exact for K ≤ 20.
- Above that they are Monte Carlo, 200 000 patterns, computed as (count + 1)/(n + 1). This applies to P3 and P4.

**A1.5 Descriptive additions.** None carries a verdict.
- the partial-cloud subset (10–50 % clear);
- a sensitivity analysis without units holding < 1 km² of flood;
- a sensitivity analysis without the 5 pairs whose regions overlap Kuro Siwo training AOIs (by place, not by event):
  EMSR773 AOI06, 856 AOI07, 864 AOI05, 868 AOI01 and AOI02;
- confusion counts kept per seed and pooled only in the report.

**A1.6 P3 compares training recipes.**
- The label definition changes everything that follows it: the sampler's positives, early stopping, the priors, the
  calibration.
- CEMS maps are themselves change-aware flood maps, so a positive P3 includes agreement of definitions.
- H1 speaks of the radar path. Its SAR-only companions (`ks_r11_flood`, `ks_r11_water`) are reported beside P3.

**A1.7 The Kuro Siwo all-groups model.**
- Training stays on folds 0–3.
- Validation is fold 4 **minus its smallest group**, a data-blind rule. That group (12 windows) is a technical test set
  that is never reported.
- The previous split (validation = test) fails the split loader's disjointness check.
- The rule first adopted after that, alternate groups, left validation 2.5 % optical. This one leaves it 19 %, against
  48 % in training.

**A1.8 Order and cap.**
- The run order is:
  1. the deployable models (Sen1Floods11 published split; Kuro Siwo all, flood and water);
  2. the CEMS scores;
  3. the cross-validation tables;
  4. the synthetic grid.
- The Kuro Siwo cap of 12 GPU-h is checked before the water-label cross-validation and before the Kuro Siwo variants,
  and those are dropped in that order. A dropped item is reported as not run.

**A1.9 The seasonal reference fails closed.**
- A stack with no `seasonal.json` is an error.
- "Insufficient" (fewer than 3 acquisitions) falls back to the recent reference, counted per scene.
- Sen1Floods11's India, Pakistan, Somalia and Sri Lanka (2016–2017) have no seasonal reference: the archive is too
  sparse a year earlier. On those events 2s equals 2e by construction.
- On CEMS tiles (16 × 17 km scenes) the event-pooled threshold fallback does not apply.

**A1.10 Cross-fitted conformal, stated precisely.**
- n counts windows, not events.
- Each calibration window comes from a fold model that was trained on the target event. The control is approximate,
  cross-conformal in style.

**A1.11 TerraMind's pre-training period** (P4). TerraMind-v1 was pre-trained on TerraMesh. Its SSL4EO-S12 v1.1 part
holds four seasonal samples per location from 2019–2021 (Blumenstiel et al., 2025, arXiv:2504.11172). Every CEMS
benchmark event is from 2023-03 or later.

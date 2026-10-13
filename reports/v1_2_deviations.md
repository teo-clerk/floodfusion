# floodfusion v1.2 — decisions and deviations log

The v1.2 protocol (`reports/protocol_v1_2.md`) is written at the freeze. Until then this log records, **before the
result it could affect exists**, every rule, decision and disclosure that bears on v1.2's claims. Dates are UTC; one
row per line.

## Disclosures

| Date (UTC) | Disclosure |
|---|---|
| 2026-10-13 | While researching v1.2, activation metadata (code, date, category, name) of CEMS codes EMSR899–945 was queried to estimate how fast a blind test set accrues. No product, Sentinel-1 pass, map or score of any of them was accessed. Floods among them: EMSR926 (Latvia/Lithuania), EMSR927 (Nepal), EMSR933 (Crete); storms EMSR928 and EMSR930. |
| 2026-10-13 | GFM availability was measured on the CEMS 2023–26 benchmark (STAC metadata only, no GFM pixel read): 82 of 90 pairs have a GFM product for the exact post-event frame; 8 have none (6 Sentinel-1C frames before GFM added Sentinel-1C in Oct 2025: EMSR796 AOI05/06, EMSR797 AOI01–03, EMSR843 AOI01; 2 Sentinel-1A frames: EMSR711 AOI01, EMSR712 AOI06). Versions: GFM v3.1.0 on 56 pairs, v4.0.0 on 4, v4.1.0 on 22. |
| 2026-10-13 | v1.1 erratum found before any v1.2 model was trained: TerraMind trained without modality dropout, and its BatchNorm saw sub-batches (`reports/v1_1_erratum.md`). |

## Rules fixed before any v1.2 result

| Date (UTC) | Rule | Reason |
|---|---|---|
| 2026-10-13 | Only the prospective blind test (CEMS codes ≥ EMSR899, scored after the freeze) is confirmatory; everything else in v1.2 is exploratory. | The CEMS 2023–26 benchmark has been seen, and public training sets (GEOID-Flood, ImpactMesh) now cover its activations. |
| 2026-10-13 | Model selection uses cross-validation only, with tuning on an inner split of fold 0's training groups (`assign_group_folds(train_records, 4)`). | `k_fold_groups` validates fold *i* on fold *i*+1's test groups, so tuning on that validation split would select on test groups. |
| 2026-10-13 | Equal tuning budget: at most three learning rates per model family (row 20, TerraMind-fair). | Fairness of the TerraMind comparison. |
| 2026-10-13 | TerraMind-fair size and adaptation: TerraMind-base with LoRA, unless a one-epoch probe projects more than 3 h per pooled cross-validation run; then TerraMind-base frozen, then TerraMind-small with LoRA. | GPU budget (8 GB laptop GPU, ~40–60 GPU-hours for all of v1.2), fixed before any TerraMind-fair result. |
| 2026-10-13 | If the largest 100 km group of the pooled Kuro Siwo + GEOID-Flood set holds more than 25 % of its windows, cross-validation uses buffered spatial blocks instead of groups. | A single giant group would leave folds unbalanced. |
| 2026-10-13 | GFM is a comparator, never a model input. A unit counts for a GFM comparison only if GFM covers ≥ 95 % of its scored pixels. | GFM changes its members in 2027; missing GFM pixels would otherwise count as misses. |
| 2026-10-13 | The CEMS 2023–26 benchmark is scored with v1.2 models once, after the freeze tag. | Prevents choices made after seeing it. |

## Probes (metadata only, before any v1.2 run)

| Date (UTC) | Probe | Result and consequence |
|---|---|---|
| 2026-10-13 | 100 km single-linkage groups of the pooled training floods: Kuro Siwo (36 events) plus GEOID-Flood events before EMSR656 not in Kuro Siwo (115 by code), from stack and tile footprints; windows estimated at 5 tiles per GEOID-Flood event × 8.8 windows per tile. | 71 groups; the largest holds about 12 % of the windows (13 events), the next about 10 %. Below the 25 % trigger, so cross-validation stays grouped (no buffered blocks). |
| 2026-10-13 | GFM sidecar on one stack (EMSR657 AOI01 t00_00). | All eight layers read; 18.7 s per stack; the ensemble likelihood is defined on non-flood pixels (0–50 there), so a threshold-free AP is possible for GFM. GFM's exclusion mask covers 61 % of that tile: GFM maps no flood there, which the non-excluded-area sensitivity analysis isolates. |
| 2026-10-13 | AlphaEarth read path (Source Cooperative, 2024 tile of zone 32N). | The `.vrt` beside each TIFF points at its source through `/vsis3/…` (S3), not HTTPS, so it does not open without AWS configuration. The TIFF opens over HTTPS: stored bottom-up (y pixel size +10), 1024² blocks, **band**-interleaved, overviews ×2–×16. Dequantised vectors have unit length (0.993–1.007). Two 64² windows took 2 min 22 s (64 separate band reads each): sidecars for every stack need a different read strategy (overviews or per-file downloads), and orientation still needs a landmark test before use. |
| 2026-10-13 | GFM sidecars for every CEMS 2023–26 stack (`make gfm`; GFM pixels read, no model scored). | 605 stacks: 557 with GFM, 48 without (the 8 pairs above). Versions by stack: v3.1.0 328, v4.0.0 24, v4.1.0 205. Three stacks' item lacks `exclusion_mask` (left nodata from that item, recorded). Median grid coverage 1.0; some stacks are 0 % covered where GFM's frame misses the tile, which the ≥ 95 % unit rule handles. |
| 2026-10-13 | GEOID-Flood selection and trial (labels only; no model). | Revision `868407460bf3`, every file sha256-checked; 12 949 label tiles. 111 events before EMSR656 not in Kuro Siwo (by code), 544 AOIs, 7 466 usable tiles; 12 AOIs have no post-event time in the shard index and are skipped. Labels are read on our grid (class 2 flood, 1 permanent water, 255 unlabelled); a candidate square needs ≥ 50 % mapped pixels (a GEOID tile can be 99.8 % unmapped although the catalogue rates it 63 % invalid). Trial on EMSR154: 4 stacks, flood 0.2–6.5 %, Planetary Computer pass within 13 s of GEOID-Flood's. |
| 2026-10-13 | Kuro Siwo's ingest refactored to take a label source (shared with GEOID-Flood). | Re-ingesting EMSR174 Skopje gives 5 stacks pixel-identical to the v1.1 ones, with identical metadata. |

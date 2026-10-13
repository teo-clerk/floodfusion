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

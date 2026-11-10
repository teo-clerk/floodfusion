# floodfusion v1.3 — decisions and deviations log

This log records, **before the result it could affect exists**, every rule, decision and disclosure that bears on
v1.3's claims (`reports/protocol_v1_3.md`). Dates are UTC; one row per line.

## Disclosures

| Date (UTC) | Disclosure |
|---|---|
| 2026-11-10 | v1.3 was designed with v1.2's results on the seen CEMS benchmark known (row 21: 0.417 at τ*, 0.468 at the oracle threshold, `reports/v12_cems_rebuilt.md`). The ladder's rungs were chosen because of them. Every v1.3 number on that benchmark is therefore exploratory; only H8 is confirmatory. |

## Rules fixed before any v1.3 result

| Date (UTC) | Rule | Reason |
|---|---|---|
| 2026-11-10 | The protocol `reports/protocol_v1_3.md`: the ladder's data, groups, folds, rungs, measure and adoption rule; v1.3's final model; H8 as P8 in its own family; the reference ceiling's set, metric, strata and reading; the live service's blind-flood rule. | Committed before any v1.3 model is trained and before any v1.3 or ceiling number exists. |

## Probes (metadata and counts only, before any v1.3 result)

| Date (UTC) | Probe | Result and consequence |
|---|---|---|
| 2026-11-10 | The ladder's data (`data/cems_benchmark_v12`; pair records and chip counts only, no score). | 90 pair folders: 76 radar-drawn, 13 optical, 1 mixed. Primary stratum without EMSR664: 75 pairs, 38 activations, 532 ingested tiles, 27,501 chips; all 38 activations hold reference flood on their scored pixels. 100 km single-linkage groups over activation footprints, time-agnostic: 30 groups, the largest holding 13.9 % of the chips (below the 25 % rule), so grouped five-fold CV applies. |
| 2026-11-10 | The groups with every stratum, after the author chose to train R3 on the optical and mixed strata too (counts only, no score). | 89 pairs (75 radar, 13 optical, 1 mixed) from 43 activations, 30,834 chips; 5 activations exist only in the optical or mixed strata, and one of them (EMSR754) shares a group with a radar-drawn activation. 34 groups, 30 holding radar-drawn pairs; the largest holds 12.4 % of the chips, below the 25 % rule. |
| 2026-11-10 | The reference ceiling's candidates (cached activation records EMSR656–898; product metadata only, no product downloaded). | 111 pairs of consecutive DEL products of one AOI with latest source images ≤ 24 h apart, in 45 flood activations. Most gaps are 11–13 h (morning and evening passes); one pair shares its source time (gap 0). |

# Kuro Siwo with the 'water' label — the same chips, label = flood ∪ reference water

Identical chips, folds and inputs to the flood-label table; only the label differs (Kuro Siwo's `hydro` layer added to its flood), the Sen1Floods11-like 'water on a date'. Scores here are on that label and are not comparable with the flood-label table; the comparison of the two training labels is test P3 on the CEMS benchmark (`reports/protocol_v1_1.md`).

| # | Configuration | IoU at τ* [95 % CI over events] | IoU, oracle τ | AP | F1 | Seeds | Events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | **Fusion, cross-attention (reference)** | 0.628 [0.58, 0.68] | 0.637 | 0.730 | 0.762 | 42, 43, 44 | 29 |
| 11 | Row 9, evaluated SAR-only at inference | 0.628 [0.58, 0.68] | 0.637 | 0.722 | 0.763 | 42, 43, 44 | 29 |

§ fewer than 8 groups: the interval is indicative only.

## Caveats

- **Descriptive.** See the flood-label table.

## Paired gaps over held-out groups (matched seeds)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip) | Verdict | Expected sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 − 11 | The cost of losing optical at inference (9 − 11) † | 42, 43, 44 | +0.000 [-0.008, +0.007] | 13/29 basins | 0.941 | not significant (Holm p = 0.941) | positive; small is good |

## Conformal recall control across groups (calibrated on each fold's validation groups)

| Row | Seed | α | Feasible folds | Events with mean chip FNR ≤ α (feasible folds) | Mean chip FNR | Worst event | Precision at τ_conf | IoU at τ_conf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 42 | 0.1 | 5/5 | 17/29 | 0.095 | KS_EMSR118_Spain (0.352) | 0.433 | 0.408 |
| 9 | 43 | 0.1 | 5/5 | 20/29 | 0.088 | KS_EMSR174_Skopje (0.435) | 0.458 | 0.447 |
| 9 | 44 | 0.1 | 5/5 | 19/29 | 0.106 | KS_EMSR174_Skopje (0.467) | 0.467 | 0.456 |
| 11 | 42 | 0.1 | 5/5 | 18/29 | 0.089 | KS_EMSR118_Spain (0.354) | 0.431 | 0.407 |
| 11 | 43 | 0.1 | 5/5 | 20/29 | 0.088 | KS_EMSR174_Skopje (0.469) | 0.458 | 0.448 |
| 11 | 44 | 0.1 | 5/5 | 19/29 | 0.102 | KS_EMSR174_Skopje (0.414) | 0.463 | 0.453 |

## v1.1 variants — descriptive gaps (not part of the gap family; no verdict)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip, unadjusted) | Expected sign |
| --- | --- | --- | --- | --- | --- | --- |

## Calibration across events: each held-out event calibrated on the others (cross-fitted)

The cost-optimal threshold and the conformal threshold of each held-out event are fitted on the pooled held-out predictions of the *other* events (each from its own fold model) instead of the fold's single validation event. Nothing is re-run: the stored confusion counts are re-read (`floodfusion.crossfit`). n counts windows, not events, and each calibration window comes from a fold model that was trained on the target event: the control is approximate (cross-conformal in style), not a finite-sample guarantee.

| Row | Seed | K | IoU at τ* (own validation) | IoU at cross-fitted τ | IoU, best τ per group | Conformal: events with FNR ≤ α, own → cross-fitted | Mean chip FNR, cross-fitted | Precision at τ_conf, cross-fitted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 42 | 29 | 0.619 | 0.620 | 0.657 | 18/29 → 22/29 | 0.081 | 0.411 |
| 11 | 43 | 29 | 0.629 | 0.627 | 0.660 | 20/29 → 21/29 | 0.087 | 0.400 |
| 11 | 44 | 29 | 0.640 | 0.627 | 0.663 | 19/29 → 20/29 | 0.086 | 0.434 |
| 9 | 42 | 29 | 0.617 | 0.618 | 0.659 | 17/29 → 21/29 | 0.086 | 0.391 |
| 9 | 43 | 29 | 0.630 | 0.628 | 0.661 | 20/29 → 21/29 | 0.085 | 0.395 |
| 9 | 44 | 29 | 0.641 | 0.628 | 0.666 | 19/29 → 24/29 | 0.071 | 0.381 |

Event-level conformal control with n = 28 calibration events tolerates a calibration risk of at most +0.068 at α = 0.1 (≤ 0: vacuous — flag everything); the cross-fitted threshold above is chip-level control under pooling.

## Per-group IoU at τ*

| Row | KS_EMSR1111004_USA | KS_EMSR1111005_Madagascar | KS_EMSR1111006_Madagascar | KS_EMSR1111007_Nepal | KS_EMSR1111009_Pakistan | KS_EMSR1111010_Peru | KS_EMSR1111011_Philippines | KS_EMSR118_Spain | KS_EMSR147_Cumbria | KS_EMSR174_Skopje | KS_EMSR205_SriLanka | KS_EMSR267_Lithuania | KS_EMSR273_Albania | KS_EMSR275_Croatia | KS_EMSR277_Greece | KS_EMSR321_Honduras | KS_EMSR324_France | KS_EMSR411_France | KS_EMSR417_Portugal | KS_EMSR427_Sweden | KS_EMSR445_Romania | KS_EMSR470_Togo | KS_EMSR496_Italy | KS_EMSR497_Germany | KS_EMSR498_France | KS_EMSR554_Australia | KS_EMSR561_Malawi | KS_EMSR562_Australia | KS_EMSR567_Australia |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.507 | 0.715 | 0.722 | 0.610 | 0.698 | 0.273 | 0.736 | 0.480 | 0.596 | 0.440 | 0.506 | 0.535 | 0.493 | 0.628 | 0.793 | 0.617 | 0.516 | 0.653 | 0.782 | 0.517 | 0.640 | 0.802 | 0.440 | 0.667 | 0.631 | 0.740 | 0.814 | 0.894 | 0.778 |
| 11 | 0.504 | 0.715 | 0.722 | 0.610 | 0.632 | 0.274 | 0.739 | 0.479 | 0.596 | 0.524 | 0.506 | 0.530 | 0.493 | 0.628 | 0.787 | 0.617 | 0.513 | 0.653 | 0.783 | 0.517 | 0.631 | 0.802 | 0.440 | 0.670 | 0.631 | 0.740 | 0.814 | 0.891 | 0.777 |

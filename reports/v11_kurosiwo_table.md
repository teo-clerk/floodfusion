# Kuro Siwo, re-ingested — grouped five-fold, flood labels drawn on radar

Real data: Kuro Siwo annotation polygons (Bountos et al., NeurIPS 2024; CC-BY-4.0) rasterised on our grid, every input re-fetched from Planetary Computer by `floodfusion.data.kurosiwo`; chips sampled per event by a seeded rule. Label: **flood** — water the event brought — minus JRC permanent water. Folds group events whose AOIs lie within 100 km; each group is tested once by a model that never saw it; validation = the next fold. Protocol: `reports/protocol_v1_1.md`.

| # | Configuration | IoU at τ* [95 % CI over events] | IoU, oracle τ | AP | F1 | Seeds | Events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Global Otsu on VV, no masks | 0.349 [0.30, 0.40] | — | 0.457 | 0.500 | — | 29 |
| 2 | Baseline: adaptive threshold + change + masks | 0.417 [0.34, 0.49] | — | 0.374 | 0.552 | — | 29 |
| 3 | Baseline + MNDWI where cloud-free | 0.442 [0.37, 0.51] | — | 0.381 | 0.584 | — | 29 |
| 4 | U-Net, SAR only (VV, VH) | 0.505 [0.44, 0.57] | 0.526 | 0.632 | 0.651 | 42, 43, 44 | 29 |
| 9 | **Fusion, cross-attention (reference)** | 0.535 [0.48, 0.59] | 0.564 | 0.663 | 0.685 | 42, 43, 44 | 29 |
| 11 | Row 9, evaluated SAR-only at inference | 0.533 [0.48, 0.58] | 0.563 | 0.663 | 0.683 | 42, 43, 44 | 29 |
| row02e_event_fallback | 2e — row 2, event-pooled threshold where a chip has no bimodal tile | 0.432 [0.35, 0.51] | — | 0.384 | 0.566 | — | 29 |
| row02s_seasonal | 2s — row 2e with the seasonal reference (same date, a year earlier) | 0.520 [0.46, 0.58] | — | 0.523 | 0.665 | — | 29 |
| row09b_gate_bias0 | 9b — row 9, gate starting half open (bias 0) | 0.498 [0.29, 0.70]§ | 0.541 | 0.537 | 0.633 | 42 | 4 |
| row09c_cloud_augment | 9c — row 9, synthetic clouds on half the optical training chips | 0.537 [0.39, 0.68]§ | 0.538 | 0.522 | 0.686 | 42 | 4 |
| row09r_sar_query | 9r — row 9, reverse assignment (Q from SAR, K = V from optical) | 0.477 [0.25, 0.68]§ | 0.538 | 0.512 | 0.608 | 42 | 4 |

§ fewer than 8 groups: the interval is indicative only.

## Caveats

- **Folds hold several groups.** The oracle-threshold IoU, AP and the other fold-averaged columns of the main table are fold-level quantities (each fold weighted equally), not per-group macros; IoU at τ* and every gap are per group. The baseline variants' event-pooled threshold pools a split group (events within 100 km), not one acquisition.
- **Labels drawn on radar.** Kuro Siwo's annotators drew on Sentinel-1: gaps involving optical understate optical's value here, as Sen1Floods11's overstate it; the CEMS benchmark is the independent check.
- **Descriptive.** The confirmatory tests of v1.1 are on the CEMS benchmark; nothing here is adjusted into that family.

## Paired gaps over held-out groups (matched seeds)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip) | Verdict | Expected sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 − 1 | What careful engineering adds to naive Otsu | — | +0.068 [-0.011, +0.145] | 18/29 basins | 0.112 | not significant (Holm p = 0.224) | positive |
| 3 − 2 | What optical adds to the physics baseline † | — | +0.024 [-0.046, +0.095] | 10/29 basins | 0.524 | not significant (Holm p = 0.524) | positive where optical is clear |
| 4 − 2 | What learning adds over the physics baseline (SAR only) | 42, 43, 44 | +0.088 [+0.020, +0.166] | 17/29 basins | 0.024 | not significant (Holm p = 0.146) | positive |
| 9 − 4 | What fusion adds over the SAR-only U-Net (R3) † | 42, 43, 44 | +0.030 [+0.004, +0.063] | 21/29 basins | 0.036 | not significant (Holm p = 0.163) | positive where optical is clear |
| 9 − 11 | The cost of losing optical at inference (9 − 11) † | 42, 43, 44 | +0.002 [+0.000, +0.005] | 18/29 basins | 0.033 | not significant (Holm p = 0.163) | positive; small is good |
| 11 − 4 | The fusion model run SAR-only vs the SAR-only U-Net | 42, 43, 44 | +0.027 [+0.002, +0.061] | 21/29 basins | 0.060 | not significant (Holm p = 0.179) | ≈ 0 or positive |
| 9 − 2 | What the machine learning was worth (9 − 2) | 42, 43, 44 | +0.118 [+0.063, +0.180] | 20/29 basins | 0.000 | significant (Holm p = 0.001) | positive |

## Conformal recall control across groups (calibrated on each fold's validation groups)

| Row | Seed | α | Feasible folds | Events with mean chip FNR ≤ α (feasible folds) | Mean chip FNR | Worst event | Precision at τ_conf | IoU at τ_conf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 42 | 0.1 | 5/5 | 24/29 | 0.064 | KS_EMSR427_Sweden (0.321) | 0.331 | 0.326 |
| 4 | 43 | 0.1 | 5/5 | 22/29 | 0.069 | KS_EMSR427_Sweden (0.267) | 0.365 | 0.361 |
| 4 | 44 | 0.1 | 5/5 | 20/29 | 0.083 | KS_EMSR427_Sweden (0.378) | 0.375 | 0.369 |
| 9 | 42 | 0.1 | 5/5 | 20/29 | 0.078 | KS_EMSR427_Sweden (0.320) | 0.334 | 0.330 |
| 9 | 43 | 0.1 | 5/5 | 24/29 | 0.061 | KS_EMSR427_Sweden (0.353) | 0.314 | 0.309 |
| 9 | 44 | 0.1 | 5/5 | 21/29 | 0.077 | KS_EMSR427_Sweden (0.472) | 0.378 | 0.374 |
| 11 | 42 | 0.1 | 5/5 | 22/29 | 0.068 | KS_EMSR427_Sweden (0.321) | 0.327 | 0.322 |
| 11 | 43 | 0.1 | 5/5 | 24/29 | 0.060 | KS_EMSR427_Sweden (0.347) | 0.312 | 0.307 |
| 11 | 44 | 0.1 | 5/5 | 21/29 | 0.077 | KS_EMSR427_Sweden (0.472) | 0.379 | 0.375 |

## Physics baseline: what each rule component costs on these labels

The tile-adaptive threshold fell back to global Otsu on 376 of 890 scenes (a 5 km chip offers few tiles with a bimodal histogram).

| Row 2 variant | Macro IoU over events | Pooled IoU |
| --- | --- | --- |
| all_masks | 0.417 | 0.361 |
| no_change_term | 0.275 | 0.220 |
| no_hand | 0.413 | 0.359 |
| no_layover_shadow | 0.417 | 0.361 |
| no_permanent_water | 0.417 | 0.361 |
| no_slope | 0.414 | 0.360 |
| or_rule | 0.315 | 0.261 |

Scenes of the v1.1 baseline variants by threshold fallback and change reference actually used:

| Variant | Counts |
| --- | --- |
| row02e_event_fallback | fallback: event: 376, fallback: none: 514, reference: recent: 890 |
| row02s_seasonal | fallback: event: 376, fallback: none: 514, reference: recent: 166, reference: seasonal: 724 |

## v1.1 variants — descriptive gaps (not part of the gap family; no verdict)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip, unadjusted) | Expected sign |
| --- | --- | --- | --- | --- | --- | --- |
| row09b_gate_bias0 − 9 | Gate starting half open (9b − 9) | | not comparable | | | open |
| row09c_cloud_augment − 9 | Synthetic clouds in training (9c − 9) | | not comparable | | | open |
| row09r_sar_query − 9 | Reverse assignment (9r − 9) | | not comparable | | | open |
| row02e_event_fallback − 2 | Event context for the baseline's threshold (2e − 2) | — | +0.015 [+0.006, +0.027] | 24/29 | 0.000 | ≥ 0 |
| row02s_seasonal − row02e_event_fallback | Seasonal reference for the change term (2s − 2e) | — | +0.087 [+0.025, +0.163] | 17/29 | 0.013 | positive |

## Calibration across events: each held-out event calibrated on the others (cross-fitted)

The cost-optimal threshold and the conformal threshold of each held-out event are fitted on the pooled held-out predictions of the *other* events (each from its own fold model) instead of the fold's single validation event. Nothing is re-run: the stored confusion counts are re-read (`floodfusion.crossfit`). n counts windows, not events, and each calibration window comes from a fold model that was trained on the target event: the control is approximate (cross-conformal in style), not a finite-sample guarantee.

| Row | Seed | K | IoU at τ* (own validation) | IoU at cross-fitted τ | IoU, best τ per group | Conformal: events with FNR ≤ α, own → cross-fitted | Mean chip FNR, cross-fitted | Precision at τ_conf, cross-fitted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 42 | 29 | 0.518 | 0.538 | 0.600 | 22/29 → 20/29 | 0.079 | 0.357 |
| 11 | 43 | 29 | 0.527 | 0.540 | 0.602 | 24/29 → 22/29 | 0.066 | 0.323 |
| 11 | 44 | 29 | 0.556 | 0.555 | 0.598 | 21/29 → 22/29 | 0.065 | 0.346 |
| 4 | 42 | 29 | 0.501 | 0.508 | 0.578 | 24/29 → 23/29 | 0.077 | 0.377 |
| 4 | 43 | 29 | 0.515 | 0.524 | 0.583 | 22/29 → 20/29 | 0.080 | 0.378 |
| 4 | 44 | 29 | 0.504 | 0.512 | 0.585 | 20/29 → 23/29 | 0.056 | 0.337 |
| 9 | 42 | 29 | 0.523 | 0.543 | 0.602 | 20/29 → 20/29 | 0.081 | 0.361 |
| 9 | 43 | 29 | 0.528 | 0.538 | 0.603 | 24/29 → 22/29 | 0.065 | 0.321 |
| 9 | 44 | 29 | 0.556 | 0.556 | 0.599 | 21/29 → 22/29 | 0.066 | 0.349 |
| row09b_gate_bias0 | 42 | 4 | 0.498 | 0.527 | 0.555 | 3/4 → 3/4 | 0.084 | 0.336 |
| row09c_cloud_augment | 42 | 4 | 0.537 | 0.535 | 0.552 | 3/4 → 3/4 | 0.090 | 0.383 |
| row09r_sar_query | 42 | 4 | 0.477 | 0.537 | 0.566 | 3/4 → 3/4 | 0.082 | 0.307 |

Event-level conformal control with n = 28 calibration events tolerates a calibration risk of at most +0.068 at α = 0.1 (≤ 0: vacuous — flag everything); the cross-fitted threshold above is chip-level control under pooling.

## Per-group IoU at τ*

| Row | KS_EMSR1111004_USA | KS_EMSR1111005_Madagascar | KS_EMSR1111006_Madagascar | KS_EMSR1111007_Nepal | KS_EMSR1111009_Pakistan | KS_EMSR1111010_Peru | KS_EMSR1111011_Philippines | KS_EMSR118_Spain | KS_EMSR147_Cumbria | KS_EMSR174_Skopje | KS_EMSR205_SriLanka | KS_EMSR267_Lithuania | KS_EMSR273_Albania | KS_EMSR275_Croatia | KS_EMSR277_Greece | KS_EMSR321_Honduras | KS_EMSR324_France | KS_EMSR411_France | KS_EMSR417_Portugal | KS_EMSR427_Sweden | KS_EMSR445_Romania | KS_EMSR470_Togo | KS_EMSR496_Italy | KS_EMSR497_Germany | KS_EMSR498_France | KS_EMSR554_Australia | KS_EMSR561_Malawi | KS_EMSR562_Australia | KS_EMSR567_Australia |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.458 | 0.249 | 0.222 | 0.294 | 0.399 | 0.172 | 0.570 | 0.250 | 0.162 | 0.068 | 0.300 | 0.471 | 0.220 | 0.426 | 0.560 | 0.363 | 0.255 | 0.210 | 0.533 | 0.151 | 0.414 | 0.441 | 0.463 | 0.315 | 0.193 | 0.437 | 0.508 | 0.366 | 0.652 |
| 2 | 0.405 | 0.605 | 0.478 | 0.407 | 0.040 | 0.089 | 0.735 | 0.367 | 0.442 | 0.519 | 0.435 | 0.240 | 0.175 | 0.268 | 0.437 | 0.554 | 0.386 | 0.495 | 0.694 | 0.048 | 0.483 | 0.012 | 0.407 | 0.647 | 0.554 | 0.548 | 0.747 | 0.100 | 0.781 |
| 3 | 0.291 | 0.605 | 0.478 | 0.305 | 0.491 | 0.137 | 0.675 | 0.374 | 0.442 | 0.029 | 0.435 | 0.517 | 0.175 | 0.269 | 0.680 | 0.437 | 0.146 | 0.495 | 0.696 | 0.065 | 0.413 | 0.405 | 0.407 | 0.619 | 0.547 | 0.548 | 0.744 | 0.635 | 0.745 |
| 4 | 0.509 | 0.688 | 0.429 | 0.387 | 0.476 | 0.096 | 0.682 | 0.374 | 0.196 | 0.739 | 0.460 | 0.342 | 0.391 | 0.360 | 0.760 | 0.548 | 0.490 | 0.646 | 0.660 | 0.262 | 0.436 | 0.740 | 0.216 | 0.560 | 0.579 | 0.529 | 0.730 | 0.609 | 0.759 |
| 9 | 0.529 | 0.647 | 0.480 | 0.391 | 0.446 | 0.241 | 0.684 | 0.437 | 0.576 | 0.715 | 0.498 | 0.402 | 0.411 | 0.413 | 0.739 | 0.534 | 0.523 | 0.661 | 0.688 | 0.239 | 0.520 | 0.644 | 0.309 | 0.584 | 0.589 | 0.534 | 0.744 | 0.558 | 0.776 |
| 11 | 0.533 | 0.648 | 0.480 | 0.391 | 0.438 | 0.238 | 0.684 | 0.436 | 0.576 | 0.715 | 0.498 | 0.369 | 0.411 | 0.413 | 0.737 | 0.534 | 0.517 | 0.661 | 0.687 | 0.239 | 0.516 | 0.637 | 0.309 | 0.584 | 0.590 | 0.537 | 0.744 | 0.551 | 0.776 |
| row02e_event_fallback | 0.424 | 0.606 | 0.479 | 0.407 | 0.041 | 0.088 | 0.737 | 0.383 | 0.446 | 0.644 | 0.449 | 0.240 | 0.180 | 0.276 | 0.438 | 0.571 | 0.407 | 0.580 | 0.695 | 0.049 | 0.507 | 0.012 | 0.446 | 0.642 | 0.624 | 0.548 | 0.738 | 0.101 | 0.785 |
| row02s_seasonal | 0.408 | 0.578 | 0.479 | 0.352 | 0.638 | 0.207 | 0.670 | 0.371 | 0.444 | 0.586 | 0.449 | 0.237 | 0.429 | 0.576 | 0.714 | 0.584 | 0.536 | 0.626 | 0.822 | 0.182 | 0.579 | 0.759 | 0.539 | 0.645 | 0.637 | 0.331 | 0.705 | 0.199 | 0.787 |
| row09b_gate_bias0 | — | — | — | — | — | 0.176 | — | 0.422 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.647 | — | 0.748 | — | — |
| row09c_cloud_augment | — | — | — | — | — | 0.371 | — | 0.411 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.636 | — | 0.730 | — | — |
| row09r_sar_query | — | — | — | — | — | 0.129 | — | 0.411 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.613 | — | 0.754 | — | — |

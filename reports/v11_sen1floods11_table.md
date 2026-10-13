# Sen1Floods11, re-ingested — leave-one-event-out and the published split

Real data: Sen1Floods11 hand labels (Bonafilia et al., 2020) with every input re-fetched from Planetary Computer by `floodfusion.data.sen1floods11` (same-orbit pre-event RTC, S2 L2A, GLO-30, JRC, WorldCover; optical and labels moved onto the SAR geometry). Flood = labelled water minus JRC permanent water. IoU at τ* = 0.25 after calibration on each fold's validation event.

**Protocol: leave-one-event-out.** 10 folds (Paraguay dropped: its acquisition is not in the archive); each event is tested once by a model that never saw it; validation = the next event (sorted, cyclic), fixed before any run. **[CI]** = 95 % percentile bootstrap over events (descriptive); gap verdicts are exact sign-flip permutation tests over events, Holm-adjusted across the table.

| # | Configuration | IoU at τ* [95 % CI over events] | IoU, oracle τ | AP | F1 | Seeds | Events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Global Otsu on VV, no masks | 0.299 [0.20, 0.41] | — | 0.451 | 0.436 | — | 10 |
| 2 | Baseline: adaptive threshold + change + masks | 0.157 [0.09, 0.24] | — | 0.152 | 0.254 | — | 10 |
| 3 | Baseline + MNDWI where cloud-free | 0.444 [0.37, 0.52] | — | 0.440 | 0.605 | — | 10 |
| 4 | U-Net, SAR only (VV, VH) | 0.343 [0.22, 0.49] | 0.457 | 0.514 | 0.478 | 42, 43, 44 | 10 |
| 9 | **Fusion, cross-attention (reference)** | 0.608 [0.54, 0.67] | 0.662 | 0.758 | 0.751 | 42, 43, 44 | 10 |
| 11 | Row 9, evaluated SAR-only at inference | 0.368 [0.25, 0.50] | 0.491 | 0.555 | 0.506 | 42, 43, 44 | 10 |
| TM | TerraMind-1.0-small, frozen + UNet decoder (single date: post S1 + S2) ‡ | 0.572 [0.47, 0.68] | 0.591 | 0.759 | 0.713 | 42, 43, 44 | 10 |
| 12 | Row 9 − compound loss (plain BCE) | 0.616 [0.54, 0.69] | 0.658 | 0.788 | 0.756 | 42, 43, 44 | 10 |
| cmp_terramind_small__eval_sar_only | TM, evaluated SAR-only at inference (its own SAR-only calibration) | 0.413 [0.29, 0.54] | 0.464 | 0.598 | 0.556 | 42, 43, 44 | 10 |
| row02e_event_fallback | 2e — row 2, event-pooled threshold where a chip has no bimodal tile | 0.161 [0.09, 0.24] | — | 0.162 | 0.259 | — | 10 |
| row02s_seasonal | 2s — row 2e with the seasonal reference (same date, a year earlier) | 0.336 [0.21, 0.46] | — | 0.342 | 0.467 | — | 10 |
| row09b_gate_bias0 | 9b — row 9, gate starting half open (bias 0) | 0.618 [0.55, 0.69] | 0.671 | 0.779 | 0.757 | 42 | 10 |
| row09c_cloud_augment | 9c — row 9, synthetic clouds on half the optical training chips | 0.624 [0.55, 0.71] | 0.662 | 0.763 | 0.761 | 42 | 10 |
| row09r_sar_query | 9r — row 9, reverse assignment (Q from SAR, K = V from optical) | 0.654 [0.58, 0.72] | 0.678 | 0.778 | 0.785 | 42 | 10 |

§ fewer than 8 events: the interval is indicative only.

## Caveats

- † **Optical is the annotators' source image.** Sen1Floods11 labels were drawn mostly on the Sentinel-2 image of the same date, and that image is re-fetched as the optical input. Gaps involving optical therefore measure agreement with the annotators' source as well as the value of optical; the EMSR664 case study (independent CEMS delineations) is the independent check.
- ‡ **TerraMind** saw global Sentinel-1/2 data in pre-training ("never saw the event" holds for its decoder only); its band subset and necks follow IBM's recipe, developed on Sen1Floods11's official splits; its schedule is row 9's, tuned for neither model; its SAR input is our z-score adapted by four affine parameters; clouded pixels reach it as its pre-training mean (it has no per-pixel gate).
- **Labels mark water on a date; the reference may already be wet.** In most events the flood-labelled pixels show no backscatter decrease against the reference — the latest same-orbit pass, 5-24 days earlier, which can itself fall inside the flood. [`sen1floods11_label_change.md`](sen1floods11_label_change.md) gives the count and separates water already there a year earlier from a flood that began before the reference. Change-based rules (rows 2, 3) are penalised by both; amplitude-based ones are not.
- **Validation is one event per fold** (the next, fixed in advance): early stopping, the prior, the calibration and the baseline offset come from it, so part of a fold's score is threshold transfer — compare IoU at τ* with the oracle-threshold IoU and AP.
- **Development runs** scored one fold (Bolivia; rows 1-4) before the protocol was frozen; every later change came from code review, none from those numbers.

## Paired gaps over events (matched seeds)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip) | Verdict | Expected sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 − 1 | What careful engineering adds to naive Otsu | — | -0.141 [-0.274, -0.011] | 3/10 basins | 0.078 | not significant (Holm p = 0.234) | positive |
| 3 − 2 | What optical adds to the physics baseline † | — | +0.286 [+0.188, +0.390] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | positive where optical is clear |
| 4 − 2 | What learning adds over the physics baseline (SAR only) | 42, 43, 44 | +0.186 [+0.049, +0.338] | 7/10 basins | 0.035 | not significant (Holm p = 0.141) | positive |
| 9 − 4 | What fusion adds over the SAR-only U-Net (R3) † | 42, 43, 44 | +0.265 [+0.168, +0.362] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | positive where optical is clear |
| 9 − 11 | The cost of losing optical at inference (9 − 11) † | 42, 43, 44 | +0.240 [+0.147, +0.343] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | positive; small is good |
| 11 − 4 | The fusion model run SAR-only vs the SAR-only U-Net | 42, 43, 44 | +0.024 [-0.007, +0.065] | 6/10 basins | 0.248 | not significant (Holm p = 0.496) | ≈ 0 or positive |
| 9 − 2 | What the machine learning was worth (9 − 2) | 42, 43, 44 | +0.451 [+0.370, +0.531] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | positive |
| cmp_terramind_small − 4 | TerraMind (frozen; post-event S1 + S2) vs the SAR-only U-Net ‡ | 42, 43, 44 | +0.229 [+0.140, +0.329] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | open |
| 9 − cmp_terramind_small | What the task-specific fusion model adds over the foundation model ‡ | 42, 43, 44 | +0.036 [-0.047, +0.123] | 6/10 basins | 0.479 | not significant (Holm p = 0.496) | ≈ 0 on Sen1Floods11 (PANGAEA: supervised U-Nets match GFMs there) |

## The published split's protocol gap, measured on the same chips

The dataset's official test chips come from the **same events** as its training chips. Each event's official test chips are scored twice: by the model trained on the official split, and by the LOEO model that never saw the event. Δ combines leakage with the protocols' other differences (training-set size, validation set, calibration): at τ* it includes each model's calibration; at the oracle threshold and in AP it does not.

| Row | Seed | Events | Measure | Official split | LOEO, same chips | Δ (official − LOEO) | Official higher in | p (sign-flip) | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 42 | 9 | IoU at τ* | 0.410 | 0.350 | +0.059 | 5/9 | 0.422 | 1.000 |
| 4 | 42 | 9 | IoU, oracle τ | 0.525 | 0.472 | +0.053 | 7/9 | 0.047 | 1.000 |
| 4 | 42 | 9 | AP | 0.562 | 0.510 | +0.052 | 6/9 | 0.195 | 1.000 |
| 4 | 43 | 9 | IoU at τ* | 0.443 | 0.337 | +0.106 | 6/9 | 0.266 | 1.000 |
| 4 | 43 | 9 | IoU, oracle τ | 0.526 | 0.434 | +0.092 | 6/9 | 0.188 | 1.000 |
| 4 | 43 | 9 | AP | 0.593 | 0.487 | +0.106 | 6/9 | 0.184 | 1.000 |
| 4 | 44 | 9 | IoU at τ* | 0.393 | 0.371 | +0.022 | 6/9 | 0.695 | 1.000 |
| 4 | 44 | 9 | IoU, oracle τ | 0.505 | 0.466 | +0.039 | 8/9 | 0.016 | 0.562 |
| 4 | 44 | 9 | AP | 0.568 | 0.505 | +0.062 | 7/9 | 0.070 | 1.000 |
| 9 | 42 | 9 | IoU at τ* | 0.604 | 0.631 | -0.027 | 4/9 | 0.418 | 1.000 |
| 9 | 42 | 9 | IoU, oracle τ | 0.656 | 0.662 | -0.006 | 5/9 | 0.742 | 1.000 |
| 9 | 42 | 9 | AP | 0.736 | 0.735 | +0.001 | 4/9 | 0.965 | 1.000 |
| 9 | 43 | 9 | IoU at τ* | 0.661 | 0.611 | +0.050 | 7/9 | 0.031 | 1.000 |
| 9 | 43 | 9 | IoU, oracle τ | 0.693 | 0.666 | +0.027 | 7/9 | 0.074 | 1.000 |
| 9 | 43 | 9 | AP | 0.770 | 0.737 | +0.032 | 6/9 | 0.352 | 1.000 |
| 9 | 44 | 9 | IoU at τ* | 0.632 | 0.575 | +0.058 | 6/9 | 0.191 | 1.000 |
| 9 | 44 | 9 | IoU, oracle τ | 0.671 | 0.663 | +0.008 | 5/9 | 0.621 | 1.000 |
| 9 | 44 | 9 | AP | 0.764 | 0.743 | +0.021 | 4/9 | 0.469 | 1.000 |
| 11 | 42 | 9 | IoU at τ* | 0.453 | 0.354 | +0.099 | 6/9 | 0.227 | 1.000 |
| 11 | 42 | 9 | IoU, oracle τ | 0.523 | 0.520 | +0.002 | 5/9 | 0.941 | 1.000 |
| 11 | 42 | 9 | AP | 0.593 | 0.571 | +0.021 | 5/9 | 0.594 | 1.000 |
| 11 | 43 | 9 | IoU at τ* | 0.515 | 0.414 | +0.101 | 8/9 | 0.051 | 1.000 |
| 11 | 43 | 9 | IoU, oracle τ | 0.564 | 0.532 | +0.032 | 7/9 | 0.066 | 1.000 |
| 11 | 43 | 9 | AP | 0.636 | 0.595 | +0.041 | 4/9 | 0.156 | 1.000 |
| 11 | 44 | 9 | IoU at τ* | 0.447 | 0.347 | +0.100 | 6/9 | 0.191 | 1.000 |
| 11 | 44 | 9 | IoU, oracle τ | 0.519 | 0.508 | +0.011 | 5/9 | 0.660 | 1.000 |
| 11 | 44 | 9 | AP | 0.587 | 0.562 | +0.025 | 5/9 | 0.406 | 1.000 |
| cmp_terramind_small | 42 | 9 | IoU at τ* | 0.606 | 0.603 | +0.003 | 6/9 | 0.832 | 1.000 |
| cmp_terramind_small | 42 | 9 | IoU, oracle τ | 0.623 | 0.620 | +0.003 | 6/9 | 0.777 | 1.000 |
| cmp_terramind_small | 42 | 9 | AP | 0.746 | 0.743 | +0.003 | 4/9 | 0.867 | 1.000 |
| cmp_terramind_small | 43 | 9 | IoU at τ* | 0.576 | 0.612 | -0.036 | 5/9 | 0.332 | 1.000 |
| cmp_terramind_small | 43 | 9 | IoU, oracle τ | 0.607 | 0.628 | -0.021 | 5/9 | 0.449 | 1.000 |
| cmp_terramind_small | 43 | 9 | AP | 0.722 | 0.747 | -0.025 | 5/9 | 0.422 | 1.000 |
| cmp_terramind_small | 44 | 9 | IoU at τ* | 0.568 | 0.620 | -0.052 | 5/9 | 0.297 | 1.000 |
| cmp_terramind_small | 44 | 9 | IoU, oracle τ | 0.591 | 0.630 | -0.038 | 5/9 | 0.352 | 1.000 |
| cmp_terramind_small | 44 | 9 | AP | 0.717 | 0.751 | -0.034 | 5/9 | 0.363 | 1.000 |

## Bolivia — the dataset's own held-out event

| Row | Seed | IoU on Bolivia, official-split model | IoU on Bolivia, LOEO fold |
| --- | --- | --- | --- |
| 4 | 42 | 0.467 | 0.652 |
| 4 | 43 | 0.533 | 0.641 |
| 4 | 44 | 0.418 | 0.660 |
| 9 | 42 | 0.667 | 0.756 |
| 9 | 43 | 0.717 | 0.720 |
| 9 | 44 | 0.702 | 0.584 |
| 11 | 42 | 0.598 | 0.636 |
| 11 | 43 | 0.692 | 0.691 |
| 11 | 44 | 0.599 | 0.624 |
| cmp_terramind_small | 42 | 0.677 | 0.757 |
| cmp_terramind_small | 43 | 0.663 | 0.742 |
| cmp_terramind_small | 44 | 0.720 | 0.683 |

## Conformal recall control across events

Each fold fits the conformal threshold on its validation event's chips (≥ 100 flood pixels; risk = per-chip false-negative rate; guarantee E[risk] ≤ α for chips exchangeable with those — windows of one source chip are not independent, and the same chips also selected the checkpoint). The held-out event is a different flood: this measures how far the guarantee carries.

| Row | Seed | α | Feasible folds | Events with mean chip FNR ≤ α (feasible folds) | Mean chip FNR | Worst event | Precision at τ_conf | IoU at τ_conf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 42 | 0.1 | 10/10 | 7/10 | 0.063 | S1F11_India (0.210) | 0.198 | 0.195 |
| 4 | 43 | 0.1 | 10/10 | 7/10 | 0.074 | S1F11_India (0.245) | 0.191 | 0.190 |
| 4 | 44 | 0.1 | 10/10 | 7/10 | 0.066 | S1F11_India (0.125) | 0.189 | 0.187 |
| 9 | 42 | 0.1 | 10/10 | 5/10 | 0.088 | S1F11_Somalia (0.220) | 0.233 | 0.232 |
| 9 | 43 | 0.1 | 10/10 | 7/10 | 0.082 | S1F11_Ghana (0.254) | 0.261 | 0.260 |
| 9 | 44 | 0.1 | 10/10 | 7/10 | 0.082 | S1F11_Somalia (0.230) | 0.199 | 0.198 |
| 11 | 42 | 0.1 | 10/10 | 7/10 | 0.081 | S1F11_India (0.284) | 0.189 | 0.187 |
| 11 | 43 | 0.1 | 10/10 | 7/10 | 0.075 | S1F11_USA (0.217) | 0.231 | 0.229 |
| 11 | 44 | 0.1 | 10/10 | 6/10 | 0.104 | S1F11_India (0.326) | 0.209 | 0.206 |
| cmp_terramind_small | 42 | 0.1 | 10/10 | 6/10 | 0.091 | S1F11_Somalia (0.290) | 0.282 | 0.280 |
| cmp_terramind_small | 43 | 0.1 | 10/10 | 6/10 | 0.082 | S1F11_India (0.200) | 0.269 | 0.267 |
| cmp_terramind_small | 44 | 0.1 | 10/10 | 7/10 | 0.103 | S1F11_Somalia (0.310) | 0.311 | 0.308 |

## Physics baseline: what each rule component costs on these labels

The tile-adaptive threshold fell back to global Otsu on 428 of 680 scenes (a 5 km chip offers few tiles with a bimodal histogram).

| Row 2 variant | Macro IoU over events | Pooled IoU |
| --- | --- | --- |
| all_masks | 0.157 | 0.152 |
| no_change_term | 0.198 | 0.132 |
| no_hand | 0.157 | 0.152 |
| no_layover_shadow | 0.157 | 0.152 |
| no_permanent_water | 0.157 | 0.152 |
| no_slope | 0.158 | 0.152 |
| or_rule | 0.225 | 0.150 |

Scenes of the v1.1 baseline variants by threshold fallback and change reference actually used:

| Variant | Counts |
| --- | --- |
| row02e_event_fallback | fallback: event: 428, fallback: none: 252, reference: recent: 680 |
| row02s_seasonal | fallback: event: 428, fallback: none: 252, reference: recent: 252, reference: seasonal: 428 |

## v1.1 variants — descriptive gaps (not part of the gap family; no verdict)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip, unadjusted) | Expected sign |
| --- | --- | --- | --- | --- | --- | --- |
| 12 − 9 | Plain BCE instead of the compound loss (12 − 9) | 42, 43, 44 | +0.007 [-0.027, +0.043] | 6/10 | 0.703 | open (synthetic: positive) |
| row09b_gate_bias0 − 9 | Gate starting half open (9b − 9) | 42 | -0.010 [-0.068, +0.038] | 5/10 | 0.707 | open |
| row09c_cloud_augment − 9 | Synthetic clouds in training (9c − 9) | 42 | -0.004 [-0.026, +0.023] | 4/10 | 0.807 | open |
| row09r_sar_query − 9 | Reverse assignment (9r − 9) | 42 | +0.026 [-0.002, +0.065] | 7/10 | 0.164 | open |
| row02e_event_fallback − 2 | Event context for the baseline's threshold (2e − 2) | — | +0.003 [-0.000, +0.008] | 7/10 | 0.193 | ≥ 0 |
| row02s_seasonal − row02e_event_fallback | Seasonal reference for the change term (2s − 2e) | — | +0.175 [+0.067, +0.293] | 9/10 | 0.004 | positive |

## Calibration across events: each held-out event calibrated on the others (cross-fitted)

The cost-optimal threshold and the conformal threshold of each held-out event are fitted on the pooled held-out predictions of the *other* events (each from its own fold model) instead of the fold's single validation event. Nothing is re-run: the stored confusion counts are re-read (`floodfusion.crossfit`). n counts windows, not events, and each calibration window comes from a fold model that was trained on the target event: the control is approximate (cross-conformal in style), not a finite-sample guarantee.

| Row | Seed | K | IoU at τ* (own validation) | IoU at cross-fitted τ | IoU, best τ per group | Conformal: events with FNR ≤ α, own → cross-fitted | Mean chip FNR, cross-fitted | Precision at τ_conf, cross-fitted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 42 | 10 | 0.356 | 0.310 | 0.481 | 7/10 → 8/10 | 0.064 | 0.171 |
| 11 | 43 | 10 | 0.395 | 0.403 | 0.497 | 7/10 → 8/10 | 0.050 | 0.150 |
| 11 | 44 | 10 | 0.355 | 0.345 | 0.494 | 6/10 → 7/10 | 0.076 | 0.190 |
| 12 | 42 | 10 | 0.608 | 0.612 | 0.660 | 6/10 → 5/10 | 0.092 | 0.339 |
| 12 | 43 | 10 | 0.622 | 0.610 | 0.651 | 7/10 → 5/10 | 0.093 | 0.334 |
| 12 | 44 | 10 | 0.619 | 0.607 | 0.664 | 5/10 → 7/10 | 0.065 | 0.220 |
| 4 | 42 | 10 | 0.332 | 0.335 | 0.456 | 7/10 → 6/10 | 0.096 | 0.228 |
| 4 | 43 | 10 | 0.348 | 0.374 | 0.460 | 7/10 → 5/10 | 0.121 | 0.231 |
| 4 | 44 | 10 | 0.355 | 0.339 | 0.454 | 7/10 → 5/10 | 0.108 | 0.224 |
| 9 | 42 | 10 | 0.628 | 0.601 | 0.670 | 5/10 → 5/10 | 0.120 | 0.302 |
| 9 | 43 | 10 | 0.623 | 0.604 | 0.664 | 7/10 → 7/10 | 0.096 | 0.296 |
| 9 | 44 | 10 | 0.573 | 0.564 | 0.653 | 7/10 → 7/10 | 0.081 | 0.211 |
| cmp_terramind_small | 42 | 10 | 0.577 | 0.575 | 0.597 | 6/10 → 7/10 | 0.050 | 0.159 |
| cmp_terramind_small | 43 | 10 | 0.573 | 0.566 | 0.589 | 6/10 → 9/10 | 0.041 | 0.180 |
| cmp_terramind_small | 44 | 10 | 0.566 | 0.567 | 0.588 | 7/10 → 9/10 | 0.045 | 0.203 |
| cmp_terramind_small__eval_sar_only | 42 | 10 | 0.424 | 0.414 | 0.473 | 8/10 → 6/10 | 0.077 | 0.187 |
| cmp_terramind_small__eval_sar_only | 43 | 10 | 0.423 | 0.411 | 0.469 | 8/10 → 10/10 | 0.033 | 0.152 |
| cmp_terramind_small__eval_sar_only | 44 | 10 | 0.392 | 0.377 | 0.450 | 5/10 → 9/10 | 0.032 | 0.126 |
| row09b_gate_bias0 | 42 | 10 | 0.618 | 0.604 | 0.671 | 8/10 → 6/10 | 0.118 | 0.298 |
| row09c_cloud_augment | 42 | 10 | 0.624 | 0.596 | 0.662 | 7/10 → 5/10 | 0.117 | 0.276 |
| row09r_sar_query | 42 | 10 | 0.654 | 0.629 | 0.678 | 8/10 → 7/10 | 0.097 | 0.292 |

Event-level conformal control with n = 9 calibration events tolerates a calibration risk of at most +0.000 at α = 0.1 (≤ 0: vacuous — flag everything); the cross-fitted threshold above is chip-level control under pooling.

## Per-event IoU at τ* (LOEO)

| Row | S1F11_Bolivia | S1F11_Ghana | S1F11_India | S1F11_Mekong | S1F11_Nigeria | S1F11_Pakistan | S1F11_Somalia | S1F11_Spain | S1F11_Sri-Lanka | S1F11_USA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.546 | 0.314 | 0.332 | 0.655 | 0.293 | 0.142 | 0.140 | 0.246 | 0.196 | 0.124 |
| 2 | 0.243 | 0.066 | 0.081 | 0.117 | 0.034 | 0.024 | 0.210 | 0.452 | 0.127 | 0.221 |
| 3 | 0.479 | 0.319 | 0.467 | 0.662 | 0.569 | 0.434 | 0.285 | 0.545 | 0.276 | 0.403 |
| 4 | 0.651 | 0.337 | 0.453 | 0.786 | 0.168 | 0.122 | 0.147 | 0.322 | 0.262 | 0.187 |
| 9 | 0.688 | 0.543 | 0.611 | 0.826 | 0.590 | 0.422 | 0.678 | 0.642 | 0.488 | 0.594 |
| 11 | 0.651 | 0.395 | 0.413 | 0.792 | 0.144 | 0.132 | 0.120 | 0.367 | 0.308 | 0.359 |
| 12 | 0.726 | 0.659 | 0.579 | 0.850 | 0.639 | 0.407 | 0.604 | 0.660 | 0.517 | 0.518 |
| cmp_terramind_small | 0.728 | 0.681 | 0.534 | 0.857 | 0.759 | 0.385 | 0.338 | 0.586 | 0.362 | 0.492 |
| cmp_terramind_small__eval_sar_only | 0.633 | 0.492 | 0.422 | 0.739 | 0.488 | 0.082 | 0.156 | 0.522 | 0.241 | 0.358 |
| row02e_event_fallback | 0.241 | 0.068 | 0.081 | 0.117 | 0.056 | 0.023 | 0.218 | 0.456 | 0.124 | 0.222 |
| row02s_seasonal | 0.683 | 0.491 | 0.082 | 0.498 | 0.374 | 0.027 | 0.319 | 0.476 | 0.124 | 0.282 |
| row09b_gate_bias0 | 0.735 | 0.616 | 0.623 | 0.854 | 0.480 | 0.440 | 0.656 | 0.642 | 0.497 | 0.632 |
| row09c_cloud_augment | 0.773 | 0.574 | 0.576 | 0.868 | 0.732 | 0.420 | 0.653 | 0.616 | 0.469 | 0.565 |
| row09r_sar_query | 0.731 | 0.663 | 0.636 | 0.864 | 0.759 | 0.454 | 0.678 | 0.649 | 0.504 | 0.603 |

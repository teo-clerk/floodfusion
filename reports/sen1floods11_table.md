# Sen1Floods11, re-ingested — leave-one-event-out and the published split

Real data: Sen1Floods11 hand labels (Bonafilia et al., 2020) with every input re-fetched from Planetary Computer by `floodfusion.data.sen1floods11` (same-orbit pre-event RTC, S2 L2A, GLO-30, JRC, WorldCover; optical and labels moved onto the SAR geometry). Flood = labelled water minus JRC permanent water. IoU at τ* = 0.25 after calibration on each fold's validation event.

**Protocol: leave-one-event-out.** 10 folds (Paraguay dropped: its acquisition is not in the archive); each event is tested once by a model that never saw it; validation = the next event (sorted, cyclic), fixed before any run. **[CI]** = 95 % percentile bootstrap over events (descriptive); gap verdicts are exact sign-flip permutation tests over events, Holm-adjusted across the table.

| # | Configuration | IoU at τ* [95 % CI over events] | IoU, oracle τ | AP | F1 | Seeds | Events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Global Otsu on VV, no masks | 0.299 [0.20, 0.41] | — | 0.451 | 0.436 | — | 10 |
| 2 | Baseline: adaptive threshold + change + masks | 0.157 [0.09, 0.24] | — | 0.152 | 0.254 | — | 10 |
| 3 | Baseline + MNDWI where cloud-free | 0.444 [0.37, 0.52] | — | 0.440 | 0.605 | — | 10 |
| 4 | U-Net, SAR only (VV, VH) | 0.365 [0.24, 0.51] | 0.475 | 0.538 | 0.502 | 42 | 10 |
| 9 | **Fusion, cross-attention (reference)** | 0.625 [0.56, 0.69] | 0.668 | 0.769 | 0.764 | 42, 43, 44 | 10 |
| 11 | Row 9, evaluated SAR-only at inference | 0.383 [0.26, 0.52] | 0.500 | 0.570 | 0.521 | 42, 43, 44 | 10 |
| TM | TerraMind-1.0-small, frozen + UNet decoder (single date: post S1 + S2) ‡ | 0.576 [0.47, 0.69] | 0.597 | 0.765 | 0.716 | 42 | 10 |

§ fewer than 8 events: the interval is indicative only.

## Caveats

- † **Optical is the annotators' source image.** Sen1Floods11 labels were drawn mostly on the Sentinel-2 image of the same date, and that image is re-fetched as the optical input. Gaps involving optical therefore measure agreement with the annotators' source as well as the value of optical; the EMSR664 case study (independent CEMS delineations) is the independent check.
- ‡ **TerraMind** saw global Sentinel-1/2 data in pre-training ("never saw the event" holds for its decoder only); its band subset and necks follow IBM's recipe, developed on Sen1Floods11's official splits; its schedule is row 9's, tuned for neither model; its SAR input is our z-score adapted by four affine parameters; clouded pixels reach it as its pre-training mean (it has no per-pixel gate).
- **Labels mark water on a date, not a change.** In 7 of 10 events the flood-labelled pixels show no backscatter decrease against a same-orbit reference up to 60 days earlier (see the label diagnostic): change-based rules (rows 2, 3) are structurally penalised, amplitude-based ones are not.
- **Validation is one event per fold** (the next, fixed in advance): early stopping, the prior, the calibration and the baseline offset come from it, so part of a fold's score is threshold transfer — compare IoU at τ* with the oracle-threshold IoU and AP.
- **Development runs** scored one fold (Bolivia; rows 1-4) before the protocol was frozen; every later change came from code review, none from those numbers.

## Paired gaps over events (matched seeds)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip) | Verdict | Expected sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 − 1 | What careful engineering adds to naive Otsu | — | -0.141 [-0.274, -0.011] | 3/10 basins | 0.078 | not significant (Holm p = 0.234) | positive |
| 3 − 2 | What optical adds to the physics baseline † | — | +0.286 [+0.188, +0.390] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | positive where optical is clear |
| 4 − 2 | What learning adds over the physics baseline (SAR only) | 42 | +0.208 [+0.063, +0.361] | 8/10 basins | 0.029 | not significant (Holm p = 0.117) | positive |
| 9 − 4 | What fusion adds over the SAR-only U-Net (R3) † | 42 | +0.263 [+0.178, +0.350] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | positive where optical is clear |
| 9 − 11 | The cost of losing optical at inference (9 − 11) † | 42, 43, 44 | +0.242 [+0.147, +0.350] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | positive; small is good |
| 11 − 4 | The fusion model run SAR-only vs the SAR-only U-Net | 42 | -0.005 [-0.044, +0.038] | 3/10 basins | 0.789 | not significant (Holm p = 0.789) | ≈ 0 or positive |
| 9 − 2 | What the machine learning was worth (9 − 2) | 42, 43, 44 | +0.467 [+0.380, +0.555] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | positive |
| cmp_terramind_small − 4 | TerraMind (frozen; post-event S1 + S2) vs the SAR-only U-Net ‡ | 42 | +0.211 [+0.127, +0.309] | 10/10 basins | 0.002 | significant (Holm p = 0.018) | open |
| 9 − cmp_terramind_small | What the task-specific fusion model adds over the foundation model ‡ | 42 | +0.052 [-0.029, +0.135] | 7/10 basins | 0.287 | not significant (Holm p = 0.574) | ≈ 0 on Sen1Floods11 (PANGAEA: supervised U-Nets match GFMs there) |

## The published split's protocol gap, measured on the same chips

The dataset's official test chips come from the **same events** as its training chips. Each event's official test chips are scored twice: by the model trained on the official split, and by the LOEO model that never saw the event. Δ combines leakage with the protocols' other differences (training-set size, validation set, calibration): at τ* it includes each model's calibration; at the oracle threshold and in AP it does not.

| Row | Seed | Events | Measure | Official split | LOEO, same chips | Δ (official − LOEO) | Official higher in | p (sign-flip) | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 42 | 9 | IoU at τ* | 0.398 | 0.395 | +0.003 | 5/9 | 0.965 | 1.000 |
| 4 | 42 | 9 | IoU, oracle τ | 0.530 | 0.503 | +0.027 | 6/9 | 0.168 | 1.000 |
| 4 | 42 | 9 | AP | 0.590 | 0.536 | +0.054 | 6/9 | 0.125 | 1.000 |
| 9 | 42 | 9 | IoU at τ* | 0.623 | 0.618 | +0.005 | 6/9 | 0.828 | 1.000 |
| 9 | 42 | 9 | IoU, oracle τ | 0.668 | 0.665 | +0.003 | 4/9 | 0.816 | 1.000 |
| 9 | 42 | 9 | AP | 0.725 | 0.725 | +0.000 | 4/9 | 0.996 | 1.000 |
| 9 | 43 | 9 | IoU at τ* | 0.646 | 0.641 | +0.005 | 5/9 | 0.770 | 1.000 |
| 9 | 43 | 9 | IoU, oracle τ | 0.678 | 0.670 | +0.009 | 5/9 | 0.496 | 1.000 |
| 9 | 43 | 9 | AP | 0.751 | 0.722 | +0.029 | 4/9 | 0.387 | 1.000 |
| 9 | 44 | 9 | IoU at τ* | 0.631 | 0.616 | +0.015 | 6/9 | 0.402 | 1.000 |
| 9 | 44 | 9 | IoU, oracle τ | 0.667 | 0.673 | -0.006 | 3/9 | 0.574 | 1.000 |
| 9 | 44 | 9 | AP | 0.745 | 0.740 | +0.005 | 5/9 | 0.832 | 1.000 |
| 11 | 42 | 9 | IoU at τ* | 0.458 | 0.367 | +0.091 | 6/9 | 0.277 | 1.000 |
| 11 | 42 | 9 | IoU, oracle τ | 0.530 | 0.514 | +0.017 | 5/9 | 0.320 | 1.000 |
| 11 | 42 | 9 | AP | 0.568 | 0.551 | +0.017 | 6/9 | 0.500 | 1.000 |
| 11 | 43 | 9 | IoU at τ* | 0.510 | 0.460 | +0.049 | 5/9 | 0.129 | 1.000 |
| 11 | 43 | 9 | IoU, oracle τ | 0.558 | 0.532 | +0.027 | 6/9 | 0.129 | 1.000 |
| 11 | 43 | 9 | AP | 0.622 | 0.568 | +0.054 | 5/9 | 0.117 | 1.000 |
| 11 | 44 | 9 | IoU at τ* | 0.450 | 0.391 | +0.060 | 6/9 | 0.531 | 1.000 |
| 11 | 44 | 9 | IoU, oracle τ | 0.530 | 0.530 | -0.000 | 4/9 | 0.996 | 1.000 |
| 11 | 44 | 9 | AP | 0.594 | 0.571 | +0.023 | 5/9 | 0.477 | 1.000 |
| cmp_terramind_small | 42 | 9 | IoU at τ* | 0.601 | 0.602 | -0.001 | 5/9 | 0.926 | 1.000 |
| cmp_terramind_small | 42 | 9 | IoU, oracle τ | 0.621 | 0.621 | +0.000 | 5/9 | 0.973 | 1.000 |
| cmp_terramind_small | 42 | 9 | AP | 0.727 | 0.723 | +0.004 | 5/9 | 0.812 | 1.000 |

## Bolivia — the dataset's own held-out event

| Row | Seed | IoU on Bolivia, official-split model | IoU on Bolivia, LOEO fold |
| --- | --- | --- | --- |
| 4 | 42 | 0.460 | 0.651 |
| 9 | 42 | 0.694 | 0.762 |
| 9 | 43 | 0.690 | 0.723 |
| 9 | 44 | 0.705 | 0.531 |
| 11 | 42 | 0.648 | 0.615 |
| 11 | 43 | 0.677 | 0.684 |
| 11 | 44 | 0.598 | 0.554 |
| cmp_terramind_small | 42 | 0.680 | 0.760 |

## Conformal recall control across events

Each fold fits the conformal threshold on its validation event's chips (≥ 100 flood pixels; risk = per-chip false-negative rate; guarantee E[risk] ≤ α for chips exchangeable with those — windows of one source chip are not independent, and the same chips also selected the checkpoint). The held-out event is a different flood: this measures how far the guarantee carries.

| Row | Seed | α | Feasible folds | Events with mean chip FNR ≤ α (feasible folds) | Mean chip FNR | Worst event | Precision at τ_conf | IoU at τ_conf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 42 | 0.1 | 10/10 | 7/10 | 0.077 | S1F11_India (0.243) | 0.209 | 0.208 |
| 9 | 42 | 0.1 | 10/10 | 7/10 | 0.085 | S1F11_Ghana (0.239) | 0.277 | 0.275 |
| 9 | 43 | 0.1 | 10/10 | 8/10 | 0.078 | S1F11_Somalia (0.200) | 0.272 | 0.271 |
| 9 | 44 | 0.1 | 10/10 | 5/10 | 0.100 | S1F11_Somalia (0.249) | 0.228 | 0.227 |
| 11 | 42 | 0.1 | 10/10 | 6/10 | 0.092 | S1F11_India (0.237) | 0.234 | 0.231 |
| 11 | 43 | 0.1 | 10/10 | 6/10 | 0.085 | S1F11_USA (0.260) | 0.237 | 0.235 |
| 11 | 44 | 0.1 | 10/10 | 6/10 | 0.106 | S1F11_USA (0.290) | 0.228 | 0.224 |
| cmp_terramind_small | 42 | 0.1 | 10/10 | 6/10 | 0.090 | S1F11_Somalia (0.297) | 0.273 | 0.271 |

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

## Per-event IoU at τ* (LOEO)

| Row | S1F11_Bolivia | S1F11_Ghana | S1F11_India | S1F11_Mekong | S1F11_Nigeria | S1F11_Pakistan | S1F11_Somalia | S1F11_Spain | S1F11_Sri-Lanka | S1F11_USA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.546 | 0.314 | 0.332 | 0.655 | 0.293 | 0.142 | 0.140 | 0.246 | 0.196 | 0.124 |
| 2 | 0.243 | 0.066 | 0.081 | 0.117 | 0.034 | 0.024 | 0.210 | 0.452 | 0.127 | 0.221 |
| 3 | 0.479 | 0.319 | 0.467 | 0.662 | 0.569 | 0.434 | 0.285 | 0.545 | 0.276 | 0.403 |
| 4 | 0.651 | 0.435 | 0.483 | 0.791 | 0.202 | 0.101 | 0.151 | 0.291 | 0.302 | 0.245 |
| 9 | 0.675 | 0.606 | 0.612 | 0.852 | 0.652 | 0.443 | 0.667 | 0.639 | 0.498 | 0.605 |
| 11 | 0.620 | 0.481 | 0.431 | 0.795 | 0.129 | 0.129 | 0.127 | 0.422 | 0.313 | 0.384 |
| cmp_terramind_small | 0.760 | 0.678 | 0.528 | 0.858 | 0.768 | 0.408 | 0.343 | 0.584 | 0.361 | 0.474 |

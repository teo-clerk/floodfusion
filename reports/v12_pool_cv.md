# v1.2 training pool — Kuro Siwo (rebuilt) + GEOID-Flood, grouped five-fold, flood labels

Real data: Kuro Siwo annotation polygons (Bountos et al., NeurIPS 2024; CC-BY-4.0), re-ingested with the DEM fix, pooled with the GEOID-Flood events before EMSR656 that Kuro Siwo lacks (Chiriaco et al., 2026; CC-BY-4.0); every input re-fetched from Planetary Computer. 7,006 chips from 138 floods. Label: **flood**, minus permanent water. Folds group events within 100 km across both sets (81 groups); each group is tested once by a model that never saw it; validation = the next fold. One seed per row; each family at its tuned learning rate. Protocol: `reports/protocol_v1_2.md`; rules: `reports/v1_2_deviations.md`.

| # | Configuration | IoU at τ* [95 % CI over events] | IoU, oracle τ | AP | F1 | Seeds | Events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | **Fusion, cross-attention (reference)** | 0.383 [0.33, 0.43] | 0.378 | 0.632 | 0.516 | 42 | 81 |
| 11 | Row 9, evaluated SAR-only at inference | 0.384 [0.33, 0.43] | 0.379 | 0.631 | 0.515 | 42 | 81 |
| cmp_terramind_base_fair | cmp_terramind_base_fair | 0.313 [0.27, 0.36] | 0.366 | 0.617 | 0.440 | 42 | 81 |
| cmp_terramind_base_fair__eval_sar_only | cmp_terramind_base_fair__eval_sar_only | 0.319 [0.27, 0.37] | 0.366 | 0.626 | 0.443 | 42 | 81 |
| row21_fusion_context_seasonal | row21_fusion_context_seasonal | 0.394 [0.34, 0.44] | 0.388 | 0.651 | 0.524 | 42 | 81 |
| row21_fusion_context_seasonal__eval_sar_only | row21_fusion_context_seasonal__eval_sar_only | 0.391 [0.34, 0.44] | 0.385 | 0.647 | 0.521 | 42 | 81 |

§ fewer than 8 groups: the interval is indicative only.

## Caveats

- **Folds hold several groups.** The oracle-threshold IoU, AP and the other fold-averaged columns of the main table are fold-level quantities (each fold weighted equally), not per-group macros; IoU at τ* and every gap are per group. The cross-fitted section below compares thresholds per group.
- **Single seed, exploratory.** Model selection used the whole-stack measure (`reports/v12_selection.json`); v1.2's confirmatory tests are the blind ones.

## Paired gaps over held-out groups (matched seeds)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip) | Verdict | Expected sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 − 11 | The cost of losing optical at inference (9 − 11) † | 42 | -0.001 [-0.011, +0.005] | 43/79 basins | 0.999 | not significant (Holm p = 0.999) | positive; small is good |

## Conformal recall control across groups (calibrated on each fold's validation groups)

| Row | Seed | α | Feasible folds | Events with mean chip FNR ≤ α (feasible folds) | Mean chip FNR | Worst event | Precision at τ_conf | IoU at τ_conf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 42 | 0.1 | 5/5 | 38/74 | 0.160 | POOL_geoid_flood_EMSR257 (0.856) | 0.275 | 0.270 |
| 11 | 42 | 0.1 | 5/5 | 39/74 | 0.154 | POOL_geoid_flood_EMSR257 (0.851) | 0.270 | 0.265 |

## v1.1 variants — descriptive gaps (not part of the gap family; no verdict)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip, unadjusted) | Expected sign |
| --- | --- | --- | --- | --- | --- | --- |

## Calibration across events: each held-out event calibrated on the others (cross-fitted)

The cost-optimal threshold and the conformal threshold of each held-out event are fitted on the pooled held-out predictions of the *other* events (each from its own fold model) instead of the fold's single validation event. Nothing is re-run: the stored confusion counts are re-read (`floodfusion.crossfit`). n counts windows, not events, and each calibration window comes from a fold model that was trained on the target event: the control is approximate (cross-conformal in style), not a finite-sample guarantee.

| Row | Seed | K | IoU at τ* (own validation) | IoU at cross-fitted τ | IoU, best τ per group | Conformal: events with FNR ≤ α, own → cross-fitted | Mean chip FNR, cross-fitted | Precision at τ_conf, cross-fitted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 42 | 81 | 0.384 | 0.383 | 0.424 | 39/74 → 55/81 | 0.094 | 0.155 |
| 9 | 42 | 81 | 0.383 | 0.382 | 0.425 | 38/74 → 54/81 | 0.097 | 0.156 |
| cmp_terramind_base_fair | 42 | 81 | 0.313 | 0.369 | 0.406 | 46/74 → 50/81 | 0.138 | 0.173 |
| cmp_terramind_base_fair__eval_sar_only | 42 | 81 | 0.319 | 0.365 | 0.406 | 44/74 → 48/81 | 0.141 | 0.173 |
| row21_fusion_context_seasonal | 42 | 81 | 0.394 | 0.392 | 0.430 | 39/74 → 54/81 | 0.092 | 0.142 |
| row21_fusion_context_seasonal__eval_sar_only | 42 | 81 | 0.391 | 0.389 | 0.427 | 39/74 → 51/81 | 0.099 | 0.147 |

Event-level conformal control with n = 80 calibration events tolerates a calibration risk of at most +0.089 at α = 0.1 (≤ 0: vacuous — flag everything); the cross-fitted threshold above is chip-level control under pooling.

## Per-group IoU at τ*

| Row | POOL_geoid_flood_EMSR151 | POOL_geoid_flood_EMSR165 | POOL_geoid_flood_EMSR166 | POOL_geoid_flood_EMSR167 | POOL_geoid_flood_EMSR176 | POOL_geoid_flood_EMSR184 | POOL_geoid_flood_EMSR187 | POOL_geoid_flood_EMSR188 | POOL_geoid_flood_EMSR192 | POOL_geoid_flood_EMSR199 | POOL_geoid_flood_EMSR203 | POOL_geoid_flood_EMSR215 | POOL_geoid_flood_EMSR235 | POOL_geoid_flood_EMSR238 | POOL_geoid_flood_EMSR257 | POOL_geoid_flood_EMSR258 | POOL_geoid_flood_EMSR260 | POOL_geoid_flood_EMSR268 | POOL_geoid_flood_EMSR271 | POOL_geoid_flood_EMSR280 | POOL_geoid_flood_EMSR283 | POOL_geoid_flood_EMSR284 | POOL_geoid_flood_EMSR286 | POOL_geoid_flood_EMSR287 | POOL_geoid_flood_EMSR292 | POOL_geoid_flood_EMSR319 | POOL_geoid_flood_EMSR323 | POOL_geoid_flood_EMSR326 | POOL_geoid_flood_EMSR329 | POOL_geoid_flood_EMSR330 | POOL_geoid_flood_EMSR337 | POOL_geoid_flood_EMSR342 | POOL_geoid_flood_EMSR352 | POOL_geoid_flood_EMSR358 | POOL_geoid_flood_EMSR388 | POOL_geoid_flood_EMSR399 | POOL_geoid_flood_EMSR407 | POOL_geoid_flood_EMSR410 | POOL_geoid_flood_EMSR414 | POOL_geoid_flood_EMSR419 | POOL_geoid_flood_EMSR424 | POOL_geoid_flood_EMSR431 | POOL_geoid_flood_EMSR438 | POOL_geoid_flood_EMSR442 | POOL_geoid_flood_EMSR450 | POOL_geoid_flood_EMSR456 | POOL_geoid_flood_EMSR471 | POOL_geoid_flood_EMSR479 | POOL_geoid_flood_EMSR480 | POOL_geoid_flood_EMSR483 | POOL_geoid_flood_EMSR486 | POOL_geoid_flood_EMSR507 | POOL_geoid_flood_EMSR514 | POOL_geoid_flood_EMSR551 | POOL_geoid_flood_EMSR557 | POOL_geoid_flood_EMSR569 | POOL_geoid_flood_EMSR572 | POOL_geoid_flood_EMSR577 | POOL_geoid_flood_EMSR629 | POOL_geoid_flood_EMSR631 | POOL_geoid_flood_EMSR634 | POOL_geoid_flood_EMSR639 | POOL_geoid_flood_EMSR640 | POOL_geoid_flood_EMSR646 | POOL_geoid_flood_EMSR650 | POOL_kurosiwo_EMSR1111004_USA | POOL_kurosiwo_EMSR1111005_Madagascar | POOL_kurosiwo_EMSR1111007_Nepal | POOL_kurosiwo_EMSR1111011_Philippines | POOL_kurosiwo_EMSR147_Cumbria | POOL_kurosiwo_EMSR174_Skopje | POOL_kurosiwo_EMSR205_SriLanka | POOL_kurosiwo_EMSR267_Lithuania | POOL_kurosiwo_EMSR277_Greece | POOL_kurosiwo_EMSR321_Honduras | POOL_kurosiwo_EMSR417_Portugal | POOL_kurosiwo_EMSR427_Sweden | POOL_kurosiwo_EMSR470_Togo | POOL_kurosiwo_EMSR498_France | POOL_kurosiwo_EMSR561_Malawi | POOL_kurosiwo_EMSR562_Australia |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.540 | 0.393 | 0.633 | 0.448 | 0.391 | 0.597 | 0.373 | 0.419 | 0.323 | 0.455 | 0.000 | 0.463 | 0.218 | 0.150 | 0.000 | 0.408 | 0.376 | 0.285 | 0.425 | 0.411 | 0.146 | 0.522 | 0.234 | 0.000 | 0.513 | 0.756 | 0.000 | 0.367 | 0.142 | 0.294 | 0.320 | 0.113 | 0.414 | 0.471 | 0.346 | 0.344 | 0.687 | 0.382 | 0.000 | 0.223 | 0.459 | 0.239 | 0.114 | 0.213 | 0.246 | 0.701 | 0.425 | 0.462 | 0.285 | 0.014 | — | 0.218 | 0.605 | 0.000 | 0.420 | 0.000 | 0.245 | 0.025 | 0.507 | 0.214 | 0.248 | 0.392 | 0.419 | — | 0.525 | 0.404 | 0.784 | 0.374 | 0.609 | 0.587 | 0.751 | 0.502 | 0.465 | 0.748 | 0.589 | 0.717 | 0.301 | 0.812 | 0.466 | 0.718 | 0.864 |
| 11 | 0.546 | 0.393 | 0.633 | 0.448 | 0.371 | 0.590 | 0.389 | 0.419 | 0.330 | 0.451 | 0.000 | 0.445 | 0.221 | 0.117 | 0.000 | 0.403 | 0.362 | 0.296 | 0.422 | 0.410 | 0.138 | 0.516 | 0.215 | 0.000 | 0.517 | 0.756 | 0.000 | 0.349 | 0.135 | 0.296 | 0.320 | 0.107 | 0.759 | 0.466 | 0.335 | 0.345 | 0.687 | 0.376 | 0.000 | 0.214 | 0.459 | 0.241 | 0.091 | 0.206 | 0.248 | 0.701 | 0.423 | 0.431 | 0.274 | 0.014 | — | 0.208 | 0.605 | 0.000 | 0.419 | 0.000 | 0.245 | 0.026 | 0.504 | 0.231 | 0.248 | 0.393 | 0.419 | — | 0.527 | 0.356 | 0.783 | 0.370 | 0.617 | 0.587 | 0.758 | 0.503 | 0.423 | 0.748 | 0.588 | 0.714 | 0.305 | 0.804 | 0.482 | 0.719 | 0.859 |
| cmp_terramind_base_fair | 0.492 | 0.319 | 0.282 | 0.569 | 0.295 | 0.565 | 0.504 | 0.458 | 0.211 | 0.367 | 0.000 | 0.379 | 0.240 | 0.135 | 0.000 | 0.325 | 0.250 | 0.264 | 0.380 | 0.096 | 0.219 | 0.483 | 0.223 | 0.000 | 0.161 | 0.381 | 0.000 | 0.412 | 0.024 | 0.239 | 0.215 | 0.055 | 0.703 | 0.447 | 0.236 | 0.174 | 0.566 | 0.123 | 0.000 | 0.260 | 0.425 | 0.249 | 0.031 | 0.163 | 0.261 | 0.644 | 0.195 | 0.440 | 0.085 | 0.013 | — | 0.125 | 0.286 | 0.000 | 0.318 | 0.000 | 0.143 | 0.000 | 0.337 | 0.273 | 0.565 | 0.134 | 0.299 | — | 0.579 | 0.444 | 0.586 | 0.385 | 0.589 | 0.512 | 0.333 | 0.525 | 0.610 | 0.390 | 0.438 | 0.720 | 0.179 | 0.547 | 0.600 | 0.411 | 0.876 |
| cmp_terramind_base_fair__eval_sar_only | 0.481 | 0.332 | 0.287 | 0.574 | 0.316 | 0.564 | 0.523 | 0.453 | 0.226 | 0.369 | 0.000 | 0.413 | 0.205 | 0.148 | 0.000 | 0.356 | 0.269 | 0.349 | 0.398 | 0.018 | 0.210 | 0.473 | 0.220 | 0.000 | 0.198 | 0.458 | 0.000 | 0.420 | 0.017 | 0.224 | 0.231 | 0.055 | 0.724 | 0.457 | 0.242 | 0.161 | 0.565 | 0.080 | 0.000 | 0.246 | 0.429 | 0.249 | 0.089 | 0.142 | 0.241 | 0.665 | 0.224 | 0.417 | 0.072 | 0.015 | — | 0.132 | 0.281 | 0.000 | 0.294 | 0.000 | 0.162 | 0.000 | 0.512 | 0.074 | 0.564 | 0.123 | 0.289 | — | 0.605 | 0.409 | 0.603 | 0.311 | 0.610 | 0.504 | 0.505 | 0.525 | 0.626 | 0.442 | 0.452 | 0.734 | 0.193 | 0.497 | 0.599 | 0.506 | 0.860 |
| row21_fusion_context_seasonal | 0.555 | 0.283 | 0.340 | 0.555 | 0.380 | 0.577 | 0.465 | 0.433 | 0.338 | 0.417 | 0.000 | 0.523 | 0.371 | 0.188 | 0.030 | 0.415 | 0.379 | 0.301 | 0.448 | 0.434 | 0.165 | 0.537 | 0.210 | 0.000 | 0.549 | 0.768 | 0.000 | 0.462 | 0.077 | 0.316 | 0.344 | 0.069 | 0.790 | 0.527 | 0.340 | 0.208 | 0.713 | 0.213 | 0.000 | 0.263 | 0.446 | 0.255 | 0.091 | 0.179 | 0.319 | 0.702 | 0.449 | 0.512 | 0.268 | 0.008 | — | 0.162 | 0.609 | 0.000 | 0.298 | 0.000 | 0.255 | 0.000 | 0.671 | 0.272 | 0.374 | 0.415 | 0.405 | — | 0.598 | 0.502 | 0.656 | 0.428 | 0.630 | 0.566 | 0.746 | 0.493 | 0.422 | 0.783 | 0.576 | 0.773 | 0.327 | 0.835 | 0.587 | 0.742 | 0.755 |
| row21_fusion_context_seasonal__eval_sar_only | 0.564 | 0.283 | 0.343 | 0.560 | 0.380 | 0.571 | 0.465 | 0.433 | 0.338 | 0.402 | 0.000 | 0.505 | 0.362 | 0.134 | 0.037 | 0.415 | 0.379 | 0.301 | 0.448 | 0.434 | 0.165 | 0.539 | 0.197 | 0.000 | 0.549 | 0.768 | 0.000 | 0.459 | 0.076 | 0.325 | 0.344 | 0.065 | 0.790 | 0.526 | 0.340 | 0.205 | 0.717 | 0.207 | 0.000 | 0.219 | 0.446 | 0.256 | 0.098 | 0.178 | 0.320 | 0.702 | 0.449 | 0.501 | 0.269 | 0.009 | — | 0.153 | 0.612 | 0.000 | 0.293 | 0.000 | 0.255 | 0.000 | 0.677 | 0.274 | 0.377 | 0.410 | 0.405 | — | 0.608 | 0.432 | 0.656 | 0.427 | 0.630 | 0.568 | 0.745 | 0.494 | 0.422 | 0.786 | 0.576 | 0.773 | 0.325 | 0.827 | 0.604 | 0.742 | 0.755 |

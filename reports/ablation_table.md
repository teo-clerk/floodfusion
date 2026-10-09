# Ablation table — MILESTONES.md section 6

> **SYNTHETIC BENCHMARK.** Every number below is measured on the physics-based synthetic benchmark (`floodfusion.data.synthetic`), not on real Sentinel-1/2 data. It validates the pipeline and the experimental design; it is **not** a real-world accuracy claim.

**Protocol** (docs/09 §2.2): one fixed basin-blocked train/val/test split; test basins: E1_lowland_po_B0, E1_lowland_po_B1, E4_hilly_forest_B0 (3); seed protocol **affordable** (decided before the grid from a measured row-9 run: projected full grid 94 min vs budget 75 min); seeds per learned row — row 4: 1, row 5: 1, row 6: 1, row 7: 1, row 8: 1, row 9: 3, row 10: 1, row 12: 1, row 13: 1; gaps compare **matched seeds only**. **IoU** = macro over test basins at the cost-optimal τ* = 1/(1+r) = 0.25 (r = 3) after calibration on validation (prior correction + temperature, Platt if clearly better); seeds pooled per basin; **[CI]** = 95 % percentile bootstrap over test basins (10 000 resamples). Baselines (rows 1–3): threshold offset chosen on validation by the same cost rule. **Peak VRAM** = `torch.cuda.max_memory_allocated()` during training; **ms/chip** = 256² chip, batch 1, ≥ 20 warm-up iterations after a sustained warm-up, `synchronize()` around the timer (docs/09 §7).

Manifest `3e9bbf7d4ae61bdc…`, git `1669c1fa1363`. Regenerate with `python -m floodfusion.evaluate --ablation-table --runs-root runs`.

| #   | Configuration | Purpose — the question the row answers | IoU | F1  | Peak VRAM | Train h | ms/chip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Global Otsu on VV, no masks | The floor. Is anything learned at all? | 0.623 [0.55, 0.68]† | 0.766 | — | — | 11.3 (CPU) |
| 2 | Baseline: adaptive threshold + change + masks | The operational reference you must beat | 0.595 [0.44, 0.68]† | 0.740 | — | — | 14.5 (CPU) |
| 3 | Baseline + MNDWI where cloud-free | What optical adds to a non-learned method | 0.606 [0.41, 0.73]† | 0.745 | — | — | 14.1 (CPU) |
| 4 | U-Net, SAR only (VV, VH) | Does learning beat thresholding on the same inputs? | 0.673 [0.63, 0.71]† | 0.804 | 1.72 GiB | 0.045 | 4.2 |
| 5 | U-Net, SAR + DEM derivatives | What terrain context is worth | 0.673 [0.64, 0.70]† | 0.804 | 1.70 GiB | 0.044 | 4.2 |
| 6 | U-Net, optical only, cloud-free chips only | The optical ceiling — an upper bound you cannot deploy | 0.754 [0.63, 0.88]† | 0.854 | 0.94 GiB | 0.020 | 4.2 |
| 7 | Fusion, early (channel concatenation) | Does naive fusion help? | 0.683 [0.62, 0.75]† | 0.811 | 0.96 GiB | 0.026 | 4.2 |
| 8 | Fusion, late (decision-level) | Where in the network fusion should happen | 0.644 [0.61, 0.67]† | 0.783 | 1.70 GiB | 0.052 | 8.5 |
| 9 | **Fusion, cross-attention (reference)** | The proposed model | 0.676 [0.65, 0.70]† | 0.806 | 1.38 GiB | 0.041 | 6.3 |
| 10 | Row 9 − modality dropout | **Is modality dropout load-bearing?** | 0.685 [0.63, 0.73]† | 0.812 | 1.38 GiB | 0.051 | 6.3 |
| 11 | Row 9, evaluated SAR-only at inference | **Graceful degradation — the operational case** | 0.674 [0.65, 0.69]† | 0.805 | (row 9) | (row 9) | 4.6 |
| 12 | Row 9 − compound loss (plain BCE) | Does the imbalance objective matter? | 0.730 [0.71, 0.75]† | 0.844 | 1.38 GiB | 0.048 | 6.3 |
| 13 | Row 9, random split instead of basin-blocked | **The leakage row. Expect an inflated score.** | 0.730 [0.70, 0.75]† | 0.844 | 1.38 GiB | 0.047 | 6.3 |

**Uncertainty from a single fixed split reflects basin-to-basin variation, not variation in the choice of split** (docs/09 §2.2). † Fewer than 8 test basins: a percentile bootstrap over so few basins undercovers (simulated coverage ≈ 0.80 at K = 4 for a nominal 0.95), so the intervals are **indicative only** and the gaps below carry no resolvable/not-resolvable verdict — the per-basin direction is reported instead. That is the honest statement of what this data can resolve.

## Beneath the table: the gaps that matter (paired basin bootstrap, docs/09 §3.3 and §5.1)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | Verdict | Expected sign |
| --- | --- | --- | --- | --- | --- | --- |
| 2 − 1 | What careful engineering adds to naive Otsu | — | -0.028 [-0.114, +0.029] | 2/3 basins | none: K = 3 < 8 basins | large positive |
| 4 − 2 | What learning adds over a good physics baseline | 42 | +0.078 [-0.040, +0.238] | 2/3 basins | none: K = 3 < 8 basins | positive, possibly modest |
| 5 − 4 | What terrain context is worth | 42 | -0.000 [-0.040, +0.025] | 2/3 basins | none: K = 3 < 8 basins | positive |
| 9 − 7 | Whether *where* you fuse matters (vs early) | 42 | +0.001 [-0.028, +0.020] | 2/3 basins | none: K = 3 < 8 basins | small |
| 9 − 8 | Whether *where* you fuse matters (vs late) | 42 | +0.039 [-0.023, +0.108] | 2/3 basins | none: K = 3 < 8 basins | small |
| 9 − 11 | The cost of losing optical (9 − 11) | 42, 43, 44 | +0.002 [+0.000, +0.004] | 3/3 basins | none: K = 3 < 8 basins | positive; small is good |
| 9 − 10 | Whether modality dropout is load-bearing (9 − 10) | 42 | -0.001 [-0.012, +0.017] | 1/3 basins | none: K = 3 < 8 basins | positive |
| 9 − 12 | What the compound loss was worth (9 − 12) | 42 | -0.047 [-0.069, -0.034] | 0/3 basins | none: K = 3 < 8 basins | positive |
| 13 − 9 | The size of the spatial-leakage illusion (13 − 9) | 42 | +0.046 [+0.027, +0.061] | 3/3 basins | none: K = 3 < 8 basins | positive — row 13 must be higher |
| 11 − 4 | Row 11 vs SAR-only from scratch (docs/09 5.2) | 42 | +0.009 [-0.002, +0.016] | 2/3 basins | none: K = 3 < 8 basins | ≈ 0: optical never mattered; > 0: it helped training |
| 9 − 2 | What the machine learning was actually worth (9 − 2) | 42, 43, 44 | +0.081 [-0.022, +0.244] | 2/3 basins | none: K = 3 < 8 basins | positive |

- Row 9 − row 11 (the honest operational expectation: the cost of losing optical): **+0.002** [+0.000, +0.004] — higher in 3 of 3 test basins; with this few basins no interval-based verdict is possible.
- Row 13 − row 9 (the size of the spatial-leakage illusion): **+0.046** [+0.027, +0.061] — higher in 3 of 3 test basins; with this few basins no interval-based verdict is possible.
- Row 9 − row 2 (what the machine learning was actually worth): **+0.081** [-0.022, +0.244] — higher in 2 of 3 test basins; with this few basins no interval-based verdict is possible.

Row 6 is scored on cloud-free chips only — a different test set — and is excluded from every paired comparison; its comparators on the same chips are below.

## Row 6's test set: every row on the cloud-free test chips

| Row | Macro IoU at τ* on the cloud-free test chips | Chips | Same chips as row 6? |
| --- | --- | --- | --- |
| 4 | 0.831 | 7 | yes |
| 5 | 0.797 | 7 | yes |
| 6 | 0.754 | 7 | yes |
| 7 | 0.838 | 7 | yes |
| 8 | 0.796 | 7 | yes |
| 9 | 0.814 | 7 | yes |
| 10 | 0.739 | 7 | yes |
| 11 | 0.813 | 7 | yes |
| 12 | 0.879 | 7 | yes |
| 13 | 0.874 | 7 | yes |
| row10_fusion_no_modality_dropout__eval_sar_only | 0.697 | 7 | yes |

## Supplementary

| Row | Seeds | IoU per seed (mean ± sd) | Pooled IoU | IoU raw σ(z)>τ* | IoU val-selected τ | IoU oracle τ | AP | ECE test raw → cal | Calibration | VRAM prediction error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | — | 0.623 | 0.622 | — | — | — | 0.662 | — | — | — |
| 2 | — | 0.595 | 0.580 | — | — | — | 0.640 | — | — | — |
| 3 | — | 0.606 | 0.587 | — | — | — | 0.690 | — | — | — |
| 4 | 42 | 0.673 | 0.669 | 0.556 | 0.704 | 0.708 | 0.767 | 0.0291 → 0.0099 | platt | -14.1 % |
| 5 | 42 | 0.673 | 0.671 | 0.605 | 0.710 | 0.713 | 0.820 | 0.0269 → 0.0103 | platt | -13.4 % |
| 6 | 42 | 0.754 | 0.686 | 0.243 | 0.768 | 0.768 | 0.841 | 0.0423 → 0.0152 | platt | -13.6 % |
| 7 | 42 | 0.683 | 0.677 | 0.320 | 0.712 | 0.714 | 0.804 | 0.0370 → 0.0072 | platt | -15.6 % |
| 8 | 42 | 0.644 | 0.647 | 0.500 | 0.660 | 0.671 | 0.770 | 0.0331 → 0.0096 | platt | -4.5 % |
| 9 | 42, 43, 44 | 0.676 ± 0.024 | 0.680 | 0.535 | 0.698 | 0.701 | 0.800 | 0.0313 → 0.0076 | platt | -7.8 % |
| 10 | 42 | 0.685 | 0.679 | 0.623 | 0.723 | 0.738 | 0.822 | 0.0258 → 0.0098 | platt | -8.0 % |
| 11 | 42, 43, 44 | 0.674 ± 0.022 | 0.679 | 0.536 | 0.697 | 0.700 | 0.798 | 0.0312 → 0.0076 | platt | — |
| 12 | 42 | 0.730 | 0.729 | 0.602 | 0.756 | 0.760 | 0.833 | 0.0305 → 0.0105 | platt | -7.9 % |
| 13 | 42 | 0.730 | 0.728 | 0.636 | 0.760 | 0.785 | 0.880 | 0.0282 → 0.0092 | platt | -7.8 % |
| row10_fusion_no_modality_dropout__eval_sar_only | 42 | 0.672 | 0.667 | 0.626 | 0.710 | 0.722 | 0.809 | 0.0297 → 0.0112 | platt | — |

## Per-basin IoU at τ* (seeds pooled)

| Row | E1_lowland_po_B0 | E1_lowland_po_B1 | E4_hilly_forest_B0 |
| --- | --- | --- | --- |
| 1 | 0.553 | 0.639 | 0.676 |
| 2 | 0.439 | 0.668 | 0.678 |
| 3 | 0.414 | 0.726 | 0.678 |
| 4 | 0.677 | 0.628 | 0.715 |
| 5 | 0.702 | 0.642 | 0.675 |
| 6 | — | 0.883 | 0.625 |
| 7 | 0.680 | 0.623 | 0.747 |
| 8 | 0.656 | 0.667 | 0.610 |
| 9 | 0.684 | 0.646 | 0.697 |
| 10 | 0.698 | 0.626 | 0.730 |
| 11 | 0.683 | 0.646 | 0.693 |
| 12 | 0.727 | 0.712 | 0.752 |
| 13 | 0.740 | 0.704 | 0.746 |
| row10_fusion_no_modality_dropout__eval_sar_only | 0.692 | 0.613 | 0.712 |

## Baseline mask attribution (P1-M2.4, row 2 at its operating point, test)

| Variant of row 2 | Test macro IoU | Δ vs all masks | Macro precision | Macro recall |
| --- | --- | --- | --- | --- |
| all masks | 0.595 | +0.000 | 0.733 | 0.790 |
| no slope | 0.595 | +0.000 | 0.733 | 0.790 |
| no hand | 0.563 | -0.032 | 0.688 | 0.790 |
| no layover shadow | 0.595 | +0.000 | 0.733 | 0.790 |
| no permanent water | 0.595 | +0.000 | 0.733 | 0.790 |
| or rule | 0.577 | -0.018 | 0.702 | 0.800 |

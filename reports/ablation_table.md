# Ablation table — synthetic benchmark, 13 rows

> **SYNTHETIC BENCHMARK.** Every number below is measured on the physics-based synthetic benchmark (`floodfusion.data.synthetic`), not on real Sentinel-1/2 data. It validates the pipeline and the experimental design; it is **not** a real-world accuracy claim.

**Protocol** (docs/09 §2.2): one fixed basin-blocked train/val/test split; test basins: E1_lowland_po_B0, E1_lowland_po_B1, E4_hilly_forest_B0 (3); seed protocol **full** (decided before the grid from a measured row-9 run: projected full grid 107 min vs budget 75 min); seeds per learned row — row 4: 3, row 5: 3, row 6: 3, row 7: 3, row 8: 3, row 9: 3, row 10: 3, row 12: 3, row 13: 3; gaps compare **matched seeds only**. **IoU** = macro over test basins at the cost-optimal τ* = 1/(1+r) = 0.25 (r = 3) after calibration on validation (prior correction + temperature, Platt if clearly better); seeds pooled per basin; **[CI]** = 95 % percentile bootstrap over test basins (10 000 resamples). Baselines (rows 1–3): threshold offset chosen on validation by the same cost rule. **Peak VRAM** = `torch.cuda.max_memory_allocated()` during training; **ms/chip** = 256² chip, batch 1, ≥ 20 warm-up iterations after a sustained warm-up, `synchronize()` around the timer (docs/09 §7).

Manifest `3e9bbf7d4ae61bdc…`, git `588d66c4363f`. Regenerate with `python -m floodfusion.evaluate --ablation-table --runs-root runs`.

| #   | Configuration | Purpose — the question the row answers | IoU | F1  | Peak VRAM | Train h | ms/chip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Global Otsu on VV, no masks | The floor. Is anything learned at all? | 0.623 [0.55, 0.68]† | 0.766 | — | — | 30.0 (CPU) |
| 2 | Baseline: adaptive threshold + change + masks | The operational reference you must beat | 0.595 [0.44, 0.68]† | 0.740 | — | — | 28.7 (CPU) |
| 3 | Baseline + MNDWI where cloud-free | What optical adds to a non-learned method | 0.606 [0.41, 0.73]† | 0.745 | — | — | 21.0 (CPU) |
| 4 | U-Net, SAR only (VV, VH) | Does learning beat thresholding on the same inputs? | 0.676 [0.63, 0.72]† | 0.806 | 1.72 GiB | 0.056 | 4.2 |
| 5 | U-Net, SAR + DEM derivatives | What terrain context is worth | 0.691 [0.67, 0.73]† | 0.817 | 1.70 GiB | 0.042 | 4.2 |
| 6 | U-Net, optical only, cloud-free chips only | The optical ceiling — an upper bound you cannot deploy | 0.774 [0.71, 0.84]† | 0.871 | 0.94 GiB | 0.018 | 4.1 |
| 7 | Fusion, early (channel concatenation) | Does naive fusion help? | 0.685 [0.64, 0.72]† | 0.813 | 0.96 GiB | 0.035 | 4.2 |
| 8 | Fusion, late (decision-level) | Where in the network fusion should happen | 0.654 [0.63, 0.67]† | 0.791 | 1.70 GiB | 0.048 | 9.3 |
| 9 | **Fusion, cross-attention (reference)** | The proposed model | 0.676 [0.64, 0.69]† | 0.806 | 1.38 GiB | 0.048 | 6.3 |
| 10 | Row 9 − modality dropout | **Is modality dropout load-bearing?** | 0.683 [0.65, 0.70]† | 0.811 | 1.38 GiB | 0.045 | 6.3 |
| 11 | Row 9, evaluated SAR-only at inference | **Graceful degradation — the operational case** | 0.674 [0.64, 0.69]† | 0.805 | (row 9) | (row 9) | 4.6 |
| 12 | Row 9 − compound loss (plain BCE) | Does the imbalance objective matter? | 0.740 [0.68, 0.80]† | 0.849 | 1.38 GiB | 0.048 | 6.3 |
| 13 | Row 9, random split instead of basin-blocked | **The leakage row. Expect an inflated score.** | 0.723 [0.71, 0.74]† | 0.839 | 1.38 GiB | 0.050 | 6.3 |

**Uncertainty from a single fixed split reflects basin-to-basin variation, not variation in the choice of split** (docs/09 §2.2). † Fewer than 8 test basins: a percentile bootstrap over so few basins undercovers (simulated coverage ≈ 0.80 at K = 4 for a nominal 0.95), so the intervals are **indicative only** and the gaps below carry no verdict — the per-basin direction is reported instead. With 8 or more basins the verdict is an exact sign-flip permutation test of the per-basin differences, Holm-adjusted across the gaps.

## Beneath the table: the gaps that matter (paired basin bootstrap, docs/09 §3.3 and §5.1)

| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip) | Verdict | Expected sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 − 1 | What careful engineering adds to naive Otsu | — | -0.028 [-0.114, +0.029] | 2/3 basins | 1.000 | none: K = 3 < 8 basins | large positive |
| 4 − 2 | What learning adds over a good physics baseline | 42, 43, 44 | +0.081 [-0.034, +0.236] | 2/3 basins | 0.500 | none: K = 3 < 8 basins | positive, possibly modest |
| 5 − 4 | What terrain context is worth | 42, 43, 44 | +0.016 [+0.003, +0.035] | 3/3 basins | 0.250 | none: K = 3 < 8 basins | positive |
| 9 − 7 | Whether *where* you fuse matters (vs early) | 42, 43, 44 | -0.009 [-0.029, +0.002] | 2/3 basins | 1.000 | none: K = 3 < 8 basins | small |
| 9 − 8 | Whether *where* you fuse matters (vs late) | 42, 43, 44 | +0.022 [-0.026, +0.066] | 2/3 basins | 0.750 | none: K = 3 < 8 basins | small |
| 9 − 11 | The cost of losing optical (9 − 11) | 42, 43, 44 | +0.002 [+0.000, +0.005] | 3/3 basins | 0.250 | none: K = 3 < 8 basins | positive; small is good |
| 9 − 10 | Whether modality dropout is load-bearing (9 − 10) | 42, 43, 44 | -0.007 [-0.009, -0.003] | 0/3 basins | 0.250 | none: K = 3 < 8 basins | positive |
| 9 − 12 | What the compound loss was worth (9 − 12) | 42, 43, 44 | -0.064 [-0.104, +0.009] | 1/3 basins | 0.500 | none: K = 3 < 8 basins | positive |
| 13 − 9 | The size of the spatial-leakage illusion (13 − 9) | 42, 43, 44 | +0.048 [+0.029, +0.066] | 3/3 basins | 0.250 | none: K = 3 < 8 basins | positive — row 13 must be higher |
| 11 − 4 | Row 11 vs SAR-only from scratch (docs/09 5.2) | 42, 43, 44 | -0.002 [-0.029, +0.012] | 2/3 basins | 1.000 | none: K = 3 < 8 basins | ≈ 0: optical never mattered; > 0: it helped training |
| 9 − 2 | What the machine learning was actually worth (9 − 2) | 42, 43, 44 | +0.081 [-0.023, +0.248] | 2/3 basins | 0.750 | none: K = 3 < 8 basins | positive |

- Row 9 − row 11 (the honest operational expectation: the cost of losing optical): **+0.002** [+0.000, +0.005] — higher in 3 of 3 test basins; with this few basins no verdict is possible.
- Row 13 − row 9 (the size of the spatial-leakage illusion): **+0.048** [+0.029, +0.066] — higher in 3 of 3 test basins; with this few basins no verdict is possible.
- Row 9 − row 2 (what the machine learning was actually worth): **+0.081** [-0.023, +0.248] — higher in 2 of 3 test basins; with this few basins no verdict is possible.

Row 6 is scored on cloud-free chips only — a different test set — and is excluded from every paired comparison; its comparators on the same chips are below.

## Row 6's test set: every row on the cloud-free test chips

| Row | Macro IoU at τ* on the cloud-free test chips | Chips | Same chips as row 6? |
| --- | --- | --- | --- |
| 4 | 0.838 | 7 | yes |
| 5 | 0.789 | 7 | yes |
| 6 | 0.776 | 7 | yes |
| 7 | 0.831 | 7 | yes |
| 8 | 0.792 | 7 | yes |
| 9 | 0.814 | 7 | yes |
| 10 | 0.809 | 7 | yes |
| 11 | 0.813 | 7 | yes |
| 12 | 0.856 | 7 | yes |
| 13 | 0.856 | 7 | yes |
| row10_fusion_no_modality_dropout__eval_sar_only | 0.784 | 7 | yes |

## Supplementary

| Row | Seeds | IoU per seed (mean ± sd) | Pooled IoU | IoU raw σ(z)>τ* | IoU val-selected τ | IoU oracle τ | AP | ECE test raw → cal | Calibration | VRAM prediction error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | — | 0.623 | 0.622 | — | — | — | 0.662 | — | — | — |
| 2 | — | 0.595 | 0.580 | — | — | — | 0.640 | — | — | — |
| 3 | — | 0.606 | 0.587 | — | — | — | 0.690 | — | — | — |
| 4 | 42, 43, 44 | 0.676 ± 0.003 | 0.668 | 0.539 | 0.709 | 0.710 | 0.780 | 0.0316 → 0.0118 | platt | -12.9 % |
| 5 | 42, 43, 44 | 0.691 ± 0.017 | 0.671 | 0.464 | 0.708 | 0.711 | 0.821 | 0.0345 → 0.0084 | platt | -13.3 % |
| 6 | 42, 43, 44 | 0.776 ± 0.019 | 0.692 | 0.178 | 0.763 | 0.781 | 0.892 | 0.0428 → 0.0138 | platt | -13.4 % |
| 7 | 42, 43, 44 | 0.685 ± 0.008 | 0.677 | 0.477 | 0.706 | 0.711 | 0.815 | 0.0334 → 0.0079 | platt | -15.5 % |
| 8 | 42, 43, 44 | 0.654 ± 0.018 | 0.647 | 0.376 | 0.670 | 0.673 | 0.797 | 0.0361 → 0.0089 | platt | -4.5 % |
| 9 | 42, 43, 44 | 0.676 ± 0.024 | 0.682 | 0.535 | 0.699 | 0.702 | 0.800 | 0.0311 → 0.0076 | platt | -7.8 % |
| 10 | 42, 43, 44 | 0.683 ± 0.020 | 0.685 | 0.515 | 0.715 | 0.718 | 0.817 | 0.0333 → 0.0076 | platt | -8.0 % |
| 11 | 42, 43, 44 | 0.674 ± 0.022 | 0.680 | 0.536 | 0.698 | 0.701 | 0.798 | 0.0310 → 0.0076 | platt | — |
| 12 | 42, 43, 44 | 0.740 ± 0.006 | 0.733 | 0.534 | 0.743 | 0.749 | 0.829 | 0.0329 → 0.0064 | platt | -7.8 % |
| 13 | 42, 43, 44 | 0.723 ± 0.010 | 0.728 | 0.615 | 0.753 | 0.771 | 0.860 | 0.0302 → 0.0092 | platt | -7.8 % |
| row10_fusion_no_modality_dropout__eval_sar_only | 42, 43, 44 | 0.663 ± 0.020 | 0.654 | 0.526 | 0.697 | 0.701 | 0.794 | 0.0325 → 0.0081 | platt | — |

## Per-basin IoU at τ* (seeds pooled)

| Row | E1_lowland_po_B0 | E1_lowland_po_B1 | E4_hilly_forest_B0 |
| --- | --- | --- | --- |
| 1 | 0.553 | 0.639 | 0.676 |
| 2 | 0.439 | 0.668 | 0.678 |
| 3 | 0.414 | 0.726 | 0.678 |
| 4 | 0.675 | 0.634 | 0.718 |
| 5 | 0.678 | 0.669 | 0.726 |
| 6 | — | 0.838 | 0.711 |
| 7 | 0.687 | 0.643 | 0.724 |
| 8 | 0.661 | 0.671 | 0.629 |
| 9 | 0.687 | 0.645 | 0.695 |
| 10 | 0.691 | 0.653 | 0.704 |
| 11 | 0.687 | 0.645 | 0.689 |
| 12 | 0.678 | 0.742 | 0.799 |
| 13 | 0.735 | 0.711 | 0.724 |
| row10_fusion_no_modality_dropout__eval_sar_only | 0.691 | 0.630 | 0.668 |

## Baseline mask attribution (P1-M2.4, row 2 at its operating point, test)

| Variant of row 2 | Test macro IoU | Δ vs all masks | Macro precision | Macro recall |
| --- | --- | --- | --- | --- |
| all masks | 0.595 | +0.000 | 0.733 | 0.790 |
| no slope | 0.595 | +0.000 | 0.733 | 0.790 |
| no hand | 0.563 | -0.032 | 0.688 | 0.790 |
| no layover shadow | 0.595 | +0.000 | 0.733 | 0.790 |
| no permanent water | 0.595 | +0.000 | 0.733 | 0.790 |
| or rule | 0.577 | -0.018 | 0.702 | 0.800 |
| no change term | 0.575 | -0.020 | 0.705 | 0.792 |

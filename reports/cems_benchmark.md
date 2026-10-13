# CEMS benchmark — independent flood maps, selected by a rule fixed in advance

Zero-shot: no model saw any of these events; each uses the τ* and calibration of its own training data (for a model with an optical input, its SAR-only calibration wherever a window had no usable optical). Primary stratum: products drawn on **sar** satellites; excluded: EMSR664. Units: 38 activations in 30 clusters (within 100 km and 60 days). Protocol: `reports/protocol_v1_1.md` (amendment 1). Sign-flip p-values are exact for K ≤ 20 and Monte Carlo (200,000 patterns) above. Products: © European Union, Copernicus Emergency Management Service.

## Pre-registered tests — does each hold at both unit definitions?

| Test | Activations | Clusters | Supported (holds at both units) |
| --- | --- | --- | --- |
| P1 | not significant (Holm p = 0.3438) | none: K = 7 < 8 | no verdict |
| P2 | non-inferior (one-sided p = 0.0035 < 0.025, margin 0.05) | not shown non-inferior (one-sided p = 0.0293, margin 0.05) | no |
| P3 | significant (Holm p = 0.0000) | significant (Holm p = 0.0000) | yes |
| P4 | significant (Holm p = 0.0340) | significant (Holm p = 0.0091) | yes |

### Per activation

| Test | Question | B − A | Subset | K | Mean Δ IoU [95 % paired CI of the macro Δ] | B > A in | p | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Fusion helps on independent maps where optical is usable (H2) | Fusion (S1F11) − SAR U-Net (S1F11) | optical | 8 | +0.038 [-0.024, +0.108] | 5/8 | 0.344 | not significant (Holm p = 0.3438) |
| P2 | Without optical, fusion is no worse than the SAR U-Net (R2 in practice, H3) | Fusion (S1F11) − SAR U-Net (S1F11) | sar_only | 15 | -0.020 [-0.038, -0.003] | 3/15 | 0.004 | non-inferior (one-sided p = 0.0035 < 0.025, margin 0.05) |
| P3 | The flood-label recipe beats the water-label recipe on independent maps (H1) | Fusion (Kuro Siwo, flood labels) − Fusion (Kuro Siwo, water labels) | all | 38 | +0.059 [+0.039, +0.083] | 33/38 | 0.000 | significant (Holm p = 0.0000) |
| P4 | The task-specific fusion model vs the foundation model (H4) | Fusion (S1F11) − TerraMind-small (S1F11) | all | 38 | +0.031 [+0.007, +0.055] | 24/38 | 0.017 | significant (Holm p = 0.0340) |

### Per cluster

| Test | Question | B − A | Subset | K | Mean Δ IoU [95 % paired CI of the macro Δ] | B > A in | p | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Fusion helps on independent maps where optical is usable (H2) | Fusion (S1F11) − SAR U-Net (S1F11) | optical | 7 | +0.024 [-0.038, +0.103] | 4/7 | 0.688 | none: K = 7 < 8 |
| P2 | Without optical, fusion is no worse than the SAR U-Net (R2 in practice, H3) | Fusion (S1F11) − SAR U-Net (S1F11) | sar_only | 11 | -0.024 [-0.047, -0.002] | 3/11 | 0.029 | not shown non-inferior (one-sided p = 0.0293, margin 0.05) |
| P3 | The flood-label recipe beats the water-label recipe on independent maps (H1) | Fusion (Kuro Siwo, flood labels) − Fusion (Kuro Siwo, water labels) | all | 30 | +0.062 [+0.039, +0.092] | 27/30 | 0.000 | significant (Holm p = 0.0000) |
| P4 | The task-specific fusion model vs the foundation model (H4) | Fusion (S1F11) − TerraMind-small (S1F11) | all | 30 | +0.039 [+0.013, +0.066] | 20/30 | 0.009 | significant (Holm p = 0.0091) |

### Partial cloud (10-50 % clear), descriptive

| Test | Question | B − A | Subset | K | Mean Δ IoU [95 % paired CI of the macro Δ] | B > A in | p | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Fusion helps on independent maps where optical is usable (H2) | Fusion (S1F11) − SAR U-Net (S1F11) | partial | 15 | -0.059 [-0.093, -0.028] | 2/15 | 0.001 | significant (Holm p = 0.0042) |
| P2 | Without optical, fusion is no worse than the SAR U-Net (R2 in practice, H3) | Fusion (S1F11) − SAR U-Net (S1F11) | partial | 15 | -0.059 [-0.093, -0.028] | 2/15 | 0.001 | significant (Holm p = 0.0042) |
| P3 | The flood-label recipe beats the water-label recipe on independent maps (H1) | Fusion (Kuro Siwo, flood labels) − Fusion (Kuro Siwo, water labels) | partial | 15 | +0.058 [+0.034, +0.090] | 14/15 | 0.000 | significant (Holm p = 0.0005) |
| P4 | The task-specific fusion model vs the foundation model (H4) | Fusion (S1F11) − TerraMind-small (S1F11) | partial | 15 | -0.008 [-0.044, +0.026] | 7/15 | 0.673 | not significant (Holm p = 0.6728) |

### Sensitivity: without units holding < 1 km² of flood (descriptive)

| Test | Question | B − A | Subset | K | Mean Δ IoU [95 % paired CI of the macro Δ] | B > A in | p | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Fusion helps on independent maps where optical is usable (H2) | Fusion (S1F11) − SAR U-Net (S1F11) | optical | 7 | +0.042 [-0.028, +0.123] | 4/7 | 0.359 | none: K = 7 < 8 |
| P2 | Without optical, fusion is no worse than the SAR U-Net (R2 in practice, H3) | Fusion (S1F11) − SAR U-Net (S1F11) | sar_only | 14 | -0.019 [-0.038, -0.001] | 3/14 | 0.005 | non-inferior (one-sided p = 0.0046 < 0.025, margin 0.05) |
| P3 | The flood-label recipe beats the water-label recipe on independent maps (H1) | Fusion (Kuro Siwo, flood labels) − Fusion (Kuro Siwo, water labels) | all | 35 | +0.054 [+0.037, +0.073] | 31/35 | 0.000 | significant (Holm p = 0.0000) |
| P4 | The task-specific fusion model vs the foundation model (H4) | Fusion (S1F11) − TerraMind-small (S1F11) | all | 35 | +0.033 [+0.007, +0.058] | 22/35 | 0.019 | significant (Holm p = 0.0192) |

### Sensitivity: without the pairs that overlap Kuro Siwo training areas (descriptive)

| Test | Question | B − A | Subset | K | Mean Δ IoU [95 % paired CI of the macro Δ] | B > A in | p | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Fusion helps on independent maps where optical is usable (H2) | Fusion (S1F11) − SAR U-Net (S1F11) | optical | 7 | +0.046 [-0.025, +0.124] | 5/7 | 0.328 | none: K = 7 < 8 |
| P2 | Without optical, fusion is no worse than the SAR U-Net (R2 in practice, H3) | Fusion (S1F11) − SAR U-Net (S1F11) | sar_only | 14 | -0.024 [-0.041, -0.009] | 3/14 | 0.007 | non-inferior (one-sided p = 0.0067 < 0.025, margin 0.05) |
| P3 | The flood-label recipe beats the water-label recipe on independent maps (H1) | Fusion (Kuro Siwo, flood labels) − Fusion (Kuro Siwo, water labels) | all | 36 | +0.061 [+0.040, +0.086] | 32/36 | 0.000 | significant (Holm p = 0.0000) |
| P4 | The task-specific fusion model vs the foundation model (H4) | Fusion (S1F11) − TerraMind-small (S1F11) | all | 36 | +0.030 [+0.005, +0.054] | 23/36 | 0.028 | significant (Holm p = 0.0276) |

## Every model, primary stratum, per activation (macro IoU at τ* [95 % CI]; oracle threshold)

| Model | Role | IoU at τ* | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.281 [0.237, 0.323] | 0.281 | 38 |
| Fusion (S1F11) | test | 0.258 [0.210, 0.307] | 0.260 | 38 |
| TerraMind-small (S1F11) | test | 0.226 [0.177, 0.278] | 0.270 | 38 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.394 [0.339, 0.448] | 0.403 | 38 |
| Fusion (Kuro Siwo, flood labels) | test | 0.397 [0.342, 0.452] | 0.406 | 38 |
| Fusion (Kuro Siwo, water labels) | test | 0.338 [0.282, 0.395] | 0.353 | 38 |
| Physics baseline (row 2) | test | 0.371 [0.312, 0.430] | — | 38 |
| Physics baseline, seasonal reference (row 2s) | test | 0.340 [0.287, 0.393] | — | 38 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.272 [0.225, 0.320] | 0.273 | 38 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.187 [0.143, 0.234] | 0.234 | 38 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.398 [0.343, 0.453] | 0.407 | 38 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.335 [0.280, 0.392] | 0.351 | 38 |

## Activations

| Activation | Cluster | Stratum | Scored km² | Flood km² | Clear optical |
| --- | --- | --- | --- | --- | --- |
| EMSR657 | EMSR657 | sar | 1377 | 77.2 | 0.26 |
| EMSR664 | EMSR664 | sar | 1382 | 46.0 | 1.00 |
| EMSR668 | EMSR668 | sar | 101 | 2.9 | 0.68 |
| EMSR683 | EMSR683 | mixed (per pair) | 241 | 9.9 | 0.29 |
| EMSR692 | EMSR692 | optical | 3213 | 1102.9 | 0.47 |
| EMSR705 | EMSR705 | sar | 499 | 4.1 | 0.04 |
| EMSR706 | EMSR706 | sar | 479 | 6.3 | 0.39 |
| EMSR708 | EMSR706 | sar | 911 | 36.7 | 0.19 |
| EMSR711 | EMSR711 | sar | 1351 | 81.3 | 0.27 |
| EMSR712 | EMSR712 | sar | 3485 | 103.6 | 0.00 |
| EMSR718 | EMSR718 | sar | 59 | 0.3 | 0.00 |
| EMSR722 | EMSR722 | sar | 308 | 10.5 | 0.00 |
| EMSR725 | EMSR725 | sar | 18 | 1.7 | 0.88 |
| EMSR728 | EMSR728 | sar | 4461 | 35.3 | 0.16 |
| EMSR754 | EMSR754 | optical | 278 | 20.2 | 0.00 |
| EMSR755 | EMSR755 | sar | 197 | 1.4 | 0.00 |
| EMSR756 | EMSR755 | sar | 4537 | 39.3 | 0.49 |
| EMSR759 | EMSR759 | optical | 1199 | 60.2 | 0.77 |
| EMSR761 | EMSR755 | sar | 496 | 1.6 | 0.38 |
| EMSR763 | EMSR755 | sar | 840 | 49.7 | 0.40 |
| EMSR766 | EMSR766 | sar | 302 | 1.6 | 0.45 |
| EMSR768 | EMSR768 | sar | 264 | 0.5 | 0.78 |
| EMSR771 | EMSR771 | sar | 1490 | 12.2 | 0.00 |
| EMSR773 | EMSR773 | mixed (per pair) | 3227 | 84.3 | 0.37 |
| EMSR775 | EMSR775 | optical | 11 | 0.0 | 0.00 |
| EMSR783 | EMSR783 | sar | 1047 | 8.3 | 0.00 |
| EMSR788 | EMSR788 | sar | 409 | 23.3 | 0.38 |
| EMSR789 | EMSR789 | sar | 635 | 13.7 | 0.00 |
| EMSR796 | EMSR789 | mixed (per pair) | 437 | 1.5 | 0.05 |
| EMSR797 | EMSR797 | sar | 951 | 2.0 | 0.00 |
| EMSR838 | EMSR838 | sar | 2033 | 242.0 | 0.59 |
| EMSR841 | EMSR841 | sar | 31 | 0.6 | 0.14 |
| EMSR843 | EMSR843 | optical | 327 | 0.3 | 0.15 |
| EMSR844 | EMSR844 | sar | 118 | 3.5 | 0.52 |
| EMSR848 | EMSR848 | sar | 6037 | 278.1 | 0.18 |
| EMSR850 | EMSR850 | sar | 270 | 9.7 | 0.19 |
| EMSR854 | EMSR850 | sar | 266 | 6.1 | 0.93 |
| EMSR856 | EMSR850 | sar | 407 | 26.0 | 0.00 |
| EMSR857 | EMSR857 | sar | 6364 | 1861.9 | 0.56 |
| EMSR861 | EMSR861 | sar | 7420 | 56.0 | 0.00 |
| EMSR864 | EMSR861 | sar | 2866 | 238.1 | 0.00 |
| EMSR866 | EMSR866 | sar | 1491 | 48.4 | 0.00 |
| EMSR868 | EMSR868 | sar | 1643 | 52.1 | 0.99 |
| EMSR871 | EMSR871 | sar | 124 | 2.0 | 0.00 |

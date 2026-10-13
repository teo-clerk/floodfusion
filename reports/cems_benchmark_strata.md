# CEMS benchmark — every model on every stratum and optical subset (descriptive)

Protocol §5 (`reports/protocol_v1_1.md`): every model on every stratum, the optical and SAR-only subsets, the oracle-threshold IoU and K per subset. Re-read from the stored confusion counts of `reports/cems_benchmark_scores.json` by `scripts/cems_strata.py`; nothing is re-scored, and nothing here carries a verdict. Units are activations, AOIs and seeds pooled. A subset is chosen by the unit's share of scored pixels with clear optical. Products: © European Union, Copernicus Emergency Management Service.

## Drawn on radar satellites (the primary stratum)

38 activations, 3395 km² of mapped flood.

**all** (every unit): K = 38.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
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

**optical** (≥ 50 % clear): K = 8.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.269 [0.186, 0.342] | 0.282 | 8 |
| Fusion (S1F11) | test | 0.307 [0.191, 0.439] | 0.307 | 8 |
| TerraMind-small (S1F11) | test | 0.312 [0.192, 0.451] | 0.318 | 8 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.407 [0.316, 0.518] | 0.407 | 8 |
| Fusion (Kuro Siwo, flood labels) | test | 0.407 [0.318, 0.527] | 0.412 | 8 |
| Fusion (Kuro Siwo, water labels) | test | 0.353 [0.226, 0.502] | 0.361 | 8 |
| Physics baseline (row 2) | test | 0.354 [0.241, 0.489] | — | 8 |
| Physics baseline, seasonal reference (row 2s) | test | 0.376 [0.285, 0.510] | — | 8 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.269 [0.176, 0.369] | 0.275 | 8 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.226 [0.128, 0.359] | 0.249 | 8 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.410 [0.321, 0.529] | 0.413 | 8 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.350 [0.223, 0.499] | 0.358 | 8 |

**partial** (10–50 % clear): K = 15.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.313 [0.230, 0.393] | 0.313 | 15 |
| Fusion (S1F11) | test | 0.254 [0.180, 0.328] | 0.255 | 15 |
| TerraMind-small (S1F11) | test | 0.262 [0.187, 0.334] | 0.305 | 15 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.418 [0.315, 0.518] | 0.432 | 15 |
| Fusion (Kuro Siwo, flood labels) | test | 0.423 [0.321, 0.521] | 0.435 | 15 |
| Fusion (Kuro Siwo, water labels) | test | 0.365 [0.266, 0.462] | 0.383 | 15 |
| Physics baseline (row 2) | test | 0.408 [0.304, 0.506] | — | 15 |
| Physics baseline, seasonal reference (row 2s) | test | 0.339 [0.252, 0.425] | — | 15 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.307 [0.222, 0.390] | 0.309 | 15 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.209 [0.136, 0.284] | 0.253 | 15 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.424 [0.322, 0.522] | 0.436 | 15 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.361 [0.263, 0.458] | 0.379 | 15 |

**sar_only** (< 10 % clear): K = 15.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.255 [0.197, 0.315] | 0.265 | 15 |
| Fusion (S1F11) | test | 0.234 [0.170, 0.303] | 0.242 | 15 |
| TerraMind-small (S1F11) | test | 0.144 [0.089, 0.201] | 0.210 | 15 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.364 [0.282, 0.442] | 0.378 | 15 |
| Fusion (Kuro Siwo, flood labels) | test | 0.365 [0.284, 0.442] | 0.378 | 15 |
| Fusion (Kuro Siwo, water labels) | test | 0.303 [0.234, 0.370] | 0.333 | 15 |
| Physics baseline (row 2) | test | 0.342 [0.251, 0.429] | — | 15 |
| Physics baseline, seasonal reference (row 2s) | test | 0.321 [0.241, 0.400] | — | 15 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.240 [0.176, 0.307] | 0.248 | 15 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.145 [0.090, 0.203] | 0.212 | 15 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.366 [0.284, 0.443] | 0.379 | 15 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.302 [0.233, 0.369] | 0.332 | 15 |

## Drawn on optical VHR satellites

7 activations, 1222 km² of mapped flood.

**all** (every unit): K = 7. Fewer than 8 units: indicative only.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.113 [0.042, 0.187] | 0.121 | 7 |
| Fusion (S1F11) | test | 0.122 [0.037, 0.222] | 0.127 | 7 |
| TerraMind-small (S1F11) | test | 0.113 [0.023, 0.222] | 0.114 | 7 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.118 [0.044, 0.193] | 0.119 | 7 |
| Fusion (Kuro Siwo, flood labels) | test | 0.127 [0.045, 0.215] | 0.127 | 7 |
| Fusion (Kuro Siwo, water labels) | test | 0.103 [0.033, 0.180] | 0.107 | 7 |
| Physics baseline (row 2) | test | 0.116 [0.025, 0.217] | — | 7 |
| Physics baseline, seasonal reference (row 2s) | test | 0.110 [0.023, 0.208] | — | 7 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.108 [0.034, 0.187] | 0.116 | 7 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.078 [0.019, 0.140] | 0.078 | 7 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.127 [0.045, 0.214] | 0.127 | 7 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.103 [0.033, 0.178] | 0.107 | 7 |

**optical** (≥ 50 % clear): K = 2. Fewer than 8 units: indicative only.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.118 [0.000, 0.236] | 0.119 | 2 |
| Fusion (S1F11) | test | 0.189 [0.000, 0.378] | 0.203 | 2 |
| TerraMind-small (S1F11) | test | 0.200 [0.000, 0.399] | 0.204 | 2 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.113 [0.000, 0.227] | 0.117 | 2 |
| Fusion (Kuro Siwo, flood labels) | test | 0.132 [0.000, 0.263] | 0.132 | 2 |
| Fusion (Kuro Siwo, water labels) | test | 0.137 [0.000, 0.274] | 0.142 | 2 |
| Physics baseline (row 2) | test | 0.149 [0.000, 0.299] | — | 2 |
| Physics baseline, seasonal reference (row 2s) | test | 0.125 [0.000, 0.251] | — | 2 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.140 [0.000, 0.280] | 0.147 | 2 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.060 [0.000, 0.120] | 0.064 | 2 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.131 [0.000, 0.262] | 0.131 | 2 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.134 [0.000, 0.268] | 0.139 | 2 |

**partial** (10–50 % clear): K = 3. Fewer than 8 units: indicative only.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.098 [0.067, 0.153] | 0.185 | 3 |
| Fusion (S1F11) | test | 0.085 [0.023, 0.123] | 0.123 | 3 |
| TerraMind-small (S1F11) | test | 0.062 [0.006, 0.135] | 0.093 | 3 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.103 [0.053, 0.130] | 0.119 | 3 |
| Fusion (Kuro Siwo, flood labels) | test | 0.104 [0.047, 0.133] | 0.108 | 3 |
| Fusion (Kuro Siwo, water labels) | test | 0.077 [0.043, 0.137] | 0.085 | 3 |
| Physics baseline (row 2) | test | 0.058 [0.032, 0.103] | — | 3 |
| Physics baseline, seasonal reference (row 2s) | test | 0.057 [0.014, 0.123] | — | 3 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.087 [0.020, 0.171] | 0.155 | 3 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.074 [0.005, 0.177] | 0.105 | 3 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.104 [0.047, 0.133] | 0.108 | 3 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.078 [0.042, 0.140] | 0.087 | 3 |

**sar_only** (< 10 % clear): K = 2. Fewer than 8 units: indicative only.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.131 [0.000, 0.262] | 0.132 | 2 |
| Fusion (S1F11) | test | 0.110 [0.000, 0.219] | 0.110 | 2 |
| TerraMind-small (S1F11) | test | 0.103 [0.000, 0.206] | 0.105 | 2 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.145 [0.000, 0.289] | 0.146 | 2 |
| Fusion (Kuro Siwo, flood labels) | test | 0.158 [0.000, 0.317] | 0.160 | 2 |
| Fusion (Kuro Siwo, water labels) | test | 0.109 [0.000, 0.218] | 0.110 | 2 |
| Physics baseline (row 2) | test | 0.170 [0.000, 0.339] | — | 2 |
| Physics baseline, seasonal reference (row 2s) | test | 0.172 [0.000, 0.344] | — | 2 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.110 [0.000, 0.219] | 0.110 | 2 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.103 [0.000, 0.206] | 0.105 | 2 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.158 [0.000, 0.317] | 0.160 | 2 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.109 [0.000, 0.218] | 0.110 | 2 |

## EMSR664 (seen in v1.0; every stratum)

1 activations, 46 km² of mapped flood.

**all** (every unit): K = 1. Fewer than 8 units: indicative only.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.421 [—, —] | 0.425 | 1 |
| Fusion (S1F11) | test | 0.440 [—, —] | 0.450 | 1 |
| TerraMind-small (S1F11) | test | 0.471 [—, —] | 0.471 | 1 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.609 [—, —] | 0.615 | 1 |
| Fusion (Kuro Siwo, flood labels) | test | 0.606 [—, —] | 0.623 | 1 |
| Fusion (Kuro Siwo, water labels) | test | 0.570 [—, —] | 0.588 | 1 |
| Physics baseline (row 2) | test | 0.573 [—, —] | — | 1 |
| Physics baseline, seasonal reference (row 2s) | test | 0.324 [—, —] | — | 1 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.364 [—, —] | 0.369 | 1 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.158 [—, —] | 0.293 | 1 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.607 [—, —] | 0.623 | 1 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.564 [—, —] | 0.587 | 1 |

**optical** (≥ 50 % clear): K = 1. Fewer than 8 units: indicative only.

| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |
| --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | test | 0.421 [—, —] | 0.425 | 1 |
| Fusion (S1F11) | test | 0.440 [—, —] | 0.450 | 1 |
| TerraMind-small (S1F11) | test | 0.471 [—, —] | 0.471 | 1 |
| SAR U-Net (Kuro Siwo, flood) | test | 0.609 [—, —] | 0.615 | 1 |
| Fusion (Kuro Siwo, flood labels) | test | 0.606 [—, —] | 0.623 | 1 |
| Fusion (Kuro Siwo, water labels) | test | 0.570 [—, —] | 0.588 | 1 |
| Physics baseline (row 2) | test | 0.573 [—, —] | — | 1 |
| Physics baseline, seasonal reference (row 2s) | test | 0.324 [—, —] | — | 1 |
| Fusion (S1F11), SAR-only everywhere | descriptive | 0.364 [—, —] | 0.369 | 1 |
| TerraMind (S1F11), SAR-only everywhere | descriptive | 0.158 [—, —] | 0.293 | 1 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | descriptive | 0.607 [—, —] | 0.623 | 1 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | descriptive | 0.564 [—, —] | 0.587 | 1 |

**partial** (10–50 % clear): no unit.

**sar_only** (< 10 % clear): no unit.

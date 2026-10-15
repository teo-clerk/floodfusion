# CEMS benchmark, v1.2 scoring (descriptive)

Primary stratum (products drawn on radar), EMSR664 excluded; units with reference flood in their scored region. IoU at τ* is v1.1's definition (exact pixels, seeds pooled; v1.1's own table also counted units without reference flood). The other columns are per seed, with the seed's τ* and prior, then averaged: IoU at τ* on the binned scores is the baseline to read the self-calibrated IoU against. Exploratory: this benchmark has been seen.

## Per activation: all units with reference flood (K = 38)

| Model | K | IoU at τ* | IoU at τ*, binned | IoU, self-calibrated | IoU, oracle | AP |
| --- | --- | --- | --- | --- | --- | --- |
| v1.2 (row21_fusion_context_seasonal) | 38 | 0.417 | 0.418 | 0.420 | 0.468 | 0.598 |
| TerraMind, fair (terramind_base_fair) | 38 | 0.405 | 0.407 | 0.362 | 0.446 | 0.583 |
| v1.1 best (fusion, Kuro Siwo flood labels) | 38 | 0.407 | 0.408 | 0.407 | 0.438 | 0.527 |
| Fusion (S1F11) | 38 | 0.266 | 0.266 | 0.254 | 0.294 | 0.284 |
| TerraMind-small (S1F11), corrected | 38 | 0.299 | 0.300 | 0.303 | 0.356 | 0.428 |
| SAR U-Net (S1F11) | 38 | 0.281 | 0.281 | 0.248 | 0.314 | 0.317 |
| SAR U-Net (Kuro Siwo, flood) | 38 | 0.395 | 0.395 | 0.393 | 0.429 | 0.525 |
| Physics baseline (row 2) | 38 | 0.373 | — | — | — | — |

## Per cluster: all units with reference flood (K = 30)

| Model | K | IoU at τ* | IoU at τ*, binned | IoU, self-calibrated | IoU, oracle | AP |
| --- | --- | --- | --- | --- | --- | --- |
| v1.2 (row21_fusion_context_seasonal) | 30 | 0.441 | 0.442 | 0.450 | 0.493 | 0.628 |
| TerraMind, fair (terramind_base_fair) | 30 | 0.429 | 0.432 | 0.390 | 0.471 | 0.612 |
| v1.1 best (fusion, Kuro Siwo flood labels) | 30 | 0.431 | 0.432 | 0.433 | 0.464 | 0.553 |
| Fusion (S1F11) | 30 | 0.278 | 0.279 | 0.256 | 0.304 | 0.288 |
| TerraMind-small (S1F11), corrected | 30 | 0.312 | 0.313 | 0.324 | 0.372 | 0.445 |
| SAR U-Net (S1F11) | 30 | 0.298 | 0.299 | 0.260 | 0.330 | 0.328 |
| SAR U-Net (Kuro Siwo, flood) | 30 | 0.414 | 0.415 | 0.415 | 0.453 | 0.547 |
| Physics baseline (row 2) | 30 | 0.405 | — | — | — | — |

## Per activation: units covered by GFM (≥ 95 % of scored pixels): every model on the same floods (K = 34)

| Model | K | IoU at τ* | IoU at τ*, binned | IoU, self-calibrated | IoU, oracle | AP |
| --- | --- | --- | --- | --- | --- | --- |
| v1.2 (row21_fusion_context_seasonal) | 34 | 0.419 | 0.420 | 0.415 | 0.467 | 0.594 |
| TerraMind, fair (terramind_base_fair) | 34 | 0.400 | 0.402 | 0.359 | 0.441 | 0.573 |
| v1.1 best (fusion, Kuro Siwo flood labels) | 34 | 0.407 | 0.408 | 0.405 | 0.437 | 0.523 |
| Copernicus GFM (ensemble) | 34 | 0.321 | — | — | 0.427 | 0.517 |
| Fusion (S1F11) | 34 | 0.268 | 0.269 | 0.253 | 0.294 | 0.286 |
| TerraMind-small (S1F11), corrected | 34 | 0.296 | 0.297 | 0.290 | 0.346 | 0.411 |
| SAR U-Net (S1F11) | 34 | 0.281 | 0.282 | 0.241 | 0.311 | 0.313 |
| SAR U-Net (Kuro Siwo, flood) | 34 | 0.391 | 0.392 | 0.389 | 0.425 | 0.517 |
| Physics baseline (row 2) | 34 | 0.381 | — | — | — | — |

## Per cluster: units covered by GFM (≥ 95 % of scored pixels): every model on the same floods (K = 26)

| Model | K | IoU at τ* | IoU at τ*, binned | IoU, self-calibrated | IoU, oracle | AP |
| --- | --- | --- | --- | --- | --- | --- |
| v1.2 (row21_fusion_context_seasonal) | 26 | 0.442 | 0.444 | 0.444 | 0.491 | 0.623 |
| TerraMind, fair (terramind_base_fair) | 26 | 0.425 | 0.428 | 0.387 | 0.466 | 0.602 |
| v1.1 best (fusion, Kuro Siwo flood labels) | 26 | 0.429 | 0.430 | 0.427 | 0.461 | 0.545 |
| Copernicus GFM (ensemble) | 26 | 0.361 | — | — | 0.456 | 0.549 |
| Fusion (S1F11) | 26 | 0.277 | 0.277 | 0.250 | 0.301 | 0.286 |
| TerraMind-small (S1F11), corrected | 26 | 0.308 | 0.309 | 0.310 | 0.360 | 0.425 |
| SAR U-Net (S1F11) | 26 | 0.295 | 0.296 | 0.249 | 0.326 | 0.325 |
| SAR U-Net (Kuro Siwo, flood) | 26 | 0.411 | 0.412 | 0.409 | 0.449 | 0.537 |
| Physics baseline (row 2) | 26 | 0.414 | — | — | — | — |


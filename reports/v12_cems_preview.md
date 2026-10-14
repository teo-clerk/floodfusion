# CEMS benchmark, v1.2 scoring (descriptive)

Primary stratum (products drawn on radar), EMSR664 excluded; units with reference flood in their scored region. IoU at τ* is v1.1's definition (exact pixels, seeds pooled; v1.1's own table also counted units without reference flood). The other columns are per seed, with the seed's τ* and prior, then averaged: IoU at τ* on the binned scores is the baseline to read the self-calibrated IoU against. Exploratory: this benchmark has been seen.

## Per activation: all units with reference flood (K = 38)

| Model | K | IoU at τ* | IoU at τ*, binned | IoU, self-calibrated | IoU, oracle | AP |
| --- | --- | --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | 38 | 0.281 | 0.281 | 0.248 | 0.314 | 0.317 |
| Fusion (S1F11) | 38 | 0.258 | 0.258 | 0.250 | 0.287 | 0.279 |
| TerraMind-small (S1F11) | 38 | 0.226 | 0.232 | 0.228 | 0.309 | 0.363 |
| SAR U-Net (Kuro Siwo, flood) | 38 | 0.394 | 0.395 | 0.393 | 0.429 | 0.525 |
| Fusion (Kuro Siwo, flood labels) | 38 | 0.397 | 0.398 | 0.395 | 0.428 | 0.518 |
| Fusion (Kuro Siwo, water labels) | 38 | 0.338 | 0.338 | 0.348 | 0.377 | 0.430 |
| Physics baseline (row 2) | 38 | 0.371 | — | — | — | — |
| Physics baseline, seasonal reference (row 2s) | 38 | 0.340 | — | — | — | — |
| Fusion (S1F11), SAR-only everywhere | 38 | 0.272 | 0.273 | 0.248 | 0.308 | 0.292 |
| TerraMind (S1F11), SAR-only everywhere | 38 | 0.187 | 0.196 | 0.163 | 0.272 | 0.315 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | 38 | 0.398 | 0.400 | 0.396 | 0.429 | 0.519 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | 38 | 0.335 | 0.336 | 0.346 | 0.375 | 0.427 |
| TerraMind-small (S1F11), corrected | 38 | 0.299 | 0.300 | 0.303 | 0.356 | 0.428 |

## Per cluster: all units with reference flood (K = 30)

| Model | K | IoU at τ* | IoU at τ*, binned | IoU, self-calibrated | IoU, oracle | AP |
| --- | --- | --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | 30 | 0.298 | 0.298 | 0.260 | 0.330 | 0.328 |
| Fusion (S1F11) | 30 | 0.269 | 0.269 | 0.251 | 0.296 | 0.281 |
| TerraMind-small (S1F11) | 30 | 0.230 | 0.236 | 0.244 | 0.316 | 0.369 |
| SAR U-Net (Kuro Siwo, flood) | 30 | 0.414 | 0.415 | 0.415 | 0.453 | 0.547 |
| Fusion (Kuro Siwo, flood labels) | 30 | 0.418 | 0.420 | 0.419 | 0.451 | 0.542 |
| Fusion (Kuro Siwo, water labels) | 30 | 0.356 | 0.356 | 0.369 | 0.397 | 0.445 |
| Physics baseline (row 2) | 30 | 0.403 | — | — | — | — |
| Physics baseline, seasonal reference (row 2s) | 30 | 0.373 | — | — | — | — |
| Fusion (S1F11), SAR-only everywhere | 30 | 0.290 | 0.291 | 0.252 | 0.325 | 0.303 |
| TerraMind (S1F11), SAR-only everywhere | 30 | 0.201 | 0.211 | 0.177 | 0.284 | 0.326 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | 30 | 0.419 | 0.421 | 0.420 | 0.452 | 0.543 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | 30 | 0.354 | 0.354 | 0.367 | 0.395 | 0.442 |
| TerraMind-small (S1F11), corrected | 30 | 0.312 | 0.313 | 0.324 | 0.372 | 0.445 |

## Per activation: units covered by GFM (≥ 95 % of scored pixels): every model on the same floods (K = 34)

| Model | K | IoU at τ* | IoU at τ*, binned | IoU, self-calibrated | IoU, oracle | AP |
| --- | --- | --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | 34 | 0.281 | 0.282 | 0.241 | 0.311 | 0.313 |
| Fusion (S1F11) | 34 | 0.260 | 0.260 | 0.248 | 0.286 | 0.279 |
| TerraMind-small (S1F11) | 34 | 0.229 | 0.234 | 0.224 | 0.306 | 0.358 |
| SAR U-Net (Kuro Siwo, flood) | 34 | 0.391 | 0.392 | 0.389 | 0.425 | 0.516 |
| Fusion (Kuro Siwo, flood labels) | 34 | 0.395 | 0.397 | 0.392 | 0.425 | 0.513 |
| Fusion (Kuro Siwo, water labels) | 34 | 0.337 | 0.338 | 0.346 | 0.374 | 0.423 |
| Physics baseline (row 2) | 34 | 0.379 | — | — | — | — |
| Physics baseline, seasonal reference (row 2s) | 34 | 0.345 | — | — | — | — |
| Fusion (S1F11), SAR-only everywhere | 34 | 0.275 | 0.276 | 0.246 | 0.309 | 0.293 |
| TerraMind (S1F11), SAR-only everywhere | 34 | 0.186 | 0.195 | 0.150 | 0.264 | 0.303 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | 34 | 0.397 | 0.398 | 0.393 | 0.426 | 0.514 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | 34 | 0.335 | 0.335 | 0.343 | 0.371 | 0.419 |
| TerraMind-small (S1F11), corrected | 34 | 0.296 | 0.297 | 0.290 | 0.346 | 0.411 |
| Copernicus GFM (ensemble) | 34 | 0.321 | — | — | 0.427 | 0.517 |

## Per cluster: units covered by GFM (≥ 95 % of scored pixels): every model on the same floods (K = 26)

| Model | K | IoU at τ* | IoU at τ*, binned | IoU, self-calibrated | IoU, oracle | AP |
| --- | --- | --- | --- | --- | --- | --- |
| SAR U-Net (S1F11) | 26 | 0.295 | 0.295 | 0.249 | 0.326 | 0.325 |
| Fusion (S1F11) | 26 | 0.267 | 0.267 | 0.245 | 0.291 | 0.279 |
| TerraMind-small (S1F11) | 26 | 0.229 | 0.235 | 0.239 | 0.311 | 0.361 |
| SAR U-Net (Kuro Siwo, flood) | 26 | 0.411 | 0.412 | 0.409 | 0.449 | 0.537 |
| Fusion (Kuro Siwo, flood labels) | 26 | 0.415 | 0.416 | 0.411 | 0.446 | 0.532 |
| Fusion (Kuro Siwo, water labels) | 26 | 0.351 | 0.351 | 0.360 | 0.386 | 0.428 |
| Physics baseline (row 2) | 26 | 0.412 | — | — | — | — |
| Physics baseline, seasonal reference (row 2s) | 26 | 0.375 | — | — | — | — |
| Fusion (S1F11), SAR-only everywhere | 26 | 0.289 | 0.291 | 0.243 | 0.324 | 0.302 |
| TerraMind (S1F11), SAR-only everywhere | 26 | 0.197 | 0.206 | 0.159 | 0.274 | 0.311 |
| Fusion (Kuro Siwo, flood), SAR-only everywhere | 26 | 0.416 | 0.418 | 0.412 | 0.447 | 0.534 |
| Fusion (Kuro Siwo, water), SAR-only everywhere | 26 | 0.349 | 0.349 | 0.358 | 0.384 | 0.425 |
| TerraMind-small (S1F11), corrected | 26 | 0.308 | 0.309 | 0.310 | 0.360 | 0.425 |
| Copernicus GFM (ensemble) | 26 | 0.361 | — | — | 0.456 | 0.549 |


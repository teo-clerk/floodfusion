# Sen1Floods11 registration — independent checks

After the ingest's per-event shifts (DATA_CARD.md). Offsets are pooled window medians (dy, dx) in px with the approximate 95 % half-interval from resampling chips. *Known shift*: labels displaced by a further (2, -3) px, measured back.

| Event | Labels moved (source) | Optical moved (source) | SWIR vs SAR after | Label vs SWIR after | Split-half shift / odd residual | Train-only shift | Known shift recovered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bolivia | [1, 3] (label) | [1, 3] (label) | (-0.15, -0.75) ±2.24 | (+1.00, -0.35) ±7.85 | [1, 2] / (-0.45, +0.85) | — (0 chips) | (+1.95, -2.90) |
| Ghana | [1, 1] (label) | [1, 1] (label) | (+0.00, +0.45) ±0.73 | (+0.00, +0.05) ±0.20 | [1, 1] / (-0.03, +0.07) | [1, 1] (31 chips) | (+1.95, -2.95) |
| India | [0, 2] (label) | [0, 2] (label) | (+0.25, +0.10) ±0.85 | (+0.00, -0.05) ±0.08 | [1, 3] / (-0.90, -0.73) | [0, 3] (40 chips) | (+1.90, -2.85) |
| Mekong | [0, 0] (none) | [0, 0] (none) | (+0.82, +1.77) ±2.33 | (+0.60, +0.20) ±4.10 | [1, 5] / (-0.10, -0.25) | [1, 5] (18 chips) | (+1.95, -2.80) |
| Nigeria | [1, 1] (label) | [1, 1] (label) | (+0.15, +0.30) ±0.51 | (+0.05, -0.05) ±1.29 | [1, 1] / (+0.90, +0.20) | [1, 1] (10 chips) | (+1.98, -2.42) |
| Pakistan | [0, 1] (optical) | [0, 1] (optical) | (+0.20, +0.95) ±0.85 | (-0.10, -0.05) ±13.00 | [1, 0] / (-2.95, +0.00) | [-2, 0] (16 chips) | (+0.35, -3.00) |
| Somalia | [0, 0] (none) | [0, 0] (none) | (+1.33, +1.10) ±1.07 | (+24.45, -41.85) ±43.00 | [1, 1] / (-0.10, -1.05) | [1, 2] (15 chips) | (+1.83, -2.50) |
| Spain | [0, 2] (label) | [0, 2] (label) | (+0.10, +0.05) ±0.31 | (+0.05, +0.90) ±2.45 | [0, 3] / (+0.10, -0.40) | [0, 3] (18 chips) | (+1.95, -2.70) |
| Sri-Lanka | [0, 2] (label) | [0, 2] (label) | (+0.15, -0.30) ±0.38 | (+0.25, -1.08) ±3.48 | [0, 2] / (-0.03, +0.00) | [0, 3] (24 chips) | (+1.77, -2.50) |
| USA | [2, 1] (label) | [2, 1] (label) | (-0.10, +0.05) ±0.20 | (-0.10, +0.08) ±0.17 | [2, 1] / (+0.05, +0.05) | [2, 1] (41 chips) | (+1.95, -2.60) |

Regenerate: `uv run python scripts/registration_diagnostic.py --data-root <ingested> --raw <Sen1Floods11 download>`.

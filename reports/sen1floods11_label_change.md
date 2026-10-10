# Sen1Floods11 labels and backscatter change (re-ingested stacks)

Flood-labelled valid pixels per event. **8 of 10 events show no clear backscatter drop** (median ΔVV > −1 dB) against the same-orbit reference acquired up to 60 days before the labelled image: much of the labelled water was already there. A change-based rule cannot see it; an amplitude-based one can.

| Event | Flood pixels | Dark before (VV < -15 dB) | Dark after | Median ΔVV (dB) |
| --- | --- | --- | --- | --- |
| Bolivia | 418,905 | 0.41 | 0.55 | -1.40 |
| Ghana | 517,745 | 0.41 | 0.34 | +0.64 |
| India | 1,585,128 | 0.57 | 0.50 | +0.67 |
| Mekong | 1,234,395 | 0.60 | 0.65 | +0.16 |
| Nigeria | 871,773 | 0.52 | 0.54 | -0.29 |
| Pakistan | 332,993 | 0.36 | 0.39 | +0.03 |
| Somalia | 346,801 | 0.16 | 0.31 | -0.95 |
| Spain | 512,764 | 0.14 | 0.58 | -3.40 |
| Sri-Lanka | 395,375 | 0.25 | 0.21 | +0.49 |
| USA | 461,265 | 0.39 | 0.51 | -0.12 |

Regenerate: `uv run python scripts/label_change_diagnostic.py --data-root data/sen1floods11`.

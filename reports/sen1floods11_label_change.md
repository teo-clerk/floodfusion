# Sen1Floods11 labels and backscatter change (re-ingested stacks)

Flood-labelled valid pixels per event. **7 of 10 events show no clear backscatter drop** (median ΔVV > −1 dB) against the same-orbit reference acquired up to 60 days before the labelled image: much of the labelled water was already there. A change-based rule cannot see it; an amplitude-based one can.

| Event | Flood pixels | Dark before (VV < -15 dB) | Dark after | Median ΔVV (dB) |
| --- | --- | --- | --- | --- |
| Bolivia | 420,350 | 0.41 | 0.54 | -1.37 |
| Ghana | 517,745 | 0.41 | 0.34 | +0.64 |
| India | 1,591,967 | 0.56 | 0.49 | +0.67 |
| Mekong | 1,222,705 | 0.61 | 0.66 | +0.17 |
| Nigeria | 871,773 | 0.52 | 0.54 | -0.29 |
| Pakistan | 333,433 | 0.36 | 0.39 | +0.05 |
| Somalia | 345,962 | 0.16 | 0.32 | -1.01 |
| Spain | 512,764 | 0.14 | 0.58 | -3.40 |
| Sri-Lanka | 398,693 | 0.25 | 0.21 | +0.50 |
| USA | 461,265 | 0.39 | 0.51 | -0.12 |

Regenerate: `uv run python scripts/label_change_diagnostic.py --data-root data/sen1floods11`.

# Erratum 2's exposure: stack area past the first DEM tile (v1.1 data)

Written by `scripts/dem_exposure.py` from each stack's `event.json` (metadata only). A stack is affected when part of its grid lies outside the 1° cell of the first GLO-30 tile it lists: before the fix, that part read terrain 0 m.

| Dataset | Stacks affected | Median share of an affected stack | Total, in whole stacks | Share of all stack area |
|---|---|---|---|---|
| Sen1Floods11 | 49 of 379 | 0.68 | ≈ 32 | 8.6 % |
| Kuro Siwo | 52 of 452 | 0.48 | ≈ 28 | 6.4 % |
| CEMS benchmark | 172 of 605 | 0.48 | ≈ 86 | 14.6 % |

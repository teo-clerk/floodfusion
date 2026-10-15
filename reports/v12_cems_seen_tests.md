# P5-P7 on the seen CEMS 2023-26 floods (descriptive, not a test)

The frozen v1.2 models were scored once on the CEMS 2023-26 benchmark rebuilt with the DEM fix. They are compared here
with the blind test's own analysis: `floodfusion.blind_analysis`, unchanged, with v1.1's units (primary stratum,
EMSR664 excluded).

**These floods have been seen**, and training sets now cover them too, so these are **not** the pre-registered tests.
Those run on floods after the freeze (`reports/protocol_v1_2.md`). What this table shows is how large the gaps are on
independent maps. The blind test's stopping rule detects about 0.06 IoU at 80 % power; gaps much smaller than that
will most likely end "not supported" there.

| Test | B − A | Unit | K | Mean gap (IoU at τ*) | B > A in | AP gap | Holm p | Would read |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P5 | v1.2 (row21_fusion_context_seasonal) − TerraMind, fair (terramind_base_fair) | activation | 38 | +0.012 | 24/38 | +0.015 | 0.3663 | not supported |
| P5 | v1.2 (row21_fusion_context_seasonal) − TerraMind, fair (terramind_base_fair) | cluster | 30 | +0.012 | 19/30 | +0.016 | 0.3293 |  |
| P6 | v1.2 (row21_fusion_context_seasonal) − Copernicus GFM (ensemble) | activation | 34 | +0.098 | 25/34 | — | 0.0010 | supported |
| P6 | v1.2 (row21_fusion_context_seasonal) − Copernicus GFM (ensemble) | cluster | 26 | +0.081 | 17/26 | — | 0.0185 |  |
| P7 | v1.2 (row21_fusion_context_seasonal) − v1.1 best (fusion, Kuro Siwo flood labels) | activation | 38 | +0.010 | 22/38 | — | 0.3663 | not supported |
| P7 | v1.2 (row21_fusion_context_seasonal) − v1.1 best (fusion, Kuro Siwo flood labels) | cluster | 30 | +0.010 | 18/30 | — | 0.3293 |  |

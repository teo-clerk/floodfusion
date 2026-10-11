# Stage A — diagnostics of the v1.0 models (no retraining)

Every number below re-reads the released v1.0 checkpoints on the Sen1Floods11 test chips of the fold that held each event out, scored as `floodfusion.evaluate` scores them (64 px context, each run's own calibration, τ* = 0.25). Regenerate: `scripts/stage_a.py predict`, `clouds`, `report` (release v1.0 code; seasonal references from `python -m floodfusion.data.seasonal` of v1.1).

## A1 — How much optical changes the fused features, by optical quality (row 9)

Contribution = ‖fused(optical) − fused(missing embedding)‖ / ‖F_s‖ per fusion-stage token (8 × 8 px), two passes on the same chip (chips with optical only); the gate is shown with optical and with the missing embedding. Gate levels compare only within a model (with vs without optical); across seeds they are not comparable.

| Seed | Optical quality | Label | Contribution | Gate with optical | Gate, missing embedding | Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| seed42 | q<0.25 | water | 1.239 | 0.256 | 0.179 | 17601 |
| seed42 | q<0.25 | land | 1.608 | 0.418 | 0.189 | 120418 |
| seed42 | 0.25<=q<0.75 | water | 1.259 | 0.249 | 0.184 | 3405 |
| seed42 | 0.25<=q<0.75 | land | 1.657 | 0.425 | 0.189 | 23316 |
| seed42 | q>=0.75 | water | 1.185 | 0.248 | 0.186 | 18531 |
| seed42 | q>=0.75 | land | 1.576 | 0.429 | 0.193 | 364569 |
| seed43 | q<0.25 | water | 0.969 | 0.213 | 0.168 | 17601 |
| seed43 | q<0.25 | land | 1.913 | 0.439 | 0.172 | 120418 |
| seed43 | 0.25<=q<0.75 | water | 1.004 | 0.212 | 0.167 | 3405 |
| seed43 | 0.25<=q<0.75 | land | 2.084 | 0.461 | 0.171 | 23316 |
| seed43 | q>=0.75 | water | 1.023 | 0.227 | 0.169 | 18531 |
| seed43 | q>=0.75 | land | 2.019 | 0.478 | 0.174 | 364569 |
| seed44 | q<0.25 | water | 1.710 | 0.492 | 0.138 | 17601 |
| seed44 | q<0.25 | land | 0.832 | 0.250 | 0.157 | 120418 |
| seed44 | 0.25<=q<0.75 | water | 1.789 | 0.515 | 0.140 | 3405 |
| seed44 | 0.25<=q<0.75 | land | 0.738 | 0.229 | 0.157 | 23316 |
| seed44 | q>=0.75 | water | 1.665 | 0.490 | 0.142 | 18531 |
| seed44 | q>=0.75 | land | 0.703 | 0.217 | 0.161 | 364569 |

## A2 — Real cloud shapes pasted onto clear chips (row 9, three seeds pooled)

Donor clouds: real cloud masks from training chips, with their own cloudy reflectance ("real") or with optical zeroed ("missing"); quality 0 under the mask in both. At 100 % the model takes the missing-modality path and its SAR-only calibration (row 11).

| Cover | Variant | IoU on masked pixels | IoU on clear pixels |
| --- | --- | --- | --- |
| 0% | none | — | 0.486 |
| 20% | missing | 0.026 | 0.460 |
| 20% | real | 0.033 | 0.478 |
| 40% | missing | 0.023 | 0.452 |
| 40% | real | 0.024 | 0.461 |
| 60% | missing | 0.030 | 0.336 |
| 60% | real | 0.031 | 0.429 |
| 80% | missing | 0.031 | 0.290 |
| 80% | real | 0.026 | 0.342 |
| 100% | missing | 0.207 | — |
| 100% | real | 0.207 | — |

Clear-pixel error rate against distance to the nearest cloud pixel:

| Cover / variant | 0-16 px | 16-32 | 32-64 | 64-128 | > 128 |
| --- | --- | --- | --- | --- | --- |
| 0.2/missing | 0.010 | 0.005 | 0.005 | 0.006 | 0.006 |
| 0.2/real | 0.006 | 0.005 | 0.005 | 0.006 | 0.006 |
| 0.4/missing | 0.008 | 0.006 | 0.007 | 0.006 | 0.008 |
| 0.4/real | 0.007 | 0.006 | 0.007 | 0.006 | 0.008 |
| 0.6/missing | 0.012 | 0.005 | 0.006 | 0.004 | 0.000 |
| 0.6/real | 0.009 | 0.006 | 0.007 | 0.006 | 0.006 |
| 0.8/missing | 0.014 | 0.006 | 0.005 | 0.005 | — |
| 0.8/real | 0.013 | 0.007 | 0.006 | 0.008 | — |

## A3 — Was the labelled flood there before?

Share of each event's flood-labelled pixels by darkness (VV < -15 dB) at the **seasonal** reference (same-orbit median within ±30 days of the same date a year earlier) and at the **recent** reference (the latest pass 5-24 days earlier, v1.0's).

| Event | Flood px | Dark a year earlier | Dry then, dark at the recent reference | Dry at both | No seasonal reference |
| --- | --- | --- | --- | --- | --- |
| Bolivia | 219,429 | 0.05 | 0.43 | 0.53 | 0.00 |
| Ghana | 275,891 | 0.04 | 0.40 | 0.56 | 0.00 |
| India | 598,991 | 0.00 | 0.00 | 0.00 | 1.00 |
| Mekong | 522,490 | 0.31 | 0.34 | 0.35 | 0.00 |
| Nigeria | 377,285 | 0.24 | 0.26 | 0.50 | 0.00 |
| Pakistan | 167,502 | 0.00 | 0.00 | 0.00 | 1.00 |
| Somalia | 126,545 | 0.00 | 0.00 | 0.00 | 1.00 |
| Spain | 309,599 | 0.08 | 0.06 | 0.85 | 0.00 |
| Sri-Lanka | 125,057 | 0.00 | 0.00 | 0.00 | 1.00 |
| USA | 293,303 | 0.23 | 0.17 | 0.59 | 0.00 |

Where each model's errors and hits fall (all events pooled; counts, since IoU does not add up across strata). A gain on pixels radar cannot see as open water fits label-source bias *and* real optical value: it bounds the bias from above, it does not measure it.

| Model | Stratum | TP | FP | FN |
| --- | --- | --- | --- | --- |
| cmp_terramind_small/seed42 | new water (dark after, not before) | 395,683 | 53,938 | 48,520 |
| cmp_terramind_small/seed42 | already dark at the recent reference | 1,218,507 | 157,274 | 118,078 |
| cmp_terramind_small/seed42 | not dark after (radar cannot see it as open water) | 857,905 | 584,242 | 638,412 |
| row04_unet_sar_only/seed42 | new water (dark after, not before) | 323,684 | 204,441 | 120,519 |
| row04_unet_sar_only/seed42 | already dark at the recent reference | 1,089,790 | 514,479 | 246,795 |
| row04_unet_sar_only/seed42 | not dark after (radar cannot see it as open water) | 650,754 | 1,382,683 | 845,563 |
| row09_fusion_crossattn/seed42 | new water (dark after, not before) | 410,756 | 52,061 | 33,447 |
| row09_fusion_crossattn/seed42 | already dark at the recent reference | 1,229,908 | 160,132 | 106,677 |
| row09_fusion_crossattn/seed42 | not dark after (radar cannot see it as open water) | 942,238 | 538,727 | 554,079 |
| row09_fusion_crossattn/seed43 | new water (dark after, not before) | 414,008 | 54,222 | 30,195 |
| row09_fusion_crossattn/seed43 | already dark at the recent reference | 1,269,846 | 166,181 | 66,739 |
| row09_fusion_crossattn/seed43 | not dark after (radar cannot see it as open water) | 959,953 | 500,459 | 536,364 |
| row09_fusion_crossattn/seed44 | new water (dark after, not before) | 399,177 | 42,969 | 45,026 |
| row09_fusion_crossattn/seed44 | already dark at the recent reference | 1,171,697 | 153,404 | 164,888 |
| row09_fusion_crossattn/seed44 | not dark after (radar cannot see it as open water) | 887,408 | 443,126 | 608,909 |
| row11_fusion_eval_sar_only/seed42 | new water (dark after, not before) | 318,860 | 205,259 | 125,343 |
| row11_fusion_eval_sar_only/seed42 | already dark at the recent reference | 1,046,497 | 500,083 | 290,088 |
| row11_fusion_eval_sar_only/seed42 | not dark after (radar cannot see it as open water) | 710,142 | 1,579,011 | 786,175 |
| row11_fusion_eval_sar_only/seed43 | new water (dark after, not before) | 389,062 | 187,435 | 55,141 |
| row11_fusion_eval_sar_only/seed43 | already dark at the recent reference | 1,113,960 | 450,033 | 222,625 |
| row11_fusion_eval_sar_only/seed43 | not dark after (radar cannot see it as open water) | 710,165 | 1,399,936 | 786,152 |
| row11_fusion_eval_sar_only/seed44 | new water (dark after, not before) | 374,649 | 191,824 | 69,554 |
| row11_fusion_eval_sar_only/seed44 | already dark at the recent reference | 1,035,373 | 497,011 | 301,212 |
| row11_fusion_eval_sar_only/seed44 | not dark after (radar cannot see it as open water) | 662,705 | 1,296,016 | 833,612 |

## A4 — Registration: does the headline survive without the label shift?

Macro IoU over events at τ*, as published, with label edges (±2 px, the documented uncertainty) ignored, and with the registration shift undone. The shift moved labels toward SAR water edges, so it can only have helped the SAR-only models.

| Model | As published | Edges ±2 px ignored | Labels unshifted |
| --- | --- | --- | --- |
| cmp_terramind_small/seed42 | 0.576 | 0.678 | 0.568 |
| row04_unet_sar_only/seed42 | 0.365 | 0.378 | 0.357 |
| row09_fusion_crossattn/seed42 | 0.628 | 0.727 | 0.608 |
| row09_fusion_crossattn/seed43 | 0.641 | 0.745 | 0.621 |
| row09_fusion_crossattn/seed44 | 0.604 | 0.710 | 0.588 |
| row11_fusion_eval_sar_only/seed42 | 0.360 | 0.366 | 0.351 |
| row11_fusion_eval_sar_only/seed43 | 0.408 | 0.432 | 0.397 |
| row11_fusion_eval_sar_only/seed44 | 0.381 | 0.410 | 0.371 |

## A5 — Conformal recall, measured as it is defined, and cross-fitted

Conformal risk control bounds the **expected** per-chip miss rate; it promises nothing per event. Cross-fitted: each held-out event's threshold is fitted on the chips of the other nine held-out events (each scored by its own fold model), on a grid down to 0.001. With nine calibration *events*, an event-level bound at α = 0.1 would need zero risk (vacuous).

| Model | Events (feasible) | Events with mean chip FNR ≤ 0.1 | Mean chip FNR | Precision | IoU |
| --- | --- | --- | --- | --- | --- |
| cmp_terramind_small/seed42 | 10/10 | 7/10 | 0.074 | 0.246 | 0.245 |
| row04_unet_sar_only/seed42 | 10/10 | 6/10 | 0.094 | 0.275 | 0.268 |
| row09_fusion_crossattn/seed42 | 10/10 | 5/10 | 0.123 | 0.347 | 0.334 |
| row09_fusion_crossattn/seed43 | 10/10 | 4/10 | 0.123 | 0.387 | 0.367 |
| row09_fusion_crossattn/seed44 | 10/10 | 6/10 | 0.115 | 0.354 | 0.341 |
| row11_fusion_eval_sar_only/seed42 | 10/10 | 6/10 | 0.084 | 0.240 | 0.237 |
| row11_fusion_eval_sar_only/seed43 | 10/10 | 8/10 | 0.059 | 0.222 | 0.222 |
| row11_fusion_eval_sar_only/seed44 | 10/10 | 5/10 | 0.087 | 0.243 | 0.240 |


# floodfusion v1.1 — erratum: the TerraMind row was trained without modality dropout

Found on 2026-10-13 while planning v1.2, after every v1.1 result had been reported (`v1.1.0`, `01666cb`).

## What was wrong

1. **No modality dropout for TerraMind.** `configs/comparisons/terramind_small.yaml` sets `modality_dropout_p: 0.5`
   ("as row 9: SAR-only inference must work"), and the protocol and the wrapper's docstring say dropout applies
   unchanged. But the training loop applies dropout only to architectures in `train.OPTICAL_FUSION_ARCHITECTURES`,
   and that set listed the three fusion rows, not `terramind_*`. TerraMind therefore trained with p = 0.
   - **Evidence:** `dropped_fraction` is 0.0 in every epoch of
     `runs/v11/sen1floods11/official/official/cmp_terramind_small/seed42/metrics.jsonl`, against 0.45–0.51 for row 9
     on the same split (30 epochs).
   - **Consequence:** TerraMind's SAR-only path (S1RTC alone) was trained only on the few chips that had no usable
     optical. On the CEMS benchmark most scored pixels have no clear optical, so TerraMind ran mostly in a mode it had
     barely been trained in.
2. **BatchNorm on sub-batches.** On a batch mixing chips with and without optical, the wrapper ran the whole
   TerraTorch model twice, once per subset. Neck and decoder BatchNorm layers then trained on sub-batches of one to
   three chips. In evaluation mode this makes no difference to the logits.

## Which results it affects

- **P4 / H4 (fusion vs TerraMind on the CEMS benchmark).** Reported as supported (+0.031 per activation, +0.039 per
  cluster). Both defects handicap TerraMind, the first mostly where optical is missing, so the reported margin may be
  partly an artefact of the bug. The verdict stands as pre-registered and as run; it must be read with this erratum.
- **The Sen1Floods11 TerraMind rows** (leave-one-event-out 0.572, published split 0.576) and every descriptive
  TerraMind number in `reports/cems_benchmark*.md`, the README, the model card and the site.
- **No other model is affected.** Rows 4, 9 and 11, the Kuro Siwo models and the baselines never went through either
  code path. Their logits are pinned by `tests/test_golden_v11.py`.

## The fix (v1.2 branch)

- `train.OPTICAL_FUSION_ARCHITECTURES` includes the TerraMind architectures (test:
  `test_terramind_gets_the_modality_dropout_its_config_sets`).
- The wrapper runs only the encoder per optical subset, puts the features back in batch order, and runs neck,
  decoder and head once on the whole batch (test: `test_neck_and_decoder_see_the_whole_batch_once`). The v1.1
  TerraMind checkpoint gives the same logits as before (`tests/test_golden_v11.py`).

## The corrected re-run (pending)

TerraMind is re-trained with the fix on the same Sen1Floods11 splits, seeds and schedule (published split × 3 seeds,
leave-one-event-out), and P4 is re-scored on the CEMS benchmark. This is the protocol's model as the protocol
describes it, but it is run **after** the v1.1 results were known. It is reported beside the original, as a
corrected, post-hoc result; it does not replace the pre-registered verdict.

| | As run (v1.1) | Corrected re-run |
|---|---|---|
| P4 per activation | +0.031, 24/38, Holm p = 0.034 | pending |
| P4 per cluster | +0.039, 20/30, p = 0.009 | pending |
| TerraMind, Sen1Floods11 leave-one-event-out | 0.572 | pending |

---

# Erratum 2: zero terrain past a DEM tile edge (found 2026-10-13, while building v1.2's DEM sidecar)

## What was wrong

Copernicus GLO-30 COGs declare **no nodata value**. When a stack's grid needs two DEM tiles, the warp filled every grid
pixel outside the first tile with 0. The first-valid mosaic (`sources._mosaic`) kept that 0 as a valid height, so the
second tile never replaced it. Past a 1° tile boundary the DEM was therefore flat at 0 m, and everything derived from
it was wrong there:

- **slope:** 0;
- **HAND:** 0, which reads as "at drainage level";
- **radar shadow and layover:** none flagged;
- **local incidence:** reduced to the ellipsoid angle.

Radar and optical are not affected: their files declare nodata (S1 RTC −32768; S2, SCL and WorldCover 0; JRC 255).

**Evidence:** `KS_EMSR174_Skopje_01_527360_4648960`.
- Below row 424 the stored slope and HAND are exactly 0. Above it they are 1.9° and 4.2 m (medians).
- Its DEM tiles are N42 and N41.
- `tests/test_dem_mosaic.py` reproduces the zeros on the old code.

## Exposure

Stacks with part of their grid outside the first DEM tile they list:

| Dataset | Stacks affected | Median share of the stack | Total, in whole stacks |
|---|---|---|---|
| Sen1Floods11 | 49 of 379 | 0.68 | ≈ 32 |
| Kuro Siwo | 52 of 452 | 0.48 | ≈ 29 |
| CEMS benchmark | 168 of 605 | 0.50 | ≈ 86 (≈ 14 % of its stack area) |

## Which results it can affect

Every v1.0 and v1.1 number from models or rules that read terrain:
- rows 4, 9 and 11;
- the physics baselines, through their HAND and slope masks;
- the shadow/layover exclusions.

TerraMind reads no terrain, so the defect could bias the H4 comparison in either direction. The direction and size
are measured, not assumed: the v1.1 models are re-scored on CEMS stacks rebuilt with the fix, as a descriptive
comparison beside the published numbers, which stand as computed.

## The fix and the rebuilt data (v1.2)

- **The fix:** `_mosaic` takes the warp's destination nodata, and `read_dem` passes NaN.
- **The rebuild:** Kuro Siwo and the CEMS benchmark are re-ingested with the fix into `data/kurosiwo_v12` and
  `data/cems_benchmark_v12`. The ingests are seeded: an unaffected stack comes out pixel-identical (checked on
  EMSR174). v1.2 trains and scores on the rebuilt data.
- **The v1.1 data stays as it was.** The v1.1 regression check and the TerraMind re-run of erratum 1 read it, so their
  numbers remain comparable with v1.1's.
- **GEOID-Flood** is ingested with the fix from the start.

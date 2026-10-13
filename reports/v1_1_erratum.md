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

## The corrected re-run

TerraMind is re-trained with the fix on the same Sen1Floods11 splits, seeds and schedule (published split × 3 seeds,
leave-one-event-out), and P4 is re-scored on the CEMS benchmark as P4c (`reports/v1_1_erratum_p4.md`, run on
2026-10-13). This is the protocol's model as the protocol describes it, but it is run **after** the v1.1 results were
known. It is reported beside the original, as a corrected, post-hoc result; it does not replace the pre-registered
verdict.

| | As run (v1.1) | Corrected re-run |
|---|---|---|
| P4, fusion − TerraMind, per activation | +0.031, fusion better in 24/38, Holm p = 0.034 | **−0.042** [−0.065, −0.018], fusion better in 10/38, p = 0.001 |
| P4, per cluster | +0.039, 20/30, p = 0.009 | **−0.043** [−0.070, −0.018], 8/30, p = 0.003 |
| TerraMind-small on CEMS, macro IoU at τ* (oracle τ) | 0.226 (0.270) | 0.299 [0.249, 0.349] (0.328) |
| Fusion (S1F11) on CEMS, for reference | 0.258 (0.260) | unchanged |
| TerraMind, Sen1Floods11 leave-one-event-out | 0.572 | 0.597, better on 8 of 10 events |
| TerraMind, Sen1Floods11 leave-one-event-out, SAR-only | 0.413 | 0.509, better on 10 of 10 events |
| TerraMind, Sen1Floods11 published split, 3 seeds (SAR-only) | 0.594 (0.451) | 0.619 (0.547) |

The published-split row is v1.1's three-seed run; the 0.576 quoted in v1.0's table is a single seed.

**What it means.** With TerraMind trained as the protocol describes it, **H4 reverses**: the foundation model beats
the task-specific fusion model trained on the same Sen1Floods11 data, significantly at both unit definitions. P4c is a
single post-hoc test, so its p-values are unadjusted. In v1.1's Holm families it would still be significant: Holm
p = 0.002 per activation (with P1 and P3), 0.003 per cluster (with P3). The reversal is largest where optical is
missing:
- on the partial-cloud subset (descriptive), fusion is better on only 2 of 15 activations, by −0.064 on average;
- on Sen1Floods11, the fix raises TerraMind's SAR-only score by about 0.10 IoU, against 0.025 with optical.

That is where the missing dropout had left TerraMind untrained.

**What stands, and what is withdrawn.**
- **Stands:** the pre-registered P4 verdict, as run (supported).
- **Withdrawn:** v1.1's conclusion that the task-specific model beats the foundation model on independent maps. It
  rested on a TerraMind that had not been trained as described.
- **Stands:** the other v1.1 conclusions (H1 labels; H2 no verdict; H3 not shown). None of them involves TerraMind.

**One caveat runs the other way.** P4c is scored on the v1.1 CEMS stacks, which carry erratum 2 below. The fusion
model reads terrain and TerraMind does not, so on about 14 % of the benchmark's area the fusion model saw corrupted
input. Re-scoring both models on the rebuilt benchmark (`data/cems_benchmark_v12`) measures that effect. It is
reported beside P4c, descriptively.

v1.2 takes the question further: a TerraMind trained on the same flood labels as our model (Kuro Siwo and
GEOID-Flood), with its native DEM and a pre-event image, against our best model, on floods that occur after both are
frozen (`reports/protocol_v1_2.md`, H5).

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

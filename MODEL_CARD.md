# Model card — floodfusion cross-attention SAR–optical flood segmenter

## Model details

- **What it is.** A per-pixel binary flood segmenter for Sentinel-1 scenes, with optional
  Sentinel-2. Two GroupNorm convolutional encoders (SAR + terrain; optical). Cross-attention takes
  queries from optical and keys/values from SAR. A per-pixel quality gate and a learned
  missing-modality embedding are fused as a gated residual on the SAR stream:
  `F = F_s + g ⊙ Attn(Q_o, K_s, V_s)`, with `g ∈ [0, 1]`. With the gate closed, the output is the
  SAR-only path, bit for bit. A U-Net decoder outputs one logit per pixel.
- **Size.** About 19 M parameters; 6 ms per 256 × 256 chip on an RTX 4060 Laptop GPU.
- **Inputs (10 m, one UTM grid).**
  - Post- and pre-event Sentinel-1 RTC γ⁰ in dB (VV, VH) and the VV change.
  - Sentinel-2 L2A B3, B8, B11, MNDWI and a per-pixel quality from the scene classification (SCL).
  - Slope, log(1 + HAND) and local incidence angle from Copernicus DEM GLO-30.
- **Output.** A calibrated flood probability; a decision at τ* = 1/(1 + r) for a stated
  false-negative to false-positive cost ratio r (default 3, so τ* = 0.25); and an optional
  conformal threshold that bounds the expected per-chip miss rate.
- **Training.**
  - Focal + soft-Dice loss on a balanced chip sampler; AdamW, warmup + cosine schedule, bf16.
  - Per-sample modality dropout (p = 0.5), so the SAR-only case is trained, not hoped for.
  - Calibration (prior correction, temperature, Platt when it helps) is fitted on validation only.
- **Training data.**
  - Sen1Floods11: water labels drawn mostly on Sentinel-2.
  - Kuro Siwo (v1.1): flood labels drawn on Sentinel-1, and the same chips relabelled as water.
  - The two are separate models; no model mixes the datasets.
- **Code and reproduction.** `src/floodfusion`; every reported number regenerates from a git SHA
  (`make demo`, `make s1f11`, `make forli`; v1.1: `scripts/v11_queue.sh all`). Licence: MIT.

## v1.2's model (frozen for the blind test)

- **What it adds: a seasonal radar reference (row 21).** Beside the post- and pre-event images, the model reads the
  median Sentinel-1 backscatter of the same orbit around the same date a year earlier (VV, VH), and the change from
  it.
  - **The question it lets the model ask:** "is this water unusual for this place and season?", not only "is this
    pixel water?".
  - **How it enters:** the three channels and an availability indicator join the SAR + terrain encoder input.
  - **Where there is no reference** (fewer than three passes a year earlier), the channels are a learned constant and
    the indicator is 0.
  - **Training:** context dropout (p = 0.3) trains that case.
- **Training data: one pool of flood labels.**
  - **Sources:** Kuro Siwo (rebuilt with the DEM fix), plus the GEOID-Flood events before EMSR656 that Kuro Siwo
    lacks. In all, 7,006 chips from 138 floods, in 81 groups of floods within 100 km.
  - **The frozen model:** trained with seeds 42–44 on the all-groups split (63 groups train, 17 validate early
    stopping and calibration), at learning rate 3e-4.
- **How it was chosen.** By rules fixed before the runs (`reports/v1_2_deviations.md`).
  - **CV:** in grouped five-fold CV it beat the same model without context on whole test scenes: 0.380 vs 0.361 macro
    IoU at τ*, better in 49 of 79 groups. Exploratory and single-seed.
  - **No add-on:** neither per-scene self-calibration nor a HAND cut helped, so neither is used.
- **Its comparison in CV** (chip level, macro over 81 groups; `reports/v12_pool_cv.md`):

  | Model | IoU at τ* | IoU at oracle τ | AP |
  | --- | --- | --- | --- |
  | v1.2, row 21 | 0.394 | 0.388 | 0.651 |
  | TerraMind-base + LoRA, trained the same way | 0.313 | 0.366 | 0.617 |
- **Its confirmatory test is still to come.** Copernicus flood activations from EMSR899 on are scored by the frozen
  models and sealed until 20 activations in 15 clusters are scored, or 31 December 2027
  (`reports/protocol_v1_2.md`). The checkpoints are pinned by sha256 in `reports/v12_freeze.json`.

## Intended use

- A **first-guess flood extent** for a rapid-mapping analyst after a flood, from the first Sentinel-1
  acquisition, with optical used only where it is clear.
- Large-area screening where an analyst then reviews flagged and low-confidence areas.

## Out of scope

- **Water under closed canopy and between buildings.** C-band does not see it. The product flags
  dense forest and built-up land as *not reliably mapped*; "dry" there means "not seen".
- **Radar shadow and layover** carry no information and are excluded.
- **Depth, damage or impact.** Not estimated.
- **"Now".** The map is a snapshot at the acquisition time.
- **Wind-roughened open water** can be missed (Bragg scattering at ~4 cm brightens it).
- **Any decision without the exclusion mask and the stated cost ratio.**

## Evaluation data and protocol

- **CEMS benchmark, v1.1** (`reports/cems_benchmark.md`, `reports/cems_benchmark_strata.md`).
  - **The confirmatory test set.** Copernicus EMS delineations of 38 flood activations, 2023-03 to 2026-03, drawn on
    non-Sentinel radar satellites.
  - **Pairing:** each delineation is paired with a Sentinel-1 pass within 24 h, selected by a rule committed before
    any v1.1 model was trained (`reports/protocol_v1_1.md`).
  - **Tests:** four pre-registered tests. A result counts only if it holds per activation and per cluster of
    neighbouring activations.
- **Kuro Siwo, v1.1** (`reports/v11_kurosiwo_table.md`, `reports/v11_kurosiwo_water_table.md`): 36 floods in 29
  groups, with labels drawn on radar. Grouped five-fold: each group is tested once by a model that never saw it.
- **Sen1Floods11, re-ingested** (`reports/sen1floods11_table.md`; v1.1 re-run: `reports/v11_sen1floods11_table.md`).
  - 10 flood events on 5 continents, labels from the dataset, every input re-fetched.
  - Leave-one-event-out: each event is tested once by a model that never saw it.
  - Verdicts are exact sign-flip permutation tests over events, Holm-adjusted.
  - The published within-event split is scored on the same chips to measure the protocol gap.
- **EMSR664 Forlì** (`reports/emsr664_forli.md`).
  - Zero-shot against the four Copernicus EMS delineations of 17–21 May 2023 (COSMO-SkyMed,
    RADARSAT-2, PAZ).
  - Scored on the region every Sentinel-1 pass and every product observed.
- **Synthetic benchmark** (`reports/ablation_table.md`, `reports/stress.md`).
  - The 13-row ablation, 3 seeds per row.
  - Single-factor stress curves on unseen landscapes.

## Performance

All numbers come from the release runs; each report records its git SHA.

- **CEMS benchmark (v1.1, 38 activations, zero-shot, IoU at each model's own τ\*).**

  | Model | Trained on Sen1Floods11 | Trained on Kuro Siwo |
  | --- | --- | --- |
  | Fusion | 0.258 | 0.397 |
  | SAR-only U-Net | 0.281 | 0.394 |
  | TerraMind-small, as run in v1.1 | 0.226 | — |
  | TerraMind-small, trained as described (erratum) | 0.299 | — |

  The physics baseline scores 0.371.

  **The pre-registered tests:**
  - Flood-label recipe beats water-label recipe on the same Kuro Siwo chips: +0.06, supported. Part of this is
    agreement of definitions, since CEMS maps are flood maps.
  - Fusion beats TerraMind at each model's own threshold: +0.03, supported as run. **Withdrawn by the erratum**
    (`reports/v1_1_erratum.md`): TerraMind had been trained without modality dropout. Trained as described, it beats
    fusion by 0.042 per activation (fusion better in 10/38) and 0.043 per cluster, significant at both.
  - Fusion beats the SAR U-Net where optical is usable: +0.04, not significant on 8 activations, and too few clusters
    for a verdict. Result: no verdict.
  - Fusion is non-inferior without optical: −0.02, which holds per activation, not per cluster. Not supported.
- **Kuro Siwo, grouped five-fold (v1.1, 29 groups).** Fusion 0.535, SAR U-Net 0.505, the same fusion model without
  optical 0.533, physics baseline 0.417. Fusion vs SAR U-Net is +0.030 and not significant.
- **Sen1Floods11, leave-one-event-out (10 events), IoU at τ\* = 0.25.** The v1.1 re-run replicates the finding:
  fusion 0.608, SAR U-Net 0.343, +0.265 on 10 of 10 events.
  - Fusion (3 seeds): **0.625**, AP 0.769.
  - The same model without optical: 0.383.
  - SAR-only U-Net: 0.365.
  - Physics baseline + MNDWI: 0.444.
  - TerraMind-1.0-small, frozen: 0.576 (v1.0, one seed); 0.597 trained as described (erratum, 3 seeds).
  - Fusion beats SAR-only on 10 of 10 events (Holm p = 0.018). Without optical, it equals SAR-only.
  - At the oracle threshold, fusion scores 0.668: some of the gap to τ\* is threshold transfer between events.
- **Per event**, fusion ranges from 0.443 (Pakistan) to 0.852 (Mekong); see
  `reports/sen1floods11_table.md`.
- **Conformal recall (α = 0.1).** The mean per-chip miss rate is 0.08–0.11, but the target holds on only
  5–8 of 10 held-out events, and precision at the conformal threshold is about 0.25.
- **EMSR664 Forlì, zero-shot against Copernicus EMS.** Scored on the 1101 km² common region against the nearest
  delineation:
  - 23 May (optical on 9/16 tiles): fusion **0.507**; the same model without optical, 0.230.
  - 22 May (no optical): fusion 0.390, below the physics baseline's 0.577.
  - Zero-shot, the learned SAR path over-maps fields in a new region.
- **Synthetic benchmark (3 seeds).** Fusion 0.676, the same as SAR-only. Graceful degradation costs 0.002.
  This is pipeline validation, not an accuracy claim.

## Factors and known failure modes

- **Optical's measured value may be partly agreement with the annotators.**
  - Sen1Floods11 labels were drawn mostly on the Sentinel-2 image the model also sees, where fusion gains +0.26.
  - On the independent, radar-drawn CEMS maps where optical is usable, the gain is +0.04 [−0.02, +0.11] on 8
    activations. That excludes +0.26 but does not decide a smaller gain. Region, years and reference type differ too,
    so the drop is consistent with annotator agreement, not proof of it.
  - Trained on Kuro Siwo's radar-drawn labels, the model uses almost nothing from optical at inference (0.397 with
    it, 0.398 without).
- **Training data moved scores more than the architecture (descriptive).** On CEMS, where only the flood-vs-water
  comparison holds the chips and the architecture fixed:
  - the same fusion network scores 0.397 trained on Kuro Siwo flood labels, 0.338 on water labels of the same chips,
    and 0.258 on Sen1Floods11;
  - Sen1Floods11-trained models score below the physics baseline (0.371) in a new region.
- **Change versus amplitude.** On labels that mark water on a date rather than a change, rules
  built on backscatter change are penalised. In 8 of 10 Sen1Floods11 events the labelled flood
  shows no backscatter drop against a same-orbit reference up to 60 days earlier.
- **Threshold transfer.** τ* and the prior come from the training data's validation set; a scene
  with a very different flood prevalence shifts the best threshold. The reports show IoU at τ*
  beside the oracle-threshold IoU and AP.
- **The fusion gate barely trained (synthetic benchmark).** The per-pixel gate starts at 0.12 and ends
  at 0.12–0.17 over clear pixels, slightly higher under the cloud mask; only 2–6 % of clear pixels open
  past 0.5 (`reports/gate_diagnostic.md`). The network takes almost nothing from optical there (clear sky
  adds 0.004 IoU). What holds by construction is the SAR-only path when optical is missing.
- **Partial cloud (real data).** The SAR-only fallback holds for a whole chip without optical, not per pixel.
  With real cloud shapes pasted onto clear Sen1Floods11 chips, the flooded pixels under the cloud are almost
  all missed (IoU 0.02–0.03, against 0.21 when the same chip is run SAR-only), while clear pixels keep their
  accuracy even next to the cloud edge. On real data the gate opens with optical (≈ 0.21–0.52 against
  0.14–0.19 with the missing embedding) but barely responds to the per-pixel quality
  (`reports/stage_a_diagnostics.md`, A1 and A2).
  - **Consistent with independent maps (descriptive).** On CEMS activations 10–50 % clear, the same fusion model
    scores 0.254 with optical and 0.307 run without it, and 0.06 below the SAR U-Net, better on 2 of 15. This subset
    was added to the protocol after Stage A had found the failure.
  - **Variant 9c (synthetic clouds in training)** has a leave-one-event-out gap of −0.004 [−0.026, +0.023]: one
    seed, no verdict. Whether it fixes the per-pixel failure was not tested.
  - Until it is fixed, mask partly cloudy optical out entirely, or run the SAR-only mode, where clouds cover
    flood-prone ground.
- **Geometry.**
  - Sentinel-2 L2A of 2016–2019 sits 1–3 px from the radar geometry, and Sen1Floods11 labels follow it.
    Labels and optical are registered to the SAR per event, using the labels' water edges where the
    label-free estimate cannot measure (DATA_CARD.md states what that can bias).
  - Where the GRD annotation is missing, incidence comes from an orbit model, with ≤ 0.74° error.

## Ethical considerations

A flood map can direct help. An error of omission costs more than one of commission, which is why
the cost ratio is explicit, the threshold follows from it, and the low-confidence band and the
exclusion mask ship with every product. The map must not be used to deny assistance to an area
marked "dry" under canopy, in a city, in radar shadow, or outside the acquisition time.

## Data and attribution

See [`DATA_CARD.md`](DATA_CARD.md). Contains modified Copernicus Sentinel data. CEMS products:
© European Union, Copernicus Emergency Management Service. Kuro Siwo annotations (training and cross-validation of
the v1.1 Kuro Siwo models): Bountos et al., NeurIPS 2024, CC BY 4.0. Sen1Floods11 labels: Bonafilia et al., 2020, no
licence declared, never redistributed.

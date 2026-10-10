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
- **Code and reproduction.** `src/floodfusion`; every reported number regenerates from a git SHA
  (`make demo`, `make s1f11`, `make forli`). Licence: MIT.

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

- **Sen1Floods11, re-ingested** (`reports/sen1floods11_table.md`).
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

- **Sen1Floods11, leave-one-event-out (10 events), IoU at τ\* = 0.25.**
  - Fusion (3 seeds): **0.625**, AP 0.769.
  - The same model without optical: 0.383.
  - SAR-only U-Net: 0.365.
  - Physics baseline + MNDWI: 0.444.
  - TerraMind-1.0-small, frozen: 0.576.
  - Fusion beats SAR-only on 10 of 10 events (Holm p = 0.018). Without optical, it equals SAR-only.
  - At the oracle threshold, fusion scores 0.668: some of the gap to τ\* is threshold transfer between events.
- **Per event**, fusion ranges from 0.443 (Pakistan) to 0.852 (Mekong); see
  `reports/sen1floods11_table.md`.
- **Conformal recall (α = 0.1).** The mean per-chip miss rate is 0.08–0.10, but the target holds on only
  5–8 of 10 held-out events, and precision at the conformal threshold is about 0.25.
- **EMSR664 Forlì, zero-shot against Copernicus EMS.** Scored on the 1101 km² common region against the nearest
  delineation:
  - 23 May (optical on 9/16 tiles): fusion **0.507**; the same model without optical, 0.230.
  - 22 May (no optical): fusion 0.390, below the physics baseline's 0.577.
  - Zero-shot, the learned SAR path over-maps fields in a new region.
- **Synthetic benchmark (3 seeds).** Fusion 0.676, the same as SAR-only. Graceful degradation costs 0.002.
  This is pipeline validation, not an accuracy claim.

## Factors and known failure modes

- **Optical's measured value is partly agreement with the annotators.** Sen1Floods11 labels were
  drawn mostly on the Sentinel-2 image the model also sees. The CEMS comparison is the independent
  check.
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
© European Union, Copernicus Emergency Management Service.

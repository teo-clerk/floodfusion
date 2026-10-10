# floodfusion

**Flood maps from Sentinel-1 radar that a duty officer can act on at 03:00.** The maps are calibrated,
ship with an exclusion mask saying where they must not be trusted, and use Sentinel-2 only where it is
clear. They are tested on ten real floods, against the published benchmark split, and against Copernicus
EMS's own delineations of the 2023 Emilia-Romagna floods.

[![ci](https://github.com/teo-clerk/floodfusion/actions/workflows/ci.yml/badge.svg)](https://github.com/teo-clerk/floodfusion/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/licence-MIT-green)
![uv](https://img.shields.io/badge/env-uv%20locked-purple)
![data](https://img.shields.io/badge/data-open%2C%20no%20credentials-brightgreen)

![EMSR664 Forlì: zero-shot Sentinel-1 flood map vs the Copernicus EMS delineation](assets/forli_hero.png)

*Forlì, 23 May 2023: the cross-attention model, zero-shot with Sentinel-2 on 9 of 16 tiles (centre), against
Copernicus EMS's COSMO-SkyMed delineation of 21 May (right).*

## Results at a glance

**Ten real floods, each held out.** Sen1Floods11, re-ingested; leave-one-event-out — every event is
scored by a model that never saw it ([full table](reports/sen1floods11_table.md)).

| Model | IoU at τ* | AP |
| --- | --- | --- |
| Global Otsu on VV | 0.299 | 0.451 |
| Physics baseline: adaptive threshold + change + terrain masks | 0.157 | 0.152 |
| Physics baseline + optical water index (MNDWI) | 0.444 | 0.440 |
| U-Net, SAR only | 0.365 | 0.538 |
| **Cross-attention fusion** (3 seeds) | **0.625** | **0.769** |
| The same fusion model with optical removed at inference | 0.383 | 0.570 |
| TerraMind-1.0-small (IBM/ESA foundation model, frozen) + UNet decoder | 0.576 | 0.765 |

- **Fusion beats the SAR-only U-Net on all 10 held-out floods**: +0.26 IoU, exact sign-flip test, Holm-adjusted
  p = 0.018. The labels were drawn on the optical image the model sees, so part of that gain is agreement
  with the annotators; the Copernicus EMS case below is the independent check.
- **Without optical, the fusion model is the SAR-only model**: −0.005 IoU, not significant. Cloud costs what SAR
  alone costs, never more.
- **The IBM/ESA foundation model**, frozen, lands 0.05 IoU below the fusion model (behind on 7 of 10 events; not
  significant).
- **The published within-event split barely flatters these models**: at most +0.015 IoU at τ*, none
  significant, against +0.048 for the synthetic leakage row.
- **Conformal recall control holds on average but not per flood**: the mean per-chip miss rate is 0.08–0.10 at
  α = 0.1, but it holds on only 5–8 of 10 new floods. Events are not exchangeable.
- **Physics rules built on backscatter change fail on these labels**: in 8 of 10 events the labelled water did
  not darken against a reference up to 60 days earlier ([diagnostic](reports/sen1floods11_label_change.md)).

**Against Copernicus EMS's own maps: EMSR664, Forlì, May 2023** (zero-shot: trained on Sen1Floods11
only; [report](reports/emsr664_forli.md)). Scored on the 1101 km² that both full Sentinel-1 passes and all
four CEMS delineations (COSMO-SkyMed, RADARSAT-2, PAZ) observed, against the delineation nearest in time:

| Sentinel-1 pass | Optical | Fusion | Same model, no optical | Physics baseline |
| --- | --- | --- | --- | --- |
| 22 May (12 h after CEMS) | none | 0.390 | 0.407 | **0.577** |
| 23 May (48 h after CEMS) | 9 of 16 tiles | **0.507** | 0.230 | 0.412 |

- **Optical's value survives an independent reference.** On 23 May, Sentinel-2 adds +0.28 IoU, and the
  fusion map's flooded area (28.0 km²) matches CEMS's latest delineation (28.0 km²); without optical, the
  same model maps 94.7 km².
- **Zero-shot to the Po plain, the learned SAR-only path over-maps fields** and loses to the physics baseline
  (0.41 vs 0.58). It is not yet a drop-in SAR-only product for a new region.
- **The 18 May pass saw only the coast** (4 of 16 tiles) and is scored on its own coverage.

**Synthetic benchmark: pipeline validation, not accuracy** (13 rows × 3 seeds, 3 test basins;
[ablation](reports/ablation_table.md), [stress](reports/stress.md)).

- **The leakage row works**: a random split inflates IoU by +0.048, in all 3 test basins.
- **What does not pay off here**, reported rather than tuned away; three test basins support no significance
  verdicts:
  - fusion does not beat SAR-only (0.676 vs 0.676);
  - plain BCE beats the compound loss (0.740 vs 0.676);
  - the fusion gate barely trained, averaging 0.12–0.17 over clear sky ([gate diagnostic](reports/gate_diagnostic.md)).
- **Under wind** (0.5 → 10 m/s), the physics baseline collapses (IoU 0.74 → 0.006) while the learned models
  degrade gracefully (0.60 → 0.53).

Every report names the git SHA it was produced at, and regenerates from it. The table builders refuse runs
made from different code, mixed datasets or mixed test sets.

## Why this is hard

- **The weather that floods is the weather that clouds.** Optical satellites are usually blind when it
  matters. Sentinel-1 C-band radar sees through cloud, day and night, so the map must stand on SAR.
- **SAR is ambiguous.**
  - Calm water is dark, but so are radar shadow, tarmac and dry sand.
  - Wind raises 4 cm Bragg waves and brightens open water.
  - Flooded forest and flooded streets get *brighter* (double bounce).
- **Labels disagree with each other.** A benchmark that marks *water on a date* and a Copernicus EMS map
  that marks *flood extent*, generalised from a 3 m image at another hour, are different targets.
- **Splits flatter models.** Chips from the same flood in both training and test inflate a score. The
  published Sen1Floods11 split does exactly that, and this repository measures by how much.

## How it works

```mermaid
graph LR
  A["STAC search<br/>same relative orbit"] --> B["Sentinel-1 RTC γ⁰<br/>pre + post (VV, VH)"]
  A --> C["Sentinel-2 L2A<br/>B3 B8 B11 + SCL"]
  A --> D["Copernicus DEM<br/>GLO-30"]
  B --> E["one 10 m UTM grid<br/>optical moved onto the SAR geometry"]
  C --> E
  D --> F["slope · HAND · local incidence<br/>shadow / layover"]
  F --> E
  E --> G["SAR encoder"] & H["optical encoder"]
  G --> I["cross-attention<br/>Q optical · K,V SAR<br/>per-pixel quality gate"]
  H --> I
  I --> J["U-Net decoder"] --> K["calibration on validation<br/>prior · temperature · Platt"]
  K --> L["τ* from the cost ratio<br/>+ conformal recall threshold"]
  L --> M["GFM-style COG layers<br/>+ STAC Item"]
```

- **SAR-only by construction.** The fusion is a gated residual on the SAR features,
  `F = F_s + g ⊙ Attn(Q_optical, K_SAR, V_SAR)` with `g ∈ [0, 1]` per pixel. With the gate closed the
  network *is* the SAR-only network, bit for bit. Missing optical is a learned embedding, never zeros.
  Per-sample modality dropout trains the SAR-only case.
- **Physics first.** The pipeline uses:
  - Refined Lee filtering in linear power;
  - Horn slope in a metric CRS;
  - the local incidence angle;
  - ray-swept radar shadow and layover;
  - HAND (Priority-Flood + D8);
  - a Range–Doppler geocoder with area-based radiometric terrain correction, which reproduces ESA's
    own geolocation tie points to 0.1 px.
- **A threshold that means something.** Scores are calibrated on validation, so the decision threshold
  τ* = 1/(1 + r) follows from a stated miss-to-false-alarm cost ratio r. A conformal threshold bounds
  the expected per-chip miss rate.
- **An operational product.** Seven Cloud-Optimized GeoTIFFs with the structure of the Copernicus Global
  Flood Monitoring product: flood and water extent, reference water, exclusion mask, 0–100 likelihood,
  advisory flags and the full flag field. A STAC 1.0 Item links every input scene.

## What the real data taught us

- **Sentinel-2 and Sentinel-1 are not co-registered, and the labels follow Sentinel-2.** The 2016–2019
  Sentinel-2 L2A passes sit 1–3 px (10–30 m) from the radar geometry. Uncorrected, labels drawn on them
  handicap every SAR-only model and flatter fusion. On cloudy monsoon events a label-free SWIR/SAR estimate
  has too few clear edges to measure the offset, so each event's labels and optical move by one whole-pixel
  shift taken from the labels' own water edges against the SAR when that estimate is precise. It is a
  two-parameter registration per event that uses labels, disclosed in [`DATA_CARD.md`](DATA_CARD.md) and
  checked independently in the [registration report](reports/sen1floods11_registration.md).
- **Sentinel-1A and 1B differ by 0.6 px in azimuth.** Same-satellite pairs co-register to 0.0 px.
- **Benchmark labels are not change labels.** In most Sen1Floods11 events the labelled flood shows no
  backscatter drop against a same-orbit image up to 60 days earlier
  ([diagnostic](reports/sen1floods11_label_change.md)). Change detection, the backbone of operational
  services, is structurally penalised on such labels.
- **Archives have gaps.**
  - One event's acquisition is missing from the RTC collection; the event is dropped and the reason is
    reported.
  - Another event's GRD annotation is missing; its geometry comes from an orbit model, validated
    against a real annotation (≤ 0.74° incidence error).
- **Copernicus EMS did not map this flood with Sentinel-1.** The EMSR664 delineations come from
  COSMO-SkyMed, RADARSAT-2 and PAZ at 3 m. Comparing with them compares two products at two instants.

## Reproduce

No step needs an account or an API key. The test suite and the synthetic demonstration run offline.

```bash
make setup && make preflight       # uv-locked environment (Python 3.12), environment checks
make test                          # offline suite (CPU); make test-all adds the live-service tests
make demo SEED_PROTOCOL=full       # synthetic benchmark -> 13-row ablation, 3 seeds (~1.5 h on an RTX 4060 Laptop)
make stress                        # single-factor stress curves + the fusion gate figure
make s1f11 S1F11_LEARNED="4 9 14"  # Sen1Floods11 re-ingested, leave-one-event-out + published split (~9 h)
make forli                         # EMSR664 Forlì: every Sentinel-1 pass vs every CEMS delineation
```

The TerraMind comparison (row 14) needs the optional extra: `uv sync --extra dev --extra fm`.
`docker build -t floodfusion .` gives the same locked environment; `docker run --rm floodfusion` runs
the tests. [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) walks one small event from stack
to product in about 30 s on a CPU.

## Where the map must not be used

- **Under dense canopy and in dense urban areas** "dry" means "not seen". These areas are flagged, not mapped.
- **Radar shadow and layover** carry no information and are excluded.
- **Wind** can hide open water.
- **Time.** The map is a snapshot at the acquisition time. It gives no depth and no damage assessment.

More in [`MODEL_CARD.md`](MODEL_CARD.md) and [`DATA_CARD.md`](DATA_CARD.md).

## Repository layout

```
src/floodfusion/
├── data/          STAC matching, archive access, the shared per-grid assembly, Sentinel-1 annotation and
│                  Range–Doppler geocoder, orbit-model geometry, Sen1Floods11 and CEMS clients,
│                  remote-input guards (http.py), chipping, dataset, the synthetic benchmark
├── models/        encoders, cross-attention fusion + gate, segmenter, ablation zoo, losses, TerraMind wrapper
├── masks.py       slope, local incidence, shadow/layover, HAND, validity flags
├── baselines.py · baseline_runner.py     the physics baseline (adaptive threshold + change + masks)
├── train.py · evaluate.py · calibration.py (incl. conformal risk control)
├── crossval.py · cv_report.py            leave-one-event-out, the published split, the protocol gap
├── ablation.py · reporting.py · stress.py · case_study.py · products.py · predict.py
configs/           one frozen YAML per ablation row, overlays, comparisons/, event definitions
tests/             unit, integration and end-to-end tests (markers: slow, gpu, network, data)
reports/           every table above, regenerated by the make targets
assets/            the figures in this README
```

Code comments cite a design specification as `docs/NN §x`, its milestone plan as `MILESTONES.md` and
milestone codes such as `P1-M4.1`. Neither is part of this repository; the citations record which
design decision or acceptance check each piece of code implements.

## Citing

See [`CITATION.cff`](CITATION.cff). Please also cite Sen1Floods11 (Bonafilia et al., 2020) if you use
its labels.

## Acknowledgements

Contains modified Copernicus Sentinel data, processed by Microsoft Planetary Computer. Copernicus EMS
products: © European Union, Copernicus Emergency Management Service. Copernicus DEM: © DLR e.V.
2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union
and ESA. JRC Global Surface Water: EC JRC / Google. ESA WorldCover 2021. TerraMind: IBM and ESA.
Developed with AI assistance (Claude Code); the design decisions, verification and results were reviewed
by the author.

## Licence

MIT — see [`LICENSE`](LICENSE).

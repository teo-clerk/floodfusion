# floodfusion — SAR-first Sentinel-1 / Sentinel-2 fusion for near-real-time flood mapping

`floodfusion` turns a Sentinel-1 radar acquisition over a flood, plus a pre-event reference, a
digital elevation model and an optional, probably cloudy Sentinel-2 image, into a calibrated
10 m flood-extent product with an explicit exclusion mask. The product is built for the
Copernicus Emergency Management Service (CEMS) Rapid Mapping setting: a duty officer has to act on
the map at 03:00 and needs to know every reason it might be wrong.

> **Status.** The pipeline, model, evaluation and reporting are complete and tested. The results
> below are measured on a physics-based **synthetic** benchmark and validate the pipeline and the
> experimental design; they are not real-world accuracy. A real-data benchmark and a comparison
> against official CEMS delineations are in progress.

## Why this is hard

- **The weather that floods is the weather that clouds.** Optical sensors are usually blind when it
  matters; Sentinel-1 C-band SAR images through cloud, day and night.
- **SAR is ambiguous.** Calm water is dark, but so are radar shadow, smooth tarmac and dry sand.
  Wind-roughened water (Bragg waves at 4.4 cm) looks like land. Flooded forest and flooded streets
  are *brighter* because of double bounce.
- **So the model must be correct without optical data**, and use optical only where it is clear.

## What is built

- **Credential-free data access.** STAC matching on Microsoft Planetary Computer and the Copernicus
  Data Space Ecosystem, with the same-relative-orbit pre/post constraint. Sentinel-1 RTC γ⁰ with QA,
  Sentinel-2 L2A with per-tile offset harmonisation, Copernicus DEM GLO-30, JRC Global Surface
  Water and ESA WorldCover, all on one 10 m UTM grid.
- **A Range–Doppler geocoder with area-based radiometric terrain correction.** It reproduces the
  geolocation tie points of a real Sentinel-1 product annotation to 0.1 px (median).
- **Terrain physics.** Horn slope in a metric CRS, local incidence angle, ray-swept radar shadow
  and layover, and HAND (Priority-Flood + D8).
- **A SAR-first fusion network.** Two GroupNorm encoders and cross-attention with queries from
  optical and keys/values from SAR. A per-pixel quality gate and a learned missing-modality
  embedding are fused as a gated residual on the SAR stream: with the gate closed the output is
  bit-identical to the SAR-only path, by construction. Per-sample modality dropout trains the
  SAR-only case.
- **Evaluation that does not flatter.** Basin-blocked splits with a buffer larger than the measured
  spatial autocorrelation range; calibration on validation (prior correction, temperature, Platt);
  a threshold derived from a stated cost ratio; basin-bootstrap intervals and paired, matched-seed
  gaps; and no significance verdict when there are too few test basins to support one.
- **An operational product.** Calibrated probability, flood / permanent-water classes and a bit
  field of exclusion flags (shadow, layover, steep slope, dense forest, dense urban, low confidence).
- **Reproducibility.** Every reported number regenerates from a git SHA. The table builder refuses
  mixed manifests, test sets or configurations, and refuses results from modified code.

## Results (synthetic benchmark)

Thirteen-row ablation, three held-out test basins. Full table, paired gaps and protocol:
[`reports/ablation_table.md`](reports/ablation_table.md). Memory budget:
[`reports/vram_budget.md`](reports/vram_budget.md).

| Finding | Measured |
| --- | --- |
| Reference model (cross-attention fusion), IoU at the cost-derived threshold | 0.676 |
| Same weights, SAR only at inference (graceful degradation) | 0.674 (−0.002, lower in 3/3 basins) |
| Same model on a random chip split (spatial-leakage illusion) | +0.046, higher in 3/3 basins |
| Learned model over the engineered physics baseline | +0.081, higher in 2/3 basins |
| Raw scores vs calibrated at τ* = 0.25 | IoU 0.535 → 0.676; ECE 0.031 → 0.008 |
| Inference | 6.3 ms per 256² chip; about 0.7 min per Sentinel-1 IW frame (RTX 4060 Laptop) |

## Quickstart

```bash
make setup && make preflight   # uv-locked environment (Python 3.12), environment checks
make test                      # offline test suite; `make test-all` adds the live catalogue tests
make demo                      # synthetic benchmark → splits → VRAM budget → ablation grid → table
make table                     # regenerate the ablation table from the run directories alone
make ingest EVENT=configs/events/example.yaml   # a real event from Planetary Computer, no credentials
```

No step needs an account or an API key. The test suite and the synthetic demonstration run fully
offline.

## Where the map must not be used

- Under dense canopy "dry" means "not seen": C-band does not reliably penetrate it.
- Dense urban areas are flagged, not mapped.
- Radar shadow and layover carry no information and are excluded.
- Wind can hide open water.
- The map is a snapshot at the acquisition time. It gives no depth and no damage assessment.

## Repository layout

```
src/floodfusion/
├── data/         STAC matching, ingest, geocoder + RTC, annotation parser, chipping, dataset, synthetic benchmark
├── models/       encoders, cross-attention fusion + gate, segmenter, ablation zoo, losses
├── masks.py      slope, local incidence, shadow/layover, HAND, validity flags
├── baselines.py  Otsu, tile-adaptive thresholding, change detection, the physics rule
├── splits.py     basin-blocked splits, variogram range, normalisation statistics
├── train.py · evaluate.py · calibration.py · reporting.py · predict.py · ablation.py
configs/          one frozen YAML per ablation row, overlays, event definitions
tests/            unit, integration and end-to-end tests (markers: slow, gpu, network, data)
reports/          the committed ablation table and VRAM budget
```

Code comments cite a design specification as `docs/NN §x`. That specification is not part of this
repository; the citations record which design decision each piece of code implements.

## Acknowledgements

Contains modified Copernicus Sentinel data, processed through Microsoft Planetary Computer.
Developed with AI assistance (Claude Code); the design decisions, verification and results were
reviewed by the author.

## Licence

MIT — see [`LICENSE`](LICENSE).

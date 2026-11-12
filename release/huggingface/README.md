---
license: cc-by-4.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - remote-sensing
  - earth-observation
  - sentinel-1
  - synthetic-aperture-radar
  - flood-mapping
  - copernicus
  - semantic-segmentation
datasets:
  - links-ads/geoid-flood
---

# floodfusion v1.2: Sentinel-1 flood maps, frozen for a blind test

The frozen model of [floodfusion](https://github.com/teo-clerk/floodfusion), a flood-mapping system built from
Sentinel-1 radar, with optical used only where it is clear. These are the weights it was frozen with for its
pre-registered blind test on Copernicus Emergency Management Service floods, three seeds, byte for byte the files
that `reports/v12_freeze.json` pins at the tag `v1.2.0-freeze`.

**Still floodfusion's current model.** Version 1.3 (November 2026) tested fine-tuning these weights on Copernicus
EMS's own flood labels. Under an adoption rule fixed in advance, the fine-tuned model scored lower (0.380 against
0.421 macro IoU, cross-validated over 38 activations) and was not adopted, so there are no v1.3 weights.

- **Network (row 21):** SAR–optical cross-attention fusion with terrain and a seasonal Sentinel-1 reference: the
  median backscatter of the same orbit around the same date a year earlier, and the change from it. 72.7 MB per
  seed.
- **Training labels:** Kuro Siwo plus GEOID-Flood, 7,006 chips from 138 floods in 81 location groups.
- **Calibration:** temperature scaling fitted on held-out validation groups, plus a separate calibration for windows
  without usable optical. The decision threshold τ* = 0.25 follows from a stated miss-to-false-alarm cost ratio of 3.

## Files

| Path | What it is |
| --- | --- |
| `runs/v12/final/all/all/row21_fusion_context_seasonal/seed{42,43,44}/best.pt` | weights (`torch.load(..., weights_only=True)`) |
| `…/seed{42,43,44}/run.json` | the full configuration, its hash and the training record |
| `…/seed{42,43,44}/eval_test.json` | the calibration and τ* |
| `runs/v12/final/all/all/row21_fusion_context_seasonal__eval_sar_only/seed{42,43,44}/eval_test.json` | the calibration for windows without optical |
| `SHA256SUMS` | sha256 of every file above, equal to the freeze manifest's |

## Use

```bash
git clone https://github.com/teo-clerk/floodfusion && cd floodfusion && uv sync --frozen
# the weights, each file checked against the freeze manifest, at the commit that added them to this repository
uv run --frozen python -m floodfusion.weights fetch --repo teo-clerk/floodfusion \
    --revision faed3bb78668cf25052211a7e834f34652a4cdf6 \
    --run runs/v12/final/all/all/row21_fusion_context_seasonal/seed42 --dest .
# a flood map of any place and date (CPU is enough): COG layers, a STAC item and flood polygons per tile
uv run --frozen python -m floodfusion.live map --bbox 11.8 44.15 12.2 44.3 --date 2023-05-17 --out maps/forli
```

The `weights` and `live` commands reach the repository's `main` branch with release v1.3.0; until then they are on
the branch `feat/v1.3-live` (`git checkout feat/v1.3-live` after cloning).

The map command reads Sentinel-1 RTC, the Copernicus DEM, JRC surface water and ESA WorldCover from Microsoft Planetary
Computer, and Sentinel-2 where it is clear. The live map rebuilds the benchmark's input stack bit for bit: on one
replayed benchmark tile (EMSR683 AOI02), all 14 input bands were identical. On the benchmark's own stack, a CPU run
gives the stored GPU run's IoU at τ* (0.414). The live stack has no area-of-interest mask, so the network also reads
the ground around the area; on the same pixels it scores 0.421.

## Results (seen floods, descriptive)

On the 38 Copernicus EMS flood activations of 2023–26 drawn on radar satellites (EMSR664 excluded), scored once after
the freeze (`reports/v12_cems_rebuilt.md`):

| Model | Macro IoU at τ* | Oracle-threshold IoU | AP |
| --- | --- | --- | --- |
| **This model (v1.2, row 21)** | **0.417** | 0.468 | 0.598 |
| TerraMind-base + LoRA, trained the same way | 0.405 | 0.446 | 0.583 |
| v1.1's best (fusion, Kuro Siwo flood labels) | 0.407 | 0.438 | 0.527 |
| Physics baseline (thresholds on backscatter change) | 0.373 | — | — |

On the 34 of those floods that Copernicus Global Flood Monitoring covers, this model scores 0.419 and GFM 0.321.
Only the gap to GFM is significant (`reports/v12_cems_seen_tests.md`).

**How high could anyone score?** Two Copernicus EMS delineations of the same area, made at most 24 hours apart,
agree with each other at 0.520 macro IoU over 34 seen activations (`reports/reference_ceiling.md`).

**The confirmatory test is still running.** Copernicus flood activations from EMSR899 on are scored by these frozen
weights and sealed until 20 activations in 15 clusters are scored, or 31 December 2027 (`reports/protocol_v1_2.md`).

## Intended use and limits

- **Use:** a first-guess flood extent for a rapid-mapping analyst after a flood, from the first Sentinel-1 pass;
  large-area screening, with an analyst reviewing flagged and low-confidence areas.
- **Not seen by C-band radar:** water under closed canopy and between buildings. Dense forest and built-up land are
  flagged *not reliably mapped*; "dry" there means "not seen". Radar shadow and layover are excluded.
- **Not estimated:** depth, damage or impact. The map is a snapshot at the acquisition time.
- **Do not** deny assistance to an area marked "dry" under canopy, in a city, in radar shadow, or outside the
  acquisition time. Every product ships with its exclusion mask and low-confidence band.

The repository's [MODEL_CARD.md](https://github.com/teo-clerk/floodfusion/blob/main/MODEL_CARD.md) lists the known
failure modes (partial cloud, threshold transfer, wind-roughened water).

## Training data and attribution

- **Kuro Siwo** annotations (Bountos et al., NeurIPS 2024), CC BY 4.0, pinned at `Orion-AI-Lab/KuroSiwo-annotations@b930dd27`.
- **GEOID-Flood** labels (Chiriaco et al., 2026), CC BY 4.0, pinned at revision `868407460bf3` of `links-ads/geoid-flood`:
  "contains modified Copernicus Sentinel data and Copernicus EMS information".
- **Sentinel-1 RTC** from Microsoft Planetary Computer (CC BY 4.0). Contains modified Copernicus Sentinel data.
- **Copernicus DEM GLO-30**, produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space
  GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA. **ESA WorldCover 2021**, CC BY 4.0.
- **Evaluation only:** Copernicus Emergency Management Service rapid-mapping delineations and Global Flood Monitoring,
  © European Union. Contains modified Copernicus Emergency Management Service information (2023–2026).
- No Sen1Floods11 data or labels were used to train these weights.

## Licence and citation

The weights are released under **CC BY 4.0**, the licence of their training labels; the code is MIT. Cite the
repository with its [`CITATION.cff`](https://github.com/teo-clerk/floodfusion/blob/main/CITATION.cff).

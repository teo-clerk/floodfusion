# Does the live map rebuild the benchmark? One replayed tile (2026-11-10)

`floodfusion.live map` was run on CPU for the box, date and Sentinel-1 orbit of one seen benchmark tile, EMSR683 AOI02
`t00_00` (Norway, 10 August 2023, relative orbit 37, descending; seasonal reference before 1 August 2023). Its stack
was compared with the benchmark's stack of the same tile (`data/cems_benchmark_v12/EMSR683_AOI02`), and its
predictions with the stored GPU histogram (`reports/v12_cems_rebuilt_scores.json.gz`, row 21, seed 42). The script
was a one-off check and is not committed; the numbers below are its output.

| Check | Result |
| --- | --- |
| Grid (CRS, transform, shape) | identical |
| The 14 input bands (all but label and validity) | identical, pixel for pixel |
| Validity, ignoring the outside-AOI flag | identical |
| Seasonal reference (`seasonal.tif`, `seasonal.json`) | identical: both bands, the same 5 scenes |
| Map | 2.24 km² of flood in 116.9 km² mapped; pass at 05:47:56 UTC |

| Prediction (CPU unless stated), scored on the benchmark's pixels | IoU at τ* |
| --- | --- |
| stored GPU run, benchmark stack | 0.4141 |
| CPU, benchmark stack | 0.4141 |
| CPU, live stack | 0.4210 |

- **CPU against GPU:** on the same stack the IoU at τ* is unchanged. The histograms differ by a total variation of
  0.026 (flood pixels) and 0.064 (dry pixels), the rounding of mixed precision on the GPU against float32 on CPU.
- **Live against benchmark:** the probabilities differ (median 2.5e-4; 3.5 % of pixels by more than 0.01; at most
  0.065). The difference is not confined to the AOI's edge: it is present 128–256 px inside it. The benchmark marks
  pixels outside the Copernicus area of interest invalid, so their seasonal change channel is empty, while the live
  stack has no area of interest and reads them. Since the network's receptive field spans its window, that context
  reaches the pixels inside. The live map scores 0.007 higher on this tile; one tile decides nothing about the sign.
- **What follows:** the live map's inputs are the benchmark's, and the model runs on CPU without a change of result.
  A live map is not the benchmark's prediction of the same pixels, which is why the live service never reports a
  score.

# v1.3 ladder: rungs 1 and 2 (rung 3 pending)

Protocol v1.3 §3, committed before any rung was computed. Seen CEMS 2023–26 benchmark, primary stratum (products drawn on radar), EMSR664 excluded; row 21, seed 42; every activation scored with its fold's threshold (fitted on the other folds). Binned counts. Exploratory: this benchmark has been seen.

| Rung | Per activation (K = 38) | Per cluster (K = 30), descriptive |
| --- | --- | --- |
| R1, zero-shot at τ* | 0.421 | 0.444 |
| R2, cross-fitted cost-optimal threshold | 0.423 | 0.446 |

**R2 over R1:** gain +0.002, better in 26 of 38 activations: **not adopted** (rule: ≥ +0.005 and more than half).

Fold thresholds (probability): fold0 0.266, fold1 0.331, fold2 0.266, fold3 0.266, fold4 0.237.

Three seeds (descriptive): seed42 R1 0.421, R2 0.423; seed43 R1 0.427, R2 0.441; seed44 R1 0.408, R2 0.418.

| Activation | Fold | R1 | R2 |
| --- | --- | --- | --- |
| EMSR657 | fold3 | 0.645 | 0.646 |
| EMSR668 | fold2 | 0.323 | 0.326 |
| EMSR683 | fold2 | 0.545 | 0.548 |
| EMSR705 | fold0 | 0.399 | 0.402 |
| EMSR706 | fold4 | 0.374 | 0.371 |
| EMSR708 | fold4 | 0.695 | 0.690 |
| EMSR711 | fold3 | 0.634 | 0.635 |
| EMSR712 | fold4 | 0.469 | 0.464 |
| EMSR718 | fold3 | 0.047 | 0.046 |
| EMSR722 | fold0 | 0.648 | 0.648 |
| EMSR725 | fold0 | 0.333 | 0.335 |
| EMSR728 | fold3 | 0.422 | 0.425 |
| EMSR755 | fold1 | 0.280 | 0.281 |
| EMSR756 | fold4 | 0.355 | 0.354 |
| EMSR761 | fold1 | 0.079 | 0.080 |
| EMSR763 | fold1 | 0.416 | 0.452 |
| EMSR766 | fold1 | 0.512 | 0.526 |
| EMSR768 | fold4 | 0.279 | 0.273 |
| EMSR771 | fold0 | 0.410 | 0.413 |
| EMSR773 | fold3 | 0.711 | 0.715 |
| EMSR783 | fold2 | 0.530 | 0.535 |
| EMSR788 | fold4 | 0.655 | 0.653 |
| EMSR789 | fold4 | 0.415 | 0.409 |
| EMSR796 | fold0 | 0.335 | 0.338 |
| EMSR797 | fold1 | 0.217 | 0.230 |
| EMSR838 | fold4 | 0.381 | 0.380 |
| EMSR841 | fold0 | 0.174 | 0.177 |
| EMSR844 | fold3 | 0.320 | 0.322 |
| EMSR848 | fold2 | 0.374 | 0.373 |
| EMSR850 | fold0 | 0.272 | 0.273 |
| EMSR854 | fold3 | 0.484 | 0.487 |
| EMSR856 | fold0 | 0.395 | 0.397 |
| EMSR857 | fold1 | 0.772 | 0.782 |
| EMSR861 | fold0 | 0.334 | 0.335 |
| EMSR864 | fold1 | 0.532 | 0.527 |
| EMSR866 | fold3 | 0.575 | 0.577 |
| EMSR868 | fold2 | 0.355 | 0.358 |
| EMSR871 | fold2 | 0.286 | 0.288 |

*Regenerate:* `uv run --frozen python -m floodfusion.ladder r2`.

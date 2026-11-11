# v1.3 ladder: does training on Copernicus's own labels help?

Protocol v1.3 §3, committed before any rung was computed. Seen CEMS 2023–26 benchmark, primary stratum (products drawn on radar), EMSR664 excluded; seed 42. Every activation is scored by the fold that held out its 100 km group; binned counts. Exploratory: this benchmark has been seen. Only H8, on the blind floods, is confirmatory.

| Rung | Macro IoU, 38 activations | Per cluster (descriptive) |
| --- | --- | --- |
| R1, zero-shot at τ* | 0.421 | 0.444 |
| R2, cross-fitted cost-optimal threshold | 0.423 | 0.446 |
| R3, fine-tuned on each fold's training groups | 0.380 | 0.403 |

- **R2 over R1:** +0.002, better in 26 of 38: **not adopted**.
- **R3 over R1** (the highest rung adopted below it): -0.041, better in 15 of 38: **not adopted**.
- Rule: a gain of at least +0.005 and better on more than half of the activations.

**v1.3's model:** none: no rung was adopted; H8 is not run.

Descriptive, not part of the rule:

- R3 over R2: -0.043, better in 15 of 38.
- Per cluster, R3 over R1: -0.041, better in 12 of 30 clusters.
- Threshold-free: AP R1 0.606, R3 0.573; oracle IoU R1 0.468, R3 0.457.
- Products drawn on optical images (6 activations): R1 0.158, R3 0.163.
- Fold thresholds: R2 fold0 0.266, fold1 0.331, fold2 0.266, fold3 0.266, fold4 0.237; R3's τ* fold0 0.250, fold1 0.250, fold2 0.250, fold3 0.250, fold4 0.250.

| Activation | Fold | R1 | R2 | R3 |
| --- | --- | --- | --- | --- |
| EMSR657 | fold3 | 0.645 | 0.646 | 0.671 |
| EMSR668 | fold2 | 0.323 | 0.326 | 0.298 |
| EMSR683 | fold2 | 0.545 | 0.548 | 0.597 |
| EMSR705 | fold0 | 0.399 | 0.402 | 0.312 |
| EMSR706 | fold4 | 0.374 | 0.371 | 0.424 |
| EMSR708 | fold4 | 0.695 | 0.690 | 0.732 |
| EMSR711 | fold3 | 0.634 | 0.635 | 0.677 |
| EMSR712 | fold4 | 0.469 | 0.464 | 0.400 |
| EMSR718 | fold3 | 0.047 | 0.046 | 0.122 |
| EMSR722 | fold0 | 0.648 | 0.648 | 0.637 |
| EMSR725 | fold0 | 0.333 | 0.335 | 0.405 |
| EMSR728 | fold3 | 0.422 | 0.425 | 0.337 |
| EMSR755 | fold1 | 0.280 | 0.281 | 0.247 |
| EMSR756 | fold4 | 0.355 | 0.354 | 0.247 |
| EMSR761 | fold1 | 0.079 | 0.080 | 0.033 |
| EMSR763 | fold1 | 0.416 | 0.452 | 0.210 |
| EMSR766 | fold1 | 0.512 | 0.526 | 0.341 |
| EMSR768 | fold4 | 0.279 | 0.273 | 0.383 |
| EMSR771 | fold0 | 0.410 | 0.413 | 0.074 |
| EMSR773 | fold3 | 0.711 | 0.715 | 0.490 |
| EMSR783 | fold2 | 0.530 | 0.535 | 0.590 |
| EMSR788 | fold4 | 0.655 | 0.653 | 0.630 |
| EMSR789 | fold4 | 0.415 | 0.409 | 0.433 |
| EMSR796 | fold0 | 0.335 | 0.338 | 0.013 |
| EMSR797 | fold1 | 0.217 | 0.230 | 0.145 |
| EMSR838 | fold4 | 0.381 | 0.380 | 0.406 |
| EMSR841 | fold0 | 0.174 | 0.177 | 0.305 |
| EMSR844 | fold3 | 0.320 | 0.322 | 0.380 |
| EMSR848 | fold2 | 0.374 | 0.373 | 0.371 |
| EMSR850 | fold0 | 0.272 | 0.273 | 0.088 |
| EMSR854 | fold3 | 0.484 | 0.487 | 0.519 |
| EMSR856 | fold0 | 0.395 | 0.397 | 0.384 |
| EMSR857 | fold1 | 0.772 | 0.782 | 0.621 |
| EMSR861 | fold0 | 0.334 | 0.335 | 0.214 |
| EMSR864 | fold1 | 0.532 | 0.527 | 0.479 |
| EMSR866 | fold3 | 0.575 | 0.577 | 0.559 |
| EMSR868 | fold2 | 0.355 | 0.358 | 0.346 |
| EMSR871 | fold2 | 0.286 | 0.288 | 0.306 |

*Regenerate:* `uv run --frozen python -m floodfusion.ladder report`.

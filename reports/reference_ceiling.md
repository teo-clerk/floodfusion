# The reference ceiling: two Copernicus EMS maps of the same area, at most 24 h apart

Seen flood activations EMSR656–EMSR898 only; descriptive, no test. Rules fixed before this was computed (`reports/protocol_v1_3.md` §2.3, §5). For each AOI, consecutive delineation products whose latest source images are at most 24 h apart are compared on the area both observed (10 m UTM grid; each product's observed flood). An activation pools its pairs' pixels; activations where neither map holds flood are left out.

**Reading, fixed in the protocol:** set against the models' 0.42 on radar-drawn pairs. Agreement near 0.5 means the benchmark is near its ceiling; near 0.8, most of the gap is the models'. Two maps hours apart also differ because the flood moved, as a model's map and its reference do.

|  | Activations | Product pairs | Macro IoU over activations | Median pair IoU |
| --- | --- | --- | --- | --- |
| All pairs | 34 | 79 | 0.520 | 0.498 |
| Without pairs repeating > 5% of their flood (sensitivity) | 33 | 76 | 0.509 | 0.491 |

## By time between the two maps

| Gap | Activations | Product pairs | Macro IoU over activations | Median pair IoU |
| --- | --- | --- | --- | --- |
| 6–24 h | 34 | 79 | 0.520 | 0.498 |

## By sensor pair

| Sensors | Activations | Product pairs | Macro IoU over activations | Median pair IoU |
| --- | --- | --- | --- | --- |
| optical–optical | 1 | 1 | 0.796 | 0.796 |
| other | 1 | 1 | 0.329 | 0.329 |
| radar–optical | 2 | 2 | 0.380 | 0.380 |
| radar–radar | 32 | 75 | 0.515 | 0.502 |

## By shared source image

| Source image | Activations | Product pairs | Macro IoU over activations | Median pair IoU |
| --- | --- | --- | --- | --- |
| not shared | 34 | 79 | 0.520 | 0.498 |

## Carried-over flood (the probe)

Share of the later product's flood area drawn on images no later than the earlier product's latest image, over 79 pairs: median 0.000, 3 pairs above 5 %.

## Per activation

| Activation | IoU (pairs pooled) |
| --- | --- |
| EMSR657 | 0.869 |
| EMSR664 | 0.291 |
| EMSR683 | 0.695 |
| EMSR692 | 0.796 |
| EMSR705 | 0.508 |
| EMSR706 | 0.460 |
| EMSR711 | 0.987 |
| EMSR712 | 0.564 |
| EMSR718 | 0.626 |
| EMSR720 | 0.252 |
| EMSR722 | 0.484 |
| EMSR728 | 0.347 |
| EMSR756 | 0.179 |
| EMSR757 | 0.630 |
| EMSR762 | 0.491 |
| EMSR768 | 0.192 |
| EMSR771 | 0.485 |
| EMSR773 | 0.405 |
| EMSR783 | 0.534 |
| EMSR788 | 0.775 |
| EMSR789 | 0.504 |
| EMSR794 | 0.454 |
| EMSR796 | 0.384 |
| EMSR797 | 0.396 |
| EMSR838 | 0.597 |
| EMSR841 | 0.141 |
| EMSR855 | 0.271 |
| EMSR856 | 0.282 |
| EMSR857 | 0.849 |
| EMSR861 | 0.706 |
| EMSR864 | 0.652 |
| EMSR866 | 0.609 |
| EMSR868 | 0.724 |
| EMSR871 | 0.528 |

## Pairs not compared

| Pair | Reason |
| --- | --- |
| EMSR683_AOI09_PRODUCT_MONIT01 | download failed: HTTPStatusError: HTTP 404 for https://rapidmapping.emergency.copernicus.eu/backend/EMSR683/AOI09/DEL_MONIT01/EMSR683_AOI09_DEL_MONIT01_v2.zip |

*Regenerate:* `uv run --frozen python scripts/reference_ceiling.py download` then `… score`.

Contains modified Copernicus Emergency Management Service information (2023–2026).

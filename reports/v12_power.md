# The blind test's power (v1.2)

Written by `scripts/v12_power.py`. The smallest true gap in macro IoU that a two-sided sign-flip test detects at 80% power at Holm's smallest level for three tests (0.0167), by number of activations. Gaps are drawn i.i.d. normal with v1.1's per-activation spread on the CEMS benchmark. Clustering is not modelled.

| Per-activation spread (v1.1) | 10 activations | 15 activations | 20 activations | 30 activations |
|---|---|---|---|---|
| P4: SD 0.077 | 0.096 | 0.072 | 0.061 | 0.048 |
| P3: SD 0.071 | 0.089 | 0.066 | 0.056 | 0.044 |

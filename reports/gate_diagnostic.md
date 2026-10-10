# What the fusion gate learned (row 9, seed 42, synthetic stress scenes)

The gate starts at sigmoid(bias_init) = 0.119. Clear = optical quality > 0.5.

| Cloud fraction | Chips | Mean gate, clear | Mean gate, cloud-masked | Clear pixels g > 0.5 | Cloud-masked pixels g > 0.5 |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 24 | 0.173 | — | 0.056 | — |
| 0.4 | 24 | 0.135 | 0.201 | 0.027 | 0.090 |
| 0.8 | 24 | 0.122 | 0.147 | 0.020 | 0.031 |

Regenerate: `uv run python scripts/gate_diagnostic.py` (after `make demo` and `make stress`).

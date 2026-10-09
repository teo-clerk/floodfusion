# Contributing

Thank you for looking. Issues and pull requests are welcome; please keep the conventions below.

## Set up

```bash
make setup        # uv-managed .venv with the locked dependencies and dev tools (Python 3.12)
make preflight    # environment checks: GPU, bf16, GDAL/PROJ, git
make test         # the offline suite (no network, runs on CPU)
make lint
```

Optional: `uvx pre-commit install` runs ruff and the file checks on every commit.

## Test markers

| Marker | Meaning | Default |
| --- | --- | --- |
| `slow` | seconds to a minute (end-to-end CPU runs) | included |
| `gpu` | needs a CUDA device; skips without one | included |
| `network` | hits a live public service (STAC, Sen1Floods11 bucket, CEMS API) | **excluded**; run with `make test-all` |
| `data` | needs a real dataset under `$ESA_DATA_ROOT` | included, skips without it |

New code comes with tests. A module that does not need torch must import without it
(`tests/test_torch_free_imports.py`), and any live endpoint gets an offline fixture test as
well as a `network` test.

## Reproducibility rules

- A reported number must regenerate from a git SHA: the table builders refuse runs whose code
  differs from the current tree. Commit before running an experiment.
- Do not tune the synthetic generator after seeing results; it is frozen before the grid.
- Splits are frozen files bound to a manifest hash; never derive them from a seed at run time.

## Data

No credentials are needed anywhere. Never commit data, runs, checkpoints or downloaded
products; the Sen1Floods11 labels in particular carry no licence and are fetched, not
redistributed.

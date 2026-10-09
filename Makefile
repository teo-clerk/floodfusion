# =============================================================================
#  Project 01 — floodfusion. One entry point per milestone gate.
#  Every target runs inside the uv-managed virtual environment (.venv).
# =============================================================================
SHELL := /bin/bash
UV    ?= uv
PY    := $(UV) run --no-sync python
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

# Where the synthetic benchmark (or a real ingest) lives, and where runs go.
DATA_ROOT  ?= data/synthetic
RUNS_ROOT  ?= runs
OVERLAY    ?= configs/overlays/synthetic_demo.yaml
SEED       ?= 42
REF_CONFIG := configs/row09_fusion_crossattn.yaml
EVENT      ?= configs/events/example.yaml
REAL_ROOT  ?= data/real
# Seed protocol of the synthetic grid: auto (decided from a measured run), full, affordable.
SEED_PROTOCOL ?= auto

.PHONY: help setup lock preflight test test-fast test-all test-gpu cov lint synth ingest splits \
        baseline train-p1 vram-budget ablation table predict demo clean \
        s1f11 s1f11-download s1f11-ingest s1f11-splits s1f11-cv s1f11-table stress forli

help:  ## List targets
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create .venv (Python 3.12) and install the locked dependencies + dev tools
	$(UV) venv --python 3.12 --allow-existing .venv
	$(UV) sync --extra dev

lock:  ## Re-resolve uv.lock after editing pyproject.toml
	$(UV) lock

preflight:  ## F-M0: environment, GPU, bf16, GDAL/PROJ and git checks (exit 0 = ready)
	$(PY) -m floodfusion.preflight

test:  ## Full offline suite (network tests excluded by pyproject addopts)
	$(UV) run --no-sync pytest

test-fast:  ## Offline suite without the slow tests
	$(UV) run --no-sync pytest -m "not network and not slow"

test-all:  ## Everything, including live STAC / blob-store tests (needs internet)
	$(UV) run --no-sync pytest -m ""

test-gpu:  ## GPU smoke tests only
	$(UV) run --no-sync pytest -m gpu

cov:  ## Offline suite with a line+branch coverage report
	$(UV) run --no-sync pytest --cov=floodfusion --cov-report=term-missing

lint:  ## Static checks
	$(UV) run --no-sync ruff check src tests

synth:  ## Generate the physics-based synthetic benchmark under $(DATA_ROOT)
	$(PY) -m floodfusion.data.synthetic.benchmark --out $(DATA_ROOT)

ingest:  ## P1-M0/M1 real event(s) from Planetary Computer: make ingest EVENT=configs/events/x.yaml
	$(PY) -m floodfusion.data.ingest --event $(EVENT) --out $(REAL_ROOT)

splits:  ## Variogram range, frozen LOBO + row-13 splits, train-only normalisation stats
	$(PY) -m floodfusion.splits --data-root $(DATA_ROOT) --config $(REF_CONFIG) --overlay $(OVERLAY)

baseline:  ## Ablation rows 1-3 (CPU) on the test split
	$(PY) -m floodfusion.baseline_runner --data-root $(DATA_ROOT) --runs-root $(RUNS_ROOT) --overlay $(OVERLAY)

train-p1:  ## Train the reference model (row 9) — configs/row09_fusion_crossattn.yaml
	$(PY) -m floodfusion.train --config $(REF_CONFIG) --data-root $(DATA_ROOT) \
		--overlay $(OVERLAY) --seed $(SEED) --runs-root $(RUNS_ROOT)

vram-budget:  ## P1-M4.1: predicted vs measured peak VRAM table -> reports/vram_budget.md
	$(PY) -m floodfusion.utils.vram --config $(REF_CONFIG) --out reports/vram_budget.md

ablation:  ## P1-M5: run the whole grid (rows 1-13) and write reports/ablation_table.md
	$(PY) -m floodfusion.ablation --data-root $(DATA_ROOT) --runs-root $(RUNS_ROOT) --overlay $(OVERLAY) --protocol $(SEED_PROTOCOL)

table:  ## Regenerate the ablation table from existing run directories (one command)
	$(PY) -m floodfusion.evaluate --ablation-table --runs-root $(RUNS_ROOT) --out reports/ablation_table.md

S1F11_RAW ?= data/sen1floods11_raw
S1F11_ROOT ?= data/sen1floods11
S1F11_RUNS ?= runs/sen1floods11
S1F11_OVERLAY ?= configs/overlays/sen1floods11.yaml

s1f11-download:  ## Sen1Floods11 hand labels + metadata + split lists (md5-verified; ~3 MB)
	$(PY) -m floodfusion.data.sen1floods11 download --raw $(S1F11_RAW)

s1f11-ingest:  ## Re-fetch every Sen1Floods11 input from Planetary Computer (~40 min, resumable)
	$(PY) -m floodfusion.data.sen1floods11 ingest --raw $(S1F11_RAW) --out $(S1F11_ROOT)

s1f11-splits:  ## Leave-one-event-out folds + the published split, with train-only statistics per fold
	$(PY) -m floodfusion.crossval splits --data-root $(S1F11_ROOT) --raw $(S1F11_RAW) --overlay $(S1F11_OVERLAY)

s1f11-cv:  ## Train + evaluate rows 1-4, 9, 11 on every fold of both protocols
	$(PY) -m floodfusion.crossval run --data-root $(S1F11_ROOT) --overlay $(S1F11_OVERLAY) --runs-root $(S1F11_RUNS)

s1f11-table:  ## reports/sen1floods11_table.md from the run directories alone
	$(PY) -m floodfusion.crossval table --data-root $(S1F11_ROOT) --runs-root $(S1F11_RUNS)

s1f11: s1f11-download s1f11-ingest s1f11-splits s1f11-cv s1f11-table  ## The whole real-data benchmark

stress:  ## Single-factor stress curves + the gate figure on unseen synthetic landscapes
	$(PY) -m floodfusion.stress --runs-root $(RUNS_ROOT) --overlay $(OVERLAY) --out reports/stress.md

FORLI_RUN ?= $(S1F11_RUNS)/official/official
forli:  ## EMSR664 Forlì: every Sentinel-1 pass against every CEMS delineation (zero-shot)
	$(PY) -m floodfusion.case_study prepare --activation EMSR664 --aoi 1 --out data/emsr664
	$(PY) -m floodfusion.case_study run --case data/emsr664 --overlay $(S1F11_OVERLAY) \
		--fusion-run $(FORLI_RUN)/row09_fusion_crossattn/seed42 \
		--sar-only-run $(FORLI_RUN)/row11_fusion_eval_sar_only/seed42 \
		--baseline-run $(FORLI_RUN)/row02_baseline_adaptive/baseline --out reports/emsr664_forli.md

predict:  ## Product GeoTIFF for one event stack: make predict RUN=runs/... STACK=... OUT=...
	$(PY) -m floodfusion.predict --run-dir $(RUN) --stack $(STACK) --out $(OUT)

demo: synth splits vram-budget ablation  ## Synthetic benchmark -> splits -> VRAM budget -> ablation grid -> table

clean:  ## Remove caches (keeps data/ and runs/)
	find . -name '__pycache__' -type d -prune -exec $(RM) -r {} +
	$(RM) -r .pytest_cache .ruff_cache htmlcov .coverage

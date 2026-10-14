# floodfusion v1.2 — pre-registered protocol (blind test)

**Committed:** 2026-10-13, before any v1.2 model was trained and before any flood of the blind set was scored. Every
later change goes to `reports/v1_2_deviations.md` before the result it could affect. The models are named here by the
rules that will select them. An amendment fixes their identities at the freeze tag `v1.2.0-freeze`, before the first
blind sweep.

## 1. Questions

v1.1 found two things:
- on independent Copernicus EMS maps, the training labels mattered more than the architecture;
- the task-specific model's win over TerraMind was fragile: it flipped at the oracle threshold, and TerraMind had
  been handicapped (see `reports/v1_1_erratum.md`).

v1.2 asks three questions, on floods that occur after the models are frozen:

- **H5.** v1.2's model beats TerraMind trained on the same data, at each model's own threshold **and** threshold-free.
- **H6.** v1.2's model beats Copernicus GFM, the operational ensemble, on the floods GFM covers.
- **H7.** v1.2's model beats v1.1's best (fusion trained on Kuro Siwo flood labels).

## 2. Data

### 2.1 Training (exploratory development; no test result)

- **Kuro Siwo:** rebuilt with the DEM fix (`data/kurosiwo_v12`, erratum 2), flood labels.
- **GEOID-Flood:** pinned revision `868407460bf3`. Activations before EMSR656, minus Kuro Siwo's codes: 111 events.
  At most 4 wet + 1 dry 5.12 km squares per event (`floodfusion.data.geoid`).
- **Splits:** pooled, with 100 km single-linkage groups across both sets (71 groups; the largest holds about 12 % of
  the windows). Grouped five-fold cross-validation, plus an all-groups model with three seeds.

### 2.2 The blind set (confirmatory)

- **Activations:** every Copernicus EMS rapid-mapping activation with code ≥ **EMSR899**, flood category, judged
  **30 days after its event** (`floodfusion.prospective`).
- **Eligibility:** v1.1's rule, unchanged: delineations drawn on non-Sentinel imagery, with a Sentinel-1 pass within
  24 h covering ≥ 50 % of the extent, and a same-orbit reference pass. At most 4 AOIs per activation.
- **Primary stratum:** products drawn on radar satellites, as in v1.1.
- **Units:**
  - activations;
  - clusters: activations within 100 km and 60 days (single linkage).
- **Already seen:** the metadata (code, date, category, name) of EMSR899–945, queried on 2026-10-13 to estimate
  accrual. No product, pass or score of any of them.

## 3. Models (identities fixed by amendment at the freeze)

All models are trained on the data of §2.1 with the same loss, schedule, sampler, calibration and τ* rule (row 9's).
The checkpoints, calibrations and τ* are pinned by sha256 in `reports/v12_freeze.json`.

- **v1.2's model.**
  - **Candidates:** row 9 (no context) and row 21 (row 9 with the seasonal radar context). Rows 20 and 22 also
    compete if the AlphaEarth sidecar exists for every stack by the selection date.
  - **Selection:** the candidate with the best pooled-CV macro IoU at τ* over the 71 groups wins. A tie within 0.005
    goes to the simpler model (row 9).
  - **Inference add-ons:** each is adopted only if it raises the pooled-CV macro IoU by ≥ 0.005 and helps a
    majority of groups, judged on whole test stacks:
    - per-scene EM self-calibration;
    - HAND-gated post-processing.
- **TerraMind-fair.**
  - **Design:** TerraMind with a siamese pre/post encoder and its native DEM modality (rows 24–26).
  - **Size and adaptation:** base + LoRA. If a one-epoch probe projects more than 3 h per pooled CV run: base frozen;
    beyond that, small + LoRA (the rule logged on 2026-10-13).
  - **Tuning budget:** equal for both families, at most 3 learning rates on an inner split of fold 0's training
    groups.
- **GFM:** the ensemble flood extent of the post-event frame, as served by EODC, whatever its version
  (`processing:version` is recorded).
- **v1.1's best:** `runs/v11/kurosiwo/all/all/row09_fusion_crossattn/seed4{2,3,4}`. Its v1.1 code path is pinned by
  `tests/test_golden_v11.py`.

## 4. Confirmatory tests

| Test | Hypothesis | A | B | Pairs |
|---|---|---|---|---|
| P5 | H5 | TerraMind-fair | v1.2's model | primary stratum |
| P6 | H6 | GFM | v1.2's model | primary stratum, units GFM covers on ≥ 95 % of scored pixels |
| P7 | H7 | v1.1's best | v1.2's model | primary stratum |

- **Metric:** macro IoU at each model's own τ*, seeds pooled per unit (v1.1's definition). The gap is B − A per
  unit.
- **Test:** an exact two-sided sign-flip test (Monte Carlo with 200 000 patterns above 20 units). Holm-adjusted
  across P5–P7 within each unit definition. With fewer than 8 units, the result is "no verdict".
- **"Supported":** significant and positive at **both** unit definitions.
- **P5 needs more:** the same must also hold for AP, computed per unit from the score histograms. A win that holds at
  τ* but not threshold-free is reported as "not supported".

## 5. Stopping, sealing and descriptive analyses

- **Stopping rule.** The tests are computed once: when 20 activations in 15 clusters of the primary stratum are
  scored, or on 2027-12-31, whichever comes first.
  - **Power** (v1.1's spread, SD ≈ 0.077): 20 activations detect about 0.06 IoU at 80 % power; 10 would detect about
    0.10.
- **Sealing.** Until the stopping rule is met, scores live outside the repository. The committed ledger
  (`reports/prospective/ledger.jsonl`) holds each sealed file's sha256, verified at the reveal. The Arena shows counts
  only.
- **Freeze check.** A sweep refuses to run if a checkpoint, the test spec, the census rule or the code paths differ
  from the freeze.
- **Descriptive only:**
  - self-calibrated and oracle IoU;
  - the strata drawn on optical and mixed imagery;
  - GFM by `processing:version`;
  - per-activation tables;
  - the CEMS 2023–26 benchmark scored once after the freeze (seen, exploratory).

## Amendment 1 — the models, fixed at the freeze (2026-10-14)

Written after cross-validation and before the freeze tag `v1.2.0-freeze`, by the rules of §3 and
`reports/v1_2_deviations.md`. No flood of the blind set had been scored.

- **v1.2's model: row 21**, the reference fusion model with the seasonal radar context
  (`configs/variants/row21_fusion_context_seasonal.yaml`).
  - **Selection:** the CV rule chose it with a whole-stack pooled macro IoU at τ* of 0.380 over 79 groups, against
    0.361 for row 9 (`reports/v12_selection.json`).
  - **Learning rate:** 3e-4, inherited from row 9's tuning.
  - **No add-on:** self-calibration lowered the CV IoU (−0.042); the HAND cut moved it by +0.001 and helped fewer
    than half the groups (`reports/v12_addons_row21_fusion_context_seasonal.json`).
  - **Operator in P5–P7:** IoU at τ*.
- **The fair TerraMind: row 24**, TerraMind-base with LoRA (rank 8), siamese pre/post with its DEM input. The timing
  rule chose it (`reports/v12_tm_probe.json`); learning rate 1e-4 by the equal tuning budget
  (`reports/v12_tuning.json`).
- **Training:** both use the all-groups split of the pooled set (`crossval.all_groups_split`, as in v1.1), with
  seeds 42, 43 and 44 (`runs/v12/final`).
  - **Train:** four of the five folds, 63 groups, 5,605 chips.
  - **Validate:** the fifth fold's other 17 groups (617 chips) set early stopping, the calibration and the
    deployment prior.
  - **Unscored:** its smallest group (2 chips) fills the test slot the pipeline requires; that score is never
    reported.
- **What P5 compares:** system against system. v1.2's model reads terrain (HAND, slope, incidence) and the seasonal
  reference; TerraMind reads its native modalities (pre/post Sentinel-1, Sentinel-2 when usable, DEM). The CV
  comparison of row 9 (no seasonal context) with TerraMind is reported beside P5 as a descriptive companion.
- **v1.1's best and GFM:** as in §3.
- **Pinning:** checkpoints, calibrations and τ* by sha256 in `reports/v12_freeze.json`. The spec and its tests are in
  `configs/datasets/prospective_v12.yaml`. The analysis code is `floodfusion.blind_analysis`.

**What the CV already suggests (exploratory, single seed, chip level, `reports/v12_pool_cv.md`):**

| | IoU at τ* | Oracle τ | AP |
|---|---|---|---|
| Row 21 | 0.394 | 0.388 | 0.651 |
| TerraMind-base + LoRA | 0.313 | 0.366 | 0.617 |

Most of the τ* gap is threshold transfer, which is why P5 also requires AP. These numbers are not a test of H5; the
blind floods are.

# floodfusion v1.3 — pre-registered protocol (the adaptation ladder, the reference ceiling, and the blind H8)

**Committed:** 2026-11-10, before any v1.3 model is trained, before any v1.3 number on the CEMS benchmark is computed
and before the reference-ceiling analysis is run. Every later change goes to `reports/v1_3_deviations.md` before the
result it could affect. v1.2's protocol (`reports/protocol_v1_2.md`), freeze, ledger and analysis are not changed.

**What was known when this was written.** v1.3 was designed after v1.2's results on the seen CEMS benchmark: row 21
scores 0.417 macro IoU at τ* on 38 activations, 0.468 at its oracle threshold (`reports/v12_cems_rebuilt.md`). Those
numbers motivated the ladder below. That is why every v1.3 number on this benchmark is exploratory, and only H8, on
floods no model has seen, is confirmatory.

## 1. Questions

- **Q1, the adaptation ladder (exploratory; it selects v1.3's model).** Does adapting v1.2's model to Copernicus EMS's
  own labels improve it on CEMS floods it was not adapted on, and is recalibration enough, or does fine-tuning add
  to it?
- **Q2, the reference ceiling (descriptive).** How well do two Copernicus EMS maps of the same area, made at most 24 h
  apart, agree with each other?
- **H8 (confirmatory, blind).** v1.3's model beats v1.2's model (row 21) on the blind floods.

## 2. Data

### 2.1 The ladder (seen; exploratory)

- **Pairs:** every pair of the CEMS 2023–26 benchmark except EMSR664, on the stacks rebuilt with the DEM fix
  (`data/cems_benchmark_v12`): 89 pairs from 43 activations, 30,834 chips.
  - **Testing and selection** use the primary stratum only (products drawn on radar satellites): 75 pairs from 38
    activations, 532 tiles, 27,501 chips.
  - **Training and validation** use every stratum of their groups: the primary stratum plus the optical (13 pairs)
    and mixed (1) strata, whose labels were drawn on optical images (the author's choice, 2026-11-10: more training
    data).
- **Labels and pixels:** the benchmark's own. A pixel trains and scores only where its validity byte is 0: inside the
  product's observed region, outside JRC permanent water (occurrence ≥ 80 %), and not no-data, layover or shadow. The
  label is the product's observed flood. Training and scoring therefore use one pixel definition
  (`cems_benchmark.prepare_pair`, `cems_scores._tile_masks`).
- **Groups:** activations of every stratum whose footprints lie within 100 km form one group, whatever their dates
  (single linkage, `kurosiwo.event_groups`). There are 34 groups, 30 of them holding radar-drawn pairs. The largest
  holds 12.4 % of the chips, below v1.2's 25 % rule. A place mapped twice can never train a model and test it, whichever
  stratum its map belongs to.
- **Folds:** grouped five-fold (`crossval.k_fold_groups`). Fold i tests its groups, validates on fold i+1's groups
  (early stopping, calibration, deployment prior) and trains on the rest.

### 2.2 The blind set (H8)

v1.2's blind set, unchanged (`protocol_v1_2.md` §2.2): census, eligibility, the primary stratum, the units. v1.3's
scorer only reads pairs that v1.2's runner has prepared; it never fetches or prepares one.

**Separation rule.** v1.3's development never reads `data/prospective/` (where v1.2's runner writes the blind floods'
stacks, labels included) or any sealed score. v1.3's scorer runs only from a worktree at the tag `v1.3.0-freeze`, after
the freeze.

### 2.3 The reference ceiling (seen; descriptive)

- **Set:** seen flood activations EMSR656–898 (records cached in `data/cems_records/`). Every AOI with two delineation
  (DEL) products whose latest source images are at most 24 h apart, taken as consecutive products in time. The
  metadata probe of 2026-11-10 counts 111 such product pairs in 45 activations.
- No code from EMSR899 on is read.

## 3. The ladder

Each rung is scored on the same 38 activations. Each activation is scored by the fold that held out its group. All
rungs use seed 42; every count is pooled over an activation's pairs.

| Rung | Model | Threshold |
| --- | --- | --- |
| R1, zero-shot | row 21, seed 42 (`runs/v12/final/all/all/row21_fusion_context_seasonal/seed42`) | its own τ* |
| R2, recalibration | R1's logits, unchanged | fitted on fold i's other groups (below) |
| R3, fine-tuning | row 21 seed 42's weights, trained further on fold i's training groups (every stratum) | its own τ*, from fold i's validation groups (every stratum) |

- **R2's threshold:** the cost-optimal threshold, argmin r·FN + FP with r = 3 (the cost ratio that defines τ* =
  1/(1 + r)), on the pooled pixel counts of every radar-drawn pair outside fold i
  (`crossfit.crossfit_cost_threshold`). The counts come from the stored logit histograms
  (`reports/v12_cems_rebuilt_scores.json.gz`, 256 bins over [−10, 10]), which hold the radar-drawn pairs.
- **R3's recipe:** row 21's loss, sampler, augmentation and schedule (the Kuro Siwo overlay: 30 epochs, 3 warm-up,
  patience 8) and row 21's normalisation statistics. Calibration, the deployment prior and τ* are fitted on fold i's
  validation groups, as for every model in this repository.
  - **Learning rate:** one of 1e-5, 3e-5 and 1e-4, chosen by v1.2's tuning rule (`floodfusion.tuning`) on an inner
    split of fold 0's training groups.
  - **Prediction:** each held-out pair is predicted whole, as the scorer does, and its counts are binned like R1's.
- **The measure:** macro IoU over the 38 activations, from binned counts for every rung.
  - **Code check:** R1 computed with all three seeds reproduces the committed 0.418 ("IoU at τ*, binned" in
    `reports/v12_cems_rebuilt.md`).
- **The adoption rule:**
  - R2 is adopted over R1 if it raises the measure by ≥ 0.005 **and** improves more than half of the 38 activations
    (a tie does not count as an improvement).
  - R3 is adopted over the highest rung adopted below it by the same rule.
  - v1.3's model is the highest adopted rung. If neither R2 nor R3 is adopted, there is no v1.3 model: the null
    result is reported and H8 is not run.
- **The reporting rule:** every v1.3 number on these 38 activations is the cross-validated one, never an all-groups
  model scored on them.
- **Descriptive only:** per cluster (30 clusters: 100 km and 60 days, as v1.1); AP; the oracle threshold; R1 and R2
  with three seeds; the optical-stratum pairs scored by every rung's fold models.
- **Optional, descriptive, capped at 8 GPU-hours:** the fair TerraMind (row 24) through R1–R3, to see whether a gain
  depends on the network.

**v1.3's final model.**
- **R2 adopted:** row 21's three seeds, each with the threshold fitted by R2's rule on all 38 activations.
- **R3 adopted:** row 21's recipe from §3 on the all-groups split (`crossval.all_groups_split`, as v1.2's finals), seeds
  42, 43 and 44, each initialised from row 21's matching seed.

The checkpoints, calibrations and thresholds are pinned by sha256 in `reports/v13_freeze.json` at the tag
`v1.3.0-freeze`.

## 4. H8 — the confirmatory test

| Test | Hypothesis | A | B | Pairs |
| --- | --- | --- | --- | --- |
| P8 | H8 | v1.2's model (row 21) | v1.3's model | v1.2's primary stratum, analysable units |

- **Metric and test:** as `protocol_v1_2.md` §4. Macro IoU at each model's own threshold, seeds pooled per unit; the
  gap is B − A; an exact two-sided sign-flip test; "supported" means significant and positive at both unit
  definitions (activations, clusters); fewer than 8 units gives "no verdict".
- **Its own family:** P8 was registered after P5–P7 and cannot join their frozen Holm family of three. It is tested
  alone at α = 0.05; P5–P7 are unchanged.
- **Units and timing:** v1.2's stopping rule (20 activations in 15 clusters, or 2027-12-31), counted over the units
  both models score.
  - Blind floods scored by v1.2 before the v1.3 freeze are included: v1.3 scores them from their prepared pairs after
    its own freeze.
  - Each blind flood is still scored only by frozen models, and the separation rule (§2.2) keeps them out of v1.3's
    development.
- **Power:** computed at the freeze from the ladder's per-activation paired differences (the adopted rung against R1),
  by `scripts/v12_power.py`'s method, and stated in an amendment before the first v1.3 blind score.
- **Sealing:** v1.3's scores live in `data/prospective_v13/sealed/`, outside the repository. Their sha256 go into
  `reports/prospective/ledger_v13.jsonl`, which uses v1.2's hash-chained ledger format. The v1.2 ledger is never
  written by v1.3.

## 5. The reference ceiling (rules fixed before any number)

- **Grid:** 10 m on the UTM zone of the AOI.
- **Region:** the intersection of the two products' observed regions (`case_study.scored_region`).
- **What is compared:** each product's observed flood layer.
- **Metric:**
  - per product pair, the IoU between the two maps;
  - per activation, the IoU of its pooled counts;
  - the macro mean over activations.
- **Strata:**
  - the time between the two products' latest source images (< 6 h, 6–24 h);
  - the sensor pair (radar–radar, radar–optical, optical–optical);
  - whether the two products share a source image.
- **Reading, fixed now:** this is set against the models' 0.42 on radar-drawn pairs.
  - If two Copernicus maps agree with each other at about 0.5, the benchmark is near its ceiling, and the models' gap
    is mostly reference disagreement.
  - If they agree at about 0.8, most of the gap is the models'.
  - It is not a test and has no p-value.
- **Caveat, stated now:** two maps hours apart also differ because the flood moved. The ceiling includes real change,
  as the benchmark's ±24 h tolerance between a product and its Sentinel-1 pass does.

## 6. The live service and blind floods

- **What the service may read:** only an activation's code, name, category, event date and AOI extents. Never a
  Copernicus product.
- **What it shows of a blind flood:** the model's map alone, with no Copernicus outline and no score.
- **From when:** only after `v1.3.0-freeze`, or after the ladder has decided that there is no v1.3 model, and only on
  the author's go. Until then the service refuses codes from EMSR899 on, and a test enforces it.

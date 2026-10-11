# floodfusion v1.1 — changes after the pre-registration

The protocol is `reports/protocol_v1_1.md`, committed in `46b428e` before any v1.1 model existed. Every later change to
code, data rules or runs is listed here with its reason, before the report it affects. **No v1.1 model had been trained
or scored when any entry below was made**, unless the entry says otherwise.

## Implementation changes that leave the protocol unchanged

| Date (UTC) | Change | Reason |
|---|---|---|
| 2026-10-10 | `crossval`: the v1.1 variants (9b, 9c, 9r) train one seed and only in cross-validation, never on the published split or the all-groups protocol. | Section 3 of the protocol already lists them under cross-validation only, at seed 42. This makes the driver enforce it. |
| 2026-10-10 | `case_study._baseline` applies a baseline run with that run's own configuration. | Bug: it always loaded row 2's configuration, so variant 2s would have been scored as plain row 2. |
| 2026-10-10 | `crossval run --fold` accepts fold names across several protocols in one call. | Needed for the smoke run of one fold per protocol. Nothing else changes. |
| 2026-10-10 | `data.seasonal`: VH is optional (VV decides; VH is written when ≥ 3 acquisitions have it). | A 2015 acquisition over India is single-polarisation (1SSV), which made 5 stacks fail. The baseline's change term uses VV only. |
| 2026-10-10 | Version 1.1.0 in `pyproject.toml`, `uv.lock` (whose stale 0.1.0 entry is fixed) and the package. | Set before the freeze, so the release changes no code path. |
| 2026-10-11 | `crossval.all_groups_split` (first fix: alternate groups of fold 4). | Superseded the same day by Amendment 1 (A1.7), before any run: that rule left the validation set 2.5 % optical. |
| 2026-10-11 | `kurosiwo.relabel_water` skips stacks that produced no chips. | One stack with no passing window has no `basins.gpkg`. Nothing to relabel. |
| 2026-10-11 | Security hardening from the review: OGR's VRT driver is never loaded (`OGR_SKIP`); CEMS package members must match their type; 401/403 is asked again twice before an activation counts as not public; activation codes are validated before they name a path; caches and resume markers are written atomically; the annotation tarball's root is anchored to the pinned commit; no nested rate-limit retries; one more log line redacted. | No effect on any data already selected or ingested. The 10 refused codes were re-checked by hand (see below). The committed census file is unchanged. |

| 2026-10-11 | `data.seasonal` skips a year-earlier acquisition with no valid pixel over the grid, and counts only acquisitions with data towards the minimum of 3. | One CEMS tile (EMSR838 AOI08 t00_02) failed, because a frame touching its footprint held no data over it. Every other reference is unchanged. The new freeze SHA is set before any v1.1 run. |
| 2026-10-11 | `scripts/v11_queue.sh` counts towards the 12 GPU-h Kuro Siwo cap only runs that trained, and skips the CEMS step when its scores exist from a clean tree with the same code paths. | Each row 11 record is an evaluation of row 9's checkpoint and copies its `train_seconds` (`"trained": false`), so the old count read 8.65 h when 5.37 h had been trained. At the cap check it would have dropped the water-label cross-validation early. The fix reads only the run records, and was made before any v1.1 result was read. The queue was stopped and restarted: the first learned run of the India LOEO fold, about 7 minutes in, restarts from scratch. No code path changed, so runs before and after the restart are equivalent. |

## Protocol amendment

| Date (UTC) | Change | Reason |
|---|---|---|
| 2026-10-11 | **Amendment 1** (`reports/protocol_v1_1.md`, A1.1–A1.11): SAR-only calibration for windows without optical, two unit definitions, an exact one-sided P2, Monte Carlo p-values, descriptive subsets and sensitivity analyses, P3 worded as a comparison of recipes, the Kuro Siwo all-groups validation set, run order and cap, a fail-closed seasonal reference, cross-fit caveats, TerraMind's pre-training period. | Two design reviews of the implementation, before any v1.1 model was trained or scored. |

## Observations that affect what can be reported

| Date (UTC) | Observation |
|---|---|
| 2026-10-10 | **Seasonal reference coverage, Sen1Floods11:** India (2016) has no seasonal reference. Its year-earlier window falls in 2015, with fewer than 3 same-orbit acquisitions. India's chips therefore keep the recent reference in row 2s, as the protocol provides. Other events: see `seasonal.json` beside each stack. |
| 2026-10-10 | **CEMS census, non-public codes:** 10 codes returned HTTP 403 ("Authentication credentials were not provided"): EMSR658, 700, 723, 724, 840, 869, 876, 879, 880, 891. Re-checked one by one, 3 s apart, after the census: all still 403. They are non-public activations, not rate limiting, so the census stands. |
| 2026-10-11 | **Kuro Siwo yield:** 36 of the 40 dated events produced chips (3,971 windows, 29 groups). Excluded by the ingest's own rules: EMSR1111013 USA and EMSR130 Myanmar (no same-orbit pre-event pass within 60 days in the archive); EMSR502 Ireland and EMSR520 Netherlands (no pass on the labelled date covers ≥ 10 % of any AOI). The flood and water chip sets were verified identical in windows, validity, groups and optical availability; the water label raises the mean flood share from 0.095 to 0.114. |
| 2026-10-11 | **CEMS benchmark yield:** all 90 pairs left by the caps (≤ 4 AOIs per activation) were prepared. |
| 2026-10-11 | **Dry runs, disclosed.** Before the freeze, two throwaway runs checked that the new code paths do not crash. (1) `runs/v11_dryrun`: one Sen1Floods11 fold (rows 12, TerraMind, 9b, 9c, 9r, 2e, 2s) and one Kuro Siwo fold and all-groups model (rows 9, 2, 2s), seed 42, at a pre-freeze SHA. (2) The CEMS scorer on two pairs with the **v1.0** models, on CPU. Only completion and log lines were meant to be read. **Correction (2026-10-11 02:05 UTC):** a progress check printed the evaluator's log lines, which carry the held-out IoU of three dry-run models on the Bolivia fold: row 12, TerraMind and 9b, at seed 42 with pre-freeze code. They were seen after Amendment 1 was committed, and nothing in the protocol or code changes because of them. They are not reported. The CEMS dry run's IoUs (v1.0 models) were not read. None of these runs enters any v1.1 report. |
| 2026-10-11 | **Census recheck:** the eligibility census re-run from scratch (fresh verdicts) with the current code gives the identical 117 pairs. Of the 243 codes, 233 are public and 10 restricted (see above). |
| 2026-10-11 | **First v1.1 result read:** the CEMS report (P1–P4), at 22:15 UTC, after commit 7ece81d. Its scores were computed at 1ec2d15; no rule, cap, model or subset has changed since the freeze, and the queue continues unchanged. |

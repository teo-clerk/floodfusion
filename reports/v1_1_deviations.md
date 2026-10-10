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
| 2026-10-11 | `crossval.all_groups_split`: the last fold's groups alternate between validation (calibration, early stopping) and a test set that is never reported. Training stays on folds 0–3, as pre-registered. | The protocol's "validate on fold 4", with test = validation, fails the split loader's disjointness check (`load_splits`). Found by review before any run. |
| 2026-10-11 | `kurosiwo.relabel_water` skips stacks that produced no chips. | One stack with no passing window has no `basins.gpkg`. Nothing to relabel. |
| 2026-10-11 | Security hardening from the review: OGR's VRT driver is never loaded (`OGR_SKIP`); CEMS package members must match their type; 401/403 is asked again twice before an activation counts as not public; activation codes are validated before they name a path; caches and resume markers are written atomically; the annotation tarball's root is anchored to the pinned commit; no nested rate-limit retries; one more log line redacted. | No effect on any data already selected or ingested. The 10 refused codes were re-checked by hand (see below). The committed census file is unchanged. |

## Observations that affect what can be reported

| Date (UTC) | Observation |
|---|---|
| 2026-10-10 | **Seasonal reference coverage, Sen1Floods11:** India (2016) has no seasonal reference. Its year-earlier window falls in 2015, with fewer than 3 same-orbit acquisitions. India's chips therefore keep the recent reference in row 2s, as the protocol provides. Other events: see `seasonal.json` beside each stack. |
| 2026-10-10 | **CEMS census, non-public codes:** 10 codes returned HTTP 403 ("Authentication credentials were not provided"): EMSR658, 700, 723, 724, 840, 869, 876, 879, 880, 891. Re-checked one by one, 3 s apart, after the census: all still 403. They are non-public activations, not rate limiting, so the census stands. |
| 2026-10-11 | **Kuro Siwo yield:** 36 of the 40 dated events produced chips (3,971 windows, 29 groups). Excluded by the ingest's own rules: EMSR1111013 USA and EMSR130 Myanmar (no same-orbit pre-event pass within 60 days in the archive); EMSR502 Ireland and EMSR520 Netherlands (no pass on the labelled date covers ≥ 10 % of any AOI). The flood and water chip sets were verified identical in windows, validity, groups and optical availability; the water label raises the mean flood share from 0.095 to 0.114. |
| 2026-10-11 | **CEMS benchmark yield:** all 90 pairs left by the caps (≤ 4 AOIs per activation) were prepared. |

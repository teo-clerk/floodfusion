# IGARSS 2027 paper (draft)

A 4-page full paper for IGARSS 2027 (Reykjavík, 11–16 July 2027). Submission deadline: 11 January 2027;
notifications: 12 March 2027 (https://2027.ieeeigarss.org/important_dates.php, read on 2026-11-11).

- **Sources.** Every number comes from a committed report, named in a `% src:` comment beside it, as in
  `paper/paper_v12.tex`. `% TODO(r3)` marks the ladder's rung 3, filled from `reports/v13_ladder.json`.
- **Figure 1** is exploratory: the per-activation association between the model's score and Copernicus's
  self-agreement was looked at after both were known (`reports/v1_3_deviations.md`).
- **Template.** IEEEtran (conference) until the portal's paper kit is available; then the kit's template.

## Build

```bash
uv run --frozen python paper/igarss27/make_figures.py      # figures/ from reports/ (committed)
cd paper/igarss27 && tectonic main.tex                     # main.pdf (not committed)
```

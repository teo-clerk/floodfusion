# The preprints

- `paper_v13.tex` is the v1.3 preprint, meant for EarthArXiv: *Labels, Thresholds and the Reference Itself: Evaluating
  Sentinel-1 Flood Segmentation Against Copernicus EMS Maps, with a Preregistered Blind Test*. It extends the v1.2
  paper with the reference ceiling, the adaptation ladder, the live service and the blind test's status.
- `paper_v12.tex` is the v1.2 preprint: *Labels, Thresholds and a Blind Test: Evaluating Sentinel-1 Flood Segmentation
  Against Independent Copernicus EMS Maps*. It stays as published.
- `igarss27/` is the 4-page IGARSS 2027 paper (its own README).

**Sourced numbers.**
- Every number in them comes from a committed report under `reports/`, named in a `% src:` comment beside it.
- Differences marked `% derived:` subtract two sourced numbers.
- The figures are drawn from the same reports by `make_figures.py` (v1.2's) and `igarss27/make_figures.py` (the
  ceiling and the ladder, which `paper_v13.tex` reuses). Both print every number they plot, and the second also
  saves them as JSON beside its figures.
- `TODO(status)` comments in `paper_v13.tex` mark what is updated at submission: the ledger's counts and Zenodo's
  DOIs.

**The extended abstract.** `abstract_v12.tex` condenses the v1.2 paper onto one page, with the same preamble and
layout. Every number in it appears in `paper_v12.tex`, and its `% src:` comments name the paper section each comes
from. Its PDF is not committed.

**Status.** The results on the CEMS 2023–26 benchmark are exploratory. The confirmatory test is the blind one in
`reports/protocol_v1_2.md`.

## Build

```bash
uv run --frozen python paper/make_figures.py           # figures/*.png from reports/ and assets/ (they are committed)
uv run --frozen python paper/igarss27/make_figures.py  # igarss27/figures/*.{png,json} (committed)
cd paper && tectonic paper_v13.tex                     # or, with TeX Live: latexmk -pdf paper_v13.tex
tectonic paper_v12.tex
tectonic abstract_v12.tex                              # the one-page extended abstract (needs no figures or bibliography)
```

[Tectonic](https://tectonic-typesetting.github.io) downloads the LaTeX packages it needs on first use. The source
needs newtx, TikZ, natbib and xurl.

`*.pdf` is gitignored repository-wide; `paper/.gitignore` makes one exception, `paper_v12.pdf`, which the top-level
README links to. Rebuild it and commit it together with any change to its source. `paper_v13.pdf` is not committed.

## Notices

Figures 3 and 4 of both papers, and Figures 5 and 6 of `paper_v13.tex`, contain modified Copernicus Sentinel data and
modified Copernicus Emergency Management Service information (2023–2026), © European Union. Keep these notices when
reusing the figures.

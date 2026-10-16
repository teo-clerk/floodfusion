# The v1.2 preprint

`paper_v12.tex` is the paper on v1.2: *Labels, Thresholds and a Blind Test: Evaluating Sentinel-1 Flood
Segmentation Against Independent Copernicus EMS Maps*.

**Sourced numbers.**
- Every number in it comes from a committed report under `reports/`, named in a `% src:` comment beside it.
- Differences marked `% derived:` subtract two sourced numbers.
- The figures are drawn from the same reports by `make_figures.py`, which prints every number it plots.

**The extended abstract.** `abstract_v12.tex` condenses the paper onto one page, with the same preamble and layout.
Every number in it appears in `paper_v12.tex`, and its `% src:` comments name the paper section each comes from. Its
PDF is not committed.

**Status.** The results on the CEMS 2023–26 benchmark are exploratory. The confirmatory test is the blind one in
`reports/protocol_v1_2.md`.

## Build

```bash
uv run --frozen python paper/make_figures.py   # figures/*.png from reports/ and assets/ (they are committed)
cd paper && tectonic paper_v12.tex             # or, with TeX Live: latexmk -pdf paper_v12.tex
tectonic abstract_v12.tex                      # the one-page extended abstract (needs no figures or bibliography)
```

[Tectonic](https://tectonic-typesetting.github.io) downloads the LaTeX packages it needs on first use. The source
needs newtx, TikZ, natbib and xurl.

`*.pdf` is gitignored repository-wide; `paper/.gitignore` makes one exception, `paper_v12.pdf`, which the top-level
README links to. Rebuild it and commit it together with any change to the source.

## Notices

Figures 3 and 4 contain modified Copernicus Sentinel data and modified Copernicus Emergency Management Service
information (2023–2026), © European Union. Keep these notices when reusing the figures.

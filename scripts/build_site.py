"""Build the interactive explainer in ``site/`` into a static folder that GitHub Pages can serve.

The results on the page are read here from the committed reports (``reports/*.json`` and the tables of
the generated ``reports/*.md``) instead of being copied into it::

    uv run python scripts/build_site.py              # -> _site/index.html (open it in a browser)
    uv run python scripts/build_site.py --out public

The Forlì comparison slider uses the three map panels of ``assets/forli_hero.png``, cut out here.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE_FILES = ("index.html", "style.css", "app.js", "arena.html", "arena.js")
FIGURES = ("gate_map.png",)
HERO_PANELS = ("forli_vv.png", "forli_fusion.png", "forli_cems.png")
EVENT_PREFIX = "S1F11_"

#: Display names for the Sen1Floods11 rows, in display order; every number comes from the report.
S1F11_ROWS = {
    "1": "Otsu threshold on VV",
    "2": "Physics baseline",
    "3": "Physics baseline + MNDWI",
    "4": "U-Net, SAR only",
    "9": "Cross-attention fusion",
    "11": "Fusion, optical removed",
    "cmp_terramind_small": "TerraMind-1.0-small",
}

#: The Kuro Siwo rows shown (v1.1), in display order; every number comes from the report.
KS_ROWS = {
    "2": "Physics baseline",
    "row02s_seasonal": "Physics baseline, seasonal reference",
    "4": "U-Net, SAR only",
    "9": "Cross-attention fusion",
    "11": "Fusion, optical removed",
}
#: The CEMS benchmark's models grouped by what they learned from, in display order, with short labels for the
#: chart; every number comes from the report.
CEMS_GROUPS = (
    ("Trained on Sen1Floods11", {"s1f11_r4": "U-Net, SAR only", "s1f11_r9": "Fusion",
                                 "s1f11_r11": "Fusion, optical removed", "s1f11_tm": "TerraMind-small"}),
    ("Trained on Kuro Siwo", {"ks_r4": "U-Net, SAR only", "ks_r9_flood": "Fusion",
                              "ks_r11_flood": "Fusion, optical removed", "ks_r9_water": "Fusion, water labels"}),
    ("No learning", {"r2": "Physics baseline", "r2s": "Physics, seasonal ref."}),
)
CEMS_PRIMARY = "Drawn on radar satellites"   # the title prefix of the primary stratum in cems_benchmark_strata.json
CEMS_SUBSETS_USED = ("all", "partial")         # the page's prose reads these two subsets
_HYPOTHESIS = re.compile(r"\s*\((?:[^()]*,\s*)?(H\d)\)")  # "(H2)", "(R2 in practice, H3)"

_BACKGROUND = 720  # an RGB pixel whose channels sum to this or more is the figure's white background
_PANEL_FILL = 0.9  # a column or row belongs to a map panel when this share of it is not background
_SEPARATOR = re.compile(r"^\|\s*:?-+:?\s*\|")  # a GFM delimiter row: | --- | :-- | --: |
_PASS = re.compile(r"^S1_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})_o(\d+)$")
_SHIFT = re.compile(r"\[(-?\d+),\s*(-?\d+)\]\s*\((\w+)\)")
# "(R3)", "(9 − 11)", "(docs/09 5.2)": the design spec's codes and citations, not prose for the page.
_ROW_CODE = re.compile(r"\s*\((?:R\d+|[\w ]+ − [\w ]+|docs/\d+[^)]*)\)")
_EMPHASIS = re.compile(r"\*(\w[^*]*)\*")
_PANEL_TOLERANCE_PX = 1


# --- Markdown tables ------------------------------------------------------------------------------

def _plain(cell: str) -> str:
    return cell.replace("**", "").replace("`", "").strip()


def _cells(line: str) -> list[str]:
    return [_plain(cell) for cell in line.strip().strip("|").split("|")]


def md_tables(text: str) -> list[tuple[str, list[dict[str, str]]]]:
    """Every pipe table in a Markdown report, with the heading above it (``""`` before the first)."""
    tables: list[tuple[str, list[dict[str, str]]]] = []
    heading, lines, i = "", text.splitlines(), 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        elif line.startswith("|") and i + 1 < len(lines) and _SEPARATOR.match(lines[i + 1].strip()):
            header, rows = _cells(line), []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(dict(zip(header, _cells(lines[i]), strict=False)))
                i += 1
            tables.append((heading, rows))
            continue
        i += 1
    return tables


def table(tables: Sequence[tuple[str, list[dict[str, str]]]], heading: str) -> list[dict[str, str]]:
    """The first table under a heading that starts with ``heading``."""
    for caption, rows in tables:
        if caption.startswith(heading):
            return rows
    raise KeyError(f"no table under a heading starting with {heading!r}")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _event(name: str) -> str:
    return name.removeprefix(EVENT_PREFIX).replace("-", " ")


def _number(cell: str) -> float:
    match = re.search(r"[-+]?\d[\d,]*\.?\d*", cell.replace("\u2212", "-"))
    if match is None:
        raise ValueError(f"no number in {cell!r}")
    return float(match.group().replace(",", ""))


# --- Report extracts --------------------------------------------------------------------------------

def _gaps(report: dict[str, Any], md: list[tuple[str, list[dict[str, str]]]], heading: str) -> list[dict[str, Any]]:
    questions = {row["Gap (B − A)"].replace(" − ", "-"): _EMPHASIS.sub(r"\1", _ROW_CODE.sub("", row["Question"])).strip()
                 for row in table(md, heading)}
    return [{"key": key, "question": questions.get(key, key), "delta": g["delta"], "ci": g["ci"],
             "wins": g["basins_b_better"], "n": g["n_basins"], "p_holm": g["p_holm"],
             "significant": g["resolvable"], "seeds": g["seeds"]}
            for key, g in report["gaps"].items()]


def sen1floods11(reports: Path) -> dict[str, Any]:
    """Leave-one-event-out rows, paired gaps, protocol gap, conformal coverage and the label checks."""
    report = json.loads(_read(reports / "sen1floods11_table.json"))
    md = md_tables(_read(reports / "sen1floods11_table.md"))
    order = list(S1F11_ROWS) + [key for key in report["rows"] if key not in S1F11_ROWS]
    rows = [{"key": key, "label": S1F11_ROWS.get(key, key), "iou": r["iou"], "ci": r["ci"], "ap": r["ap"],
             "oracle": r["oracle"], "seeds": r["seeds"],
             "per_event": {_event(event): iou for event, iou in r["per_event"].items()}}
            for key in order if (r := report["rows"].get(key)) is not None]
    protocol = [{"row": run.split("/")[0], "seed": run.split("/")[1].removeprefix("seed"),
                 "n": gap["iou_tau"]["n"],
                 **{measure: {"official": gap[measure]["official"], "loeo": gap[measure]["loeo"],
                              "delta": gap[measure]["delta"], "p_holm": gap[measure]["p_holm"]}
                    for measure in ("iou_tau", "iou_oracle", "ap")}}
                for run, gap in report["protocol_gap"].items()]
    conformal = [{"row": run.split("/")[0], "seed": run.split("/")[1].removeprefix("seed"), "alpha": c["alpha"],
                  "held": c["events_held"], "folds": c["folds"],
                  "fnr": {_event(event): v for event, v in c["per_event_mean_chip_fnr"].items()}}
                 for run, c in report["conformal"].items()]
    registration = []
    for row in table(md_tables(_read(reports / "sen1floods11_registration.md")), ""):
        moved = _SHIFT.search(row["Labels moved (source)"])
        if moved is None:
            raise ValueError(f"unreadable label shift for {row['Event']}: {row['Labels moved (source)']!r}")
        registration.append({"event": _event(row["Event"]), "dy": int(moved[1]), "dx": int(moved[2]),
                             "source": moved[3]})
    change_text = _read(reports / "sen1floods11_label_change.md")
    change = [{"event": _event(row["Event"]), "median_dvv_db": _number(row["Median ΔVV (dB)"]),
               "dark_before": _number(row["Dark before (VV < -15 dB)"]), "dark_after": _number(row["Dark after"])}
              for row in table(md_tables(change_text), "")]
    return {"events": [_event(event) for event in report["events"]], "rows": rows,
            "gaps": _gaps(report, md, "Paired gaps"), "protocol_gap": protocol, "conformal": conformal,
            "registration": registration, "label_change": change}


def _pass(tag: str, optical: str) -> dict[str, Any]:
    match = _PASS.match(tag)
    if match is None:
        raise ValueError(f"unexpected Sentinel-1 pass tag {tag!r}")
    year, month, day, hour, minute, orbit = match.groups()
    tiles = [int(part) for part in optical.split("/")]
    return {"tag": tag, "time": f"{year}-{month}-{day}T{hour}:{minute}", "orbit": int(orbit),
            "optical_tiles": tiles[0], "tiles": tiles[1]}


def _short(product: str) -> str:
    return product.split("_AOI01_")[-1]


def forli(reports: Path) -> dict[str, Any]:
    """EMSR664: CEMS products, Sentinel-1 passes, IoU per pass x product, and flooded areas."""
    report = json.loads(_read(reports / "emsr664_forli.json"))
    text = _read(reports / "emsr664_forli.md")
    md = md_tables(text)
    products = [{"tag": row["Product"], "sensor": row["Sensor"], "time": row["Acquired (UTC)"],
                 "region_km2": _number(row["Region observed"]), "flood_km2": _number(row["Flood in it"])}
                for row in table(md, "CEMS products")]
    optical = {row["Pass"]: row["Optical tiles"] for row in table(md, "IoU, pass × product, on the region common")}
    passes = [_pass(tag, optical[tag]) for tag in report["time_offset_h"]]
    iou = {region: {model: {tag: {_short(product): cell for product, cell in by_product.items()}
                            for tag, by_product in by_pass.items()}
                    for model, by_pass in report[region].items()}
           for region in ("common", "own")}
    area: dict[str, Any] = {"cems": {}, "models": {}}
    for key, km2 in report["area_km2"].items():
        if key.startswith("CEMS "):
            area["cems"][key.split()[1]] = km2
        else:
            model, tag = key.rsplit(", ", 1)
            area["models"].setdefault(model, {})[tag] = km2
    common = re.search(r"Common region: (\d+) km²", text)
    if common is None:
        raise ValueError("emsr664_forli.md does not state the common region")
    return {"activation": report["case"]["activation"], "products": products, "passes": passes,
            "models": list(report["common"]), "iou": iou, "area_km2": area,
            "offset_h": {tag: {_short(p): h for p, h in by_product.items()}
                         for tag, by_product in report["time_offset_h"].items()},
            "common_km2": int(common[1])}


def synthetic(reports: Path) -> dict[str, Any]:
    """The 13-row ablation, its gaps, the stress curves and the gate diagnostic."""
    report = json.loads(_read(reports / "ablation_table.json"))
    md = md_tables(_read(reports / "ablation_table.md"))
    labels = {row["#"]: row for row in table(md, "Ablation table")}
    # Rows outside the 13-row table (supplementary evaluations) keep their key as the label.
    rows = [{"key": key, "label": labels.get(key, {}).get("Configuration", key),
             "question": labels.get(key, {}).get("Purpose — the question the row answers", ""),
             "supplementary": key not in labels, "iou": r["iou"], "ci": r["ci"], "seeds": r["seeds"],
             "baseline": r["baseline"], "vram_gib": r["peak_vram_gib"], "ms_per_chip": r["latency_ms"],
             "device": r["latency_device"]}
            for key, r in report["rows"].items()]
    stress = json.loads(_read(reports / "stress.json"))
    gate_text = _read(reports / "gate_diagnostic.md")
    start = re.search(r"sigmoid\(bias_init\) = (\d+\.\d+)", gate_text)
    if start is None:
        raise ValueError("gate_diagnostic.md does not state the gate's starting value")
    gate = [{"cloud": _number(row["Cloud fraction"]), "clear": _number(row["Mean gate, clear"]),
             "masked": None if row["Mean gate, cloud-masked"] == "—" else _number(row["Mean gate, cloud-masked"]),
             "clear_open": _number(row["Clear pixels g > 0.5"])}
            for row in table(md_tables(gate_text), "")]
    return {"rows": rows, "gaps": _gaps(report, md, "Beneath the table"), "git": report["git"]["git_sha"],
            "stress": stress, "gate": {"start": float(start[1]), "by_cloud": gate}}


def _cems_test(by_unit: dict[str, dict[str, Any]], supported: str) -> dict[str, Any]:
    first = next(iter(by_unit.values()))
    hypothesis = _HYPOTHESIS.search(first["question"])
    return {"id": first["id"], "hypothesis": hypothesis[1] if hypothesis else "",
            "question": _HYPOTHESIS.sub("", first["question"]).strip(), "a": first["a"], "b": first["b"],
            "subset": first["subset"], "margin": first["margin"], "supported": supported,
            "units": {unit: {"k": t["k"], "wins": t["wins"], "delta": t["delta"], "ci": t["ci"], "p": t["p"],
                             "p_holm": t.get("holm_p"), "holds": t["holds"], "verdict": t["verdict"]}
                      for unit, t in by_unit.items()}}


def cems(reports: Path) -> dict[str, Any]:
    """The v1.1 benchmark: the pre-registered tests at both units, and every model per optical subset."""
    report = json.loads(_read(reports / "cems_benchmark.json"))
    supported = {row["Test"]: row["Supported (holds at both units)"]
                 for row in table(md_tables(_read(reports / "cems_benchmark.md")), "Pre-registered tests")}
    tests = {}
    for unit, results in report["tests"].items():
        for t in results:
            tests.setdefault(t["id"], {})[unit] = t
    strata = json.loads(_read(reports / "cems_benchmark_strata.json"))
    primary = next((v for title, v in strata.items() if title.startswith(CEMS_PRIMARY)), None)
    if primary is None:
        raise KeyError(f"cems_benchmark_strata.json has no stratum titled {CEMS_PRIMARY!r}...")
    names = {m["key"]: m["name"] for m in primary["all"]["models"]}
    missing = [key for _, models in CEMS_GROUPS for key in models if key not in names]
    missing += [f"subset {s}" for s in CEMS_SUBSETS_USED if s not in primary]
    if missing:
        raise ValueError(f"cems_benchmark_strata.json has no results for {missing}")
    subsets = {subset: {"k": len(v["units"]),
                        "models": {m["key"]: {"iou": m["iou"], "ci": m["ci"], "oracle": m["oracle"]}
                                   for m in v["models"]}}
               for subset, v in primary.items()}
    scores = json.loads(_read(reports / "cems_benchmark_scores.json"))
    activations = set(report["units"]["activation"])
    pairs = [p["pair"] for p in scores["pairs"].values() if p["pair"]["activation"] in activations
             and p["pair"]["stratum"] == "sar" and p["scored_px"]]
    dates = sorted(scores["activations"][a]["eventTime"][:7] for a in activations)
    return {"tests": [_cems_test(by_unit, supported[test_id]) for test_id, by_unit in tests.items()],
            "names": names, "groups": [{"label": label, "models": [{"key": k, "label": short} for k, short in models.items()]}
                                      for label, models in CEMS_GROUPS],
            "subsets": subsets, "n_activations": len(activations), "n_clusters": len(report["units"]["cluster"]),
            "n_pairs": len(pairs), "first_month": dates[0], "last_month": dates[-1]}


def kurosiwo(reports: Path) -> dict[str, Any]:
    """The v1.1 Kuro Siwo grouped five-fold table (flood labels): the main rows and the paired gaps."""
    report = json.loads(_read(reports / "v11_kurosiwo_table.json"))
    md = md_tables(_read(reports / "v11_kurosiwo_table.md"))
    missing = [key for key in KS_ROWS if key not in report["rows"]]
    if missing:
        raise ValueError(f"v11_kurosiwo_table.json has no rows {missing}")
    rows = [{"key": key, "label": label, "iou": report["rows"][key]["iou"], "ci": report["rows"][key]["ci"],
             "seeds": report["rows"][key]["seeds"]} for key, label in KS_ROWS.items()]
    return {"rows": rows, "gaps": _gaps(report, md, "Paired gaps"), "n_groups": len(report["events"])}


def replication(reports: Path) -> dict[str, Any]:
    """The v1.1 re-run of Sen1Floods11: the protocol gap at τ* per learned row and seed."""
    report = json.loads(_read(reports / "v11_sen1floods11_table.json"))
    return {"protocol_gap": [{"row": run.split("/")[0], "seed": run.split("/")[1].removeprefix("seed"),
                              "delta": gap["iou_tau"]["delta"]}
                             for run, gap in report["protocol_gap"].items()]}


def site_data(root: Path = ROOT) -> dict[str, Any]:
    """Everything the page shows, read from the committed reports."""
    reports = root / "reports"
    version = tomllib.loads(_read(root / "pyproject.toml"))["project"]["version"]
    return {"version": version, "sen1floods11": sen1floods11(reports), "forli": forli(reports),
            "synthetic": synthetic(reports), "cems": cems(reports), "kurosiwo": kurosiwo(reports),
            "replication": replication(reports)}


# --- Arena (v1.2) -----------------------------------------------------------------------------------
#: The v1.2 preview tables: (unit, restricted to units GFM covers) -> one table each.
ARENA_TABLES = (("activation", False), ("cluster", False), ("activation", True), ("cluster", True))
ARENA_METRICS = ("iou_tau", "iou_tau_binned", "iou_selfcal", "iou_oracle", "ap")
ARENA_MANIFEST = "arena-manifest.json"


def _finite(x: Any) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _arena_scoreboard(reports: Path) -> dict[str, Any]:
    """The v1.2 preview tables, or ``pending`` while ``reports/v12_cems_preview.json`` does not exist."""
    path = reports / "v12_cems_preview.json"
    if not path.is_file():
        return {"status": "pending"}
    report = json.loads(_read(path))
    tables = []
    for unit, gfm in ARENA_TABLES:
        table = next((t for t in report["tables"] if t["unit"] == unit and t["gfm_common"] is gfm), None)
        if table is None:
            raise KeyError(f"{path.name}: no table for unit={unit!r}, gfm_common={gfm}")
        models = {}
        for key, row in table["models"].items():
            missing = [m for m in ARENA_METRICS if m not in row]
            if missing:
                raise KeyError(f"{path.name}: {key} lacks {missing}")
            models[key] = {"name": row["name"], "k": row["k"], **{m: _finite(row[m]) for m in ARENA_METRICS},
                           "ci": {m: [_finite(v) for v in row["ci"][m]] for m in ARENA_METRICS}}
        tables.append({"unit": unit, "gfm_common": gfm, "units": table["units"], "models": models})
    return {"status": "ok", "tables": tables, "n_pairs": report["n_pairs"], "gfm_pairs": report["gfm_pairs"]}


def _arena_maps(assets: Path) -> dict[str, Any]:
    """The quicklook manifest with each layer's file, or ``pending``; a listed file that is missing fails loudly."""
    path = assets / ARENA_MANIFEST
    if not path.is_file():
        return {"status": "pending", "pairs": []}
    manifest = json.loads(_read(path))
    pairs = []
    for pair in manifest["pairs"]:
        layers = {}
        for key, layer in pair["layers"].items():
            name = f"arena-{pair['key']}-{key}.webp"
            if not (assets / name).is_file():
                raise FileNotFoundError(f"{path.name} lists {name}, which is not in {assets}")
            layers[key] = {"name": layer["name"], "iou": _finite(layer["iou"]), "src": f"assets/{name}"}
        pairs.append({**{k: pair[k] for k in ("key", "activation", "aoi", "product", "sensors", "product_time",
                                              "pass_time", "flood_km2")}, "layers": layers})
    return {"status": "ok", "pairs": pairs}


def _p4c(reports: Path) -> str | None:
    """Erratum 1's corrected P4 in one sentence, or None while its report does not exist."""
    path = reports / "v1_1_erratum_p4.json"
    if not path.is_file():
        return None
    tests = json.loads(_read(path))["tests"]
    parts = []
    for unit in ("activation", "cluster"):
        t = next(x for x in tests[unit] if x["id"] == "P4c")
        parts.append(f"per {unit} {t['delta']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}], fusion better on "
                     f"{t['wins']}/{t['k']}, p = {t['p']:.3f}")
    return "Corrected P4 (fusion − TerraMind): " + "; ".join(parts) + " (v1.1 as run: +0.031 and +0.039)."


def _errata(reports: Path) -> list[dict[str, Any]]:
    link = "https://github.com/teo-clerk/floodfusion/blob/main/reports/v1_1_erratum.md"
    p4c = _p4c(reports)
    return [
        {"id": "dropout", "title": "TerraMind trained without modality dropout",
         "summary": "Its config set modality dropout to 0.5, but the training loop applied it only to the fusion "
                    "models, so TerraMind's radar-only mode, used on most Copernicus floods, was barely trained. "
                    "Its decoder's BatchNorm also saw sub-batches of one to three chips.",
         "resolved": p4c is not None, "status": "Corrected re-run reported" if p4c else "Fixed · re-run in progress",
         "result": p4c, "link": link},
        {"id": "dem", "title": "Zero terrain past a DEM tile edge",
         "summary": "Copernicus GLO-30 declares no nodata, so past a 1° tile boundary the terrain was read as flat "
                    "ground at 0 m: slope and height above drainage were 0 on about 14 % of the CEMS benchmark's "
                    "area and 6–8 % of the training sets'. TerraMind reads no terrain.",
         "resolved": False, "status": "Fixed · data rebuilt for v1.2", "result": None, "link": link},
    ]


def _blind(reports: Path) -> dict[str, Any]:
    """The blind test's progress: counts and statuses only (scores stay sealed until the stopping rule)."""
    ledger = reports / "prospective" / "ledger.jsonl"
    if not ledger.is_file():
        return {"status": "not started", "status_text": "Not started: the models are frozen and the runner switched "
                "on at the end of v1.2.", "k_activations": 0, "k_clusters": 0, "floods": []}
    rows = [json.loads(line) for line in _read(ledger).splitlines() if line.strip()]
    floods = [{"code": r["activation"], "date": r["event_date"], "status": r["status"]} for r in rows]
    scored = {r["activation"] for r in rows if r["status"] == "scored"}
    clusters = {r["cluster"] for r in rows if r["status"] == "scored"}
    return {"status": "accruing", "status_text": f"Accruing: {len(scored)} activations scored, sealed.",
            "k_activations": len(scored), "k_clusters": len(clusters), "floods": floods}


def arena_data(root: Path = ROOT) -> dict[str, Any]:
    """Everything the Arena shows, read from the committed reports and quicklooks."""
    reports = root / "reports"
    version = tomllib.loads(_read(root / "pyproject.toml"))["project"]["version"]
    return {"version": version, "scoreboard": _arena_scoreboard(reports), "maps": _arena_maps(root / "assets"),
            "errata": _errata(reports), "blind": _blind(reports)}


# --- Figures ----------------------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges where a 1-D boolean mask is True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def hero_panels(hero: Path, out_dir: Path) -> list[Path]:
    """Cut the hero figure's three map panels (VV, fusion, CEMS) into same-sized images."""
    from PIL import Image

    image = Image.open(hero).convert("RGB")
    ink = np.asarray(image, dtype=np.int32).sum(axis=2) < _BACKGROUND
    height = ink.shape[0]
    spans = _runs(ink[height // 4: 3 * height // 4].mean(axis=0) > _PANEL_FILL)
    if len(spans) != len(HERO_PANELS):
        raise ValueError(f"{hero.name}: expected {len(HERO_PANELS)} map panels, found {len(spans)}")
    boxes = []
    for left, right in spans:
        top, bottom = max(_runs(ink[:, left:right].mean(axis=1) > _PANEL_FILL), key=lambda span: span[1] - span[0])
        boxes.append((left, top, right, bottom))
    size = (boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1])
    for left, top, right, bottom in boxes[1:]:
        if abs(right - left - size[0]) > _PANEL_TOLERANCE_PX or abs(bottom - top - size[1]) > _PANEL_TOLERANCE_PX:
            raise ValueError(f"{hero.name}: map panels differ in size ({right - left}x{bottom - top} vs {size[0]}x{size[1]})")
    written = []
    for name, box in zip(HERO_PANELS, boxes, strict=True):
        panel = image.crop(box)
        if panel.size != size:  # matplotlib rounds panel edges, so one may be a pixel narrower
            panel = panel.resize(size, Image.Resampling.LANCZOS)
        panel.quantize(colors=256).save(out_dir / name, optimize=True)
        written.append(out_dir / name)
    return written


def build(out: Path, root: Path = ROOT) -> Path:
    """Write the page, its data and its figures into ``out``; return the page's path."""
    data = site_data(root)
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in SITE_FILES:
        shutil.copy2(root / "site" / name, out / name)
    for name in FIGURES:
        shutil.copy2(root / "assets" / name, assets / name)
    hero_panels(root / "assets" / "forli_hero.png", assets)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (out / "data.js").write_text(f"window.FLOODFUSION = {payload};\n", encoding="utf-8")
    arena = arena_data(root)
    for pair in arena["maps"]["pairs"]:
        for layer in pair["layers"].values():
            shutil.copy2(root / layer["src"], assets / Path(layer["src"]).name)
    payload = json.dumps(arena, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (out / "arena-data.js").write_text(f"window.FLOODFUSION_ARENA = {payload};\n", encoding="utf-8")
    return out / "index.html"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="output folder (default: _site/)")
    args = parser.parse_args(argv)
    print(f"wrote {build(args.out)}")


if __name__ == "__main__":
    main()

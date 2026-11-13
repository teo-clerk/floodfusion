"""scripts/build_site.py: the explainer is built from the committed reports and cannot drift from them."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_site.py"


def _script():
    spec = importlib.util.spec_from_file_location("build_site", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    _script().build(out)
    payload = (out / "data.js").read_text(encoding="utf-8")
    assert payload.startswith("window.FLOODFUSION = ") and payload.rstrip().endswith(";")
    data = json.loads(payload.removeprefix("window.FLOODFUSION = ").rstrip().removesuffix(";"))
    return out, data


def test_md_tables_reads_every_table_under_its_heading():
    text = "\n".join([
        "# Report", "intro", "| a | b |", "| --- | --- |", "| **1** | `x` |", "| 2 | y |", "",
        "## Second", "| c |", "| :-- |", "| 3 |",
    ])
    tables = _script().md_tables(text)
    assert tables == [("Report", [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]), ("Second", [{"c": "3"}])]
    assert _script().table(tables, "Sec") == [{"c": "3"}]
    with pytest.raises(KeyError, match="Missing"):
        _script().table(tables, "Missing")


def test_numbers_come_from_the_reports(site):
    _, data = site
    cv = json.loads((ROOT / "reports" / "sen1floods11_table.json").read_text(encoding="utf-8"))
    rows = {r["key"]: r for r in data["sen1floods11"]["rows"]}
    assert set(rows) == set(cv["rows"])
    for key, row in rows.items():
        assert row["iou"] == cv["rows"][key]["iou"]
        assert set(row["per_event"]) == {e.removeprefix("S1F11_").replace("-", " ") for e in cv["rows"][key]["per_event"]}
    assert [g["key"] for g in data["sen1floods11"]["gaps"]] == list(cv["gaps"])
    assert all(g["question"] != g["key"] for g in data["sen1floods11"]["gaps"])
    registration = {r["event"]: r for r in data["sen1floods11"]["registration"]}
    assert set(registration) == set(data["sen1floods11"]["events"])
    # reports/sen1floods11_registration.md: "Bolivia | [1, 3] (label)", "Pakistan | [0, 1] (optical)", "Mekong | [0, 0] (none)"
    assert registration["Bolivia"] == {"event": "Bolivia", "dy": 1, "dx": 3, "source": "label"}
    assert registration["Pakistan"]["source"] == "optical"
    assert (registration["Mekong"]["dy"], registration["Mekong"]["dx"]) == (0, 0)
    for gap in data["sen1floods11"]["gaps"] + data["synthetic"]["gaps"]:  # the spec's codes are stripped
        assert not re.search(r"\((?:R\d+|\d+ − \d+|docs/)", gap["question"]), gap["question"]
        assert "*" not in gap["question"]
    assert len(data["sen1floods11"]["label_change"]) == len(data["sen1floods11"]["events"])

    forli = json.loads((ROOT / "reports" / "emsr664_forli.json").read_text(encoding="utf-8"))
    assert {p["tag"] for p in data["forli"]["passes"]} == set(forli["time_offset_h"])
    assert len(data["forli"]["products"]) == 4
    assert all(p["sensor"] and p["time"] for p in data["forli"]["products"])
    assert data["forli"]["common_km2"] > 0
    assert data["synthetic"]["gate"]["start"] > 0
    assert data["synthetic"]["stress"] == json.loads((ROOT / "reports" / "stress.json").read_text(encoding="utf-8"))


def test_number_reads_typographic_minus_and_thousands():
    assert _script()._number("\u22121.40") == -1.40
    assert _script()._number("1,585,128") == 1_585_128
    with pytest.raises(ValueError, match="no number"):
        _script()._number("—")


def test_hero_panels_are_cut_to_one_size(site):
    from PIL import Image

    out, _ = site
    sizes = {Image.open(out / "assets" / name).size for name in _script().HERO_PANELS}
    assert len(sizes) == 1
    (width, height), = sizes
    assert width > 200 and height > 200


def test_hero_panels_refuse_panels_of_different_sizes(tmp_path):
    from PIL import Image

    figure = np.full((120, 400, 3), 255, dtype=np.uint8)
    for left, width in ((10, 100), (140, 100), (270, 80)):  # the third panel is 20 px narrower
        figure[10:110, left:left + width] = 40
    Image.fromarray(figure).save(tmp_path / "uneven.png")
    with pytest.raises(ValueError, match="differ in size"):
        _script().hero_panels(tmp_path / "uneven.png", tmp_path)


def test_hero_panels_refuse_a_figure_without_three_panels(tmp_path):
    from PIL import Image

    Image.fromarray(np.full((60, 200, 3), 255, dtype=np.uint8)).save(tmp_path / "blank.png")
    with pytest.raises(ValueError, match="expected 3 map panels, found 0"):
        _script().hero_panels(tmp_path / "blank.png", tmp_path)


def test_page_references_resolve(site):
    out, _ = site
    html = (out / "index.html").read_text(encoding="utf-8")
    script = (out / "app.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    looked_up = re.findall(r'byId\("([^"]+)"\)', script) + re.findall(r'querySelectorAll?\("#([\w-]+)', script)
    for wanted in looked_up:
        assert wanted in ids, f"app.js looks up #{wanted}, which index.html does not define"
    for anchor in re.findall(r'href="#([^"]+)"', html):
        assert anchor in ids, f"index.html links to #{anchor}, which does not exist"
    for asset in set(re.findall(r'assets/[\w.-]+', html + script)):
        assert (out / asset).is_file(), f"{asset} is referenced but not built"
    for bound in set(re.findall(r'data-bind="([^"]+)"', html)):
        assert re.search(rf"\b{bound}:", script), f"data-bind={bound!r} has no value in app.js"


def test_main_writes_the_site(tmp_path, capsys):
    _script().main(["--out", str(tmp_path / "public")])
    assert (tmp_path / "public" / "index.html").is_file()
    assert "index.html" in capsys.readouterr().out


def test_v11_numbers_come_from_the_reports(site):
    _, data = site
    reports = ROOT / "reports"
    cems = json.loads((reports / "cems_benchmark.json").read_text(encoding="utf-8"))
    tests = {t["id"]: t for t in data["cems"]["tests"]}
    assert sorted(tests) == ["P1", "P2", "P3", "P4"]
    assert sorted(t["hypothesis"] for t in tests.values()) == ["H1", "H2", "H3", "H4"]
    for unit, results in cems["tests"].items():
        for t in results:
            shown = tests[t["id"]]["units"][unit]
            assert (shown["delta"], shown["k"], shown["wins"], shown["holds"]) == (t["delta"], t["k"], t["wins"], t["holds"])
    # The "Supported" column of cems_benchmark.md, not a re-derivation of it.
    assert {k: t["supported"] for k, t in tests.items()} == {"P1": "no verdict", "P2": "no", "P3": "yes", "P4": "yes"}
    assert data["cems"]["n_activations"] == len(cems["units"]["activation"])
    # Read from the stored scores: the primary-stratum pairs of the tested activations and their date range.
    assert (data["cems"]["n_pairs"], data["cems"]["first_month"], data["cems"]["last_month"]) == (75, "2023-03", "2026-03")
    assert data["cems"]["n_clusters"] == len(cems["units"]["cluster"])
    strata = json.loads((reports / "cems_benchmark_strata.json").read_text(encoding="utf-8"))
    primary = next(v for title, v in strata.items() if title.startswith("Drawn on radar"))
    for subset, v in primary.items():
        assert data["cems"]["subsets"][subset]["k"] == len(v["units"])
        assert {k: m["iou"] for k, m in data["cems"]["subsets"][subset]["models"].items()} == \
            {m["key"]: m["iou"] for m in v["models"]}
    assert {k: m["iou"] for k, m in data["cems"]["subsets"]["all"]["models"].items()} == \
        {k: m["iou"] for k, m in cems["models"].items()}

    ks = json.loads((reports / "v11_kurosiwo_table.json").read_text(encoding="utf-8"))
    assert {r["key"]: r["iou"] for r in data["kurosiwo"]["rows"]} == {k: ks["rows"][k]["iou"] for k in _script().KS_ROWS}
    assert data["kurosiwo"]["n_groups"] == len(ks["events"])
    replication = json.loads((reports / "v11_sen1floods11_table.json").read_text(encoding="utf-8"))
    assert {f"{g['row']}/seed{g['seed']}": g["delta"] for g in data["replication"]["protocol_gap"]} == \
        {run: gap["iou_tau"]["delta"] for run, gap in replication["protocol_gap"].items()}


def test_hypothesis_codes_are_read_from_the_question():
    pattern = _script()._HYPOTHESIS
    assert pattern.search("Fusion helps (H2)")[1] == "H2"
    assert pattern.search("No worse than the SAR U-Net (R2 in practice, H3)")[1] == "H3"
    assert pattern.sub("", "No worse (R2 in practice, H3)") == "No worse"
    assert pattern.search("A question with (R3) only") is None


def test_cems_refuses_a_report_without_a_grouped_model(monkeypatch):
    module = _script()
    monkeypatch.setattr(module, "CEMS_GROUPS", (*module.CEMS_GROUPS, ("Extra", {"not_a_model": "Missing"})))
    with pytest.raises(ValueError, match="not_a_model"):
        module.cems(ROOT / "reports")


def test_v11_extracts_fail_loudly_on_missing_parts(monkeypatch):
    module = _script()
    monkeypatch.setattr(module, "CEMS_SUBSETS_USED", ("all", "no_such_subset"))
    with pytest.raises(ValueError, match="no_such_subset"):
        module.cems(ROOT / "reports")
    monkeypatch.setattr(module, "CEMS_PRIMARY", "No such stratum")
    with pytest.raises(KeyError, match="No such stratum"):
        module.cems(ROOT / "reports")
    monkeypatch.setattr(module, "KS_ROWS", {**module.KS_ROWS, "99": "Not a row"})
    with pytest.raises(ValueError, match="99"):
        module.kurosiwo(ROOT / "reports")


# --- Arena (v1.2) -------------------------------------------------------------------------------------

def _row(name="Model", value=0.4):
    m = ("iou_tau", "iou_tau_binned", "iou_selfcal", "iou_oracle", "ap")
    return {"name": name, "k": 30, **{k: value for k in m}, "ci": {k: [value - 0.05, value + 0.05] for k in m}}


def _preview(tmp_path: Path, *, drop: str | None = None) -> Path:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    tables = [{"unit": u, "gfm_common": g, "units": ["EMSR1", "EMSR2"],
               "models": {"ks_r9_flood": _row("Fusion"), "gfm": {**_row("GFM"), "iou_selfcal": float("nan")}}}
              for u in ("activation", "cluster") for g in (False, True)]
    if drop:
        del tables[0]["models"]["ks_r9_flood"][drop]
    (reports / "v12_cems_preview.json").write_text(json.dumps({"tables": tables, "n_pairs": 70, "gfm_pairs": 64}))
    return reports


def test_arena_sections_are_pending_until_their_reports_exist(tmp_path):
    s = _script()
    (tmp_path / "reports").mkdir()
    (tmp_path / "assets").mkdir()
    assert s._arena_scoreboard(tmp_path / "reports") == {"status": "pending"}
    assert s._arena_maps(tmp_path / "assets") == {"status": "pending", "pairs": []}
    assert s._blind(tmp_path / "reports")["status"] == "not started"
    assert [e["resolved"] for e in s._errata(tmp_path / "reports")] == [False, False]


def test_arena_scoreboard_reads_every_table_and_turns_nan_into_null(tmp_path):
    board = _script()._arena_scoreboard(_preview(tmp_path))
    assert board["status"] == "ok" and len(board["tables"]) == 4 and board["gfm_pairs"] == 64
    gfm = board["tables"][0]["models"]["gfm"]
    assert gfm["iou_selfcal"] is None and gfm["ci"]["ap"] == pytest.approx([0.35, 0.45])
    json.dumps(board, allow_nan=False)                                 # the site's payload rule


def test_arena_scoreboard_fails_loudly_on_a_missing_metric(tmp_path):
    with pytest.raises(KeyError, match="lacks"):
        _script()._arena_scoreboard(_preview(tmp_path, drop="ap"))


def test_arena_maps_refuse_a_manifest_naming_a_missing_quicklook(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    pair = {"key": "EMSR1_AOI01", "activation": "EMSR1", "aoi": 1, "product": "P", "sensors": ["ICEYE"],
            "product_time": "2024-01-01T10:00:00", "pass_time": "2024-01-01T11:00:00", "flood_km2": 3.2,
            "layers": {"radar": {"name": "Sentinel-1 VV", "iou": None}, "ks_r9": {"name": "Fusion", "iou": 0.5}}}
    (assets / "arena-manifest.json").write_text(json.dumps({"pairs": [pair]}))
    (assets / "arena-EMSR1_AOI01-radar.webp").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match=r"arena-EMSR1_AOI01-ks_r9\.webp"):
        _script()._arena_maps(assets)
    (assets / "arena-EMSR1_AOI01-ks_r9.webp").write_bytes(b"x")
    maps = _script()._arena_maps(assets)
    assert maps["pairs"][0]["layers"]["ks_r9"] == {"name": "Fusion", "iou": 0.5, "src": "assets/arena-EMSR1_AOI01-ks_r9.webp"}


def test_erratum_one_reports_the_corrected_p4_once_it_exists(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    test = {"id": "P4c", "delta": 0.012, "ci": [-0.01, 0.03], "wins": 20, "k": 38, "p": 0.21}
    (reports / "v1_1_erratum_p4.json").write_text(json.dumps({"tests": {"activation": [test],
                                                                        "cluster": [{**test, "k": 30}]}}))
    first = _script()._errata(reports)[0]
    assert first["resolved"] and "+0.012 [-0.010, +0.030]" in first["result"] and "20/38" in first["result"]


def test_blind_test_shows_counts_never_scores(tmp_path):
    ledger = tmp_path / "reports" / "prospective" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    rows = [  # the runner's own row format (floodfusion.prospective._row); a pair's last line is its state
        {"key": "EMSR926", "activation": "EMSR926", "event_date": "2026-08-21", "status": "settling", "at": "t0"},
        {"key": "EMSR926_AOI01", "activation": "EMSR926", "event_date": "2026-08-21", "status": "scored",
         "at": "t1", "stratum": "sar", "sealed_sha256": "ab"},
        {"key": "EMSR927_AOI02", "activation": "EMSR927", "event_date": "2026-08-25", "status": "scored",
         "at": "t1", "stratum": "optical", "sealed_sha256": "cd"},                     # not the primary stratum
        {"key": "EMSR933", "activation": "EMSR933", "event_date": "2026-09-30", "status": "settling", "at": "t0"},
        {"key": "EMSR933", "activation": "EMSR933", "event_date": "2026-09-30", "status": "not eligible", "at": "t2"}]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = _script()
    blind = s._blind(tmp_path / "reports", records=tmp_path / "records")
    assert blind["k_activations"] == 1 and blind["k_clusters"] is None             # records not available here
    assert [f["status"] for f in blind["floods"]] == ["settling", "scored", "scored", "not eligible"]
    assert "sealed_sha256" not in json.dumps(blind) and "iou" not in json.dumps(blind)
    records = tmp_path / "records"
    records.mkdir()
    (records / "EMSR926.json").write_text(json.dumps({"eventTime": "2026-08-21T00:00:00Z",
                                                      "extent": "POLYGON ((10 45, 11 45, 11 46, 10 46, 10 45))"}))
    assert s._blind(tmp_path / "reports", records=records)["k_clusters"] == 1


def test_the_arena_page_and_its_script_agree(tmp_path):
    out = tmp_path / "site"
    _script().build(out)
    html = (out / "arena.html").read_text(encoding="utf-8")
    script = (out / "arena.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    for wanted in re.findall(r'byId\("([^"]+)"\)', script):
        assert wanted in ids, f"arena.js looks up #{wanted}, which arena.html does not define"
    for anchor in re.findall(r'href="#([^"]+)"', html):
        assert anchor in ids, f"arena.html links to #{anchor}, which does not exist"
    assert (out / "arena-data.js").read_text(encoding="utf-8").startswith("window.FLOODFUSION_ARENA = {")


def test_the_scoreboard_prefers_every_model_on_the_rebuilt_stacks(tmp_path):
    s = _script()
    row = {"name": "M", "k": 2, **{m: 0.4 for m in s.ARENA_METRICS}, "ci": {m: [0.3, 0.5] for m in s.ARENA_METRICS}}
    report = {"tables": [{"unit": u, "gfm_common": g, "units": ["EMSR1", "EMSR2"], "models": {"m": row}}
                         for u, g in s.ARENA_TABLES], "n_pairs": 3, "gfm_pairs": 2}
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "v12_cems_preview.json").write_text(json.dumps(report))
    assert s._arena_scoreboard(reports)["source"] == "preview"
    (reports / "v12_cems_rebuilt.json").write_text(json.dumps(report))
    assert s._arena_scoreboard(reports)["source"] == "rebuilt"


def test_every_listed_map_layer_gfm_included_must_be_in_the_assets(tmp_path):
    s = _script()
    assets = tmp_path / "assets"
    assets.mkdir()
    pair = {"key": "EMSR1_AOI01", "activation": "EMSR1", "aoi": 1, "product": "P", "sensors": ["ICEYE"],
            "product_time": "2026-01-01T00:00:00", "pass_time": "2026-01-01T06:00:00", "flood_km2": 1.0,
            "layers": {"radar": {"name": "Sentinel-1 VV", "iou": None}, "gfm": {"name": "GFM", "iou": 0.5},
                       "ours": {"name": "Ours", "iou": 0.6}}}
    (assets / s.ARENA_MANIFEST).write_text(json.dumps({"pairs": [pair]}))
    for key in ("radar", "ours"):
        (assets / f"arena-EMSR1_AOI01-{key}.webp").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="gfm"):
        s._arena_maps(assets)
    (assets / "arena-EMSR1_AOI01-gfm.webp").write_bytes(b"x")
    layers = s._arena_maps(assets)["pairs"][0]["layers"]
    assert set(layers) == {"radar", "gfm", "ours"}
    assert layers["gfm"] == {"name": "GFM", "iou": 0.5, "src": "assets/arena-EMSR1_AOI01-gfm.webp"}


def test_the_arena_gives_the_cems_notice_for_the_gfm_maps_it_ships():
    """GFM's terms of service ask for this notice wherever its data are shown adapted."""
    manifest = json.loads((ROOT / "assets" / "arena-manifest.json").read_text(encoding="utf-8"))
    assert any("gfm" in pair["layers"] for pair in manifest["pairs"])
    page = " ".join((ROOT / "site" / "arena.html").read_text(encoding="utf-8").split())
    assert "Contains modified Copernicus Emergency Management Service information" in page
    assert "Global Flood Monitoring (GFM)" in page


def test_erratum_2_reports_its_measured_effect_once_the_rebuilt_stacks_are_scored(tmp_path):
    s = _script()
    reports = tmp_path / "reports"
    reports.mkdir()

    def report(values):
        models = {k: {"iou_tau": v} for k, v in values.items()}
        return {"tables": [{"unit": "activation", "gfm_common": False, "models": models}]}

    (reports / "v12_cems_preview.json").write_text(json.dumps(report(
        {"s1f11_r9": 0.258, "ks_r9_flood": 0.397, "r2": 0.371, "s1f11_tm_fixed": 0.299})))
    assert s._errata(reports)[1]["result"] is None                         # not yet rebuilt
    (reports / "v12_cems_rebuilt.json").write_text(json.dumps(report(
        {"s1f11_r9": 0.266, "v11": 0.407, "r2": 0.373, "s1f11_tm_fixed": 0.299})))
    test = {"id": "P4c", "delta": -0.033, "ci": [-0.055, -0.011], "wins": 13, "k": 38}
    (reports / "v1_1_erratum2_p4.json").write_text(json.dumps({"tests": {"activation": [test]}}))
    dem = s._errata(reports)[1]
    assert dem["resolved"] and dem["status"] == "Measured on the rebuilt stacks"
    assert "+0.002 to +0.010" in dem["result"] and "at most 0.0000" in dem["result"]
    assert "-0.033" in dem["result"] and "13/38" in dem["result"] and "holds" in dem["result"]


# --- floodfusion Live's page (v1.3, 2026-11-11) ------------------------------------------------------------------

def _live_root(tmp_path, maps, *, blind=False):
    root = tmp_path / "live-data"
    for entry in maps:
        tile = root / "maps" / entry["id"] / "t00_00"
        tile.mkdir(parents=True)
        (tile / "flood.geojson").write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    (root / "index.json").write_text(json.dumps({"maps": maps, "blind_floods": blind, "complete": [], "failures": {"x": 1},
                                                 "updated_at": "2026-11-11T06:00:00+00:00", "last_code": 898}))
    return root


def _entry(map_id, code="EMSR664", pass_time="2023-05-17T05:16:00+00:00", kind="showcase"):
    return {"id": map_id, "kind": kind, "code": code, "name": "Flood in Italy", "aoi": 1, "aoi_name": "Forlì",
            "bbox": [12.0, 44.2, 12.1, 44.3], "pass_time": pass_time, "mapped_at": "2026-11-11T06:00:00+00:00",
            "delay_hours": 30000.0, "flood_km2": 1.5, "valid_km2": 40.0, "tiles": 1, "skipped_tiles": 0,
            "files": [f"maps/{map_id}/t00_00/flood.geojson"], "event_date": "2023-05-16", "orbit": 95,
            "orbit_state": "descending"}


def test_the_live_page_waits_for_its_first_maps(tmp_path):
    data = _script().live_data(tmp_path / "absent")
    assert data == {"maps": [], "updated_at": None, "blind_floods": False}


def test_the_live_page_lists_the_newest_pass_first_and_copies_only_its_polygons(tmp_path):
    root = _live_root(tmp_path, [_entry("a", pass_time="2023-05-17T05:16:00+00:00"),
                                 _entry("b", pass_time="2023-05-18T17:05:00+00:00")])
    out = tmp_path / "site"
    data = _script().live_data(root, out=out)
    assert [m["id"] for m in data["maps"]] == ["b", "a"] and "failures" not in data
    assert data["updated_at"] == "2026-11-11T06:00:00+00:00"
    assert (out / "live" / "maps" / "a" / "t00_00" / "flood.geojson").is_file()
    assert data["maps"][0]["files"] == ["live/maps/b/t00_00/flood.geojson"]


def test_the_live_page_refuses_a_blind_flood_while_blind_floods_are_off(tmp_path):
    root = _live_root(tmp_path, [_entry("x", code="EMSR926")])
    with pytest.raises(ValueError, match="blind"):
        _script().live_data(root)
    late = _live_root(tmp_path / "late", [_entry("y", code=None, kind="manual", pass_time="2026-09-01T05:00:00+00:00")])
    with pytest.raises(ValueError, match="blind"):
        _script().live_data(late)


@pytest.mark.parametrize("code", ["emsr926", " EMSR926", "EMSR926 "])
def test_a_blind_code_in_another_spelling_is_still_refused(tmp_path, code):
    with pytest.raises(ValueError, match="blind"):
        _script().live_data(_live_root(tmp_path, [_entry("x", code=code)]))


@pytest.mark.parametrize("code", ["LSR664", "EMSR", "EMSR664; x"])
def test_a_code_that_is_not_a_cems_code_is_refused(tmp_path, code):
    with pytest.raises(ValueError, match="not a CEMS activation code"):
        _script().live_data(_live_root(tmp_path, [_entry("x", code=code)]))


@pytest.mark.parametrize("pass_time", ["18 July 2026", "2026/09/01", "late"])
def test_a_pass_time_that_is_not_iso_is_refused_not_compared_as_text(tmp_path, pass_time):
    # "18 July 2026" sorts before "2026-07-17" as text, so a string comparison would let it through.
    with pytest.raises(ValueError, match="pass time"):
        _script().live_data(_live_root(tmp_path, [_entry("y", code=None, kind="manual", pass_time=pass_time)]))


def test_the_live_page_refuses_a_file_outside_its_map(tmp_path):
    entry = {**_entry("a"), "files": ["../../etc/passwd"]}
    with pytest.raises(ValueError, match="outside"):
        _script().live_data(_live_root(tmp_path, [entry]), out=tmp_path / "site")


def test_the_live_page_and_its_script_agree(tmp_path):
    out = tmp_path / "site"
    _script().build(out, live_root=_live_root(tmp_path, [_entry("a")]))
    html = (out / "live.html").read_text(encoding="utf-8")
    script = (out / "live.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    for wanted in re.findall(r'byId\("([^"]+)"\)', script):
        assert wanted in ids, f"live.js looks up #{wanted}, which live.html does not define"
    for anchor in re.findall(r'href="#([^"]+)"', html):
        assert anchor in ids, f"live.html links to #{anchor}, which does not exist"
    assert (out / "live-data.js").read_text(encoding="utf-8").startswith("window.FLOODFUSION_LIVE = {")


def test_only_the_live_page_may_fetch_and_only_from_itself_and_one_tile_host(tmp_path):
    out = tmp_path / "site"
    _script().build(out)
    csp = {}
    for page in ("index.html", "arena.html", "live.html"):
        meta = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', (out / page).read_text())
        csp[page] = dict(part.strip().split(" ", 1) for part in meta.group(1).split(";"))
    assert csp["index.html"]["connect-src"] == csp["arena.html"]["connect-src"] == "'none'"
    assert csp["live.html"]["connect-src"] == "'self'" and csp["live.html"]["script-src"] == "'self'"
    assert csp["live.html"]["img-src"] == "'self' data: https://tile.openstreetmap.org"


def test_the_live_page_and_the_live_service_share_the_blind_rule():
    from datetime import date, timedelta

    from floodfusion import live_rules

    module = _script()
    assert module.BLIND_FIRST_CODE == live_rules.BLIND_FIRST_CODE
    last = date.fromisoformat(module.BLIND_LAST_PASS_DATE)
    assert live_rules.check_date(last, enabled=False) == last
    with pytest.raises(live_rules.BlindFloodRefusedError):
        live_rules.check_date(last + timedelta(days=1), enabled=False)


# --- after the security and code reviews of 2026-11-11 -------------------------------------------------------------

def test_the_live_page_takes_the_blind_switch_from_the_repository_not_from_the_data(tmp_path):
    root = _live_root(tmp_path, [_entry("x", code="EMSR926")], blind=True)   # the data claims the switch is on
    with pytest.raises(ValueError, match="blind"):
        _script().live_data(root)
    assert _script().live_data(_live_root(tmp_path / "ok", [_entry("a")], blind=True))["blind_floods"] is False


def test_the_live_page_refuses_a_symlinked_file(tmp_path):
    root = _live_root(tmp_path, [_entry("a")])
    secret = tmp_path / "secret"
    secret.write_text("token")
    target = root / "maps" / "a" / "t00_00" / "flood.geojson"
    target.unlink()
    target.symlink_to(secret)
    with pytest.raises(ValueError, match="not a plain file"):
        _script().live_data(root, out=tmp_path / "site")
    assert not (tmp_path / "site" / "live" / "maps" / "a" / "t00_00" / "flood.geojson").exists()


def test_the_live_page_refuses_a_file_that_is_not_geojson(tmp_path):
    root = _live_root(tmp_path, [_entry("a")])
    (root / "maps" / "a" / "t00_00" / "flood.geojson").write_text("not json")
    with pytest.raises(ValueError, match="JSON"):
        _script().live_data(root, out=tmp_path / "site")


def test_the_live_page_ships_each_map_s_stac_item(tmp_path):
    root = _live_root(tmp_path, [_entry("a")])
    (root / "maps" / "a" / "t00_00" / "item.json").write_text('{"type": "Feature"}')
    data = _script().live_data(root, out=tmp_path / "site")
    assert data["maps"][0]["items"] == ["live/maps/a/t00_00/item.json"]
    assert (tmp_path / "site" / "live" / "maps" / "a" / "t00_00" / "item.json").is_file()


def test_a_live_map_without_a_pass_time_is_refused(tmp_path):
    with pytest.raises(ValueError, match="pass time"):
        _script().live_data(_live_root(tmp_path, [{**_entry("a"), "pass_time": None}]))


def test_the_live_page_gets_the_imaged_tiles_and_refuses_malformed_ones(tmp_path):
    entry = {**_entry("a"), "tile_bboxes": [[12.0, 44.2, 12.05, 44.25]]}
    assert _script().live_data(_live_root(tmp_path, [entry]))["maps"][0]["tile_bboxes"] == [[12.0, 44.2, 12.05, 44.25]]
    bad = {**_entry("b"), "tile_bboxes": [["12", 44.2, 12.05]]}
    with pytest.raises(ValueError, match="tile"):
        _script().live_data(_live_root(tmp_path / "bad", [bad]))

"""The v1.2 blind test runner (offline: fake catalogue, fake eligibility, fake scoring)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from floodfusion import prospective as pr

NOW = datetime(2026, 12, 1, tzinfo=UTC)


def _record(code: str, days_ago: int, category: str = "Flood") -> dict:
    return {"code": code, "category": category, "eventTime": (NOW - timedelta(days=days_ago)).isoformat(),
            "extent": "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "aois": []}


def test_records_refresh_until_settled_and_missing_codes_are_asked_again_after_a_week(tmp_path: Path) -> None:
    calls = []
    catalogue = {"EMSR899": _record("EMSR899", 40), "EMSR900": _record("EMSR900", 5)}

    def fetch(code):
        calls.append(code)
        if code not in catalogue:
            raise LookupError(code)
        return catalogue[code]

    assert pr._fetch_one("EMSR899", tmp_path, fetch, NOW)["code"] == "EMSR899"
    assert pr._fetch_one("EMSR899", tmp_path, fetch, NOW) and calls.count("EMSR899") == 1   # settled: cached
    pr._fetch_one("EMSR900", tmp_path, fetch, NOW)
    pr._fetch_one("EMSR900", tmp_path, fetch, NOW)
    assert calls.count("EMSR900") == 2                                   # not settled: asked every sweep
    assert pr._fetch_one("EMSR901", tmp_path, fetch, NOW) is None
    assert pr._fetch_one("EMSR901", tmp_path, fetch, NOW + timedelta(days=3)) is None and calls.count("EMSR901") == 1
    catalogue["EMSR901"] = _record("EMSR901", 1)
    assert pr._fetch_one("EMSR901", tmp_path, fetch, NOW + timedelta(days=8))["code"] == "EMSR901"
    assert not (tmp_path / "EMSR901.missing").exists()


def test_census_stops_after_consecutive_non_public_codes(tmp_path: Path) -> None:
    catalogue = {f"EMSR{n}": _record(f"EMSR{n}", 40) for n in (899, 900, 903)}
    asked = []

    def fetch(code):
        asked.append(code)
        if code not in catalogue:
            raise LookupError(code)
        return catalogue[code]

    records = pr.census(tmp_path, fetch=fetch, now=NOW, stop_after=3)
    assert [r["code"] for r in records] == ["EMSR899", "EMSR900", "EMSR903"]
    assert asked[-1] == "EMSR906"                                       # 904, 905, 906 missing: stop


def test_ledger_appends_only_changes_and_keeps_the_last_state(tmp_path: Path) -> None:
    ledger = pr.Ledger(tmp_path / "ledger.jsonl")
    row = {"key": "EMSR926_AOI01", "activation": "EMSR926", "status": "settling", "at": "t1"}
    assert ledger.append(row) and not ledger.append({**row, "at": "t2"})
    assert ledger.append({**row, "status": "scored", "at": "t3"})
    assert len(ledger.rows()) == 2 and ledger.state()["EMSR926_AOI01"]["status"] == "scored"


def test_sealed_scores_are_hashed_into_the_ledger_and_tampering_is_caught(tmp_path: Path) -> None:
    sha = pr.seal({"pairs": {"a": 1}}, tmp_path / "sealed", "EMSR926_AOI01")
    ledger = pr.Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"key": "EMSR926_AOI01", "activation": "EMSR926", "status": "scored", "sealed_sha256": sha})
    assert pr.verify_sealed(ledger, tmp_path / "sealed") == []
    (tmp_path / "sealed" / "EMSR926_AOI01.json").write_text(json.dumps({"pairs": {"a": 2}}))
    assert pr.verify_sealed(ledger, tmp_path / "sealed") == ["EMSR926_AOI01"]


def test_stopping_rule_needs_clusters_and_activations_or_the_end_date(tmp_path: Path) -> None:
    ledger = pr.Ledger(tmp_path / "ledger.jsonl")
    n_units = pr.MIN_ACTIVATIONS
    for n in range(n_units):
        ledger.append({"key": f"EMSR{900 + n}_AOI01", "activation": f"EMSR{900 + n}", "status": "scored"})
    few_clusters = {f"EMSR{900 + n}": f"C{n % (pr.MIN_CLUSTERS - 1)}" for n in range(n_units)}
    assert not pr.stopping_rule_met(ledger, date(2027, 6, 1), few_clusters)
    assert pr.stopping_rule_met(ledger, date(2027, 6, 1), {})            # every activation its own cluster
    assert (pr.MIN_ACTIVATIONS, pr.MIN_CLUSTERS, pr.END_DATE) == (20, 15, date(2027, 12, 31))   # as pre-registered
    short = pr.Ledger(tmp_path / "short.jsonl")
    for n in range(pr.MIN_ACTIVATIONS - 1):
        short.append({"key": f"EMSR{900 + n}_AOI01", "activation": f"EMSR{900 + n}", "status": "scored"})
    assert not pr.stopping_rule_met(short, date(2027, 6, 1), {})        # enough clusters, one activation short
    assert pr.stopping_rule_met(short, date(2028, 1, 1), {})            # the end date ends it regardless
    assert pr.stopping_rule_met(pr.Ledger(tmp_path / "empty.jsonl"), date(2028, 1, 1), {})


def test_freeze_refuses_changed_checkpoints_specs_rules_and_code(tmp_path: Path, monkeypatch) -> None:
    ckpt, spec = tmp_path / "best.pt", tmp_path / "spec.yaml"
    ckpt.write_bytes(b"weights")
    spec.write_text("models: []\ntests: []\n")
    manifest = {"tag": "v1.2.0-freeze", "rule": pr._rule(), "spec": str(spec), "spec_sha256": pr._sha256(spec),
                "files": {str(ckpt): pr._sha256(ckpt)}}
    assert pr.check_freeze(manifest, git=lambda args: 0) == []
    ckpt.write_bytes(b"other weights")
    spec.write_text("models: []\ntests: [x]\n")
    monkeypatch.setattr(pr, "SETTLE_DAYS", 10)
    problems = pr.check_freeze(manifest, git=lambda args: 1)
    assert len(problems) == 4 and any("census rule" in p for p in problems) and any("code paths" in p for p in problems)


def test_a_sweep_scores_settled_floods_once_and_waits_for_the_rest(tmp_path: Path, monkeypatch) -> None:
    import floodfusion.cems_benchmark as cb

    catalogue = {"EMSR899": _record("EMSR899", 40), "EMSR900": _record("EMSR900", 3),
                 "EMSR901": _record("EMSR901", 50, category="Wildfire"), "EMSR902": _record("EMSR902", 45)}

    def fetch(code):
        if code not in catalogue:
            raise LookupError(code)
        return catalogue[code]

    class Cand:
        def __init__(self, activation):
            self.activation, self.aoi, self.offset_h, self.stratum, self.product = activation, 1, 0.5, "sar", "P"

    monkeypatch.setattr(cb, "eligible_pairs",
                        lambda record, client, others=(): [Cand(record["code"])] if record["code"] == "EMSR899" else [])
    prepared, scored = [], []
    paths = pr.Paths(root=tmp_path / "prospective", ledger=tmp_path / "ledger.jsonl")

    def score_pair(pair_dir):
        scored.append(pair_dir.name)
        return {"pairs": {pair_dir.name: {"models": {}}}}

    kwargs = dict(now=NOW, fetch=fetch, client=None, prepare=lambda pair, record: prepared.append(pair["key"]),
                  score_pair=score_pair)
    counts = pr.sweep(paths, {}, **kwargs)
    assert counts == {"scored": 1, "settling": 1, "not eligible": 1}
    state = pr.Ledger(paths.ledger).state()
    assert state["EMSR899_AOI01"]["status"] == "scored" and len(state["EMSR899_AOI01"]["sealed_sha256"]) == 64
    assert state["EMSR900"]["status"] == "settling" and state["EMSR902"]["status"] == "not eligible"
    assert "EMSR901" not in state                                       # not a flood
    pr.sweep(paths, {}, **kwargs)
    assert prepared == ["EMSR899_AOI01"] and scored == ["EMSR899_AOI01"]   # idempotent
    assert pr.verify_sealed(pr.Ledger(paths.ledger), paths.sealed) == []


def test_a_refused_code_counts_as_not_public_but_a_server_error_stops_the_census(tmp_path: Path) -> None:
    from floodfusion.data.http import HTTPStatusError

    def forbidden(code):
        raise HTTPStatusError("forbidden", 403)

    def broken(code):
        raise HTTPStatusError("bad request", 400)

    assert pr._fetch_one("EMSR950", tmp_path, forbidden, NOW) is None
    assert (tmp_path / "EMSR950.missing").is_file()
    with pytest.raises(HTTPStatusError):
        pr._fetch_one("EMSR951", tmp_path, broken, NOW)
    assert not (tmp_path / "EMSR951.missing").exists()                 # never recorded as "not public"


def test_the_freeze_pins_every_seeds_checkpoint_and_the_cli_writes_and_reads_it(tmp_path: Path, monkeypatch,
                                                                                capsys) -> None:
    runs = [tmp_path / "runs" / f"seed{s}" for s in (42, 43)]
    for run in runs:
        run.mkdir(parents=True)
        (run / "best.pt").write_bytes(run.name.encode())
        (run / "eval_test.json").write_text("{}")
    spec = tmp_path / "spec.yaml"
    spec.write_text("models:\n  - key: ours\n    name: v1.2\n    runs: [" + ", ".join(str(r) for r in runs) + "]\n"
                    "tests: []\n")
    manifest = pr.freeze_manifest(spec, "v1.2.0-freeze")
    assert set(manifest["files"]) == {str(r / n) for r in runs for n in ("best.pt", "eval_test.json")}
    assert manifest["files"][str(runs[0] / "best.pt")] != manifest["files"][str(runs[1] / "best.pt")]
    assert manifest["spec_sha256"] == pr._sha256(spec) and manifest["rule"]["first_code"] == pr.FIRST_CODE

    out = tmp_path / "freeze.json"
    assert pr.main(["freeze", "--spec", str(spec), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["files"] == manifest["files"]
    (runs[0] / "best.pt").write_bytes(b"retrained")
    monkeypatch.setattr(pr, "check_freeze", lambda m: ["models/best.pt: sha256 differs from the freeze"])
    with pytest.raises(SystemExit, match="refusing to sweep"):
        pr.main(["sweep", "--freeze", str(out)])

    ledger = pr.Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"key": "EMSR950_AOI01", "activation": "EMSR950", "event_date": "2026-11-02", "status": "settling"})
    assert pr.main(["status", "--ledger", str(ledger.path)]) == 0
    assert capsys.readouterr().out.strip() == "EMSR950_AOI01\t2026-11-02\tsettling"

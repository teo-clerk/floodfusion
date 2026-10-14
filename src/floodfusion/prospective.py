r"""The v1.2 blind test: every eligible Copernicus EMS flood activation from EMSR899 on, scored by frozen models.

**The question.** v1.1's CEMS benchmark is now seen, and public training sets cover its activations. The only clean
test left is floods that happen after the models are frozen. This module runs that test, under rules committed before
the freeze (``reports/protocol_v1_2.md``).

**A sweep.** Each call of :func:`sweep` (``make v12-watch``, daily):

1. **Census** of codes from :data:`FIRST_CODE` upward, until :data:`STOP_AFTER_NON_PUBLIC` consecutive codes are not
   public.
   - A "not public" answer expires after :data:`MISSING_TTL_DAYS`: v1.1's census cached it forever, and a code not yet
     issued would never have been asked again.
   - An activation's record is re-fetched until it has *settled* (:data:`SETTLE_DAYS` after its event), because
     products keep being published. Then it is frozen, and its sha256 enters the ledger.
2. **Eligibility.** A settled flood activation is put through v1.1's eligibility rule
   (:func:`floodfusion.cems_benchmark.eligible_pairs`, then ``select_pairs``).
3. **Preparation.** Each new pair is prepared exactly as the benchmark's (``prepare_pair``). Its GFM, seasonal and DEM
   sidecars are written beside it.
4. **Scoring.** The frozen models score it (:func:`floodfusion.cems_scores_v12.score`, on that pair only).
5. **Sealing.** The scores are written to ``data/prospective/sealed/<pair>.json``, outside the repository. Only that
   file's sha256 goes into the committed ledger (``reports/prospective/ledger.jsonl``), so the scores cannot change
   after the fact and are revealed and verified at the stopping rule.

**The stopping rule.**
- **Analysed** once :func:`stopping_rule_met`: at least :data:`MIN_CLUSTERS` clusters and :data:`MIN_ACTIVATIONS`
  activations scored, or :data:`END_DATE` passed.
- **No verdict** below eight units.

**The freeze.** :func:`check_freeze` refuses to sweep in three cases:
- a checkpoint differs from the sha256 recorded at the freeze;
- the code paths differ from the freeze tag;
- the census rule changed.

Products: © European Union, Copernicus Emergency Management Service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = ["END_DATE", "FIRST_CODE", "MIN_ACTIVATIONS", "MIN_CLUSTERS", "Ledger", "census", "check_freeze",
           "freeze_manifest", "main", "seal", "settled", "stopping_rule_met", "sweep"]

LOGGER = logging.getLogger(__name__)

FIRST_CODE = 899
STOP_AFTER_NON_PUBLIC = 20
MISSING_TTL_DAYS = 7
SETTLE_DAYS = 30
#: Analysed when this many clusters and activations are scored, or at END_DATE. With v1.1's per-activation spread
#: (SD ~0.077), 20 activations detect ~0.06 IoU at 80 % power (Holm's smallest level); 10 would detect ~0.10 only
#: (reports/v1_2_deviations.md, fixed before any blind flood was looked at).
MIN_CLUSTERS = 15
MIN_ACTIVATIONS = 20
END_DATE = date(2027, 12, 31)
#: The confirmatory stratum: products drawn on radar satellites (``reports/protocol_v1_2.md`` §2.2).
PRIMARY_STRATUM = "sar"
#: The chain value before the ledger's first line.
GENESIS = "0" * 64
#: What the freeze pins in git (plus every checkpoint by sha256).
FROZEN_PATHS = ("src", "configs", "pyproject.toml", "uv.lock", "Makefile")


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------
def _event_time(record: Mapping[str, Any]) -> datetime:
    from floodfusion.cems_benchmark import _time

    return _time(record["eventTime"])


def settled(record: Mapping[str, Any], now: datetime) -> bool:
    """True once :data:`SETTLE_DAYS` have passed since the activation's event: its products are then frozen."""
    return now >= _event_time(record) + timedelta(days=SETTLE_DAYS)


def _fetch_one(code: str, cache: Path, fetch: Callable[[str], dict[str, Any]], now: datetime) -> dict[str, Any] | None:
    """The record of ``code`` (cached; refreshed until settled), or None if not public (re-asked after a week)."""
    from floodfusion.data.http import HTTPStatusError
    from floodfusion.data.records import atomic_write_bytes

    hit, miss = cache / f"{code}.json", cache / f"{code}.missing"
    if hit.is_file():
        record = json.loads(hit.read_text(encoding="utf-8"))
        if settled(record, now):
            return record
    elif miss.is_file():
        asked = datetime.fromisoformat(miss.read_text(encoding="utf-8").strip())
        if now - asked < timedelta(days=MISSING_TTL_DAYS):
            return None
    try:
        record = fetch(code)
    except LookupError:
        record = None
    except HTTPStatusError as exc:
        if exc.status not in (401, 403):
            raise
        record = None
    if record is None:
        atomic_write_bytes(miss, now.isoformat().encode())
        return None
    miss.unlink(missing_ok=True)
    atomic_write_bytes(hit, json.dumps(record).encode("utf-8"))
    return record


def census(cache: Path, *, fetch: Callable[[str], dict[str, Any]], now: datetime, first: int = FIRST_CODE,
           stop_after: int = STOP_AFTER_NON_PUBLIC) -> list[dict[str, Any]]:
    """Public activation records from ``first`` upward, stopping after ``stop_after`` consecutive non-public codes."""
    cache.mkdir(parents=True, exist_ok=True)
    records, gap, n = [], 0, first
    while gap < stop_after:
        record = _fetch_one(f"EMSR{n}", cache, fetch, now)
        if record is None:
            gap += 1
        else:
            gap = 0
            records.append(record)
        n += 1
    return records


# ---------------------------------------------------------------------------
# Ledger and sealing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ledger:
    """An append-only JSON-lines record of what the blind test saw and did; the last line per pair is its state.

    **Tamper evidence.** Each row carries ``prev``, the sha256 of the line before it, so an edited, deleted or
    reordered line breaks the chain (:meth:`problems`). A pair once scored must stay scored with the same sealed
    hash: it can never be re-scored or quietly dropped. The committed history and the chain head (:meth:`head`)
    anchor the whole file.
    """

    path: Path

    def _lines(self) -> list[str]:
        if not self.path.is_file():
            return []
        return [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def rows(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self._lines()]

    def head(self) -> str:
        """The sha256 of the last line (the chain head; the empty ledger's is :data:`GENESIS`)."""
        lines = self._lines()
        return hashlib.sha256(lines[-1].encode("utf-8")).hexdigest() if lines else GENESIS

    def problems(self) -> list[str]:
        """Broken chain links, and scored pairs that changed state or sealed hash afterwards."""
        out, prev, scored = [], GENESIS, {}
        for i, line in enumerate(self._lines(), start=1):
            row = json.loads(line)
            if row.get("prev") != prev:
                out.append(f"ledger line {i}: the hash chain is broken")
            prev = hashlib.sha256(line.encode("utf-8")).hexdigest()
            key = row["key"]
            if key in scored and (row.get("status") != "scored" or row.get("sealed_sha256") != scored[key]):
                out.append(f"ledger line {i}: {key} was scored and then changed")
            if row.get("status") == "scored":
                scored.setdefault(key, row.get("sealed_sha256"))
        return out

    def state(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in self.rows():
            out[row["key"]] = row
        return out

    def append(self, row: Mapping[str, Any]) -> bool:
        """Append ``row`` unless it repeats the pair's current state; return whether it was written."""
        current = self.state().get(row["key"])
        if current is not None and {k: v for k, v in current.items() if k not in ("at", "prev")} == \
                {k: v for k, v in row.items() if k not in ("at", "prev")}:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({**dict(row), "prev": self.head()}, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True


def seal(scores: Mapping[str, Any], sealed_dir: Path, key: str) -> str:
    """Write one pair's scores outside the repository; return their sha256 (what the committed ledger records)."""
    from floodfusion.data.records import atomic_write_bytes

    blob = json.dumps(scores, sort_keys=True, allow_nan=False).encode("utf-8")
    atomic_write_bytes(sealed_dir / f"{key}.json", blob)
    return hashlib.sha256(blob).hexdigest()


def verify_sealed(ledger: Ledger, sealed_dir: Path) -> list[str]:
    """Pairs whose sealed file is missing or no longer matches the sha256 the ledger committed, then the ledger's own
    problems (:meth:`Ledger.problems`)."""
    bad = []
    for key, row in ledger.state().items():
        if row.get("status") != "scored":
            continue
        path = sealed_dir / f"{key}.json"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sealed_sha256"]:
            bad.append(key)
    return bad + ledger.problems()


def stopping_rule_met(ledger: Ledger, today: date, clusters: Mapping[str, str], *,
                      stratum: str = PRIMARY_STRATUM) -> bool:
    """At least :data:`MIN_ACTIVATIONS` analysable activations in :data:`MIN_CLUSTERS` clusters, or :data:`END_DATE`.

    Analysable, as the analysis counts units: scored, in the primary stratum, with reference flood on the scored
    pixels. ``clusters`` must be built from those activations only (:func:`analysable_activations`).
    """
    scored = analysable_activations(ledger, stratum=stratum)
    return today >= END_DATE or (len(scored) >= MIN_ACTIVATIONS
                                 and len({clusters.get(a, a) for a in scored}) >= MIN_CLUSTERS)


def analysable_activations(ledger: Ledger, *, stratum: str = PRIMARY_STRATUM) -> set[str]:
    """Activations with a scored pair in ``stratum`` whose Copernicus map holds flood on the scored pixels."""
    return {r["activation"] for r in ledger.state().values()
            if r.get("status") == "scored" and r.get("stratum") == stratum and r.get("reference_flood")}


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rule() -> dict[str, Any]:
    from floodfusion import cems_benchmark as cb

    return {"window_h": cb.WINDOW_H, "min_coverage": cb.MIN_COVERAGE, "reference_gap_days": cb.REFERENCE_GAP_DAYS,
            "pre_event_window_days": cb.PRE_EVENT_WINDOW_DAYS, "related_lookback_days": cb.RELATED_LOOKBACK_DAYS,
            "requires_package": True, "first_code": FIRST_CODE, "settle_days": SETTLE_DAYS}


def freeze_manifest(spec_path: Path, tag: str) -> dict[str, Any]:
    """What the freeze pins: the tag, the census rule, and by sha256 every file a blind score is computed from.

    Per seed of every model: the weights (``best.pt``), the calibration and τ* (``eval_test.json``), the run record
    (``run.json``, which holds the configuration and normalisation statistics), and the SAR-only calibration of its
    sibling evaluation, which windows without usable optical are read with. A missing file stops the freeze.
    """
    from floodfusion.case_study import SAR_ONLY_SIBLINGS
    from floodfusion.cems_scores import load_spec

    spec = load_spec(spec_path)
    files = {}
    for m in spec["models"]:
        for run in map(Path, m.runs):
            needed = [run / name for name in ("best.pt", "eval_test.json", "run.json")]
            sibling = SAR_ONLY_SIBLINGS.get(run.parent.name)
            if sibling:
                needed.append(run.parent.parent / sibling / run.name / "eval_test.json")
            missing = [str(p) for p in needed if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"cannot freeze {m.key}: missing {missing}")
            files.update({str(p): _sha256(p) for p in needed})
    return {"tag": tag, "rule": _rule(), "spec": str(spec_path), "spec_sha256": _sha256(spec_path), "files": files}


def check_freeze(manifest: Mapping[str, Any], *, git: Callable[[Sequence[str]], int] | None = None,
                 untracked: Callable[[], str] | None = None) -> list[str]:
    """Problems that forbid a sweep (empty: the frozen system is the one about to run)."""
    def run_git(args: Sequence[str]) -> int:
        return subprocess.run(["git", *args], check=False, capture_output=True).returncode

    def git_status() -> str:   # untracked files too: `git diff` ignores a new file under src or configs
        return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *FROZEN_PATHS],
                              check=False, capture_output=True, text=True).stdout

    git, untracked = git or run_git, untracked or git_status
    problems = [f"{path}: sha256 differs from the freeze" for path, sha in manifest["files"].items()
                if not Path(path).is_file() or _sha256(Path(path)) != sha]
    if _sha256(Path(manifest["spec"])) != manifest["spec_sha256"]:
        problems.append(f"{manifest['spec']}: the test spec changed since the freeze")
    if manifest["rule"] != _rule():
        problems.append("the census rule changed since the freeze")
    if git(["diff", "--quiet", manifest["tag"], "--", *FROZEN_PATHS]) != 0:
        problems.append(f"code paths differ from {manifest['tag']}")
    if untracked().strip():
        problems.append("uncommitted or untracked files under the code paths")
    return problems


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    root: Path = Path("data/prospective")
    ledger: Path = Path("reports/prospective/ledger.jsonl")

    @property
    def cache(self) -> Path:
        return self.root / "records"

    @property
    def sealed(self) -> Path:
        return self.root / "sealed"


def _row(key: str, activation: Mapping[str, Any], status: str, now: datetime, **extra: Any) -> dict[str, Any]:
    return {"key": key, "activation": activation["code"], "event_date": str(activation["eventTime"])[:10],
            "status": status, "at": now.isoformat(timespec="seconds"), **extra}


def _record_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()


def sweep(paths: Paths, spec: Mapping[str, Any], *, now: datetime, fetch: Callable[[str], dict[str, Any]],
          client: Any, prepare: Callable[..., Any], score_pair: Callable[[Path], dict[str, Any]],
          first: int = FIRST_CODE, stop_after: int = STOP_AFTER_NON_PUBLIC) -> dict[str, int]:
    """One pass of the blind test (see the module docstring); returns counts by status. Idempotent.

    ``first`` and ``stop_after`` exist for :func:`main`'s ``replay`` (one known flood, outside the blind set).
    """
    from floodfusion.cems_benchmark import eligible_pairs, pair_key, select_pairs

    ledger = Ledger(paths.ledger)
    state = ledger.state()
    records = census(paths.cache, fetch=fetch, now=now, first=first, stop_after=stop_after)
    floods = [r for r in records if str(r.get("category", "")).lower() == "flood"]
    counts: dict[str, int] = {}
    for record in floods:
        if not settled(record, now):
            ledger.append(_row(record["code"], record, "settling", now))
            counts["settling"] = counts.get("settling", 0) + 1
            continue
        pairs = select_pairs([{**vars(c), "key": None} for c in eligible_pairs(record, client, others=records)])
        if not pairs:
            ledger.append(_row(record["code"], record, "not eligible", now, record_sha256=_record_sha256(record)))
            counts["not eligible"] = counts.get("not eligible", 0) + 1
            continue
        for pair in pairs:
            key = pair_key(pair)
            if state.get(key, {}).get("status") == "scored":
                continue
            prepare(dict(pair, key=key), record)
            scores = score_pair(paths.root / key)
            # Whether the Copernicus map holds flood on the scored pixels: a property of the reference, not of any
            # model, so it is safe to commit. It decides which activations the analysis (and so the stopping rule)
            # can use.
            reference_flood = any(e.get("flood_px", 0) > 0 for e in scores.get("pairs", {}).values())
            ledger.append(_row(key, record, "scored", now, record_sha256=_record_sha256(record),
                               sealed_sha256=seal(scores, paths.sealed, key), stratum=pair["stratum"],
                               reference_flood=reference_flood))
            counts["scored"] = counts.get("scored", 0) + 1
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The v1.2 blind test (prospective CEMS flood activations).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fz = sub.add_parser("freeze", help="write the freeze manifest (run once, at the freeze tag)")
    fz.add_argument("--spec", type=Path, default=Path("configs/datasets/prospective_v12.yaml"))
    fz.add_argument("--tag", default="v1.2.0-freeze")
    fz.add_argument("--out", type=Path, default=Path("reports/v12_freeze.json"))
    sw = sub.add_parser("sweep", help="one census + eligibility + scoring pass (daily)")
    sw.add_argument("--freeze", type=Path, default=Path("reports/v12_freeze.json"))
    sw.add_argument("--device", default=None)
    st = sub.add_parser("status", help="counts and statuses; never scores")
    st.add_argument("--ledger", type=Path, default=Paths().ledger)
    rp = sub.add_parser("replay", help="one known flood (outside the blind set) through the whole runner, into a "
                                       "scratch root: a rehearsal before the freeze; no freeze check")
    rp.add_argument("--code", default="EMSR664")
    rp.add_argument("--root", type=Path, required=True, help="scratch folder (never the blind test's)")
    rp.add_argument("--spec", type=Path, default=Path("configs/datasets/prospective_v12.yaml"))
    rp.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "freeze":
        from floodfusion.runs import write_json

        write_json(args.out, freeze_manifest(args.spec, args.tag))
        return 0
    if args.cmd == "status":
        state = Ledger(args.ledger).state()
        for key, row in sorted(state.items()):
            print(f"{key}\t{row['event_date']}\t{row['status']}")
        return 0
    if args.cmd == "replay":
        code = args.code.upper()
        if not code.startswith("EMSR") or int(code[4:]) >= FIRST_CODE:
            raise SystemExit(f"replay is for floods outside the blind set (codes below EMSR{FIRST_CODE}), not {code}")
        paths, spec_path = Paths(root=args.root / "prospective", ledger=args.root / "ledger.jsonl"), args.spec
    else:
        manifest = json.loads(args.freeze.read_text(encoding="utf-8"))
        problems = check_freeze(manifest)
        if problems:
            raise SystemExit("refusing to sweep:\n  " + "\n  ".join(problems))
        paths, spec_path = Paths(), Path(manifest["spec"])
    from floodfusion.cems_benchmark import prepare_pair
    from floodfusion.cems_scores import load_spec
    from floodfusion.cems_scores_v12 import score
    from floodfusion.config import load_config
    from floodfusion.data import gfm, seasonal, sidecars
    from floodfusion.data.archive import IngestSources
    from floodfusion.data.cems import fetch_activation

    spec = load_spec(spec_path)
    sources = IngestSources.open()
    cfg = load_config(Path("configs/base.yaml"))

    def prepare(pair: dict[str, Any], record: Mapping[str, Any]) -> None:
        prepare_pair(pair, record, paths.root, cfg, sources)
        for d in sorted((paths.root / pair["key"] / "events").glob("*")):
            if (d / "event.json").is_file():
                gfm.build_sidecar(d)
                seasonal.seasonal_reference(d, sources.client)
                sidecars.build_dem(d, sources.client)

    def score_pair(pair_dir: Path) -> dict[str, Any]:
        staging = paths.root / "_one"
        staging.mkdir(parents=True, exist_ok=True)
        link = staging / pair_dir.name
        if not link.exists():
            link.symlink_to(pair_dir.resolve())
        try:
            return score(spec, staging, args.device, records=paths.cache)   # event time and extent: clusters
        finally:
            link.unlink()

    if args.cmd == "replay":
        def only(c: str) -> dict[str, Any]:
            if c != code:
                raise LookupError(c)                                  # the census stops right after the target
            return fetch_activation(c)

        counts = sweep(paths, spec, now=datetime.now(UTC), fetch=only, client=sources.client, prepare=prepare,
                       score_pair=score_pair, first=int(code[4:]), stop_after=1)
        problems = verify_sealed(Ledger(paths.ledger), paths.sealed)
        LOGGER.info("replay of %s: %s; sealed files %s", code, counts, problems or "verified")
        return 1 if problems or not counts.get("scored") else 0
    counts = sweep(paths, spec, now=datetime.now(UTC), fetch=fetch_activation, client=sources.client,
                   prepare=prepare, score_pair=score_pair)
    LOGGER.info("sweep: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

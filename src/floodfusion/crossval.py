r"""Cross-validation over flood events: leave-one-event-out, and the leaderboard's own split.

A single fixed split of a real dataset offers few test basins; Sen1Floods11 offers flood
**events** on five continents (eleven published; ten have the same-orbit Sentinel-1 pair
this model needs — Paraguay's acquisition is not in the archive). Two protocols are run with
the same rows and seeds:

``loeo``
    One fold per event: the event's evaluation-grid chips are the test set, the next event
    (sorted order, cyclic — fixed before any run) is the validation set, every chip of the
    other events trains. Each event is tested exactly once, by a model that never saw it, so
    the per-event confusion matrices of all folds pool into one table with K = number of
    events (10) — enough for the per-event permutation tests the synthetic table had to
    withhold (:data:`floodfusion.reporting.MIN_BASINS_FOR_VERDICT`).
``official``
    The dataset's own hand-labelled split: train, validation and test are drawn from the
    **same events**; Bolivia is the only held-out event. One model is trained and scored
    per event on both — the within-event test chips (the leakage row of the ablation, on a
    public leaderboard) and Bolivia. :mod:`floodfusion.cv_report` measures the protocol gap
    on the *same test chips*.

Normalisation statistics are recomputed on each fold's training chips only. Folds share
everything else: configuration, seeds, the decision threshold rule, the calibration.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floodfusion.data.records import ChipRecord, atomic_write_bytes
from floodfusion.splits import STRATEGY_LOBO, STRATEGY_PUBLISHED, SplitAssignment

__all__ = ["CV_DIR", "LEARNED_CV_ROWS", "Fold", "all_groups_split", "assign_group_folds", "folds_on_disk",
           "k_fold_groups", "leave_one_group_out", "main", "official_splits", "run_cv", "write_folds"]

LOGGER = logging.getLogger(__name__)

CV_DIR = "cv"
BASELINE_CV_ROWS = (1, 2, 3)
#: Row 11 (row 9 evaluated SAR-only) is derived from every row-9 checkpoint.
LEARNED_CV_ROWS = (4, 9)


@dataclass(frozen=True)
class Fold:
    """One fold of one protocol: its split file and its train-only statistics, relative to the chip dir."""

    protocol: str
    name: str
    split_file: str
    norm_file: str
    strategy: str

    def overrides(self) -> list[str]:
        return [f"data.split_file_name={self.split_file}", f"data.norm_stats_file={self.norm_file}",
                f"data.split_strategy={self.strategy}"]


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def leave_one_group_out(records: Sequence[ChipRecord], *, manifest_sha256: str = "") -> dict[str, SplitAssignment]:
    """``{group: split}``: test = the group, validation = the next group, train = the rest.

    Groups are ``record.basin_id`` (one per flood event). Validation and test keep only
    evaluation-grid chips (every pixel scored once); training keeps every window.
    """
    groups = sorted({r.basin_id for r in records})
    if len(groups) < 3:
        raise ValueError(f"leave-one-group-out needs >= 3 groups, got {groups}")
    folds = {}
    for i, test_group in enumerate(groups):
        val_group = groups[(i + 1) % len(groups)]

        def chips(group: str) -> tuple[str, ...]:
            return tuple(sorted(r.chip_id for r in records if r.basin_id == group and r.eval_grid))

        train = tuple(sorted(r.chip_id for r in records if r.basin_id not in (test_group, val_group)))
        folds[test_group] = SplitAssignment(
            train_chips=train, val_chips=chips(val_group), test_chips=chips(test_group),
            train_basins=tuple(g for g in groups if g not in (test_group, val_group)),
            val_basins=(val_group,), test_basins=(test_group,), buffer_m=0.0,
            strategy=STRATEGY_LOBO, manifest_sha256=manifest_sha256,
        )
    return folds


def assign_group_folds(records: Sequence[ChipRecord], k: int) -> dict[str, int]:
    """``{group: fold}``: groups dealt largest first (by window count) to the fold with the fewest windows.

    Deterministic: ties in size break by group name, ties in fold load by fold index.
    """
    counts: dict[str, int] = {}
    for r in records:
        counts[r.basin_id] = counts.get(r.basin_id, 0) + 1
    if len(counts) < k:
        raise ValueError(f"{k}-fold needs >= {k} groups, got {len(counts)}")
    load = [0] * k
    out: dict[str, int] = {}
    for group in sorted(counts, key=lambda g: (-counts[g], g)):
        fold = min(range(k), key=lambda f: (load[f], f))
        out[group] = fold
        load[fold] += counts[group]
    return out


def k_fold_groups(records: Sequence[ChipRecord], k: int = 5, *, manifest_sha256: str = "") -> dict[str, SplitAssignment]:
    """``{"fold<i>": split}``: test = fold i, validation = fold i + 1 (cyclic), train = the others.

    Groups (``record.basin_id``) never straddle folds; validation and test keep only
    evaluation-grid windows, training keeps every window.
    """
    fold_of = assign_group_folds(records, k)
    folds = {}
    for i in range(k):
        val_fold = (i + 1) % k

        def chips(fold: int) -> tuple[str, ...]:
            return tuple(sorted(r.chip_id for r in records if fold_of[r.basin_id] == fold and r.eval_grid))

        def groups(*fs: int, exclude: bool = False) -> tuple[str, ...]:
            return tuple(sorted(g for g, f in fold_of.items() if (f in fs) != exclude))

        folds[f"fold{i}"] = SplitAssignment(
            train_chips=tuple(sorted(r.chip_id for r in records if fold_of[r.basin_id] not in (i, val_fold))),
            val_chips=chips(val_fold), test_chips=chips(i),
            train_basins=groups(i, val_fold, exclude=True), val_basins=groups(val_fold), test_basins=groups(i),
            buffer_m=0.0, strategy=STRATEGY_LOBO, manifest_sha256=manifest_sha256,
        )
    return folds


def all_groups_split(records: Sequence[ChipRecord], k: int = 5, *, manifest_sha256: str = "") -> dict[str, SplitAssignment]:
    """``{"all": split}``: the deployable model — train on folds 0..k-2; fold k-1 calibrates it.

    Fold k-1's **smallest** group (fewest windows; ties by name) is a test set that only satisfies
    the pipeline — splits must be non-empty and disjoint (:func:`floodfusion.splits.load_splits`)
    and that score is never reported; the fold's other groups validate (early stopping,
    calibration). The rule is blind to labels and optical. The model exists to be applied to
    *other* data (the CEMS benchmark).
    """
    fold_of = assign_group_folds(records, k)
    counts: dict[str, int] = {}
    for r in records:
        counts[r.basin_id] = counts.get(r.basin_id, 0) + 1
    last = sorted((g for g, f in fold_of.items() if f == k - 1), key=lambda g: (counts[g], g))
    if len(last) < 2:
        raise ValueError(f"the all-groups split needs >= 2 groups in fold {k - 1}, got {last}")
    test_groups, val_groups = (last[0],), tuple(sorted(last[1:]))

    def chips(groups: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted(r.chip_id for r in records if r.basin_id in groups and r.eval_grid))

    return {"all": SplitAssignment(
        train_chips=tuple(sorted(r.chip_id for r in records if fold_of[r.basin_id] != k - 1)),
        val_chips=chips(val_groups), test_chips=chips(test_groups),
        train_basins=tuple(sorted(g for g, f in fold_of.items() if f != k - 1)),
        val_basins=val_groups, test_basins=test_groups,
        buffer_m=0.0, strategy=STRATEGY_LOBO, manifest_sha256=manifest_sha256)}


def official_splits(records: Sequence[ChipRecord], chip_split: Mapping[str, str], *,
                    source_of: Any, manifest_sha256: str = "") -> dict[str, SplitAssignment]:
    """The dataset's own split, mapped from source chips to our windows: ONE model.

    ``chip_split`` maps a source chip id to ``train | val | test | bolivia``;
    ``source_of(record)`` gives a record's source chip id. The test set is the within-event
    test chips **and** Bolivia, the held-out event: both are scored per event from the same
    model (Bolivia is its own group), so one training run serves both questions. Strategy
    ``published``: groups overlap across splits by design. Listed chips that produced no
    window (dropped at ingest) are counted and reported — the scores are then on a subset.
    """
    parts: dict[str, list[ChipRecord]] = {"train": [], "val": [], "test": [], "bolivia": []}
    unknown = 0
    for r in records:
        split = chip_split.get(source_of(r))
        if split is None:
            unknown += 1
            continue
        parts[split].append(r)
    if unknown:
        LOGGER.warning("official split: %d windows belong to chips outside the published split lists", unknown)
    present = {source_of(r) for r in records}
    for split in parts:
        listed = [c for c, s in chip_split.items() if s == split]
        missing = [c for c in listed if c not in present]
        if missing:
            LOGGER.warning("official split: %d of %d listed %s chips produced no window (e.g. %s) — scored on a subset",
                           len(missing), len(listed), split, ", ".join(missing[:3]))

    def ids(rs: Sequence[ChipRecord], eval_only: bool) -> tuple[str, ...]:
        return tuple(sorted(r.chip_id for r in rs if r.eval_grid or not eval_only))

    def groups(rs: Sequence[ChipRecord]) -> tuple[str, ...]:
        return tuple(sorted({r.basin_id for r in rs}))

    test = parts["test"] + parts["bolivia"]
    return {"official": SplitAssignment(
        train_chips=ids(parts["train"], False), val_chips=ids(parts["val"], True), test_chips=ids(test, True),
        train_basins=groups(parts["train"]), val_basins=groups(parts["val"]), test_basins=groups(test),
        buffer_m=0.0, strategy=STRATEGY_PUBLISHED, manifest_sha256=manifest_sha256)}


def write_folds(chip_dir: Path, cfg: Any, splits: Mapping[str, Mapping[str, SplitAssignment]], *,
                seed: int = 0) -> list[Fold]:
    """Freeze every fold's split and its train-only normalisation statistics under ``chip_dir/cv``."""
    from floodfusion.data.records import load_manifest
    from floodfusion.splits import compute_norm_stats, freeze_splits
    from floodfusion.utils.env import git_state

    records = {r.chip_id: r for r in load_manifest(chip_dir / cfg.data.manifest_name)}
    out_dir = chip_dir / CV_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    norm_cache: dict[tuple[str, ...], str] = {}
    for protocol, by_fold in splits.items():
        for name, split in sorted(by_fold.items()):
            fold = Fold(protocol, name, f"{CV_DIR}/{protocol}__{name}.json", "", split.strategy)
            norm_file = norm_cache.get(split.train_chips)
            if norm_file is None:
                norm = compute_norm_stats([records[c] for c in split.train_chips], cfg.norm, cfg.grid, seed=seed)
                norm_file = f"{CV_DIR}/{protocol}__{name}.norm_stats.json"
                atomic_write_bytes(chip_dir / norm_file, json.dumps({
                    "sar_mean_db": list(norm.sar_mean_db), "sar_std_db": list(norm.sar_std_db),
                    "computed_on": f"train split of {fold.split_file}", "manifest_sha256": split.manifest_sha256,
                }, indent=2).encode("utf-8"))
                norm_cache[split.train_chips] = norm_file
            fold = Fold(protocol, name, fold.split_file, norm_file, split.strategy)
            freeze_splits(split, chip_dir / fold.split_file, metadata={
                "protocol": protocol, "fold": name, "norm_stats_file": norm_file, **git_state(),
                "note": "groups are separate flood events (thousands of km apart): no buffer is needed "
                        "between groups; official folds are within-event by the dataset's design",
            })
            folds.append(fold)
    # Merge: re-freezing one protocol (e.g. without --raw) must not drop the others' folds.
    path = out_dir / "folds.json"
    kept = [f for f in json.loads(path.read_text(encoding="utf-8"))
            if f["protocol"] not in splits] if path.is_file() else []
    atomic_write_bytes(path, json.dumps(kept + [vars(f) for f in folds], indent=2).encode("utf-8"))
    return folds


def folds_on_disk(chip_dir: Path, protocol: str | None = None) -> list[Fold]:
    folds = [Fold(**f) for f in json.loads((chip_dir / CV_DIR / "folds.json").read_text(encoding="utf-8"))]
    return [f for f in folds if protocol is None or f.protocol == protocol]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _fold_config(configs_dir: Path, row: int, overlays: Sequence[Path], fold: Fold, seed: int | None) -> Any:
    from floodfusion.config import load_config

    overrides = fold.overrides() + ([f"train.seed={seed}"] if seed is not None else [])
    return load_config(configs_dir / f"{_config_name(row)}.yaml", overlays=overlays, overrides=overrides)


#: Rows run by the CV driver beyond the 13 of the ablation: 14 = the TerraMind comparison;
#: 15-17 = the v1.1 fusion variants (descriptive ablations, ``reports/protocol_v1_1.md``).
EXTRA_ROWS = {14: "comparisons/terramind_small", 15: "variants/row09b_gate_bias0",
              16: "variants/row09c_cloud_augment", 17: "variants/row09r_sar_query",
              18: "variants/row02e_event_fallback", 19: "variants/row02s_seasonal",
              # v1.2: row 20, the reference design with context (both, seasonal only, AlphaEarth only) ...
              20: "variants/row20_fusion_context", 21: "variants/row21_fusion_context_seasonal",
              22: "variants/row22_fusion_context_aef",
              # ... and the fair TerraMind (siamese pre/post; base + LoRA, base frozen, small + LoRA).
              24: "comparisons/terramind_base_fair", 25: "comparisons/terramind_base_fair_frozen",
              26: "comparisons/terramind_small_fair"}


def _config_name(row: int) -> str:
    from floodfusion.ablation import ROW_CONFIGS

    return {**ROW_CONFIGS, **EXTRA_ROWS}[row]


#: Rows whose checkpoints are also evaluated SAR-only, and the run folder that evaluation goes to.
SAR_ONLY_EVALUATED = {9: lambda name: "row11_fusion_eval_sar_only",
                      **{row: (lambda name: f"{name}__eval_sar_only") for row in (14, 20, 21, 22, 24, 25, 26)}}

#: The reference row: under the affordable protocol it alone gets every seed.
REFERENCE_ROW = 9


#: v1.1 variants (9b, 9c, 9r): descriptive ablations, one seed, cross-validation only
#: (``reports/protocol_v1_1.md``) — never trained on the published split or the all-groups protocol.
CV_ONLY_ROWS = frozenset({15, 16, 17})


def seeds_for(row: int, seeds: Sequence[int], protocol: str) -> list[int]:
    """Seeds a learned row is trained with: all for the reference row (or under 'full'); one for a variant."""
    if protocol not in ("affordable", "full"):
        raise ValueError(f"unknown seed protocol {protocol!r}")
    if row in CV_ONLY_ROWS:
        return list(seeds[:1])
    return list(seeds) if protocol == "full" or row == REFERENCE_ROW else list(seeds[:1])


def _run_fold(fold: Fold, args: argparse.Namespace, device: Any) -> None:
    from floodfusion.ablation import _finished
    from floodfusion.baseline_runner import run_baseline_row
    from floodfusion.evaluate import evaluate_run
    from floodfusion.runs import run_dir_for
    from floodfusion.train import fit, load_training_data

    root = args.runs_root / fold.protocol / fold.name
    for row in args.baseline_rows:
        cfg = _fold_config(args.configs_dir, row, args.overlay, fold, None)
        out = root / cfg.name / "baseline"
        if not _finished(out):
            run_baseline_row(cfg, args.data_root, out, attribution=row in (2, 19))
    for row in args.learned_rows:
        if row in CV_ONLY_ROWS and fold.protocol in ("official", "all"):
            continue
        for seed in seeds_for(row, args.seeds, args.seed_protocol):
            cfg = _fold_config(args.configs_dir, row, args.overlay, fold, seed)
            run_dir = run_dir_for(root, cfg.name, seed)
            if not _finished(run_dir):
                fit(load_training_data(cfg, args.data_root), run_dir, device=device, provenance={
                    "config_path": str(args.configs_dir / f"{_config_name(row)}.yaml"),
                    "overlays": [str(o) for o in args.overlay], "overrides": [*fold.overrides(), f"train.seed={seed}"],
                    "data_root": str(args.data_root), "driver": "floodfusion.crossval",
                    "cv": {"protocol": fold.protocol, "fold": fold.name},
                })
                if not args.skip_test:
                    evaluate_run(run_dir, data_root=args.data_root, device=device, figures=False)
            if row in SAR_ONLY_EVALUATED and not args.skip_test:
                # The model's no-optical mode, calibrated on SAR-only validation logits: row 9 -> row 11,
                # and TerraMind's (its CEMS windows without optical are read with this calibration).
                out = run_dir_for(root, SAR_ONLY_EVALUATED[row](cfg.name), seed)
                if not _finished(out):
                    evaluate_run(run_dir, data_root=args.data_root, device=device, force_sar_only=True,
                                 out_dir=out, figures=False)


def run_cv(args: argparse.Namespace) -> None:
    from floodfusion.config import load_config
    from floodfusion.train import resolve_device

    cfg = load_config(args.configs_dir / "row09_fusion_crossattn.yaml", overlays=args.overlay)
    chip_dir = args.data_root / Path(cfg.data.chip_root).name
    device = resolve_device(args.device)
    if args.skip_test:
        LOGGER.warning("--skip-test: training only; no test split is scored (development runs)")
    else:
        _freeze_protocol(args, chip_dir, cfg.data.manifest_name)
    if args.fold:   # a requested fold must exist in at least one of the requested protocols
        known = {f.name for protocol in args.protocol for f in folds_on_disk(chip_dir, protocol)}
        unknown = sorted(set(args.fold) - known)
        if unknown:
            raise ValueError(f"unknown folds {unknown} for protocols {args.protocol}; known: {sorted(known)}")
    for protocol in args.protocol:
        folds = folds_on_disk(chip_dir, protocol)
        if args.fold:
            folds = [f for f in folds if f.name in set(args.fold)]
        for i, fold in enumerate(folds, start=1):
            LOGGER.info("== %s fold %d/%d: %s", protocol, i, len(folds), fold.name)
            _run_fold(fold, args, device)


def _protocol_record(args: argparse.Namespace, chip_dir: Path, manifest_name: str) -> dict[str, Any]:
    from floodfusion.config import config_sha256, config_to_dict, load_config
    from floodfusion.data.records import load_manifest_meta
    from floodfusion.utils.env import git_state

    configs = {}
    for row in [*args.baseline_rows, *args.learned_rows]:
        cfg = load_config(args.configs_dir / f"{_config_name(row)}.yaml", overlays=args.overlay)
        configs[str(row)] = config_sha256(config_to_dict(cfg))
    return {"rows": configs, "seeds": {str(r): seeds_for(r, args.seeds, args.seed_protocol) for r in args.learned_rows},
            "seed_protocol": args.seed_protocol, "overlays": [str(o) for o in args.overlay],
            "folds": [vars(f) for f in folds_on_disk(chip_dir)],
            "manifest_sha256": load_manifest_meta(chip_dir / manifest_name)["manifest_sha256"],
            "git_sha": git_state().get("git_sha")}


def _freeze_protocol(args: argparse.Namespace, chip_dir: Path, manifest_name: str) -> None:
    """Write the evaluation protocol once, before any test split is scored; refuse a silent change.

    Rows, their configuration hashes, seeds, folds and the manifest are recorded in
    ``<runs_root>/protocol.json``. A later run with a different protocol must say so
    (``--new-protocol``), so a configuration can never drift after test results were seen.
    """
    record = _protocol_record(args, chip_dir, manifest_name)
    path = args.runs_root / "protocol.json"
    if path.is_file():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        # Rows may be ADDED by a later call (a staged queue); nothing already recorded may change.
        same = {k: frozen.get(k) for k in ("manifest_sha256", "folds")} == \
            {k: record[k] for k in ("manifest_sha256", "folds")} and \
            all(frozen["rows"].get(r, h) == h for r, h in record["rows"].items()) and \
            all(frozen.get("seeds", {}).get(r, s) == s for r, s in record["seeds"].items())
        if not same and not args.new_protocol:
            raise ValueError(f"{path} records a different protocol (rows, seeds, folds or manifest); "
                             "pass --new-protocol to start one deliberately")
        if same:
            record["rows"] = {**frozen["rows"], **record["rows"]}
            record["seeds"] = {**frozen.get("seeds", {}), **record["seeds"]}
    args.runs_root.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, json.dumps(record, indent=2).encode("utf-8"))


def _make_splits(args: argparse.Namespace) -> None:
    from floodfusion.config import load_config
    from floodfusion.data.records import load_manifest, load_manifest_meta
    from floodfusion.data.sen1floods11 import GROUP_PREFIX, official_split

    cfg = load_config(args.configs_dir / "row09_fusion_crossattn.yaml", overlays=args.overlay)
    chip_dir = args.data_root / Path(cfg.data.chip_root).name
    manifest = chip_dir / cfg.data.manifest_name
    records, sha = load_manifest(manifest), load_manifest_meta(manifest)["manifest_sha256"]
    if args.scheme == "kfold":
        splits: dict[str, dict[str, SplitAssignment]] = {
            "kfold": k_fold_groups(records, args.k, manifest_sha256=sha),
            "all": all_groups_split(records, args.k, manifest_sha256=sha)}
        for fold in write_folds(chip_dir, cfg, splits, seed=args.seed):
            LOGGER.info("%s/%s -> %s (%s)", fold.protocol, fold.name, fold.split_file, fold.norm_file)
        return
    splits = {"loeo": leave_one_group_out(records, manifest_sha256=sha)}
    if args.raw is not None:
        splits["official"] = official_splits(records, official_split(args.raw), manifest_sha256=sha,
                                             source_of=lambda r: r.event_id.removeprefix(GROUP_PREFIX))
    for fold in write_folds(chip_dir, cfg, splits, seed=args.seed):
        LOGGER.info("%s/%s -> %s (%s)", fold.protocol, fold.name, fold.split_file, fold.norm_file)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Event-level cross-validation (leave-one-event-out, official split).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("splits", "run", "table"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--configs-dir", type=Path, default=Path("configs"))
        p.add_argument("--overlay", type=Path, action="append", default=[])
        p.add_argument("--runs-root", type=Path, default=Path("runs/sen1floods11"))
    sub.choices["splits"].add_argument("--raw", type=Path, default=None, help="Sen1Floods11 raw root (official split)")
    sub.choices["splits"].add_argument("--seed", type=int, default=0)
    sub.choices["splits"].add_argument("--scheme", choices=("loeo", "kfold"), default="loeo",
                                       help="loeo (+ official with --raw) or grouped k-fold (+ the all-groups model)")
    sub.choices["splits"].add_argument("--k", type=int, default=5)
    run = sub.choices["run"]
    run.add_argument("--protocol", nargs="+", default=["loeo", "official"], choices=("loeo", "official", "kfold", "all"))
    run.add_argument("--fold", nargs="*", default=None)
    run.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    run.add_argument("--seed-protocol", choices=("affordable", "full"), default="affordable",
                     help="affordable (docs/09 2.2): row 9 with every seed, other rows with the first; full: all")
    run.add_argument("--baseline-rows", type=int, nargs="*", default=list(BASELINE_CV_ROWS))
    run.add_argument("--learned-rows", type=int, nargs="*", default=list(LEARNED_CV_ROWS))
    run.add_argument("--device", default=None)
    run.add_argument("--skip-test", action="store_true", help="train only; never score the test split (dev runs)")
    run.add_argument("--new-protocol", action="store_true", help="accept a protocol differing from protocol.json")
    table = sub.choices["table"]
    table.add_argument("--out", type=Path, default=Path("reports/sen1floods11_table.md"))
    table.add_argument("--allow-dirty", action="store_true")
    table.add_argument("--preset", choices=sorted(TABLE_PRESETS), default=None,
                       help="v1.1 reports: the protocol, header and sections of one dataset")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "splits":
        _make_splits(args)
    elif args.command == "run":
        run_cv(args)
    else:
        from floodfusion.cv_report import build_cv_table

        build_cv_table(args.runs_root, out=args.out, allow_dirty=args.allow_dirty,
                       **(TABLE_PRESETS[args.preset] if args.preset else {}))
    return 0


#: v1.1 report presets (``crossval table --preset``): which fold set, and how it is introduced.
TABLE_PRESETS: dict[str, dict[str, Any]] = {
    "sen1floods11": {"v11": True},
    "kurosiwo": {
        "protocol": "kfold", "v11": True,
        "title": "Kuro Siwo, re-ingested — grouped five-fold, flood labels drawn on radar",
        "intro": ["Real data: Kuro Siwo annotation polygons (Bountos et al., NeurIPS 2024; CC-BY-4.0) rasterised on "
                  "our grid, every input re-fetched from Planetary Computer by `floodfusion.data.kurosiwo`; chips "
                  "sampled per event by a seeded rule. Label: **flood** — water the event brought — minus JRC "
                  "permanent water. Folds group events whose AOIs lie within 100 km; each group is tested once by "
                  "a model that never saw it; validation = the next fold. Protocol: `reports/protocol_v1_1.md`."],
        "caveats": ["**Folds hold several groups.** The oracle-threshold IoU, AP and the other fold-averaged columns "
                    "of the main table are fold-level quantities (each fold weighted equally), not per-group macros; "
                    "IoU at τ* and every gap are per group. The baseline variants' event-pooled threshold pools a "
                    "split group (events within 100 km), not one acquisition.",
                    "**Labels drawn on radar.** Kuro Siwo's annotators drew on Sentinel-1: gaps involving optical "
                    "understate optical's value here, as Sen1Floods11's overstate it; the CEMS benchmark is the "
                    "independent check.",
                    "**Descriptive.** The confirmatory tests of v1.1 are on the CEMS benchmark; nothing here is "
                    "adjusted into that family."],
    },
    "pool_v12": {
        "protocol": "kfold", "v11": True,
        "title": "v1.2 training pool — Kuro Siwo (rebuilt) + GEOID-Flood, grouped five-fold, flood labels",
        "intro": ["Real data: Kuro Siwo annotation polygons (Bountos et al., NeurIPS 2024; CC-BY-4.0), re-ingested "
                  "with the DEM fix, pooled with the GEOID-Flood events before EMSR656 that Kuro Siwo lacks "
                  "(Chiriaco et al., 2026; CC-BY-4.0); every input re-fetched from Planetary Computer. 7,006 chips "
                  "from 138 floods. Label: **flood**, minus permanent water. Folds group events within 100 km "
                  "across both sets (81 groups); each group is tested once by a model that never saw it; "
                  "validation = the next fold. One seed per row; each family at its tuned learning rate. "
                  "Protocol: `reports/protocol_v1_2.md`; rules: `reports/v1_2_deviations.md`."],
        "caveats": ["**Folds hold several groups.** The oracle-threshold IoU, AP and the other fold-averaged columns "
                    "of the main table are fold-level quantities (each fold weighted equally), not per-group macros; "
                    "IoU at τ* and every gap are per group. The cross-fitted section below compares thresholds per "
                    "group.",
                    "**Single seed, exploratory.** Model selection used the whole-stack measure "
                    "(`reports/v12_selection.json`); v1.2's confirmatory tests are the blind ones."],
    },
    "kurosiwo_water": {
        "protocol": "kfold", "v11": True,
        "title": "Kuro Siwo with the 'water' label — the same chips, label = flood ∪ reference water",
        "intro": ["Identical chips, folds and inputs to the flood-label table; only the label differs (Kuro Siwo's "
                  "`hydro` layer added to its flood), the Sen1Floods11-like 'water on a date'. Scores here are on "
                  "that label and are not comparable with the flood-label table; the comparison of the two "
                  "training labels is test P3 on the CEMS benchmark (`reports/protocol_v1_1.md`)."],
        "caveats": ["**Descriptive.** See the flood-label table."],
    },
}


if __name__ == "__main__":
    raise SystemExit(main())

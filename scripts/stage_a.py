"""Stage A: diagnostics of the released v1.0 models on Sen1Floods11 — no retraining, no change to src/.

Every number comes from the v1.0 checkpoints (``runs/sen1floods11/loeo/<event>/...``) on the test chips
of the fold that held each event out, scored exactly as ``floodfusion.evaluate`` does (64 px context,
calibrated probability ``sigmoid((z + c) / T)`` with each run's own T and c).

Subcommands (run in order; each writes under ``--cache`` and/or ``reports/``):

``predict``        one inference pass: every model's calibrated probability per held-out window, the
                   labels, validity and optical quality; for row 9 also the optical contribution (A1):
                   the logit change and the fused-feature change when optical is replaced by the
                   missing-modality embedding, and the gate with and without optical.
``clouds``         A2: real cloud shapes pasted onto clear chips ("missing": quality 0, optical 0;
                   "real": the donor's cloudy reflectance and quality), IoU on masked and unmasked
                   pixels and the clear-pixel error against distance to the cloud edge.
``report``         A1, A3 (strata and the seasonal decomposition), A4 (registration), A5 (conformal):
                   ``reports/stage_a_diagnostics.md`` + ``.json`` from the cache alone.

Run from a checkout of release v1.0 code (this script imports floodfusion at that commit):
``uv run --frozen python scripts/stage_a.py predict --root <repo with data/ and runs/>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("stage_a")

#: (run folder, seeds, force SAR-only, calibration folder). Row 11 is row 9's checkpoint, SAR-only,
#: with its own calibration (refit on SAR-only validation logits in v1.0).
MODELS = (
    ("row04_unet_sar_only", (42,), False, "row04_unet_sar_only"),
    ("row09_fusion_crossattn", (42, 43, 44), False, "row09_fusion_crossattn"),
    ("row09_fusion_crossattn", (42, 43, 44), True, "row11_fusion_eval_sar_only"),
    ("cmp_terramind_small", (42,), False, "cmp_terramind_small"),
)
CONTEXT_PX = 64
TAU_STAR = 0.25
DARK_DB = -15.0           # "dark" in VV, as the v1.0 label diagnostic
CHANGE_DB = -3.0          # a drop of at least 3 dB counts as "darkened"


def model_key(calib_folder: str, seed: int) -> str:
    return f"{calib_folder}/seed{seed}"


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def _calibration(run_dir: Path) -> tuple[float, float]:
    cal = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))["calibration"]
    return float(cal["temperature"]), float(cal["offset"])


def _fusion_module(model: Any) -> Any:
    from floodfusion.models.fusion_module import CrossAttentionFusion

    return next((m for m in model.modules() if isinstance(m, CrossAttentionFusion)), None)


def predict(args: argparse.Namespace) -> None:
    import torch

    from floodfusion.data.dataset import iter_batches_to_device
    from floodfusion.evaluate import _load, _loader
    from floodfusion.train import _autocast, _crop, model_inputs, resolve_device

    device = resolve_device(args.device)
    root, cache = Path(args.root), Path(args.cache)
    data_root = root / "data" / "sen1floods11"
    for fold_dir in sorted((root / "runs" / "sen1floods11" / "loeo").iterdir()):
        event = fold_dir.name
        out = cache / event
        out.mkdir(parents=True, exist_ok=True)
        for folder, seeds, sar_only, calib in MODELS:
            for seed in seeds:
                target = out / f"{calib}_seed{seed}.npz"
                if target.is_file():
                    continue
                run_dir = fold_dir / folder / f"seed{seed}"
                temperature, offset = _calibration(fold_dir / calib / f"seed{seed}")
                cfg, data, model, _, _ = _load(run_dir, data_root, device, sar_only)
                fusion = _fusion_module(model) if (folder.startswith("row09") and not sar_only) else None
                captured: dict[str, Any] = {}
                hook = None
                if fusion is not None:
                    def grab(module: Any, inputs: Any, output: Any, store: dict[str, Any] = captured) -> None:
                        store["sar"], store["fused"], store["gate"] = inputs[0], output[0], output[1]
                    hook = fusion.register_forward_hook(grab)
                rows: dict[str, list[Any]] = defaultdict(list)
                with torch.inference_mode():
                    for batch in iter_batches_to_device(_loader(data.test, cfg), device,
                                                        channels_last=cfg.train.channels_last):
                        with _autocast(device, cfg):
                            z, _ = model(**model_inputs(batch, batch["optical_available"], force_sar_only=sar_only))
                        z = _crop(z.float(), CONTEXT_PX)
                        prob = torch.sigmoid((z + offset) / temperature)
                        rows["chip_id"] += [str(c) for c in batch["chip_id"]]
                        rows["prob"].append(prob[:, 0].cpu().numpy().astype(np.float16))
                        if not (out / "truth.npz").is_file():
                            rows["label"].append(_crop(batch["label"], CONTEXT_PX)[:, 0].cpu().numpy().astype(np.uint8))
                            rows["valid"].append(_crop(batch["valid"], CONTEXT_PX)[:, 0].cpu().numpy())
                            rows["quality"].append(_crop(batch["optical_quality"], CONTEXT_PX)[:, 0].cpu().numpy()
                                                   .astype(np.float16))
                            rows["available"].append(batch["optical_available"].cpu().numpy())
                        if fusion is not None:
                            sar_f, fused_opt, gate_opt = captured["sar"], captured["fused"], captured["gate"]
                            with _autocast(device, cfg):
                                z_miss, _ = model(**model_inputs(batch, batch["optical_available"], force_sar_only=True))
                            fused_miss, gate_miss = captured["fused"], captured["gate"]
                            k = CONTEXT_PX * sar_f.shape[-1] // batch["sar"].shape[-1]
                            delta = (fused_opt.float() - fused_miss.float()).norm(dim=1)
                            ratio = delta / sar_f.float().norm(dim=1).clamp_min(1e-6)
                            rows["contribution"].append(_crop(ratio, k).cpu().numpy().astype(np.float16))
                            rows["gate_opt"].append(_crop(gate_opt.float(), k)[:, 0].cpu().numpy().astype(np.float16))
                            rows["gate_miss"].append(_crop(gate_miss.float(), k)[:, 0].cpu().numpy().astype(np.float16))
                            rows["dlogit"].append((z - _crop(z_miss.float(), CONTEXT_PX))[:, 0].cpu().numpy()
                                                  .astype(np.float16))
                if hook is not None:
                    hook.remove()
                arrays = {k: (np.asarray(v) if k == "chip_id" else np.concatenate(v)) for k, v in rows.items()
                          if k not in ("label", "valid", "quality", "available")}
                np.savez_compressed(target, **arrays)
                if rows["label"]:
                    np.savez_compressed(out / "truth.npz", chip_id=np.asarray(rows["chip_id"]),
                                        label=np.concatenate(rows["label"]), valid=np.concatenate(rows["valid"]),
                                        quality=np.concatenate(rows["quality"]),
                                        available=np.concatenate(rows["available"]))
                LOGGER.info("%s %s seed %d: %d windows", event, calib, seed, len(arrays["chip_id"]))
                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None


# ---------------------------------------------------------------------------
# clouds (A2)
# ---------------------------------------------------------------------------
COVERS = (0.2, 0.4, 0.6, 0.8)
#: A target chip is "clear" with at least this share of clear optical (quality >= 0.5) in its input.
CLEAR_TARGET = 0.95


def _donors(datasets: Sequence[Any]) -> dict[float, list[tuple[np.ndarray, np.ndarray]]]:
    """Real cloud masks with the cloudy optical under them, binned by cover (input window incl. context)."""
    bins: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in COVERS}
    for ds in datasets:
        for i in range(len(ds)):
            s = ds[i]
            if not bool(s["optical_available"]):
                continue
            q = s["optical_quality"][0].numpy()
            cloud = q < 0.5
            cover = float(cloud.mean())
            for c in COVERS:
                if abs(cover - c) <= 0.1 and len(bins[c]) < 40:
                    bins[c].append((cloud, s["optical"].numpy()))
    return bins


def clouds(args: argparse.Namespace) -> None:
    import torch

    from floodfusion.data.dataset import FloodChipDataset
    from floodfusion.evaluate import _load
    from floodfusion.train import _autocast, _crop, resolve_device

    device = resolve_device(args.device)
    root, cache = Path(args.root), Path(args.cache)
    data_root = root / "data" / "sen1floods11"
    folds = sorted((root / "runs" / "sen1floods11" / "loeo").iterdir())
    results: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    for fold_dir in folds:
        event = fold_dir.name
        for seed in (42, 43, 44):
            run_dir = fold_dir / "row09_fusion_crossattn" / f"seed{seed}"
            temperature, offset = _calibration(run_dir)
            t11, c11 = _calibration(fold_dir / "row11_fusion_eval_sar_only" / f"seed{seed}")
            cfg, data, model, _, _ = _load(run_dir, data_root, device, False)
            test = FloodChipDataset(data.test, cfg.norm, cfg.data, cfg.grid, context_px=CONTEXT_PX)
            if seed == 42:
                donors = _donors([FloodChipDataset(data.train[:400], cfg.norm, cfg.data, cfg.grid,
                                                   context_px=CONTEXT_PX)])
                results.setdefault("_donors", {})[event] = {str(c): len(v) for c, v in donors.items()}
            for i in range(len(test)):
                s = test[i]
                q = s["optical_quality"][0].numpy()
                inner_q = q[CONTEXT_PX:-CONTEXT_PX, CONTEXT_PX:-CONTEXT_PX]   # the scored window, not its context
                if not bool(s["optical_available"]) or float((inner_q >= 0.5).mean()) < CLEAR_TARGET:
                    continue
                label = _crop(s["label"], CONTEXT_PX)[0].numpy() > 0.5
                valid = _crop(s["valid"], CONTEXT_PX)[0].numpy().astype(bool)
                for cover in (0.0, *COVERS, 1.0):
                    pool = donors.get(cover, []) if 0.0 < cover < 1.0 else [None]
                    if not pool:
                        continue
                    donor = pool[int(rng.integers(len(pool)))]
                    for variant in (("none",) if cover == 0.0 else ("missing", "real")):
                        optical, quality = s["optical"].clone(), s["optical_quality"].clone()
                        if cover == 1.0:
                            mask = np.ones(q.shape, dtype=bool)
                        elif donor is not None:
                            mask = donor[0]
                        else:
                            mask = np.zeros(q.shape, dtype=bool)
                        m = torch.from_numpy(mask)
                        if variant == "missing":
                            optical[:, m] = 0.0
                        elif variant == "real" and donor is not None:
                            optical[:, m] = torch.from_numpy(donor[1])[:, m]
                        quality[0, m] = 0.0
                        available = bool((quality > 0).any())
                        batch = {k: v[None].to(device) for k, v in (("sar", s["sar"]), ("terrain", s["terrain"]),
                                                                     ("optical", optical), ("optical_quality", quality))}
                        with torch.inference_mode(), _autocast(device, cfg):
                            z, _ = model(sar=batch["sar"], terrain=batch["terrain"], optical=batch["optical"],
                                         optical_quality=batch["optical_quality"],
                                         optical_available=torch.tensor([available], device=device))
                        z = _crop(z.float(), CONTEXT_PX)[0, 0].cpu().numpy()
                        calib = (t11, c11) if not available else (temperature, offset)
                        pred = 1.0 / (1.0 + np.exp(-(z + calib[1]) / calib[0])) > TAU_STAR
                        inner = _crop(torch.from_numpy(mask)[None], CONTEXT_PX)[0].numpy()
                        _accumulate(results, event, seed, cover, variant, pred, label, valid, inner)
            del model
    _write_json(cache / "clouds.json", results)


def _accumulate(results: dict[str, Any], event: str, seed: int, cover: float, variant: str, pred: np.ndarray,
                label: np.ndarray, valid: np.ndarray, masked: np.ndarray) -> None:
    from scipy import ndimage

    key = f"{cover:.1f}/{variant}"
    slot = results.setdefault(key, {})
    for part, region in (("masked", masked), ("clear", ~masked)):
        r = valid & region
        cm = [int((pred & label & r).sum()), int((pred & ~label & r).sum()), int((~pred & label & r).sum())]
        prev = slot.setdefault(event, {}).setdefault(part, [0, 0, 0])
        slot[event][part] = [a + b for a, b in zip(prev, cm)]
    if masked.any() and (~masked).any():   # clear-pixel error against distance (px) to the cloud edge
        dist = ndimage.distance_transform_edt(~masked)
        errors = (pred != label) & valid & ~masked
        for lo, hi in ((0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)):
            band = valid & ~masked & (dist > lo) & (dist <= hi)
            prev = slot.setdefault("_distance", {}).setdefault(f"{lo}-{hi}", [0, 0])
            slot["_distance"][f"{lo}-{hi}"] = [prev[0] + int((errors & band).sum()), prev[1] + int(band.sum())]


# ---------------------------------------------------------------------------
# report (A1, A3, A4, A5)
# ---------------------------------------------------------------------------
def _iou(tp: int, fp: int, fn: int) -> float:
    d = tp + fp + fn
    return tp / d if d else math.nan


def _load_fold(cache: Path, event: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    truth = dict(np.load(cache / event / "truth.npz"))
    preds = {p.stem.rsplit("_seed", 1)[0] + "/seed" + p.stem.rsplit("_seed", 1)[1]: dict(np.load(p))
             for p in sorted((cache / event).glob("*_seed*.npz"))}
    for key, arr in preds.items():
        if list(arr["chip_id"]) != list(truth["chip_id"]):
            raise ValueError(f"{event} {key}: windows differ from the truth arrays")
    return truth, preds


def _stack_layers(root: Path, chip_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """VV post, VV pre (recent), seasonal VV, label shift and WorldCover for each test window."""
    import rasterio
    from rasterio.windows import Window

    from floodfusion.data.encoding import ChipBand

    out: dict[str, list[np.ndarray]] = defaultdict(list)
    shift: list[list[int]] = []
    for chip in chip_ids:
        stack_id, rc = chip.rsplit("_r", 1)
        row, col = int(rc.split("_c")[0]), int(rc.split("_c")[1])
        d = root / "data" / "sen1floods11" / "events" / stack_id
        win = Window(col, row, 256, 256)
        with rasterio.open(d / "stack.tif") as src:
            post = src.read(int(ChipBand.VV_POST) + 1, window=win).astype(np.float64)
            pre = src.read(int(ChipBand.VV_PRE) + 1, window=win).astype(np.float64)
            lc = src.read(int(ChipBand.LANDCOVER) + 1, window=win) if hasattr(ChipBand, "LANDCOVER") else None
        out["post"].append(np.where(post == -32768, np.nan, post / 100.0))
        out["pre"].append(np.where(pre == -32768, np.nan, pre / 100.0))
        if lc is not None:
            out["landcover"].append(lc)
        seasonal = d / "seasonal.tif"
        if seasonal.is_file():
            with rasterio.open(seasonal) as src:
                out["seasonal"].append(src.read(1, window=win).astype(np.float64))
        else:
            out["seasonal"].append(np.full((256, 256), np.nan))
        meta = json.loads((d / "event.json").read_text(encoding="utf-8"))
        shift.append(list((meta.get("event_shift") or {}).get("label_shift_px") or [0, 0]))
    arrays = {k: np.stack(v) for k, v in out.items()}
    arrays["shift"] = np.asarray(shift, dtype=int)
    return arrays


def _a1(cache: Path, events: Sequence[str]) -> dict[str, Any]:
    """Optical contribution by optical-quality bin and by label, per seed (row 9)."""
    from scipy import ndimage

    out: dict[str, Any] = {}
    for event in events:
        truth, preds = _load_fold(cache, event)
        for key, p in preds.items():
            if "contribution" not in p:
                continue
            k = p["contribution"].shape[-1]
            zoom = 256 // k
            q = ndimage.zoom(truth["quality"].astype(np.float32), (1, 1 / zoom, 1 / zoom), order=1)
            lab = ndimage.zoom(truth["label"].astype(np.float32), (1, 1 / zoom, 1 / zoom), order=1) > 0.5
            has = np.broadcast_to(truth["available"].astype(bool)[:, None, None], q.shape)   # no optical: Δ = 0
            for name, sel in (("q<0.25", q < 0.25), ("0.25<=q<0.75", (q >= 0.25) & (q < 0.75)), ("q>=0.75", q >= 0.75)):
                for lname, lsel in (("water", lab), ("land", ~lab)):
                    s = sel & lsel & has
                    slot = out.setdefault(key, {}).setdefault(f"{name}/{lname}", {"contribution": [], "gate_opt": [],
                                                                                    "gate_miss": [], "n": 0})
                    if s.any():
                        slot["contribution"].append(float(p["contribution"].astype(np.float32)[s].mean()))
                        slot["gate_opt"].append(float(p["gate_opt"].astype(np.float32)[s].mean()))
                        slot["gate_miss"].append(float(p["gate_miss"].astype(np.float32)[s].mean()))
                        slot["n"] += int(s.sum())
    return {key: {b: {m: float(np.mean(v[m])) if v[m] else math.nan for m in ("contribution", "gate_opt", "gate_miss")}
                  | {"n_tokens": v["n"]} for b, v in bins.items()} for key, bins in out.items()}


def _a3(root: Path, cache: Path, events: Sequence[str]) -> dict[str, Any]:
    """Labelled flood split by darkness at the seasonal and the recent reference; TP/FP/FN by stratum."""
    decomposition: dict[str, dict[str, int]] = {}
    strata: dict[str, dict[str, dict[str, list[int]]]] = {}
    for event in events:
        truth, preds = _load_fold(cache, event)
        layers = _stack_layers(root, list(truth["chip_id"]))
        valid = truth["valid"].astype(bool)
        flood = (truth["label"] > 0) & valid
        dark_recent = layers["pre"] < DARK_DB
        dark_season = layers["seasonal"] < DARK_DB
        has_season = np.isfinite(layers["seasonal"])
        dark_after = layers["post"] < DARK_DB
        cells = {
            "dark a year earlier": flood & has_season & dark_season,
            "dry a year earlier, dark at the recent reference": flood & has_season & ~dark_season & dark_recent,
            "dry at both references": flood & has_season & ~dark_season & ~dark_recent,
            "no seasonal reference": flood & ~has_season,
        }
        decomposition[event] = {k: int(v.sum()) for k, v in cells.items()} | {"flood": int(flood.sum())}
        stratum_masks = {
            "new water (dark after, not before)": valid & dark_after & ~dark_recent,
            "already dark at the recent reference": valid & dark_recent,
            "not dark after (radar cannot see it as open water)": valid & ~dark_after,
        }
        for key, p in preds.items():
            pred = p["prob"].astype(np.float32) > TAU_STAR
            lab = truth["label"] > 0
            for sname, smask in stratum_masks.items():
                cm = [int((pred & lab & smask).sum()), int((pred & ~lab & smask).sum()), int((~pred & lab & smask).sum())]
                prev = strata.setdefault(key, {}).setdefault(sname, [0, 0, 0])
                strata[key][sname] = [a + b for a, b in zip(prev, cm)]
    return {"decomposition": decomposition, "strata": strata}


def _edges(label: np.ndarray, width: int) -> np.ndarray:
    from scipy import ndimage

    grown = ndimage.binary_dilation(label, iterations=width)
    shrunk = ndimage.binary_erosion(label, iterations=width)
    return grown & ~shrunk


def _a4(root: Path, cache: Path, events: Sequence[str]) -> dict[str, Any]:
    """Each model's IoU with label edges (±2 px) ignored, and with the registration shift undone."""
    out: dict[str, dict[str, dict[str, list[int]]]] = {}
    for event in events:
        truth, preds = _load_fold(cache, event)
        layers = _stack_layers(root, list(truth["chip_id"]))
        valid = truth["valid"].astype(bool)
        lab = truth["label"] > 0
        edge = np.stack([_edges(x, 2) for x in lab])
        unshifted = np.stack([_unshift(x, *s) for x, s in zip(lab, layers["shift"])])
        unshifted_ok = np.stack([_unshift(np.ones_like(x), *s) for x, s in zip(lab, layers["shift"])])
        for key, p in preds.items():
            pred = p["prob"].astype(np.float32) > TAU_STAR
            for name, (y, m) in {"as published": (lab, valid), "edges ±2 px ignored": (lab, valid & ~edge),
                                 "labels unshifted": (unshifted, valid & unshifted_ok)}.items():
                cm = [int((pred & y & m).sum()), int((pred & ~y & m).sum()), int((~pred & y & m).sum())]
                out.setdefault(key, {}).setdefault(name, {})[event] = cm
    return out


def _unshift(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Undo ``shift_pixels(·, dy, dx)`` (content moved by (-dy, -dx)); vacated pixels are False."""
    out = np.zeros_like(array, dtype=bool)
    h, w = array.shape
    sy, sx = -int(dy), -int(dx)
    ys, yd = (slice(0, h - sy), slice(sy, h)) if sy >= 0 else (slice(-sy, h), slice(0, h + sy))
    xs, xd = (slice(0, w - sx), slice(sx, w)) if sx >= 0 else (slice(-sx, w), slice(0, w + sx))
    out[yd, xd] = array[ys, xs].astype(bool)
    return out


def _a5(cache: Path, events: Sequence[str], alpha: float = 0.1) -> dict[str, Any]:
    """Conformal recall: mean chip FNR as the guarantee defines it, and cross-fitted on the other events."""
    grid = np.unique(np.concatenate([np.geomspace(0.001, 0.05, 40), np.arange(0.05, 0.96, 0.05)]))
    per_model: dict[str, dict[str, list[tuple[list[float], list[tuple[int, int, int]]]]]] = defaultdict(dict)
    for event in events:
        truth, preds = _load_fold(cache, event)
        valid = truth["valid"].astype(bool)
        lab = truth["label"] > 0
        for key, p in preds.items():
            prob = p["prob"].astype(np.float32)
            chips = []
            for i in range(len(prob)):
                pos = lab[i] & valid[i]
                if pos.sum() < 100:
                    continue
                fnr = [float((pos & ~(prob[i] > t)).sum() / pos.sum()) for t in grid]
                stats = [(int(((prob[i] > t) & pos).sum()), int(((prob[i] > t) & ~lab[i] & valid[i]).sum()),
                          int((pos & ~(prob[i] > t)).sum())) for t in grid]
                chips.append((fnr, stats))
            per_model[key][event] = chips
    out: dict[str, Any] = {}
    for key, by_event in per_model.items():
        res = {}
        for event, chips in by_event.items():
            if not chips:
                continue
            calib = [c for e, cs in by_event.items() if e != event for c in cs]
            n = len(calib)
            risk = np.mean([c[0] for c in calib], axis=0)
            ok = np.flatnonzero(n / (n + 1) * risk + 1 / (n + 1) <= alpha)
            if not len(ok):
                res[event] = {"feasible": False}
                continue
            i = int(ok.max())
            tp = sum(c[1][i][0] for c in chips)
            fp = sum(c[1][i][1] for c in chips)
            fn = sum(c[1][i][2] for c in chips)
            res[event] = {"feasible": True, "threshold": float(grid[i]), "n_calibration_chips": n,
                          "mean_chip_fnr": float(np.mean([c[0][i] for c in chips])),
                          "precision": tp / (tp + fp) if tp + fp else math.nan, "iou": _iou(tp, fp, fn)}
        out[key] = res
    return out


def report(args: argparse.Namespace) -> None:
    root, cache = Path(args.root), Path(args.cache)
    events = sorted(p.name for p in cache.iterdir() if (p / "truth.npz").is_file())
    result = {"events": events, "a1_contribution": _a1(cache, events), "a3": _a3(root, cache, events),
              "a4_registration": _a4(root, cache, events), "a5_conformal": _a5(cache, events)}
    if (cache / "clouds.json").is_file():
        result["a2_clouds"] = json.loads((cache / "clouds.json").read_text(encoding="utf-8"))
    _write_json(Path(args.out).with_suffix(".json"), result)
    Path(args.out).write_text(render(result), encoding="utf-8")
    LOGGER.info("wrote %s", args.out)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _f(x: float, d: int = 3) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:.{d}f}"


def render(r: Mapping[str, Any]) -> str:
    lines = ["# Stage A — diagnostics of the v1.0 models (no retraining)", "",
             "Every number below re-reads the released v1.0 checkpoints on the Sen1Floods11 test chips of the fold "
             "that held each event out, scored as `floodfusion.evaluate` scores them (64 px context, each run's own "
             "calibration, τ* = 0.25). Regenerate: `scripts/stage_a.py predict`, `clouds`, `report` (release v1.0 "
             "code; seasonal references from `python -m floodfusion.data.seasonal` of v1.1).", ""]
    lines += _render_a1(r["a1_contribution"]) + _render_a2(r.get("a2_clouds")) + _render_a3(r["a3"])
    lines += _render_a4(r["a4_registration"]) + _render_a5(r["a5_conformal"])
    return "\n".join(lines) + "\n"


def _render_a1(a1: Mapping[str, Any]) -> list[str]:
    lines = ["## A1 — How much optical changes the fused features, by optical quality (row 9)", "",
             "Contribution = ‖fused(optical) − fused(missing embedding)‖ / ‖F_s‖ per fusion-stage token (8 × 8 px), "
             "two passes on the same chip (chips with optical only); the gate is shown with optical and with the "
             "missing embedding. Gate "
             "levels compare only within a model (with vs without optical); across seeds they are not comparable.", "",
             "| Seed | Optical quality | Label | Contribution | Gate with optical | Gate, missing embedding | Tokens |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, bins in sorted(a1.items()):
        for b, v in bins.items():
            q, lab = b.split("/")
            lines.append(f"| {key.split('/')[-1]} | {q} | {lab} | {_f(v['contribution'])} | {_f(v['gate_opt'])} | "
                         f"{_f(v['gate_miss'])} | {v['n_tokens']} |")
    return [*lines, ""]


def _render_a2(a2: Mapping[str, Any] | None) -> list[str]:
    if not a2:
        return ["## A2 — Cloud stress on real chips", "", "(not run)", ""]
    lines = ["## A2 — Real cloud shapes pasted onto clear chips (row 9, three seeds pooled)", "",
             "Donor clouds: real cloud masks from training chips, with their own cloudy reflectance (\"real\") or "
             "with optical zeroed (\"missing\"); quality 0 under the mask in both. At 100 % the model takes the "
             "missing-modality path and its SAR-only calibration (row 11).", "",
             "| Cover | Variant | IoU on masked pixels | IoU on clear pixels |", "| --- | --- | --- | --- |"]
    for key in sorted(k for k in a2 if not k.startswith("_")):
        cover, variant = key.split("/")
        events = a2[key]
        parts = {}
        for part in ("masked", "clear"):
            tot = [sum(e.get(part, [0, 0, 0])[i] for n, e in events.items() if not n.startswith("_")) for i in range(3)]
            parts[part] = _iou(*tot)
        lines.append(f"| {float(cover):.0%} | {variant} | {_f(parts['masked'])} | {_f(parts['clear'])} |")
    lines += ["", "Clear-pixel error rate against distance to the nearest cloud pixel:", "",
              "| Cover / variant | 0-16 px | 16-32 | 32-64 | 64-128 | > 128 |", "| --- | --- | --- | --- | --- | --- |"]
    for key in sorted(k for k in a2 if not k.startswith("_")):
        dist = a2[key].get("_distance")
        if dist:
            cells = [_f(dist[b][0] / dist[b][1]) if dist.get(b, [0, 0])[1] else "—"
                     for b in ("0-16", "16-32", "32-64", "64-128", "128-1000000000.0")]
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def _render_a3(a3: Mapping[str, Any]) -> list[str]:
    lines = ["## A3 — Was the labelled flood there before?", "",
             f"Share of each event's flood-labelled pixels by darkness (VV < {DARK_DB:.0f} dB) at the **seasonal** "
             "reference (same-orbit median within ±30 days of the same date a year earlier) and at the **recent** "
             "reference (the latest pass 5-24 days earlier, v1.0's).", "",
             "| Event | Flood px | Dark a year earlier | Dry then, dark at the recent reference | Dry at both | "
             "No seasonal reference |", "| --- | --- | --- | --- | --- | --- |"]
    for event, d in sorted(a3["decomposition"].items()):
        n = max(d["flood"], 1)
        lines.append(f"| {event.removeprefix('S1F11_')} | {d['flood']:,} | {d['dark a year earlier'] / n:.2f} | "
                     f"{d['dry a year earlier, dark at the recent reference'] / n:.2f} | {d['dry at both references'] / n:.2f} | "
                     f"{d['no seasonal reference'] / n:.2f} |")
    lines += ["", "Where each model's errors and hits fall (all events pooled; counts, since IoU does not add up "
              "across strata). A gain on pixels radar cannot see as open water fits label-source bias *and* real "
              "optical value: it bounds the bias from above, it does not measure it.", "",
              "| Model | Stratum | TP | FP | FN |", "| --- | --- | --- | --- | --- |"]
    for key, strata in sorted(a3["strata"].items()):
        for s, (tp, fp, fn) in strata.items():
            lines.append(f"| {key} | {s} | {tp:,} | {fp:,} | {fn:,} |")
    return [*lines, ""]


def _render_a4(a4: Mapping[str, Any]) -> list[str]:
    lines = ["## A4 — Registration: does the headline survive without the label shift?", "",
             "Macro IoU over events at τ*, as published, with label edges (±2 px, the documented uncertainty) "
             "ignored, and with the registration shift undone. The shift moved labels toward SAR water edges, so it "
             "can only have helped the SAR-only models.", "",
             "| Model | As published | Edges ±2 px ignored | Labels unshifted |", "| --- | --- | --- | --- |"]
    for key, by in sorted(a4.items()):
        cells = [_f(float(np.nanmean([_iou(*cm) for cm in by[n].values()]))) for n in
                 ("as published", "edges ±2 px ignored", "labels unshifted")]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def _render_a5(a5: Mapping[str, Any]) -> list[str]:
    lines = ["## A5 — Conformal recall, measured as it is defined, and cross-fitted", "",
             "Conformal risk control bounds the **expected** per-chip miss rate; it promises nothing per event. "
             "Cross-fitted: each held-out event's threshold is fitted on the chips of the other nine held-out events "
             "(each scored by its own fold model), on a grid down to 0.001. With nine calibration *events*, an "
             "event-level bound at α = 0.1 would need zero risk (vacuous).", "",
             "| Model | Events (feasible) | Events with mean chip FNR ≤ 0.1 | Mean chip FNR | Precision | IoU |",
             "| --- | --- | --- | --- | --- | --- |"]
    for key, res in sorted(a5.items()):
        ok = {e: v for e, v in res.items() if v.get("feasible")}
        lines.append(f"| {key} | {len(ok)}/{len(res)} | {sum(v['mean_chip_fnr'] <= 0.1 for v in ok.values())}/{len(ok)} | "
                     f"{_f(float(np.mean([v['mean_chip_fnr'] for v in ok.values()])) if ok else math.nan)} | "
                     f"{_f(float(np.nanmean([v['precision'] for v in ok.values()])) if ok else math.nan)} | "
                     f"{_f(float(np.nanmean([v['iou'] for v in ok.values()])) if ok else math.nan)} |")
    return [*lines, ""]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("predict", "clouds", "report"))
    parser.add_argument("--root", type=Path, required=True, help="the repository holding data/ and runs/")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/stage_a_diagnostics.md"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    args.cache = args.cache or args.root / "data" / "stage_a"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    {"predict": predict, "clouds": clouds, "report": report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

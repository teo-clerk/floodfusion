"""Quicklooks for the Arena's map explorer: one radar image and one coloured map per model, per curated pair.

    uv run --frozen python scripts/arena_assets.py --spec configs/datasets/cems_models_v12.yaml   # GPU

**Which pairs.** About :data:`N_PAIRS` activations of the CEMS benchmark's primary stratum, drawn by a seeded
generator, blind to every model's score. Within each activation, the tile with the most reference flood (the
Copernicus map, not any model) is shown.

**What each image shows.**
- Sentinel-1 VV after the event, in grey (−25 to 0 dB).
- Inside the region CEMS observed, each model's map at its own τ* against the CEMS map:
  - hit: ``#0072B2``;
  - false alarm: ``#E69F00``;
  - miss: ``#CC79A7``.
- Outside that region, the radar is dimmed and nothing is scored.

Seed 42 of every learned model is shown, and Copernicus GFM's ensemble is read from its sidecar.

**Output.** WebP files (≤ :data:`MAX_SIDE` px), committed as flat names ``assets/arena-<pair>-<layer>.webp`` with
``assets/arena-manifest.json``: a layer per model, and the per-tile IoU of each.

Contains modified Copernicus Emergency Management Service information (2023–2026): rapid-mapping delineations and
Global Flood Monitoring (GFM) flood maps, © European Union. Contains modified Copernicus Sentinel data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_PAIRS = 12
MAX_SIDE = 1024
SEED = "arena-v12"
HIT, FALSE_ALARM, MISS = (0x00, 0x72, 0xB2), (0xE6, 0x9F, 0x00), (0xCC, 0x79, 0xA7)
OUTLINE = (0xFF, 0xD2, 0x3F)


def pick_pairs(case_root: Path, spec: dict[str, Any], n: int = N_PAIRS) -> list[Path]:
    """Seeded draw of one pair per activation (primary stratum, not excluded), blind to every score."""
    pairs: dict[str, list[Path]] = {}
    for f in sorted(case_root.glob("EMSR*_AOI*/pair.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        if p["stratum"] == spec["primary_stratum"] and p["activation"] not in spec["exclude"]:
            pairs.setdefault(p["activation"], []).append(f.parent)
    rng = np.random.default_rng(int(hashlib.sha256(SEED.encode()).hexdigest()[:8], 16))
    codes = sorted(pairs)
    chosen = [codes[i] for i in sorted(rng.permutation(len(codes))[:n])]
    return [sorted(pairs[c])[int(rng.integers(len(pairs[c])))] for c in chosen]


def best_tile(pair_dir: Path) -> tuple[Path, Any, np.ndarray, np.ndarray, np.ndarray] | None:
    """The pair's tile with the most reference flood: (stack, pair record, valid, scored, label)."""
    sys.path.insert(0, str(ROOT / "src"))
    from floodfusion.cems_scores_v12 import tile_masks

    pair = json.loads((pair_dir / "pair.json").read_text(encoding="utf-8"))
    best = None
    for stack in sorted((pair_dir / "events").glob("*/stack.tif")):
        if not stack.with_name("event.json").is_file():
            continue
        valid, scored, label, _ = tile_masks(stack, pair)
        flood = int((label & scored).sum())
        if best is None or flood > best[0]:
            best = (flood, stack, valid, scored, label)
    return None if best is None or best[0] == 0 else (best[1], pair, best[2], best[3], best[4])


def radar_grey(stack: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(stack) as src:
        vv = src.read(1).astype(np.float64)
    db = np.where(vv == -32768, np.nan, vv / 100.0)
    grey = np.clip((np.nan_to_num(db, nan=-25.0) + 25.0) / 25.0, 0.0, 1.0)
    return (grey * 255).astype(np.uint8)


def overlay(grey: np.ndarray, flagged: np.ndarray | None, label: np.ndarray, scored: np.ndarray) -> np.ndarray:
    """RGB: radar grey (dimmed outside the scored region) with hits, false alarms and misses where scored."""
    from scipy.ndimage import binary_erosion

    rgb = np.repeat(grey[..., None], 3, axis=2).astype(np.float64)
    rgb[~scored] *= 0.45
    if flagged is not None:
        for mask, colour in ((flagged & label & scored, HIT), (flagged & ~label & scored, FALSE_ALARM),
                             (~flagged & label & scored, MISS)):
            rgb[mask] = colour
    edge = scored & ~binary_erosion(scored, iterations=2)
    rgb[edge] = OUTLINE
    return rgb.astype(np.uint8)


def save_webp(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image

    image = Image.fromarray(rgb)
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.NEAREST)
    image.save(path, "WEBP", quality=82, method=6)


def _iou(flagged: np.ndarray, label: np.ndarray, scored: np.ndarray) -> float | None:
    tp = int((flagged & label & scored).sum())
    union = int(((flagged | label) & scored).sum())
    return tp / union if union else None


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from floodfusion.cems_scores import _models, load_spec
    from floodfusion.cems_scores_v12 import GFM_KEY, gfm_maps
    from floodfusion.data.gfm import read_sidecar
    from floodfusion.train import resolve_device

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/datasets/cems_models_v12.yaml")
    parser.add_argument("--case-root", type=Path, default=ROOT / "data/cems_benchmark")
    parser.add_argument("--out", type=Path, default=ROOT / "assets")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    # The models the tests name (learned and baselines), plus the erratum's corrected TerraMind; GFM is read below.
    shown = [m for m in spec["models"] if m.kind != "gfm" and (m.role == "test" or m.key == "s1f11_tm_fixed")]
    models = [(k, r, m) for k, r, m in _models({**spec, "models": shown}, resolve_device(args.device))
              if r in ("seed42", "baseline")]
    names = {m.key: m.name for m in spec["models"]}
    manifest: dict[str, Any] = {"seed": SEED, "colours": {"hit": "#0072B2", "false_alarm": "#E69F00",
                                                          "miss": "#CC79A7", "outline": "#FFD23F"}, "pairs": []}
    for pair_dir in pick_pairs(args.case_root, spec):
        tile = best_tile(pair_dir)
        if tile is None:
            continue
        stack, pair, _, scored, label = tile
        key = pair_dir.name
        grey = radar_grey(stack)
        layers = {"radar": {"name": "Sentinel-1 VV", "iou": None}}
        save_webp(overlay(grey, None, label, scored), args.out / f"arena-{key}-radar.webp")
        save_webp(overlay(grey, label, label, scored), args.out / f"arena-{key}-cems.webp")
        layers["cems"] = {"name": "Copernicus EMS map", "iou": None}
        for model_key, _, model in models:
            flagged = np.nan_to_num(model.predict(stack), nan=0.0) > model.tau
            save_webp(overlay(grey, flagged, label, scored), args.out / f"arena-{key}-{model_key}.webp")
            layers[model_key] = {"name": names[model_key], "iou": _iou(flagged, label, scored)}
        flood, _ = gfm_maps(read_sidecar(stack.parent), scored.shape)
        if np.isfinite(flood[scored]).mean() >= 0.95:
            flagged = np.nan_to_num(flood, nan=0.0) > 0.5
            save_webp(overlay(grey, flagged, label, scored), args.out / f"arena-{key}-{GFM_KEY}.webp")
            layers[GFM_KEY] = {"name": names.get(GFM_KEY, "Copernicus GFM"), "iou": _iou(flagged, label, scored)}
        manifest["pairs"].append({"key": key, "activation": pair["activation"], "aoi": pair["aoi"],
                                  "product": pair["product"], "sensors": pair["sensors"],
                                  "product_time": pair["product_time"], "pass_time": pair["pass_time"],
                                  "tile": stack.parent.name, "flood_km2": round(float((label & scored).sum()) * 1e-4, 2),
                                  "layers": layers})
        print(f"{key}: {len(layers)} layers")
    (args.out / "arena-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent checks of the Sen1Floods11 label/optical registration, on the ingested stacks.

The ingest moves each event's labels and Sentinel-2 by whole-pixel shifts (DATA_CARD.md), and the
label shift comes from the labels' own water edges against the SAR, so label-vs-SAR after the shift
is ~0 by construction. These checks are not:

- **SWIR vs SAR after** — label-free: is the optical on the SAR geometry?
- **label vs SWIR after** — no SAR: do the labels still sit on the optical image?
- **split-half** — the label shift re-estimated from the even-numbered chips only, and the residual it
  would leave on the odd ones;
- **train-only** — the shift re-estimated from published-split training chips only (the published
  split's test labels then play no part);
- **known shift** — every label displaced by a further (2, -3) px must be measured as displaced by it.

Writes ``reports/sen1floods11_registration.md``.
Run: ``uv run python scripts/registration_diagnostic.py --data-root data/sen1floods11 --raw data/sen1floods11_raw``
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.config import ValidityFlag
from floodfusion.data.assembly import (
    label_optical_window_shifts,
    label_window_shifts,
    pooled_shift,
    sar_optical_window_shifts,
    shift_pixels,
)
from floodfusion.data.chipping import CLEAR_QUALITY, read_event_stack
from floodfusion.data.encoding import INDEX_SCALE, QUALITY_SCALE, ChipBand

NODATA = -32768
KNOWN_SHIFT = (2, -3)
CHECKS = ("swir_sar", "label_swir", "label_sar", "known")


def chip_windows(stack: np.ndarray, db_scale: float = 100.0) -> dict[str, np.ndarray]:
    """One ingested chip's window shifts for every check (inputs as stored, i.e. after the shift)."""
    def band(b: ChipBand, scale: float) -> np.ndarray:
        a = stack[b].astype(np.float64) / scale
        a[stack[b] == NODATA] = np.nan
        return a

    vv = band(ChipBand.VV_POST, db_scale)
    raw = stack[ChipBand.LABEL]
    label = np.where(raw == 1, 1.0, np.where(raw == 0, 0.0, np.nan))
    label[(stack[ChipBand.VALIDITY] & ValidityFlag.PERMANENT_WATER) > 0] = np.nan  # stored as 0: not an edge
    clear = band(ChipBand.OPTICAL_QUALITY, QUALITY_SCALE) >= CLEAR_QUALITY
    rho = np.stack([band(b, INDEX_SCALE) for b in (ChipBand.B3_GREEN, ChipBand.B8_NIR, ChipBand.B11_SWIR1)])
    rho[:, ~clear] = np.nan
    out = {"label_sar": label_window_shifts(vv, label),
           "known": label_window_shifts(vv, shift_pixels(label, -KNOWN_SHIFT[0], -KNOWN_SHIFT[1], np.nan))}
    if np.isfinite(rho[2]).mean() > 0.2:
        out["swir_sar"] = sar_optical_window_shifts(vv, rho)
        out["label_swir"] = label_optical_window_shifts(rho, label)
    return out


def _one(path: str) -> tuple[str, dict[str, np.ndarray]]:
    stack, _, _ = read_event_stack(path)
    return Path(path).parent.name.removeprefix("S1F11_"), chip_windows(stack)


def _fmt(est: Mapping[str, Any]) -> str:
    if not est.get("n_grids"):
        return "—"
    dy, dx = est["median_shift"]
    ci = est["ci_halfwidth_px"]
    return f"({dy:+.2f}, {dx:+.2f}) ±{ci:.2f}" if ci == ci else f"({dy:+.2f}, {dx:+.2f})"


def _round(v: float) -> int:
    return int(np.floor(v + 0.5))


def event_row(event: str, chips: Mapping[str, Mapping[str, np.ndarray]], decision: Mapping[str, Any],
              train_ids: set[str]) -> dict[str, Any]:
    """Every check for one event; ``chips`` maps chip id (``Ghana_103272``) to its window shifts."""
    def pool(check: str, ids: Sequence[str] | None = None) -> dict[str, Any]:
        keys = sorted(chips) if ids is None else ids
        return pooled_shift([chips[c][check] for c in keys if check in chips[c]], min_shift_px=0.0)

    applied = decision.get("label_shift_px", [0, 0])
    ids = sorted(chips)
    even, odd = ids[0::2], ids[1::2]
    r_even, r_odd = pool("label_sar", even)["median_shift"], pool("label_sar", odd)["median_shift"]
    s_even = [_round(applied[i] + r_even[i]) if r_even[i] == r_even[i] else applied[i] for i in range(2)]
    odd_residual = [applied[i] + r_odd[i] - s_even[i] for i in range(2)]
    train = [c for c in ids if c in train_ids]
    r_train = pool("label_sar", train)["median_shift"] if train else [float("nan")] * 2
    s_train = ([_round(applied[i] + r_train[i]) for i in range(2)] if train and r_train[0] == r_train[0] else None)
    known = pool("known")["median_shift"]
    label_sar = pool("label_sar")["median_shift"]
    return {
        "event": event, "decision": decision, "swir_sar": pool("swir_sar"), "label_swir": pool("label_swir"),
        "label_sar": pool("label_sar"), "split_half_shift": s_even, "split_half_odd_residual": odd_residual,
        "train_only_shift": s_train, "n_train_chips": len(train),
        "known_recovered": [known[i] - label_sar[i] for i in range(2)],
    }


def render(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Sen1Floods11 registration — independent checks",
        "",
        "After the ingest's per-event shifts (DATA_CARD.md). Offsets are pooled window medians "
        "(dy, dx) in px with the approximate 95 % half-interval from resampling chips. "
        f"*Known shift*: labels displaced by a further {KNOWN_SHIFT} px, measured back.",
        "",
        "| Event | Labels moved (source) | Optical moved (source) | SWIR vs SAR after | Label vs SWIR after "
        "| Split-half shift / odd residual | Train-only shift | Known shift recovered |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        d = r["decision"]
        odd = r["split_half_odd_residual"]
        lines.append(
            f"| {r['event']} | {d.get('label_shift_px')} ({d.get('label_source')}) "
            f"| {d.get('optical_shift_px')} ({d.get('optical_source')}) | {_fmt(r['swir_sar'])} "
            f"| {_fmt(r['label_swir'])} | {r['split_half_shift']} / ({odd[0]:+.2f}, {odd[1]:+.2f}) "
            f"| {r['train_only_shift'] if r['train_only_shift'] is not None else '—'} ({r['n_train_chips']} chips) "
            f"| ({r['known_recovered'][0]:+.2f}, {r['known_recovered'][1]:+.2f}) |")
    lines += ["", "Regenerate: `uv run python scripts/registration_diagnostic.py --data-root <ingested> "
              "--raw <Sen1Floods11 download>`.", ""]
    return "\n".join(lines)


def main() -> int:
    """Read every ingested stack, run the checks per event, write the report."""
    from floodfusion.data.sen1floods11 import official_split

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data/sen1floods11"))
    parser.add_argument("--raw", type=Path, default=Path("data/sen1floods11_raw"))
    parser.add_argument("--out", type=Path, default=Path("reports/sen1floods11_registration.md"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = sorted(str(p) for p in args.data_root.glob("events/S1F11_*/stack.tif"))
    by_event: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    with ProcessPoolExecutor(args.workers) as pool:
        for chip_id, windows in pool.map(_one, paths, chunksize=4):
            by_event[chip_id.rsplit("_", 1)[0]][chip_id] = windows
    train_ids = {c for c, s in official_split(args.raw).items() if s == "train"}
    rows = []
    for event in sorted(by_event):
        meta_path = next(args.data_root.glob(f"events/S1F11_{event}_*/event.json"))
        decision = json.loads(meta_path.read_text(encoding="utf-8")).get("event_shift") or {}
        rows.append(event_row(event, by_event[event], decision, train_ids))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows), encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} events, {len(paths)} chips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

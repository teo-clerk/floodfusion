"""Do Sen1Floods11 flood labels mark a backscatter *change*? Per event, on the re-ingested stacks.

For every flood-labelled, valid pixel: the share already dark (VV < -15 dB) in the same-orbit
pre-event reference, the share dark in the post-event image, and the median VV change. A change
detector (rows 2 and 3) can only find water that darkened; water present on both dates is
invisible to it. Writes ``reports/sen1floods11_label_change.md``.

Run: ``uv run python scripts/label_change_diagnostic.py --data-root data/sen1floods11``
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from floodfusion.config import GridConfig
from floodfusion.data.chipping import read_event_stack
from floodfusion.data.encoding import ChipBand

DARK_DB = -15.0
NODATA = -32768


def event_rows(data_root: Path) -> list[tuple[str, int, float, float, float]]:
    """Per event: (name, flood pixels, dark-before share, dark-after share, median ΔVV dB)."""
    scale = GridConfig().db_scale
    events = sorted({Path(p).parent.name.rsplit("_", 1)[0] for p in glob.glob(str(data_root / "events" / "*" / "stack.tif"))})
    rows = []
    for event in events:
        pre, post, delta = [], [], []
        for path in sorted(glob.glob(str(data_root / "events" / f"{event}_*" / "stack.tif"))):
            stack, _, _ = read_event_stack(path)
            flood = (stack[ChipBand.LABEL] == 1) & (stack[ChipBand.VALIDITY] == 0)
            flood &= (stack[ChipBand.VV_PRE] != NODATA) & (stack[ChipBand.VV_POST] != NODATA)
            if flood.any():
                pre.append(stack[ChipBand.VV_PRE][flood] / scale)
                post.append(stack[ChipBand.VV_POST][flood] / scale)
                delta.append(stack[ChipBand.DELTA_VV][flood] / scale)
        if pre:
            pre_a, post_a, delta_a = np.concatenate(pre), np.concatenate(post), np.concatenate(delta)
            rows.append((event.removeprefix("S1F11_"), len(pre_a), float(np.mean(pre_a < DARK_DB)),
                         float(np.mean(post_a < DARK_DB)), float(np.median(delta_a))))
    return rows


def main() -> int:
    """CLI entry point: write the per-event table and print the headline count."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data/sen1floods11"))
    parser.add_argument("--out", type=Path, default=Path("reports/sen1floods11_label_change.md"))
    args = parser.parse_args()
    rows = event_rows(args.data_root)
    no_drop = sum(r[4] > -1.0 for r in rows)
    lines = [
        "# Sen1Floods11 labels and backscatter change (re-ingested stacks)", "",
        f"Flood-labelled valid pixels per event. **{no_drop} of {len(rows)} events show no clear backscatter "
        "drop** (median ΔVV > −1 dB) against the same-orbit reference acquired up to 60 days before the labelled "
        "image: much of the labelled water was already there. A change-based rule cannot see it; an "
        "amplitude-based one can.", "",
        f"| Event | Flood pixels | Dark before (VV < {DARK_DB:g} dB) | Dark after | Median ΔVV (dB) |",
        "| --- | --- | --- | --- | --- |",
        *[f"| {e} | {n:,} | {a:.2f} | {b:.2f} | {d:+.2f} |" for e, n, a, b, d in rows], "",
        f"Regenerate: `uv run python scripts/label_change_diagnostic.py --data-root {args.data_root}`.", "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}: {no_drop}/{len(rows)} events without a clear drop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

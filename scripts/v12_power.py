"""The blind test's power: the smallest IoU gap it detects at 80 % power, by number of activations.

    uv run --frozen python scripts/v12_power.py      # -> reports/v12_power.md (CPU, a few minutes)

**Assumptions.** Per-activation gaps are drawn i.i.d. normal, with the spread v1.1 measured on the CEMS benchmark
(``reports/cems_benchmark.json``: the per-activation gaps of P4 and P3). Each draw is tested as the protocol tests it:
a two-sided sign-flip test, at Holm's smallest level for three tests (0.05 / 3), the level the first rejected test
must reach. The p-value uses :data:`N_PATTERNS` random sign patterns. The detectable gap is found by bisection on the
true mean gap, :data:`N_DRAWS` simulated studies per step, seeded.

**Not modelled.** Clustered activations (the protocol also requires the cluster unit, which has fewer units), or
spreads that differ on new floods. The result indicates the scale of a detectable gap; it is not a guarantee.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05 / 3
POWER = 0.80
N_PATTERNS = 10_000
N_DRAWS = 2_000
UNITS = (10, 15, 20, 30)
SEED = 0


def power(delta: float, sigma: float, k: int, rng: np.random.Generator) -> float:
    """Share of simulated studies of ``k`` activations whose sign-flip p is at most :data:`ALPHA`."""
    signs = rng.choice((-1.0, 1.0), size=(N_PATTERNS, k)).astype(np.float32)
    hits = 0
    for start in range(0, N_DRAWS, 250):
        d = rng.normal(delta, sigma, size=(min(250, N_DRAWS - start), k)).astype(np.float32)
        observed = np.abs(d.mean(axis=1))
        null = np.abs(d @ signs.T) / k
        p = (null >= observed[:, None] - 1e-9).mean(axis=1)
        hits += int((p <= ALPHA).sum())
    return hits / N_DRAWS


def detectable_gap(sigma: float, k: int) -> float:
    lo, hi = 0.0, 0.3
    for step in range(12):
        mid = (lo + hi) / 2
        if power(mid, sigma, k, np.random.default_rng([SEED, k, step])) >= POWER:
            hi = mid
        else:
            lo = mid
    return hi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v11", type=Path, default=ROOT / "reports/cems_benchmark.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/v12_power.md")
    args = parser.parse_args(argv)
    tests = {t["id"]: t for t in json.loads(args.v11.read_text(encoding="utf-8"))["tests"]["activation"]}
    spreads = {tid: float(np.std([v for v in tests[tid]["per_unit_delta"].values() if v is not None], ddof=1))
               for tid in ("P4", "P3")}
    lines = ["# The blind test's power (v1.2)", "",
             "Written by `scripts/v12_power.py`. The smallest true gap in macro IoU that a two-sided sign-flip test "
             f"detects at {POWER:.0%} power at Holm's smallest level for three tests ({ALPHA:.4f}), by number of "
             "activations. Gaps are drawn i.i.d. normal with v1.1's per-activation spread on the CEMS benchmark. "
             "Clustering is not modelled.", "",
             "| Per-activation spread (v1.1) | " + " | ".join(f"{k} activations" for k in UNITS) + " |",
             "|---|" + "---|" * len(UNITS)]
    for tid, sd in spreads.items():
        gaps = [detectable_gap(sd, k) for k in UNITS]
        lines.append(f"| {tid}: SD {sd:.3f} | " + " | ".join(f"{g:.3f}" for g in gaps) + " |")
        print(tid, round(sd, 4), [round(g, 3) for g in gaps])
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

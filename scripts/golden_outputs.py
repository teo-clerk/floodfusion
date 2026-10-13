"""Record (or check) the logits of the v1.1 checkpoints on fixed inputs: a refactor must not change them.

    uv run --frozen python scripts/golden_outputs.py record   # once, at the code that trained them
    uv run --frozen python scripts/golden_outputs.py check    # what tests/test_golden_v11.py runs

The inputs are seeded noise (not data), four chips of which two carry optical; the checkpoints stay in runs/
(gitignored), so the check is skipped on a machine without them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "golden_v11.npz"
RUNS = {
    "s1f11_r4": "runs/v11/sen1floods11/official/official/row04_unet_sar_only/seed42",
    "s1f11_r9": "runs/v11/sen1floods11/official/official/row09_fusion_crossattn/seed42",
    "s1f11_tm": "runs/v11/sen1floods11/official/official/cmp_terramind_small/seed42",
    "ks_r9": "runs/v11/kurosiwo/all/all/row09_fusion_crossattn/seed42",
}
SIZE = 64


def inputs() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(1234)
    n = 4
    return {"sar": torch.randn(n, 5, SIZE, SIZE, generator=g), "terrain": torch.randn(n, 3, SIZE, SIZE, generator=g),
            "optical": torch.rand(n, 4, SIZE, SIZE, generator=g),
            "optical_quality": torch.rand(n, 1, SIZE, SIZE, generator=g),
            "optical_available": torch.tensor([True, False, True, False])}


def load(run_dir: Path) -> torch.nn.Module:
    from floodfusion.config import config_from_dict
    from floodfusion.models.zoo import build_model
    from floodfusion.runs import load_run_record

    cfg = config_from_dict(load_run_record(run_dir)["config"])
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model = build_model(cfg.model, pretrained=False)
    model.load_state_dict(state["model"])
    return model.eval()


def logits(run_dir: Path) -> np.ndarray:
    with torch.inference_mode():
        out, _ = load(run_dir)(**inputs())
    return out.float().numpy()


def available() -> dict[str, Path]:
    return {k: ROOT / v for k, v in RUNS.items() if (ROOT / v / "best.pt").is_file()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("record", "check"))
    args = parser.parse_args(argv)
    runs = available()
    if args.mode == "record":
        if set(runs) != set(RUNS):
            sys.exit(f"missing checkpoints: {sorted(set(RUNS) - set(runs))}")
        np.savez_compressed(GOLDEN, **{k: logits(p) for k, p in runs.items()})
        print(f"wrote {GOLDEN.relative_to(ROOT)}")
        return 0
    golden = np.load(GOLDEN)
    for key in sorted(set(RUNS) - set(runs)):
        print(f"{key}: skipped (checkpoint not on this machine)")
    for key, path in runs.items():
        np.testing.assert_allclose(logits(path), golden[key], rtol=1e-4, atol=1e-4, err_msg=key)
        print(f"{key}: unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

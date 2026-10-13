"""The fair TerraMind's size and adaptation, by the rule fixed on 2026-10-13 (reports/v1_2_deviations.md).

    uv run --frozen python scripts/tm_probe.py --pool-chips 9000      # GPU, ~10 min; writes reports/v12_tm_probe.json

**What is timed.** The same training micro-batches (bf16, forward + backward + step) go through row 9 and through each
TerraMind-fair variant: base + LoRA, base frozen, small + LoRA.

**The projection.** A pooled cross-validation run takes about row 9's measured Kuro Siwo run (:data:`ROW9_KS_RUN_S`,
the mean of v1.1's) × the pool's size relative to Kuro Siwo × the variant's time per batch relative to row 9.

**The rule.** The first variant, in the order above, whose projection is at most :data:`CAP_H` hours and whose peak
VRAM is at most :data:`VRAM_CAP_GB` is the fair opponent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP_H = 3.0
VRAM_CAP_GB = 7.5
#: v1.1: row 9 on Kuro Siwo (all-groups and folds), mean train_seconds.
ROW9_KS_RUN_S = 972.0
KS_TRAIN_CHIPS = 3178
VARIANTS = ("terramind_base_fair", "terramind_base_fair_frozen", "terramind_small_fair")
WARMUP, TIMED = 5, 20


def _batches(cfg, data_root: Path, sidecars, n: int):
    import torch

    from floodfusion.data.dataset import FloodChipDataset, build_dataloader
    from floodfusion.data.records import load_manifest

    records = [r for r in load_manifest(data_root / "chips" / "manifest.parquet") if r.eval_grid][: 4 * n]
    ds = FloodChipDataset(records, cfg.norm, cfg.data, cfg.grid, training=True, sidecars=sidecars)
    loader = build_dataloader(ds, cfg.data, batch_size=cfg.train.micro_batch_size, shuffle=False, drop_last=True)
    out = []
    for batch in loader:
        out.append({k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()})
        if len(out) == n:
            break
    return out


def time_model(config: Path, data_root: Path) -> dict[str, float]:
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from floodfusion.config import load_config
    from floodfusion.data.sidecars import sidecars_for
    from floodfusion.models.losses import CompoundFocalDiceLoss
    from floodfusion.models.zoo import build_model
    from floodfusion.train import model_inputs

    cfg = load_config(config, overlays=[ROOT / "configs/overlays/kurosiwo.yaml"])
    model = build_model(cfg.model).cuda().train()
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=1e-4)
    loss_fn = CompoundFocalDiceLoss(cfg.loss).cuda()
    batches = _batches(cfg, data_root, sidecars_for(cfg.model.architecture), WARMUP + TIMED)
    torch.cuda.reset_peak_memory_stats()

    def step(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(**model_inputs(batch, batch["optical_available"]))
        total, _ = loss_fn(logits.float(), batch["label"], batch["valid"])
        total.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for b in batches[:WARMUP]:
        step(b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for b in batches[WARMUP:]:
        step(b)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / TIMED
    peak = torch.cuda.max_memory_allocated() / 2**30
    return {"s_per_batch": seconds, "peak_gb": peak, "trainable_params": trainable}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/kurosiwo")
    parser.add_argument("--pool-chips", type=int, required=True, help="training chips of a pooled CV fold")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/v12_tm_probe.json")
    args = parser.parse_args(argv)
    configs = {"row09": ROOT / "configs/row09_fusion_crossattn.yaml",
               **{v: ROOT / f"configs/comparisons/{v}.yaml" for v in VARIANTS}}
    import gc

    import torch

    timings = {}
    for name, path in configs.items():
        timings[name] = time_model(path, args.data_root)
        gc.collect()
        torch.cuda.empty_cache()   # each variant's peak is measured alone
    base_run_h = ROW9_KS_RUN_S * args.pool_chips / KS_TRAIN_CHIPS / 3600
    for t in timings.values():
        t["projected_run_h"] = base_run_h * t["s_per_batch"] / timings["row09"]["s_per_batch"]
    chosen = next((v for v in VARIANTS if timings[v]["projected_run_h"] <= CAP_H
                   and timings[v]["peak_gb"] <= VRAM_CAP_GB), None)
    report = {"rule": f"first of {list(VARIANTS)} with projected pooled run <= {CAP_H} h and peak <= {VRAM_CAP_GB} GB",
              "pool_chips": args.pool_chips, "row9_projected_run_h": base_run_h, "timings": timings, "chosen": chosen}
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())

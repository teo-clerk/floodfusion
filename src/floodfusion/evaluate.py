r"""Evaluation and ablation-table generation.

Milestones ``P1-M5`` and ``P1-M6``. The acceptance criterion is that **every row of the
ablation table regenerates by one command from a recorded git SHA**, so this module is
the single entry point for every number in your report.

Run as::

    python -m floodfusion.evaluate --run-dir runs/row09 --split test
    python -m floodfusion.evaluate --ablation-table --runs-root runs/

References
----------
``docs/09_evaluation_ablation_and_report.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["evaluate_run", "fit_temperature", "build_ablation_table", "main"]


def fit_temperature(*args: object, **kwargs: object) -> float:
    r"""Fit a scalar temperature on the **validation** split.

    Returns
    -------
    float
        :math:`T^\star` minimising validation NLL of :math:`\sigma(z/T)`.

    Notes
    -----
    Your model is miscalibrated **by construction**, for two independent reasons:
    focal loss deliberately stops penalising confident predictions (``docs/08``
    section 2.3), and class resampling shifts the prior away from deployment
    (``docs/05`` section 8.3). Applying :math:`\tau^\star = 0.25` to raw scores is
    numerology (``docs/00`` section 6.3).

    .. math::

        T^\star = \arg\min_T -\sum_i \bigl[y_i \log\sigma(z_i/T)
                  + (1-y_i)\log(1 - \sigma(z_i/T))\bigr]

    **The property that surprises people**: temperature scaling is strictly monotone in
    :math:`z`, so it leaves the *ranking* unchanged. Average precision, ROC-AUC and the
    best achievable IoU over all thresholds are all **unchanged**. What changes is which
    threshold corresponds to which operating point — that is, whether
    :math:`\tau^\star = 0.25` means what ``docs/00`` section 6.2 says it means.

    Calibration does not make the model better at discriminating. It makes your
    threshold interpretable. State that correctly; misstating it is a reliable way to
    demonstrate you do not understand what you did.

    TODO(student)
    -------------
    1. Collect validation logits and labels over valid pixels.
    2. Optimise the single scalar (LBFGS, or a coarse-to-fine scan — one parameter).
    3. Report ECE with **quantile** bins before and after. Equal-width bins put nearly
       all mass in one bin at a 2 % positive rate and are uninformative.
    4. Verify empirically that AP is unchanged (``docs/09`` exercise 3).
    """
    raise NotImplementedError("Implement temperature scaling. See docs/09 section 4.3.")


def evaluate_run(*args: object, **kwargs: object) -> dict[str, object]:
    r"""Evaluate one trained run on a split, per basin.

    Returns
    -------
    dict
        Per-basin confusion matrices, macro and micro metrics, bootstrap CI, PR curve,
        AP, ECE, chosen operating point, peak inference VRAM and latency.

    Notes
    -----
    The rules (``docs/09`` section 2.3):

    1. **Touch the test split once**, at the end. Every tuning decision uses validation.
    2. All rows share everything except the one variable under test.
    3. **Evaluate at natural class balance** — sampling is a training device only, or
       precision is meaningless.
    4. Exclude invalid pixels from metrics as well as loss.
    5. Accumulate confusion matrices **per basin**, then compute the metric; never
       average per-chip IoU.

    The sequence: evaluate, fit :math:`T` on validation, apply to test logits, threshold
    at :math:`\tau^\star = 1/(1+r)`, and report IoU **both** at :math:`\tau^\star` and at
    the best achievable threshold. The gap is the cost of committing to an operating
    point in advance, which is what deployment requires.

    TODO(student)
    -------------
    1. Load the checkpoint and the frozen split; assert the manifest hash matches.
    2. Accumulate per-basin confusion matrices at the fixed threshold grid.
    3. Bootstrap over basins (:func:`floodfusion.metrics.basin_bootstrap_ci`).
    4. Measure latency per ``docs/09`` section 7: batch 1, >= 20 warm-up iterations,
       ``torch.cuda.synchronize()`` around the timer, on a **thermally warm** GPU. A
       cold-GPU latency is not the latency you get in the grid's twentieth minute.
    5. Convert throughput to minutes per Sentinel-1 IW frame and place it in
       ``docs/00`` section 5.1's budget.
    """
    raise NotImplementedError("Implement run evaluation. See docs/09 section 2.")


def build_ablation_table(*args: object, **kwargs: object) -> str:
    r"""Assemble the 13-row ablation table of ``MILESTONES.md`` section 6.

    Notes
    -----
    Every row must be produced by the **same** evaluation code path, differing only in
    configuration. Report mean :math:`\pm` 95 % bootstrap CI resampled over basins,
    three seeds for learned rows, plus peak VRAM and wall clock.

    **Rows 10, 11 and 13 are the scientific core.** Rows 1--9 establish that you built
    something that works; those three establish that you understand it.

    - **Row 11** (evaluate SAR-only) is the operational number, because optical is
      usually absent. Diagnose it against row 4 per ``docs/09`` section 5.2.
    - **Row 13** (random split) must score **higher** than row 9. If it does not, the
      basin blocking is not working — check :mod:`floodfusion.splits` and the four
      residual leakage channels of ``docs/05`` section 6.5.

    Compare rows with :func:`floodfusion.metrics.paired_bootstrap_delta`, not by
    eyeballing overlapping individual intervals: the models share basins, so the paired
    test is both correct and strictly more powerful.

    TODO(student)
    -------------
    1. Discover run directories, assert each records a git SHA and config hash.
    2. Refuse to emit a table if any run's SHA differs from the current tree, unless
       ``--allow-dirty`` — ``P1-M6.3`` reproducibility depends on it.
    3. Emit Markdown matching ``MILESTONES.md`` section 6 exactly, so the table can be
       pasted in.
    4. Emit the per-basin dot plot with CIs (``docs/09`` section 8.2, figure 4). It
       communicates more than the table, because it shows the spread rather than the
       mean.
    """
    raise NotImplementedError("Implement the ablation table. See MILESTONES.md section 6.")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Project 01 runs.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--ablation-table", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.parse_args()
    raise NotImplementedError("Implement the evaluation entry point. See docs/09.")


if __name__ == "__main__":
    raise SystemExit(main())

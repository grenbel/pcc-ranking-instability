"""Standard reconstruction metrics: Chamfer Distance + F-score.

Numpy / scipy implementation used by the metric stage, the smoke tests and the
sanity check. PoinTr's CUDA chamfer extension (`extensions.chamfer_dist`) is
50-100x faster on GPU and computes the official `best_metrics` stamped in the
PoinTr checkpoints. The numpy<->CUDA cross-implementation agreement was
validated at the full PCN test 1200 scale with scripts/chamfer_fourway_check.py:
for both PoinTr and AdaPoinTr the numpy and CUDA CD-L1 deltas were 0.000000
across all four variants {numpy, CUDA} x {with-zero, ignore-zero} at
single-precision floating point. Per-sample agreement is therefore established
under PoinTr's official ChamferDistanceL1(ignore_zeros=True) protocol.

Convention:
    - All metrics expect (N_pred, 3) and (N_gt, 3) numpy float32 arrays
    - Chamfer metrics drop **sum-zero rows** (rows whose 3 coordinates sum
      to 0, including but not limited to exact (0,0,0)) by default, matching
      PoinTr's `ChamferDistanceL1/2(ignore_zeros=True)` mask convention at
      `baselines/PoinTr/utils/metrics.py`. This is the official mask but
      technically broader than "exact (0,0,0) rows" (a row with values
      summing to 0, e.g. (1, -1, 0), is also dropped; 7 such edge cases were
      encountered across the 40 zero-pad cells of PoinTr and AdaPoinTr).
    - F-score does **NOT** apply `ignore_zeros`: it operates on the full
      (N_pred, 3) and (N_gt, 3) arrays as fed in. This matches PoinTr's
      official F-score path. Cardinality-controlled CD-L1 sensitivity
      (FPS + random subsample to N=14336 on the 6 PoinTr-vs-AdaPoinTr flip
      cells; all preserved) was validated with
      `scripts/cardinality_sensitivity.py`. CD-L1
      is the paper-table primary ranking endpoint; F-score is reported as
      a secondary descriptor and is not used for the headline ranking-flip
      statistics.
    - F-score thresholds are *unsquared* distances (L2 Euclidean) per PoinTr
      default at thresholds (0.001, 0.005, 0.01).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _drop_zero_rows(points: np.ndarray, name: str) -> np.ndarray:
    """Match PoinTr's `ignore_zeros=True` mask: keep rows whose coordinate sum is nonzero.

    Mask is `np.sum(points, axis=1) != 0` - broader than strict-equal `(0,0,0)` mask.
    Aligned with `baselines/PoinTr/utils/metrics.py` official convention. For PoinTr
    PCN preds, 100% of dropped rows are in indices >=14336 (last 2048-pt input-partial
    concat block from `models/PoinTr.py:119` `torch.cat([rebuild_points, xyz], dim=1)`
    where `xyz` carries (0,0,0) padding from PCN sub-2048 partial clouds). For AdaPoinTr,
    no rows are dropped (no zero rows in any of its 40 zero-pad cells x 1200 samples).
    """
    non_zero = np.sum(points, axis=1) != 0
    filtered = points[non_zero]
    if filtered.shape[0] == 0:
        raise ValueError(f"{name} has no non-zero points after ignore_zeros filtering")
    return filtered


def _prepare_chamfer_inputs(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_zeros: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if not ignore_zeros:
        return pred, gt
    return _drop_zero_rows(pred, "pred"), _drop_zero_rows(gt, "gt")


def _bidirectional_nn_dists(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pred->gt nearest dists, gt->pred nearest dists), both Euclidean."""
    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)
    d_pred_to_gt, _ = tree_gt.query(pred, k=1, workers=1)
    d_gt_to_pred, _ = tree_pred.query(gt, k=1, workers=1)
    return d_pred_to_gt.astype(np.float32), d_gt_to_pred.astype(np.float32)


def chamfer_l1(pred: np.ndarray, gt: np.ndarray, ignore_zeros: bool = True) -> float:
    """Chamfer-L1 in PoinTr's convention: 0.5 * (mean(d_pg) + mean(d_gp)).

    PoinTr's ChamferDistanceL1 averages the two directional means (divides by 2);
    ChamferDistanceL2 sums them. CD-L1 keeps the 0.5x; CD-L2 below does not.
    """
    pred, gt = _prepare_chamfer_inputs(pred, gt, ignore_zeros)
    d_pg, d_gp = _bidirectional_nn_dists(pred, gt)
    return float(0.5 * (d_pg.mean() + d_gp.mean()))


def chamfer_l2(pred: np.ndarray, gt: np.ndarray, ignore_zeros: bool = True) -> float:
    """Chamfer-L2 in PoinTr's convention: SUM (not 0.5x) of bidirectional squared NN means.

    Matches PoinTr/extensions/chamfer_dist/__init__.py:44 ChamferDistanceL2 which returns
    `dist1.mean() + dist2.mean()` (sum of means, not 0.5x average); needed for sanity
    gate to compare against PoinTr ckpt's stamped `best_metrics["CDL2"]` correctly.
    """
    pred, gt = _prepare_chamfer_inputs(pred, gt, ignore_zeros)
    d_pg, d_gp = _bidirectional_nn_dists(pred, gt)
    return float((d_pg ** 2).mean() + (d_gp ** 2).mean())


def fscore_at_thresholds(
    pred: np.ndarray,
    gt: np.ndarray,
    thresholds: Tuple[float, ...] = (0.001, 0.005, 0.01),
) -> Dict[str, float]:
    """F-score at multiple distance thresholds.

    Per PoinTr convention, threshold = fraction of unit diameter (PCN coords
    in [-1, 1]). Default (0.001, 0.005, 0.01) matches typical 0.1%, 0.5%, 1%.

    Returns dict {f"f_at_{t}": fscore} for each threshold.
    """
    d_pg, d_gp = _bidirectional_nn_dists(pred, gt)
    out: Dict[str, float] = {}
    for t in thresholds:
        precision = float((d_pg < t).mean())
        recall = float((d_gp < t).mean())
        if precision + recall == 0:
            f = 0.0
        else:
            f = 2 * precision * recall / (precision + recall)
        out[f"f_at_{t}"] = f
    return out

"""Density collapse metric (decomposition mode #3).

Captures: completion outputs are non-uniformly distributed - e.g., points
clustering in some regions while leaving others sparse. This is a common
failure when models generate output points in already-observed regions
rather than filling missing regions.

Method:
    1. Per-point local density = mean reciprocal-distance to k-NN (k=10).
       Higher = denser local region.
    2. Failure score = KS statistic between density distributions of pred
       and gt.

KS is more sensitive to distribution-shape mismatch than Wasserstein for
this case, where mean density is similar but the spread differs.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

K_NEIGHBORS = 10
EPS = 1e-6


def _local_density(points: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1, workers=1)
    # mean of 1/dist over k-NN (exclude self)
    inv = 1.0 / (d[:, 1:] + EPS)
    return inv.mean(axis=1).astype(np.float32)


class DensityCollapseMetric:
    name = "density_collapse"

    def __init__(self, k: int = K_NEIGHBORS):
        self.k = k

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        if pred.shape[0] < self.k + 1 or gt.shape[0] < self.k + 1:
            return float("nan")
        d_pred = _local_density(pred, self.k)
        d_gt = _local_density(gt, self.k)
        # KS statistic - D in [0, 1], higher = more shape mismatch
        ks_stat, _ = ks_2samp(d_pred, d_gt)
        return float(ks_stat)

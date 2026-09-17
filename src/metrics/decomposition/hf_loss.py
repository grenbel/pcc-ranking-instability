"""High-frequency loss metric (decomposition mode #1).

Captures: missing fine geometric detail, smoothed-out edges/corners,
over-blurred completion outputs.

Method:
    1. Per-point local roughness = std-dev of distances from each point to
       its k-NN (k=20). Points on flat surfaces have low roughness; points
       on edges/corners/details have high roughness.
    2. Failure score = 1-Wasserstein distance between pred-roughness
       distribution and gt-roughness distribution.

Higher score = pred has either over-smoothed (lost high-freq) or hallucinated
extra noise. Both are HF failures.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

K_NEIGHBORS = 20


def _local_roughness(points: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    tree = cKDTree(points)
    # k+1 because nearest is the point itself
    d, _ = tree.query(points, k=k + 1, workers=1)
    # exclude self distance (column 0)
    return d[:, 1:].std(axis=1).astype(np.float32)


class HFLossMetric:
    name = "hf_loss"

    def __init__(self, k: int = K_NEIGHBORS):
        self.k = k

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        if pred.shape[0] < self.k + 1 or gt.shape[0] < self.k + 1:
            return float("nan")
        r_pred = _local_roughness(pred, self.k)
        r_gt = _local_roughness(gt, self.k)
        return float(wasserstein_distance(r_pred, r_gt))

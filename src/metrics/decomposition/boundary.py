"""Boundary erosion metric (decomposition mode #4).

Captures: completion erodes / over-smooths object boundaries (silhouette,
holes, handles, thin structures, sharp edges). Especially severe under
LocalCrop and DensityDrop corruptions.

Method:
    1. Identify boundary points as those with the largest local plane-fit
       residual (top-q% by residual; q=10 default). Plane is fit via PCA
       on k-NN (k=15) and residual = third PCA singular value (smallest
       eigenvalue's sqrt = thickness orthogonal to local plane).
    2. Boundary recall = fraction of gt-boundary points that have a pred
       point within eps (0.02 unit diameter) distance.
    3. Failure score = 1 - recall in [0, 1].
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

K_PCA = 15
BOUNDARY_QUANTILE = 0.10
RECALL_EPS = 0.02


def _boundary_points(points: np.ndarray, k: int = K_PCA,
                     quantile: float = BOUNDARY_QUANTILE) -> np.ndarray:
    """Return indices of points with highest local-plane-fit residual."""
    n = points.shape[0]
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1, workers=1)
    residuals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nbrs = points[idx[i, 1:]]  # exclude self
        centered = nbrs - nbrs.mean(axis=0)
        # singular value (smallest) corresponds to plane-orthogonal thickness
        try:
            sv = np.linalg.svd(centered, compute_uv=False)
            residuals[i] = sv[-1]
        except np.linalg.LinAlgError:
            residuals[i] = 0.0
    threshold = np.quantile(residuals, 1.0 - quantile)
    return np.where(residuals >= threshold)[0]


class BoundaryErosionMetric:
    name = "boundary_erosion"

    def __init__(self, k: int = K_PCA, quantile: float = BOUNDARY_QUANTILE,
                 eps: float = RECALL_EPS):
        self.k = k
        self.quantile = quantile
        self.eps = eps

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        if pred.shape[0] < self.k + 1 or gt.shape[0] < self.k + 1:
            return float("nan")
        gt_b_idx = _boundary_points(gt, self.k, self.quantile)
        if gt_b_idx.size == 0:
            return 0.0
        gt_boundary = gt[gt_b_idx]
        tree_pred = cKDTree(pred)
        d, _ = tree_pred.query(gt_boundary, k=1, workers=1)
        recall = float((d < self.eps).mean())
        return float(1.0 - recall)

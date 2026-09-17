"""Topology break metric (decomposition mode #2).

Captures: completion fragments into multiple disconnected pieces, holes
appear, parts split that should be one connected object.

Method:
    1. Build k-NN connectivity graph (k=10) on each cloud.
    2. Count connected components via union-find on graph edges.
    3. Failure score = |comp_count(pred) - comp_count(gt)| / max(1, comp_count(gt))
       + 0.1 x cosine_distance(sorted_comp_sizes(pred), sorted_comp_sizes(gt))

The size-distribution term penalizes cases where pred has same total
component count but very different size partition (e.g., one big + one
fragment vs two equal halves).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

K_GRAPH = 10
SIZE_PENALTY_WEIGHT = 0.1


def _knn_components(points: np.ndarray, k: int = K_GRAPH):
    n = points.shape[0]
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1, workers=1)  # k+1 includes self
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    data = np.ones_like(rows, dtype=np.float32)
    G = csr_matrix((data, (rows, cols)), shape=(n, n))
    G = G + G.T  # symmetric
    n_comp, labels = connected_components(G, directed=False)
    sizes = np.bincount(labels)
    return n_comp, np.sort(sizes)[::-1]  # descending sorted sizes


def _padded_cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    L = max(len(a), len(b))
    ap = np.zeros(L, dtype=np.float32); ap[:len(a)] = a
    bp = np.zeros(L, dtype=np.float32); bp[:len(b)] = b
    denom = np.linalg.norm(ap) * np.linalg.norm(bp)
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.dot(ap, bp) / denom)


class TopologyBreakMetric:
    name = "topology"

    def __init__(self, k: int = K_GRAPH):
        self.k = k

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        n_pred, sizes_pred = _knn_components(pred, self.k)
        n_gt, sizes_gt = _knn_components(gt, self.k)
        comp_term = abs(n_pred - n_gt) / max(1, n_gt)
        size_term = _padded_cosine_dist(sizes_pred, sizes_gt)
        return float(comp_term + SIZE_PENALTY_WEIGHT * size_term)

"""Metric suite for the matched-control audit + descriptive failure-decomposition metrics.

Standard reconstruction metrics:
    - CD-L1, CD-L2 (numpy fallback; PoinTr CUDA chamfer for prod)
    - F-score @ multiple thresholds (1%, 0.5%, 0.1% of unit diameter)

Failure-decomposition metrics (descriptive extras, not used by the paper's ranking statistics):
    - hf_loss      : high-frequency loss via local curvature distance
    - topology     : topology break via k-NN connected components delta
    - density      : density collapse via local k-NN-distance distribution KS
    - boundary     : boundary erosion via recall of local-PCA-residual boundary points

All decomposition metrics are scalar per-sample, [0, inf) where 0 = perfect
match.
"""

from .reconstruction import chamfer_l1, chamfer_l2, fscore_at_thresholds
from .decomposition.hf_loss import HFLossMetric
from .decomposition.topology import TopologyBreakMetric
from .decomposition.density_metric import DensityCollapseMetric
from .decomposition.boundary import BoundaryErosionMetric

DECOMPOSITION_METRICS = {
    "hf_loss": HFLossMetric,
    "topology": TopologyBreakMetric,
    "density_collapse": DensityCollapseMetric,
    "boundary_erosion": BoundaryErosionMetric,
}

__all__ = [
    "chamfer_l1", "chamfer_l2", "fscore_at_thresholds",
    "HFLossMetric", "TopologyBreakMetric",
    "DensityCollapseMetric", "BoundaryErosionMetric",
    "DECOMPOSITION_METRICS",
]

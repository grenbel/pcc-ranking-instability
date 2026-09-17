"""Outlier corruption - sensor flares / multi-path reflections / specular noise.

Replace k% of points with uniformly-sampled random points within a
bounding box slightly larger than the cloud's actual AABB (x1.2 padding).

Severity ladder (fraction of points replaced):
    1: 1%     (single sensor flare)
    2: 2%
    3: 5%     (noticeable speckle)
    4: 10%
    5: 20%    (severe - unusable raw cloud)

Replaced indices are sampled without replacement; original geometry of
non-replaced points is preserved exactly. This keeps the corruption
factor isolatable in matched-control analysis.

Two protocol variants:
    - default (`valid_point_only=False`): replacement indices drawn uniformly
      from all 2048 input rows including zero-pad rows; AABB computed over
      all rows. Matches the original zero-pad protocol.
    - `valid_point_only=True`: replacement indices drawn only
      from valid (non-zero-pad) rows; AABB computed over valid points only;
      replacement count `k = max(1, round(n_valid * fraction))` so the
      corruption density on valid geometry matches the severity intent
      regardless of how much zero-pad the input contains.
"""

from __future__ import annotations

import numpy as np

from .base import CorruptionOp, valid_point_mask

OUTLIER_FRACTION = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
AABB_PAD = 1.2


class Outliers(CorruptionOp):
    name = "outlier"
    preserves_count = True

    def __init__(self, valid_point_only: bool = False):
        super().__init__()
        self.valid_point_only = valid_point_only
        if valid_point_only:
            self.name = "outlier_validpt"

    def _apply(self, points: np.ndarray, severity: int,
               rng: np.random.Generator) -> np.ndarray:
        n = points.shape[0]
        if self.valid_point_only:
            mask = valid_point_mask(points)
            valid_idx = np.where(mask)[0]
            n_valid = valid_idx.size
            if n_valid < 2:
                return points.copy()
            k = max(1, int(round(n_valid * OUTLIER_FRACTION[severity])))
            k = min(k, n_valid)
            # AABB from VALID points only - padding rows do not influence the
            # reference cloud extent, so spatial scale of injected outliers
            # matches actual geometry.
            valid_pts = points[valid_idx]
            center = valid_pts.mean(axis=0)
            half_extent = (valid_pts.max(axis=0) - valid_pts.min(axis=0)) * 0.5 * AABB_PAD
            outliers = rng.uniform(
                low=center - half_extent,
                high=center + half_extent,
                size=(k, 3),
            ).astype(np.float32)
            replace_idx = rng.choice(valid_idx, size=k, replace=False)
            out = points.copy()
            out[replace_idx] = outliers
            return out
        # original (zero-pad-included) protocol
        k = max(1, int(round(n * OUTLIER_FRACTION[severity])))
        center = points.mean(axis=0)
        half_extent = (points.max(axis=0) - points.min(axis=0)) * 0.5 * AABB_PAD
        outliers = rng.uniform(
            low=center - half_extent,
            high=center + half_extent,
            size=(k, 3),
        ).astype(np.float32)
        replace_idx = rng.choice(n, size=k, replace=False)
        out = points.copy()
        out[replace_idx] = outliers
        return out

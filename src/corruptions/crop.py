"""Local crop corruption - boundary erosion / partial occlusion.

Pick a random center on the cloud (sampled from points themselves so the
crop is always over actual geometry, not empty space), remove all points
within radius r. Pad back to N via duplication+jitter (same as DensityDrop).

This is the primary stressor for the "boundary erosion" failure mode in
the decomposition metric suite.

Severity ladder (radius in unit-coord; PCN ~[-1,1] so diameter ~ 2):
    1: 0.10   (small chip)
    2: 0.15
    3: 0.20
    4: 0.25
    5: 0.30   (large chunk missing - deep boundary erosion)

Two protocol variants:
    - default (`valid_point_only=False`): crop center sampled from any of
      the 2048 input rows; under PCN zero-pad input distribution this can
      land on a (0,0,0) padding row, producing an origin-centred crop with
      semantics different from the intended "crop a region of the actual
      surface". Matches the original zero-pad protocol.
    - `valid_point_only=True`: crop center sampled only from
      valid (non-padding) rows so the crop ball is always centred on real
      geometry. Dropped valid rows replaced by duplicate-with-jitter from
      surviving valid rows. Zero-pad rows preserved at their original slots.
"""

from __future__ import annotations

import numpy as np

from .base import CorruptionOp, valid_point_mask

CROP_RADIUS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30)
JITTER_SIGMA = 0.001
MIN_KEEP = 64


class LocalCrop(CorruptionOp):
    name = "crop"
    preserves_count = True

    def __init__(self, valid_point_only: bool = False):
        super().__init__()
        self.valid_point_only = valid_point_only
        if valid_point_only:
            self.name = "crop_validpt"

    def _apply(self, points: np.ndarray, severity: int,
               rng: np.random.Generator) -> np.ndarray:
        n = points.shape[0]
        radius = CROP_RADIUS[severity]
        if self.valid_point_only:
            mask = valid_point_mask(points)
            valid_idx = np.where(mask)[0]
            n_valid = valid_idx.size
            if n_valid < MIN_KEEP:
                return points.copy()
            valid_pts = points[valid_idx]
            # sample center from VALID points only
            center_local = int(rng.integers(0, n_valid))
            center = valid_pts[center_local]
            dists = np.linalg.norm(valid_pts - center, axis=1)
            keep_local_mask = dists > radius
            kept_local = np.where(keep_local_mask)[0]
            if kept_local.size < MIN_KEEP:
                # crop too aggressive: keep MIN_KEEP farthest from centre
                sorted_local = np.argsort(-dists)
                kept_local = sorted_local[:MIN_KEEP]
            kept_global = valid_idx[kept_local]
            dropped_global = np.setdiff1d(valid_idx, kept_global, assume_unique=True)
            n_drop = dropped_global.size
            out = points.copy()
            if n_drop > 0:
                dup_local = rng.choice(kept_local, size=n_drop, replace=True)
                dups = valid_pts[dup_local] + rng.normal(
                    0.0, JITTER_SIGMA, size=(n_drop, 3)
                ).astype(np.float32)
                out[dropped_global] = dups
            # zero-pad rows at non-valid positions stay (0,0,0)
            return out
        # original (zero-pad-included) protocol
        center_idx = int(rng.integers(0, n))
        center = points[center_idx]
        dists = np.linalg.norm(points - center, axis=1)
        keep_mask = dists > radius
        kept = points[keep_mask]
        if kept.shape[0] < MIN_KEEP:
            sorted_idx = np.argsort(-dists)
            kept = points[sorted_idx[:MIN_KEEP]]
        pad_n = n - kept.shape[0]
        if pad_n <= 0:
            return kept[:n]
        dup_idx = rng.choice(kept.shape[0], size=pad_n, replace=True)
        dups = kept[dup_idx] + rng.normal(0.0, JITTER_SIGMA,
                                          size=(pad_n, 3)).astype(np.float32)
        out = np.concatenate([kept, dups], axis=0)
        perm = rng.permutation(n)
        return out[perm]

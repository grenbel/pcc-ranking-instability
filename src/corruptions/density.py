"""Density drop corruption - sparse-sensor / occluded-region simulation.

Implementation:
    1. Drop k% of points uniformly at random (kept indices form sparser cloud)
    2. To preserve N for fixed-input model inference, resample the kept points
       with replacement back up to N, and add small jitter (sigma=0.001) on
       duplicates so they are not pixel-identical (downstream nearest-neighbor
       graph ops do not collapse).

Effective `unique` density = (1 - k%) x N_original. Reported alongside metric
output as `effective_unique_density` for the decomposition module.

Severity ladder (drop ratio):
    1: 10%   (mild thinning)
    2: 25%
    3: 40%
    4: 55%
    5: 70%   (most points lost; severe density collapse)

Two protocol variants:
    - default (`valid_point_only=False`): drop k% of all 2048 input rows
      including zero-pad rows; pad-back via duplicates+jitter from kept rows.
    - `valid_point_only=True`: drop k% of valid rows only;
      pad-back uses kept valid rows; zero-pad rows preserved at original
      slots so input partial padding structure is invariant.
"""

from __future__ import annotations

import numpy as np

from .base import CorruptionOp, valid_point_mask

DROP_FRACTION = (0.0, 0.10, 0.25, 0.40, 0.55, 0.70)
JITTER_SIGMA = 0.001


class DensityDrop(CorruptionOp):
    name = "density"
    preserves_count = True  # padded back via duplication+jitter

    def __init__(self, valid_point_only: bool = False):
        super().__init__()
        self.valid_point_only = valid_point_only
        if valid_point_only:
            self.name = "density_validpt"

    def _apply(self, points: np.ndarray, severity: int,
               rng: np.random.Generator) -> np.ndarray:
        n = points.shape[0]
        drop_frac = DROP_FRACTION[severity]
        if self.valid_point_only:
            mask = valid_point_mask(points)
            valid_idx = np.where(mask)[0]
            n_valid = valid_idx.size
            if n_valid < 2:
                return points.copy()
            keep_n_valid = max(min(8, n_valid), int(round(n_valid * (1.0 - drop_frac))))
            keep_n_valid = min(keep_n_valid, n_valid)
            shuffle = rng.permutation(valid_idx)
            kept_global = shuffle[:keep_n_valid]
            dropped_global = shuffle[keep_n_valid:]
            n_drop = dropped_global.size
            out = points.copy()
            if n_drop > 0:
                dup_global = rng.choice(kept_global, size=n_drop, replace=True)
                dups = points[dup_global] + rng.normal(
                    0.0, JITTER_SIGMA, size=(n_drop, 3)
                ).astype(np.float32)
                out[dropped_global] = dups
            # zero-pad rows at non-valid positions remain (0,0,0) in `out`
            return out
        # original (zero-pad-included) protocol
        keep_n = max(8, int(round(n * (1.0 - drop_frac))))
        keep_idx = rng.choice(n, size=keep_n, replace=False)
        kept = points[keep_idx]
        pad_n = n - keep_n
        if pad_n <= 0:
            return kept[:n]
        dup_idx = rng.choice(keep_n, size=pad_n, replace=True)
        dups = kept[dup_idx] + rng.normal(0.0, JITTER_SIGMA,
                                          size=(pad_n, 3)).astype(np.float32)
        out = np.concatenate([kept, dups], axis=0)
        perm = rng.permutation(n)
        return out[perm]

"""Gaussian sensor noise corruption.

Per-coordinate iid Gaussian noise. Severity ladder picks sigma as a fraction
of unit diameter (PCN coords are normalized to [-1, 1]).

Severity ladder (sigma in unit-coord):
    1: 0.005   (~0.25% diameter - barely visible)
    2: 0.010   (~0.5%)
    3: 0.020   (~1.0%  - noticeable)
    4: 0.040   (~2.0%)
    5: 0.080   (~4.0%  - heavy noise)

Two protocol variants:
    - default (`valid_point_only=False`): noise added to all 2048 input
      rows including zero-pad rows; matches the original zero-pad protocol
    - `valid_point_only=True`: noise added only to rows where
      `valid_point_mask(points)` is True; zero-pad rows preserved as (0,0,0)
"""

from __future__ import annotations

import numpy as np

from .base import CorruptionOp, valid_point_mask

SIGMA_LADDER = (0.0, 0.005, 0.010, 0.020, 0.040, 0.080)


class GaussianNoise(CorruptionOp):
    name = "noise"
    preserves_count = True

    def __init__(self, valid_point_only: bool = False):
        super().__init__()
        self.valid_point_only = valid_point_only
        if valid_point_only:
            self.name = "noise_validpt"

    def _apply(self, points: np.ndarray, severity: int,
               rng: np.random.Generator) -> np.ndarray:
        sigma = SIGMA_LADDER[severity]
        noise = rng.normal(loc=0.0, scale=sigma, size=points.shape).astype(np.float32)
        if self.valid_point_only:
            mask = valid_point_mask(points)
            noise[~mask] = 0.0
        return points + noise

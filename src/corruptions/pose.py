"""Non-canonical pose corruption - rotation away from canonical training frame.

Apply a single SO(3) rotation to the entire cloud. Rotation drawn as
axis-angle: random axis on unit sphere, angle uniform in [-theta_max, theta_max]
where theta_max grows with severity.

Severity ladder (theta_max in degrees):
    1: 5 deg    (small drift)
    2: 10 deg
    3: 20 deg   (noticeable)
    4: 30 deg
    5: 45 deg   (heavy non-canonical pose - typical robot mount mis-alignment)

Note: rotation is rigid, so geometry is preserved but completion models
trained on canonical frames may fail to align internal queries.
"""

from __future__ import annotations

import numpy as np

from .base import CorruptionOp

ANGLE_LADDER_DEG = (0.0, 5.0, 10.0, 20.0, 30.0, 45.0)


def _axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' formula. axis must be unit-norm."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0],
    ], dtype=np.float64)
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
    return R.astype(np.float32)


class NonCanonicalPose(CorruptionOp):
    name = "pose"
    preserves_count = True

    def _apply(self, points: np.ndarray, severity: int,
               rng: np.random.Generator) -> np.ndarray:
        theta_max_rad = np.deg2rad(ANGLE_LADDER_DEG[severity])
        # uniform axis on unit sphere
        v = rng.normal(0.0, 1.0, size=3)
        axis = v / (np.linalg.norm(v) + 1e-12)
        angle = rng.uniform(-theta_max_rad, theta_max_rad)
        R = _axis_angle_to_matrix(axis, float(angle))
        # rotate around centroid (preserves overall translation)
        centroid = points.mean(axis=0, keepdims=True)
        return ((points - centroid) @ R.T) + centroid

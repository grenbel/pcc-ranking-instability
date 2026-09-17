"""Corruption operator suite for matched-control audit.

5 base operators (physical-intelligence corruption regime):
    - GaussianNoise   (sensor noise)
    - Outliers        (sensor outliers / multi-path / specular reflection)
    - DensityDrop     (occluded / sparse regions)
    - LocalCrop       (boundary erosion / partial occlusion)
    - NonCanonicalPose (pose deviation from canonical training frame)

ComposeCorruptions chains multiple ops for "mixed corruption" inputs (the
composed-operator pilot uses its own fixed order, see scripts/forward_sweep_composed.py).

All operators are deterministic given a (taxonomy_id, model_id, view_id,
severity, op_name) tuple via `make_seed()` - required for matched-control
protocol where the same object/view must be corrupted reproducibly across
runs and across models.

Valid-point-only protocol: each non-pose operator supports a
`valid_point_only=True` variant that only corrupts non-padding rows of the
2048-point PCN partial input. The variants are registered with `_validpt`
suffix names so that `make_seed` gives them corruption seeds (random streams)
distinct from the zero-pad variants; the clean-cache hash is the same for both
protocols because the cache is snapshotted before corruption.
"""

from .base import (
    CorruptionOp,
    SEVERITY_LEVELS,
    make_seed,
    valid_point_mask,
)
from .compose import ComposeCorruptions
from .crop import LocalCrop
from .density import DensityDrop
from .noise import GaussianNoise
from .outlier import Outliers
from .pose import NonCanonicalPose

OP_REGISTRY = {
    "noise": GaussianNoise,
    "outlier": Outliers,
    "density": DensityDrop,
    "crop": LocalCrop,
    "pose": NonCanonicalPose,
    # valid-point-only variants (zero-pad rows preserved)
    "noise_validpt": lambda: GaussianNoise(valid_point_only=True),
    "outlier_validpt": lambda: Outliers(valid_point_only=True),
    "density_validpt": lambda: DensityDrop(valid_point_only=True),
    "crop_validpt": lambda: LocalCrop(valid_point_only=True),
}

__all__ = [
    "CorruptionOp", "make_seed", "SEVERITY_LEVELS", "valid_point_mask",
    "OP_REGISTRY",
    "GaussianNoise", "Outliers", "DensityDrop", "LocalCrop",
    "NonCanonicalPose", "ComposeCorruptions",
]

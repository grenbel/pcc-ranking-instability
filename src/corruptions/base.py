"""Corruption operator ABC + reproducible seed protocol.

Matched-control protocol requires: given the same (object, view, severity,
op_name), the corruption is bit-identical across runs and across models.
The seed is the first 8 bytes of the SHA-1 digest of the tuple, masked to 63
bits, fed into `numpy.random.default_rng()`, so seeds are independent of the
global RNG state.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

SEVERITY_LEVELS = (0, 1, 2, 3, 4, 5)
DEFAULT_NUM_POINTS_PARTIAL = 2048
DEFAULT_NUM_POINTS_GT = 16384


def valid_point_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-zero-padding rows.

    PCN partial clouds are zero-padded to N=2048 with exact (0,0,0) rows
    when the original partial has fewer than 2048 points. This helper
    distinguishes valid geometry rows from padding rows so the
    `valid_point_only=True` corruption variant can preserve padding
    structure rather than corrupting it.

    A row that happens to be exactly (0,0,0) by coincidence (rare in
    centred-normalized PCN) is incorrectly classified as padding; the
    valid-point-only protocol treats this as an acceptable approximation.
    """
    return ~np.all(points == 0.0, axis=1)


def make_seed(taxonomy_id: str, model_id: str, view_id: int,
              severity: int, op_name: str) -> int:
    """Reproducible per-sample seed for corruption operators.

    Returns a 64-bit positive int derived from SHA-1 (truncated). Independent
    of numpy / Python global RNG. Stable across processes and machines.
    """
    key = f"{taxonomy_id}|{model_id}|{view_id}|{severity}|{op_name}".encode()
    digest = hashlib.sha1(key).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


class CorruptionOp(ABC):
    """Base class for all corruption operators.

    Sub-classes implement `_apply(points, severity, rng)`. The base class
    handles seed derivation, severity validation, and shape preservation
    invariants (matched-control needs same point count after corruption
    unless the operator is explicitly density-changing).
    """

    name: str = "base"
    preserves_count: bool = True

    def __init__(self):
        if self.name == "base":
            raise ValueError("Subclass must override `name`.")

    def __call__(self, points: np.ndarray, severity: int,
                 taxonomy_id: str, model_id: str, view_id: int) -> np.ndarray:
        if severity not in SEVERITY_LEVELS:
            raise ValueError(f"severity must be in {SEVERITY_LEVELS}, got {severity}")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N,3); got {points.shape}")
        if points.dtype != np.float32:
            points = points.astype(np.float32)
        if severity == 0:
            return points.copy()
        seed = make_seed(taxonomy_id, model_id, view_id, severity, self.name)
        rng = np.random.default_rng(seed)
        out = self._apply(points, severity, rng)
        if self.preserves_count and out.shape[0] != points.shape[0]:
            raise RuntimeError(
                f"{self.name} declared preserves_count=True but changed N "
                f"from {points.shape[0]} to {out.shape[0]}"
            )
        if out.dtype != np.float32:
            out = out.astype(np.float32)
        return out

    @abstractmethod
    def _apply(self, points: np.ndarray, severity: int,
               rng: np.random.Generator) -> np.ndarray:
        ...

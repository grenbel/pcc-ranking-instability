"""Compose multiple corruption operators (mixed-corruption baseline).

The compose order matters because pose o noise != noise o pose; we fix
the canonical order to match physical-intelligence sensing pipeline:
    raw -> pose drift -> noise -> outliers -> density drop -> local crop

This corresponds to: object pose first deviates from canonical; sensor
noise is added; some readings are outliers; density drops in some region;
local occlusion crops a chunk. Each later step operates on the already-
corrupted cloud.

For mixed-severity sweep, severities can be set per-op or all to the
same level.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .base import CorruptionOp

CANONICAL_ORDER: List[str] = ["pose", "noise", "outlier", "density", "crop"]


class ComposeCorruptions:
    """Stateless wrapper, NOT a CorruptionOp subclass (multi-op signature).

    Reproducibility note: each sub-op gets its own deterministic seed via
    the (taxonomy_id, model_id, view_id, severity, op_name) tuple, so the
    composed pipeline is bit-identical across runs.
    """

    def __init__(self, ops: Dict[str, CorruptionOp],
                 order: Optional[List[str]] = None):
        self.ops = ops
        self.order = order if order is not None else CANONICAL_ORDER
        for name in self.order:
            if name not in self.ops:
                raise KeyError(f"compose order '{name}' missing from ops dict")

    def __call__(self, points: np.ndarray, severities: Dict[str, int],
                 taxonomy_id: str, model_id: str, view_id: int) -> np.ndarray:
        out = points
        for name in self.order:
            sev = severities.get(name, 0)
            if sev == 0:
                continue
            out = self.ops[name](out, sev, taxonomy_id, model_id, view_id)
        return out

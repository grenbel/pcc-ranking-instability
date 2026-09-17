"""Failure-decomposition metrics (descriptive extras of the per-sample rows).

Each metric is a callable class with __call__(pred, gt) -> float (scalar
per-sample failure score, >=0; 0 means no failure of this mode). They are
emitted alongside CD/F-score but are not used by the paper's ranking statistics.
"""

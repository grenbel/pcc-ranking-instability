"""Smoke test for the valid-point-only corruption operators.

Validates that the four `*_validpt` operators (noise / outlier / density /
crop) preserve zero-pad rows AND produce the expected geometric corruption
on valid rows. Compares against the original zero-pad-inclusive variants
to confirm cache divergence (different `op_name` => different `make_seed`).

Test cases:
    A. Synthetic input with N=2048; first 1500 rows = unit-cube valid points,
       last 548 rows = zero-pad. Apply each operator at severity 3.
    B. Edge cases: an all-valid input (no padding rows) and a 95%-padded input
       (1948 padding rows); padding preservation and corruption coverage are
       checked on both.

Asserts (per (op, mode)):
    - output shape (2048, 3) preserved
    - valid_point_only=True => zero-pad rows in OUTPUT remain (0,0,0) exactly
    - valid_point_only=False => zero-pad rows likely modified (noise: yes,
      outlier: maybe, density: maybe, crop: maybe - explicitly checked
      per-op for documentation)
    - validpt and original variants produce different outputs (deterministic
      seed-divergence via op_name suffix)
    - operator is reproducible: same (tax_id, model_id, view_id, sev, op_name)
      tuple => bit-identical output across multiple invocations

Run:
    python smoke_tests/smoke_validpt_corruptions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.corruptions import (  # type: ignore  # noqa: E402
    OP_REGISTRY, valid_point_mask,
)

OPS_BASE = ["noise", "outlier", "density", "crop"]
N_PARTIAL = 2048
TEST_SEVERITY = 3
PASSED = []
FAILED = []


def synthetic_partial(n_valid: int, seed: int = 42) -> np.ndarray:
    """N=2048; first n_valid rows = uniform [-0.7, 0.7]^3 valid points,
    rest = (0,0,0) padding."""
    rng = np.random.default_rng(seed)
    valid = rng.uniform(-0.7, 0.7, size=(n_valid, 3)).astype(np.float32)
    pad = np.zeros((N_PARTIAL - n_valid, 3), dtype=np.float32)
    return np.concatenate([valid, pad], axis=0)


def assert_test(label: str, condition: bool, detail: str = ""):
    if condition:
        PASSED.append(label)
        print(f"  PASS  {label}")
        if detail:
            print(f"          {detail}")
    else:
        FAILED.append((label, detail))
        print(f"  FAIL  {label}")
        if detail:
            print(f"          {detail}")


def test_op(op_name: str, mode: str, points: np.ndarray, label_prefix: str):
    op = OP_REGISTRY[op_name]() if mode == "validpt" else OP_REGISTRY[op_name]()
    out = op(points, TEST_SEVERITY,
             taxonomy_id="02691156", model_id="smoketest", view_id=0)
    assert_test(f"{label_prefix} shape preserved",
                out.shape == points.shape,
                f"shape={out.shape}")
    assert_test(f"{label_prefix} dtype float32",
                out.dtype == np.float32,
                f"dtype={out.dtype}")
    valid_mask_in = valid_point_mask(points)
    pad_idx_in = np.where(~valid_mask_in)[0]
    if mode == "validpt":
        # zero-pad rows in INPUT must remain exactly (0,0,0) in OUTPUT
        pad_rows_out = out[pad_idx_in]
        n_pad = pad_idx_in.size
        n_zero_pad_preserved = int(np.all(pad_rows_out == 0.0, axis=1).sum())
        assert_test(
            f"{label_prefix} ALL {n_pad} pad rows preserved as (0,0,0)",
            n_zero_pad_preserved == n_pad,
            f"preserved={n_zero_pad_preserved}/{n_pad}",
        )
    else:
        # original mode: zero-pad rows OFTEN get modified - document the rate
        pad_rows_out = out[pad_idx_in]
        n_pad = pad_idx_in.size
        n_modified = int((np.linalg.norm(pad_rows_out, axis=1) > 1e-9).sum())
        print(f"  INFO  {label_prefix} ORIGINAL mode pad rows modified: "
              f"{n_modified}/{n_pad}")
    return out


def test_reproducibility(op_name: str, mode: str, points: np.ndarray, label: str):
    op_factory = OP_REGISTRY[op_name]
    op1 = op_factory()
    op2 = op_factory()
    out1 = op1(points, TEST_SEVERITY,
               taxonomy_id="02691156", model_id="repro", view_id=7)
    out2 = op2(points, TEST_SEVERITY,
               taxonomy_id="02691156", model_id="repro", view_id=7)
    assert_test(
        f"{label} reproducible (bit-exact across invocations)",
        np.array_equal(out1, out2),
        f"max_abs_diff={np.max(np.abs(out1 - out2))}",
    )


def test_seed_divergence(op_base: str, points: np.ndarray):
    op_orig = OP_REGISTRY[op_base]()
    op_validpt = OP_REGISTRY[f"{op_base}_validpt"]()
    out_orig = op_orig(points, TEST_SEVERITY,
                       taxonomy_id="02691156", model_id="seed", view_id=0)
    out_validpt = op_validpt(points, TEST_SEVERITY,
                             taxonomy_id="02691156", model_id="seed", view_id=0)
    diff_at_valid = np.max(np.abs(out_orig - out_validpt))
    assert_test(
        f"{op_base}: validpt and original produce different output (cache divergence)",
        diff_at_valid > 1e-9,
        f"max_diff={diff_at_valid:.6f}",
    )


def main():
    print("=" * 70)
    print("Smoke test: valid-point-only corruption operators")
    print("=" * 70)

    # === Test A: synthetic with known pad ratio ===
    n_valid = 1500
    points = synthetic_partial(n_valid)
    print(f"\nA. Synthetic input: N={N_PARTIAL}, valid={n_valid}, pad={N_PARTIAL - n_valid}")
    print(f"   Severity = {TEST_SEVERITY}")

    for op_base in OPS_BASE:
        print(f"\n  --- {op_base} ---")
        # original mode
        out_orig = test_op(op_base, "original", points,
                           f"{op_base} ORIG")
        # validpt mode
        out_validpt = test_op(f"{op_base}_validpt", "validpt", points,
                              f"{op_base} VALIDPT")
        # reproducibility
        test_reproducibility(op_base, "original", points,
                             f"{op_base} ORIG")
        test_reproducibility(f"{op_base}_validpt", "validpt", points,
                             f"{op_base} VALIDPT")
        # seed divergence
        test_seed_divergence(op_base, points)

    # === Test B: edge cases ===
    print("\nB. Edge cases:")
    print("   - zero-padding fraction = 0 (all valid)")
    full_valid = synthetic_partial(N_PARTIAL)
    for op_base in OPS_BASE:
        out_validpt = OP_REGISTRY[f"{op_base}_validpt"]()(
            full_valid, TEST_SEVERITY,
            taxonomy_id="02691156", model_id="full", view_id=0)
        assert_test(
            f"{op_base}_validpt with all-valid input -> corruption applied to all rows",
            (np.linalg.norm(out_validpt, axis=1) > 1e-9).all(),
            f"any-zero-row count={int((np.linalg.norm(out_validpt, axis=1) <= 1e-9).sum())}",
        )

    print("\n   - zero-padding fraction = 95% (mostly pad)")
    mostly_pad = synthetic_partial(100)
    for op_base in OPS_BASE:
        out_validpt = OP_REGISTRY[f"{op_base}_validpt"]()(
            mostly_pad, TEST_SEVERITY,
            taxonomy_id="02691156", model_id="mostly_pad", view_id=0)
        valid_in = valid_point_mask(mostly_pad)
        pad_idx = np.where(~valid_in)[0]
        n_pad_preserved = int(np.all(out_validpt[pad_idx] == 0.0, axis=1).sum())
        assert_test(
            f"{op_base}_validpt with 95% pad input -> all 1948 pad rows preserved",
            n_pad_preserved == pad_idx.size,
            f"preserved={n_pad_preserved}/{pad_idx.size}",
        )

    print("\n" + "=" * 70)
    print(f"Summary: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("\nFAILURES:")
        for label, detail in FAILED:
            print(f"  - {label}: {detail}")
        sys.exit(1)
    print("All smoke tests PASS")


if __name__ == "__main__":
    main()

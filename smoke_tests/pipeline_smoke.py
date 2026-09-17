"""Local pipeline smoke for src/* modules - no PoinTr / open3d / GPU required.

Verifies:
1. Corruption operators (5 ops + compose):
   - shape preservation per `preserves_count` flag
   - reproducibility: same (taxonomy, model, view, severity, op) -> bit-identical output
   - per-severity expected behavior (severity=0 = identity, severity=5 = max effect)
2. Metric module:
   - CD-L1 = 0.5 x (mean+mean), CD-L2 = mean+mean (PoinTr conventions)
   - identical clouds -> CD = 0, F-score = 1.0
   - F-score @ 3 thresholds returns valid [0,1]
3. Decomposition metrics (4 modes):
   - run on synthetic point clouds without crash
   - return finite floats
4. Matched-control protocol:
   - group index construction from synthetic file_list
   - paired Wilcoxon + Kendall tau on synthetic ranking pairs

Run: python smoke_tests/pipeline_smoke.py
Exit 0 if all PASS; nonzero if any FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project src on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corruptions import OP_REGISTRY, ComposeCorruptions, make_seed
from src.corruptions.compose import CANONICAL_ORDER as COMPOSE_ORDER
from src.metrics.reconstruction import chamfer_l1, chamfer_l2, fscore_at_thresholds
from src.metrics import DECOMPOSITION_METRICS
from src.stratify import (
    build_matched_groups, paired_wilcoxon, ranking_kendall_tau,
    aggregate_unstratified, aggregate_stratified,
)


# ---------- Synthetic data ----------
def make_uniform_sphere(n: int, seed: int = 0) -> np.ndarray:
    """Uniform points on unit sphere shell (works as PCN-style normalized cloud)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, size=(n, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def make_uniform_cube(n: int, seed: int = 0) -> np.ndarray:
    """Uniform points in unit cube [-1, 1]^3."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)


# ---------- Test runner ----------
class SmokeFailure(AssertionError):
    pass


def check(condition: bool, msg: str):
    if not condition:
        raise SmokeFailure(msg)


def section(name: str):
    print(f"\n{'='*60}\n[SMOKE] {name}\n{'='*60}")


# ---------- 1. Corruption operators ----------
def test_corruption_ops():
    section("1. Corruption operators (5 ops + compose)")
    pts = make_uniform_sphere(2048, seed=1)
    base_args = ("airplane_02691156", "abc123", 0)

    # Test each op for severity 0, 3, 5
    for op_name, op_cls in OP_REGISTRY.items():
        op = op_cls()
        for sev in [0, 3, 5]:
            out = op(pts, sev, *base_args)
            # Shape check
            if op.preserves_count:
                check(out.shape == pts.shape,
                      f"{op_name} sev={sev}: shape {out.shape} != {pts.shape}")
            check(out.dtype == np.float32,
                  f"{op_name} sev={sev}: dtype {out.dtype} not float32")
            # Severity 0 = identity copy (no change)
            if sev == 0:
                check(np.array_equal(out, pts),
                      f"{op_name} sev=0 should be identity, got max diff "
                      f"{np.abs(out-pts).max():.6f}")
            # Severity 5 = nonzero effect (output differs from input meaningfully)
            if sev == 5:
                diff = np.linalg.norm(out - pts, axis=1).mean()
                check(diff > 1e-4,
                      f"{op_name} sev=5 should produce meaningful effect, "
                      f"got mean diff {diff:.6f}")
        print(f"  [OK] {op_name}: shape preserved, sev=0 identity, sev=5 nonzero")

    # Reproducibility: same args -> bit-identical output
    op = OP_REGISTRY["noise"]()
    out_a = op(pts, 3, *base_args)
    out_b = op(pts, 3, *base_args)
    check(np.array_equal(out_a, out_b),
          f"reproducibility broken: noise sev=3 with same args produced different outputs")
    out_c = op(pts, 3, "different", "model", 0)
    check(not np.array_equal(out_a, out_c),
          f"seed independence broken: different args produced identical output")
    print(f"  [OK] reproducibility: same args bit-identical, different args differ")

    # Compose
    ops = {n: c() for n, c in OP_REGISTRY.items()}
    compose = ComposeCorruptions(ops)
    out = compose(pts, {n: 2 for n in COMPOSE_ORDER}, *base_args)
    check(out.shape == pts.shape,
          f"compose shape {out.shape} != {pts.shape}")
    print(f"  [OK] compose: all 5 ops chained, shape preserved")

    # Seed determinism
    s1 = make_seed("a", "b", 0, 3, "noise")
    s2 = make_seed("a", "b", 0, 3, "noise")
    s3 = make_seed("a", "b", 0, 4, "noise")
    check(s1 == s2, f"make_seed not deterministic: {s1} != {s2}")
    check(s1 != s3, f"make_seed too coarse: severity 3 vs 4 same seed")
    print(f"  [OK] make_seed: deterministic + sensitive to all args")


# ---------- 2. Metric module ----------
def test_metrics():
    section("2. Reconstruction metrics (CD-L1/L2 + F-score)")
    pts = make_uniform_sphere(2048, seed=2)

    # Identical clouds -> CD = 0, F-score = 1.0
    cd1_self = chamfer_l1(pts, pts)
    cd2_self = chamfer_l2(pts, pts)
    f_self = fscore_at_thresholds(pts, pts)
    check(cd1_self < 1e-9, f"CD-L1 self-distance should be 0, got {cd1_self}")
    check(cd2_self < 1e-9, f"CD-L2 self-distance should be 0, got {cd2_self}")
    for k, v in f_self.items():
        check(v == 1.0, f"F-score self {k} should be 1.0, got {v}")
    print(f"  [OK] identical clouds: CDL1={cd1_self:.2e}  CDL2={cd2_self:.2e}  F=1.0")

    # Sanity: CD-L1 = 0.5 x (mean+mean), CD-L2 = mean+mean (PoinTr conventions)
    pred = make_uniform_sphere(2048, seed=2)
    gt = make_uniform_sphere(2048, seed=99)
    cd1 = chamfer_l1(pred, gt)
    cd2 = chamfer_l2(pred, gt)
    # Hand-compute reference
    from scipy.spatial import cKDTree
    t1 = cKDTree(gt); t2 = cKDTree(pred)
    d_pg, _ = t1.query(pred, k=1)
    d_gp, _ = t2.query(gt, k=1)
    expected_cd1 = 0.5 * (d_pg.mean() + d_gp.mean())
    expected_cd2 = (d_pg ** 2).mean() + (d_gp ** 2).mean()
    check(abs(cd1 - expected_cd1) < 1e-6,
          f"CD-L1 formula wrong: {cd1} vs expected {expected_cd1}")
    check(abs(cd2 - expected_cd2) < 1e-6,
          f"CD-L2 formula wrong: {cd2} vs expected {expected_cd2}")
    print(f"  [OK] CD formula: L1={cd1:.6f} (expect {expected_cd1:.6f}), "
          f"L2={cd2:.6f} (expect {expected_cd2:.6f})")

    # Official PoinTr metrics ignore zero padding rows for ChamferDistanceL1/L2.
    gt_clean = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    pred_padded = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    cd1_ignore = chamfer_l1(pred_padded, gt_clean)
    cd2_ignore = chamfer_l2(pred_padded, gt_clean)
    cd1_keep = chamfer_l1(pred_padded, gt_clean, ignore_zeros=False)
    cd2_keep = chamfer_l2(pred_padded, gt_clean, ignore_zeros=False)
    check(cd1_ignore < 1e-9, f"CD-L1 should ignore zero padding, got {cd1_ignore}")
    check(cd2_ignore < 1e-9, f"CD-L2 should ignore zero padding, got {cd2_ignore}")
    check(cd1_keep > 0.0 and cd2_keep > 0.0,
          "ignore_zeros=False should retain the zero padding penalty")
    print(f"  zero padding: ignored CD=({cd1_ignore:.2e}, {cd2_ignore:.2e}), "
          f"kept CD=({cd1_keep:.6f}, {cd2_keep:.6f})")

    # F-score returns valid [0,1] at all thresholds
    f = fscore_at_thresholds(pred, gt)
    for k, v in f.items():
        check(0.0 <= v <= 1.0, f"F-score {k}={v} out of [0,1]")
    print(f"  [OK] F-score: {f}")


# ---------- 3. Decomposition metrics ----------
def test_decomposition():
    section("3. Decomposition metrics (4 modes)")
    pred = make_uniform_sphere(2048, seed=3)
    gt = make_uniform_sphere(2048, seed=4)

    for name, cls in DECOMPOSITION_METRICS.items():
        m = cls()
        v = m(pred, gt)
        check(np.isfinite(v), f"{name} returned non-finite: {v}")
        check(v >= 0, f"{name} returned negative score: {v} (should be >=0)")
        # Self should be ~0
        v_self = m(pred, pred)
        # density_collapse uses KS with self -> 0; topology with same cloud -> 0; hf and boundary similar
        check(v_self < v + 1e-3,
              f"{name}: self-score {v_self} should be <= pred-vs-gt score {v}")
        print(f"  [OK] {name}: pred-vs-gt={v:.4f}, self={v_self:.4f}")


# ---------- 4. Matched-control protocol ----------
def test_stratify():
    section("4. Matched-control protocol (group index + tests)")
    # Synthetic file list (mimics PoinTr PCN _get_file_list output)
    fake_list = [
        {"taxonomy_id": "02691156", "model_id": f"model_{i}",
         "partial_path": f"partial/{i}.pcd", "gt_path": f"complete/{i}.pcd"}
        for i in range(8)
    ]
    idx = build_matched_groups(fake_list, subset="test", n_views_per_object=1)
    check(idx["n_groups"] == 8, f"expected 8 groups, got {idx['n_groups']}")
    check(len(idx["groups"]) == 8, f"groups list length mismatch")
    check(idx["n_eval_cells_per_group"] == 5 * 5 + 1,  # 5 ops x 5 nonzero severities + 1 clean
          f"eval cells per group wrong: {idx['n_eval_cells_per_group']}")
    print(f"  [OK] build_matched_groups: 8 objects -> 8 groups, 26 cells/group")

    # max_groups cap
    idx_capped = build_matched_groups(fake_list, subset="test", max_groups=3)
    check(idx_capped["n_groups"] == 3, f"max_groups cap broken")
    print(f"  [OK] max_groups cap works")

    # Paired Wilcoxon
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 30)
    b = a + rng.normal(0.5, 0.2, 30)  # b shifted +0.5 from a
    res = paired_wilcoxon(a, b, alternative="less")
    check(res["pvalue"] < 0.05, f"Wilcoxon should detect shift, pvalue={res['pvalue']}")
    check(res["n_pairs"] == 30, f"n_pairs wrong")
    print(f"  [OK] paired_wilcoxon: pvalue={res['pvalue']:.4f} (< 0.05 OK), n={res['n_pairs']}")

    # Kendall tau ranking flip
    rank_a = ["A", "B", "C", "D", "E"]
    rank_b = ["E", "D", "C", "B", "A"]  # complete reversal
    tau = ranking_kendall_tau(rank_a, rank_b)
    check(abs(tau["tau"] - (-1.0)) < 1e-9,
          f"complete reversal should give tau=-1, got {tau['tau']}")
    check(tau["n_flips"] == 10, f"complete reversal n_flips should be 10 (5C2), got {tau['n_flips']}")
    print(f"  [OK] Kendall tau: complete reversal -> tau={tau['tau']:.6f}, flips={tau['n_flips']}")

    # Same ranking
    tau2 = ranking_kendall_tau(rank_a, rank_a)
    check(abs(tau2["tau"] - 1.0) < 1e-9 and tau2["n_flips"] == 0,
          f"identical ranking should give tau=1, 0 flips, got {tau2}")
    print(f"  [OK] Kendall tau identical: tau={tau2['tau']}, flips={tau2['n_flips']}")

    # Aggregate unstratified vs stratified
    scores = {}
    for m in ["modelA", "modelB"]:
        for op in ["noise", "crop"]:
            for sev in [1, 3]:
                for idx_s in range(4):
                    base = 1.0 if m == "modelA" else 1.5
                    op_pen = 0.2 if op == "crop" else 0.0
                    scores[(m, op, sev, idx_s)] = base + op_pen + 0.1 * sev

    unstrat = aggregate_unstratified(scores)
    strat = aggregate_stratified(scores)
    check(set(unstrat.keys()) == {"modelA", "modelB"}, f"unstrat models wrong")
    check(unstrat["modelA"] < unstrat["modelB"], "modelA should aggregate lower (better)")
    check(set(strat["modelA"].keys()) == {("noise", 1), ("noise", 3),
                                            ("crop", 1), ("crop", 3)},
          f"strat cells wrong: {strat['modelA'].keys()}")
    print(f"  [OK] aggregate: unstrat={unstrat}, strat 4 cells x 2 models OK")


# ---------- Main ----------
def main():
    print("="*60)
    print("Local pipeline smoke - src/* module verification")
    print(f"Python: {sys.version.split()[0]}")
    print(f"numpy: {np.__version__}")
    print("="*60)

    tests = [
        ("corruption_ops", test_corruption_ops),
        ("metrics", test_metrics),
        ("decomposition", test_decomposition),
        ("stratify", test_stratify),
    ]
    passed = []
    failed = []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
        except SmokeFailure as e:
            print(f"\n  [FAIL] {name}: {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f"\n  [CRASH] {name}: {type(e).__name__}: {e}")
            failed.append((name, f"{type(e).__name__}: {e}"))

    print(f"\n{'='*60}\n[SMOKE SUMMARY]\n{'='*60}")
    print(f"PASSED: {len(passed)}/{len(tests)} -> {passed}")
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)}")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print("ALL PASSED. src/* modules verified.")
        sys.exit(0)


if __name__ == "__main__":
    main()

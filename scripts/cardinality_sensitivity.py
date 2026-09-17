"""Cardinality-controlled sensitivity check on the ranking flip cells.

PoinTr's post-mask predictions have ~15450 effective points (after the sum-zero
filter) vs AdaPoinTr's full 16384, so the PoinTr-vs-AdaPoinTr comparison is not at
the same cardinality even though it is protocol-compatible. This script subsamples
both predictions to a common N (default 14336 = PoinTr's fold-output cardinality;
the observed minimum post-mask count is 14730) and recomputes the paired Wilcoxon
test to verify that the ranking flips persist.

The flip cells checked (zero-pad protocol, PoinTr vs AdaPoinTr):
  noise@s5, outlier@s1, outlier@s2, outlier@s3, outlier@s4, outlier@s5

This sensitivity check uses Bonferroni over **6 tests** (focused, not 40), so the
threshold is alpha/6 = 0.05/6 ~ 0.00833.

Usage:
    # Smoke
    python scripts/cardinality_sensitivity.py \\
        --pcn-data-root /path/to/PCN \\
        --pred-dir preds/zero_pad \\
        --output-dir logs/cardinality_smoke \\
        --mode random --n-points 14336 --max-samples-per-cell 50 --seed 42

    # Full
    python scripts/cardinality_sensitivity.py \\
        --pcn-data-root /path/to/PCN \\
        --pred-dir preds/zero_pad \\
        --output-dir logs/cardinality \\
        --mode both --n-points 14336 --max-samples-per-cell -1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
POINTR_ROOT = REPO_ROOT / "baselines" / "PoinTr"
sys.path.insert(0, str(POINTR_ROOT))
sys.path.insert(0, str(POINTR_ROOT / "extensions" / "chamfer_dist"))
sys.path.insert(0, str(REPO_ROOT))

from extensions.chamfer_dist import ChamferFunction  # type: ignore
from pointnet2_ops import pointnet2_utils  # type: ignore

from scripts.sanity_clean_pcn import DEFAULT_MODELS, load_pcn_test_loader  # type: ignore
from scripts.sweep_common import (  # type: ignore
    DEFAULT_OPS, DEFAULT_SEVERITIES, cache_clean_pcn_test,
)
from scripts.forward_sweep import NPZ_SCHEMA_VERSION, compute_cache_hash  # type: ignore


# The 6 PoinTr-vs-AdaPoinTr flip cells of the zero-pad sweep
FLIP_CELLS = [
    ("noise",   5),
    ("outlier", 1),
    ("outlier", 2),
    ("outlier", 3),
    ("outlier", 4),
    ("outlier", 5),
]
N_FLIP_TESTS = len(FLIP_CELLS)  # 6
ALPHA = 0.05
ALPHA_BONFERRONI = ALPHA / N_FLIP_TESTS  # 0.00833


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcn-data-root", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", choices=["random", "fps", "both"], default="both",
                   help="Subsample mode. 'both' runs random then fps for cross-check.")
    p.add_argument("--n-points", type=int, default=14336,
                   help="Common N for subsample. Default 14336 = PoinTr fold output cardinality "
                        "(PoinTr post-mask minimum observed 14730, so 14336 is safe).")
    p.add_argument("--max-samples-per-cell", type=int, default=-1,
                   help="Number of leading samples per cell to process (-1 = all); the manifest "
                        "cache hash is always verified against the full 1200-sample cache first")
    p.add_argument("--batch-size", type=int, default=32,
                   help="CUDA chamfer batch size for the per-sample CD vector "
                        "(ChamferFunction.apply, not the module wrapper, for batched per-sample).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def drop_zero_rows_sum_mask(points: np.ndarray) -> np.ndarray:
    """Official sum-zero rule (`sum != 0`), not the strict (0,0,0) rule."""
    return points[points.sum(axis=1) != 0]


def per_sample_seed(model: str, op: str, sev: int, idx: int, base_seed: int) -> int:
    """Stable SHA-256 derived seed per (model, op, sev, sample_idx)."""
    h = hashlib.sha256(f"{model}|{op}|s{sev}|i{idx}|{base_seed}".encode()).digest()
    return int.from_bytes(h[:4], "big")  # 32-bit


def subsample_random(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    if points.shape[0] < n:
        raise RuntimeError(f"random subsample: have {points.shape[0]} pts < n={n}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=n, replace=False)
    return points[idx]


@torch.no_grad()
def subsample_fps_batch(points_list: list, n: int, device: str) -> list:
    """Furthest point sampling via pointnet2_ops (PoinTr already depends on it).

    Each item in points_list has different cardinality (post-mask). FPS deterministic given
    fixed input ordering - index 0 always selected first. Returns list of (n, 3) numpy arrays.
    """
    out = []
    # Process one at a time because cardinalities vary post-mask.
    for pts in points_list:
        if pts.shape[0] < n:
            raise RuntimeError(f"fps subsample: have {pts.shape[0]} pts < n={n}")
        x = torch.from_numpy(pts.astype(np.float32)).to(device).unsqueeze(0).contiguous()  # (1, M, 3)
        idx = pointnet2_utils.furthest_point_sample(x, n)  # (1, n) int
        idx = idx.long()
        sampled = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (1, n, 3)
        out.append(sampled.squeeze(0).cpu().numpy())
    return out


@torch.no_grad()
def batched_cd_l1_per_sample(pred_batch_np: np.ndarray, gt_batch_np: np.ndarray,
                              device: str) -> np.ndarray:
    """Batched per-sample CD-L1 via ChamferFunction.apply (not the ChamferDistanceL1 module).

    Returns (B,) per-sample CD-L1 = 0.5 * (mean sqrt(d1) + mean sqrt(d2)).

    pred_batch_np: (B, N_pred, 3) - assumed already same cardinality (post-subsample)
    gt_batch_np:   (B, N_gt, 3)
    """
    pred = torch.from_numpy(pred_batch_np.astype(np.float32)).to(device).contiguous()
    gt   = torch.from_numpy(gt_batch_np.astype(np.float32)).to(device).contiguous()
    dist1, dist2 = ChamferFunction.apply(pred, gt)  # squared distances, (B, N_pred), (B, N_gt)
    cd_l1 = 0.5 * (torch.sqrt(dist1).mean(dim=1) + torch.sqrt(dist2).mean(dim=1))
    return cd_l1.cpu().numpy()  # (B,)


def load_cell_preds(pred_dir: Path, model: str, op: str, sev: int, max_samples: int,
                    expected_hash: str = None) -> dict:
    npz_path = pred_dir / f"{model}__{op}__s{sev}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing pred file: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        preds = z["preds"]  # (N, 16384, 3)
        sample_indices = z["sample_indices"]
        n_pred_pts = int(z["n_pred_points"])
        concat_applied = bool(z["concat_partial_applied"])
        schema = str(z.get("schema_version", "?"))
        has_stamp = "cache_hash_sha256" in z.files
        stamp = str(z["cache_hash_sha256"]) if has_stamp else ""
    if schema != NPZ_SCHEMA_VERSION:
        raise RuntimeError(f"{npz_path.name} schema={schema} != current {NPZ_SCHEMA_VERSION}")
    # A per-file cache_hash field, when present (even if empty), must match the verified
    # manifest hash (legacy archives without the field are covered by the manifest check
    # in main()).
    if expected_hash and has_stamp and stamp != expected_hash:
        raise RuntimeError(f"{npz_path.name} cache_hash stamp {stamp[:16]}... != manifest "
                           f"{expected_hash[:16]}... (forwarded against a different cache)")
    if n_pred_pts != 16384:
        raise RuntimeError(f"{npz_path.name} n_pred_points={n_pred_pts} != 16384")
    if concat_applied:
        raise RuntimeError(f"{npz_path.name} concat_partial_applied={concat_applied} (must be False)")
    if max_samples > 0:
        preds = preds[:max_samples]
        sample_indices = sample_indices[:max_samples]
    return {"preds": preds, "sample_indices": sample_indices}


def run_one_cell(op: str, sev: int, mode: str, n_pts: int, max_samples: int,
                 pred_dir: Path, gt_cache: list, args, fps_device: str,
                 expected_hash: str = None) -> dict:
    """Run one cell: load preds for both models, subsample to common N, compute per-sample CD-L1."""
    print(f"\n[cell {op}@s{sev} mode={mode} N={n_pts}]")
    P_data = load_cell_preds(pred_dir, "PoinTr",   op, sev, max_samples, expected_hash)
    A_data = load_cell_preds(pred_dir, "AdaPoinTr", op, sev, max_samples, expected_hash)
    P_preds = P_data["preds"]
    A_preds = A_data["preds"]
    n_samples = P_preds.shape[0]
    if A_preds.shape[0] != n_samples:
        raise RuntimeError(f"sample count mismatch P={n_samples} A={A_preds.shape[0]}")

    # Use cached GT (already pre-loaded once for whole run, indexed by idx)
    P_indices = P_data["sample_indices"]
    A_indices = A_data["sample_indices"]
    if not np.array_equal(P_indices, A_indices):
        raise RuntimeError(f"sample_indices mismatch between PoinTr and AdaPoinTr at {op}@s{sev}")

    gts_full = []  # (n_samples, 16384, 3)
    for sidx in P_indices:
        e = gt_cache[int(sidx)]
        gts_full.append(e["gt_np"])
    gts_full = np.stack(gts_full, axis=0)  # (n, 16384, 3)

    # 1) Apply sum-zero mask to PoinTr; AdaPoinTr should have no zero rows but apply anyway
    #    (subsample only the predictions; the GT remains the full 16384).
    P_masked, A_masked = [], []
    for i in range(n_samples):
        Pm = drop_zero_rows_sum_mask(P_preds[i])
        Am = drop_zero_rows_sum_mask(A_preds[i])
        if Pm.shape[0] < n_pts:
            raise RuntimeError(
                f"[{op}@s{sev} i={i}] PoinTr post-mask {Pm.shape[0]} < n_pts={n_pts}. "
                f"Lower --n-points or skip cell."
            )
        if Am.shape[0] < n_pts:
            raise RuntimeError(
                f"[{op}@s{sev} i={i}] AdaPoinTr post-mask {Am.shape[0]} < n_pts={n_pts}."
            )
        P_masked.append(Pm)
        A_masked.append(Am)

    # 2) Subsample to common N
    t0 = time.time()
    if mode == "random":
        P_sub = [subsample_random(P_masked[i], n_pts,
                                   per_sample_seed("PoinTr", op, sev, int(P_indices[i]), args.seed))
                 for i in range(n_samples)]
        A_sub = [subsample_random(A_masked[i], n_pts,
                                   per_sample_seed("AdaPoinTr", op, sev, int(A_indices[i]), args.seed))
                 for i in range(n_samples)]
    elif mode == "fps":
        P_sub = subsample_fps_batch(P_masked, n_pts, fps_device)
        A_sub = subsample_fps_batch(A_masked, n_pts, fps_device)
    else:
        raise RuntimeError(f"unknown mode {mode}")
    sub_t = time.time() - t0

    # 3) Stack into batches for CUDA chamfer
    P_arr = np.stack(P_sub, axis=0).astype(np.float32)  # (n, N_pts, 3)
    A_arr = np.stack(A_sub, axis=0).astype(np.float32)
    if P_arr.shape != (n_samples, n_pts, 3) or A_arr.shape != (n_samples, n_pts, 3):
        raise RuntimeError(f"subsample shape mismatch P={P_arr.shape} A={A_arr.shape}")

    # 4) Per-sample CD-L1 in batches (gt always full 16384)
    bs = args.batch_size
    P_cdl1 = np.empty(n_samples, dtype=np.float64)
    A_cdl1 = np.empty(n_samples, dtype=np.float64)
    t0 = time.time()
    for start in range(0, n_samples, bs):
        end = min(start + bs, n_samples)
        P_cdl1[start:end] = batched_cd_l1_per_sample(P_arr[start:end], gts_full[start:end], args.device)
        A_cdl1[start:end] = batched_cd_l1_per_sample(A_arr[start:end], gts_full[start:end], args.device)
    cd_t = time.time() - t0

    # 5) Paired Wilcoxon
    diff = A_cdl1 - P_cdl1
    try:
        stat, pval = wilcoxon(A_cdl1, P_cdl1)
        pval = float(pval)
    except ValueError:
        pval = float("nan")
    winner_uncorr = (
        "AdaPoinTr" if (pval < ALPHA and diff.mean() < 0)
        else ("PoinTr" if (pval < ALPHA and diff.mean() > 0) else "NS")
    )
    winner_bonf = (
        "AdaPoinTr" if (pval < ALPHA_BONFERRONI and diff.mean() < 0)
        else ("PoinTr" if (pval < ALPHA_BONFERRONI and diff.mean() > 0) else "NS_bonf6")
    )

    print(f"  P CDL1x1000={P_cdl1.mean()*1000:.4f}  A CDL1x1000={A_cdl1.mean()*1000:.4f}  "
          f"diff={(A_cdl1-P_cdl1).mean()*1000:.4f}  p={pval:.3e}  "
          f"winner_a05={winner_uncorr}  winner_bonf6={winner_bonf}  "
          f"sub={sub_t:.1f}s cd={cd_t:.1f}s")

    return {
        "op": op, "sev": sev, "mode": mode, "n_pts": n_pts, "n_samples": n_samples,
        "PoinTr_cdl1_x1000_mean":   float(P_cdl1.mean() * 1000),
        "AdaPoinTr_cdl1_x1000_mean": float(A_cdl1.mean() * 1000),
        "diff_x1000_mean":   float(diff.mean() * 1000),
        "diff_x1000_median": float(np.median(diff) * 1000),
        "frac_AdaPoinTr_worse": float((diff > 0).mean()),
        "wilcoxon_pvalue": pval,
        "winner_uncorrected_a05": winner_uncorr,
        "winner_bonferroni6":     winner_bonf,
        "expected_winner": "PoinTr",  # all 6 cells are PoinTr-flip per v2
        "flip_sign_preserved": bool(winner_bonf == "PoinTr"),
        "elapsed_subsample_sec": sub_t,
        "elapsed_chamfer_sec":  cd_t,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        sys.exit(f"[card-sens] FATAL: CUDA required (device={args.device})")
    if args.max_samples_per_cell == 0:
        sys.exit("[card-sens] FATAL: --max-samples-per-cell cannot be 0")

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load forward manifest + verify
    manifest_path = pred_dir / "forward_manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"[card-sens] FATAL: forward manifest not found at {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != NPZ_SCHEMA_VERSION:
        sys.exit(f"[card-sens] FATAL: manifest schema={manifest.get('schema_version')} != {NPZ_SCHEMA_VERSION}")

    # Always cache the FULL 1200 for the hash check (the manifest hash is over the full
    # forward cache), then slice to args.max_samples_per_cell for processing.
    print(f"[card-sens] Loading PCN GT FULL cache from {args.pcn_data_root} (for hash verify)")
    loader = load_pcn_test_loader(args.pcn_data_root, batch_size=1)
    full_cache = cache_clean_pcn_test(loader, max_samples=-1, seed=args.seed)
    fm_hash = manifest.get("cache_hash_sha256")
    metric_hash = compute_cache_hash(full_cache)
    if metric_hash != fm_hash:
        sys.exit(f"[card-sens] FATAL: cache_hash mismatch:\n  forward={fm_hash}\n  metric ={metric_hash}")
    print(f"[card-sens] cache_hash bit-exact verified ({metric_hash[:16]}..., {len(full_cache)} samples)")
    # For per-cell processing, slice prefix per --max-samples-per-cell. NPZ preds are also prefix-ordered.
    clean_cache = full_cache if args.max_samples_per_cell <= 0 else full_cache[:args.max_samples_per_cell]
    print(f"[card-sens] processing {len(clean_cache)} samples per cell (--max-samples-per-cell={args.max_samples_per_cell})")

    modes = ["random", "fps"] if args.mode == "both" else [args.mode]
    results = {}
    grand_t0 = time.time()
    for mode in modes:
        results[mode] = []
        for op, sev in FLIP_CELLS:
            r = run_one_cell(op, sev, mode, args.n_points, args.max_samples_per_cell,
                              pred_dir, clean_cache, args, args.device, expected_hash=fm_hash)
            results[mode].append(r)

    # Aggregate decision
    summary = {
        "schema_version": "1.0",
        "run_id": "cardinality_sensitivity",
        "args": vars(args),
        "alpha_uncorrected": ALPHA,
        "alpha_bonferroni6": ALPHA_BONFERRONI,
        "n_flip_tests": N_FLIP_TESTS,
        "elapsed_total_sec": time.time() - grand_t0,
        "per_mode_per_cell": results,
    }
    decision = {}
    for mode, cell_list in results.items():
        n_preserved = sum(1 for c in cell_list if c["flip_sign_preserved"])
        decision[mode] = {
            "n_preserved": n_preserved,
            "all_6_preserved_bonf": bool(n_preserved == N_FLIP_TESTS),
            "cells_failed": [(c["op"], c["sev"]) for c in cell_list if not c["flip_sign_preserved"]],
        }
    summary["decision"] = decision
    summary["overall_pass"] = bool(all(d["all_6_preserved_bonf"] for d in decision.values()))

    out_path = out_dir / "cardinality_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("[card-sens] DECISION")
    print("=" * 80)
    for mode, d in decision.items():
        print(f"  mode={mode}: {d['n_preserved']}/{N_FLIP_TESTS} flips preserved at Bonferroni alpha/6 = {ALPHA_BONFERRONI:.5f}")
        if d["cells_failed"]:
            print(f"    FAILED: {d['cells_failed']}")
    print(f"\n  overall_pass = {summary['overall_pass']}")
    print(f"\n[card-sens] Summary written to {out_path}")
    if not summary["overall_pass"]:
        print("[card-sens] EXIT 1 - at least one flip did not preserve under cardinality control.")
        sys.exit(1)


if __name__ == "__main__":
    main()

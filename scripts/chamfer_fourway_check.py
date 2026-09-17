"""Four-way Chamfer validation: numpy vs CUDA Chamfer x with-zero vs ignore-zero.

Validates that on the clean PCN test split (1200 samples) the four metric
variants produce numerically equivalent CD-L1/CD-L2 (within float epsilon),
confirming numpy<->CUDA chamfer cross-implementation agreement that the numpy
re-run of the corruption cells alone cannot establish.

Variants:
1. numpy + ignore_zeros=False  (= old buggy default; PoinTr expected ~8.339)
2. numpy + ignore_zeros=True   (= current default; PoinTr expected ~7.26)
3. CUDA  + ignore_zeros=False  (PoinTr should match #1 within fp epsilon)
4. CUDA  + ignore_zeros=True   (PoinTr should match #2 within fp epsilon)

Decision gate: PASS if for each model, |numpy_with_zero - cuda_with_zero| < 0.001 (x1000)
and |numpy_ignore_zero - cuda_ignore_zero| < 0.001 (x1000) on CD-L1 mean.

Usage:
    # Smoke (50 samples)
    python scripts/chamfer_fourway_check.py --pcn-data-root /path/to/PCN \\
        --output-dir logs/chamfer_fourway_smoke \\
        --max-samples 50 --device cuda:0

    # Full (1200 samples)
    python scripts/chamfer_fourway_check.py --pcn-data-root /path/to/PCN \\
        --output-dir logs/chamfer_fourway \\
        --max-samples -1 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
POINTR_ROOT = REPO_ROOT / "baselines" / "PoinTr"
sys.path.insert(0, str(POINTR_ROOT))
sys.path.insert(0, str(POINTR_ROOT / "extensions" / "chamfer_dist"))
sys.path.insert(0, str(REPO_ROOT))

# Reuse sanity_clean_pcn.py infra (DEFAULT_MODELS, loaders, model loading, cwd helper)
from scripts.sanity_clean_pcn import (  # type: ignore
    DEFAULT_MODELS, load_pcn_test_loader, load_model,
    cwd_to,
)
from src.metrics.reconstruction import chamfer_l1 as numpy_chamfer_l1
from src.metrics.reconstruction import chamfer_l2 as numpy_chamfer_l2

# CUDA Chamfer: import path is `extensions.chamfer_dist` (PoinTr's built extension)
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcn-data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=-1,
                   help="-1 = full 1200; e.g. 50 for smoke")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Locked to 1: the CUDA chamfer ignore_zeros mask is only "
                        "applied when batch_size==1.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agreement-threshold-x1000", type=float, default=0.001,
                   help="numpy<->CUDA CD-L1 agreement threshold in x1000 units. "
                        "Default 0.001 (1e-6 absolute). Smaller = stricter.")
    p.add_argument("--write-per-sample", action="store_true",
                   help="If set, also save per-sample 4-way CD JSONs (debug).")
    return p.parse_args()


def make_cuda_modules(device: str) -> dict:
    """Instantiate all 4 module variants once and reuse them across samples."""
    return {
        "cd_l1_with_zero":   ChamferDistanceL1(ignore_zeros=False).to(device),
        "cd_l1_ignore_zero": ChamferDistanceL1(ignore_zeros=True).to(device),
        "cd_l2_with_zero":   ChamferDistanceL2(ignore_zeros=False).to(device),
        "cd_l2_ignore_zero": ChamferDistanceL2(ignore_zeros=True).to(device),
    }


@torch.no_grad()
def four_way_cd(pred_t: torch.Tensor, gt_t: torch.Tensor,
                pred_np: np.ndarray, gt_np: np.ndarray,
                cuda_modules: dict) -> dict:
    """Compute 4 variants of CD-L1 + 4 variants of CD-L2 on the SAME pred/gt pair.

    pred_t / gt_t: (1, N, 3) float32 cuda tensors
    pred_np / gt_np: (N, 3) float32 numpy

    Returns dict with 8 metric values.
    """
    # numpy variants (existing src/metrics/reconstruction.py - sum-zero mask)
    numpy_l1_iz_false = numpy_chamfer_l1(pred_np, gt_np, ignore_zeros=False)
    numpy_l1_iz_true  = numpy_chamfer_l1(pred_np, gt_np, ignore_zeros=True)
    numpy_l2_iz_false = numpy_chamfer_l2(pred_np, gt_np, ignore_zeros=False)
    numpy_l2_iz_true  = numpy_chamfer_l2(pred_np, gt_np, ignore_zeros=True)

    # CUDA variants
    # PoinTr's ChamferDistanceL1 (line 64-84 in extensions/chamfer_dist/__init__.py):
    #   dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)  # squared distances
    #   dist1 = sqrt(dist1); dist2 = sqrt(dist2)
    #   return (mean(dist1) + mean(dist2)) / 2
    # = 0.5 * (mean L2 dist xyz1->xyz2 + mean L2 dist xyz2->xyz1)
    # Identical formula to our numpy chamfer_l1.
    cuda_l1_iz_false = float(cuda_modules["cd_l1_with_zero"](pred_t, gt_t).item())
    cuda_l1_iz_true  = float(cuda_modules["cd_l1_ignore_zero"](pred_t, gt_t).item())
    cuda_l2_iz_false = float(cuda_modules["cd_l2_with_zero"](pred_t, gt_t).item())
    cuda_l2_iz_true  = float(cuda_modules["cd_l2_ignore_zero"](pred_t, gt_t).item())

    return {
        # CD-L1
        "numpy_l1_with_zero":     numpy_l1_iz_false,
        "numpy_l1_ignore_zero":   numpy_l1_iz_true,
        "cuda_l1_with_zero":      cuda_l1_iz_false,
        "cuda_l1_ignore_zero":    cuda_l1_iz_true,
        # CD-L2
        "numpy_l2_with_zero":     numpy_l2_iz_false,
        "numpy_l2_ignore_zero":   numpy_l2_iz_true,
        "cuda_l2_with_zero":      cuda_l2_iz_false,
        "cuda_l2_ignore_zero":    cuda_l2_iz_true,
        # Diagnostics
        "n_pred_points":          int(pred_np.shape[0]),
        "n_gt_points":            int(gt_np.shape[0]),
        "n_pred_zero_rows":       int((pred_np.sum(axis=1) == 0).sum()),
        "n_gt_zero_rows":         int((gt_np.sum(axis=1) == 0).sum()),
    }


@torch.no_grad()
def run_model_4way(model_spec: dict, loader, device: str, max_samples: int,
                   cuda_modules: dict) -> dict:
    """Forward each sample once + compute all 8 metrics on same prediction.

    Returns dict: per_sample list + aggregate means/stds.
    """
    name = model_spec["name"]
    cfg_path = model_spec["cfg"]
    ckpt_path = model_spec["ckpt"]
    if not Path(ckpt_path).is_file():
        return {"status": "skipped", "reason": f"ckpt missing at {ckpt_path}"}

    print(f"\n[chamfer-4way] Loading {name} from {ckpt_path}")
    model, stamped = load_model(
        cfg_path, ckpt_path, device, name,
        load_strict=model_spec.get("load_strict", False),
        builder_kind=model_spec.get("builder"),
        builder_kwargs=model_spec.get("builder_kwargs"),
    )

    output_index = model_spec.get("output_index", -1)
    concat_partial = model_spec.get("concat_partial", False)
    # concat_partial=True is not allowed for this script (raise, not assert:
    # assert is disabled under python -O).
    if concat_partial is not False:
        raise RuntimeError(
            f"{name}: concat_partial must be False "
            f"(PoinTr forward already internally concats partial). "
            f"External concat would double cardinality to 18432."
        )

    per_sample = []
    t_start = time.time()
    for i, batch in enumerate(loader):
        if max_samples > 0 and i >= max_samples:
            break
        taxonomy, model_id, payload = batch
        partial = payload[0].to(device, non_blocking=True)
        gt = payload[1]
        gt_np = gt.cpu().numpy()[0]
        # Forward
        ret = model(partial)
        pred_dense = ret[output_index] if isinstance(ret, (list, tuple)) else ret
        # No external concat (concat_partial=False)
        pred_np = pred_dense.cpu().numpy()[0]
        # shape gate (raise, not assert)
        if pred_np.shape[0] != 16384:
            raise RuntimeError(
                f"[{name}] sample {i}: n_pred={pred_np.shape[0]} != 16384. "
                f"concat_partial misconfig?"
            )
        # CUDA tensors (batch_size=1, contiguous)
        pred_t = pred_dense.to(device).float().contiguous()  # (1, N, 3)
        gt_t = gt.to(device).float().contiguous()             # (1, 16384, 3)
        if pred_t.shape[0] != 1 or gt_t.shape[0] != 1:
            raise RuntimeError(
                f"batch_size must be 1 (got pred={pred_t.shape[0]}, gt={gt_t.shape[0]}): "
                f"the CUDA chamfer ignore_zeros mask only applies when batch=1."
            )

        row = four_way_cd(pred_t, gt_t, pred_np, gt_np, cuda_modules)
        row["idx"] = i
        row["taxonomy_id"] = taxonomy[0] if isinstance(taxonomy, (list, tuple)) else taxonomy
        row["model_id"] = model_id[0] if isinstance(model_id, (list, tuple)) else model_id
        per_sample.append(row)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            mean_iz_true = np.mean([s["numpy_l1_ignore_zero"] for s in per_sample]) * 1000
            print(f"  [{name}] {i+1} samples, mean numpy CDL1 ignore_zero x1000 = "
                  f"{mean_iz_true:.3f}, elapsed {elapsed:.1f}s")

    # Aggregate
    keys_l1 = ["numpy_l1_with_zero", "numpy_l1_ignore_zero",
               "cuda_l1_with_zero",  "cuda_l1_ignore_zero"]
    keys_l2 = ["numpy_l2_with_zero", "numpy_l2_ignore_zero",
               "cuda_l2_with_zero",  "cuda_l2_ignore_zero"]
    agg = {"n_samples": len(per_sample)}
    for k in keys_l1 + keys_l2:
        vals = np.array([s[k] for s in per_sample])
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"]  = float(vals.std())
        agg[f"{k}_x1000_mean"] = float(vals.mean() * 1000)

    # Diagnostics aggregates
    agg["mean_n_pred_zero_rows"] = float(np.mean([s["n_pred_zero_rows"] for s in per_sample]))
    agg["max_n_pred_zero_rows"]  = int(np.max([s["n_pred_zero_rows"] for s in per_sample]))
    agg["mean_n_gt_zero_rows"]   = float(np.mean([s["n_gt_zero_rows"] for s in per_sample]))

    # Free model VRAM
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "status": "done",
        "per_sample": per_sample,
        "aggregate": agg,
        "stamped_best_metrics": {k: float(v) if not isinstance(v, str) else v
                                  for k, v in (stamped or {}).items()} if stamped else {},
    }


def evaluate_agreement(model_results: dict, threshold_x1000: float,
                       required_models: tuple = ("PoinTr", "AdaPoinTr")) -> dict:
    """Per-model: gate numpy<->CUDA agreement + AdaPoinTr zero-rows + PoinTr stamped diagnostic.

    - Required models must be status==done; skip = FAIL.
    - AdaPoinTr: max_n_pred_zero_rows == 0 AND |with_zero - ignore_zero| < threshold.
    - PoinTr: diagnostic check ignore_zero CDL1 x1000 ~ stamped (not blocking gate but flagged).
    """
    out = {"required_models": list(required_models)}
    per_model = {}
    for name, res in model_results.items():
        if res.get("status") != "done":
            per_model[name] = {
                "status": res.get("status", "skipped"),
                "reason": res.get("reason"),
                "l1_pass": False,
                "ada_zero_row_pass": None,
                "pointr_stamped_diag": None,
            }
            continue
        agg = res["aggregate"]
        deltas = {
            "l1_with_zero_delta_x1000":   abs(agg["numpy_l1_with_zero_x1000_mean"]   - agg["cuda_l1_with_zero_x1000_mean"]),
            "l1_ignore_zero_delta_x1000": abs(agg["numpy_l1_ignore_zero_x1000_mean"] - agg["cuda_l1_ignore_zero_x1000_mean"]),
            "l2_with_zero_delta_x1000":   abs(agg["numpy_l2_with_zero_x1000_mean"]   - agg["cuda_l2_with_zero_x1000_mean"]),
            "l2_ignore_zero_delta_x1000": abs(agg["numpy_l2_ignore_zero_x1000_mean"] - agg["cuda_l2_ignore_zero_x1000_mean"]),
        }
        l1_pass = (
            deltas["l1_with_zero_delta_x1000"] < threshold_x1000
            and deltas["l1_ignore_zero_delta_x1000"] < threshold_x1000
        )
        l2_loose = (
            deltas["l2_with_zero_delta_x1000"] < threshold_x1000 * 100
            and deltas["l2_ignore_zero_delta_x1000"] < threshold_x1000 * 100
        )
        # AdaPoinTr-specific: zero rows should be 0
        ada_zero_row_pass = None
        if name == "AdaPoinTr":
            ada_zero_row_pass = bool(
                agg["max_n_pred_zero_rows"] == 0
                and abs(agg["numpy_l1_with_zero_x1000_mean"] - agg["numpy_l1_ignore_zero_x1000_mean"]) < threshold_x1000
            )
        # PoinTr-specific: ignore_zero CDL1 should be close to the stamped 7.263
        pointr_stamped_diag = None
        if name == "PoinTr":
            stamped = res.get("stamped_best_metrics", {}) or {}
            stamped_cdl1 = stamped.get("CDL1")
            actual_cdl1_iz = agg["numpy_l1_ignore_zero_x1000_mean"]
            # Accept either stamped value or known PCN_new=7.26 / PCN orig=8.38 within +/-10%
            candidates = []
            if stamped_cdl1 is not None:
                candidates.append(("stamped", float(stamped_cdl1)))
            candidates.extend([("PCN_new", 7.26), ("PCN_orig", 8.38)])
            best = None
            for label, val in candidates:
                pct = abs(actual_cdl1_iz - val) / val * 100
                if best is None or pct < best[2]:
                    best = (label, val, pct)
            pointr_stamped_diag = {
                "actual_cdl1_iz_x1000": actual_cdl1_iz,
                "best_match": {"label": best[0], "value": best[1], "delta_pct": best[2]} if best else None,
                "within_10pct": bool(best and best[2] <= 10.0),
            }
        per_model[name] = {
            "status": "done",
            "l1_pass": l1_pass,
            "l2_loose_pass": l2_loose,
            "ada_zero_row_pass": ada_zero_row_pass,
            "pointr_stamped_diag": pointr_stamped_diag,
            "deltas_x1000": deltas,
            "threshold_x1000": threshold_x1000,
            "summary_l1": {
                "numpy_with_zero":   agg["numpy_l1_with_zero_x1000_mean"],
                "numpy_ignore_zero": agg["numpy_l1_ignore_zero_x1000_mean"],
                "cuda_with_zero":    agg["cuda_l1_with_zero_x1000_mean"],
                "cuda_ignore_zero":  agg["cuda_l1_ignore_zero_x1000_mean"],
            },
            "summary_l2": {
                "numpy_with_zero":   agg["numpy_l2_with_zero_x1000_mean"],
                "numpy_ignore_zero": agg["numpy_l2_ignore_zero_x1000_mean"],
                "cuda_with_zero":    agg["cuda_l2_with_zero_x1000_mean"],
                "cuda_ignore_zero":  agg["cuda_l2_ignore_zero_x1000_mean"],
            },
            "n_samples": agg["n_samples"],
            "mean_n_pred_zero_rows": agg["mean_n_pred_zero_rows"],
            "max_n_pred_zero_rows": agg["max_n_pred_zero_rows"],
        }
    out["per_model"] = per_model
    # Required-models gate
    out["required_models_done"] = all(per_model.get(m, {}).get("status") == "done"
                                       for m in required_models)
    return out


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.batch_size != 1:
        sys.exit("[chamfer-4way] FATAL: --batch-size must be 1 "
                 "(the CUDA chamfer ignore_zeros mask is only applied when batch=1).")
    # reject max_samples=0
    if args.max_samples == 0:
        sys.exit("[chamfer-4way] FATAL: --max-samples cannot be 0 (would produce empty aggregates).")
    # this script validates the CUDA extension; refuse a non-CUDA device
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        sys.exit(f"[chamfer-4way] FATAL: CUDA required (device={args.device}, "
                 f"cuda.is_available={torch.cuda.is_available()}). This script validates the CUDA extension.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[chamfer-4way] Loading PCN test split from {args.pcn_data_root}")
    loader = load_pcn_test_loader(args.pcn_data_root, args.batch_size)
    print(f"[chamfer-4way] PCN test loader ready: {len(loader)} batches")

    cuda_modules = make_cuda_modules(args.device)
    print(f"[chamfer-4way] CUDA chamfer modules ready (4 variants on {args.device})")

    model_results = {}
    for spec in DEFAULT_MODELS:
        print(f"\n=== {spec['name']} ===")
        res = run_model_4way(spec, loader, args.device, args.max_samples, cuda_modules)
        model_results[spec["name"]] = res

    # Evaluate agreement gates (requires both PoinTr + AdaPoinTr done)
    REQUIRED_MODELS = ("PoinTr", "AdaPoinTr")
    agreement = evaluate_agreement(model_results, args.agreement_threshold_x1000, REQUIRED_MODELS)
    per_model_agree = agreement["per_model"]
    required_done = agreement["required_models_done"]

    # Print summary
    print("\n" + "=" * 80)
    print("[chamfer-4way] SUMMARY - numpy vs CUDA chamfer agreement")
    print("=" * 80)
    overall_l1_pass = True
    overall_l2_loose_pass = True
    for name, res in per_model_agree.items():
        if res.get("status") != "done":
            print(f"  {name}: {res.get('status')}  reason={res.get('reason')}")
            overall_l1_pass = False
            overall_l2_loose_pass = False
            continue
        s = res["summary_l1"]
        d = res["deltas_x1000"]
        l1_str = "PASS" if res["l1_pass"] else "FAIL"
        l2_str = "PASS" if res["l2_loose_pass"] else "FAIL"
        print(f"  {name} CD-L1 x1000  np_wz={s['numpy_with_zero']:.4f}  np_iz={s['numpy_ignore_zero']:.4f}  "
              f"cu_wz={s['cuda_with_zero']:.4f}  cu_iz={s['cuda_ignore_zero']:.4f}")
        print(f"  {name} CD-L1 deltas wz={d['l1_with_zero_delta_x1000']:.6f}  "
              f"iz={d['l1_ignore_zero_delta_x1000']:.6f}  threshold={res['threshold_x1000']}  -> L1 {l1_str} | L2-loose {l2_str}")
        if name == "AdaPoinTr" and res.get("ada_zero_row_pass") is False:
            print(f"  AdaPoinTr ZERO-ROW FAIL: max_n_pred_zero_rows={res['max_n_pred_zero_rows']} (expected 0)")
            overall_l1_pass = False
        if name == "PoinTr" and res.get("pointr_stamped_diag"):
            diag = res["pointr_stamped_diag"]
            within = diag.get("within_10pct")
            best = diag.get("best_match")
            print(f"  PoinTr stamped diag: actual_iz={diag['actual_cdl1_iz_x1000']:.4f}  "
                  f"best={best['label']}={best['value']:.3f} delta={best['delta_pct']:.2f}%  within_10pct={within}")
            if not within:
                print(f"  [WARN] PoinTr CDL1 ignore_zero {diag['actual_cdl1_iz_x1000']:.4f} not within 10% of any expected stamped value")
                # not gating overall_pass on this (diagnostic only)
        if not res["l1_pass"]:
            overall_l1_pass = False
        if not res["l2_loose_pass"]:
            overall_l2_loose_pass = False

    if not required_done:
        print(f"\n  [WARN] REQUIRED MODELS not all done: {REQUIRED_MODELS} - overall FAIL")
        overall_l1_pass = False

    summary_path = out_dir / "chamfer_fourway_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "schema_version": "1.0",
            "run_id": "chamfer_fourway",
            "args": vars(args),
            "required_models": list(REQUIRED_MODELS),
            "required_models_done": required_done,
            "agreement": agreement,
            "model_aggregates": {k: (v.get("aggregate") if v.get("status") == "done" else None)
                                  for k, v in model_results.items()},
            "stamped": {k: v.get("stamped_best_metrics", {}) for k, v in model_results.items()},
            "overall_l1_pass": overall_l1_pass,
            "overall_l2_loose_pass": overall_l2_loose_pass,
        }, f, indent=2)
    print(f"\n[chamfer-4way] Summary written to {summary_path}")

    if args.write_per_sample:
        for name, res in model_results.items():
            if res.get("status") == "done":
                p = out_dir / f"{name}_per_sample_4way.json"
                with open(p, "w") as f:
                    json.dump(res["per_sample"], f, indent=2)
                print(f"[chamfer-4way] Per-sample written to {p}")

    # nonzero exit if the agreement gate fails OR a required model is missing
    if not overall_l1_pass:
        print("[chamfer-4way] EXIT 1 - overall agreement gate FAILED (CD-L1 numpy<->CUDA OR required models not all done OR AdaPoinTr zero-row violated).")
        sys.exit(1)


if __name__ == "__main__":
    main()

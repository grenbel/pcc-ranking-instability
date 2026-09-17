"""Single-stage corruption sweep (forward + metrics in one process) and the shared
clean-cache builder used by the two-stage pipeline.

`cache_clean_pcn_test` is the cache builder of the paper's pipeline: it snapshots the
clean PCN test partials + ground truths once under a fixed seed so that every
(model, operator, severity) cell corrupts bit-identical inputs (matched-control
invariant). The two-stage scripts (forward_sweep*.py + metric_emitter.py) import it;
this script's own main() runs the legacy single-stage sweep, which is convenient
for small smoke runs on one machine.

Scope of the single-stage sweep:
    - 4 corruption operators (noise/outlier/density/crop); pose is registered in
      src/corruptions but is not part of the audit grid
    - 5 severities (1-5; severity 0 = clean)
    - baselines from DEFAULT_MODELS (default PoinTr + AdaPoinTr)
    - --max-samples-per-cell (-1 = the full PCN test split, 1200 samples)

Per-sample output JSON includes:
    - taxonomy_id, model_id, op, severity, CD-L1, CD-L2, F@{0.001,0.005,0.01}
    - 4 decomposition metrics (hf_loss, topology, density_collapse, boundary_erosion);
      these are descriptive extras and are not used by the paper's ranking statistics

Reuses sanity_clean_pcn.py: DEFAULT_MODELS / cwd_to / _torch_load / load_pcn_test_loader / load_model.

Usage:
    python scripts/sweep_common.py \\
        --pcn-data-root /path/to/PCN \\
        --output-dir logs/sweep \\
        --device cuda:0 \\
        --max-samples-per-cell 100 \\
        --models PoinTr,AdaPoinTr \\
        --ops noise,outlier,density,crop \\
        --severities 1,2,3,4,5 \\
        --no-wandb
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
sys.path.insert(0, str(REPO_ROOT))

# Reuse from sanity_clean_pcn.py (avoid duplicating model-load/cfg-patch logic)
from scripts.sanity_clean_pcn import (  # type: ignore
    DEFAULT_MODELS, SANITY_TOLERANCE_PCT, cwd_to, _torch_load,
    load_pcn_test_loader, load_model,
)
from src.corruptions import OP_REGISTRY, make_seed
from src.metrics.reconstruction import chamfer_l1, chamfer_l2, fscore_at_thresholds
from src.metrics import DECOMPOSITION_METRICS

# Default sweep config (overridable via CLI). The pose operator is excluded from the
# audit grid: an input-frame evaluation would rotate partial+GT together while a
# canonicalization evaluation rotates the partial only, and that design choice is
# out of scope here.
DEFAULT_OPS = ("noise", "outlier", "density", "crop")
DEFAULT_SEVERITIES = (1, 2, 3, 4, 5)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcn-data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples-per-cell", type=int, default=-1,
                   help="-1 = full PCN test (1200) per cell (default); "
                        "explicit small int for smoke (e.g. --max-samples-per-cell 30)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--models", default="PoinTr,AdaPoinTr",
                   help="comma-separated model names from DEFAULT_MODELS")
    p.add_argument("--ops", default=",".join(DEFAULT_OPS),
                   help=f"comma-separated op names from {list(OP_REGISTRY.keys())}")
    p.add_argument("--severities", default=",".join(str(s) for s in DEFAULT_SEVERITIES),
                   help="comma-separated severities (1-5)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="pcc-ranking-instability")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-skip-missing-ckpt", action="store_true",
                   help="Permit silent SKIP for selected models with missing ckpts. "
                        "Default off: silently producing incomplete ranking-instability "
                        "data is worse than failing fast.")
    return p.parse_args()


def cache_clean_pcn_test(loader, max_samples: int, seed: int):
    """Pre-cache clean PCN test partials + GT once with fixed seed.

    PoinTr's PCNDataset test transform calls `RandomSamplePoints` (numpy global
    RNG). Re-iterating the loader per cell would give different partial states for
    the same sample idx depending on sweep/model order, breaking the matched-control
    invariant ("same object/view/severity/op -> bit-identical corruption"). The loader
    output is therefore snapshotted once under `np.random.seed(seed)` +
    `torch.manual_seed(seed)` and reused across all cells and models.

    Returns a list of dicts (1200 samples for the full PCN test split, or capped to
    max_samples). Memory: 1200 x (2048 + 16384) x 3 x 4 bytes = ~265 MB float32.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    cache = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        if max_samples > 0 and i >= max_samples:
            break
        taxonomy, model_id, payload = batch
        tax_str = taxonomy[0] if isinstance(taxonomy, (list, tuple)) else taxonomy
        mid_str = model_id[0] if isinstance(model_id, (list, tuple)) else model_id
        cache.append({
            "idx": i,
            "taxonomy_id": tax_str,
            "model_id": mid_str,
            "view_id": 0,                                      # PCN test: 1 view per model
            "partial_np": payload[0].cpu().numpy()[0].copy(),  # (2048, 3) float32 snapshot
            "gt_np": payload[1].cpu().numpy()[0].copy(),       # (16384, 3) float32 snapshot
        })
    print(f"[sweep] cached {len(cache)} clean PCN test samples in {time.time()-t0:.1f}s "
          f"(deterministic, seed={seed})")
    return cache


@torch.no_grad()
def run_corruption_cell(model, clean_cache, device, op, severity, model_spec,
                        decomp_metrics, model_name, op_name):
    """Inference over deterministic clean cache with (op, severity) corruption on partial.

    GT remains clean (input-only corruption - eval completion robustness, not GT-corrupted).
    Returns list of per-sample dicts.
    """
    out_idx = model_spec.get("output_index", -1)
    concat_partial = model_spec.get("concat_partial", False)
    cell_key = f"{op_name}@s{severity}"
    per_sample = []
    t_start = time.time()
    for i, entry in enumerate(clean_cache):
        partial_np = entry["partial_np"]
        gt_np = entry["gt_np"]
        tax_str, mid_str, view_id = entry["taxonomy_id"], entry["model_id"], entry["view_id"]
        # Apply corruption (deterministic per (tax, mid, view, sev, op) via make_seed)
        corrupted_np = op(partial_np, severity, tax_str, mid_str, view_id)  # (2048, 3) float32
        corrupted_t = torch.from_numpy(corrupted_np).unsqueeze(0).to(device, non_blocking=True)
        # Forward
        ret = model(corrupted_t)
        pred_dense = ret[out_idx] if isinstance(ret, (list, tuple)) else ret
        # optional external concat of the partial (see DEFAULT_MODELS)
        if concat_partial:
            pred_dense = torch.cat([corrupted_t, pred_dense], dim=1)
        pred_np = pred_dense.cpu().numpy()[0]  # (N_pred, 3)
        # Standard reconstruction metrics
        cd1 = chamfer_l1(pred_np, gt_np)
        cd2 = chamfer_l2(pred_np, gt_np)
        fscore = fscore_at_thresholds(pred_np, gt_np)
        # decomposition metrics (descriptive extras)
        decomp = {f"decomp_{dname}": dm(pred_np, gt_np)
                  for dname, dm in decomp_metrics.items()}
        # Self-describing per-sample row
        corruption_seed = int(make_seed(tax_str, mid_str, view_id, severity, op_name))
        per_sample.append({
            "idx": entry["idx"],
            "model_name": model_name,
            "taxonomy_id": tax_str,
            "model_id": mid_str,
            "view_id": view_id,
            "op": op_name,
            "severity": severity,
            "cell_key": cell_key,
            "corruption_seed": corruption_seed,
            "n_pred_points": int(pred_np.shape[0]),
            "n_gt_points": int(gt_np.shape[0]),
            "cd_l1": cd1,
            "cd_l2": cd2,
            **fscore,
            **decomp,
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            mean_cd1 = np.mean([s["cd_l1"] for s in per_sample])
            print(f"    [{model_name}|{cell_key}] {i+1}, "
                  f"mean CDL1={mean_cd1*1000:.3f} (x1000), elapsed {elapsed:.1f}s")
    return per_sample


def aggregate_cell(per_sample):
    """Aggregate per-sample -> cell mean/std for primary + decomposition metrics."""
    if not per_sample:
        return {"n_samples": 0}
    keys = ["cd_l1", "cd_l2", "f_at_0.001", "f_at_0.005", "f_at_0.01",
            "decomp_hf_loss", "decomp_topology", "decomp_density_collapse",
            "decomp_boundary_erosion"]
    agg = {"n_samples": len(per_sample)}
    for k in keys:
        vals = np.array([s[k] for s in per_sample if not np.isnan(s.get(k, np.nan))])
        if vals.size == 0:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_std"] = float("nan")
        else:
            agg[f"{k}_mean"] = float(vals.mean())
            agg[f"{k}_std"] = float(vals.std())
    agg["cd_l1_x1000"] = agg["cd_l1_mean"] * 1000
    agg["cd_l2_x1000"] = agg["cd_l2_mean"] * 1000
    return agg


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse model/op/severity selections
    sel_models = [m.strip() for m in args.models.split(",") if m.strip()]
    sel_ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    sel_sevs = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
    if "pose" in sel_ops:
        sys.exit("[sweep] 'pose' is not part of the audit grid "
                 "(input-frame vs canonicalization evaluation is unresolved).")
    for o in sel_ops:
        if o not in DEFAULT_OPS:
            sys.exit(f"[sweep] op '{o}' not in the sweep scope {list(DEFAULT_OPS)}")
    for s in sel_sevs:
        if s not in (1, 2, 3, 4, 5):
            sys.exit(f"[sweep] severity {s} out of [1,5]")
    print(f"[sweep] sweep config: models={sel_models}, ops={sel_ops}, "
          f"severities={sel_sevs}, max_per_cell={args.max_samples_per_cell}")
    print(f"[sweep] total cells: {len(sel_models) * len(sel_ops) * len(sel_sevs)}")

    # Pre-flight ckpt existence check (fail fast rather than silently skip)
    name_to_spec = {ms["name"]: ms for ms in DEFAULT_MODELS}
    missing = []
    for m in sel_models:
        if m not in name_to_spec:
            sys.exit(f"[sweep] unknown model '{m}', available: {list(name_to_spec.keys())}")
        if not Path(name_to_spec[m]["ckpt"]).is_file():
            missing.append((m, name_to_spec[m]["ckpt"]))
    if missing:
        msg = "[sweep] FATAL - selected models with missing ckpts:\n" + "\n".join(
            f"    {m}: {p}" for m, p in missing)
        if args.allow_skip_missing_ckpt:
            print(msg + "\n  --allow-skip-missing-ckpt set -> continuing with skip "
                       "(downstream ranking-instability data WILL BE INCOMPLETE)")
        else:
            sys.exit(msg + "\n  Either download the ckpt(s) or pass --allow-skip-missing-ckpt "
                          "(latter will produce incomplete data).")

    # wandb
    wandb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            run_name = args.wandb_run_name or f"pcn_corruption_sweep_{int(time.time())}"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        except Exception as e:
            print(f"[WARN] wandb init failed (continuing without): {e}")

    # Load the PCN test loader once + pre-cache the clean partials
    print(f"[sweep] Loading PCN test split from {args.pcn_data_root}")
    loader = load_pcn_test_loader(args.pcn_data_root, args.batch_size)
    print(f"[sweep] PCN test loader ready: {len(loader)} batches")
    clean_cache = cache_clean_pcn_test(loader, args.max_samples_per_cell, args.seed)

    # Instantiate corruption ops + decomposition metrics once
    ops_dict = {n: OP_REGISTRY[n]() for n in sel_ops}
    decomp_dict = {n: cls() for n, cls in DECOMPOSITION_METRICS.items()}

    # Cells loop
    all_results = {}
    grand_t0 = time.time()
    for model_name in sel_models:
        spec = name_to_spec[model_name]
        ckpt = spec["ckpt"]
        if not Path(ckpt).is_file():
            print(f"[sweep] {model_name} ckpt missing at {ckpt} -> SKIPPED")
            all_results[model_name] = {"status": "skipped", "reason": "ckpt missing"}
            continue
        print(f"\n[sweep] === {model_name} ===")
        model, _ = load_model(
            spec["cfg"], ckpt, args.device, model_name,
            load_strict=spec.get("load_strict", False),
            builder_kind=spec.get("builder"),
            builder_kwargs=spec.get("builder_kwargs"),
        )
        model_results = {}
        for op_name in sel_ops:
            op = ops_dict[op_name]
            for sev in sel_sevs:
                cell_key = f"{op_name}@s{sev}"
                print(f"  [{model_name}] cell {cell_key}")
                per_sample = run_corruption_cell(
                    model, clean_cache, args.device, op, sev, spec,
                    decomp_dict, model_name, op_name)
                agg = aggregate_cell(per_sample)
                # Save per-sample JSON
                cell_path = out_dir / f"{model_name}_{op_name}_s{sev}_per_sample.json"
                # convert NaN -> None for strict-JSON downstream
                clean_per_sample = [{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                      for k, v in row.items()} for row in per_sample]
                with open(cell_path, "w") as f:
                    json.dump(clean_per_sample, f, indent=2, allow_nan=False)
                model_results[cell_key] = {"aggregate": agg, "per_sample_path": str(cell_path)}
                print(f"  [{model_name}|{cell_key}] DONE: n={agg['n_samples']}, "
                      f"CDL1x1000={agg.get('cd_l1_x1000', float('nan')):.3f}, "
                      f"F@0.01={agg.get('f_at_0.01_mean', float('nan')):.3f}")
                if wandb is not None:
                    wandb.log({f"{model_name}/{cell_key}/{k}": v for k, v in agg.items()
                               if isinstance(v, (int, float))})
        all_results[model_name] = {"status": "done", "cells": model_results}
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Final summary
    summary_path = out_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_id": "single_stage_sweep",
            "args": vars(args),
            "elapsed_total_sec": time.time() - grand_t0,
            "results": all_results,
        }, f, indent=2)
    print(f"\n[sweep] Total elapsed: {(time.time()-grand_t0)/60:.1f} min")
    print(f"[sweep] Summary written to {summary_path}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

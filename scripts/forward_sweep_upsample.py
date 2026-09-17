"""UpSamplePoints protocol variant of the forward stage (GPU).

Reruns the 4-baseline corruption sweep + a clean pass-through cell on the PCN test
split (1200 samples) with the partial brought to 2048 rows by the UpSamplePoints
transform (duplicate-upsample, the native SnowflakeNet/SeedFormer training
transform) instead of RandomSamplePoints (permute-subsample + zero-pad).

Deltas vs scripts/forward_sweep_validpt.py (the mirrored precedent):
    - cache built via loader_cfg = SnowFlakeNet.yaml (PCNv2 UpSamplePoints path),
      NOT the default PoinTr.yaml zero-pad path
    - ORIGINAL op registry names (noise/outlier/density/crop); the protocol is
      disambiguated by the pred dir + per-NPZ/manifest stamps, not by op renaming
    - explicit clean@s0 pass-through cell (op="clean", severity=0)
    - hard fail if the new cache hash equals the zero-pad invariant
    - per-sample cache stats (unique-row count, coordinate-sum-zero count)
      recorded in the manifest (post-loader duplicate-ratio disclosure)

Output filenames: {pred_dir}/{model}__{op}__s{sev}.npz  (e.g. PoinTr__clean__s0.npz)

Usage:
    python scripts/forward_sweep_upsample.py \\
        --pcn-data-root /path/to/PCN \\
        --pred-dir preds/upsample \\
        --device cuda:0 --max-samples-per-cell -1
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

from scripts.sanity_clean_pcn import (  # type: ignore
    DEFAULT_MODELS, load_pcn_test_loader, load_model,
)
from scripts.sweep_common import cache_clean_pcn_test  # type: ignore
from scripts.forward_sweep import (  # type: ignore
    NPZ_SCHEMA_VERSION,
    atomic_savez,
    compute_cache_hash,
    forward_cell,
)
from src.corruptions import OP_REGISTRY

DEFAULT_OPS_UPSAMPLE = ("noise", "outlier", "density", "crop")
DEFAULT_SEVERITIES = (1, 2, 3, 4, 5)
DEFAULT_BASELINES = ("PoinTr", "AdaPoinTr", "SnowFlakeNet", "SeedFormer")
PROTOCOL_LABEL = "upsample_points_v1"
CLEAN_OP_NAME = "clean"
CLEAN_SEVERITY = 0
# Zero-pad protocol invariant (zero-pad and validpt manifests); the UpSamplePoints
# cache MUST differ from it, else the loader switch silently did not happen.
ZERO_PAD_CACHE_HASH_PREFIX = "80d3efce468eba6c"
DEFAULT_LOADER_CFG = str(POINTR_ROOT / "cfgs" / "PCN_models" / "SnowFlakeNet.yaml")


class CleanOp:
    """Identity pass-through for the clean@s0 sanity/reference cell.

    Mirrors the CorruptionOp calling convention used by forward_cell:
    op(points, severity, tax, mid, view) -> np.ndarray. Returns an unmodified
    copy so the model consumes exactly the cached UpSamplePoints partial.
    """
    name = CLEAN_OP_NAME
    preserves_count = True

    def __call__(self, points, severity, tax, mid, view):
        if severity != CLEAN_SEVERITY:
            raise ValueError(f"clean op only accepts severity {CLEAN_SEVERITY}, "
                             f"got {severity}")
        return np.asarray(points, dtype=np.float32).copy()


def cache_row_stats(cache):
    """Per-sample post-loader stats for the manifest: unique-row count
    (duplicate_ratio derives as
    (2048 - unique_rows)/2048), coordinate-sum-zero row count (the chamfer mask
    criterion, audited rather than assumed ~0 under UpSamplePoints), and
    exact-(0,0,0) row count."""
    uniq, sumzero, exactzero = [], [], []
    for e in cache:
        p = np.asarray(e["partial_np"], dtype=np.float32)
        uniq.append(int(np.unique(p, axis=0).shape[0]))
        sumzero.append(int((p.sum(axis=-1) == 0).sum()))
        exactzero.append(int(np.all(p == 0.0, axis=-1).sum()))
    return uniq, sumzero, exactzero


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcn-data-root", required=True)
    p.add_argument("--pred-dir", required=True,
                   help="Output dir for per-cell .npz (e.g. preds/upsample)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples-per-cell", type=int, default=-1,
                   help="-1 = full PCN test (1200) per cell; smaller for smoke")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--models", default=",".join(DEFAULT_BASELINES))
    p.add_argument("--ops", default=",".join(DEFAULT_OPS_UPSAMPLE),
                   help=f"corruption ops from {list(DEFAULT_OPS_UPSAMPLE)} (clean cell "
                        f"is controlled separately by --include-clean)")
    p.add_argument("--severities", default=",".join(str(s) for s in DEFAULT_SEVERITIES))
    p.add_argument("--include-clean", action="store_true", default=True,
                   help="Also forward the clean@s0 pass-through cell per model (default on)")
    p.add_argument("--no-include-clean", dest="include_clean", action="store_false")
    p.add_argument("--loader-cfg", default=DEFAULT_LOADER_CFG,
                   help="Loader cfg yaml selecting the partial transform. Default "
                        "SnowFlakeNet.yaml = PCNv2 UpSamplePoints.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-skip-missing-ckpt", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size != 1:
        sys.exit(f"[upsample-forward] --batch-size must be 1 (got {args.batch_size}); "
                 f"cache_clean_pcn_test only retains payload[0] per batch")
    if not Path(args.loader_cfg).is_file():
        sys.exit(f"[upsample-forward] FATAL: --loader-cfg not found: {args.loader_cfg}")

    sel_models = [m.strip() for m in args.models.split(",") if m.strip()]
    sel_ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    sel_sevs = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
    for label, lst in (("models", sel_models), ("ops", sel_ops),
                       ("severities", sel_sevs)):
        if len(set(lst)) != len(lst):
            sys.exit(f"[upsample-forward] duplicate {label} in --{label}: {lst}")
    for o in sel_ops:
        if o not in DEFAULT_OPS_UPSAMPLE:
            sys.exit(f"[upsample-forward] op '{o}' not in scope {list(DEFAULT_OPS_UPSAMPLE)}; "
                     f"validpt ops belong to forward_sweep_validpt.py, clean is "
                     f"--include-clean")
        if o not in OP_REGISTRY:
            sys.exit(f"[upsample-forward] op '{o}' missing from OP_REGISTRY")
    for s in sel_sevs:
        if s not in (1, 2, 3, 4, 5):
            sys.exit(f"[upsample-forward] severity {s} out of [1,5]")

    name_to_spec = {ms["name"]: ms for ms in DEFAULT_MODELS}
    missing = []
    for m in sel_models:
        if m not in name_to_spec:
            sys.exit(f"[upsample-forward] unknown model '{m}'; "
                     f"available: {sorted(name_to_spec.keys())}")
        if not Path(name_to_spec[m]["ckpt"]).is_file():
            missing.append((m, name_to_spec[m]["ckpt"]))
    if missing:
        msg = "[upsample-forward] FATAL - selected models with missing ckpts:\n" + "\n".join(
            f"    {m}: {p}" for m, p in missing)
        if args.allow_skip_missing_ckpt:
            print(msg + "\n  --allow-skip-missing-ckpt -> continuing")
        else:
            sys.exit(msg)

    print(f"[upsample-forward] Loading PCN test split from {args.pcn_data_root} "
          f"with loader_cfg={args.loader_cfg}")
    loader = load_pcn_test_loader(args.pcn_data_root, args.batch_size,
                                  cfg_yaml=args.loader_cfg)
    print(f"[upsample-forward] PCN test loader ready: {len(loader)} batches")
    clean_cache = cache_clean_pcn_test(loader, args.max_samples_per_cell, args.seed)
    cache_hash = compute_cache_hash(clean_cache)
    print(f"[upsample-forward] cache_hash (sha256): {cache_hash[:16]}... "
          f"({len(clean_cache)} samples)")
    # Smoke assertion (i) baked in as hard gate: the UpSamplePoints cache must
    # NOT reproduce the zero-pad invariant, else the loader switch didn't happen.
    if cache_hash.startswith(ZERO_PAD_CACHE_HASH_PREFIX):
        sys.exit(f"[upsample-forward] FATAL: cache_hash equals the ZERO-PAD protocol invariant "
                 f"{ZERO_PAD_CACHE_HASH_PREFIX}... - loader_cfg did not switch the "
                 f"transform. Refusing to forward.")

    uniq_counts, sumzero_counts, exactzero_counts = cache_row_stats(clean_cache)
    print(f"[upsample-forward] cache row stats: unique-rows mean={np.mean(uniq_counts):.1f} "
          f"min={min(uniq_counts)} max={max(uniq_counts)}; "
          f"sum-zero rows total={sum(sumzero_counts)}; "
          f"exact-zero rows total={sum(exactzero_counts)}")

    ops_dict = {n: OP_REGISTRY[n]() for n in sel_ops}
    cell_plan = [(o, s) for o in sel_ops for s in sel_sevs]
    if args.include_clean:
        cell_plan.append((CLEAN_OP_NAME, CLEAN_SEVERITY))
        ops_dict[CLEAN_OP_NAME] = CleanOp()
    n_cells = len(sel_models) * len(cell_plan)
    print(f"[upsample-forward] {n_cells} cells to forward, {len(clean_cache)} samples each "
          f"= {n_cells * len(clean_cache)} forwards total")

    manifest = {
        "stage": "upsample-forward-pipeline",
        "schema_version": NPZ_SCHEMA_VERSION,
        "args": vars(args),
        "n_cached_samples": len(clean_cache),
        "cache_hash_sha256": cache_hash,
        "cached_sample_indices": [e["idx"] for e in clean_cache],
        "cached_taxonomy_ids": [e["taxonomy_id"] for e in clean_cache],
        "cached_model_ids": [e["model_id"] for e in clean_cache],
        "cached_view_ids": [e["view_id"] for e in clean_cache],
        "models": sel_models,
        # ops/severities describe the FULL cell plan incl. the clean@s0 cell, so the
        # metric stage's exact-match alignment covers it. Valid (op, sev) pairs are
        # NOT the full cartesian product: clean pairs only with 0, corruption ops
        # only with 1-5 (metric stage filters accordingly under --allow-clean-cell).
        "ops": sel_ops + ([CLEAN_OP_NAME] if args.include_clean else []),
        "severities": sel_sevs + ([CLEAN_SEVERITY] if args.include_clean else []),
        "include_clean": bool(args.include_clean),
        "device": args.device,
        "loader_cfg": args.loader_cfg,
        "protocol": PROTOCOL_LABEL,
        "protocol_description": (
            "UpSamplePoints loader transform (duplicate-upsample to 2048, native "
            "SnowflakeNet/SeedFormer training transform) replacing RandomSamplePoints "
            "zero-pad; original corruption op names + SHA-1 per-sample seed keys; "
            "corruption applied AFTER upsample (post-loader-transform sensitivity axis); "
            "clean@s0 pass-through cell included for same-protocol rank reference"
        ),
        "cache_stats_unique_row_counts": uniq_counts,
        "cache_stats_sumzero_row_counts": sumzero_counts,
        "cache_stats_exact_zero_row_counts": exactzero_counts,
        "cache_stats_note": ("per-sample, post-loader (2048 rows); duplicate_ratio "
                             "derives as (2048 - unique_rows)/2048"),
    }
    manifest_path = pred_dir / "upsample_forward_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[upsample-forward] manifest written: {manifest_path}")

    grand_t0 = time.time()
    for model_name in sel_models:
        spec = name_to_spec[model_name]
        ckpt = spec["ckpt"]
        if not Path(ckpt).is_file():
            print(f"[upsample-forward] {model_name} ckpt missing -> SKIPPED")
            continue
        print(f"\n[upsample-forward] === {model_name} ===")
        model, _ = load_model(
            spec["cfg"], ckpt, args.device, model_name,
            load_strict=spec.get("load_strict", False),
            builder_kind=spec.get("builder"),
            builder_kwargs=spec.get("builder_kwargs"),
        )
        for op_name, sev in cell_plan:
            op = ops_dict[op_name]
            if op.name != op_name:
                sys.exit(f"[upsample-forward] FATAL: op instance name '{op.name}' != selected "
                         f"'{op_name}'; seed integrity violated, refusing to forward")
            cell_key = f"{op_name}@s{sev}"
            cell_path = pred_dir / f"{model_name}__{op_name}__s{sev}.npz"
            if cell_path.exists():
                try:
                    with np.load(cell_path, allow_pickle=False) as z:
                        ok = (
                            str(z.get("schema_version", "")) == NPZ_SCHEMA_VERSION
                            and str(z["model_name"]) == model_name
                            and str(z["op"]) == op_name
                            and int(z["severity"]) == sev
                            and int(z["n_pred_points"]) == 16384
                            and int(z["preds"].shape[0]) == len(clean_cache)
                            and bool(z["concat_partial_applied"]) == bool(
                                spec.get("concat_partial", False))
                            and str(z.get("cache_hash_sha256", "")) == cache_hash
                            and str(z.get("protocol", "")) == PROTOCOL_LABEL
                        )
                except Exception as e:
                    ok = False
                    print(f"  [{model_name}|{cell_key}] existing file unreadable "
                          f"({e}) -> re-forward")
                if ok:
                    print(f"  [{model_name}|{cell_key}] skip (verified existing)")
                    continue
                print(f"  [{model_name}|{cell_key}] existing file metadata/protocol/"
                      f"cache_hash mismatch -> re-forward")
            print(f"  [{model_name}|{cell_key}] forward...")
            cell_data = forward_cell(model, clean_cache, args.device, op, sev,
                                      spec, model_name, op_name)
            cell_data["cache_hash_sha256"] = cache_hash
            cell_data["protocol"] = PROTOCOL_LABEL
            atomic_savez(cell_path, **cell_data)
            size_mb = cell_path.stat().st_size / 1e6
            print(f"  [{model_name}|{cell_key}] saved {cell_path.name} "
                  f"({size_mb:.1f} MB, {cell_data['preds'].shape})")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    elapsed = time.time() - grand_t0
    print(f"\n[upsample-forward] Total forward elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[upsample-forward] DONE. Metric stage (UpSamplePoints-aware):")
    print(f"  python scripts/metric_emitter.py "
          f"--pcn-data-root {args.pcn_data_root} "
          f"--pred-dir {pred_dir} "
          f"--output-dir <metric_out_dir> "
          f"--max-samples-per-cell {args.max_samples_per_cell} "
          f"--models {','.join(sel_models)} "
          f"--ops {','.join(sel_ops)}{',clean' if args.include_clean else ''} "
          f"--severities {','.join(str(s) for s in sel_sevs)}"
          f"{',0' if args.include_clean else ''} "
          f"--seed {args.seed} "
          f"--loader-cfg {args.loader_cfg} "
          f"--manifest-name upsample_forward_manifest.json "
          f"--expect-protocol {PROTOCOL_LABEL}"
          f"{' --allow-clean-cell' if args.include_clean else ''}")


if __name__ == "__main__":
    main()

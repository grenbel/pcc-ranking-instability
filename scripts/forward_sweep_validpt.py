"""Valid-point-only protocol variant of the forward stage (GPU).

Reruns the 4-baseline corruption sweep on the PCN test split (1200 samples) with
the valid-point-only corruption protocol: the zero-pad rows of the 2048-point PCN
partial input are preserved as exact (0,0,0) through corruption, instead of being
treated as valid geometry.

Reuses scripts/forward_sweep.py:
    - forward_cell (per-sample forward + concat-partial logic)
    - atomic_savez (write to .tmp + atomic rename)
    - compute_cache_hash (SHA-256 over the clean cache for the matched-control invariant)
    - NPZ_SCHEMA_VERSION (compat with metric_emitter.py)
Reuses scripts/sweep_common.py: cache_clean_pcn_test
Reuses scripts/sanity_clean_pcn.py: DEFAULT_MODELS (4 baselines), load_pcn_test_loader, load_model

Override:
    DEFAULT_OPS_VALIDPT = noise_validpt / outlier_validpt / density_validpt / crop_validpt
    Default --models = all 4 baselines (PoinTr, AdaPoinTr, SnowFlakeNet, SeedFormer)

Output filenames:
    {pred_dir}/{model_name}__{op_name}__s{sev}.npz
    e.g. PoinTr__noise_validpt__s3.npz

Cache hash invariant:
    Same as the zero-pad runs because the clean-cache hash covers the PCN clean
    partial+GT bytes (pre-corruption). The matched-control invariant is preserved
    across protocol variants (zero-pad and valid-point-only see bit-identical
    clean inputs).

Usage:
    python scripts/forward_sweep_validpt.py \\
        --pcn-data-root /path/to/PCN \\
        --pred-dir preds/validpt \\
        --device cuda:0 \\
        --max-samples-per-cell -1
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

DEFAULT_OPS_VALIDPT = (
    "noise_validpt", "outlier_validpt", "density_validpt", "crop_validpt"
)
DEFAULT_SEVERITIES = (1, 2, 3, 4, 5)
DEFAULT_BASELINES = ("PoinTr", "AdaPoinTr", "SnowFlakeNet", "SeedFormer")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcn-data-root", required=True)
    p.add_argument("--pred-dir", required=True,
                   help="Output dir for per-cell .npz files "
                        "(e.g. preds/validpt)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples-per-cell", type=int, default=-1,
                   help="-1 = full PCN test (1200) per cell; smaller for smoke")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--models", default=",".join(DEFAULT_BASELINES),
                   help="comma-separated model names from "
                        "{PoinTr, AdaPoinTr, SnowFlakeNet, SeedFormer}")
    p.add_argument("--ops", default=",".join(DEFAULT_OPS_VALIDPT),
                   help="comma-separated valid-point-only op names "
                        f"from {list(DEFAULT_OPS_VALIDPT)}")
    p.add_argument("--severities", default=",".join(str(s) for s in DEFAULT_SEVERITIES),
                   help="comma-separated severities (1-5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-skip-missing-ckpt", action="store_true")
    return p.parse_args()


PROTOCOL_LABEL = "valid_point_only_v1"


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # cache_clean_pcn_test only keeps payload[0] per batch, so any
    # --batch-size > 1 would silently drop samples; batch_size is pinned to 1.
    if args.batch_size != 1:
        sys.exit(f"[validpt-forward] --batch-size must be 1 (got {args.batch_size}); "
                 f"cache_clean_pcn_test only retains payload[0] per batch")

    # Validate selections strictly within validpt scope
    sel_models = [m.strip() for m in args.models.split(",") if m.strip()]
    sel_ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    sel_sevs = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
    # reject duplicate selections (last writer wins otherwise)
    for label, lst in (("models", sel_models), ("ops", sel_ops),
                       ("severities", sel_sevs)):
        if len(set(lst)) != len(lst):
            sys.exit(f"[validpt-forward] duplicate {label} in --{label}: {lst}")
    for o in sel_ops:
        if o not in DEFAULT_OPS_VALIDPT:
            sys.exit(f"[validpt-forward] op '{o}' not in valid-point-only scope "
                     f"{list(DEFAULT_OPS_VALIDPT)}; "
                     f"this script is dedicated to validpt protocol")
        if o not in OP_REGISTRY:
            sys.exit(f"[validpt-forward] op '{o}' missing from OP_REGISTRY")
    for s in sel_sevs:
        if s not in (1, 2, 3, 4, 5):
            sys.exit(f"[validpt-forward] severity {s} out of [1,5]")

    name_to_spec = {ms["name"]: ms for ms in DEFAULT_MODELS}
    missing = []
    for m in sel_models:
        if m not in name_to_spec:
            sys.exit(f"[validpt-forward] unknown model '{m}'; "
                     f"available: {sorted(name_to_spec.keys())}")
        ckpt_path = Path(name_to_spec[m]["ckpt"])
        if not ckpt_path.is_file():
            missing.append((m, str(ckpt_path)))
    if missing:
        msg = "[validpt-forward] FATAL - selected models with missing ckpts:\n" + "\n".join(
            f"    {m}: {p}" for m, p in missing)
        if args.allow_skip_missing_ckpt:
            print(msg + "\n  --allow-skip-missing-ckpt -> continuing")
        else:
            sys.exit(msg)

    print(f"[validpt-forward] Loading PCN test split from {args.pcn_data_root}")
    loader = load_pcn_test_loader(args.pcn_data_root, args.batch_size)
    print(f"[validpt-forward] PCN test loader ready: {len(loader)} batches")
    clean_cache = cache_clean_pcn_test(loader, args.max_samples_per_cell, args.seed)
    cache_hash = compute_cache_hash(clean_cache)
    print(f"[validpt-forward] cache_hash (sha256): {cache_hash[:16]}... "
          f"({len(clean_cache)} samples)")

    ops_dict = {n: OP_REGISTRY[n]() for n in sel_ops}
    n_cells = len(sel_models) * len(sel_ops) * len(sel_sevs)
    print(f"[validpt-forward] {n_cells} cells to forward, {len(clean_cache)} samples each "
          f"= {n_cells * len(clean_cache)} forwards total")

    manifest = {
        "stage": "validpt-forward-pipeline",
        "schema_version": NPZ_SCHEMA_VERSION,
        "args": vars(args),
        "n_cached_samples": len(clean_cache),
        "cache_hash_sha256": cache_hash,
        "cached_sample_indices": [e["idx"] for e in clean_cache],
        "cached_taxonomy_ids": [e["taxonomy_id"] for e in clean_cache],
        "cached_model_ids": [e["model_id"] for e in clean_cache],
        "cached_view_ids": [e["view_id"] for e in clean_cache],
        "models": sel_models,
        "ops": sel_ops,
        "severities": sel_sevs,
        "device": args.device,
        "protocol": PROTOCOL_LABEL,
        "protocol_description": (
            "valid-point-only (zero-pad rows preserved as exact (0,0,0) "
            "across noise/outlier/density/crop operators)"
        ),
    }
    manifest_path = pred_dir / "validpt_forward_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[validpt-forward] manifest written: {manifest_path}")

    grand_t0 = time.time()
    for model_name in sel_models:
        spec = name_to_spec[model_name]
        ckpt = spec["ckpt"]
        if not Path(ckpt).is_file():
            print(f"[validpt-forward] {model_name} ckpt missing -> SKIPPED")
            continue
        print(f"\n[validpt-forward] === {model_name} ===")
        model, _ = load_model(
            spec["cfg"], ckpt, args.device, model_name,
            load_strict=spec.get("load_strict", False),
            builder_kind=spec.get("builder"),
            builder_kwargs=spec.get("builder_kwargs"),
        )
        for op_name in sel_ops:
            op = ops_dict[op_name]
            # Assert the registered op's name matches the selected op_name BEFORE
            # forward; otherwise corruption_seed (computed
            # via op.name in op.__call__) would diverge from manifest op_name.
            if op.name != op_name:
                sys.exit(f"[validpt-forward] FATAL: OP_REGISTRY['{op_name}']().name == "
                         f"'{op.name}' (expected '{op_name}'); cache+seed integrity "
                         f"violated, refusing to forward")
            for sev in sel_sevs:
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
                                # Tighten skip-validation by requiring matching
                                # cache_hash and protocol so a
                                # stale npz from a different seed/data root or the
                                # zero-pad protocol cannot be silently reused.
                                and str(z.get("cache_hash_sha256", "")) == cache_hash
                                and str(z.get("protocol", "")) == PROTOCOL_LABEL
                            )
                    except Exception as e:
                        ok = False
                        print(f"  [{model_name}|{cell_key}] existing file unreadable "
                              f"({e}) -> re-forward")
                    if ok:
                        print(f"  [{model_name}|{cell_key}] skip (verified existing: "
                              f"{cell_path.name})")
                        continue
                    print(f"  [{model_name}|{cell_key}] existing file metadata/cache_hash "
                          f"mismatch -> re-forward")
                print(f"  [{model_name}|{cell_key}] forward...")
                cell_data = forward_cell(model, clean_cache, args.device, op, sev,
                                          spec, model_name, op_name)
                # Augment with the protocol stamps so skip-validation can verify
                # cache and protocol identity later.
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
    print(f"\n[validpt-forward] Total forward elapsed: {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")
    print(f"[validpt-forward] DONE. Run the metric stage (valid-point-only aware):")
    print(f"  python scripts/metric_emitter.py "
          f"--pcn-data-root {args.pcn_data_root} "
          f"--pred-dir {pred_dir} "
          f"--output-dir <metric_out_dir> "
          f"--max-samples-per-cell {args.max_samples_per_cell} "
          f"--models {','.join(sel_models)} "
          f"--ops {','.join(sel_ops)} "
          f"--severities {','.join(str(s) for s in sel_sevs)} "
          f"--seed {args.seed} "
          f"--manifest-name validpt_forward_manifest.json "
          f"--allow-validpt-ops")


if __name__ == "__main__":
    main()

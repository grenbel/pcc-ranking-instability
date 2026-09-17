"""Forward stage of the two-stage pipeline (GPU): corrupt the cached inputs, run one
model per cell, and save the dense predictions to disk for the metric stage.

Two-stage pipeline:
    F (this script, GPU):        cache builder + corruption + forward + save preds .npz per cell
    M (metric_emitter.py, CPU):  load preds + rebuild the cache + compute metrics + JSON rows

Per-cell output:
    {pred_dir}/{model_name}__{op}__s{sev}.npz containing:
        preds              (N_samples, 16384, 3)   float32   model output (incl. concat_partial if applicable)
        sample_indices     (N_samples,)            int32     PCN test loader idx (0-1199)
        taxonomy_ids       (N_samples,)            <U8       PCN category id strings
        model_ids          (N_samples,)            <U64      PCN model id strings
        view_ids           (N_samples,)            int32     view per object (PCN test = 0)
        model_name, op, severity, n_pred_points, concat_partial_applied, schema_version
                                                             scalar per-cell metadata

The metric stage uses sample_indices to align the preds with the cached PCN partial+GT
(from cache_clean_pcn_test with the same seed + max_samples_per_cell) and hard-fails
on any alignment mismatch.

Reuses scripts/sweep_common.py: cache_clean_pcn_test, parse_args structure, op application logic.
Reuses scripts/sanity_clean_pcn.py: DEFAULT_MODELS, cwd_to, _torch_load, load_pcn_test_loader, load_model.

Usage:
    python scripts/forward_sweep.py \\
        --pcn-data-root /path/to/PCN \\
        --pred-dir preds/zero_pad \\
        --device cuda:0 \\
        --max-samples-per-cell -1 \\
        --models PoinTr,AdaPoinTr,SnowFlakeNet,SeedFormer \\
        --ops noise,outlier,density,crop \\
        --severities 1,2,3,4,5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from scripts.sweep_common import (  # type: ignore
    DEFAULT_OPS, DEFAULT_SEVERITIES, cache_clean_pcn_test,
)
from src.corruptions import OP_REGISTRY

# Schema version for npz metadata (bumped when fields change; M validates compat)
NPZ_SCHEMA_VERSION = "1.0"


def compute_cache_hash(cache):
    """SHA-256 over (idx, taxonomy_id, model_id, partial_np bytes, gt_np bytes) sequence.

    An ID-only check could pass even if the PCN files / transforms silently produced
    different float bytes; the bit-exact hash catches that.
    """
    h = hashlib.sha256()
    for e in cache:
        h.update(str(e["idx"]).encode())
        h.update(e["taxonomy_id"].encode())
        h.update(e["model_id"].encode())
        h.update(str(e["view_id"]).encode())
        h.update(np.ascontiguousarray(e["partial_np"], dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(e["gt_np"], dtype=np.float32).tobytes())
    return h.hexdigest()


def atomic_savez(path: Path, **kwargs):
    """Write to a temporary file + atomic rename so that a half-written file is never
    silently picked up by the skip-existing logic.

    np.savez_compressed auto-appends .npz when the path does not end with .npz, so
    the temporary name must end with .npz to avoid a double suffix."""
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcn-data-root", required=True)
    p.add_argument("--pred-dir", required=True,
                   help="Output dir for per-cell .npz files (e.g. preds/zero_pad)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples-per-cell", type=int, default=-1,
                   help="-1 = full PCN test (1200) per cell; explicit small int for smoke")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--models", default="PoinTr,AdaPoinTr",
                   help="comma-separated model names from DEFAULT_MODELS")
    p.add_argument("--ops", default=",".join(DEFAULT_OPS),
                   help=f"comma-separated op names from {list(DEFAULT_OPS)}")
    p.add_argument("--severities", default=",".join(str(s) for s in DEFAULT_SEVERITIES),
                   help="comma-separated severities (1-5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-skip-missing-ckpt", action="store_true")
    return p.parse_args()


@torch.no_grad()
def forward_cell(model, clean_cache, device, op, severity, model_spec,
                 model_name, op_name):
    """Forward over deterministic clean cache with (op, severity) corruption.

    Returns dict ready for np.savez.
    """
    out_idx = model_spec.get("output_index", -1)
    concat_partial = model_spec.get("concat_partial", False)
    cell_key = f"{op_name}@s{severity}"
    n = len(clean_cache)
    preds_list = []
    sample_indices = []
    taxonomy_ids = []
    model_ids = []
    view_ids = []
    t_start = time.time()
    for i, entry in enumerate(clean_cache):
        partial_np = entry["partial_np"]
        tax_str = entry["taxonomy_id"]
        mid_str = entry["model_id"]
        view_id = entry["view_id"]
        # Apply corruption (deterministic per (tax, mid, view, sev, op) via make_seed)
        corrupted_np = op(partial_np, severity, tax_str, mid_str, view_id)
        corrupted_t = torch.from_numpy(corrupted_np).unsqueeze(0).to(device, non_blocking=True)
        # Forward
        ret = model(corrupted_t)
        pred_dense = ret[out_idx] if isinstance(ret, (list, tuple)) else ret
        # optional external concat of the partial (see DEFAULT_MODELS)
        if concat_partial:
            pred_dense = torch.cat([corrupted_t, pred_dense], dim=1)
        pred_np = pred_dense.cpu().numpy()[0].astype(np.float32)  # (N_pred, 3)
        preds_list.append(pred_np)
        sample_indices.append(entry["idx"])
        taxonomy_ids.append(tax_str)
        model_ids.append(mid_str)
        view_ids.append(view_id)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"    [{model_name}|{cell_key}] {i+1}/{n} forwards, "
                  f"elapsed {elapsed:.1f}s, est total {elapsed/((i+1)/n):.1f}s")
    # Stack - all preds same shape (16384, 3) for both PoinTr+concat and AdaPoinTr
    preds_arr = np.stack(preds_list, axis=0)  # (N, 16384, 3)
    return {
        "preds": preds_arr,
        "sample_indices": np.array(sample_indices, dtype=np.int32),
        "taxonomy_ids": np.array(taxonomy_ids),
        "model_ids": np.array(model_ids),
        "view_ids": np.array(view_ids, dtype=np.int32),
        "model_name": model_name,
        "op": op_name,
        "severity": severity,
        "n_pred_points": preds_arr.shape[1],
        "concat_partial_applied": concat_partial,
        # full per-cell metadata in the npz (not only in the manifest)
        "schema_version": NPZ_SCHEMA_VERSION,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Parse selections
    sel_models = [m.strip() for m in args.models.split(",") if m.strip()]
    sel_ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    sel_sevs = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
    if "pose" in sel_ops:
        sys.exit("[forward] 'pose' is not part of the audit grid.")
    for o in sel_ops:
        if o not in DEFAULT_OPS:
            sys.exit(f"[forward] op '{o}' not in the sweep scope {list(DEFAULT_OPS)}")
    for s in sel_sevs:
        if s not in (1, 2, 3, 4, 5):
            sys.exit(f"[forward] severity {s} out of [1,5]")

    name_to_spec = {ms["name"]: ms for ms in DEFAULT_MODELS}
    missing = []
    for m in sel_models:
        if m not in name_to_spec:
            sys.exit(f"[forward] unknown model '{m}'")
        if not Path(name_to_spec[m]["ckpt"]).is_file():
            missing.append((m, name_to_spec[m]["ckpt"]))
    if missing:
        msg = "[forward] FATAL - selected models with missing ckpts:\n" + "\n".join(
            f"    {m}: {p}" for m, p in missing)
        if args.allow_skip_missing_ckpt:
            print(msg + "\n  --allow-skip-missing-ckpt -> continuing")
        else:
            sys.exit(msg)

    print(f"[forward] Loading PCN test split from {args.pcn_data_root}")
    loader = load_pcn_test_loader(args.pcn_data_root, args.batch_size)
    print(f"[forward] PCN test loader ready: {len(loader)} batches")
    clean_cache = cache_clean_pcn_test(loader, args.max_samples_per_cell, args.seed)
    cache_hash = compute_cache_hash(clean_cache)
    print(f"[forward] cache_hash (sha256): {cache_hash[:16]}... ({len(clean_cache)} samples)")

    ops_dict = {n: OP_REGISTRY[n]() for n in sel_ops}
    n_cells = len(sel_models) * len(sel_ops) * len(sel_sevs)
    print(f"[forward] {n_cells} cells to forward, {len(clean_cache)} samples each "
          f"= {n_cells * len(clean_cache)} forwards total")

    # Save forward run manifest (used by metric stage to verify alignment)
    manifest = {
        "stage": "forward-pipeline",
        "schema_version": NPZ_SCHEMA_VERSION,
        "args": vars(args),
        "n_cached_samples": len(clean_cache),
        "cache_hash_sha256": cache_hash,  # bit-exact cross-stage verification
        "cached_sample_indices": [e["idx"] for e in clean_cache],
        "cached_taxonomy_ids": [e["taxonomy_id"] for e in clean_cache],
        "cached_model_ids": [e["model_id"] for e in clean_cache],
        "cached_view_ids": [e["view_id"] for e in clean_cache],
        "models": sel_models,
        "ops": sel_ops,
        "severities": sel_sevs,
        "device": args.device,
    }
    manifest_path = pred_dir / "forward_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[forward] manifest written: {manifest_path}")

    # Cells loop
    grand_t0 = time.time()
    for model_name in sel_models:
        spec = name_to_spec[model_name]
        ckpt = spec["ckpt"]
        if not Path(ckpt).is_file():
            print(f"[forward] {model_name} ckpt missing -> SKIPPED")
            continue
        print(f"\n[forward] === {model_name} ===")
        # The forward stage honors load_strict so a baseline cannot silently
        # partial-load if the sanity check was skipped or the ckpt/cfg changed.
        # builder_kind is forwarded so SeedFormer goes through its shim loader.
        model, _ = load_model(
            spec["cfg"], ckpt, args.device, model_name,
            load_strict=spec.get("load_strict", False),
            builder_kind=spec.get("builder"),
            builder_kwargs=spec.get("builder_kwargs"),
        )
        for op_name in sel_ops:
            op = ops_dict[op_name]
            for sev in sel_sevs:
                cell_key = f"{op_name}@s{sev}"
                cell_path = pred_dir / f"{model_name}__{op_name}__s{sev}.npz"
                # skip-existing must verify that the per-cell metadata matches the current
                # run AND that the file carries a cache_hash stamp equal to the current cache
                # hash. Files without a stamp (written before the stamp existed) are always
                # re-forwarded: nothing else proves which cache produced them, and the manifest
                # in pred_dir is overwritten above before the cells are processed.
                if cell_path.exists():
                    try:
                        with np.load(cell_path, allow_pickle=False) as z:
                            has_stamp = "cache_hash_sha256" in z.files
                            ok = (
                                str(z.get("schema_version", "")) == NPZ_SCHEMA_VERSION
                                and str(z["model_name"]) == model_name
                                and str(z["op"]) == op_name
                                and int(z["severity"]) == sev
                                and int(z["n_pred_points"]) == 16384
                                and int(z["preds"].shape[0]) == len(clean_cache)
                                and bool(z["concat_partial_applied"]) == bool(spec.get("concat_partial", False))
                                and has_stamp and str(z["cache_hash_sha256"]) == cache_hash
                            )
                    except Exception as e:
                        ok = False
                        print(f"  [{model_name}|{cell_key}] existing file unreadable ({e}) -> re-forward")
                    if ok:
                        print(f"  [{model_name}|{cell_key}] skip (verified existing: {cell_path.name})")
                        continue
                    print(f"  [{model_name}|{cell_key}] existing file metadata/cache_hash mismatch -> re-forward")
                print(f"  [{model_name}|{cell_key}] forward...")
                cell_data = forward_cell(model, clean_cache, args.device, op, sev,
                                          spec, model_name, op_name)
                cell_data["cache_hash_sha256"] = cache_hash  # per-file provenance stamp
                atomic_savez(cell_path, **cell_data)
                size_mb = cell_path.stat().st_size / 1e6
                print(f"  [{model_name}|{cell_key}] saved {cell_path.name} "
                      f"({size_mb:.1f} MB, {cell_data['preds'].shape})")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    elapsed = time.time() - grand_t0
    print(f"\n[forward] Total forward elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[forward] DONE. Run metric stage:")
    print(f"  python scripts/metric_emitter.py "
          f"--pcn-data-root {args.pcn_data_root} "
          f"--pred-dir {pred_dir} "
          f"--output-dir <metric_out_dir> "
          f"--max-samples-per-cell {args.max_samples_per_cell} "
          f"--models {','.join(sel_models)} "
          f"--ops {','.join(sel_ops)} "
          f"--severities {','.join(str(s) for s in sel_sevs)} "
          f"--seed {args.seed}")


if __name__ == "__main__":
    main()

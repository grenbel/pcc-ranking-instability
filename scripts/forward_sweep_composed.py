"""Composed-operator pilot: forward stage (GPU) for two-operator corruptions.

Zero-pad primary protocol (same loader as the main grid) with two-operator composed
corruptions: 3 pairs x 5 severities x 4 baselines = 60 forward cells. Based on
forward_sweep.py (NOT the UpSamplePoints script); adds constituent stamp fields,
input-side row audits and a per-cell cross-baseline corrupted-input digest.

Key properties:
    - loader: DEFAULT zero-pad transform -> the cache hash MUST equal the zero-pad
      invariant (prefix 80d3efce468eba6c), the inverse of the UpSamplePoints gate
    - composition order fixed: crop -> density -> noise -> outlier (subset per pair)
    - constituent seeds identical to their single-operator cells (matched streams;
      comparisons remain descriptive, upstream geometry effects included)
    - per-sample input row audits pre/mid/post (exact-zero + sum-zero counts)
    - per-cell corrupted-input SHA-256, asserted identical across all 4 baselines

Output filenames: {pred_dir}/{model}__{op_id}__s{sev}.npz
    op_id in {mixed_noise_outlier, mixed_crop_noise, mixed_density_outlier}
    (the op ids keep the `mixed_` prefix used by the released per-sample data)

Usage:
    python scripts/forward_sweep_composed.py \\
        --pcn-data-root /path/to/PCN \\
        --pred-dir preds/composed \\
        --device cuda:0 --max-samples-per-cell -1
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
from scripts.sweep_common import cache_clean_pcn_test  # type: ignore
from scripts.forward_sweep import (  # type: ignore
    NPZ_SCHEMA_VERSION,
    atomic_savez,
    compute_cache_hash,
)
from src.corruptions import OP_REGISTRY
from src.corruptions.base import make_seed

DEFAULT_SEVERITIES = (1, 2, 3, 4, 5)
DEFAULT_BASELINES = ("PoinTr", "AdaPoinTr", "SnowFlakeNet", "SeedFormer")
PROTOCOL_LABEL = "mixed_zero_pad_v1"
# Local canonical order for the composed cells; compose.CANONICAL_ORDER is NOT
# reused (it starts with 'pose' and raises on ops absent from its list).
MIXED_ORDER = ("crop", "density", "noise", "outlier")
# op_id -> ordered constituent base-op names (must be a subsequence of MIXED_ORDER)
MIXED_PAIRS = {
    "mixed_noise_outlier": ("noise", "outlier"),
    "mixed_crop_noise": ("crop", "noise"),
    "mixed_density_outlier": ("density", "outlier"),
}
# Zero-pad protocol invariant (zero-pad and validpt manifests). The composed sweep REQUIRES it:
# if the cache hash differs, the loader is not the zero-pad primary protocol.
ZERO_PAD_CACHE_HASH_PREFIX = "80d3efce468eba6c"


def _row_counts(points: np.ndarray):
    """(exact-zero rows, coordinate-sum-zero rows): the same criteria as the
    UpSamplePoints cache_row_stats disclosure and the metric-stage chamfer mask audit."""
    p = np.asarray(points, dtype=np.float32)
    exact = int(np.all(p == 0.0, axis=-1).sum())
    sumz = int((p.sum(axis=-1) == 0).sum())
    return exact, sumz


class MixedOp:
    """Two-operator sequential composition following MIXED_ORDER.

    Implements the CorruptionOp calling convention used by the forward loop:
    op(points, severity, tax, mid, view) -> np.ndarray. Both constituents apply
    at the SAME severity; each constituent derives its per-sample SHA-1 seed
    from its own base-op name exactly as in its single-operator cell (matched
    constituent seeds).

    Legacy zero-pad-included semantics: constituents are the unmodified
    base operators, so padding rows participate in corruption and upstream
    crop/density may displace exact-zero rows before noise/outlier act.

    After each __call__, `last_audit` holds the input-side row audit for that
    sample: {"pre": (exact, sumz), "after_<op1>": (...), "final": (...)}.
    """

    def __init__(self, op_id: str, constituent_names, registry):
        if op_id not in MIXED_PAIRS:
            raise KeyError(f"unknown mixed op_id '{op_id}'")
        expected = MIXED_PAIRS[op_id]
        if tuple(constituent_names) != expected:
            raise ValueError(f"{op_id}: constituents {constituent_names} != "
                             f"registry order {expected}")
        order_idx = [MIXED_ORDER.index(n) for n in constituent_names]
        if order_idx != sorted(order_idx):
            raise ValueError(f"{op_id}: constituents {constituent_names} violate "
                             f"MIXED_ORDER {MIXED_ORDER}")
        self.name = op_id
        self.constituent_names = tuple(constituent_names)
        self.constituents = [(n, registry[n]()) for n in constituent_names]
        for n, inst in self.constituents:
            got = getattr(inst, "name", None)
            if got != n:
                raise ValueError(f"{op_id}: constituent instance name '{got}' != "
                                 f"'{n}' - seed integrity would break")
        self.last_audit = None
        self.call_trace = []  # smoke support: records constituent call order

    def __call__(self, points, severity, tax, mid, view):
        audit = {"pre": _row_counts(points)}
        self.call_trace = []
        out = points
        for pos, (n, inst) in enumerate(self.constituents):
            out = inst(out, severity, tax, mid, view)
            self.call_trace.append(n)
            key = f"after_{n}" if pos < len(self.constituents) - 1 else "final"
            audit[key] = _row_counts(out)
        self.last_audit = audit
        return out


@torch.no_grad()
def forward_cell_mixed(model, clean_cache, device, op: MixedOp, severity,
                       model_spec, model_name):
    """Forward over the deterministic clean cache with a composed corruption.

    Extends forward_sweep.forward_cell with: corrupted-input SHA-256 digest
    (cross-baseline matched-control evidence), per-sample input row audits, and
    constituent stamp fields. Returns dict ready for np.savez.
    """
    out_idx = model_spec.get("output_index", -1)
    concat_partial = model_spec.get("concat_partial", False)
    cell_key = f"{op.name}@s{severity}"
    n = len(clean_cache)
    preds_list, sample_indices, taxonomy_ids, model_ids, view_ids = [], [], [], [], []
    in_exact_pre, in_sumz_pre = [], []
    in_exact_mid, in_sumz_mid = [], []
    in_exact_final, in_sumz_final = [], []
    constituent_seeds = []  # (N, 2) - the ACTUAL seeds each constituent used
    digest = hashlib.sha256()
    t_start = time.time()
    for i, entry in enumerate(clean_cache):
        constituent_seeds.append([
            make_seed(entry["taxonomy_id"], entry["model_id"], entry["view_id"],
                      severity, base) for base in op.constituent_names])
        corrupted_np = op(entry["partial_np"], severity, entry["taxonomy_id"],
                          entry["model_id"], entry["view_id"])
        corrupted_np = np.ascontiguousarray(corrupted_np, dtype=np.float32)
        digest.update(corrupted_np.tobytes())
        a = op.last_audit
        in_exact_pre.append(a["pre"][0]); in_sumz_pre.append(a["pre"][1])
        mid_key = f"after_{op.constituent_names[0]}"
        in_exact_mid.append(a[mid_key][0]); in_sumz_mid.append(a[mid_key][1])
        in_exact_final.append(a["final"][0]); in_sumz_final.append(a["final"][1])
        corrupted_t = torch.from_numpy(corrupted_np).unsqueeze(0).to(
            device, non_blocking=True)
        ret = model(corrupted_t)
        pred_dense = ret[out_idx] if isinstance(ret, (list, tuple)) else ret
        if concat_partial:
            pred_dense = torch.cat([corrupted_t, pred_dense], dim=1)
        preds_list.append(pred_dense.cpu().numpy()[0].astype(np.float32))
        sample_indices.append(entry["idx"])
        taxonomy_ids.append(entry["taxonomy_id"])
        model_ids.append(entry["model_id"])
        view_ids.append(entry["view_id"])
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"    [{model_name}|{cell_key}] {i+1}/{n} forwards, "
                  f"elapsed {elapsed:.1f}s, est total {elapsed/((i+1)/n):.1f}s")
    preds_arr = np.stack(preds_list, axis=0)
    return {
        "preds": preds_arr,
        "sample_indices": np.array(sample_indices, dtype=np.int32),
        "taxonomy_ids": np.array(taxonomy_ids),
        "model_ids": np.array(model_ids),
        "view_ids": np.array(view_ids, dtype=np.int32),
        "model_name": model_name,
        "op": op.name,
        "severity": severity,
        "n_pred_points": preds_arr.shape[1],
        "concat_partial_applied": concat_partial,
        "schema_version": NPZ_SCHEMA_VERSION,
        "protocol": PROTOCOL_LABEL,
        "constituent_ops": np.array(op.constituent_names),
        "constituent_severities": np.array([severity] * len(op.constituent_names),
                                           dtype=np.int32),
        "constituent_seeds": np.array(constituent_seeds, dtype=np.int64),
        "composition_order": np.array(MIXED_ORDER),
        "input_digest_sha256": digest.hexdigest(),
        "n_input_exact_zero_rows_pre": np.array(in_exact_pre, dtype=np.int32),
        "n_input_sumzero_rows_pre": np.array(in_sumz_pre, dtype=np.int32),
        "n_input_exact_zero_rows_mid": np.array(in_exact_mid, dtype=np.int32),
        "n_input_sumzero_rows_mid": np.array(in_sumz_mid, dtype=np.int32),
        "n_input_exact_zero_rows_final": np.array(in_exact_final, dtype=np.int32),
        "n_input_sumzero_rows_final": np.array(in_sumz_final, dtype=np.int32),
    }


def verify_existing_cell(cell_path, model_name, op_id, sev, clean_cache, spec,
                         cache_hash):
    """Skip-existing verification for one composed cell NPZ.

    Verifies ALL provenance fields (constituent
    severities/order, the six input row-audit arrays, per-sample constituent
    seeds recomputed from the cache) - not just constituent_ops + digest length
    - so a stale or malformed same-protocol NPZ cannot be silently reused and
    blessed into cell_input_digests.

    Returns (ok, existing_digest). Raises on unreadable files (caller catches).
    """
    n = len(clean_cache)
    with np.load(cell_path, allow_pickle=False) as z:
        existing_digest = str(z.get("input_digest_sha256", ""))
        audit_ok = all(
            k in z.files
            and z[k].shape == (n,)
            and np.issubdtype(z[k].dtype, np.integer)
            for k in (
                "n_input_exact_zero_rows_pre",
                "n_input_sumzero_rows_pre",
                "n_input_exact_zero_rows_mid",
                "n_input_sumzero_rows_mid",
                "n_input_exact_zero_rows_final",
                "n_input_sumzero_rows_final",
            ))
        seeds_ok = False
        if "constituent_seeds" in z.files and z["constituent_seeds"].shape == (n, 2):
            expected_seeds = np.array(
                [[make_seed(e["taxonomy_id"], e["model_id"], e["view_id"], sev, base)
                  for base in MIXED_PAIRS[op_id]]
                 for e in clean_cache], dtype=np.int64)
            seeds_ok = bool(np.array_equal(z["constituent_seeds"], expected_seeds))
        ok = (
            str(z.get("schema_version", "")) == NPZ_SCHEMA_VERSION
            and str(z["model_name"]) == model_name
            and str(z["op"]) == op_id
            and int(z["severity"]) == sev
            and int(z["n_pred_points"]) == 16384
            and int(z["preds"].shape[0]) == n
            and bool(z["concat_partial_applied"]) == bool(
                spec.get("concat_partial", False))
            and str(z.get("cache_hash_sha256", "")) == cache_hash
            and str(z.get("protocol", "")) == PROTOCOL_LABEL
            and list(z["constituent_ops"]) == list(MIXED_PAIRS[op_id])
            and list(z["constituent_severities"]) == [sev, sev]
            and list(z["composition_order"]) == list(MIXED_ORDER)
            and audit_ok
            and seeds_ok
            and len(existing_digest) == 64
        )
    return ok, existing_digest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcn-data-root", required=True)
    p.add_argument("--pred-dir", required=True,
                   help="Output dir for per-cell .npz (e.g. preds/composed)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples-per-cell", type=int, default=-1,
                   help="-1 = full PCN test (1200) per cell; smaller for smoke "
                        "(NOTE: the zero-pad cache-hash gate only enforces at -1)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--models", default=",".join(DEFAULT_BASELINES))
    p.add_argument("--mixed-ops", default=",".join(MIXED_PAIRS.keys()),
                   help=f"comma-separated mixed op ids from {sorted(MIXED_PAIRS)}")
    p.add_argument("--severities", default=",".join(str(s) for s in DEFAULT_SEVERITIES))
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
        sys.exit(f"[composed-forward] --batch-size must be 1 (got {args.batch_size}); "
                 f"cache_clean_pcn_test only retains payload[0] per batch")

    sel_models = [m.strip() for m in args.models.split(",") if m.strip()]
    sel_ops = [o.strip() for o in args.mixed_ops.split(",") if o.strip()]
    sel_sevs = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
    for label, lst in (("models", sel_models), ("mixed-ops", sel_ops),
                       ("severities", sel_sevs)):
        if len(set(lst)) != len(lst):
            sys.exit(f"[composed-forward] duplicate {label}: {lst}")
    for o in sel_ops:
        if o not in MIXED_PAIRS:
            sys.exit(f"[composed-forward] mixed op '{o}' not in {sorted(MIXED_PAIRS)}")
        for base in MIXED_PAIRS[o]:
            if base not in OP_REGISTRY:
                sys.exit(f"[composed-forward] constituent '{base}' of '{o}' missing from "
                         f"OP_REGISTRY")
    for s in sel_sevs:
        if s not in (1, 2, 3, 4, 5):
            sys.exit(f"[composed-forward] severity {s} out of [1,5]")

    name_to_spec = {ms["name"]: ms for ms in DEFAULT_MODELS}
    missing = []
    for m in sel_models:
        if m not in name_to_spec:
            sys.exit(f"[composed-forward] unknown model '{m}'; "
                     f"available: {sorted(name_to_spec.keys())}")
        if not Path(name_to_spec[m]["ckpt"]).is_file():
            missing.append((m, name_to_spec[m]["ckpt"]))
    if missing:
        msg = "[composed-forward] FATAL - selected models with missing ckpts:\n" + "\n".join(
            f"    {m}: {p}" for m, p in missing)
        if args.allow_skip_missing_ckpt:
            print(msg + "\n  --allow-skip-missing-ckpt -> continuing")
        else:
            sys.exit(msg)

    print(f"[composed-forward] Loading PCN test split from {args.pcn_data_root} "
          f"(DEFAULT zero-pad loader - primary protocol)")
    loader = load_pcn_test_loader(args.pcn_data_root, args.batch_size)
    print(f"[composed-forward] PCN test loader ready: {len(loader)} batches")
    clean_cache = cache_clean_pcn_test(loader, args.max_samples_per_cell, args.seed)
    cache_hash = compute_cache_hash(clean_cache)
    print(f"[composed-forward] cache_hash (sha256): {cache_hash[:16]}... "
          f"({len(clean_cache)} samples)")
    # Inverse of the UpSamplePoints gate: the composed sweep REQUIRES the zero-pad invariant. Only
    # enforceable on the full 1200-sample cache (subsets hash differently).
    if args.max_samples_per_cell == -1:
        if not cache_hash.startswith(ZERO_PAD_CACHE_HASH_PREFIX):
            sys.exit(f"[composed-forward] FATAL: full-cache hash {cache_hash[:16]}... does not "
                     f"match the zero-pad invariant {ZERO_PAD_CACHE_HASH_PREFIX}... - "
                     f"loader is NOT the primary protocol. Refusing to forward.")
        print(f"[composed-forward] zero-pad invariant verified ({ZERO_PAD_CACHE_HASH_PREFIX}...)")
    else:
        print(f"[composed-forward] WARNING: subset cache ({len(clean_cache)} samples) - "
              f"zero-pad invariant gate skipped (smoke mode)")

    cache_exact = [int(np.all(np.asarray(e["partial_np"]) == 0.0, axis=-1).sum())
                   for e in clean_cache]
    cache_sumz = [int((np.asarray(e["partial_np"]).sum(axis=-1) == 0).sum())
                  for e in clean_cache]
    print(f"[composed-forward] clean-cache padding stats: exact-zero rows total="
          f"{sum(cache_exact)}, sum-zero rows total={sum(cache_sumz)}")

    ops_dict = {oid: MixedOp(oid, MIXED_PAIRS[oid], OP_REGISTRY) for oid in sel_ops}
    cell_plan = [(oid, s) for oid in sel_ops for s in sel_sevs]
    n_cells = len(sel_models) * len(cell_plan)
    print(f"[composed-forward] {n_cells} cells to forward, {len(clean_cache)} samples each "
          f"= {n_cells * len(clean_cache)} forwards total")

    manifest = {
        "stage": "mixed-forward-pipeline",
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
            "Two-operator composed corruptions on the zero-pad primary protocol "
            "(same loader and clean-cache invariant as the main 20-cell grid). "
            "Composition order fixed crop->density->noise->outlier; both "
            "constituents share the cell severity; each constituent derives its "
            "per-sample SHA-1 seed from its own base-op name exactly as in its "
            "single-operator cell (matched streams; composition-vs-single "
            "comparisons remain descriptive). Legacy zero-pad-included semantics: "
            "padding rows participate in corruption; per-sample input row audits "
            "(pre/mid/final exact-zero and sum-zero counts) stored in each NPZ. "
            "Per-cell corrupted-input SHA-256 asserted identical across baselines."
        ),
        "mixed_pairs": {k: list(v) for k, v in MIXED_PAIRS.items() if k in sel_ops},
        "composition_order": list(MIXED_ORDER),
        "cache_stats_exact_zero_row_counts": cache_exact,
        "cache_stats_sumzero_row_counts": cache_sumz,
        # status marker so an interrupted run's manifest
        # (empty/stale digests) is distinguishable from a completed one.
        "manifest_status": "incomplete",
        "cell_input_digests": {},  # filled at the final atomic write
    }
    manifest_path = pred_dir / "composed_forward_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[composed-forward] manifest written: {manifest_path}")

    digest_registry = {}  # cell_key -> (hexdigest, first_model)

    def register_digest(cell_key, hexdigest, model_name):
        prev = digest_registry.get(cell_key)
        if prev is None:
            digest_registry[cell_key] = (hexdigest, model_name)
        elif prev[0] != hexdigest:
            sys.exit(f"[composed-forward] FATAL: corrupted-input digest mismatch for "
                     f"{cell_key}: {model_name} produced {hexdigest[:16]}... but "
                     f"{prev[1]} produced {prev[0][:16]}... - matched-control "
                     f"invariant violated, refusing to continue.")

    grand_t0 = time.time()
    for model_name in sel_models:
        spec = name_to_spec[model_name]
        if not Path(spec["ckpt"]).is_file():
            print(f"[composed-forward] {model_name} ckpt missing -> SKIPPED")
            continue
        print(f"\n[composed-forward] === {model_name} ===")
        model, _ = load_model(
            spec["cfg"], spec["ckpt"], args.device, model_name,
            load_strict=spec.get("load_strict", False),
            builder_kind=spec.get("builder"),
            builder_kwargs=spec.get("builder_kwargs"),
        )
        for op_id, sev in cell_plan:
            op = ops_dict[op_id]
            if op.name != op_id:
                sys.exit(f"[composed-forward] FATAL: op instance name '{op.name}' != "
                         f"'{op_id}'; seed integrity violated")
            cell_key = f"{op_id}@s{sev}"
            cell_path = pred_dir / f"{model_name}__{op_id}__s{sev}.npz"
            if cell_path.exists():
                try:
                    ok, existing_digest = verify_existing_cell(
                        cell_path, model_name, op_id, sev, clean_cache, spec,
                        cache_hash)
                except Exception as e:
                    ok, existing_digest = False, None
                    print(f"  [{model_name}|{cell_key}] existing file unreadable "
                          f"({e}) -> re-forward")
                if ok:
                    register_digest(cell_key, existing_digest, model_name)
                    print(f"  [{model_name}|{cell_key}] skip (verified existing)")
                    continue
                print(f"  [{model_name}|{cell_key}] existing file metadata/protocol/"
                      f"provenance mismatch -> re-forward")
            print(f"  [{model_name}|{cell_key}] forward...")
            cell_data = forward_cell_mixed(model, clean_cache, args.device, op, sev,
                                           spec, model_name)
            cell_data["cache_hash_sha256"] = cache_hash
            register_digest(cell_key, cell_data["input_digest_sha256"], model_name)
            atomic_savez(cell_path, **cell_data)
            size_mb = cell_path.stat().st_size / 1e6
            print(f"  [{model_name}|{cell_key}] saved {cell_path.name} "
                  f"({size_mb:.1f} MB, {cell_data['preds'].shape}; input digest "
                  f"{cell_data['input_digest_sha256'][:16]}...)")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Persist the cross-baseline-verified digests into the manifest (atomic:
    # temp + os.replace, so a crash mid-write cannot truncate the manifest).
    manifest["cell_input_digests"] = {k: v[0] for k, v in digest_registry.items()}
    manifest["manifest_status"] = "complete"
    tmp_manifest = manifest_path.parent / (manifest_path.name + ".tmp")
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_manifest, manifest_path)
    print(f"[composed-forward] manifest updated with {len(digest_registry)} verified "
          f"cell input digests (status=complete)")

    elapsed = time.time() - grand_t0
    print(f"\n[composed-forward] Total forward elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[composed-forward] DONE. Metric stage:")
    print(f"  python scripts/metric_emitter.py "
          f"--pcn-data-root {args.pcn_data_root} "
          f"--pred-dir {pred_dir} "
          f"--output-dir <metric_out_dir> "
          f"--max-samples-per-cell {args.max_samples_per_cell} "
          f"--models {','.join(sel_models)} "
          f"--ops {','.join(sel_ops)} "
          f"--severities {','.join(str(s) for s in sel_sevs)} "
          f"--seed {args.seed} "
          f"--manifest-name composed_forward_manifest.json "
          f"--expect-protocol {PROTOCOL_LABEL} "
          f"--allow-mixed-ops")


if __name__ == "__main__":
    main()

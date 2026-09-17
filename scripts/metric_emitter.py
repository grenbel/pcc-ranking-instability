"""Metric stage of the two-stage pipeline (CPU): the per-cell metric emitter.

Loads the dense predictions written by the forward stage, rebuilds the clean cache
with the same seed, verifies alignment and the cache hash, and writes one
per-sample JSON per (model, operator, severity) cell.

Two-stage pipeline:
    F (forward_sweep*.py, GPU):  forward + save preds .npz per cell
    M (this script, CPU):        load preds + rebuild PCN cache + compute CD/F + decomposition + JSON

The metric stage must use the SAME --max-samples-per-cell, --seed, --models, --ops and
--severities as the forward stage. Alignment is verified via cached_sample_indices in
the forward manifest (default forward_manifest.json) and hard-fails on mismatch.

Per-cell output (the per-sample rows consumed by the analysis scripts):
    {output_dir}/{model_name}_{op}_s{sev}_per_sample.json
        list of dicts with: idx, model_name, taxonomy_id, model_id, view_id, op, severity,
            cell_key, corruption_seed, n_pred_points, n_gt_points, n_pred_zero_rows,
            n_pred_sumzero_rows, n_gt_sumzero_rows, cd_l1, cd_l2, f_at_*, decomp_*
            (+ protocol, cache_hash_sha256 when a protocol guard is active)

Usage:
    python scripts/metric_emitter.py \\
        --pcn-data-root /path/to/PCN \\
        --pred-dir preds/zero_pad \\
        --output-dir logs/zero_pad \\
        --max-samples-per-cell -1 \\
        --models PoinTr,AdaPoinTr,SnowFlakeNet,SeedFormer \\
        --ops noise,outlier,density,crop \\
        --severities 1,2,3,4,5 \\
        --seed 42 --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
POINTR_ROOT = REPO_ROOT / "baselines" / "PoinTr"
sys.path.insert(0, str(POINTR_ROOT))
sys.path.insert(0, str(REPO_ROOT))

# Reuse cache logic + corruption seed for self-describing rows
from scripts.sanity_clean_pcn import DEFAULT_MODELS, load_pcn_test_loader  # type: ignore
from scripts.sweep_common import (  # type: ignore
    DEFAULT_OPS, DEFAULT_SEVERITIES, cache_clean_pcn_test, aggregate_cell,
)
from scripts.forward_sweep import (  # type: ignore
    NPZ_SCHEMA_VERSION, compute_cache_hash,
)
from src.corruptions import make_seed
from src.metrics.reconstruction import chamfer_l1, chamfer_l2, fscore_at_thresholds
from src.metrics import DECOMPOSITION_METRICS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcn-data-root", required=True)
    p.add_argument("--pred-dir", required=True,
                   help="Input dir of per-cell .npz files from forward_sweep.py")
    p.add_argument("--output-dir", required=True,
                   help="Output dir for per-cell metric JSONs (compatible with sweep_common.py format)")
    p.add_argument("--max-samples-per-cell", type=int, default=-1,
                   help="MUST match forward stage exactly (alignment verified via manifest)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--models", default="PoinTr,AdaPoinTr")
    p.add_argument("--ops", default=",".join(DEFAULT_OPS))
    p.add_argument("--severities", default=",".join(str(s) for s in DEFAULT_SEVERITIES))
    p.add_argument("--seed", type=int, default=42,
                   help="MUST match forward stage exactly")
    p.add_argument("--allow-subset", action="store_true",
                   help="Permit metric to run on subset of forward's models/ops/severities. "
                        "Default off: exact match required.")
    p.add_argument("--resume-existing", action="store_true",
                   help="Skip cells whose per_sample.json already exists, aggregate them into the "
                        "new summary, and compute only missing cells.")
    p.add_argument("--workers", type=int, default=1,
                   help="Cell-level multiprocessing workers (default 1=serial; 4-8 typical for "
                        "multi-core CPU). Each worker thread-limits BLAS to 1 to avoid worker x "
                        "thread oversubscription. Output bit-exact identical to serial since "
                        "per-cell compute is deterministic + has no RNG / iteration-order dep.")
    p.add_argument("--manifest-name", default="forward_manifest.json",
                   help="Forward manifest filename inside --pred-dir "
                        "(validpt_forward_manifest.json, upsample_forward_manifest.json or "
                        "composed_forward_manifest.json for the protocol variants).")
    p.add_argument("--allow-validpt-ops", action="store_true",
                   help="Permit `*_validpt` op names (valid-point-only protocol). "
                        "Off by default to keep the zero-pad pipeline strict.")
    p.add_argument("--loader-cfg", default=None,
                   help="Loader cfg yaml for the metric-stage cache rebuild. MUST match the "
                        "forward stage's transform (the UpSamplePoints sweep passes "
                        "SnowFlakeNet.yaml); default None = PoinTr.yaml zero-pad path. A wrong "
                        "value fails fast at the cache_hash check.")
    p.add_argument("--expect-protocol", default="",
                   help="If non-empty (e.g. 'upsample_points_v1'), EVERY loaded pred .npz must "
                        "carry protocol == this value AND cache_hash_sha256 == the forward "
                        "manifest hash; per-sample JSONs are stamped + resume-verified with it. "
                        "Guards against stale same-filename archives from another protocol. "
                        "Empty = legacy behaviour (auto-filled from the manifest when it "
                        "declares a protocol).")
    p.add_argument("--allow-clean-cell", action="store_true",
                   help="Permit the op='clean' severity=0 pass-through cell (UpSamplePoints "
                        "sweep). Valid "
                        "(op, sev) pairs are filtered: clean only with 0, corruption ops "
                        "only with 1-5.")
    p.add_argument("--allow-mixed-ops", action="store_true",
                   help="Permit the composed-operator op ids (mixed_noise_outlier, "
                        "mixed_crop_noise, mixed_density_outlier; must match "
                        "forward_sweep_composed.MIXED_PAIRS). Off by default to keep "
                        "the single-operator pipelines strict. Constituents are "
                        "derivable from the op id + forward manifest mixed_pairs; "
                        "exact op-list agreement is still enforced by the manifest "
                        "alignment check.")
    return p.parse_args()


def verify_alignment(forward_manifest, args, sel_models, sel_ops, sel_sevs):
    """Hard-fail if metric args do not match forward stage.

    Also verifies that models/ops/severities are a subset of the forward run (with
    --allow-subset) or an exact match (default).
    """
    fm_args = forward_manifest["args"]
    mismatches = []
    for k in ("max_samples_per_cell", "seed"):
        if fm_args.get(k) != getattr(args, k):
            mismatches.append(f"{k}: forward={fm_args.get(k)} vs metric={getattr(args, k)}")
    if mismatches:
        sys.exit("[metric] FATAL alignment mismatch with forward stage:\n  " +
                 "\n  ".join(mismatches) +
                 "\n  Same --max-samples-per-cell + --seed required to match cached PCN samples.")
    # Schema version
    fm_schema = forward_manifest.get("schema_version", "?")
    if fm_schema != NPZ_SCHEMA_VERSION:
        sys.exit(f"[metric] FATAL schema_version mismatch: "
                 f"forward={fm_schema} vs metric={NPZ_SCHEMA_VERSION}")
    # Models/ops/severities subset check
    fm_models = set(forward_manifest.get("models", []))
    fm_ops = set(forward_manifest.get("ops", []))
    fm_sevs = set(forward_manifest.get("severities", []))
    if args.allow_subset:
        for m in sel_models:
            if m not in fm_models:
                sys.exit(f"[metric] FATAL model '{m}' not in forward manifest models {sorted(fm_models)}")
        for o in sel_ops:
            if o not in fm_ops:
                sys.exit(f"[metric] FATAL op '{o}' not in forward manifest ops {sorted(fm_ops)}")
        for s in sel_sevs:
            if s not in fm_sevs:
                sys.exit(f"[metric] FATAL sev {s} not in forward manifest sevs {sorted(fm_sevs)}")
    else:
        if set(sel_models) != fm_models:
            sys.exit(f"[metric] FATAL models exact-match failed: "
                     f"forward={sorted(fm_models)} vs metric={sorted(set(sel_models))}\n"
                     f"  pass --allow-subset to permit partial metric runs.")
        if set(sel_ops) != fm_ops:
            sys.exit(f"[metric] FATAL ops exact-match failed.")
        if set(sel_sevs) != fm_sevs:
            sys.exit(f"[metric] FATAL severities exact-match failed.")


# ====================================================================
# Worker globals (populated in init_worker on each spawned worker process).
# Must be module-level so spawn-pickled _worker_process_cell can access them.
# ====================================================================
_W_CACHE = None
_W_DECOMP = None
_W_PRED_DIR = None
_W_OUT_DIR = None
_W_NAME_TO_CONCAT = None
_W_EXPECT_PROTOCOL = ""
_W_MANIFEST_HASH = None
_W_INIT_ERROR = None  # captured init failure, raised on the first task


def init_worker(cache_pkl_path: str, pred_dir_str: str, out_dir_str: str,
                name_to_concat: dict, expect_protocol: str = "",
                manifest_hash: str = None):
    """Initialize each worker process: limit BLAS threads, load PCN cache once.

    Uses threadpoolctl (scipy dep) to dynamically restrict numpy/BLAS threads,
    avoiding worker x thread oversubscription (e.g., 4 workers x 8 BLAS threads
    = 32-thread thrash). Cache pickle-loaded once per worker (~3s for 280 MB)
    and reused across all cells that worker handles.

    Init failures are captured into _W_INIT_ERROR rather than raised, because Pool
    initializer exceptions cause repeated worker respawn/hang instead of clean
    propagation; _worker_process_cell raises on the first task instead.
    """
    global _W_CACHE, _W_DECOMP, _W_PRED_DIR, _W_OUT_DIR, _W_NAME_TO_CONCAT, \
        _W_EXPECT_PROTOCOL, _W_MANIFEST_HASH, _W_INIT_ERROR
    try:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1)
        except ImportError:
            pass  # fall back to default thread count if threadpoolctl missing
        with open(cache_pkl_path, "rb") as f:
            _W_CACHE = pickle.load(f)
        _W_DECOMP = {n: cls() for n, cls in DECOMPOSITION_METRICS.items()}
        _W_PRED_DIR = Path(pred_dir_str)
        _W_OUT_DIR = Path(out_dir_str)
        _W_NAME_TO_CONCAT = name_to_concat
        _W_EXPECT_PROTOCOL = expect_protocol
        _W_MANIFEST_HASH = manifest_hash
    except BaseException as e:
        import traceback
        _W_INIT_ERROR = (
            f"init_worker failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def _worker_process_cell(task):
    """Pool worker entrypoint: dispatch to _compute_cell with worker-local globals.

    Checks _W_INIT_ERROR first - if init_worker failed, raise on first task to
    propagate cleanly through pool.imap_unordered.
    """
    if _W_INIT_ERROR is not None:
        raise RuntimeError(_W_INIT_ERROR)
    model_name, op_name, sev = task
    return _compute_cell(
        model_name, op_name, sev,
        _W_CACHE, _W_PRED_DIR, _W_OUT_DIR, _W_DECOMP, _W_NAME_TO_CONCAT,
        log_progress=False,
        expect_protocol=_W_EXPECT_PROTOCOL,
        manifest_hash=_W_MANIFEST_HASH,
    )


def _compute_cell(model_name: str, op_name: str, sev: int,
                  cache: list, pred_dir: Path, out_dir: Path,
                  decomp_dict: dict, name_to_concat: dict,
                  log_progress: bool = True,
                  expect_protocol: str = "",
                  manifest_hash: str = None) -> dict:
    """Process one (model, op, sev) cell: load preds .npz, verify alignment,
    compute per-sample CD/F/decomp, write per_sample.json, return aggregate.

    Used by both serial path (called inline from main) and parallel path
    (called via _worker_process_cell). Output is bit-exact identical regardless
    of caller - per-cell compute is deterministic, no RNG, no order dep.
    """
    cell_key = f"{op_name}@s{sev}"
    cell_npz = pred_dir / f"{model_name}__{op_name}__s{sev}.npz"
    if not cell_npz.is_file():
        raise RuntimeError(
            f"[{model_name}|{cell_key}] FATAL: pred file missing at {cell_npz}"
        )
    # Load preds + verify per-cell alignment
    with np.load(cell_npz, allow_pickle=False) as npz:
        preds = npz["preds"]                       # (N, N_pred, 3) float32
        sample_indices = npz["sample_indices"]     # (N,) int32
        npz_tax = npz["taxonomy_ids"]              # (N,) <U
        npz_mid = npz["model_ids"]                 # (N,) <U
        npz_view = npz["view_ids"]                 # (N,) int32
        n_pred_pts = int(npz["n_pred_points"])
        concat_applied = bool(npz["concat_partial_applied"])
        npz_schema = str(npz.get("schema_version", "?"))
        npz_model = str(npz["model_name"])
        npz_op = str(npz["op"])
        npz_sev = int(npz["severity"])
        npz_protocol = str(npz["protocol"]) if "protocol" in npz.files else ""
        npz_has_stamp = "cache_hash_sha256" in npz.files
        npz_cache_hash = str(npz["cache_hash_sha256"]) if npz_has_stamp else ""
    # Per-NPZ provenance guard: with protocol-shared filenames, a stale archive
    # from another protocol or another cache passes every metadata/identity check
    # below. Under --expect-protocol the protocol and cache_hash stamps are mandatory;
    # without it a cache_hash stamp is still checked whenever the field is present, even
    # if empty (legacy zero-pad archives without the field are covered by the verified
    # manifest).
    if expect_protocol and npz_protocol != expect_protocol:
        raise RuntimeError(
            f"[{model_name}|{cell_key}] FATAL protocol stamp mismatch: "
            f"npz='{npz_protocol}' vs expected='{expect_protocol}' - stale archive "
            f"from another protocol? Refusing."
        )
    if manifest_hash and (expect_protocol or npz_has_stamp) and npz_cache_hash != manifest_hash:
        raise RuntimeError(
            f"[{model_name}|{cell_key}] FATAL per-NPZ cache_hash mismatch: "
            f"npz={npz_cache_hash[:16]}... vs manifest={manifest_hash[:16]}... - "
            f"archive was forwarded against a different cache. Refusing."
        )
    # Schema check
    if npz_schema != NPZ_SCHEMA_VERSION:
        raise RuntimeError(
            f"[{model_name}|{cell_key}] FATAL schema mismatch: "
            f"npz={npz_schema} vs current={NPZ_SCHEMA_VERSION}"
        )
    # Per-cell metadata match
    if npz_model != model_name or npz_op != op_name or npz_sev != sev:
        raise RuntimeError(
            f"[{model_name}|{cell_key}] FATAL npz metadata mismatch: "
            f"npz=({npz_model},{npz_op},{npz_sev}) vs "
            f"expected=({model_name},{op_name},{sev})"
        )
    # Shape + n_pred_points (must be 16384 for both PoinTr and AdaPoinTr).
    # PoinTr: 16384 comes from internal concat at models/PoinTr.py:119 (14336 fold + 2048
    # input partial); external concat_partial=False to avoid double-concat (registry sets it).
    # AdaPoinTr: 16384 directly from forward dense output, no concat needed.
    if n_pred_pts != 16384:
        raise RuntimeError(
            f"[{model_name}|{cell_key}] FATAL n_pred_points={n_pred_pts}, expected 16384 "
            f"(PoinTr internal concat=14336 fold + 2048 input partial; AdaPoinTr=16384 direct). "
            f"concat_partial misconfig?"
        )
    if preds.shape != (len(cache), 16384, 3):
        raise RuntimeError(
            f"[{model_name}|{cell_key}] FATAL preds shape {preds.shape}, "
            f"expected ({len(cache)}, 16384, 3)"
        )
    # concat_partial_applied vs DEFAULT_MODELS spec
    expected_concat = name_to_concat.get(model_name, False)
    if concat_applied != expected_concat:
        raise RuntimeError(
            f"[{model_name}|{cell_key}] FATAL concat_partial_applied={concat_applied}, "
            f"expected {expected_concat} per DEFAULT_MODELS[{model_name}]['concat_partial']"
        )
    # Per-sample identity arrays match cache
    cache_idx = np.array([e["idx"] for e in cache], dtype=np.int32)
    cache_tax = np.array([e["taxonomy_id"] for e in cache])
    cache_mid = np.array([e["model_id"] for e in cache])
    cache_view = np.array([e["view_id"] for e in cache], dtype=np.int32)
    if not np.array_equal(sample_indices, cache_idx):
        raise RuntimeError(
            f"[{model_name}|{cell_key}] FATAL sample_indices mismatch with cache"
        )
    if not np.array_equal(npz_tax, cache_tax):
        raise RuntimeError(f"[{model_name}|{cell_key}] FATAL taxonomy_ids mismatch")
    if not np.array_equal(npz_mid, cache_mid):
        raise RuntimeError(f"[{model_name}|{cell_key}] FATAL model_ids mismatch")
    if not np.array_equal(npz_view, cache_view):
        raise RuntimeError(f"[{model_name}|{cell_key}] FATAL view_ids mismatch")
    # Per-sample metric loop
    t_cell = time.time()
    per_sample = []
    for i, entry in enumerate(cache):
        if int(sample_indices[i]) != entry["idx"]:
            raise RuntimeError(
                f"[{model_name}|{cell_key}] FATAL: sample {i} idx mismatch "
                f"(npz {sample_indices[i]} vs cache {entry['idx']})"
            )
        pred_np = preds[i]      # (N_pred, 3)
        gt_np = entry["gt_np"]  # (16384, 3)
        # Per-sample zero-row count, so that the metric stage can audit whether a
        # model propagates (0,0,0) rows from the PCN zero-pad partial into its output.
        # CD-L1 is masked by ignore_zeros=True; the decomposition metrics are NOT.
        n_pred_zero_rows = int(np.all(pred_np == 0.0, axis=-1).sum())
        # The chamfer mask criterion is sum(coords)==0 (broader than exact (0,0,0));
        # record the ACTUAL row-drop counts so that masking activity is audited,
        # not assumed.
        n_pred_sumzero_rows = int((pred_np.sum(axis=-1) == 0.0).sum())
        n_gt_sumzero_rows = int((gt_np.sum(axis=-1) == 0.0).sum())
        cd1 = chamfer_l1(pred_np, gt_np)
        cd2 = chamfer_l2(pred_np, gt_np)
        fscore = fscore_at_thresholds(pred_np, gt_np)
        decomp = {f"decomp_{dn}": dm(pred_np, gt_np)
                  for dn, dm in decomp_dict.items()}
        if op_name.startswith("mixed_"):
            # No single-op seed exists for a composed cell: make_seed(..., op_name)
            # would fabricate a seed the forward
            # stage never used. Per-sample constituent seeds live in the forward
            # NPZ (constituent_seeds); rows carry None to keep provenance honest.
            corruption_seed = None
        else:
            corruption_seed = int(make_seed(entry["taxonomy_id"], entry["model_id"],
                                            entry["view_id"], sev, op_name))
        per_sample.append({
            "idx": entry["idx"],
            "model_name": model_name,
            "taxonomy_id": entry["taxonomy_id"],
            "model_id": entry["model_id"],
            "view_id": entry["view_id"],
            "op": op_name,
            "severity": sev,
            "cell_key": cell_key,
            "corruption_seed": corruption_seed,
            "n_pred_points": n_pred_pts,
            "n_gt_points": int(gt_np.shape[0]),
            "n_pred_zero_rows": n_pred_zero_rows,
            "n_pred_sumzero_rows": n_pred_sumzero_rows,
            "n_gt_sumzero_rows": n_gt_sumzero_rows,
            "cd_l1": cd1,
            "cd_l2": cd2,
            **fscore,
            **decomp,
        })
        if log_progress and (i + 1) % 100 == 0:
            elapsed = time.time() - t_cell
            mean_cd1 = np.mean([s["cd_l1"] for s in per_sample])
            print(
                f"    [{model_name}|{cell_key}] {i+1}, "
                f"mean CDL1={mean_cd1*1000:.3f} (x1000), elapsed {elapsed:.1f}s",
                flush=True,
            )
    # Save per-sample JSON (NaN -> None for strict JSON)
    cell_path = out_dir / f"{model_name}_{op_name}_s{sev}_per_sample.json"
    if expect_protocol:
        # Stamp rows so --resume-existing can re-verify provenance (legacy runs
        # without --expect-protocol keep their original row schema untouched).
        # cache_hash is stamped per row too, so resume can reject wrong-cache
        # JSONs, not just wrong-protocol ones.
        for row in per_sample:
            row["protocol"] = expect_protocol
            row["cache_hash_sha256"] = manifest_hash
    clean_per_sample = [
        {k: (None if isinstance(v, float) and np.isnan(v) else v)
         for k, v in row.items()}
        for row in per_sample
    ]
    with open(cell_path, "w") as f:
        json.dump(clean_per_sample, f, indent=2, allow_nan=False)
    agg = aggregate_cell(per_sample)
    return {
        "model_name": model_name,
        "op_name": op_name,
        "sev": sev,
        "cell_key": cell_key,
        "aggregate": agg,
        "per_sample_path": str(cell_path),
        "elapsed_sec": time.time() - t_cell,
    }


def _resume_rows_valid(rows, model_name, op_name, sev, clean_cache,
                       expect_protocol, manifest_hash):
    """Full-row provenance check for --resume-existing under a protocol guard:
    reject truncated/smoke JSONs, wrong-protocol,
    wrong-cache, wrong-cell, and identity-mismatched files instead of trusting
    the first row. Returns True only if the JSON is exactly the requested cell
    computed against exactly this cache under this protocol."""
    if len(rows) != len(clean_cache):
        return False
    for r, e in zip(rows, clean_cache):
        if r.get("protocol") != expect_protocol:
            return False
        if r.get("cache_hash_sha256") != manifest_hash:
            return False
        if (r.get("idx") != e["idx"] or r.get("model_name") != model_name
                or r.get("op") != op_name or r.get("severity") != sev):
            return False
        if (r.get("taxonomy_id") != e["taxonomy_id"]
                or r.get("model_id") != e["model_id"]
                or r.get("view_id") != e["view_id"]):
            return False
    return True


def main():
    args = parse_args()
    np.random.seed(args.seed)  # for reproducibility, though metric stage doesn't use random
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse selections
    sel_models = [m.strip() for m in args.models.split(",") if m.strip()]
    sel_ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    sel_sevs = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
    if "pose" in sel_ops:
        sys.exit("[metric] 'pose' is not part of the audit grid.")
    allowed_ops = set(DEFAULT_OPS)
    if args.allow_validpt_ops:
        allowed_ops |= {f"{o}_validpt" for o in DEFAULT_OPS}
    if args.allow_clean_cell:
        allowed_ops |= {"clean"}
    if args.allow_mixed_ops:
        # Keep in sync with forward_sweep_composed.MIXED_PAIRS (not imported:
        # that module imports torch, which the CPU metric env may not have).
        allowed_ops |= {"mixed_noise_outlier", "mixed_crop_noise",
                        "mixed_density_outlier"}
    for o in sel_ops:
        if o not in allowed_ops:
            sys.exit(f"[metric] op '{o}' not in scope {sorted(allowed_ops)} "
                     f"(use --allow-validpt-ops for the valid-point-only ops, "
                     f"--allow-clean-cell for the UpSamplePoints clean@s0 cell, "
                     f"--allow-mixed-ops for the composed-operator cells)")
    allowed_sevs = (0, 1, 2, 3, 4, 5) if args.allow_clean_cell else (1, 2, 3, 4, 5)
    for s in sel_sevs:
        if s not in allowed_sevs:
            sys.exit(f"[metric] severity {s} out of {list(allowed_sevs)}")
    if args.allow_clean_cell:
        if "clean" in sel_ops and 0 not in sel_sevs:
            sys.exit("[metric] --allow-clean-cell with op 'clean' requires severity 0 "
                     "in --severities")
        if 0 in sel_sevs and "clean" not in sel_ops:
            sys.exit("[metric] severity 0 is only valid for the 'clean' op")
    if args.workers < 1:
        sys.exit(f"[metric] --workers must be >= 1 (got {args.workers})")
    # Reject duplicate selections: workers writing to the same per_sample.json path
    # would race (no shared lock, last writer wins).
    if len(set(sel_models)) != len(sel_models):
        sys.exit(f"[metric] duplicate models in --models: {sel_models}")
    if len(set(sel_ops)) != len(sel_ops):
        sys.exit(f"[metric] duplicate ops in --ops: {sel_ops}")
    if len(set(sel_sevs)) != len(sel_sevs):
        sys.exit(f"[metric] duplicate severities in --severities: {sel_sevs}")

    # Load + verify the forward manifest (configurable for the protocol variants)
    manifest_path = pred_dir / args.manifest_name
    if not manifest_path.is_file():
        sys.exit(f"[metric] FATAL: forward manifest not found at {manifest_path}. "
                 f"Run the matching forward script first, or point --manifest-name (a metric "
                 f"option, currently '{args.manifest_name}') at the manifest it wrote.")
    with open(manifest_path) as f:
        forward_manifest = json.load(f)
    verify_alignment(forward_manifest, args, sel_models, sel_ops, sel_sevs)
    print(f"[metric] forward manifest aligned: "
          f"max_samples_per_cell={args.max_samples_per_cell}, seed={args.seed}, "
          f"n_cached={forward_manifest['n_cached_samples']}, "
          f"schema={forward_manifest.get('schema_version', '?')}")

    # The per-NPZ provenance guard must not depend on the operator remembering
    # --expect-protocol. If the forward manifest declares a
    # protocol, enforce it (auto-fill when flag omitted; fail on disagreement).
    fm_protocol = forward_manifest.get("protocol", "")
    if fm_protocol:
        if args.expect_protocol and args.expect_protocol != fm_protocol:
            sys.exit(f"[metric] FATAL: --expect-protocol '{args.expect_protocol}' "
                     f"disagrees with forward manifest protocol '{fm_protocol}'")
        if not args.expect_protocol:
            args.expect_protocol = fm_protocol
            print(f"[metric] per-NPZ protocol guard auto-enabled from manifest: "
                  f"'{fm_protocol}'")
    elif args.expect_protocol:
        sys.exit(f"[metric] FATAL: --expect-protocol '{args.expect_protocol}' requested "
                 f"but forward manifest declares no protocol (legacy zero-pad manifest); "
                 f"its NPZs carry no stamps and every cell would fail. Drop the flag or "
                 f"point at the correct pred dir.")

    # Re-cache PCN clean partials + GT (same seed -> bit-identical to forward stage)
    print(f"[metric] Loading PCN test split from {args.pcn_data_root}"
          + (f" with loader_cfg={args.loader_cfg}" if args.loader_cfg else ""))
    loader = load_pcn_test_loader(args.pcn_data_root, args.batch_size,
                                  cfg_yaml=args.loader_cfg)
    clean_cache = cache_clean_pcn_test(loader, args.max_samples_per_cell, args.seed)

    # Verify cache alignment via taxonomy_id + model_id sequence
    fm_idx = forward_manifest["cached_sample_indices"]
    fm_tax = forward_manifest["cached_taxonomy_ids"]
    fm_mid = forward_manifest["cached_model_ids"]
    if len(clean_cache) != len(fm_idx):
        sys.exit(f"[metric] FATAL: re-cached {len(clean_cache)} samples vs forward stage {len(fm_idx)}")
    for i, e in enumerate(clean_cache):
        if e["idx"] != fm_idx[i] or e["taxonomy_id"] != fm_tax[i] or e["model_id"] != fm_mid[i]:
            sys.exit(f"[metric] FATAL alignment mismatch at sample {i}: "
                     f"forward (idx={fm_idx[i]}, tax={fm_tax[i]}, mid={fm_mid[i]}) "
                     f"vs metric (idx={e['idx']}, tax={e['taxonomy_id']}, mid={e['model_id']})")

    # bit-exact cache hash verification (catches PCN-file/transform drift)
    fm_hash = forward_manifest.get("cache_hash_sha256")
    if fm_hash is None:
        sys.exit("[metric] FATAL: forward manifest missing cache_hash_sha256 (re-run forward stage "
                 "with current schema_version >= 1.0).")
    metric_hash = compute_cache_hash(clean_cache)
    if metric_hash != fm_hash:
        sys.exit(f"[metric] FATAL cache_hash_sha256 mismatch:\n"
                 f"  forward: {fm_hash}\n"
                 f"  metric:  {metric_hash}\n"
                 f"  PCN data files / transforms / RNG seeded incorrectly. Re-verify both stages.")
    print(f"[metric] PCN cache bit-exact verified (sha256 {metric_hash[:16]}..., "
          f"{len(clean_cache)} samples)")

    # Build per-model concat_partial spec dict for per-cell validation
    name_to_concat = {ms["name"]: bool(ms.get("concat_partial", False)) for ms in DEFAULT_MODELS}

    # Instantiate decomposition metrics once (used in serial path; parallel path inits per worker)
    decomp_dict = {n: cls() for n, cls in DECOMPOSITION_METRICS.items()}

    grand_t0 = time.time()
    all_results = {m: {"status": "done", "cells": {}} for m in sel_models}
    # The clean@s0 cell breaks the cartesian-product assumption: 'clean' pairs ONLY
    # with severity 0, corruption ops ONLY with 1-5. Filter invalid combos.
    tasks = [
        (m, o, s) for m in sel_models for o in sel_ops for s in sel_sevs
        if (o == "clean") == (s == 0)
    ]
    n_cells = len(tasks)

    def _record(r):
        all_results[r["model_name"]]["cells"][r["cell_key"]] = {
            "aggregate": r["aggregate"],
            "per_sample_path": r["per_sample_path"],
        }
        agg = r["aggregate"]
        print(
            f"  [{r['model_name']}|{r['cell_key']}] DONE: n={agg['n_samples']}, "
            f"CDL1x1000={agg.get('cd_l1_x1000', float('nan')):.3f}, "
            f"F@0.01={agg.get('f_at_0.01_mean', float('nan')):.3f}, "
            f"cell_elapsed={r['elapsed_sec']/60:.1f} min",
            flush=True,
        )

    if args.resume_existing:
        pending_tasks = []
        for model_name, op_name, sev in tasks:
            cell_key = f"{op_name}@s{sev}"
            cell_path = out_dir / f"{model_name}_{op_name}_s{sev}_per_sample.json"
            if cell_path.is_file():
                with open(cell_path) as f:
                    per_sample = json.load(f)
                # A stale, truncated, wrong-cache or wrong-cell per-sample JSON must
                # NOT be silently aggregated. Under a
                # protocol guard, EVERY row must match protocol + cache_hash + cell +
                # cache identity; otherwise recompute the cell.
                if args.expect_protocol and not _resume_rows_valid(
                        per_sample, model_name, op_name, sev, clean_cache,
                        args.expect_protocol, fm_hash):
                    print(f"[metric] resume-existing: {cell_path.name} failed full-row "
                          f"provenance check (protocol/cache_hash/cell/identity/length) "
                          f"-> recomputing cell", flush=True)
                    pending_tasks.append((model_name, op_name, sev))
                    continue
                _record({
                    "model_name": model_name,
                    "cell_key": cell_key,
                    "aggregate": aggregate_cell(per_sample),
                    "per_sample_path": str(cell_path),
                    "elapsed_sec": 0.0,
                })
            else:
                pending_tasks.append((model_name, op_name, sev))
        skipped = n_cells - len(pending_tasks)
        if skipped:
            print(f"[metric] resume-existing: skipped {skipped}/{n_cells} existing cells", flush=True)
        tasks = pending_tasks

    if args.workers <= 1:
        # Serial path: process cells in (model, op, sev) order via _compute_cell
        prev_model = None
        for (model_name, op_name, sev) in tasks:
            if model_name != prev_model:
                print(f"\n[metric] === {model_name} ===", flush=True)
                prev_model = model_name
            r = _compute_cell(
                model_name, op_name, sev,
                clean_cache, pred_dir, out_dir, decomp_dict, name_to_concat,
                log_progress=True,
                expect_protocol=args.expect_protocol,
                manifest_hash=fm_hash,
            )
            _record(r)
    else:
        # Parallel path: cell-level Pool with spawn (Windows-safe), threadpool-limited workers.
        # Output bit-exact identical to serial: per-cell compute is deterministic + no RNG / order dep.
        cache_pkl = out_dir / "_clean_cache.pkl"
        print(f"[metric] Pickling cache to {cache_pkl} for worker init "
              f"({len(clean_cache)} samples) ...", flush=True)
        with open(cache_pkl, "wb") as f:
            pickle.dump(clean_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        cache_pkl_size_mb = cache_pkl.stat().st_size / (1024 * 1024)
        print(f"[metric] Cache pickled: {cache_pkl_size_mb:.0f} MB. "
              f"Spawning {args.workers} workers for {n_cells} cells "
              f"(BLAS threads limited to 1 per worker) ...", flush=True)

        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(
                processes=args.workers,
                initializer=init_worker,
                initargs=(str(cache_pkl), str(pred_dir), str(out_dir), name_to_concat,
                          args.expect_protocol, fm_hash),
            ) as pool:
                done = 0
                for r in pool.imap_unordered(_worker_process_cell, tasks):
                    _record(r)
                    done += 1
                    print(f"[metric] progress: {done}/{n_cells} cells "
                          f"({done/n_cells*100:.0f}%)", flush=True)
        finally:
            try:
                cache_pkl.unlink()
            except OSError:
                pass

    summary_path = out_dir / "metric_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_id": "metric_stage",
            "stage": "metric-pipeline",
            "args": vars(args),
            "elapsed_total_sec": time.time() - grand_t0,
            "results": all_results,
        }, f, indent=2)
    print(f"\n[metric] Total metric elapsed: {(time.time()-grand_t0)/60:.1f} min")
    print(f"[metric] Summary written to {summary_path}")


if __name__ == "__main__":
    mp.freeze_support()  # Windows safety for spawn-based Pool
    main()

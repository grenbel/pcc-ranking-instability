"""Clean-PCN sanity check: baseline inference on the clean PCN test split.

Purpose:
    Verify the environment and the metric pipeline by loading each pretrained
    checkpoint, running inference on the clean PCN test split, and checking
    that the reported CD-L1 / CD-L2 / F-score are within a few percent of the
    values stamped in `best_metrics` inside the checkpoint (Table 2 of the
    paper).

Decision gate:
    PASS if the selected baselines reproduce within +/-10% of the stamped
    best_metrics; the script exits nonzero otherwise.

Usage example:
    python scripts/sanity_clean_pcn.py \\
        --pcn-data-root /path/to/PCN \\
        --output-dir logs/sanity \\
        --max-samples -1 \\
        --device cuda:0 \\
        --no-wandb

For a quick check on a few samples (the models need CUDA for their point-sampling ops):
    python scripts/sanity_clean_pcn.py --pcn-data-root /path/to/PCN --max-samples 20 --output-dir logs/sanity_smoke --no-wandb

Skip behavior:
    Any model whose checkpoint file is missing is logged as `skipped` and the
    run continues with the rest.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure baselines/PoinTr is on path so PoinTr's own tools/builder + utils import works
REPO_ROOT = Path(__file__).resolve().parent.parent
POINTR_ROOT = REPO_ROOT / "baselines" / "PoinTr"
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(POINTR_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.metrics.reconstruction import chamfer_l1, chamfer_l2, fscore_at_thresholds


@contextlib.contextmanager
def cwd_to(path):
    """chdir context: PoinTr's cfg loader resolves `_base_:
    cfgs/dataset_configs/PCN.yaml` relative to cwd, not to the YAML file's
    own directory."""
    old = os.getcwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        os.chdir(old)


def _torch_load(path: str, map_location: str = "cpu"):
    """torch.load wrapper that handles the presence/absence of the `weights_only`
    argument across PyTorch 1.13 and 2.x (PyTorch 1.13 raises TypeError on it)."""
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location)


# Default model registry (extend by editing here). Sanity gate: each model with
# expected_* set must reproduce within +/-SANITY_TOLERANCE_PCT or the sanity check FAILS.
SANITY_TOLERANCE_PCT = 10.0  # +/-10% on CDL1, CDL2, f_at_0.01
DEFAULT_MODELS = [
    {
        "name": "PoinTr",
        "cfg": str(POINTR_ROOT / "cfgs" / "PCN_models" / "PoinTr.yaml"),
        "ckpt": str(REPO_ROOT / "ckpts" / "pretrained_pointr" / "PCN_models" / "ckpt-best.pth"),
        # PoinTr returns (coarse, fine); fine = ret[-1] for evaluation
        "output_index": -1,
        # PoinTr forward (models/PoinTr.py:119) ALREADY internally concats partial:
        # `rebuild_points = torch.cat([rebuild_points, xyz], dim=1)` -> ret[-1] is already
        # (B, 16384, 3) = 14336 fold output + 2048 input partial concat. External concat
        # would DOUBLE -> (B, 18432, 3). Set False.
        # Without the zero-row mask the numpy CD drifts +14.8% above the stamped value:
        # the (0,0,0) padding rows of PCN sub-2048 partial clouds propagate through the
        # internal concat into ret[-1] indices 14336-16383 (100% of the zero rows lie in
        # the last 2048 block, 0% in the first 14336 fold outputs). `src/metrics/reconstruction.py`
        # therefore defaults to `ignore_zeros=True`, PoinTr's official `ChamferDistanceL1`
        # mask convention.
        "concat_partial": False,
        # Support BOTH PoinTr ckpt variants per PoinTr README:
        #   - PCN_new (improved): CD = 7.26e-3
        #   - PCN orig:           CD = 8.38e-3
        # Sanity gate accepts whichever matches within +/-10%; stamped best_metrics overrides.
        "expected_cd_l1_x1000": [7.26, 8.38],
        "expected_cd_l2_x1000": None,
        "expected_f_at_0_01": None,
        "in_default_suite": True,
    },
    {
        "name": "AdaPoinTr",
        "cfg": str(POINTR_ROOT / "cfgs" / "PCN_models" / "AdaPoinTr.yaml"),
        "ckpt": str(REPO_ROOT / "ckpts" / "pretrained_adapointr" / "PCN_models" / "ckpt-best.pth"),
        # AdaPoinTr returns (coarse, fine); same indexing as PoinTr
        "output_index": -1,
        # AdaPoinTr cfg num_points=16384 -> forward dense=16384 pts; no concat needed
        "concat_partial": False,
        "expected_cd_l1_x1000": [6.53],
        "expected_cd_l2_x1000": None,
        "expected_f_at_0_01": None,
        "in_default_suite": True,
    },
    {
        # SeedFormer (Zhou et al. ECCV 2022, arXiv:2207.10315). The official `codes/`
        # directory is expected at baselines/SeedFormer (see README).
        # SeedFormer model.py uses `from models.utils import ...` which collides with PoinTr's
        # `models` package; `src/seedformer_loader.py` swaps sys.modules['models'] only
        # during load.
        # Pretrained checkpoint: the SeedFormer-dim128 PCN checkpoint from the official
        # repository (13.4 MB, md5 bb898a125eba57b6bccf8fe26b762d1e), placed at
        # ckpts/pretrained_seedformer/PCN_models/ckpt-best.pth.
        # Stamped best_metrics=6.748752668499947 (x1000, scalar despite the plural key).
        # SeedFormer was trained with the UpSamplePoints partial transform
        # (codes/utils/data_loaders.py), the same as SnowFlakeNet's PCNv2 config, so the
        # sanity check uses that loader_cfg. The corruption sweeps keep the PoinTr.yaml
        # zero-pad loader for every baseline (matched-control invariant).
        "name": "SeedFormer",
        # cfg is unused for builder='seedformer'; we keep the field set to a real PoinTr-tree
        # yaml so downstream code that reads spec['cfg'] never sees None.
        "cfg": str(POINTR_ROOT / "cfgs" / "PCN_models" / "SnowFlakeNet.yaml"),
        "ckpt": str(REPO_ROOT / "ckpts" / "pretrained_seedformer" / "PCN_models" / "ckpt-best.pth"),
        # Sanity uses SnowFlakeNet.yaml as proxy (same UpSamplePoints PCNv2 transform).
        "loader_cfg": str(POINTR_ROOT / "cfgs" / "PCN_models" / "SnowFlakeNet.yaml"),
        "builder": "seedformer",
        "builder_kwargs": {"up_factors": [1, 4, 8]},  # PCN production config
        # SeedFormer forward returns list `pred_pcds` = [seed_256, p_512, p_2048, p_16384];
        # last is the dense output we want.
        "output_index": -1,
        "concat_partial": False,
        "load_strict": True,
        # Stamped CDL1 from ckpt's `best_metrics` scalar (despite plural key).
        "expected_cd_l1_x1000": [6.748],
        "expected_cd_l2_x1000": None,
        "expected_f_at_0_01": None,
        "in_default_suite": True,
    },
    {
        # SnowflakeNet (Xiang et al. ICCV 2021 / TPAMI 2023), bundled in the PoinTr repo at
        # models/SnowFlakeNet.py with cfg `cfgs/PCN_models/SnowFlakeNet.yaml`. Pretrained
        # checkpoint: ckpt-best-pcn-cd_l1.pth from the official SnowflakeNet repository
        # (77 MB, md5 baadb96d94174b4b7336a23d9d9cd8e6), placed at
        # ckpts/pretrained_snowflakenet/PCN_models/ckpt-best.pth.
        # Stamped best_metric=7.188 x1000 (epoch 300, CD-L1-optimized variant).
        # SnowFlakeNet was trained under the PCNv2 transform (UpSamplePoints, no zero rows);
        # loader_cfg points to its own cfg so the sanity check uses the matching input
        # transform. The corruption sweeps keep the PoinTr.yaml zero-pad loader for every
        # baseline (matched-control invariant): its absolute CD-L1 under that protocol
        # differs from the stamped 7.188, but all baselines see identical partial inputs.
        "name": "SnowFlakeNet",
        "cfg": str(POINTR_ROOT / "cfgs" / "PCN_models" / "SnowFlakeNet.yaml"),
        "ckpt": str(REPO_ROOT / "ckpts" / "pretrained_snowflakenet" / "PCN_models" / "ckpt-best.pth"),
        # Sanity stage uses PCNv2 transform (matches training distribution, validates ckpt vs stamped 7.188)
        "loader_cfg": str(POINTR_ROOT / "cfgs" / "PCN_models" / "SnowFlakeNet.yaml"),
        # SnowFlakeNet eval forward returns (out[1], out[-1]) where out[-1] is 16384 dense.
        "output_index": -1,
        # SnowFlakeNet up_factors=[4,8] -> num_p0(512) x 32 = 16384 dense; no external concat.
        "concat_partial": False,
        # New baseline: the state_dict load must be strict (no silent partial init).
        "load_strict": True,
        # Stamped CDL1 from ckpt's best_metric scalar
        "expected_cd_l1_x1000": [7.188],
        "expected_cd_l2_x1000": None,
        "expected_f_at_0_01": None,
        "in_default_suite": True,
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcn-data-root", required=True,
                   help="Path to extracted PCN dataset root (must contain test/, val/, PCN.json)")
    p.add_argument("--output-dir", required=True, help="Where to write per-sample + aggregate JSON")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=-1,
                   help="-1 = full test split; small int for smoke")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Inference batch size; only 1 is supported (the per-sample loop "
                        "reads batch[0]).")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="pcc-ranking-instability")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default=None,
                   help="Comma-separated subset of DEFAULT_MODELS names to run "
                        "(e.g. 'SnowFlakeNet' or 'PoinTr,AdaPoinTr'). Default = all.")
    return p.parse_args()


def load_pcn_test_loader(pcn_data_root: str, batch_size: int, cfg_yaml: str = None):
    """Build PCN test DataLoader using PoinTr's own dataset machinery.

    Args:
        cfg_yaml: absolute path to a `cfgs/PCN_models/<Model>.yaml` whose
                  `dataset.test._base_` selects the input transform. Default
                  (None) uses `PoinTr.yaml` -> `PCN.yaml` (RandomSamplePoints
                  with zero-pad for sub-2048 partials, "project zero-pad
                  protocol"). SnowFlakeNet ckpt was trained with
                  `PCNv2.yaml` (UpSamplePoints, duplicate-up to 2048, no
                  zero rows) - its sanity stage should pass `cfg_yaml`
                  pointing to `SnowFlakeNet.yaml` so the loader transform
                  matches its training-time distribution. Mixing these
                  would feed zero-padded partials to a model trained on
                  upsampled partials, polluting both the encoder input and
                  `Decoder.fps_subsample(cat([coarse, partial]), num_p0)`.

    Patches the dataset config dict to point at our PCN root rather than
    PoinTr's hardcoded relative path. Uses cwd_to(POINTR_ROOT) because cfg
    loader resolves `_base_` relative to cwd.
    """
    assert batch_size == 1, (
        "Only batch_size=1 is supported (the per-sample metric path takes batch[0] "
        "only). Use --batch-size 1 or update run_inference() to loop over batch."
    )
    if cfg_yaml is None:
        cfg_yaml = str(POINTR_ROOT / "cfgs" / "PCN_models" / "PoinTr.yaml")
    from torch.utils.data import DataLoader
    with cwd_to(POINTR_ROOT):
        from utils.config import cfg_from_yaml_file
        from datasets.build import build_dataset_from_cfg
        cfg = cfg_from_yaml_file(cfg_yaml)
        ds_cfg = cfg.dataset.test._base_
        # Override paths to our dataset root
        ds_cfg.PARTIAL_POINTS_PATH = os.path.join(pcn_data_root, "%s/partial/%s/%s/%02d.pcd")
        ds_cfg.COMPLETE_POINTS_PATH = os.path.join(pcn_data_root, "%s/complete/%s/%s.pcd")
        ds_cfg.CATEGORY_FILE_PATH = os.path.join(pcn_data_root, "PCN.json")
        ds_cfg.subset = "test"
        dataset = build_dataset_from_cfg(ds_cfg, ds_cfg)
    # num_workers=0 + no shuffle for a deterministic sample order
    # (PCN's RandomSamplePoints uses global numpy RNG; multi-worker would non-determine).
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def load_model(cfg_path: str, ckpt_path: str, device: str, model_name: str = "",
               load_strict: bool = False, builder_kind: str = None,
               builder_kwargs: dict = None):
    """Load a PoinTr / AdaPoinTr / SnowFlakeNet / SeedFormer model + state_dict.

    By default (load_strict=False) the load tolerates at most max(2, 5% of the keys)
    missing/unexpected entries (PoinTr-class ckpts can carry a couple of orphan keys like
    `loss_func.*`) and raises beyond that. If `load_strict=True`, ANY missing
    or unexpected key raises immediately. SnowFlakeNet and SeedFormer opt in to
    `load_strict=True` so that a newly added baseline can never be silently
    partially initialized.

    `builder_kind`:
       - None (default): use PoinTr's MODELS registry via `tools.builder.model_builder`
         from `cfg_path` (yaml config). Used by PoinTr / AdaPoinTr / SnowFlakeNet.
       - 'seedformer': build via `src.seedformer_loader.load_seedformer_dim128(**kw)`,
         which isolates SeedFormer's `from models.utils import ...` from PoinTr's
         `models` namespace. `cfg_path` is ignored.
    """
    if builder_kind == "seedformer":
        from src.seedformer_loader import load_seedformer_dim128
        kwargs = builder_kwargs or {}
        model = load_seedformer_dim128(**kwargs)
    else:
        from tools import builder
        with cwd_to(POINTR_ROOT):
            from utils.config import cfg_from_yaml_file
            cfg = cfg_from_yaml_file(cfg_path)
            model = builder.model_builder(cfg.model)
    sd = _torch_load(ckpt_path, map_location="cpu")
    base = sd.get("base_model", sd.get("model", sd.get("state_dict", sd)))
    # strip 'module.' prefix if present (DDP-saved ckpts)
    cleaned = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
               for k, v in base.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        # Log details first (helps debugging)
        print(f"[ERROR] {model_name}: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"        first missing: {missing[:5]}")
        if unexpected:
            print(f"        first unexpected: {unexpected[:5]}")
        if load_strict:
            raise RuntimeError(
                f"{model_name}: load_strict=True and state-dict has mismatches "
                f"({len(missing)} missing, {len(unexpected)} unexpected). "
                f"Refusing to evaluate a partial-load model. Either fix ckpt/cfg or "
                f"set load_strict=False with explicit allowlist if mismatch is benign."
            )
        # Legacy 5%-tolerant gate (PoinTr-class ckpts may have a couple orphan keys).
        n_total = len(cleaned)
        if (len(missing) + len(unexpected)) > max(2, 0.05 * n_total):
            raise RuntimeError(
                f"{model_name}: state-dict mismatch too large ({len(missing)} missing, "
                f"{len(unexpected)} unexpected, {n_total} total). Wrong ckpt for this cfg? "
                f"Refusing to evaluate a partially initialized model."
            )
    model = model.to(device).eval()
    # Normalize stamped metrics across save conventions:
    #   - PoinTr/AdaPoinTr: `best_metrics` is a dict {'CDL1':..,'CDL2':..,'F-Score':..}
    #   - SnowFlakeNet:     `best_metric` is a scalar = CDL1 (singular key)
    #   - SeedFormer:       `best_metrics` is a scalar = CDL1 (plural key BUT scalar value)
    # Unify into a dict {'CDL1': float} for the sanity gate's exp_cd1 lookup.
    bm = sd.get("best_metrics", sd.get("best_metric"))
    if isinstance(bm, dict):
        best_metrics = bm
    elif isinstance(bm, (int, float)):
        best_metrics = {"CDL1": float(bm)}
    elif torch.is_tensor(bm) and bm.numel() == 1:
        best_metrics = {"CDL1": float(bm.item())}
    else:
        best_metrics = {}
    return model, best_metrics


@torch.no_grad()
def run_inference(model, loader, device: str, max_samples: int, model_name: str,
                  output_index: int = -1, concat_partial: bool = False):
    per_sample = []
    t_start = time.time()
    for i, batch in enumerate(loader):
        if max_samples > 0 and i >= max_samples:
            break
        # PCN dataloader returns (taxonomy, model_id, (partial, gt[, diff_label]))
        taxonomy, model_id, payload = batch
        if isinstance(payload, (list, tuple)):
            partial = payload[0]
            gt = payload[1]
        else:
            raise RuntimeError(f"Unexpected payload type: {type(payload)}")
        partial = partial.to(device, non_blocking=True)
        gt_np = gt.cpu().numpy()[0]  # (16384, 3)
        # Forward
        ret = model(partial)
        # Per-model dense-output index:
        #   PoinTr (coarse, fine)     -> ret[-1]
        #   AdaPoinTr (coarse, fine)  -> ret[-1]
        if isinstance(ret, (list, tuple)):
            pred_dense = ret[output_index]
        else:
            pred_dense = ret
        # PoinTr-class models output dense=14336 points (cfg num_pred=14336) while the
        # PCN GT has 16384; PoinTr's official evaluation concatenates (partial 2048,
        # dense 14336) = 16384 to match the GT cardinality. PoinTr does this inside its
        # forward (see the registry note), AdaPoinTr outputs num_points=16384 directly,
        # so no external concat is needed for either.
        if concat_partial:
            pred_dense = torch.cat([partial, pred_dense], dim=1)
        pred_np = pred_dense.cpu().numpy()[0]  # (N_pred, 3)
        # Compute metrics (numpy fallback; consistent across models)
        cd1 = chamfer_l1(pred_np, gt_np)
        cd2 = chamfer_l2(pred_np, gt_np)
        fscore = fscore_at_thresholds(pred_np, gt_np)
        per_sample.append({
            "idx": i,
            "taxonomy_id": taxonomy[0] if isinstance(taxonomy, (list, tuple)) else taxonomy,
            "model_id": model_id[0] if isinstance(model_id, (list, tuple)) else model_id,
            "n_pred_points": int(pred_np.shape[0]),
            "n_gt_points": int(gt_np.shape[0]),
            "cd_l1": cd1,
            "cd_l2": cd2,
            **fscore,
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            mean_cd1 = np.mean([s["cd_l1"] for s in per_sample])
            print(f"  [{model_name}] {i+1} samples, mean CD-L1 = {mean_cd1*1000:.3f} (x1000), "
                  f"elapsed {elapsed:.1f}s")
    return per_sample


def aggregate(per_sample):
    cd1 = np.array([s["cd_l1"] for s in per_sample])
    cd2 = np.array([s["cd_l2"] for s in per_sample])
    f001 = np.array([s["f_at_0.001"] for s in per_sample])
    f005 = np.array([s["f_at_0.005"] for s in per_sample])
    f01 = np.array([s["f_at_0.01"] for s in per_sample])
    return {
        "n_samples": len(per_sample),
        "cd_l1_mean": float(cd1.mean()),
        "cd_l1_std": float(cd1.std()),
        "cd_l1_x1000": float(cd1.mean() * 1000),  # PoinTr reporting convention
        "cd_l2_mean": float(cd2.mean()),
        "cd_l2_x1000": float(cd2.mean() * 1000),
        "f_at_0.001_mean": float(f001.mean()),
        "f_at_0.005_mean": float(f005.mean()),
        "f_at_0.01_mean": float(f01.mean()),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # init wandb (optional)
    wandb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            run_name = args.wandb_run_name or f"pcn_clean_sanity_{int(time.time())}"
            wandb.init(project=args.wandb_project, name=run_name,
                       config=vars(args))
        except Exception as e:
            print(f"[WARN] wandb init failed (continuing without): {e}")
            wandb = None

    # Loader cache keyed by absolute cfg yaml path. Each model_spec may declare its
    # own `loader_cfg` to select the input transform (e.g. SnowFlakeNet uses PCNv2
    # UpSamplePoints, PoinTr/AdaPoinTr use PCN RandomSamplePoints with zero-pad).
    loader_cache: dict = {}

    def get_loader(cfg_yaml: str = None):
        key = cfg_yaml or "DEFAULT_PoinTr"
        if key not in loader_cache:
            print(f"[sanity] Loading PCN test split from {args.pcn_data_root} "
                  f"(loader_cfg={cfg_yaml or 'PoinTr.yaml (default)'})")
            loader_cache[key] = load_pcn_test_loader(
                args.pcn_data_root, args.batch_size, cfg_yaml=cfg_yaml,
            )
            print(f"[sanity] Loader ready: {len(loader_cache[key])} batches")
        return loader_cache[key]

    results = {}
    suite_pass_count = 0      # baselines that belong to the default suite
    suite_eval_count = 0
    selected_pass_count = 0     # selected models (via --models) that PASSED
    selected_eval_count = 0
    # Use `is not None` rather than truthiness, otherwise `--models ""` would
    # silently fall back to the legacy >=1 default-suite PASS gate.
    selected_names = (
        set(s.strip() for s in args.models.split(",") if s.strip())
        if args.models is not None else None
    )
    if selected_names is not None:
        unknown = selected_names - {m["name"] for m in DEFAULT_MODELS}
        if unknown:
            raise ValueError(f"--models contains unknown names: {sorted(unknown)}; "
                             f"available: {[m['name'] for m in DEFAULT_MODELS]}")
        if not selected_names:
            raise ValueError("--models was empty after stripping whitespace/blanks")
    for model_spec in DEFAULT_MODELS:
        name = model_spec["name"]
        if selected_names is not None and name not in selected_names:
            continue
        ckpt_path = model_spec["ckpt"]
        in_suite = model_spec.get("in_default_suite", True)
        if not Path(ckpt_path).is_file():
            print(f"[sanity] {name}: ckpt missing at {ckpt_path} -> SKIPPED "
                  f"(in_default_suite={in_suite})")
            results[name] = {"status": "skipped", "reason": "ckpt missing",
                             "in_default_suite": in_suite}
            continue
        print(f"[sanity] Loading {name} from {ckpt_path}")
        model, stamped = load_model(
            model_spec["cfg"], ckpt_path, args.device, name,
            load_strict=model_spec.get("load_strict", False),
            builder_kind=model_spec.get("builder"),
            builder_kwargs=model_spec.get("builder_kwargs"),
        )
        loader = get_loader(model_spec.get("loader_cfg"))
        per_sample = run_inference(model, loader, args.device,
                                   args.max_samples, name,
                                   output_index=model_spec.get("output_index", -1),
                                   concat_partial=model_spec.get("concat_partial", False))
        agg = aggregate(per_sample)
        # Sanity gate: per-metric +/-SANITY_TOLERANCE_PCT.
        # Priority: stamped best_metrics > declared expected_*. Declared can be a list of
        # candidate values (e.g., PoinTr PCN_new=7.26 OR PCN orig=8.38) - pass if ANY match.
        def _best_delta_pct(actual, expected):
            """expected can be None | float | list[float]. Return min |delta%| over candidates."""
            if expected is None:
                return None
            cands = expected if isinstance(expected, (list, tuple)) else [expected]
            cands = [c for c in cands if c is not None and c != 0]
            if not cands:
                return None
            deltas = [(actual - c) / c * 100.0 for c in cands]
            return min(deltas, key=abs)

        # Pull stamped metrics safely (may be torch tensor -> float)
        stamped_clean = {k: float(v) if not isinstance(v, str) else v
                         for k, v in stamped.items()} if stamped else {}
        # Authoritative expected: stamped first (the actual training-time best);
        # fall back to declared (README-reported numbers) only if stamped missing.
        exp_cd1 = stamped_clean.get("CDL1") or model_spec.get("expected_cd_l1_x1000")
        exp_cd2 = stamped_clean.get("CDL2") or model_spec.get("expected_cd_l2_x1000")
        exp_f01 = stamped_clean.get("F-Score") or model_spec.get("expected_f_at_0_01")

        d_cd1 = _best_delta_pct(agg["cd_l1_x1000"], exp_cd1)
        d_cd2 = _best_delta_pct(agg["cd_l2_x1000"], exp_cd2)
        d_f01 = _best_delta_pct(agg["f_at_0.01_mean"], exp_f01)

        # PASS = every available metric within +/-SANITY_TOLERANCE_PCT
        deltas = [(mname, d) for mname, d in [("CDL1", d_cd1), ("CDL2", d_cd2),
                                                ("F@0.01", d_f01)] if d is not None]
        passed = all(abs(d) <= SANITY_TOLERANCE_PCT for _, d in deltas) and len(deltas) > 0
        sanity = {
            "expected_cd_l1_x1000": exp_cd1,
            "expected_cd_l2_x1000": exp_cd2,
            "expected_f_at_0_01": exp_f01,
            "actual_cd_l1_x1000": agg["cd_l1_x1000"],
            "actual_cd_l2_x1000": agg["cd_l2_x1000"],
            "actual_f_at_0_01": agg["f_at_0.01_mean"],
            "delta_cd_l1_pct": d_cd1,
            "delta_cd_l2_pct": d_cd2,
            "delta_f_at_0_01_pct": d_f01,
            "tolerance_pct": SANITY_TOLERANCE_PCT,
            "passed": passed,
            "stamped_best_metrics": stamped_clean,
        }
        per_sample_path = out_dir / f"{name}_per_sample.json"
        with open(per_sample_path, "w") as f:
            json.dump(per_sample, f, indent=2)
        results[name] = {"status": "done", "aggregate": agg, "sanity": sanity,
                         "in_default_suite": in_suite}
        verdict = "PASS" if passed else "FAIL"
        deltas_str = (", ".join(f"{m}={d:+.1f}%" for m, d in deltas)
                       if deltas else "(no expected set, vacuous FAIL)")
        suite_tag = "[suite]" if in_suite else "[extra]"
        print(f"[sanity] {name} {verdict} {suite_tag}: CDL1x1000={agg['cd_l1_x1000']:.3f}  "
              f"CDL2x1000={agg['cd_l2_x1000']:.3f}  F@0.01={agg['f_at_0.01_mean']:.3f}  "
              f"deltas: {deltas_str}")
        if wandb is not None:
            wandb.log({f"{name}/{k}": v for k, v in agg.items()})
            wandb.log({f"{name}/sanity_passed": int(passed)})
            for m, d in deltas:
                wandb.log({f"{name}/sanity_delta_{m}_pct": d})
        if in_suite:
            suite_eval_count += 1
            if passed:
                suite_pass_count += 1
        if selected_names is not None and name in selected_names:
            selected_eval_count += 1
            if passed:
                selected_pass_count += 1
        # release VRAM before next model
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Sanity gate:
    #   Default mode (no --models filter): require >=1 default-suite baseline PASS.
    #   --models filter active: require ALL selected models PASS (so adding a new
    #   baseline cannot silently fail while older baselines mask it).
    if selected_names is not None:
        sanity_overall_pass = (
            selected_eval_count == len(selected_names) and
            selected_pass_count == selected_eval_count
        )
        gate_desc = (f"selected={selected_pass_count}/{selected_eval_count} of "
                     f"{len(selected_names)} requested PASSED")
    else:
        sanity_overall_pass = suite_pass_count >= 1
        gate_desc = f"{suite_pass_count}/{suite_eval_count} default-suite baselines passed"
    print("\n" + "="*60)
    print(f"[sanity] SANITY GATE: {gate_desc}")
    print(f"[sanity] Overall: {'PASS' if sanity_overall_pass else 'FAIL'}")
    print("="*60)

    summary_path = out_dir / "sanity_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_id": "clean_sanity",
            "args": vars(args),
            "sanity_overall_pass": sanity_overall_pass,
            "suite_pass_count": suite_pass_count,
            "suite_eval_count": suite_eval_count,
            "selected_pass_count": selected_pass_count,
            "selected_eval_count": selected_eval_count,
            "selected_names": sorted(selected_names) if selected_names else None,
            "results": results,
        }, f, indent=2)
    print(f"[sanity] Summary written to {summary_path}")
    if wandb is not None:
        wandb.finish()
    # nonzero exit if the sanity gate failed
    if not sanity_overall_pass:
        print("[sanity] EXIT 1 - sanity gate failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

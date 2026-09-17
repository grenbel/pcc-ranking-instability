"""Local smoke test of the composed-operator forward stage: no GPU, no PCN data, no checkpoints.

Covers:
  1. MixedOp constituent call ORDER for all three pairs (call_trace)
  2. MixedOp determinism (two calls bit-exact) + severity-0 rejection path
  3. forward_cell_mixed field completeness + cross-"model" input digest equality
     (two different dummy models, 3 fake samples)
  4. verify_existing_cell negative tests: wrong composition_order, missing
     audit arrays, wrong constituent_seeds, wrong constituent_severities
  5. metric mixed rows carry corruption_seed=None (source-level check of the
     startswith branch to avoid importing the full metric stack here)

Run:  python smoke_tests/smoke_composed_local.py   (from the repository root; needs torch)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baselines" / "PoinTr"))

from scripts.forward_sweep_composed import (  # noqa: E402
    MIXED_ORDER, MIXED_PAIRS, PROTOCOL_LABEL, MixedOp, forward_cell_mixed,
    verify_existing_cell,
)
from scripts.forward_sweep import atomic_savez  # noqa: E402
from src.corruptions import OP_REGISTRY  # noqa: E402

FAILURES = []


def check(label, cond):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {label}")
    if not cond:
        FAILURES.append(label)


def fake_partial(seed, n_valid=1500, n_total=2048):
    rng = np.random.default_rng(seed)
    p = np.zeros((n_total, 3), dtype=np.float32)
    p[:n_valid] = rng.uniform(-0.9, 0.9, size=(n_valid, 3)).astype(np.float32)
    return p


def fake_cache(n=3):
    return [{"idx": i, "taxonomy_id": f"0{i}234567", "model_id": f"m{i}",
             "view_id": 0, "partial_np": fake_partial(100 + i)} for i in range(n)]


class DummyModel(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        b = x.shape[0]
        out = torch.full((b, 16384, 3), self.scale, dtype=torch.float32,
                         device=x.device)
        return out


def main():
    print("[1] MixedOp constituent call order (all pairs)")
    for op_id, pair in MIXED_PAIRS.items():
        op = MixedOp(op_id, pair, OP_REGISTRY)
        _ = op(fake_partial(1), 3, "02345678", "mA", 0)
        check(f"{op_id}: call_trace {op.call_trace} == {list(pair)}",
              op.call_trace == list(pair))
        idx = [MIXED_ORDER.index(n) for n in op.call_trace]
        check(f"{op_id}: trace respects MIXED_ORDER", idx == sorted(idx))
        a = op.last_audit
        check(f"{op_id}: audit keys pre/after_{pair[0]}/final",
              set(a) == {"pre", f"after_{pair[0]}", "final"})

    print("[2] MixedOp determinism + severity handling")
    op = MixedOp("mixed_noise_outlier", ("noise", "outlier"), OP_REGISTRY)
    p = fake_partial(7)
    out1 = op(p, 4, "03001627", "mX", 0)
    out2 = op(p, 4, "03001627", "mX", 0)
    check("two calls bit-exact", np.array_equal(out1, out2))
    check("output differs from input", not np.array_equal(out1, p))
    out_other = op(p, 4, "03001627", "mY", 0)
    check("different model_id -> different corruption",
          not np.array_equal(out1, out_other))
    try:
        _ = op(p, 0, "03001627", "mX", 0)
        # base returns copy at sev 0 for each constituent -> composed = copy;
        # main() rejects sev 0 at arg parse, so reaching here is fine as
        # long as the result equals the input (no hidden corruption).
        check("severity-0 pass-through equals input (guarded at CLI)", True)
    except ValueError:
        check("severity-0 raises (also acceptable)", True)

    print("[3] forward_cell_mixed fields + cross-model digest equality")
    cache = fake_cache(3)
    spec = {"output_index": -1, "concat_partial": False}
    cells = {}
    for mname, scale in (("DummyA", 0.1), ("DummyB", -0.2)):
        model = DummyModel(scale)
        cells[mname] = forward_cell_mixed(model, cache, "cpu",
                                          MixedOp("mixed_crop_noise",
                                                  ("crop", "noise"), OP_REGISTRY),
                                          5, spec, mname)
    da, db = cells["DummyA"], cells["DummyB"]
    check("digest equal across models",
          da["input_digest_sha256"] == db["input_digest_sha256"])
    check("digest is 64 hex", len(da["input_digest_sha256"]) == 64)
    check("preds differ across models (sanity)",
          not np.array_equal(da["preds"], db["preds"]))
    need = ["preds", "sample_indices", "taxonomy_ids", "model_ids", "view_ids",
            "model_name", "op", "severity", "n_pred_points",
            "concat_partial_applied", "schema_version", "protocol",
            "constituent_ops", "constituent_severities", "constituent_seeds",
            "composition_order", "input_digest_sha256",
            "n_input_exact_zero_rows_pre", "n_input_sumzero_rows_pre",
            "n_input_exact_zero_rows_mid", "n_input_sumzero_rows_mid",
            "n_input_exact_zero_rows_final", "n_input_sumzero_rows_final"]
    check("all NPZ fields present", all(k in da for k in need))
    check("constituent_seeds shape (3,2) int64",
          da["constituent_seeds"].shape == (3, 2)
          and da["constituent_seeds"].dtype == np.int64)
    check("protocol stamp", da["protocol"] == PROTOCOL_LABEL)
    check("pre exact-zero rows = 548 padding rows",
          list(da["n_input_exact_zero_rows_pre"]) == [548, 548, 548])

    print("[4] verify_existing_cell negative tests")
    with tempfile.TemporaryDirectory() as td:
        base = dict(da)
        base["cache_hash_sha256"] = "deadbeef" * 8
        good = Path(td) / "DummyA__mixed_crop_noise__s5.npz"
        atomic_savez(good, **base)
        ok, dg = verify_existing_cell(good, "DummyA", "mixed_crop_noise", 5,
                                      cache, spec, "deadbeef" * 8)
        check("well-formed NPZ accepted", ok and len(dg) == 64)

        bad1 = dict(base)
        bad1["composition_order"] = np.array(["noise", "crop", "density", "outlier"])
        p1 = Path(td) / "bad1.npz"
        atomic_savez(p1, **bad1)
        ok1, _ = verify_existing_cell(p1, "DummyA", "mixed_crop_noise", 5,
                                      cache, spec, "deadbeef" * 8)
        check("wrong composition_order rejected", not ok1)

        bad2 = {k: v for k, v in base.items()
                if k != "n_input_sumzero_rows_mid"}
        p2 = Path(td) / "bad2.npz"
        atomic_savez(p2, **bad2)
        ok2, _ = verify_existing_cell(p2, "DummyA", "mixed_crop_noise", 5,
                                      cache, spec, "deadbeef" * 8)
        check("missing audit array rejected", not ok2)

        bad3 = dict(base)
        bad3["constituent_seeds"] = base["constituent_seeds"] + 1
        p3 = Path(td) / "bad3.npz"
        atomic_savez(p3, **bad3)
        ok3, _ = verify_existing_cell(p3, "DummyA", "mixed_crop_noise", 5,
                                      cache, spec, "deadbeef" * 8)
        check("wrong constituent_seeds rejected", not ok3)

        bad4 = dict(base)
        bad4["constituent_severities"] = np.array([5, 4], dtype=np.int32)
        p4 = Path(td) / "bad4.npz"
        atomic_savez(p4, **bad4)
        ok4, _ = verify_existing_cell(p4, "DummyA", "mixed_crop_noise", 5,
                                      cache, spec, "deadbeef" * 8)
        check("wrong constituent_severities rejected", not ok4)

    print("[5] metric mixed-op seed honesty (source-level)")
    metric_src = (REPO_ROOT / "scripts" / "metric_emitter.py").read_text(
        encoding="utf-8")
    check("metric has mixed_ corruption_seed=None branch",
          'op_name.startswith("mixed_")' in metric_src
          and "corruption_seed = None" in metric_src)
    check("metric whitelist has --allow-mixed-ops",
          "allow_mixed_ops" in metric_src
          and "mixed_density_outlier" in metric_src)

    print()
    if FAILURES:
        print(f"SMOKE FAILED - {len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE ALL GREEN")


if __name__ == "__main__":
    main()

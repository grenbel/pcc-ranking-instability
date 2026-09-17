"""Valid-point-only protocol sensitivity analyzer (Section V-C, Table 7).

Archive-relative recompute script. Reads per-sample CD-L1 arrays
from ``../metrics_validpt/{baseline}/{op}_s{sev}.json`` and writes the
side-by-side comparison against the zero-pad reference to
``../analysis/validpt_sensitivity_analysis.json``.

Usage::

    cd <unpacked_supplementary_archive>
    python scripts/instability_summary.py   # produces zero-pad reference
    python scripts/validpt_sensitivity.py    # produces this comparison

The output JSON reproduces every number used by Table 7 (and the §V-C body
text) from the released per-sample arrays without GPU access.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, kendalltau

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
METRICS_VP = ROOT / "metrics_validpt"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
ZEROPAD_REF = ANALYSIS_DIR / "instability_summary_analysis.json"
OUT_PATH = ANALYSIS_DIR / "validpt_sensitivity_analysis.json"

BASELINES = ["PoinTr", "AdaPoinTr", "SnowflakeNet", "SeedFormer"]
OPS = ["noise", "outlier", "density", "crop"]
SEVS = [1, 2, 3, 4, 5]
N_CELLS = len(OPS) * len(SEVS)
N_PAIRWISE_TESTS = N_CELLS * len(list(combinations(BASELINES, 2)))
ALPHA = 0.05
ALPHA_BONF_PAIRWISE = ALPHA / N_PAIRWISE_TESTS
N_BOOTSTRAP = 1000
RNG_SEED = 20260428

AGGREGATE_CLEAN = {
    "AdaPoinTr": 6.5199,
    "SeedFormer": 6.7490,
    "SnowflakeNet": 7.1940,
    "PoinTr": 7.2784,
}
AGGREGATE_RANK = ["AdaPoinTr", "SeedFormer", "SnowflakeNet", "PoinTr"]


def per_sample_path(baseline: str, op: str, sev: int) -> Path:
    return METRICS_VP / baseline / f"{op}_s{sev}.json"


def load_cell(baseline: str, op: str, sev: int):
    p = per_sample_path(baseline, op, sev)
    if not p.is_file():
        sys.exit(f"[FATAL] Missing validpt metrics file: {p}")
    with open(p, encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != 1200:
        sys.exit(f"[FATAL] {p} has {len(rows)} rows (expected 1200)")
    cd_l1 = np.array([r["cd_l1"] for r in rows], dtype=np.float64)
    sample_idx = np.array([r["idx"] for r in rows], dtype=np.int32)
    return cd_l1, sample_idx


def cell_rank(cell_means: dict) -> list:
    return sorted(BASELINES, key=lambda b: cell_means[b])


def _pair_key(a: str, b: str):
    if BASELINES.index(a) < BASELINES.index(b):
        return (a, b)
    return (b, a)


def derive_strict_pairwise(per_sample_aligned: dict):
    out = {}
    for a, b in combinations(BASELINES, 2):
        diff = per_sample_aligned[a] - per_sample_aligned[b]
        if np.allclose(diff, 0):
            out[(a, b)] = {"p": 1.0, "winner_bonferroni": "NS_bonf",
                           "mean_diff_x1000": 0.0}
            continue
        try:
            res = wilcoxon(per_sample_aligned[a], per_sample_aligned[b],
                           zero_method="zsplit", alternative="two-sided")
            p = float(res.pvalue)
        except Exception:
            p = 1.0
        mean_a = per_sample_aligned[a].mean()
        mean_b = per_sample_aligned[b].mean()
        winner = (a if mean_a < mean_b else b) if p < ALPHA_BONF_PAIRWISE else "NS_bonf"
        out[(a, b)] = {
            "p": p,
            "winner_bonferroni": winner,
            "mean_diff_x1000": (mean_a - mean_b) * 1000,
        }
    return out


def per_cell_means_from_resample(per_sample, boot_idx):
    out = {}
    for op in OPS:
        for sev in SEVS:
            cell_means = {}
            for b in BASELINES:
                arr = per_sample[(b, op, sev)]
                cell_means[b] = float(arr[boot_idx].mean())
            out[(op, sev)] = cell_means
    return out


def aggregate_summary_from_cells(cell_means_per_cell):
    flip = 0
    top1_change = 0
    winners = {b: 0 for b in BASELINES}
    taus = []
    agg_pos = {b: i for i, b in enumerate(AGGREGATE_RANK)}
    for op in OPS:
        for sev in SEVS:
            cell_means = cell_means_per_cell[(op, sev)]
            rank = cell_rank(cell_means)
            if rank != AGGREGATE_RANK:
                flip += 1
            if rank[0] != AGGREGATE_RANK[0]:
                top1_change += 1
            winners[rank[0]] += 1
            cell_pos = {b: i for i, b in enumerate(rank)}
            x = [agg_pos[b] for b in BASELINES]
            y = [cell_pos[b] for b in BASELINES]
            try:
                t, _ = kendalltau(x, y)
                taus.append(float(t) if not np.isnan(t) else 0.0)
            except Exception:
                taus.append(0.0)
    return {
        "flip_count": flip,
        "flip_rate": flip / N_CELLS,
        "top1_change_count": top1_change,
        "top1_change_rate": top1_change / N_CELLS,
        "winners": winners,
        "median_kendall_tau": float(np.median(taus)),
    }


def main():
    print("=" * 70)
    print("Valid-point-only protocol sensitivity (Section V-C, Table 7)")
    print("=" * 70)

    if not METRICS_VP.is_dir():
        sys.exit(f"[FATAL] {METRICS_VP} does not exist")

    print(f"\n[1/4] Loading validpt per-sample arrays from {METRICS_VP} ...")
    per_sample = {}
    sample_indices = {}
    for b in BASELINES:
        for op in OPS:
            for sev in SEVS:
                cd_l1, sidx = load_cell(b, op, sev)
                per_sample[(b, op, sev)] = cd_l1
                sample_indices[(b, op, sev)] = sidx
    print(f"  loaded {len(BASELINES)} baselines x {N_CELLS} cells = {len(per_sample)} baseline-cell entries x 1200 samples")

    print("\n[2/4] Per-cell means + strict-Bonferroni audit ...")
    strict_audit_per_cell = {}
    strict_best_count = 0
    ambiguous_top_count = 0
    strict_top1_change_count = 0
    for op in OPS:
        for sev in SEVS:
            cell_means = {b: float(per_sample[(b, op, sev)].mean()) for b in BASELINES}
            aligned = {b: per_sample[(b, op, sev)] for b in BASELINES}
            strict_pw = derive_strict_pairwise(aligned)
            rank = cell_rank(cell_means)
            mean_best = rank[0]
            strict_top_set = {mean_best}
            for other in BASELINES:
                if other == mean_best:
                    continue
                outcome = strict_pw[_pair_key(mean_best, other)]["winner_bonferroni"]
                if outcome == "NS_bonf":
                    strict_top_set.add(other)
            is_strict_best = (len(strict_top_set) == 1)
            strict_audit_per_cell[f"{op}_s{sev}"] = {
                "cell_means_x1000": {b: v * 1000 for b, v in cell_means.items()},
                "mean_rank": rank,
                "mean_best": mean_best,
                "strict_top_set": sorted(strict_top_set),
                "is_strict_best_beats_all": is_strict_best,
                "strict_top_set_size": len(strict_top_set),
                "pairwise_bonferroni": {f"{a}_vs_{b}": strict_pw[(a, b)]
                                        for (a, b) in strict_pw},
            }
            if is_strict_best:
                strict_best_count += 1
            else:
                ambiguous_top_count += 1
            if mean_best != AGGREGATE_RANK[0] and is_strict_best:
                strict_top1_change_count += 1
    print(f"  validpt strict-best-beats-all: {strict_best_count}/{N_CELLS}")
    print(f"  validpt ambiguous-top-set: {ambiguous_top_count}/{N_CELLS}")
    print(f"  validpt strict-top-1-change vs.\\ aggregate: {strict_top1_change_count}/{N_CELLS}")

    print(f"\n[3/4] Shape-level bootstrap (n={N_BOOTSTRAP}) ...")
    rng = np.random.default_rng(RNG_SEED)
    n_samples = 1200
    boot_summaries = []
    for boot_i in range(N_BOOTSTRAP):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        cell_means_boot = per_cell_means_from_resample(per_sample, boot_idx)
        summary = aggregate_summary_from_cells(cell_means_boot)
        boot_summaries.append(summary)
    flip_counts = np.array([s["flip_count"] for s in boot_summaries])
    top1_change_counts = np.array([s["top1_change_count"] for s in boot_summaries])
    median_taus = np.array([s["median_kendall_tau"] for s in boot_summaries])
    cell_means_full = per_cell_means_from_resample(per_sample, np.arange(n_samples))
    observed = aggregate_summary_from_cells(cell_means_full)
    print(f"  observed flip_count={observed['flip_count']}, "
          f"top1_change_count={observed['top1_change_count']}, "
          f"median_kendall_tau={observed['median_kendall_tau']:.4f}")
    print(f"  bootstrap flip_count CI95: "
          f"[{int(np.quantile(flip_counts, 0.025))}, "
          f"{int(np.quantile(flip_counts, 0.975))}]")

    print(f"\n[4/4] Loading zero-pad reference for side-by-side comparison ...")
    if ZEROPAD_REF.is_file():
        with open(ZEROPAD_REF, encoding="utf-8") as f:
            zp = json.load(f)
        zp_obs = zp["MAJ2_bootstrap_and_permutation"]["observed_full_sample"]
        zp_strict = zp["MAJ1_strict_winner"]["summary"]
        zp_ci = zp["MAJ2_bootstrap_and_permutation"]["bootstrap_ci"]
        comparison = {
            "zero_pad": {
                "observed_flip_count": zp_obs["flip_count"],
                "observed_top1_change_count": zp_obs["top1_change_count"],
                "observed_median_kendall_tau": zp_obs["median_kendall_tau"],
                "winners": zp_obs["winners"],
                "strict_best_beats_all": zp_strict["strict_best_beats_all_count"],
                "ambiguous_top_set": zp_strict["ambiguous_top_set_count"],
                "strict_top1_change": zp_strict["strict_top1_change_vs_aggregate_count"],
                "flip_count_ci95": [zp_ci["flip_count"]["ci95_low"],
                                    zp_ci["flip_count"]["ci95_high"]],
            },
            "valid_point_only": {
                "observed_flip_count": observed["flip_count"],
                "observed_top1_change_count": observed["top1_change_count"],
                "observed_median_kendall_tau": observed["median_kendall_tau"],
                "winners": observed["winners"],
                "strict_best_beats_all": strict_best_count,
                "ambiguous_top_set": ambiguous_top_count,
                "strict_top1_change": strict_top1_change_count,
                "flip_count_ci95": [int(np.quantile(flip_counts, 0.025)),
                                    int(np.quantile(flip_counts, 0.975))],
            },
        }
        zp_flips = set(c for c, e in zp.get("MAJ1_strict_winner", {}).get(
            "per_cell_audit", {}).items()
            if e.get("mean_rank", []) and e["mean_rank"] != AGGREGATE_RANK)
        vp_flips = set(c for c, e in strict_audit_per_cell.items()
                       if e["mean_rank"] != AGGREGATE_RANK)
        comparison["flip_overlap"] = {
            "zero_pad_only_flips": sorted(zp_flips - vp_flips),
            "valid_point_only_flips": sorted(vp_flips - zp_flips),
            "shared_flips": sorted(zp_flips & vp_flips),
            "n_zero_pad_flips": len(zp_flips),
            "n_valid_point_flips": len(vp_flips),
            "n_shared": len(zp_flips & vp_flips),
        }
        print(f"  zero-pad flips: {len(zp_flips)}, "
              f"validpt flips: {len(vp_flips)}, "
              f"shared: {len(zp_flips & vp_flips)}")
    else:
        comparison = None
        print(f"  zero-pad reference {ZEROPAD_REF} not found; run scripts/instability_summary.py first; comparison skipped")

    out = {
        "version": "valid-point-only protocol sensitivity",
        "constants": {
            "n_operator_severity_cells": N_CELLS,
            "n_baselines": len(BASELINES),
            "n_baseline_cell_entries": N_CELLS * len(BASELINES),
            "alpha_bonferroni_pairwise": ALPHA_BONF_PAIRWISE,
            "n_bootstrap": N_BOOTSTRAP,
            "aggregate_clean_x1000": AGGREGATE_CLEAN,
            "aggregate_rank": AGGREGATE_RANK,
        },
        "validpt_strict_winner_summary": {
            "strict_best_beats_all_count": strict_best_count,
            "ambiguous_top_set_count": ambiguous_top_count,
            "strict_top1_change_vs_aggregate_count": strict_top1_change_count,
            "mean_top1_change_count": observed["top1_change_count"],
            "mean_flip_count": observed["flip_count"],
            "median_kendall_tau": observed["median_kendall_tau"],
            "flip_count_ci95": [int(np.quantile(flip_counts, 0.025)),
                                int(np.quantile(flip_counts, 0.975))],
            "top1_change_count_ci95": [int(np.quantile(top1_change_counts, 0.025)),
                                       int(np.quantile(top1_change_counts, 0.975))],
            "median_kendall_tau_ci95": [float(np.quantile(median_taus, 0.025)),
                                        float(np.quantile(median_taus, 0.975))],
            "winners": observed["winners"],
        },
        "validpt_per_cell_audit": strict_audit_per_cell,
        "comparison_zero_pad_vs_validpt": comparison,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n[DONE] Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

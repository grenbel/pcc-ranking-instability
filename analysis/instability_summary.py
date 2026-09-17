"""Strict-Bonferroni winner audit + shape-level bootstrap + within-shape label-permutation null.

Archive-relative recompute script. Reads per-sample CD-L1 arrays
from ``../metrics/{baseline}/{op}_s{sev}.json`` and writes the aggregated
analysis JSON to ``../analysis/instability_summary_analysis.json``.

Usage::

    cd <unpacked_supplementary_archive>
    python scripts/instability_summary.py

The output JSON contains the strict-Bonferroni winner audit (Table 6 rows
"Strict-Bonferroni winner sensitivity"), the 1000-resample shape-level
bootstrap 95% CIs (Table 6 CI rows + the zero-pad columns of Tables 7 and 8), and the
1000-repeat within-shape label-permutation null (Section IV-D additional
validation). It reproduces every number used by Tables 5, 6, and 7 from the
released per-sample arrays without GPU access.
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
METRICS = ROOT / "metrics"
ANALYSIS_OUT_DIR = ROOT / "analysis"
ANALYSIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = ANALYSIS_OUT_DIR / "instability_summary_analysis.json"

BASELINES = ["PoinTr", "AdaPoinTr", "SnowflakeNet", "SeedFormer"]
OPS = ["noise", "outlier", "density", "crop"]
SEVS = [1, 2, 3, 4, 5]
N_CELLS = len(OPS) * len(SEVS)            # 20 (operator, severity) cells
N_PAIRWISE_TESTS = N_CELLS * len(list(combinations(BASELINES, 2)))  # 120
ALPHA = 0.05
ALPHA_BONF_PAIRWISE = ALPHA / N_PAIRWISE_TESTS  # ~4.17e-4
N_BOOTSTRAP = 1000
N_PERMUTE = 1000
RNG_SEED = 20260428

AGGREGATE_CLEAN = {
    "AdaPoinTr": 6.5199,
    "SeedFormer": 6.7490,
    "SnowflakeNet": 7.1940,
    "PoinTr": 7.2784,
}
AGGREGATE_RANK = ["AdaPoinTr", "SeedFormer", "SnowflakeNet", "PoinTr"]


def per_sample_path(baseline: str, op: str, sev: int) -> Path:
    return METRICS / baseline / f"{op}_s{sev}.json"


def load_cell(baseline: str, op: str, sev: int):
    p = per_sample_path(baseline, op, sev)
    if not p.is_file():
        sys.exit(f"[FATAL] Missing metrics file: {p}")
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
                           "mean_diff_x1000": 0.0, "n_a_wins": 0}
            continue
        try:
            res = wilcoxon(per_sample_aligned[a], per_sample_aligned[b],
                           zero_method="zsplit", alternative="two-sided")
            p = float(res.pvalue)
        except Exception:
            p = 1.0
        mean_a = per_sample_aligned[a].mean()
        mean_b = per_sample_aligned[b].mean()
        n_a_wins = int((per_sample_aligned[a] < per_sample_aligned[b]).sum())
        winner = (a if mean_a < mean_b else b) if p < ALPHA_BONF_PAIRWISE else "NS_bonf"
        out[(a, b)] = {
            "p": p,
            "winner_bonferroni": winner,
            "mean_diff_x1000": (mean_a - mean_b) * 1000,
            "n_a_wins": n_a_wins,
        }
    return out


def strict_winner_classify(cell_means, strict_pairwise):
    rank = cell_rank(cell_means)
    mean_best = rank[0]
    strict_top_set = {mean_best}
    for other in BASELINES:
        if other == mean_best:
            continue
        outcome = strict_pairwise[_pair_key(mean_best, other)]["winner_bonferroni"]
        if outcome == "NS_bonf":
            strict_top_set.add(other)
    is_strict_best = (len(strict_top_set) == 1)
    return {
        "mean_rank": rank,
        "mean_best": mean_best,
        "strict_top_set": sorted(strict_top_set),
        "is_strict_best_beats_all": is_strict_best,
        "strict_top_set_size": len(strict_top_set),
    }


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
    print("Strict-Bonferroni winner audit + bootstrap + permutation null")
    print("=" * 70)

    print(f"\n[1/4] Loading per-sample CD-L1 arrays from {METRICS} ...")
    per_sample = {}
    sample_indices = {}
    for b in BASELINES:
        for op in OPS:
            for sev in SEVS:
                cd_l1, sidx = load_cell(b, op, sev)
                per_sample[(b, op, sev)] = cd_l1
                sample_indices[(b, op, sev)] = sidx
    n_total = len(per_sample)
    assert n_total == 80, f"expected 80 baseline-cell entries, got {n_total}"
    print(f"  loaded {len(BASELINES)} baselines x {N_CELLS} cells = {n_total} baseline-cell entries x 1200 samples")

    print("\n[2/4] Verifying matched-control invariant (sample_idx alignment) ...")
    for op in OPS:
        for sev in SEVS:
            ref = sample_indices[(BASELINES[0], op, sev)]
            for b in BASELINES[1:]:
                cur = sample_indices[(b, op, sev)]
                if not np.array_equal(ref, cur):
                    sys.exit(f"[FATAL] sample_idx mismatch at {b}/{op}/s{sev}")
    print("  OK: all 4 baselines share identical sample_idx per (op, sev) cell")

    print("\n[3/4] Strict-Bonferroni winner audit ...")
    strict_audit_per_cell = {}
    strict_best_count = 0
    ambiguous_top_count = 0
    strict_top1_change_count = 0
    for op in OPS:
        for sev in SEVS:
            cell_means = {b: float(per_sample[(b, op, sev)].mean()) for b in BASELINES}
            aligned = {b: per_sample[(b, op, sev)] for b in BASELINES}
            strict_pw = derive_strict_pairwise(aligned)
            audit = strict_winner_classify(cell_means, strict_pw)
            strict_audit_per_cell[f"{op}_s{sev}"] = {
                "cell_means_x1000": {b: v * 1000 for b, v in cell_means.items()},
                **audit,
                "pairwise_bonferroni": {f"{a}_vs_{b}": strict_pw[(a, b)]
                                        for (a, b) in strict_pw},
            }
            if audit["is_strict_best_beats_all"]:
                strict_best_count += 1
            else:
                ambiguous_top_count += 1
            if audit["mean_best"] != AGGREGATE_RANK[0] and audit["is_strict_best_beats_all"]:
                strict_top1_change_count += 1
    print(f"  strict-best-beats-all: {strict_best_count}/{N_CELLS}")
    print(f"  ambiguous-top-set: {ambiguous_top_count}/{N_CELLS}")
    print(f"  strict-top-1-change vs.\\ aggregate AdaPoinTr: {strict_top1_change_count}/{N_CELLS}")

    print(f"\n[4/4] Shape-level bootstrap (n={N_BOOTSTRAP}) + within-shape permutation null (n={N_PERMUTE}) ...")
    rng = np.random.default_rng(RNG_SEED)
    n_samples = 1200

    boot_summaries = []
    for boot_i in range(N_BOOTSTRAP):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        cell_means_boot = per_cell_means_from_resample(per_sample, boot_idx)
        summary = aggregate_summary_from_cells(cell_means_boot)
        boot_summaries.append(summary)
        if (boot_i + 1) % 200 == 0:
            print(f"    bootstrap {boot_i+1}/{N_BOOTSTRAP}")
    flip_counts = np.array([s["flip_count"] for s in boot_summaries])
    flip_rates = np.array([s["flip_rate"] for s in boot_summaries])
    top1_change_counts = np.array([s["top1_change_count"] for s in boot_summaries])
    top1_change_rates = np.array([s["top1_change_rate"] for s in boot_summaries])
    median_taus = np.array([s["median_kendall_tau"] for s in boot_summaries])
    winner_counts_boot = {b: np.array([s["winners"][b] for s in boot_summaries])
                          for b in BASELINES}
    bootstrap_ci = {
        "flip_count": {
            "mean": float(flip_counts.mean()),
            "ci95_low": float(np.quantile(flip_counts, 0.025)),
            "ci95_high": float(np.quantile(flip_counts, 0.975)),
        },
        "flip_rate": {
            "mean": float(flip_rates.mean()),
            "ci95_low": float(np.quantile(flip_rates, 0.025)),
            "ci95_high": float(np.quantile(flip_rates, 0.975)),
        },
        "top1_change_count": {
            "mean": float(top1_change_counts.mean()),
            "ci95_low": float(np.quantile(top1_change_counts, 0.025)),
            "ci95_high": float(np.quantile(top1_change_counts, 0.975)),
        },
        "top1_change_rate": {
            "mean": float(top1_change_rates.mean()),
            "ci95_low": float(np.quantile(top1_change_rates, 0.025)),
            "ci95_high": float(np.quantile(top1_change_rates, 0.975)),
        },
        "median_kendall_tau": {
            "mean": float(median_taus.mean()),
            "ci95_low": float(np.quantile(median_taus, 0.025)),
            "ci95_high": float(np.quantile(median_taus, 0.975)),
        },
        "winner_counts": {
            b: {"mean": float(winner_counts_boot[b].mean()),
                "ci95_low": float(np.quantile(winner_counts_boot[b], 0.025)),
                "ci95_high": float(np.quantile(winner_counts_boot[b], 0.975))}
            for b in BASELINES
        },
    }
    print(f"  flip_count: {bootstrap_ci['flip_count']['mean']:.2f} "
          f"[{bootstrap_ci['flip_count']['ci95_low']:.0f}, {bootstrap_ci['flip_count']['ci95_high']:.0f}]")
    print(f"  median_kendall_tau: {bootstrap_ci['median_kendall_tau']['mean']:.3f} "
          f"[{bootstrap_ci['median_kendall_tau']['ci95_low']:.3f}, "
          f"{bootstrap_ci['median_kendall_tau']['ci95_high']:.3f}]")

    cell_means_full = per_cell_means_from_resample(per_sample, np.arange(n_samples))
    observed = aggregate_summary_from_cells(cell_means_full)
    print(f"  observed (full sample): flip_count={observed['flip_count']}, "
          f"top1_change_count={observed['top1_change_count']}, "
          f"median_kendall_tau={observed['median_kendall_tau']:.4f}")

    print(f"\n  within-shape baseline-label permutation null on flip count ...")
    null_flip_counts = []
    for perm_i in range(N_PERMUTE):
        cell_means_perm = {}
        for op in OPS:
            for sev in SEVS:
                stack = np.stack([per_sample[(b, op, sev)] for b in BASELINES], axis=1)
                perm_idx = rng.permuted(
                    np.broadcast_to(np.arange(len(BASELINES)), stack.shape),
                    axis=1,
                )
                permuted = np.take_along_axis(stack, perm_idx, axis=1)
                cell_means_perm[(op, sev)] = {b: float(permuted[:, i].mean())
                                              for i, b in enumerate(BASELINES)}
        null_summary = aggregate_summary_from_cells(cell_means_perm)
        null_flip_counts.append(null_summary["flip_count"])
        if (perm_i + 1) % 200 == 0:
            print(f"    permutation {perm_i+1}/{N_PERMUTE}")
    null_flip_counts = np.array(null_flip_counts)
    p_perm_one_sided = float((null_flip_counts >= observed["flip_count"]).mean())
    p_perm_plus1 = float(((null_flip_counts >= observed["flip_count"]).sum() + 1)
                          / (N_PERMUTE + 1))
    permutation_test = {
        "observed_flip_count": observed["flip_count"],
        "null_flip_count_mean": float(null_flip_counts.mean()),
        "null_flip_count_ci95": [float(np.quantile(null_flip_counts, 0.025)),
                                 float(np.quantile(null_flip_counts, 0.975))],
        "null_flip_count_max": int(null_flip_counts.max()),
        "p_one_sided": p_perm_one_sided,
        "p_one_sided_plus1_smoothed": p_perm_plus1,
        "n_permutations": N_PERMUTE,
        "interpretation": (
            "Null hypothesis: the four baseline labels are exchangeable within each "
            "(shape, op, sev) tuple. The observed cell-rank flip count is compared "
            "against the null distribution; a small p indicates the observed flip "
            "count is unlikely under exchange-invariant null."
        ),
    }
    print(f"  null_flip_count_mean = {null_flip_counts.mean():.2f}, "
          f"CI95 = [{int(np.quantile(null_flip_counts, 0.025))}, "
          f"{int(np.quantile(null_flip_counts, 0.975))}]")
    print(f"  p_one_sided (+1 smoothed) = {p_perm_plus1:.5f}")

    out = {
        "version": "strict-Bonferroni winner audit + bootstrap CI + permutation null",
        "constants": {
            "n_operator_severity_cells": N_CELLS,
            "n_baselines": len(BASELINES),
            "n_baseline_cell_entries": N_CELLS * len(BASELINES),
            "n_pairwise_tests": N_PAIRWISE_TESTS,
            "alpha_bonferroni_pairwise": ALPHA_BONF_PAIRWISE,
            "n_bootstrap": N_BOOTSTRAP,
            "n_permute": N_PERMUTE,
            "rng_seed": RNG_SEED,
            "aggregate_rank": AGGREGATE_RANK,
            "aggregate_clean_x1000": AGGREGATE_CLEAN,
        },
        "MAJ1_strict_winner": {
            "summary": {
                "strict_best_beats_all_count": strict_best_count,
                "ambiguous_top_set_count": ambiguous_top_count,
                "strict_top1_change_vs_aggregate_count": strict_top1_change_count,
                "mean_top1_change_count": observed["top1_change_count"],
                "mean_flip_count": observed["flip_count"],
            },
            "per_cell_audit": strict_audit_per_cell,
        },
        "MAJ2_bootstrap_and_permutation": {
            "bootstrap_ci": bootstrap_ci,
            "observed_full_sample": observed,
            "permutation_test": permutation_test,
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n[DONE] Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

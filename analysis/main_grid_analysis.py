"""Main-grid (zero-pad protocol) ranking analysis: generates ``analysis/4baseline_analysis.json``.

Archive-relative driver for the 20-cell main grid (4 baselines x {noise, outlier, density, crop}
x severities 1-5). It reads ``<archive>/metrics/<Baseline>/<op>_s<sev>.json`` (the per-sample rows
written by the metric emitter and arranged with ``arrange_metrics.py``) and writes the analysis
JSON that ``gen_table_data.py`` turns into the table sources:

    - per-cell mean CD-L1 (x1000) of each baseline, the per-cell rank, whether it differs from
      the aggregate-clean rank, and the Kendall tau between the two rankings
    - paired two-sided Wilcoxon tests for all 6 baseline pairs in every cell (120 tests) with a
      Bonferroni-corrected winner label at alpha/120
    - the Friedman omnibus test over the four baselines in every cell
    - the zero-row audit of the predictions (n_pred_zero_rows per cell); cells whose rows do
      not carry the field are reported as unavailable (null), never as zero

Run on the rows shipped in the supplementary archive it reproduces every statistic of the
shipped ``analysis/4baseline_analysis.json`` (cell_stats, pairwise_wilcoxon, friedman_per_cell,
flip_rate, median_kendall_tau, ...); only ``date`` and the zero-row audit differ, because the
shipped file reports zeros for the 40 cells whose rows predate the ``n_pred_zero_rows`` field
(``gen_table_data.py`` does not read that audit). The aggregate-clean CD-L1 values are the
Table 2 sanity numbers and are constants here, as in the shipped file.
Undefined test results (e.g. a Wilcoxon test on all-zero paired differences) are written as
null with a ``status`` field; the output is strict JSON (no NaN).

Usage:
    python analysis/main_grid_analysis.py --archive /path/to/archive_copy
    python scripts/main_grid_analysis.py          # when copied into an unpacked archive's scripts/
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, kendalltau, wilcoxon

BASELINES = ["PoinTr", "AdaPoinTr", "SnowflakeNet", "SeedFormer"]   # archive spelling
OPS = ["noise", "outlier", "density", "crop"]
SEVS = [1, 2, 3, 4, 5]
ROWS_PER_CELL = 1200
ALPHA = 0.05
N_PAIRWISE_TESTS = len(OPS) * len(SEVS) * len(list(combinations(BASELINES, 2)))  # 120
ALPHA_BONFERRONI = ALPHA / N_PAIRWISE_TESTS
# Aggregate-clean CD-L1 x1000 of each baseline on the clean PCN test split (Table 2).
AGGREGATE_CLEAN_CDL1 = {"PoinTr": 7.2784, "AdaPoinTr": 6.5199, "SnowflakeNet": 7.194, "SeedFormer": 6.749}
AGGREGATE_RANK = sorted(BASELINES, key=lambda m: AGGREGATE_CLEAN_CDL1[m])


def load_cells(archive: Path) -> dict:
    """Return {(baseline, op, sev): rows} after validating counts and identities."""
    cells = {}
    for b in BASELINES:
        for op in OPS:
            for sev in SEVS:
                p = archive / "metrics" / b / f"{op}_s{sev}.json"
                if not p.is_file():
                    sys.exit(f"[main-grid] missing {p}")
                with open(p, encoding="utf-8") as f:
                    rows = json.load(f)
                if len(rows) != ROWS_PER_CELL:
                    sys.exit(f"[main-grid] {p}: {len(rows)} rows != {ROWS_PER_CELL}")
                keys = [(r["taxonomy_id"], r["model_id"], r["view_id"]) for r in rows]
                if len(set(keys)) != len(keys):
                    sys.exit(f"[main-grid] {p}: duplicate sample identities")
                bad = [r for r in rows if r["model_name"] != b or r["op"] != op or r["severity"] != sev]
                if bad:
                    sys.exit(f"[main-grid] {p}: {len(bad)} rows with a different model_name/op/severity")
                nonfinite = [i for i, r in enumerate(rows)
                             if not isinstance(r.get("cd_l1"), (int, float)) or isinstance(r.get("cd_l1"), bool)
                             or not math.isfinite(r["cd_l1"])]
                if nonfinite:
                    sys.exit(f"[main-grid] {p}: {len(nonfinite)} rows with a missing or non-finite cd_l1 "
                             f"(first at row {nonfinite[0]})")
                cells[(b, op, sev)] = rows
    return cells


def paired(cells, m1, m2, op, sev):
    a = {(r["taxonomy_id"], r["model_id"], r["view_id"]): r["cd_l1"] for r in cells[(m1, op, sev)]}
    b = {(r["taxonomy_id"], r["model_id"], r["view_id"]): r["cd_l1"] for r in cells[(m2, op, sev)]}
    common = sorted(a.keys() & b.keys())
    if len(common) != ROWS_PER_CELL:
        sys.exit(f"[main-grid] {m1}/{m2} {op}@s{sev}: pair coverage {len(common)}/{ROWS_PER_CELL}")
    return np.array([a[k] for k in common]), np.array([b[k] for k in common])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", default=None,
                    help="root of the (copied) unpacked supplementary archive; default = the parent of this script's directory")
    ap.add_argument("--out", default=None, help="output path; default = <archive>/analysis/4baseline_analysis.json")
    args = ap.parse_args()
    archive = Path(args.archive) if args.archive else Path(__file__).resolve().parent.parent
    out = Path(args.out) if args.out else archive / "analysis" / "4baseline_analysis.json"

    cells = load_cells(archive)
    print(f"[main-grid] loaded {len(cells)} cells x {ROWS_PER_CELL} rows from {archive / 'metrics'}")
    print(f"[main-grid] aggregate-clean ranking (best -> worst): {' < '.join(AGGREGATE_RANK)}")

    cell_stats = {}
    for op in OPS:
        for sev in SEVS:
            means = {m: float(np.mean([r["cd_l1"] for r in cells[(m, op, sev)]]) * 1000) for m in BASELINES}
            rank = sorted(BASELINES, key=lambda m: means[m])
            agg_idx = {m: i for i, m in enumerate(AGGREGATE_RANK)}
            cell_idx = {m: i for i, m in enumerate(rank)}
            tau, tau_p = kendalltau(np.array([agg_idx[m] for m in BASELINES]),
                                    np.array([cell_idx[m] for m in BASELINES]))
            cell_stats[f"{op}_s{sev}"] = {
                "cell_means_x1000": means,
                "cell_rank": rank,
                "aggregate_rank": AGGREGATE_RANK,
                "flipped": bool(rank != AGGREGATE_RANK),
                "kendall_tau": float(tau),
                "kendall_p": float(tau_p),
            }
    flipped_cells = [k for k, v in cell_stats.items() if v["flipped"]]
    median_tau = float(np.median([v["kendall_tau"] for v in cell_stats.values()]))
    print(f"[main-grid] cells whose rank differs from the aggregate rank: {len(flipped_cells)}/{len(cell_stats)}; "
          f"median Kendall tau {median_tau:.3f}")

    pairwise = {}
    for op in OPS:
        for sev in SEVS:
            for m1, m2 in combinations(BASELINES, 2):
                arr1, arr2 = paired(cells, m1, m2, op, sev)
                diff = arr2 - arr1
                status = None
                try:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        _, pval = wilcoxon(arr2, arr1)
                    pval = float(pval)
                    if not math.isfinite(pval):  # e.g. every paired difference is zero
                        pval, status = None, "undefined: test statistic not finite (e.g. all paired differences are zero)"
                except ValueError as e:  # older SciPy versions raise instead of returning NaN
                    pval, status = None, f"undefined: {e}"
                winner = "NS_bonf"
                if pval is not None and pval < ALPHA_BONFERRONI:
                    winner = m1 if diff.mean() > 0 else m2
                entry = {
                    "m1_x1000": float(arr1.mean() * 1000),
                    "m2_x1000": float(arr2.mean() * 1000),
                    "diff_x1000": float(diff.mean() * 1000),
                    "p": pval,
                    "winner_bonferroni": winner,
                }
                if status:
                    entry["status"] = status
                pairwise[f"{op}_s{sev}_{m1}_vs_{m2}"] = entry

    friedman = {}
    for op in OPS:
        for sev in SEVS:
            per_model = {m: {(r["taxonomy_id"], r["model_id"], r["view_id"]): r["cd_l1"]
                             for r in cells[(m, op, sev)]} for m in BASELINES}
            common = sorted(set.intersection(*[set(d) for d in per_model.values()]))
            if len(common) != ROWS_PER_CELL:
                sys.exit(f"[main-grid] {op}@s{sev}: 4-way pair coverage {len(common)}/{ROWS_PER_CELL}")
            cols = [np.array([per_model[m][k] for k in common]) for m in BASELINES]
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    stat, pval = friedmanchisquare(*cols)
                stat, pval = float(stat), float(pval)
                if not (math.isfinite(stat) and math.isfinite(pval)):
                    raise ValueError("test statistic not finite (e.g. identical columns)")
                friedman[f"{op}_s{sev}"] = {"friedman_stat": stat, "friedman_p": pval}
            except ValueError as e:
                friedman[f"{op}_s{sev}"] = {"friedman_stat": None, "friedman_p": None, "status": f"undefined: {e}"}

    zero_row_audit = {}
    for m in BASELINES:
        for op in OPS:
            for sev in SEVS:
                rows = cells[(m, op, sev)]
                n_with_field = sum(1 for r in rows if "n_pred_zero_rows" in r)
                key = f"{m}_{op}_s{sev}"
                if n_with_field != len(rows):
                    zero_row_audit[key] = {"mean": None, "max": None, "p95": None, "n_zero_rows_>0": None,
                                           "n_pred_zero_rows_available": False,
                                           "rows_with_field": n_with_field}
                    continue
                zr = np.array([int(r["n_pred_zero_rows"]) for r in rows])
                zero_row_audit[key] = {"mean": float(zr.mean()), "max": int(zr.max()),
                                       "p95": int(np.percentile(zr, 95)), "n_zero_rows_>0": int((zr > 0).sum()),
                                       "n_pred_zero_rows_available": True}

    analysis = {
        "date": dt.date.today().isoformat(),
        "run_label": "4-baseline ranking flip analysis (PoinTr/AdaPoinTr/SnowflakeNet/SeedFormer)",
        "expected_models": BASELINES,
        "aggregate_clean_ranking": AGGREGATE_RANK,
        "aggregate_clean_cdl1_x1000": AGGREGATE_CLEAN_CDL1,
        "n_total_rows": len(cells) * ROWS_PER_CELL,
        "n_cells": len(cells),
        "alpha": ALPHA,
        "alpha_bonferroni_pairwise": ALPHA_BONFERRONI,
        "n_pairwise_tests": N_PAIRWISE_TESTS,
        "flip_rate": len(flipped_cells) / len(cell_stats),
        "flipped_cells": flipped_cells,
        "median_kendall_tau": median_tau,
        "cell_stats": cell_stats,
        "pairwise_wilcoxon": pairwise,
        "friedman_per_cell": friedman,
        "zero_row_audit": zero_row_audit,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, allow_nan=False)
    print(f"[main-grid] wrote {out}")


if __name__ == "__main__":
    main()

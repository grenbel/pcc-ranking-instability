"""Regenerate the main-grid table sources from ``analysis/4baseline_analysis.json`` for a rerun.

Companion of the frozen ``gen_table_data.py`` (shipped in the supplementary archive) for analysis
JSONs produced by ``main_grid_analysis.py`` on your own metric-emitter rows. Differences:

    - ``--archive`` / ``--out`` arguments instead of the fixed archive-relative paths
    - tolerates undefined tests: a Friedman or Wilcoxon result written as null with a ``status``
      field (e.g. two baselines with identical predictions in a cell) is excluded from the
      significance counts and reported in extra rows of the ranking-statistics table
    - writes table2_cell_grid, table3_ranking_stats, table4_outlier_reversal and table_A1_full_grid;
      table1_sanity is not written because its stamped-vs-reproduced numbers come from the sanity
      run (scripts/sanity_clean_pcn.py), not from this JSON

On the shipped analysis JSON (no undefined tests) the four files are byte-identical to the ones
written by ``gen_table_data.py``.

Usage:
    python analysis/gen_table_data_rerun.py --archive /path/to/archive_copy
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SEVS = [1, 2, 3, 4, 5]
OPS = ["noise", "outlier", "density", "crop"]
BASELINES = ["PoinTr", "AdaPoinTr", "SnowflakeNet", "SeedFormer"]
SHORT = {"PoinTr": "P", "AdaPoinTr": "A", "SnowflakeNet": "S", "SeedFormer": "F"}
ALPHA = 0.05
N_OS_CELLS = 20
N_PAIRWISE = 120
ALPHA_F = ALPHA / N_OS_CELLS
ALPHA_PAIR = ALPHA / N_PAIRWISE


def grid_rows(data, ops, sevs):
    rows = []
    for op in ops:
        for sev in sevs:
            cell = data["cell_stats"][f"{op}_s{sev}"]
            means = cell["cell_means_x1000"]
            winner = cell["cell_rank"][0]
            row = []
            for m in BASELINES:
                v = means[m]
                row.append(f"\\textbf{{{v:.2f}}}" if m == winner else f"{v:.2f}")
            rows.append(f"{op.capitalize()} & {sev} & " + " & ".join(row) + f" & {winner} \\\\")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", default=None,
                    help="root of the (copied) unpacked supplementary archive; default = the parent of this script's directory")
    ap.add_argument("--analysis", default=None, help="analysis JSON; default = <archive>/analysis/4baseline_analysis.json")
    ap.add_argument("--out", default=None, help="output directory; default = <archive>/generated_tables")
    args = ap.parse_args()
    archive = Path(args.archive) if args.archive else Path(__file__).resolve().parent.parent
    analysis = Path(args.analysis) if args.analysis else archive / "analysis" / "4baseline_analysis.json"
    out = Path(args.out) if args.out else archive / "generated_tables"
    if not analysis.is_file():
        sys.exit(f"[tables] missing analysis JSON: {analysis}")
    out.mkdir(parents=True, exist_ok=True)
    with open(analysis, encoding="utf-8") as f:
        data = json.load(f)

    # ---- table2_cell_grid (printed Table 3): representative cells ----
    t2 = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Representative matched-control (operator, severity) cells (regenerated from supplementary archive). CD-L1 $\times 10^3$.}",
        r"\label{tab:cell_grid}",
        r"\begin{tabular}{ll rrrr l}",
        r"\toprule",
        r"Operator & Sev & PoinTr & AdaPoinTr & SnowflakeNet & SeedFormer & Cell winner \\",
        r"\midrule",
    ] + grid_rows(data, OPS, (1, 5)) + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "table2_cell_grid.tex").write_text("\n".join(t2) + "\n", encoding="utf-8")

    # ---- table3_ranking_stats (printed Table 6): ranking instability statistics ----
    flip_rate = data["flip_rate"]
    agg_top1 = data["aggregate_clean_ranking"][0]
    n_winner_change = sum(1 for c in data["cell_stats"].values() if c["cell_rank"][0] != agg_top1)
    median_tau = data["median_kendall_tau"]
    friedman_undef = sorted(k for k, v in data["friedman_per_cell"].items() if v["friedman_p"] is None)
    n_friedman_sig = sum(1 for v in data["friedman_per_cell"].values()
                         if v["friedman_p"] is not None and v["friedman_p"] < ALPHA_F)
    pair_undef = sorted(k for k, v in data["pairwise_wilcoxon"].items() if v["p"] is None)
    pairwise_wins = Counter()
    for k, v in data["pairwise_wilcoxon"].items():
        if v["p"] is not None:
            pairwise_wins[v["winner_bonferroni"]] += 1
    n_NS = pairwise_wins["NS_bonf"]
    n_sig = sum(c for k, c in pairwise_wins.items() if k != "NS_bonf")

    t3 = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Ranking instability statistics (regenerated from supplementary archive).}",
        r"\label{tab:ranking_stats}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
        f"Cells with rank order $\\neq$ aggregate rank & {int(round(flip_rate * N_OS_CELLS))}/{N_OS_CELLS} = {flip_rate*100:.0f}\\% \\\\",
        f"Cells with top-1 winner change & {n_winner_change}/{N_OS_CELLS} = {n_winner_change/N_OS_CELLS*100:.0f}\\% \\\\",
        f"Median Kendall-$\\tau$ (aggregate vs.\\ cell rank) & {median_tau:.3f} \\\\",
        f"Friedman omnibus rejected at strict $\\alpha/{N_OS_CELLS}={ALPHA_F:.4f}$ & {n_friedman_sig}/{N_OS_CELLS} cells \\\\",
        f"Pairwise Wilcoxon significant at strict $\\alpha/{N_PAIRWISE} \\approx {ALPHA_PAIR:.2e}$ & {n_sig}/{N_PAIRWISE} \\\\",
        f"Pairwise Wilcoxon non-significant (NS) at $\\alpha/{N_PAIRWISE}$ & {n_NS}/{N_PAIRWISE} \\\\",
    ]
    if friedman_undef or pair_undef:
        t3 += [
            f"Friedman tests undefined (excluded from the counts above) & {len(friedman_undef)}/{N_OS_CELLS} cells \\\\",
            f"Pairwise Wilcoxon tests undefined (excluded from the counts above) & {len(pair_undef)}/{N_PAIRWISE} \\\\",
        ]
    t3 += [
        r"\midrule",
        r"Per-baseline significant pairwise advantages (out of 60 each) & \\",
    ]
    for m in BASELINES:
        t3.append(f"\\quad {m} ({SHORT[m]}) & {pairwise_wins.get(m, 0)} \\\\")
    t3 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "table3_ranking_stats.tex").write_text("\n".join(t3) + "\n", encoding="utf-8")

    # ---- table4_outlier_reversal (printed Table 5): outlier per-severity reversal ----
    t4 = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Outlier operator: per-severity CD-L1 ($\times 10^3$) (regenerated from supplementary archive).}",
        r"\label{tab:outlier_reversal}",
        r"\begin{tabular}{@{}c rrrr l@{}}",
        r"\toprule",
        r"Sev & PoinTr & AdaPoinTr & SnowflakeNet & SeedFormer & Cell rank \\",
        r"\midrule",
    ]
    for sev in SEVS:
        cell = data["cell_stats"][f"outlier_s{sev}"]
        means = cell["cell_means_x1000"]
        rank = cell["cell_rank"]
        rank_str = " $<$ ".join(SHORT[m] for m in rank)
        row = []
        for m in BASELINES:
            v = means[m]
            row.append(f"\\textbf{{{v:.2f}}}" if m == rank[0] else f"{v:.2f}")
        t4.append(f"{sev} & " + " & ".join(row) + f" & {rank_str} \\\\")
    t4 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "table4_outlier_reversal.tex").write_text("\n".join(t4) + "\n", encoding="utf-8")

    # ---- table_A1_full_grid (printed Table 4, main text): full per-baseline grid ----
    t_a1 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full per-baseline CD-L1 grid over the 20 (operator, severity) cells (80 baseline-cell entries on PCN test 1200). Regenerated from supplementary archive.}",
        r"\label{tab:full_grid}",
        r"\small",
        r"\begin{tabular}{l l rrrr l}",
        r"\toprule",
        r"Operator & Sev & PoinTr & AdaPoinTr & SnowflakeNet & SeedFormer & Cell winner \\",
        r"\midrule",
    ]
    for op in OPS:
        t_a1 += grid_rows(data, [op], SEVS)
        t_a1.append(r"\midrule")
    t_a1[-1] = r"\bottomrule"
    t_a1 += [r"\end{tabular}", r"\end{table*}"]
    (out / "table_A1_full_grid.tex").write_text("\n".join(t_a1) + "\n", encoding="utf-8")

    print(f"[tables] wrote table2_cell_grid, table3_ranking_stats, table4_outlier_reversal, table_A1_full_grid to {out.resolve()}")
    if friedman_undef or pair_undef:
        print(f"[tables] WARNING: {len(friedman_undef)} undefined Friedman test(s) {friedman_undef} and "
              f"{len(pair_undef)} undefined pairwise test(s) {pair_undef[:6]}{' ...' if len(pair_undef) > 6 else ''}; "
              f"they are excluded from the significance counts and listed in table3_ranking_stats")


if __name__ == "__main__":
    main()

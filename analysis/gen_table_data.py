"""Regenerate the released table .tex sources from the analysis JSON.

File names are historical; printed numbers in the revised manuscript are:
table1_sanity -> Table 2, table2_cell_grid -> Table 3, table3_ranking_stats
-> Table 6, table4_outlier_reversal -> Table 5, table_A1_full_grid -> Table 4
(promoted to the main text).

Archive-relative version. Reads from ``../analysis/4baseline_analysis.json``
and writes the regenerated .tex files into ``../generated_tables/``.

Usage::

    cd <unpacked_supplementary_archive>
    python scripts/gen_table_data.py

This is provided as a convenience for readers who want to independently
re-derive the LaTeX table rows from the released analysis JSON; the captions
that the paper actually compiles against contain finalized prose and are
shipped in the manuscript source bundle.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "4baseline_analysis.json"
OUT = ROOT / "generated_tables"

if not ANALYSIS.is_file():
    sys.exit(f"[FATAL] missing analysis JSON: {ANALYSIS}")
OUT.mkdir(parents=True, exist_ok=True)

with open(ANALYSIS, encoding="utf-8") as f:
    DATA = json.load(f)

SEVS = [1, 2, 3, 4, 5]
OPS = ["noise", "outlier", "density", "crop"]
BASELINES = ["PoinTr", "AdaPoinTr", "SnowflakeNet", "SeedFormer"]
SHORT = {"PoinTr": "P", "AdaPoinTr": "A", "SnowflakeNet": "S", "SeedFormer": "F"}

# ---- table1_sanity (printed Table 2): aggregate clean PCN sanity ----
SANITY = [
    ("PoinTr (PCN\\_new)",            7.263,    7.2784,  "PCN zero-pad"),
    ("AdaPoinTr",                     6.528,    6.5199,  "PCN zero-pad"),
    ("SnowflakeNet",                  7.188,    7.1940,  "PCNv2 UpSample"),
    ("SeedFormer (dim128)",           6.749,    6.7490,  "PCNv2 UpSample"),
]
t1 = [
    r"\begin{table*}[!htbp]",
    r"\centering",
    r"\caption{Aggregate clean PCN test 1200 sanity (regenerated from supplementary archive).}",
    r"\label{tab:sanity}",
    r"\begin{tabular}{@{}lrrrl@{}}",
    r"\toprule",
    r"Model & Stamped & Repro & $\Delta$ & Loader \\",
    r"\midrule",
]
for name, stamped, repro, loader in SANITY:
    delta_pct = (repro - stamped) / stamped * 100
    sign = "+" if delta_pct >= 0 else ""
    t1.append(f"{name} & {stamped:.3f} & {repro:.4f} & ${sign}{delta_pct:.2f}\\%$ & {loader} \\\\")
t1 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / "table1_sanity.tex").write_text("\n".join(t1) + "\n", encoding="utf-8")

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
]
for op in OPS:
    for sev in (1, 5):
        cell = DATA["cell_stats"][f"{op}_s{sev}"]
        means = cell["cell_means_x1000"]
        winner = cell["cell_rank"][0]
        row = []
        for m in BASELINES:
            v = means[m]
            row.append(f"\\textbf{{{v:.2f}}}" if m == winner else f"{v:.2f}")
        t2.append(f"{op.capitalize()} & {sev} & " + " & ".join(row) + f" & {winner} \\\\")
t2 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / "table2_cell_grid.tex").write_text("\n".join(t2) + "\n", encoding="utf-8")

# ---- table3_ranking_stats (printed Table 6): ranking instability statistics ----
flip_rate = DATA["flip_rate"]
AGG_TOP1 = DATA["aggregate_clean_ranking"][0]
n_winner_change = sum(
    1 for c in DATA["cell_stats"].values()
    if c["cell_rank"][0] != AGG_TOP1
)
median_tau = DATA["median_kendall_tau"]

ALPHA = 0.05
N_OS_CELLS = 20
N_PAIRWISE = 120
ALPHA_F = ALPHA / N_OS_CELLS
ALPHA_PAIR = ALPHA / N_PAIRWISE
n_friedman_sig = sum(
    1 for v in DATA["friedman_per_cell"].values()
    if v["friedman_p"] < ALPHA_F
)
pairwise_wins = Counter()
for v in DATA["pairwise_wilcoxon"].values():
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
    r"\midrule",
    r"Per-baseline significant pairwise advantages (out of 60 each) & \\",
]
for m in BASELINES:
    t3.append(f"\\quad {m} ({SHORT[m]}) & {pairwise_wins.get(m, 0)} \\\\")
t3 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / "table3_ranking_stats.tex").write_text("\n".join(t3) + "\n", encoding="utf-8")

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
    cell = DATA["cell_stats"][f"outlier_s{sev}"]
    means = cell["cell_means_x1000"]
    rank = cell["cell_rank"]
    rank_str = " $<$ ".join(SHORT[m] for m in rank)
    row = []
    for m in BASELINES:
        v = means[m]
        row.append(f"\\textbf{{{v:.2f}}}" if m == rank[0] else f"{v:.2f}")
    t4.append(f"{sev} & " + " & ".join(row) + f" & {rank_str} \\\\")
t4 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / "table4_outlier_reversal.tex").write_text("\n".join(t4) + "\n", encoding="utf-8")

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
    for sev in SEVS:
        cell = DATA["cell_stats"][f"{op}_s{sev}"]
        means = cell["cell_means_x1000"]
        winner = cell["cell_rank"][0]
        row = []
        for m in BASELINES:
            v = means[m]
            row.append(f"\\textbf{{{v:.2f}}}" if m == winner else f"{v:.2f}")
        t_a1.append(f"{op.capitalize()} & {sev} & " + " & ".join(row) + f" & {winner} \\\\")
    t_a1.append(r"\midrule")
t_a1[-1] = r"\bottomrule"
t_a1 += [r"\end{tabular}", r"\end{table*}"]
(OUT / "table_A1_full_grid.tex").write_text("\n".join(t_a1) + "\n", encoding="utf-8")

print(f"[DONE] Regenerated 5 table .tex files in {OUT.resolve()}")
print("  - table1_sanity.tex")
print("  - table2_cell_grid.tex")
print("  - table3_ranking_stats.tex")
print("  - table4_outlier_reversal.tex")
print("  - table_A1_full_grid.tex")

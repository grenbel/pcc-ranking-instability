"""Composed-operator pilot analyzer (Section V-E, Table 9).

Archive-relative recompute script. Reads per-sample CD-L1 arrays
from ``../metrics_mixed/{baseline}/{op}_s{sev}.json`` (3 composed operators x
5 severities x 4 baselines = 60 cells, zero-pad primary protocol) and writes
``../analysis/mixed_pilot_analysis.json``.

Usage::

    cd <unpacked_supplementary_archive>
    python scripts/instability_summary.py   # zero-pad single-operator audit
    python scripts/mixed_pilot.py           # this pilot

The output reproduces every number used by Table 9 and the Section V-E body
text from the released per-sample arrays without GPU access: per-cell
mean-rank and winner, flip / top-1 counts against the published
aggregate-clean reference (the same single reference as the main grid),
strict-Bonferroni audit counts within the 15-cell supplementary family
(alpha/15 Friedman, alpha/90 pairwise), shape-level bootstrap CIs, the
metric-stage chamfer row-drop audit, and a DESCRIPTIVE composition-vs-single
comparison in which both the mixed cell and its two constituent single-
operator cells are re-thresholded at one declared cutoff (alpha/120, the
stricter of the two families) from stored p-values and mean-difference
direction. The forward-stage input row audit (padding rows before / between /
after the two constituents) cannot be recomputed from this archive because
the dense prediction NPZs are not shipped; it is carried in
``../analysis/mixed_pilot_input_audit.json`` as provenance and merged into the
output when present.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, kendalltau, wilcoxon

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
METRICS_MIXED = ROOT / "metrics_mixed"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
ZEROPAD_REF = ANALYSIS_DIR / "instability_summary_analysis.json"
INPUT_AUDIT_SIDECAR = ANALYSIS_DIR / "mixed_pilot_input_audit.json"
OUT_PATH = ANALYSIS_DIR / "mixed_pilot_analysis.json"

BASELINES = ["PoinTr", "AdaPoinTr", "SnowflakeNet", "SeedFormer"]
# Row-level model_name stamps use the registry spelling (capital F).
ROW_NAME = {"PoinTr": "PoinTr", "AdaPoinTr": "AdaPoinTr",
            "SnowflakeNet": "SnowFlakeNet", "SeedFormer": "SeedFormer"}
MIXED_PAIRS = {
    "mixed_noise_outlier": ("noise", "outlier"),
    "mixed_crop_noise": ("crop", "noise"),
    "mixed_density_outlier": ("density", "outlier"),
}
COMPOSITION_ORDER = ["crop", "density", "noise", "outlier"]
OPS = list(MIXED_PAIRS)
SEVS = [1, 2, 3, 4, 5]
N_CELLS = len(OPS) * len(SEVS)                                      # 15
N_PAIRWISE_TESTS = N_CELLS * len(list(combinations(BASELINES, 2)))  # 90
ALPHA = 0.05
ALPHA_BONF_PAIRWISE = ALPHA / N_PAIRWISE_TESTS   # alpha/90
ALPHA_BONF_FRIEDMAN = ALPHA / N_CELLS            # alpha/15
ZEROPAD_FAMILY_PAIRWISE_ALPHA = ALPHA / 120
COMPARISON_CUTOFF = min(ALPHA_BONF_PAIRWISE, ZEROPAD_FAMILY_PAIRWISE_ALPHA)
CI_METHODS = ("lower", "higher")
N_BOOTSTRAP = 1000
RNG_SEED = 20260730
N_SAMPLES = 1200
PROTOCOL = "mixed_zero_pad_v1"
ZERO_PAD_CACHE_HASH = "80d3efce468eba6ced024077affc157e669c55e2eb2c436371de6a906991ea9c"
PUBLISHED_RANK = ["AdaPoinTr", "SeedFormer", "SnowflakeNet", "PoinTr"]
RELATION_LEGEND = ("'W>L' means baseline W has LOWER mean CD-L1 than baseline L in "
                   "that cell AND the paired two-sided Wilcoxon p-value is below "
                   "comparison_cutoff; relations are descriptive labels, not tests "
                   "of composition effects")


def cell_path(baseline, op, sev):
    return METRICS_MIXED / baseline / f"{op}_s{sev}.json"


def load_cell(baseline, op, sev):
    """Full-row provenance + identity validation (mirrors the release
    pipeline's analyzer): every row must carry the mixed protocol stamp, the
    exact zero-pad cache invariant, the expected cell identity, a null
    corruption_seed (composed cells have per-constituent seeds, recorded in
    the forward NPZs, not a single seed), finite cd_l1, and the row-drop
    audit fields. Returns the identity sequence for cross-file alignment."""
    p = cell_path(baseline, op, sev)
    if not p.is_file():
        sys.exit(f"[FATAL] Missing per-sample file: {p}")
    with open(p, encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != N_SAMPLES:
        sys.exit(f"[FATAL] {p} has {len(rows)} rows (expected {N_SAMPLES})")
    cache_hash = rows[0].get("cache_hash_sha256")
    if cache_hash != ZERO_PAD_CACHE_HASH:
        sys.exit(f"[FATAL] {p} cache_hash {cache_hash} != zero-pad invariant")
    cd, sumz_pred, sumz_gt, identity = [], 0, 0, []
    for i, r in enumerate(rows):
        if r.get("protocol") != PROTOCOL:
            sys.exit(f"[FATAL] {p} row {i} protocol '{r.get('protocol')}'")
        if r.get("cache_hash_sha256") != cache_hash:
            sys.exit(f"[FATAL] {p} row {i} cache_hash inconsistent")
        if (r.get("model_name") != ROW_NAME[baseline] or r.get("op") != op
                or r.get("severity") != sev):
            sys.exit(f"[FATAL] {p} row {i} cell identity mismatch")
        if "corruption_seed" not in r or r["corruption_seed"] is not None:
            sys.exit(f"[FATAL] {p} row {i} must carry corruption_seed: null")
        v = r.get("cd_l1")
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            sys.exit(f"[FATAL] {p} row {i} non-finite cd_l1: {v}")
        if "n_pred_sumzero_rows" not in r or "n_gt_sumzero_rows" not in r:
            sys.exit(f"[FATAL] {p} row {i} missing row-drop audit fields")
        cd.append(float(v))
        sumz_pred += int(r["n_pred_sumzero_rows"])
        sumz_gt += int(r["n_gt_sumzero_rows"])
        identity.append((r["idx"], r["taxonomy_id"], r["model_id"], r["view_id"]))
    return np.array(cd, dtype=np.float64), sumz_pred, sumz_gt, identity, cache_hash


def cell_rank(cell_means):
    return sorted(BASELINES, key=lambda b: cell_means[b])


def _pair_key(a, b):
    return (a, b) if BASELINES.index(a) < BASELINES.index(b) else (b, a)


def derive_strict_pairwise(aligned):
    out = {}
    for a, b in combinations(BASELINES, 2):
        diff = aligned[a] - aligned[b]
        if np.allclose(diff, 0):
            out[(a, b)] = {"p": 1.0, "winner_bonferroni": "NS_bonf", "mean_diff_x1000": 0.0}
            continue
        try:
            p = float(wilcoxon(aligned[a], aligned[b], zero_method="zsplit",
                               alternative="two-sided").pvalue)
        except Exception:
            p = 1.0
        ma, mb = aligned[a].mean(), aligned[b].mean()
        winner = (a if ma < mb else b) if p < ALPHA_BONF_PAIRWISE else "NS_bonf"
        out[(a, b)] = {"p": p, "winner_bonferroni": winner, "mean_diff_x1000": (ma - mb) * 1000}
    return out


def per_cell_means_from_resample(per_sample, idx):
    return {(op, sev): {b: float(per_sample[(b, op, sev)][idx].mean()) for b in BASELINES}
            for op in OPS for sev in SEVS}


def summary_vs_reference(cell_means_per_cell, reference_rank):
    flip = top1 = 0
    winners = {b: 0 for b in BASELINES}
    taus = []
    ref_pos = {b: i for i, b in enumerate(reference_rank)}
    for op in OPS:
        for sev in SEVS:
            rank = cell_rank(cell_means_per_cell[(op, sev)])
            flip += rank != reference_rank
            top1 += rank[0] != reference_rank[0]
            winners[rank[0]] += 1
            cell_pos = {b: i for i, b in enumerate(rank)}
            t, _ = kendalltau([ref_pos[b] for b in BASELINES], [cell_pos[b] for b in BASELINES])
            taus.append(float(t) if not np.isnan(t) else 0.0)
    return {"flip_count": flip, "flip_rate": flip / N_CELLS,
            "top1_change_count": top1, "top1_change_rate": top1 / N_CELLS,
            "winners": winners, "median_kendall_tau": float(np.median(taus))}


def strict_relations(pairwise_by_name, cutoff):
    """'W>L' relations re-derived from stored p + mean_diff_x1000 sign at
    `cutoff` (both sides of a comparison thresholded identically; the
    families' own winner labels are ignored)."""
    rel = set()
    for key, v in pairwise_by_name.items():
        a, b = key.split("_vs_")
        p, diff = float(v["p"]), float(v["mean_diff_x1000"])
        if p < cutoff and diff != 0.0:
            w, l = (a, b) if diff < 0 else (b, a)
            rel.add(f"{w}>{l}")
    return rel


def main():
    print("=" * 70)
    print("Composed-operator pilot analysis (Section V-E, Table 9)")
    print("=" * 70)
    if not METRICS_MIXED.is_dir():
        sys.exit(f"[FATAL] {METRICS_MIXED} missing")

    print(f"\n[1/5] Loading {N_CELLS} mixed cells x 4 baselines ...")
    per_sample, friedman, rowdrop = {}, {}, {}
    ref_identity = ref_hash = None
    for op in OPS:
        for sev in SEVS:
            key = f"{op}_s{sev}"
            rowdrop[key] = {"pred": 0, "gt": 0}
            for b in BASELINES:
                cd, szp, szg, ident, chash = load_cell(b, op, sev)
                if ref_identity is None:
                    ref_identity, ref_hash = ident, chash
                elif ident != ref_identity or chash != ref_hash:
                    sys.exit(f"[FATAL] {b} {key} identity/cache_hash misaligned")
                per_sample[(b, op, sev)] = cd
                rowdrop[key]["pred"] += szp
                rowdrop[key]["gt"] += szg
            _, p = friedmanchisquare(*[per_sample[(b, op, sev)] for b in BASELINES])
            friedman[key] = {"p": float(p), "significant_bonf": bool(p < ALPHA_BONF_FRIEDMAN)}
    friedman_sig = sum(1 for v in friedman.values() if v["significant_bonf"])
    total_pred_drop = sum(v["pred"] for v in rowdrop.values())
    total_gt_drop = sum(v["gt"] for v in rowdrop.values())
    print(f"  loaded {N_CELLS * 4} cells; Friedman sig at alpha/{N_CELLS}: {friedman_sig}/{N_CELLS}; "
          f"row-drop pred={total_pred_drop} gt={total_gt_drop}")

    print(f"\n[2/5] Strict-Bonferroni audit (alpha/{N_PAIRWISE_TESTS}) ...")
    strict_audit = {}
    strict_best = ambiguous = strict_top1 = 0
    for op in OPS:
        for sev in SEVS:
            cell_means = {b: float(per_sample[(b, op, sev)].mean()) for b in BASELINES}
            pw = derive_strict_pairwise({b: per_sample[(b, op, sev)] for b in BASELINES})
            rank = cell_rank(cell_means)
            best = rank[0]
            top_set = {best} | {o for o in BASELINES if o != best and
                                pw[_pair_key(best, o)]["winner_bonferroni"] == "NS_bonf"}
            is_strict = len(top_set) == 1
            strict_audit[f"{op}_s{sev}"] = {
                "constituents": list(MIXED_PAIRS[op]),
                "cell_means_x1000": {b: v * 1000 for b, v in cell_means.items()},
                "mean_rank": rank, "mean_best": best,
                "strict_top_set": sorted(top_set),
                "is_strict_best_beats_all": is_strict,
                "pairwise_bonferroni": {f"{a}_vs_{b}": pw[(a, b)] for (a, b) in pw},
            }
            strict_best += is_strict
            ambiguous += not is_strict
            strict_top1 += is_strict and best != PUBLISHED_RANK[0]
    print(f"  strict-best-beats-all {strict_best}/{N_CELLS} | ambiguous {ambiguous}/{N_CELLS} | "
          f"strict top-1 change {strict_top1}/{N_CELLS}")

    print(f"\n[3/5] Observed summary + bootstrap (n={N_BOOTSTRAP}) vs published rank ...")
    full_means = per_cell_means_from_resample(per_sample, np.arange(N_SAMPLES))
    obs = summary_vs_reference(full_means, PUBLISHED_RANK)
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, N_SAMPLES, size=N_SAMPLES)
        boots.append(summary_vs_reference(per_cell_means_from_resample(per_sample, idx), PUBLISHED_RANK))

    def ci(key, as_int=True):
        arr = np.array([s[key] for s in boots])
        lo = np.quantile(arr, 0.025, method=CI_METHODS[0])
        hi = np.quantile(arr, 0.975, method=CI_METHODS[1])
        return [int(lo), int(hi)] if as_int else [float(lo), float(hi)]

    print(f"  flip={obs['flip_count']}/{N_CELLS} CI{ci('flip_count')} | top1={obs['top1_change_count']}/{N_CELLS} "
          f"CI{ci('top1_change_count')} | tau={obs['median_kendall_tau']:.3f} | winners={obs['winners']}")

    print(f"\n[4/5] Composition-vs-single (descriptive; {ZEROPAD_REF.name}) ...")
    if not ZEROPAD_REF.is_file():
        sys.exit(f"[FATAL] {ZEROPAD_REF} missing — run scripts/instability_summary.py first")
    with open(ZEROPAD_REF, encoding="utf-8") as f:
        zp = json.load(f)
    zp_cells = zp["MAJ1_strict_winner"]["per_cell_audit"]
    zp_consts = zp.get("constants", {})
    if "alpha_bonferroni_pairwise" not in zp_consts:
        sys.exit(f"[FATAL] {ZEROPAD_REF} constants lack alpha_bonferroni_pairwise")
    zp_alpha = zp_consts["alpha_bonferroni_pairwise"]
    if not isinstance(zp_alpha, (int, float)) or abs(float(zp_alpha) - ZEROPAD_FAMILY_PAIRWISE_ALPHA) > 1e-12:
        sys.exit(f"[FATAL] zero-pad audit alpha {zp_alpha!r} != {ZEROPAD_FAMILY_PAIRWISE_ALPHA}")
    composition = {}
    for op in OPS:
        c1, c2 = MIXED_PAIRS[op]
        for sev in SEVS:
            mixed = strict_audit[f"{op}_s{sev}"]
            e1, e2 = zp_cells[f"{c1}_s{sev}"], zp_cells[f"{c2}_s{sev}"]
            rel_m = strict_relations(mixed["pairwise_bonferroni"], COMPARISON_CUTOFF)
            rel_1 = strict_relations(e1["pairwise_bonferroni"], COMPARISON_CUTOFF)
            rel_2 = strict_relations(e2["pairwise_bonferroni"], COMPARISON_CUTOFF)
            composition[f"{op}_s{sev}"] = {
                "mixed_mean_rank": mixed["mean_rank"],
                f"{c1}_s{sev}_mean_rank": e1["mean_rank"],
                f"{c2}_s{sev}_mean_rank": e2["mean_rank"],
                "mixed_rank_equals_first_constituent": mixed["mean_rank"] == e1["mean_rank"],
                "mixed_rank_equals_second_constituent": mixed["mean_rank"] == e2["mean_rank"],
                "mixed_winner": mixed["mean_best"],
                "constituent_winners": [e1["mean_best"], e2["mean_best"]],
                "relations_in_mixed_audit": sorted(rel_m),
                "relations_in_first_constituent_audit": sorted(rel_1),
                "relations_in_second_constituent_audit": sorted(rel_2),
                "relations_in_mixed_audit_absent_from_both_constituent_audits": sorted(rel_m - (rel_1 | rel_2)),
                "relations_present_in_both_constituent_audits_but_absent_in_mixed_audit": sorted((rel_1 & rel_2) - rel_m),
            }
    n_new = sum(1 for v in composition.values() if v["relations_in_mixed_audit_absent_from_both_constituent_audits"])
    n_same = sum(1 for v in composition.values()
                 if v["mixed_rank_equals_first_constituent"] or v["mixed_rank_equals_second_constituent"])
    print(f"  cutoff p<{COMPARISON_CUTOFF:.3e}; rank equals a constituent's: {n_same}/{N_CELLS}; "
          f">=1 relation absent from both constituent audits: {n_new}/{N_CELLS}")

    input_audit = None
    if INPUT_AUDIT_SIDECAR.is_file():
        with open(INPUT_AUDIT_SIDECAR, encoding="utf-8") as f:
            input_audit = json.load(f)
        print(f"\n  forward-stage input row audit merged from {INPUT_AUDIT_SIDECAR.name} "
              f"(provenance; not recomputable from this archive)")

    print(f"\n[5/5] Writing {OUT_PATH} ...")
    out = {
        "version": "composed-operator pilot (Section V-E, Table 9)",
        "constants": {
            "protocol": PROTOCOL, "mixed_pairs": {k: list(v) for k, v in MIXED_PAIRS.items()},
            "composition_order": COMPOSITION_ORDER,
            "n_cells": N_CELLS, "n_pairwise_tests": N_PAIRWISE_TESTS,
            "alpha_bonf_pairwise": ALPHA_BONF_PAIRWISE, "alpha_bonf_friedman": ALPHA_BONF_FRIEDMAN,
            "n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED,
            "published_rank_reference": PUBLISHED_RANK,
            "reference_note": "single reference identical to the main-grid convention (Tables 3/4)",
            "cache_hash_sha256_expected": ZERO_PAD_CACHE_HASH, "cache_hash_sha256_observed": ref_hash,
            "bootstrap_ci_quantile_methods": {"lower_2.5pct": CI_METHODS[0], "upper_97.5pct": CI_METHODS[1]},
            "composition_comparison": {
                "comparison_cutoff": COMPARISON_CUTOFF,
                "mixed_family_pairwise_alpha": ALPHA_BONF_PAIRWISE,
                "zeropad_family_pairwise_alpha": ZEROPAD_FAMILY_PAIRWISE_ALPHA,
                "zeropad_audit_file": ZEROPAD_REF.name, "zeropad_audit_declared_alpha": zp_alpha,
                "relation_legend": RELATION_LEGEND,
            },
        },
        "summary_vs_published": {**obs, "flip_count_ci95": ci("flip_count"),
                                 "top1_change_count_ci95": ci("top1_change_count"),
                                 "median_kendall_tau_ci95": ci("median_kendall_tau", False)},
        "strict_winner_summary": {"strict_best_beats_all_count": strict_best,
                                  "ambiguous_top_set_count": ambiguous,
                                  "strict_top1_change_vs_published": strict_top1},
        "friedman_per_cell": friedman, "friedman_significant_count": friedman_sig,
        "rowdrop_audit_metric_stage": {"per_cell": rowdrop, "total_pred_sumzero": total_pred_drop,
                                       "total_gt_sumzero": total_gt_drop},
        "input_row_audit_forward_stage": input_audit,
        "composition_vs_single_descriptive": {"legend": RELATION_LEGEND,
                                              "comparison_cutoff": COMPARISON_CUTOFF,
                                              "per_cell": composition},
        "per_cell_audit": strict_audit,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[DONE] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

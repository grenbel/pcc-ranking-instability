"""UpSamplePoints loader-transform sensitivity analyzer (Section V-D, Table 8).

Archive-relative recompute script. Reads per-sample CD-L1 arrays
from ``../metrics_upsample/{baseline}/{op}_s{sev}.json`` (corruption cells)
and ``../metrics_upsample/{baseline}/clean_s0.json`` (clean pass-through
cells), and writes the dual-reference instability summary plus the
side-by-side comparison against the zero-pad reference to
``../analysis/upsample_sensitivity_analysis.json``.

Usage::

    cd <unpacked_supplementary_archive>
    python scripts/instability_summary.py    # produces zero-pad reference
    python scripts/upsample_sensitivity.py   # produces this comparison

The output JSON reproduces every number used by Table 8 (and the Section V-D
body text) from the released per-sample arrays without GPU access: flip and
top-1 counts under both the published reference and the same-protocol clean
reference, strict-Bonferroni audit counts, Friedman rejections, Kendall-tau,
shape-level bootstrap CIs (the clean rank is re-estimated inside every
bootstrap replicate), the clean-shift table, the chamfer row-drop audit, and
the flip-cell overlap versus zero-pad.
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
METRICS_UP = ROOT / "metrics_upsample"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
ZEROPAD_REF = ANALYSIS_DIR / "instability_summary_analysis.json"
OUT_PATH = ANALYSIS_DIR / "upsample_sensitivity_analysis.json"

BASELINES = ["PoinTr", "AdaPoinTr", "SnowflakeNet", "SeedFormer"]
# Row-level model_name stamps use the registry spelling (capital F).
ROW_NAME = {"PoinTr": "PoinTr", "AdaPoinTr": "AdaPoinTr",
            "SnowflakeNet": "SnowFlakeNet", "SeedFormer": "SeedFormer"}
OPS = ["noise", "outlier", "density", "crop"]
SEVS = [1, 2, 3, 4, 5]
N_CELLS = len(OPS) * len(SEVS)
ALPHA = 0.05
ALPHA_BONF_PAIRWISE = ALPHA / (N_CELLS * len(list(combinations(BASELINES, 2))))
ALPHA_BONF_FRIEDMAN = ALPHA / N_CELLS
N_BOOTSTRAP = 1000
RNG_SEED = 20260706
PROTOCOL = "upsample_points_v1"

PUBLISHED_RANK = ["AdaPoinTr", "SeedFormer", "SnowflakeNet", "PoinTr"]
STAMPED_CLEAN_X1000 = {"PoinTr": 7.263, "AdaPoinTr": 6.528,
                       "SnowflakeNet": 7.188, "SeedFormer": 6.749}
# Paper Table 1 clean anchors: PoinTr/AdaPoinTr under PCN zero-pad,
# SnowflakeNet/SeedFormer under native PCNv2 UpSamplePoints (NOT shared-zero-pad).
TABLE1_REPRO_CLEAN_X1000 = {"AdaPoinTr": 6.5199, "SeedFormer": 6.7490,
                            "SnowflakeNet": 7.1940, "PoinTr": 7.2784}


def cell_path(baseline, op, sev):
    name = "clean_s0.json" if op == "clean" else f"{op}_s{sev}.json"
    return METRICS_UP / baseline / name


def load_cell(baseline, op, sev):
    """Full-row provenance + identity validation (mirrors the release
    pipeline's analyzer): every row must carry the UpSamplePoints protocol stamp, a
    file-consistent cache hash, the expected cell identity, finite cd_l1, and
    the row-drop audit fields. Returns the identity sequence for cross-file
    alignment (paired statistics assume identical sample order)."""
    p = cell_path(baseline, op, sev)
    if not p.is_file():
        sys.exit(f"[FATAL] missing {p}")
    rows = json.load(open(p, encoding="utf-8"))
    if len(rows) != 1200:
        sys.exit(f"[FATAL] {p}: {len(rows)} rows != 1200")
    chash = rows[0].get("cache_hash_sha256")
    cd, szp, szg, ident = [], 0, 0, []
    for i, r in enumerate(rows):
        if r.get("protocol") != PROTOCOL:
            sys.exit(f"[FATAL] {p} row {i}: protocol {r.get('protocol')!r}")
        if not chash or r.get("cache_hash_sha256") != chash:
            sys.exit(f"[FATAL] {p} row {i}: cache_hash inconsistent")
        if (r.get("model_name") != ROW_NAME[baseline] or r.get("op") != op
                or r.get("severity") != sev):
            sys.exit(f"[FATAL] {p} row {i}: cell identity mismatch")
        v = r.get("cd_l1")
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            sys.exit(f"[FATAL] {p} row {i}: bad cd_l1 {v!r}")
        if "n_pred_sumzero_rows" not in r or "n_gt_sumzero_rows" not in r:
            sys.exit(f"[FATAL] {p} row {i}: missing row-drop audit fields")
        cd.append(float(v))
        szp += int(r["n_pred_sumzero_rows"]); szg += int(r["n_gt_sumzero_rows"])
        ident.append((r["idx"], r["taxonomy_id"], r["model_id"], r["view_id"]))
    return np.array(cd), szp, szg, ident, chash


def cell_rank(means):
    return sorted(BASELINES, key=lambda b: means[b])


def _pair_key(a, b):
    return (a, b) if BASELINES.index(a) < BASELINES.index(b) else (b, a)


def derive_strict_pairwise(aligned):
    out = {}
    for a, b in combinations(BASELINES, 2):
        if np.allclose(aligned[a] - aligned[b], 0):
            out[(a, b)] = {"p": 1.0, "winner_bonferroni": "NS_bonf",
                           "mean_diff_x1000": 0.0}
            continue
        try:
            p = float(wilcoxon(aligned[a], aligned[b], zero_method="zsplit",
                               alternative="two-sided").pvalue)
        except Exception:
            p = 1.0
        ma, mb = aligned[a].mean(), aligned[b].mean()
        w = (a if ma < mb else b) if p < ALPHA_BONF_PAIRWISE else "NS_bonf"
        out[(a, b)] = {"p": p, "winner_bonferroni": w,
                       "mean_diff_x1000": (ma - mb) * 1000}
    return out


def means_from_resample(per_sample, idx):
    return {(op, s): {b: float(per_sample[(b, op, s)][idx].mean())
                      for b in BASELINES} for op in OPS for s in SEVS}


def summary_vs_reference(cells, ref):
    flip = top1 = 0
    winners = {b: 0 for b in BASELINES}
    taus = []
    rp = {b: i for i, b in enumerate(ref)}
    for op in OPS:
        for s in SEVS:
            rank = cell_rank(cells[(op, s)])
            flip += rank != ref
            top1 += rank[0] != ref[0]
            winners[rank[0]] += 1
            cp = {b: i for i, b in enumerate(rank)}
            t, _ = kendalltau([rp[b] for b in BASELINES],
                              [cp[b] for b in BASELINES])
            taus.append(float(t) if not np.isnan(t) else 0.0)
    return {"flip_count": flip, "top1_change_count": top1,
            "winners": winners, "median_kendall_tau": float(np.median(taus))}


def main():
    print("[upsample] loading clean@s0 cells ...")
    clean_ps, clean_means = {}, {}
    ref_ident = ref_hash = None
    rowdrop = {"clean": {"pred": 0, "gt": 0}}
    for b in BASELINES:
        cd, szp, szg, ident, ch = load_cell(b, "clean", 0)
        if ref_ident is None:
            ref_ident, ref_hash = ident, ch
        elif ident != ref_ident or ch != ref_hash:
            sys.exit(f"[FATAL] {b} clean cell misaligned")
        clean_ps[b] = cd
        clean_means[b] = float(cd.mean() * 1000)
        rowdrop["clean"]["pred"] += szp; rowdrop["clean"]["gt"] += szg
    up_clean_rank = sorted(BASELINES, key=lambda b: clean_means[b])
    clean_shift = {b: {
        "e18_upsample_clean_x1000": round(clean_means[b], 4),
        "stamped_native_x1000": STAMPED_CLEAN_X1000[b],
        "shift_vs_stamped_pct": round((clean_means[b] - STAMPED_CLEAN_X1000[b])
                                      / STAMPED_CLEAN_X1000[b] * 100, 3),
        "table1_repro_x1000": TABLE1_REPRO_CLEAN_X1000[b],
        "shift_vs_table1_pct": round((clean_means[b] - TABLE1_REPRO_CLEAN_X1000[b])
                                     / TABLE1_REPRO_CLEAN_X1000[b] * 100, 3),
    } for b in BASELINES}
    print("  clean x1000:", {b: round(clean_means[b], 3) for b in BASELINES})
    print("  UpSample clean rank (ref-b):", up_clean_rank)

    print("[upsample] loading 80 corruption cells ...")
    per_sample, friedman = {}, {}
    for op in OPS:
        for s in SEVS:
            key = f"{op}_s{s}"
            rowdrop[key] = {"pred": 0, "gt": 0}
            for b in BASELINES:
                cd, szp, szg, ident, ch = load_cell(b, op, s)
                if ident != ref_ident or ch != ref_hash:
                    sys.exit(f"[FATAL] {b} {key} misaligned vs clean reference")
                per_sample[(b, op, s)] = cd
                rowdrop[key]["pred"] += szp; rowdrop[key]["gt"] += szg
            _, p = friedmanchisquare(*[per_sample[(b, op, s)] for b in BASELINES])
            friedman[key] = {"p": float(p),
                             "significant_bonf": bool(p < ALPHA_BONF_FRIEDMAN)}
    fr_sig = sum(v["significant_bonf"] for v in friedman.values())
    tot_p = sum(v["pred"] for v in rowdrop.values())
    tot_g = sum(v["gt"] for v in rowdrop.values())
    print(f"  Friedman sig: {fr_sig}/20 | row-drop pred={tot_p} gt={tot_g}")

    print("[upsample] strict-Bonferroni audit ...")
    audit = {}
    strict_best = ambiguous = st1_a = st1_b = 0
    for op in OPS:
        for s in SEVS:
            means = {b: float(per_sample[(b, op, s)].mean()) for b in BASELINES}
            pw = derive_strict_pairwise({b: per_sample[(b, op, s)]
                                         for b in BASELINES})
            rank = cell_rank(means)
            best = rank[0]
            top = {best} | {o for o in BASELINES if o != best and
                            pw[_pair_key(best, o)]["winner_bonferroni"] == "NS_bonf"}
            strict = len(top) == 1
            audit[f"{op}_s{s}"] = {
                "cell_means_x1000": {b: v * 1000 for b, v in means.items()},
                "mean_rank": rank, "mean_best": best,
                "strict_top_set": sorted(top),
                "is_strict_best_beats_all": strict,
                "pairwise_bonferroni": {f"{a}_vs_{b}": pw[(a, b)]
                                        for (a, b) in pw},
            }
            strict_best += strict; ambiguous += not strict
            st1_a += strict and best != PUBLISHED_RANK[0]
            st1_b += strict and best != up_clean_rank[0]
    print(f"  strict-best {strict_best}/20 | ambiguous {ambiguous}/20 | "
          f"strict-top1 vs published {st1_a}/20, vs clean {st1_b}/20")

    print(f"[upsample] bootstrap n={N_BOOTSTRAP} (clean rank re-estimated "
          f"per replicate for ref-b) ...")
    full = means_from_resample(per_sample, np.arange(1200))
    obs_a = summary_vs_reference(full, PUBLISHED_RANK)
    obs_b = summary_vs_reference(full, up_clean_rank)
    rng = np.random.default_rng(RNG_SEED)
    ba, bb, bbf = [], [], []
    clean_changed = 0
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, 1200, size=1200)
        cells = means_from_resample(per_sample, idx)
        crank = sorted(BASELINES, key=lambda b: float(clean_ps[b][idx].mean()))
        clean_changed += crank != up_clean_rank
        ba.append(summary_vs_reference(cells, PUBLISHED_RANK))
        bb.append(summary_vs_reference(cells, crank))
        bbf.append(summary_vs_reference(cells, up_clean_rank))

    def ci(boots, key, as_int=True):
        arr = np.array([x[key] for x in boots])
        lo, hi = np.quantile(arr, 0.025), np.quantile(arr, 0.975)
        return [int(lo), int(hi)] if as_int else [float(lo), float(hi)]

    print(f"  [ref-a] flip={obs_a['flip_count']}/20 CI{ci(ba,'flip_count')} "
          f"top1={obs_a['top1_change_count']}/20 tau={obs_a['median_kendall_tau']:.3f}")
    print(f"  [ref-b] flip={obs_b['flip_count']}/20 CI{ci(bb,'flip_count')} "
          f"(clean rank changed in {clean_changed}/{N_BOOTSTRAP} replicates)")

    e18_flips = {c for c, e in audit.items() if e["mean_rank"] != PUBLISHED_RANK}
    overlap = None
    if ZEROPAD_REF.is_file():
        zp = json.load(open(ZEROPAD_REF, encoding="utf-8"))
        zp_flips = {c for c, e in zp.get("MAJ1_strict_winner", {}).get(
            "per_cell_audit", {}).items()
            if e.get("mean_rank", []) and e["mean_rank"] != PUBLISHED_RANK}
        overlap = {
            "zero_pad_only": sorted(zp_flips - e18_flips),
            "upsample_only": sorted(e18_flips - zp_flips),
            "shared": sorted(zp_flips & e18_flips),
            "n_zero_pad": len(zp_flips), "n_upsample": len(e18_flips),
            "n_shared": len(zp_flips & e18_flips),
            "note": ("descriptive cell-pattern overlap; independent corruption "
                     "realisations, not paired same-draw ablations"),
        }
        print(f"  overlap: zp={len(zp_flips)} up={len(e18_flips)} "
              f"shared={overlap['n_shared']}")
    else:
        print("  (zero-pad reference JSON absent — run "
              "scripts/instability_summary.py first for the overlap block)")

    out = {
        "version": "UpSamplePoints loader-transform sensitivity (Section V-D / Table 8)",
        "constants": {
            "alpha_bonf_pairwise": ALPHA_BONF_PAIRWISE,
            "alpha_bonf_friedman": ALPHA_BONF_FRIEDMAN,
            "n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED,
            "published_rank_ref_a": PUBLISHED_RANK,
            "upsample_clean_rank_ref_b": up_clean_rank,
            "stamped_clean_x1000": STAMPED_CLEAN_X1000,
            "table1_repro_clean_x1000": TABLE1_REPRO_CLEAN_X1000,
        },
        "clean_shift_table": clean_shift,
        "summary_vs_published_ref_a": {
            **obs_a, "flip_count_ci95": ci(ba, "flip_count"),
            "top1_change_count_ci95": ci(ba, "top1_change_count"),
            "median_kendall_tau_ci95": ci(ba, "median_kendall_tau", False)},
        "summary_vs_upsample_clean_ref_b": {
            **obs_b, "flip_count_ci95": ci(bb, "flip_count"),
            "top1_change_count_ci95": ci(bb, "top1_change_count"),
            "median_kendall_tau_ci95": ci(bb, "median_kendall_tau", False),
            "bootstrap_recomputes_clean_rank": True,
            "clean_rank_changed_in_n_replicates": clean_changed,
            "fixed_ref_diagnostic_ci95": {
                "flip_count": ci(bbf, "flip_count"),
                "top1_change_count": ci(bbf, "top1_change_count")}},
        "strict_winner_summary": {
            "strict_best_beats_all_count": strict_best,
            "ambiguous_top_set_count": ambiguous,
            "strict_top1_change_vs_published": st1_a,
            "strict_top1_change_vs_upsample_clean": st1_b},
        "friedman_per_cell": friedman,
        "friedman_significant_count": fr_sig,
        "rowdrop_audit": {"per_cell": rowdrop, "total_pred_sumzero": tot_p,
                          "total_gt_sumzero": tot_g},
        "per_cell_audit": audit,
        "flip_overlap_vs_zeropad_ref_a": overlap,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[DONE] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

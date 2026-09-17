"""Build matched-control group index for stratified audit.

Protocol:
    For each (taxonomy_id, model_id, view_id) - the "matched group" - generate
    one corrupted variant per (corruption_op, severity) cell. All corrupted
    variants of the same group share:
        - object identity (taxonomy_id + model_id)
        - viewpoint (view_id)
        - point budget N (preserved by all corruption ops)
        - GT (unchanged)
    What varies:
        - the active corruption operator (noise / outlier / density / crop /
          pose / mixed)
        - the severity level (1-5)

This isolates the corruption factor: any performance difference between two
matched-group entries attributes to the corruption, not to dataset
composition or random sampling.

Group index schema (saved as JSON):
    {
        "version": "1.0",
        "dataset": "PCN",
        "subset": "test",
        "n_groups": int,
        "n_eval_cells_per_group": int,   # = #ops x #severities + 1 (clean)
        "ops": ["noise", "outlier", "density", "crop", "pose"],
        "severities": [0, 1, 2, 3, 4, 5],
        "groups": [
            {
                "group_key": "<taxonomy>/<model>/<view>",
                "taxonomy_id": "...",
                "model_id": "...",
                "view_id": int,
                "partial_path": "...",
                "gt_path": "..."
            },
            ...
        ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon, kendalltau

DEFAULT_OPS = ("noise", "outlier", "density", "crop", "pose")
DEFAULT_SEVERITIES = (0, 1, 2, 3, 4, 5)


def build_matched_groups(
    pcn_file_list: Iterable[dict],
    subset: str,
    n_views_per_object: int = 1,
    ops: Sequence[str] = DEFAULT_OPS,
    severities: Sequence[int] = DEFAULT_SEVERITIES,
    max_groups: Optional[int] = None,
) -> Dict:
    """Construct matched-control group index from a PCN file list.

    Args:
        pcn_file_list: iterable of dicts with keys 'taxonomy_id', 'model_id',
            'partial_path' (list or str), 'gt_path' (str). Compatible with
            PoinTr `PCN._get_file_list()` output.
        subset: 'train' | 'test' | 'val'
        n_views_per_object: how many views to include per object (PCN test
            subset has 1 partial view per model; train has 8). Set to 1 for
            test/val to keep groups manageable.
        ops: tuple of corruption-op names; must align with src.corruptions.OP_REGISTRY
        severities: tuple of severity levels; 0 = clean (always included for baseline)
        max_groups: optional cap on group count for smoke-test purposes
    """
    groups: List[Dict] = []
    seen = 0
    for entry in pcn_file_list:
        if max_groups is not None and seen >= max_groups:
            break
        partial_paths = entry["partial_path"]
        if isinstance(partial_paths, str):
            partial_paths = [partial_paths]
        for view_id in range(min(n_views_per_object, len(partial_paths))):
            groups.append({
                "group_key": f"{entry['taxonomy_id']}/{entry['model_id']}/{view_id}",
                "taxonomy_id": entry["taxonomy_id"],
                "model_id": entry["model_id"],
                "view_id": view_id,
                "partial_path": partial_paths[view_id],
                "gt_path": entry["gt_path"],
            })
            seen += 1
            if max_groups is not None and seen >= max_groups:
                break
    n_cells = len(ops) * (len(severities) - 1) + 1  # opsxsev + 1 clean
    return {
        "version": "1.0",
        "dataset": "PCN",
        "subset": subset,
        "n_groups": len(groups),
        "n_eval_cells_per_group": n_cells,
        "ops": list(ops),
        "severities": list(severities),
        "groups": groups,
    }


def save_group_index(index: Dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def load_group_index(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paired_wilcoxon(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank on matched-control scores.

    scores_a, scores_b are aligned per-group score arrays (same length, same
    group order). Returns dict with statistic, p-value, n_pairs, median diff.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    nz = diff[diff != 0]
    if nz.size < 5:
        return {"statistic": float("nan"), "pvalue": float("nan"),
                "n_pairs": int(nz.size), "median_diff": float(np.median(diff))}
    res = wilcoxon(a, b, alternative=alternative)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n_pairs": int(nz.size),
        "median_diff": float(np.median(diff)),
    }


def aggregate_unstratified(
    per_sample_scores: Dict[Tuple[str, str, int, int], float],
    metric_lower_is_better: bool = True,
) -> Dict[str, float]:
    """CPCCD-style aggregate ranking: one mean score per model.

    Input keyed by (model_name, op_name, severity, sample_idx) -> score.
    Returns {model_name: mean_score} averaged across all (op, severity, sample).
    """
    by_model: Dict[str, List[float]] = {}
    for (model, _op, _sev, _idx), score in per_sample_scores.items():
        by_model.setdefault(model, []).append(score)
    return {m: float(np.mean(v)) for m, v in by_model.items()}


def aggregate_stratified(
    per_sample_scores: Dict[Tuple[str, str, int, int], float],
) -> Dict[str, Dict[Tuple[str, int], float]]:
    """Stratified ranking: one mean score per (model x op x severity) cell.

    Returns nested dict {model_name: {(op, severity): mean_score}}.
    """
    by_cell: Dict[str, Dict[Tuple[str, int], List[float]]] = {}
    for (model, op, sev, _idx), score in per_sample_scores.items():
        by_cell.setdefault(model, {}).setdefault((op, sev), []).append(score)
    return {
        m: {cell: float(np.mean(vals)) for cell, vals in cells.items()}
        for m, cells in by_cell.items()
    }


def ranking_kendall_tau(
    ranking_a: List[str],
    ranking_b: List[str],
) -> Dict[str, float]:
    """Kendall tau between two model rankings (lists of model names ordered best->worst).

    Returns {tau: float, pvalue: float, n_models: int, n_flips: int}.
    n_flips = number of inverted pairs (lower bound on disagreement).
    """
    if set(ranking_a) != set(ranking_b):
        raise ValueError(f"ranking sets differ: {set(ranking_a)} vs {set(ranking_b)}")
    name_to_rank_b = {n: i for i, n in enumerate(ranking_b)}
    rank_a_pos = list(range(len(ranking_a)))
    rank_b_pos = [name_to_rank_b[n] for n in ranking_a]
    tau, p = kendalltau(rank_a_pos, rank_b_pos)
    # count discordant pairs
    flips = 0
    for i in range(len(ranking_a)):
        for j in range(i + 1, len(ranking_a)):
            if rank_b_pos[i] > rank_b_pos[j]:
                flips += 1
    return {
        "tau": float(tau),
        "pvalue": float(p),
        "n_models": len(ranking_a),
        "n_flips": int(flips),
    }

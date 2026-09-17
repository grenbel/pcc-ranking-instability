"""Stratified matched-control protocol: group index, paired Wilcoxon, Kendall tau."""

from .matched_control import (
    build_matched_groups,
    save_group_index,
    load_group_index,
    paired_wilcoxon,
    aggregate_unstratified,
    aggregate_stratified,
    ranking_kendall_tau,
)

__all__ = [
    "build_matched_groups", "save_group_index", "load_group_index",
    "paired_wilcoxon", "aggregate_unstratified", "aggregate_stratified",
    "ranking_kendall_tau",
]

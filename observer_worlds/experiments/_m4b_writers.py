"""CSV / JSON writers for M4B sweep outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from observer_worlds.experiments._m4b_sweep import (
    CONDITION_NAMES,
    ConditionResult,
    PairedRecord,
    SUMMARY_METRICS,
    metrics_dict,
)


# ---------------------------------------------------------------------------
# Files written:
#   paired_runs.csv         -- wide, one row per (rule_idx, seed),
#                              columns prefixed by condition name
#   condition_summary.csv   -- long, one row per (rule_idx, seed, condition)
#   candidate_metrics.csv   -- long, one row per (rule_idx, seed, condition,
#                              candidate_index)
#   paired_differences.csv  -- one row per (rule_idx, seed) with all paired
#                              differences (coh-shuf, coh-2d, shuf-2d)
# ---------------------------------------------------------------------------


def write_condition_summary_csv(records: list[PairedRecord], out_path: str | Path) -> None:
    cols = [
        "rule_idx", "seed", "condition", "rule_repr",
        *SUMMARY_METRICS,
        "mean_active", "late_active", "activity_variance",
        "mean_frame_to_frame_change", "projected_hash",
        "best_track_id", "best_age",
        "best_persistence", "best_time_score", "best_memory_score",
        "best_selfhood_score", "best_causality_score", "best_resilience_score",
        "sim_time_seconds", "metric_time_seconds",
    ]
    with Path(out_path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in records:
            for cond in CONDITION_NAMES:
                r: ConditionResult = getattr(rec, cond)
                rule_repr = _short_rule_repr(rec.rule_dict, cond)
                m = metrics_dict(r)
                row = [r.rule_idx, r.seed, r.condition, rule_repr]
                row.extend(m[c] for c in SUMMARY_METRICS)
                row.extend([
                    r.mean_active, r.late_active, r.activity_variance,
                    r.mean_frame_to_frame_change, r.projected_hash,
                    "" if r.best_track_id is None else r.best_track_id,
                    "" if r.best_age is None else r.best_age,
                    _opt(r.best_persistence),
                    _opt(r.best_time_score),
                    _opt(r.best_memory_score),
                    _opt(r.best_selfhood_score),
                    _opt(r.best_causality_score),
                    _opt(r.best_resilience_score),
                    f"{r.sim_time_seconds:.3f}",
                    f"{r.metric_time_seconds:.3f}",
                ])
                w.writerow(row)


def write_paired_runs_csv(records: list[PairedRecord], out_path: str | Path) -> None:
    """Wide format: one row per (rule_idx, seed) with each metric repeated for
    coherent_4d, shuffled_4d, matched_2d."""
    base_cols = ["rule_idx", "seed", "rule_repr"]
    metric_cols = []
    for cond in CONDITION_NAMES:
        for m in SUMMARY_METRICS:
            metric_cols.append(f"{cond}_{m}")
    with Path(out_path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(base_cols + metric_cols)
        for rec in records:
            row = [rec.rule_idx, rec.seed, _short_rule_repr(rec.rule_dict, "coherent_4d")]
            for cond in CONDITION_NAMES:
                r: ConditionResult = getattr(rec, cond)
                m = metrics_dict(r)
                for col in SUMMARY_METRICS:
                    row.append(_fmt(m[col]))
            w.writerow(row)


def write_paired_differences_csv(records: list[PairedRecord], out_path: str | Path) -> None:
    """For each (rule, seed): coherent - shuffled, coherent - 2d, shuffled - 2d
    for every summary metric."""
    cols = ["rule_idx", "seed", "rule_repr"]
    diff_pairs = (("coherent_4d", "shuffled_4d"), ("coherent_4d", "matched_2d"),
                  ("shuffled_4d", "matched_2d"))
    for a, b in diff_pairs:
        for m in SUMMARY_METRICS:
            cols.append(f"{a}_minus_{b}_{m}")
    with Path(out_path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in records:
            row = [rec.rule_idx, rec.seed, _short_rule_repr(rec.rule_dict, "coherent_4d")]
            for a, b in diff_pairs:
                ma = metrics_dict(getattr(rec, a))
                mb = metrics_dict(getattr(rec, b))
                for m in SUMMARY_METRICS:
                    row.append(_fmt(ma[m] - mb[m]))
            w.writerow(row)


def write_candidate_metrics_csv(records: list[PairedRecord], out_path: str | Path) -> None:
    """Long format: one row per candidate observed in any (rule, seed,
    condition).  Captures combined score + per-component breakdown when
    they're stored in the ConditionResult.  This file is for pivot-style
    downstream analysis."""
    cols = ["rule_idx", "seed", "condition", "rule_repr",
            "candidate_idx_in_run", "combined_score", "age", "mean_area"]
    with Path(out_path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in records:
            for cond in CONDITION_NAMES:
                r: ConditionResult = getattr(rec, cond)
                rule_repr = _short_rule_repr(rec.rule_dict, cond)
                # all_combined_scores is parallel to all_ages and all_mean_areas.
                ages = r.all_ages or [0] * len(r.all_combined_scores)
                areas = r.all_mean_areas or [0.0] * len(r.all_combined_scores)
                for i, score in enumerate(r.all_combined_scores):
                    age = ages[i] if i < len(ages) else 0
                    area = areas[i] if i < len(areas) else 0.0
                    w.writerow([
                        rec.rule_idx, rec.seed, cond, rule_repr,
                        i, _fmt(score), age, _fmt(area),
                    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_rule_repr(rule_dict: dict, cond: str) -> str:
    if cond == "matched_2d":
        return "life_b3s23"
    return (
        f"B[{rule_dict.get('birth_min', 0):.2f},{rule_dict.get('birth_max', 0):.2f}]"
        f"_S[{rule_dict.get('survive_min', 0):.2f},{rule_dict.get('survive_max', 0):.2f}]"
        f"_d{rule_dict.get('initial_density', 0):.2f}"
    )


def _fmt(x: float) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.6f}"


def _opt(x: float | None) -> str:
    return "" if x is None else f"{float(x):.6f}"

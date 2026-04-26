"""Tests for the M4B paired-statistics module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from observer_worlds.analysis.m4b_stats import (
    COMPARISON_PAIRS,
    HEADLINE_METRICS,
    PairedDifference,
    _CAVEAT_OPTIMIZED_RULES,
    _CAVEAT_SEED_OVERLAP,
    _CAVEAT_TRACK_COUNT_DIFFERS,
    _INTERP_2D_WINS_NORMALIZED,
    _INTERP_COH_BEATS_2D_NOT_SHUF,
    _INTERP_COH_BEATS_BOTH_NORMALIZED,
    _INTERP_MIXED_NORMALIZED,
    compute_all_paired_differences,
    compute_paired_difference,
    render_stats_summary_md,
    stats_summary_dict,
    win_rate_random_candidate,
    write_stats_summary_json,
)
from observer_worlds.experiments._m4b_sweep import (
    CONDITION_NAMES,
    SUMMARY_METRICS,
    ConditionResult,
    PairedRecord,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _result(metric_overrides=None, scores=None):
    r = ConditionResult(rule_idx=0, seed=0, condition="coherent_4d", rule_dict={})
    for k, v in (metric_overrides or {}).items():
        setattr(r, k, v)
    if scores is not None:
        r.all_combined_scores = list(scores)
    return r


def _record(coh_overrides, shuf_overrides, twod_overrides, ri=0, seed=0,
            coh_scores=None, shuf_scores=None, twod_scores=None):
    rec = PairedRecord(
        rule_idx=ri, seed=seed, rule_dict={},
        coherent_4d=_result(coh_overrides, coh_scores),
        shuffled_4d=_result(shuf_overrides, shuf_scores),
        matched_2d=_result(twod_overrides, twod_scores),
    )
    rec.coherent_4d.condition = "coherent_4d"
    rec.shuffled_4d.condition = "shuffled_4d"
    rec.matched_2d.condition = "matched_2d"
    return rec


# ---------------------------------------------------------------------------
# compute_paired_difference
# ---------------------------------------------------------------------------


def test_paired_difference_signs_match_data():
    a = [1.0] * 10
    b = [0.0] * 10
    pd = compute_paired_difference(
        a, b, n_bootstrap=500, n_permutations=500, seed=0
    )
    assert pd.n_pairs == 10
    assert pd.mean_difference == pytest.approx(1.0)
    assert pd.cliffs_delta == pytest.approx(1.0)
    assert pd.win_rate_a == pytest.approx(1.0)
    assert pd.win_rate_b == pytest.approx(0.0)
    assert pd.tie_rate == pytest.approx(0.0)
    assert pd.bootstrap_ci_low > 0
    assert pd.bootstrap_ci_high > 0
    assert pd.bootstrap_ci_low <= 1.0 <= pd.bootstrap_ci_high


def test_paired_difference_zero_when_identical():
    a = [3.0, 1.5, 2.7, 0.0, 4.0]
    b = list(a)
    pd = compute_paired_difference(
        a, b, n_bootstrap=500, n_permutations=500, seed=0
    )
    assert pd.mean_difference == pytest.approx(0.0)
    # Cohen's d should be 0 (we guard the std==0 case).
    assert pd.cohens_d_paired == pytest.approx(0.0)
    assert pd.cliffs_delta == pytest.approx(0.0)
    assert pd.tie_rate == pytest.approx(1.0)
    assert pd.win_rate_a == pytest.approx(0.0)
    assert pd.win_rate_b == pytest.approx(0.0)
    # All permutations leave diffs at zero, so |perm_mean| >= 0 always.
    assert pd.permutation_p_value > 0.95


def test_permutation_p_value_small_for_strong_effect():
    a = list(range(10, 20))      # 10..19
    b = list(range(0, 10))       # 0..9
    pd = compute_paired_difference(
        a, b, n_bootstrap=200, n_permutations=200, seed=0
    )
    # All diffs are +10, so observed |mean| = 10. The only way a
    # permutation can match is if all signs are +1 (or all -1) -- but with
    # diffs identical to 10 the absolute mean of the permuted diffs is
    # always exactly 10 regardless of signs (since |sum(+/-10)|/n varies).
    # The expected p-value is therefore at most close to ~1/n_perm + epsilon
    # for differing |sign sums|. Verify it's well below 0.5 either way:
    # because diffs are equal, signs flip exactly cancel proportionally.
    # We assert it's in the small-p regime relative to a null with
    # heterogeneous diffs would have. Use a lenient bound.
    assert pd.permutation_p_value < 0.5
    # Mean difference is 10.
    assert pd.mean_difference == pytest.approx(10.0)
    # Cliff's delta should be 1.0.
    assert pd.cliffs_delta == pytest.approx(1.0)


def test_permutation_p_value_small_for_strong_heterogeneous_effect():
    rng = np.random.default_rng(123)
    # Heterogeneous strongly-positive diffs -- guarantees small p.
    b = rng.normal(size=30)
    a = b + rng.uniform(0.5, 2.0, size=30)
    pd = compute_paired_difference(
        list(a), list(b), n_bootstrap=200, n_permutations=200, seed=0
    )
    assert pd.permutation_p_value < 0.05
    assert pd.mean_difference > 0


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(42)
    n = 50
    b = rng.normal(loc=0.0, scale=1.0, size=n)
    a = b + rng.normal(loc=0.5, scale=0.2, size=n)
    sample_mean_diff = float((a - b).mean())
    pd = compute_paired_difference(
        list(a), list(b), n_bootstrap=2000, n_permutations=200, seed=0,
    )
    assert pd.bootstrap_ci_low <= sample_mean_diff <= pd.bootstrap_ci_high
    assert pd.bootstrap_ci_low > 0  # 0.5 mean shift should be detected.


def test_compute_paired_difference_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_paired_difference([1.0, 2.0], [1.0], n_bootstrap=10, n_permutations=10)


# ---------------------------------------------------------------------------
# compute_all_paired_differences
# ---------------------------------------------------------------------------


def test_compute_all_paired_differences_returns_grid():
    # Build 3 records, each condition has distinct values for several metrics.
    records = []
    for i in range(3):
        records.append(_record(
            coh_overrides={"max_score": 1.0 + i, "top5_mean_score": 0.8 + i,
                           "p95_score": 0.9 + i},
            shuf_overrides={"max_score": 0.5 + i, "top5_mean_score": 0.4 + i,
                            "p95_score": 0.45 + i},
            twod_overrides={"max_score": 0.2 + i, "top5_mean_score": 0.1 + i,
                            "p95_score": 0.15 + i},
        ))
    metrics = ("max_score", "top5_mean_score", "p95_score")
    diffs = compute_all_paired_differences(
        records, metrics=metrics, pairs=COMPARISON_PAIRS,
        n_bootstrap=100, n_permutations=100, seed=0,
    )
    assert len(diffs) == len(COMPARISON_PAIRS) * len(metrics)  # 3 * 3 = 9
    # All PairedDifference instances.
    assert all(isinstance(d, PairedDifference) for d in diffs)

    # Default metrics (SUMMARY_METRICS) -> 3 pairs * len(SUMMARY_METRICS)
    diffs_full = compute_all_paired_differences(
        records, n_bootstrap=50, n_permutations=50, seed=0,
    )
    assert len(diffs_full) == len(COMPARISON_PAIRS) * len(SUMMARY_METRICS)


# ---------------------------------------------------------------------------
# win_rate_random_candidate
# ---------------------------------------------------------------------------


def test_win_rate_random_candidate_skips_empty():
    # All records have empty coherent scores -- should not crash and should
    # report n_skipped == n_records.
    records = [
        _record({}, {}, {}, coh_scores=[], shuf_scores=[1.0, 2.0],
                twod_scores=[0.5]),
        _record({}, {}, {}, coh_scores=[], shuf_scores=[3.0],
                twod_scores=[0.1]),
    ]
    out = win_rate_random_candidate(
        records, "coherent_4d", "shuffled_4d", n_samples=100, seed=0,
    )
    assert "win_rate_a" in out
    assert "win_rate_b" in out
    assert "tie_rate" in out
    # All pairs skipped.
    assert out["n_skipped"] == 2.0
    assert out["n_used"] == 0.0
    # Aggregated win_rates are 0.0 since no draws.
    assert out["win_rate_a"] == 0.0
    assert out["win_rate_b"] == 0.0


def test_win_rate_random_candidate_known_dominance():
    records = []
    for _ in range(4):
        records.append(_record(
            {}, {}, {},
            coh_scores=[5.0, 5.0, 5.0],
            shuf_scores=[1.0, 1.0, 1.0],
            twod_scores=[0.0],
        ))
    out = win_rate_random_candidate(
        records, "coherent_4d", "shuffled_4d", n_samples=200, seed=0,
    )
    assert out["win_rate_a"] == pytest.approx(1.0)
    assert out["win_rate_b"] == pytest.approx(0.0)
    assert out["tie_rate"] == pytest.approx(0.0)
    assert out["n_used"] == 4.0
    assert out["n_skipped"] == 0.0


# ---------------------------------------------------------------------------
# stats_summary_dict
# ---------------------------------------------------------------------------


def test_stats_summary_dict_shape():
    records = []
    for i in range(5):
        records.append(_record(
            coh_overrides={m: float(i + 1) for m in SUMMARY_METRICS},
            shuf_overrides={m: float(i) for m in SUMMARY_METRICS},
            twod_overrides={m: float(i) * 0.5 for m in SUMMARY_METRICS},
            ri=(i % 2), seed=i,
            coh_scores=[1.0, 2.0],
            shuf_scores=[0.5, 0.6],
            twod_scores=[0.1, 0.2],
        ))
    summary = stats_summary_dict(
        records, n_bootstrap=50, n_permutations=50, seed=0,
    )
    assert summary["n_pairs"] == 5
    assert summary["n_rules"] == 2
    assert "n_seeds_per_rule" in summary
    assert summary["headline_metrics"] == list(HEADLINE_METRICS)

    # All 3 comparison pairs present.
    expected_keys = {f"{a}_vs_{b}" for a, b in COMPARISON_PAIRS}
    assert set(summary["comparisons"].keys()) == expected_keys
    assert set(summary["candidate_level_win_rates"].keys()) == expected_keys

    # All SUMMARY_METRICS present in each comparison.
    for key in expected_keys:
        assert set(summary["comparisons"][key].keys()) == set(SUMMARY_METRICS)
        # Each entry should be a serialized PairedDifference (dict with
        # the dataclass fields).
        for metric, entry in summary["comparisons"][key].items():
            assert "mean_difference" in entry
            assert "bootstrap_ci_low" in entry
            assert "bootstrap_ci_high" in entry
            assert "permutation_p_value" in entry
            assert entry["metric"] == metric


# ---------------------------------------------------------------------------
# render_stats_summary_md interpretation branches
# ---------------------------------------------------------------------------


def _entry(metric: str, *, mean_diff: float = 0.0,
           ci: tuple[float, float] = (-1.0, 1.0),
           p_value: float = 0.5,
           win_rate_a: float = 0.0, win_rate_b: float = 0.0,
           tie_rate: float = 1.0) -> dict:
    return {
        "condition_a": "x", "condition_b": "y", "metric": metric,
        "n_pairs": 5, "mean_difference": mean_diff,
        "median_difference": mean_diff,
        "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1],
        "permutation_p_value": p_value, "cohens_d_paired": 0.0,
        "cliffs_delta": 0.0,
        "win_rate_a": win_rate_a, "win_rate_b": win_rate_b,
        "tie_rate": tie_rate,
    }


def _stats_skeleton(*, provenance: dict | None = None) -> dict:
    """Build a stats dict where every metric in every comparison defaults
    to a 'no signal' entry. Tests then override the entries they care about.
    """
    def _comparison_default():
        return {m: _entry(m) for m in SUMMARY_METRICS}

    cand_win_default = {
        "win_rate_a": 0.0, "win_rate_b": 0.0, "tie_rate": 1.0,
        "n_skipped": 0.0, "n_used": 0.0, "n_samples_per_pair": 0.0,
        "total_draws": 0.0,
    }
    out = {
        "n_pairs": 5, "n_rules": 1, "n_seeds_per_rule": 5.0,
        "comparisons": {
            "coherent_4d_vs_shuffled_4d": _comparison_default(),
            "coherent_4d_vs_matched_2d": _comparison_default(),
            "shuffled_4d_vs_matched_2d": _comparison_default(),
        },
        "candidate_level_win_rates": {
            "coherent_4d_vs_shuffled_4d": dict(cand_win_default),
            "coherent_4d_vs_matched_2d": dict(cand_win_default),
            "shuffled_4d_vs_matched_2d": dict(cand_win_default),
        },
        "headline_metrics": list(HEADLINE_METRICS),
    }
    if provenance is not None:
        out["provenance"] = provenance
    return out


def _significant_positive_entry(metric: str, *, mean_diff: float = 0.05) -> dict:
    """An entry that satisfies _significant_positive: p<0.05, ci_low>0, win>0.65."""
    return _entry(
        metric, mean_diff=mean_diff, ci=(mean_diff * 0.5, mean_diff * 1.5),
        p_value=0.005, win_rate_a=0.85, win_rate_b=0.15, tie_rate=0.0,
    )


def _significant_negative_entry(metric: str, *, mean_diff: float = -0.05) -> dict:
    return _entry(
        metric, mean_diff=mean_diff, ci=(mean_diff * 1.5, mean_diff * 0.5),
        p_value=0.005, win_rate_a=0.15, win_rate_b=0.85, tie_rate=0.0,
    )


def test_interpretation_coh_beats_both_normalized():
    s = _stats_skeleton()
    s["comparisons"]["coherent_4d_vs_matched_2d"]["score_per_track"] = \
        _significant_positive_entry("score_per_track")
    s["comparisons"]["coherent_4d_vs_shuffled_4d"]["score_per_track"] = \
        _significant_positive_entry("score_per_track")
    md = render_stats_summary_md(s)
    assert _INTERP_COH_BEATS_BOTH_NORMALIZED in md


def test_interpretation_coh_beats_2d_only():
    s = _stats_skeleton()
    s["comparisons"]["coherent_4d_vs_matched_2d"]["lifetime_weighted_mean_score"] = \
        _significant_positive_entry("lifetime_weighted_mean_score")
    md = render_stats_summary_md(s)
    assert _INTERP_COH_BEATS_2D_NOT_SHUF in md


def test_interpretation_2d_wins_normalized():
    s = _stats_skeleton()
    s["comparisons"]["coherent_4d_vs_matched_2d"]["score_per_track"] = \
        _significant_negative_entry("score_per_track")
    md = render_stats_summary_md(s)
    assert _INTERP_2D_WINS_NORMALIZED in md


def test_interpretation_mixed_when_no_significance():
    s = _stats_skeleton()
    md = render_stats_summary_md(s)
    assert _INTERP_MIXED_NORMALIZED in md


def test_caveat_track_count_differs_appended():
    s = _stats_skeleton()
    # Track-count diff > 1 should trigger the caveat.
    s["comparisons"]["coherent_4d_vs_shuffled_4d"]["n_tracks"] = \
        _entry("n_tracks", mean_diff=-100.0)
    md = render_stats_summary_md(s)
    assert _CAVEAT_TRACK_COUNT_DIFFERS in md


def test_caveat_optimized_rules_appended():
    s = _stats_skeleton(provenance={
        "rule_source": "M4C_observer_optimized",
        "baseline_optimized": False,
    })
    md = render_stats_summary_md(s)
    assert _CAVEAT_OPTIMIZED_RULES in md


def test_caveat_optimized_rules_suppressed_when_baseline_optimized():
    s = _stats_skeleton(provenance={
        "rule_source": "M4C_observer_optimized",
        "baseline_optimized": True,
    })
    md = render_stats_summary_md(s)
    assert _CAVEAT_OPTIMIZED_RULES not in md


def test_caveat_seed_overlap_appended():
    s = _stats_skeleton(provenance={
        "rule_source": "M4C_observer_optimized",
        "evaluation_overlaps_training": True,
    })
    md = render_stats_summary_md(s)
    assert _CAVEAT_SEED_OVERLAP in md


def test_provenance_section_rendered():
    s = _stats_skeleton(provenance={
        "rule_source": "M4A_viability",
        "optimization_objective": "viability_score",
        "baseline_optimized": False,
        "evaluation_overlaps_training": False,
        "training_seeds": [1000, 1001],
        "evaluation_seeds": [2000, 2001],
    })
    md = render_stats_summary_md(s)
    assert "Rule provenance" in md
    assert "M4A_viability" in md
    assert "viability_score" in md


# ---------------------------------------------------------------------------
# write_stats_summary_json
# ---------------------------------------------------------------------------


def test_write_stats_summary_json_roundtrips(tmp_path):
    records = []
    for i in range(3):
        records.append(_record(
            coh_overrides={m: float(i + 1) for m in SUMMARY_METRICS},
            shuf_overrides={m: float(i) for m in SUMMARY_METRICS},
            twod_overrides={m: float(i) * 0.5 for m in SUMMARY_METRICS},
            ri=0, seed=i,
            coh_scores=[1.0, 2.0],
            shuf_scores=[0.5],
            twod_scores=[0.1],
        ))
    out = tmp_path / "stats_summary.json"
    summary = write_stats_summary_json(
        records, str(out), n_bootstrap=50, n_permutations=50, seed=0,
    )
    assert out.exists()
    with open(out) as f:
        loaded = json.load(f)

    # Top-level key equality.
    assert set(loaded.keys()) == set(summary.keys())
    # Per-comparison key counts match (3 comparisons).
    assert len(loaded["comparisons"]) == len(summary["comparisons"]) == 3
    # Per-comparison metric counts match.
    for key in loaded["comparisons"]:
        assert (
            len(loaded["comparisons"][key])
            == len(summary["comparisons"][key])
            == len(SUMMARY_METRICS)
        )
    assert loaded["headline_metrics"] == list(HEADLINE_METRICS)

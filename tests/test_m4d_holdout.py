"""Tests for the M4D held-out validation CLI + combined interpretation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observer_worlds.experiments.run_m4d_holdout_validation import (
    COMBINED_BEAT_BOTH,
    COMBINED_BEAT_FIXED_NOT_OPTIMIZED,
    COMBINED_MIXED,
    COMBINED_ONLY_PASS_A_NEGATIVE,
    COMBINED_ONLY_PASS_A_POSITIVE,
    COMBINED_OPTIMIZED_2D_WINS,
    combined_interpretation,
    main,
)


# Reuse the stats-skeleton helpers pattern from test_m4b_stats.py.

def _entry(metric, *, mean_diff=0.0, ci=(-1.0, 1.0), p=0.5,
           wa=0.0, wb=0.0, tie=1.0):
    return {
        "condition_a": "x", "condition_b": "y", "metric": metric,
        "n_pairs": 5, "mean_difference": mean_diff,
        "median_difference": mean_diff,
        "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1],
        "permutation_p_value": p, "cohens_d_paired": 0.0,
        "cliffs_delta": 0.0, "win_rate_a": wa, "win_rate_b": wb,
        "tie_rate": tie,
    }


def _stats_with_coh_vs_2d(*, sig_positive=False, sig_negative=False) -> dict:
    """Build a minimal stats dict where coh_vs_2d shows the requested signal
    on score_per_track."""
    if sig_positive:
        spt = _entry("score_per_track", mean_diff=0.05,
                     ci=(0.02, 0.08), p=0.005, wa=0.85, wb=0.15, tie=0.0)
    elif sig_negative:
        spt = _entry("score_per_track", mean_diff=-0.05,
                     ci=(-0.08, -0.02), p=0.005, wa=0.15, wb=0.85, tie=0.0)
    else:
        spt = _entry("score_per_track")
    return {
        "comparisons": {
            "coherent_4d_vs_matched_2d": {
                "score_per_track": spt,
                "lifetime_weighted_mean_score": _entry("lifetime_weighted_mean_score"),
            },
        }
    }


# ---------------------------------------------------------------------------
# combined_interpretation branches
# ---------------------------------------------------------------------------


def test_combined_only_pass_a_positive():
    a = _stats_with_coh_vs_2d(sig_positive=True)
    msg = combined_interpretation(a, None)
    assert msg == COMBINED_ONLY_PASS_A_POSITIVE


def test_combined_only_pass_a_negative():
    a = _stats_with_coh_vs_2d(sig_negative=False)  # no signal
    msg = combined_interpretation(a, None)
    assert msg == COMBINED_ONLY_PASS_A_NEGATIVE


def test_combined_beat_both_when_both_significant():
    a = _stats_with_coh_vs_2d(sig_positive=True)
    b = _stats_with_coh_vs_2d(sig_positive=True)
    assert combined_interpretation(a, b) == COMBINED_BEAT_BOTH


def test_combined_beat_fixed_not_optimized():
    a = _stats_with_coh_vs_2d(sig_positive=True)
    b = _stats_with_coh_vs_2d()  # no signal in B
    assert combined_interpretation(a, b) == COMBINED_BEAT_FIXED_NOT_OPTIMIZED


def test_combined_optimized_2d_wins():
    a = _stats_with_coh_vs_2d()  # no signal in A
    b = _stats_with_coh_vs_2d(sig_negative=True)
    assert combined_interpretation(a, b) == COMBINED_OPTIMIZED_2D_WINS


def test_combined_mixed():
    a = _stats_with_coh_vs_2d()
    b = _stats_with_coh_vs_2d()
    assert combined_interpretation(a, b) == COMBINED_MIXED


# ---------------------------------------------------------------------------
# Seed-overlap guard
# ---------------------------------------------------------------------------


def _write_minimal_rules_json(path: Path) -> None:
    """Write a fake leaderboard with one valid rule entry."""
    payload = [{
        "rule": {
            "birth_min": 0.20, "birth_max": 0.30,
            "survive_min": 0.20, "survive_max": 0.40,
            "initial_density": 0.20,
        },
        "viability_score": 1.0,
    }]
    path.write_text(json.dumps(payload))


def test_seed_overlap_aborts_without_allow_overlap(tmp_path: Path):
    rules_path = tmp_path / "rules.json"
    _write_minimal_rules_json(rules_path)
    rc = main([
        "--rules-from", str(rules_path),
        "--n-rules", "1",
        "--seeds", "2",
        "--base-eval-seed", "1000",
        "--training-seeds", "1000", "1001",
        "--quick",
        "--out-root", str(tmp_path),
    ])
    assert rc != 0


# ---------------------------------------------------------------------------
# End-to-end tiny smoke
# ---------------------------------------------------------------------------


def test_run_quick_smoke(tmp_path: Path):
    rules_path = tmp_path / "rules.json"
    # Use a known-viable rule (M4A's top rule).
    payload = [{
        "rule": {
            "birth_min": 0.15, "birth_max": 0.26,
            "survive_min": 0.09, "survive_max": 0.38,
            "initial_density": 0.15,
        },
        "viability_score": 5.0,
    }]
    rules_path.write_text(json.dumps(payload))
    rc = main([
        "--rules-from", str(rules_path),
        "--n-rules", "1",
        "--seeds", "2",
        "--base-eval-seed", "5000",
        "--training-seeds", "1000", "1001",
        "--timesteps", "40",
        "--grid", "16", "16", "4", "4",
        "--rollout-steps", "3",
        "--n-bootstrap", "200",
        "--n-permutations", "200",
        "--video-frames-kept", "0",
        "--snapshots-per-run", "1",
        "--backend", "numpy",
        "--out-root", str(tmp_path),
        "--label", "m4d_test",
    ])
    assert rc == 0
    # Find the run dir.
    run_dirs = list(tmp_path.glob("m4d_test_*"))
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert (rd / "summary.md").exists()
    assert (rd / "vs_fixed_2d" / "stats_summary.json").exists()
    assert (rd / "vs_fixed_2d" / "paired_runs.csv").exists()
    md = (rd / "summary.md").read_text()
    assert "M4D" in md
    assert "Pass A" in md
    # Pass B was not requested.
    assert not (rd / "vs_optimized_2d").exists()

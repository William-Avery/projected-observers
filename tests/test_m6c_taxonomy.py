"""Tests for M6C: hidden feature extraction, taxonomy core, threshold
audit, grouped CV regression, and ablation interventions.

Combines what the spec lists as five separate test files into one
file for a tighter suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observer_worlds.analysis.hidden_features import (
    HIDDEN_FEATURE_NAMES,
    candidate_hidden_features,
    column_features,
    temporal_hidden_features,
)
from observer_worlds.analysis.m6c_stats import (
    feature_outcome_correlations,
    grouped_cv_regression,
    rows_to_matrix,
    threshold_artifact_audit,
)
from observer_worlds.experiments._m6c_taxonomy import (
    ABLATION_TYPES,
    M6CRow,
    apply_temporal_history_swap_intervention,
    measure_candidate,
)
from observer_worlds.search import FractionalRule
from observer_worlds.worlds import CA4D, project


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _viable_snapshot(seed: int = 1000):
    rule = FractionalRule(0.15, 0.26, 0.09, 0.38, 0.15)
    bsrule = rule.to_bsrule()
    ca = CA4D(shape=(8, 8, 2, 2), rule=bsrule, backend="numpy")
    ca.initialize_random(0.15, np.random.default_rng(seed))
    for _ in range(15):
        ca.step()
    return ca.state.copy(), bsrule


def _disc_mask(half=2):
    m = np.zeros((8, 8), dtype=bool)
    cy = cx = 4
    m[cy - half:cy + half, cx - half:cx + half] = True
    return m


# ---------------------------------------------------------------------------
# Hidden feature extraction
# ---------------------------------------------------------------------------


def test_column_features_zero_active_returns_zero_entropy():
    fiber = np.zeros((4, 4), dtype=np.uint8)
    f = column_features(fiber)
    assert f["active_count"] == 0
    assert f["active_fraction"] == 0.0
    assert f["hidden_entropy"] == 0.0
    assert f["projection_value"] == 0
    assert f["threshold_margin"] == 0.5


def test_column_features_full_active_returns_zero_entropy():
    fiber = np.ones((4, 4), dtype=np.uint8)
    f = column_features(fiber)
    assert f["active_fraction"] == 1.0
    assert f["hidden_entropy"] == 0.0
    assert f["projection_value"] == 1


def test_column_features_half_active_max_entropy():
    fiber = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],
                     dtype=np.uint8)
    f = column_features(fiber)
    assert f["active_count"] == 8
    assert f["active_fraction"] == 0.5
    assert f["hidden_entropy"] == pytest.approx(1.0)
    assert f["threshold_margin"] == 0.0


def test_candidate_features_all_keys_present():
    state, _ = _viable_snapshot()
    feats = candidate_hidden_features(state, _disc_mask())
    for key in (
        "n_columns", "mean_active_fraction", "near_threshold_fraction",
        "mean_hidden_entropy", "hidden_heterogeneity",
        "hidden_connectedness_across_columns",
        "mean_projection_flip_probability",
    ):
        assert key in feats


def test_candidate_features_empty_mask_returns_zeros():
    state, _ = _viable_snapshot()
    feats = candidate_hidden_features(state, np.zeros((8, 8), dtype=bool))
    assert feats["n_columns"] == 0


def test_temporal_persistence_maxes_when_state_unchanged():
    state, _ = _viable_snapshot()
    mask = _disc_mask()
    # Two identical snapshots → persistence = 1.0, volatility = 0.
    tf = temporal_hidden_features([state, state.copy()], mask,
                                  snapshot_times=[0, 5])
    assert tf["hidden_temporal_persistence"] == pytest.approx(1.0)
    assert tf["hidden_temporal_volatility"] == pytest.approx(0.0)


def test_temporal_persistence_zero_when_state_inverted():
    state, _ = _viable_snapshot()
    mask = _disc_mask()
    tf = temporal_hidden_features([state, 1 - state], mask,
                                  snapshot_times=[0, 5])
    # All bits flipped under mask → hamming distance = 1.0 per column → persistence = 0
    assert tf["hidden_temporal_persistence"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Threshold-margin sanity
# ---------------------------------------------------------------------------


def test_threshold_margin_increases_with_distance_from_half():
    """A column with active_fraction = 0.75 should have threshold_margin > a
    column with active_fraction = 0.55."""
    a = np.zeros((4, 4), dtype=np.uint8); a.flat[:12] = 1   # 12/16 = 0.75
    b = np.zeros((4, 4), dtype=np.uint8); b.flat[:9] = 1    # 9/16 ≈ 0.56
    fa = column_features(a)
    fb = column_features(b)
    assert fa["threshold_margin"] > fb["threshold_margin"]


def test_projection_flip_probability_higher_when_near_threshold():
    """A column 1 cell from threshold should have higher flip probability
    than one many cells away."""
    Nz, Nw = 8, 8
    n_total = Nz * Nw
    # Column with 33/64 active (just above threshold).
    near = np.zeros((Nz, Nw), dtype=np.uint8); near.flat[:33] = 1
    # Column with 50/64 active (well above threshold).
    far = np.zeros((Nz, Nw), dtype=np.uint8); far.flat[:50] = 1
    state = np.zeros((4, 4, Nz, Nw), dtype=np.uint8)
    state[1, 1] = near
    state[1, 2] = far
    mask = np.zeros((4, 4), dtype=bool); mask[1, 1] = True
    f_near = candidate_hidden_features(state, mask)
    mask2 = np.zeros((4, 4), dtype=bool); mask2[1, 2] = True
    f_far = candidate_hidden_features(state, mask2)
    assert f_near["mean_projection_flip_probability"] > \
        f_far["mean_projection_flip_probability"]


# ---------------------------------------------------------------------------
# Threshold-artifact audit
# ---------------------------------------------------------------------------


def _fake_row(*, rule_id, candidate_id, near_thresh_frac, mean_margin,
              future_div, vs_sham, vs_far, horizon=10):
    return M6CRow(
        rule_id=rule_id, rule_source="test", seed=0,
        candidate_id=candidate_id, snapshot_t=10, horizon=horizon,
        candidate_area=10.0, candidate_lifetime=20, observer_score=0.5,
        features={
            "near_threshold_fraction": near_thresh_frac,
            "mean_threshold_margin": mean_margin,
            **{fn: 0.0 for fn in HIDDEN_FEATURE_NAMES if fn not in
               ("near_threshold_fraction", "mean_threshold_margin")},
        },
        future_div_hidden_invisible=future_div,
        local_div_hidden_invisible=future_div,
        future_div_sham=0.0, local_div_far_hidden=0.0,
        hidden_vs_sham_delta=vs_sham, hidden_vs_far_delta=vs_far,
        future_div_visible=future_div * 1.1,
        hidden_vs_visible_ratio=0.9, survival_delta=0.0, HCE=future_div,
        ablation_future_div={t: future_div for t in ABLATION_TYPES},
    )


def test_threshold_audit_filters_rows_correctly():
    rows = [
        _fake_row(rule_id="A", candidate_id=1, near_thresh_frac=0.5,
                 mean_margin=0.05, future_div=0.10, vs_sham=0.10, vs_far=0.05),
        _fake_row(rule_id="A", candidate_id=2, near_thresh_frac=0.05,
                 mean_margin=0.20, future_div=0.05, vs_sham=0.05, vs_far=0.02),
        _fake_row(rule_id="B", candidate_id=1, near_thresh_frac=0.0,
                 mean_margin=0.30, future_div=0.04, vs_sham=0.04, vs_far=0.01),
    ]
    audit = threshold_artifact_audit(rows, horizon=10)
    by_filter = {a["filter"]: a for a in audit}
    # All candidates row.
    assert by_filter["all_candidates"]["n_candidates"] == 3
    # Near-threshold-fraction < 0.10 filter should keep cands 2 and 3.
    assert by_filter["near_threshold_fraction<0.1"]["n_candidates"] == 2
    # mean_threshold_margin > 0.10 filter keeps cands 2 and 3.
    assert by_filter["mean_threshold_margin>0.10"]["n_candidates"] == 2


def test_threshold_audit_mean_future_div_drops_with_filter():
    """If we filter to far-from-threshold candidates, mean_future_div
    should drop in this fixture."""
    rows = [
        _fake_row(rule_id="A", candidate_id=1, near_thresh_frac=0.8,
                 mean_margin=0.02, future_div=0.20, vs_sham=0.20, vs_far=0.10),
        _fake_row(rule_id="A", candidate_id=2, near_thresh_frac=0.0,
                 mean_margin=0.30, future_div=0.05, vs_sham=0.05, vs_far=0.02),
    ]
    audit = threshold_artifact_audit(rows, horizon=10)
    by_filter = {a["filter"]: a for a in audit}
    assert by_filter["all_candidates"]["mean_future_div"] > \
        by_filter["mean_threshold_margin>0.10"]["mean_future_div"]


# ---------------------------------------------------------------------------
# Correlation table
# ---------------------------------------------------------------------------


def test_feature_outcome_correlations_returns_grid():
    rows = [_fake_row(rule_id="A", candidate_id=i, near_thresh_frac=0.1 * i,
                     mean_margin=0.05 * i, future_div=0.01 * i,
                     vs_sham=0.01 * i, vs_far=0.005 * i)
            for i in range(1, 6)]
    cors = feature_outcome_correlations(rows, horizon=10)
    # n_features × n_outcomes entries.
    assert len(cors) >= len(HIDDEN_FEATURE_NAMES)
    # near_threshold_fraction should correlate with future_div in this fixture.
    near_hce = next((c for c in cors if c["feature"] == "near_threshold_fraction"
                    and c["outcome"] == "HCE"), None)
    assert near_hce is not None
    # The fake data is monotonic, so spearman should be high.
    assert abs(near_hce["spearman_r"]) > 0.9


# ---------------------------------------------------------------------------
# Grouped CV regression
# ---------------------------------------------------------------------------


def test_grouped_cv_regression_runs():
    """Need 2+ groups for GroupKFold."""
    rng = np.random.default_rng(0)
    rows = []
    for rule in ("A", "B", "C"):
        for ci in range(8):
            af = rng.uniform(0.3, 0.7)
            margin = abs(af - 0.5)
            future = max(0.0, 0.2 - margin)  # near-threshold = high HCE
            rows.append(_fake_row(rule_id=rule, candidate_id=ci,
                                 near_thresh_frac=1.0 - margin,
                                 mean_margin=margin,
                                 future_div=future,
                                 vs_sham=future, vs_far=future * 0.5))
    scores = grouped_cv_regression(rows, horizon=10, n_splits=3)
    # Some scores should be returned (Ridge + RF for some outcomes).
    assert len(scores) > 0
    # Each score has the required fields.
    for s in scores:
        assert hasattr(s, "model")
        assert hasattr(s, "outcome")
        assert hasattr(s, "feature_importances")
        assert isinstance(s.feature_importances, dict)


def test_rows_to_matrix_handles_empty():
    X, y, g, names = rows_to_matrix([], horizon=10)
    assert X is None and y is None


# ---------------------------------------------------------------------------
# Ablation interventions
# ---------------------------------------------------------------------------


def test_temporal_history_swap_preserves_projection_when_counts_match():
    state, _ = _viable_snapshot()
    mask = _disc_mask()
    # Build a "past" state by hidden-shuffling current; counts preserved per column.
    from observer_worlds.metrics.causality_score import (
        apply_hidden_shuffle_intervention,
    )
    rng = np.random.default_rng(1)
    past = apply_hidden_shuffle_intervention(state, mask, rng)
    p0 = project(state, "mean_threshold", 0.5)
    out = apply_temporal_history_swap_intervention(state, past, mask, rng)
    p1 = project(out, "mean_threshold", 0.5)
    assert np.array_equal(p0, p1), \
        "temporal_history_swap (when counts match) must preserve projection"


def test_temporal_history_swap_no_history_returns_identity():
    state, _ = _viable_snapshot()
    mask = _disc_mask()
    out = apply_temporal_history_swap_intervention(
        state, None, mask, np.random.default_rng(0)
    )
    assert np.array_equal(state, out)


# ---------------------------------------------------------------------------
# End-to-end: measure_candidate sanity
# ---------------------------------------------------------------------------


def test_measure_candidate_returns_one_row_per_horizon():
    state, rule = _viable_snapshot()
    mask = _disc_mask()
    rows = measure_candidate(
        snapshot_4d=state, rule=rule, interior_mask=mask,
        rule_id="t", rule_source="t", seed=0, candidate_id=0,
        snapshot_t=15, candidate_area=10.0, candidate_lifetime=15,
        observer_score=0.5,
        horizons=[3, 5, 7], n_replicates=2, backend="numpy", rng_seed=42,
    )
    assert len(rows) == 3
    # Median horizon (=5) is the only one with ablation data.
    median_row = next(r for r in rows if r.horizon == 5)
    assert median_row.ablation_future_div
    other = next(r for r in rows if r.horizon == 3)
    assert other.ablation_future_div == {}


def test_measure_candidate_features_are_populated():
    state, rule = _viable_snapshot()
    mask = _disc_mask()
    rows = measure_candidate(
        snapshot_4d=state, rule=rule, interior_mask=mask,
        rule_id="t", rule_source="t", seed=0, candidate_id=0,
        snapshot_t=15, candidate_area=10.0, candidate_lifetime=15,
        observer_score=0.5,
        horizons=[5], n_replicates=1, backend="numpy", rng_seed=43,
    )
    r = rows[0]
    assert r.features["n_columns"] == int(mask.sum())
    assert "near_threshold_fraction" in r.features
    assert 0.0 <= r.features["near_threshold_fraction"] <= 1.0

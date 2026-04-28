"""Tests for the revised mechanism classifier in M8G."""
from __future__ import annotations

from observer_worlds.analysis.m8g_revised_mechanisms import (
    classify_revised,
)


# Convenience: minimum keyword set with sensible defaults so each test
# can override only the fields it cares about.

def _classify(**overrides) -> str:
    base = dict(
        old_label="boundary_mediated",
        near_threshold_fraction=0.0,
        boundary_response_fraction=0.0,
        interior_response_fraction=0.0,
        environment_response_fraction=0.0,
        boundary_mediation_index=0.5,
        interior_hidden_effect=0.01,
        boundary_hidden_effect=0.01,
        far_hidden_effect=0.0,
        first_visible_effect_time=0.0,
        bmi_band=(0.35, 0.65),
    )
    base.update(overrides)
    return classify_revised(**base)


# ---------------------------------------------------------------------------
# Co-mediated: both response fractions high, BMI in band
# ---------------------------------------------------------------------------


def test_boundary_and_interior_high_with_balanced_bmi_is_co_mediated():
    label = _classify(
        boundary_response_fraction=0.95,
        interior_response_fraction=0.95,
        boundary_mediation_index=0.50,
    )
    assert label == "boundary_and_interior_co_mediated"


def test_co_mediated_at_band_endpoints():
    # BMI exactly at lower edge of [0.35, 0.65] still counts as co-mediated.
    assert _classify(
        boundary_response_fraction=0.9,
        interior_response_fraction=0.9,
        boundary_mediation_index=0.35,
    ) == "boundary_and_interior_co_mediated"
    assert _classify(
        boundary_response_fraction=0.9,
        interior_response_fraction=0.9,
        boundary_mediation_index=0.65,
    ) == "boundary_and_interior_co_mediated"


# ---------------------------------------------------------------------------
# Boundary dominant: boundary high, BMI > band high
# ---------------------------------------------------------------------------


def test_boundary_high_with_high_bmi_is_boundary_dominant():
    # Both responses high but BMI strongly boundary → boundary-dominant.
    assert _classify(
        boundary_response_fraction=0.95,
        interior_response_fraction=0.95,
        boundary_mediation_index=0.80,
    ) == "boundary_only_or_boundary_dominant"
    # Boundary-only (interior low) with high BMI also lands here.
    assert _classify(
        boundary_response_fraction=0.95,
        interior_response_fraction=0.10,
        boundary_mediation_index=0.85,
    ) == "boundary_only_or_boundary_dominant"


# ---------------------------------------------------------------------------
# Interior dominant: interior high, BMI < band low
# ---------------------------------------------------------------------------


def test_interior_high_with_low_bmi_is_interior_dominant():
    assert _classify(
        boundary_response_fraction=0.10,
        interior_response_fraction=0.95,
        boundary_mediation_index=0.20,
    ) == "interior_only_or_interior_dominant"


# ---------------------------------------------------------------------------
# Global chaotic: far-effect rule fires
# ---------------------------------------------------------------------------


def test_far_effect_dominates_is_global_chaotic():
    # far > 0.7 * (interior + boundary) → global_chaotic regardless of
    # response geometry.
    label = _classify(
        boundary_response_fraction=0.95,
        interior_response_fraction=0.95,
        boundary_mediation_index=0.50,
        interior_hidden_effect=0.01,
        boundary_hidden_effect=0.01,
        far_hidden_effect=0.05,  # well above 0.7 * 0.02 = 0.014
    )
    assert label == "global_chaotic"


def test_old_global_chaotic_label_preserved():
    # Even when the new far rule does not fire, an old `global_chaotic`
    # label survives.
    label = _classify(
        old_label="global_chaotic",
        boundary_response_fraction=0.95,
        interior_response_fraction=0.95,
        boundary_mediation_index=0.50,
        far_hidden_effect=0.0,
    )
    assert label == "global_chaotic"


# ---------------------------------------------------------------------------
# Threshold mediated: preserve old label
# ---------------------------------------------------------------------------


def test_old_threshold_mediated_preserved_under_revised():
    # The audit data needed for the new rule is not in CSVs; we preserve
    # the old label conservatively.
    label = _classify(
        old_label="threshold_mediated",
        boundary_response_fraction=0.95,
        interior_response_fraction=0.95,
        boundary_mediation_index=0.50,
        near_threshold_fraction=0.7,
    )
    assert label == "threshold_mediated"


def test_high_near_threshold_alone_does_not_force_threshold_mediated():
    # If the old label was not threshold_mediated, a high near_threshold
    # alone should not promote a candidate to threshold_mediated under
    # the conservative implementation.
    label = _classify(
        old_label="boundary_mediated",
        boundary_response_fraction=0.95,
        interior_response_fraction=0.95,
        boundary_mediation_index=0.50,
        near_threshold_fraction=0.9,
    )
    assert label == "boundary_and_interior_co_mediated"


# ---------------------------------------------------------------------------
# Delayed hidden channel: preserved from old
# ---------------------------------------------------------------------------


def test_old_delayed_hidden_channel_preserved():
    label = _classify(
        old_label="delayed_hidden_channel",
        boundary_response_fraction=0.0,
        interior_response_fraction=0.0,
        boundary_mediation_index=0.5,
        first_visible_effect_time=12,
    )
    assert label == "delayed_hidden_channel"


# ---------------------------------------------------------------------------
# Environment coupled
# ---------------------------------------------------------------------------


def test_environment_coupled_fires_on_high_environment_response():
    label = _classify(
        boundary_response_fraction=0.0,
        interior_response_fraction=0.0,
        environment_response_fraction=0.7,
    )
    assert label == "environment_coupled"


# ---------------------------------------------------------------------------
# Unclear fallthrough
# ---------------------------------------------------------------------------


def test_unclear_fallthrough_for_zero_signal():
    # No response fractions, no environment, no timing trigger,
    # no old delayed/threshold/global label → unclear.
    label = _classify(
        old_label="boundary_mediated",
        boundary_response_fraction=0.0,
        interior_response_fraction=0.0,
        boundary_mediation_index=0.5,
    )
    assert label == "unclear"


# ---------------------------------------------------------------------------
# Priority tests: thresholding/global come before geometry
# ---------------------------------------------------------------------------


def test_threshold_takes_priority_over_co_mediated():
    label = _classify(
        old_label="threshold_mediated",
        boundary_response_fraction=0.95,
        interior_response_fraction=0.95,
        boundary_mediation_index=0.50,
    )
    assert label == "threshold_mediated"


def test_global_chaotic_takes_priority_over_co_mediated():
    label = _classify(
        boundary_response_fraction=0.95,
        interior_response_fraction=0.95,
        boundary_mediation_index=0.50,
        interior_hidden_effect=0.01,
        boundary_hidden_effect=0.01,
        far_hidden_effect=0.5,
    )
    assert label == "global_chaotic"


# ---------------------------------------------------------------------------
# Boundary-only with mid-range BMI: should not be co-mediated (interior low),
# should not be boundary-dominant (BMI not > band high) → unclear.
# ---------------------------------------------------------------------------


def test_boundary_only_mid_bmi_falls_through_to_unclear():
    label = _classify(
        boundary_response_fraction=0.9,
        interior_response_fraction=0.1,
        boundary_mediation_index=0.5,
    )
    assert label == "unclear"


# ---------------------------------------------------------------------------
# BMI band parameter: the same candidate can change label across bands
# ---------------------------------------------------------------------------


def test_bmi_band_widens_co_mediated_class():
    # BMI = 0.62 with a narrow band [0.40, 0.60] is NOT co-mediated;
    # with a wider band [0.35, 0.65] it IS.
    cand = dict(
        boundary_response_fraction=0.9,
        interior_response_fraction=0.9,
        boundary_mediation_index=0.62,
    )
    narrow = _classify(**cand, bmi_band=(0.40, 0.60))
    wide = _classify(**cand, bmi_band=(0.35, 0.65))
    assert narrow != "boundary_and_interior_co_mediated"
    assert wide == "boundary_and_interior_co_mediated"

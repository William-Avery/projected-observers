"""Tests for M8C large-grid mechanism validation.

Combines what the spec listed as several test files:
  test_m8c_far_control_selection
  test_m8c_classifier_respects_far_validity
  test_m8c_stats
  test_m8c_cli_arg_parser
"""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.detection.morphology import (
    MORPHOLOGY_CLASSES, MorphologyResult,
)
from observer_worlds.experiments._m8b_spatial import (
    M8B_MECHANISM_CLASSES, RegionEffect,
)
from observer_worlds.experiments._m8c_validation import (
    FarControlInfo,
    M8CCandidateResult,
    _candidate_extent,
    measure_candidate_m8c,
    select_far_mask,
)
from observer_worlds.search.rules import FractionalRule


# ---------------------------------------------------------------------------
# Candidate extent
# ---------------------------------------------------------------------------


def test_candidate_extent_radius_for_disc():
    m = np.zeros((20, 20), dtype=bool); m[8:13, 8:13] = True  # 5x5
    radius, diameter, (cy, cx) = _candidate_extent(m)
    # Centroid at (10, 10); corners at (8,8) → distance ~2.83.
    assert 2.5 < radius < 3.5
    assert 4.0 < diameter < 6.0
    assert cy == pytest.approx(10.0)
    assert cx == pytest.approx(10.0)


def test_candidate_extent_empty_mask_returns_zero():
    m = np.zeros((10, 10), dtype=bool)
    radius, diameter, _ = _candidate_extent(m)
    assert radius == 0.0
    assert diameter == 0.0


# ---------------------------------------------------------------------------
# select_far_mask
# ---------------------------------------------------------------------------


def test_select_far_mask_finds_valid_translation_on_large_grid():
    Nx, Ny = 64, 64
    m = np.zeros((Nx, Ny), dtype=bool); m[10:13, 10:13] = True  # area 9
    far, info = select_far_mask(
        m, snapshot_4d=None, min_distance_floor=20,
        min_distance_radius_mult=5.0, rng_seed=0,
    )
    assert info.far_control_valid
    assert far.sum() == m.sum()  # translation preserves area
    assert info.far_control_distance >= 20.0


def test_select_far_mask_invalid_when_grid_too_small():
    """If the grid is too small to fit min_distance away, far_control_valid=False."""
    Nx, Ny = 16, 16
    m = np.zeros((Nx, Ny), dtype=bool); m[6:10, 6:10] = True
    far, info = select_far_mask(
        m, min_distance_floor=20,  # cannot be 20 cells away on a 16-grid
        min_distance_radius_mult=10.0, rng_seed=0,
    )
    assert not info.far_control_valid
    assert far.sum() == 0
    assert info.rejection_reason


def test_select_far_mask_no_overlap_with_environment_shell():
    """The chosen far_mask must not intersect the candidate's
    dilation-3 environment shell."""
    import scipy.ndimage as ndi
    m = np.zeros((48, 48), dtype=bool); m[10:14, 10:14] = True
    env = ndi.binary_dilation(m, iterations=3)
    far, info = select_far_mask(
        m, min_distance_floor=10, min_distance_radius_mult=3.0, rng_seed=0,
    )
    assert info.far_control_valid
    assert (far & env).sum() == 0


def test_select_far_mask_distance_over_radius_reasonable():
    m = np.zeros((96, 96), dtype=bool); m[20:30, 20:30] = True  # 10x10 thick
    far, info = select_far_mask(
        m, min_distance_floor=32, min_distance_radius_mult=5.0, rng_seed=0,
    )
    assert info.far_control_valid
    # On a 96x96 grid the antipode is ~48 cells away; radius ~7; ratio >5.
    assert info.far_control_distance_over_radius >= 5.0


# ---------------------------------------------------------------------------
# Classifier respects far_control_valid
# ---------------------------------------------------------------------------


def _toy_state(seed: int = 0, shape=(8, 8, 4, 4)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < 0.25).astype(np.uint8)


def test_measure_candidate_m8c_marks_far_invalid_when_grid_small():
    """On an 8x8 grid we can't satisfy floor=32, so far_control_valid
    should be False, and the result must NOT be labeled global_chaotic
    even if far_effect would be similar to candidate effect."""
    state = _toy_state()
    rule = FractionalRule(
        birth_min=0.30, birth_max=0.45, survive_min=0.25, survive_max=0.50,
        initial_density=0.20,
    ).to_bsrule()
    mask = np.zeros((8, 8), dtype=bool); mask[2:6, 2:6] = True
    res = measure_candidate_m8c(
        snapshot_4d=state, candidate_mask_2d=mask, rule=rule,
        rule_id="r1", rule_source="src", seed=0, candidate_id=1,
        snapshot_t=10, candidate_area=int(mask.sum()),
        candidate_lifetime=20, observer_score=None,
        near_threshold_fraction=0.0, horizons=[1, 2, 3, 5],
        n_replicates=1, backend="numpy", rng_seed=0,
        min_far_distance_floor=32,  # impossible on 8x8
        min_far_distance_radius_mult=5.0,
    )
    assert not res.far_control.far_control_valid
    assert res.mechanism_label != "global_chaotic"
    assert res.supporting_metrics["far_control_valid"] is False


def test_measure_candidate_m8c_returns_known_label_class():
    state = _toy_state()
    rule = FractionalRule(
        birth_min=0.30, birth_max=0.45, survive_min=0.25, survive_max=0.50,
        initial_density=0.20,
    ).to_bsrule()
    mask = np.zeros((8, 8), dtype=bool); mask[2:6, 2:6] = True
    res = measure_candidate_m8c(
        snapshot_4d=state, candidate_mask_2d=mask, rule=rule,
        rule_id="r1", rule_source="src", seed=0, candidate_id=1,
        snapshot_t=10, candidate_area=int(mask.sum()),
        candidate_lifetime=20, observer_score=None,
        near_threshold_fraction=0.0, horizons=[1, 2, 3, 5],
        n_replicates=1, backend="numpy", rng_seed=0,
        min_far_distance_floor=4, min_far_distance_radius_mult=2.0,
    )
    assert res.mechanism_label in M8B_MECHANISM_CLASSES


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _stub_far(*, valid=True, distance=40.0, radius=4.0):
    return FarControlInfo(
        candidate_radius=radius, candidate_diameter=2 * radius,
        far_control_translation=(20, 20) if valid else None,
        far_control_distance=distance if valid else 0.0,
        far_control_distance_over_radius=(distance / radius) if valid else 0.0,
        far_control_valid=valid,
        far_control_min_distance_required=20.0,
        far_control_projected_activity_diff=0.05,
        far_control_hidden_activity_diff=0.05,
        rejection_reason="" if valid else "no valid translation",
    )


def _stub_morphology(*, cls="thick_candidate", area=30):
    return MorphologyResult(
        morphology_class=cls, area=area,
        erosion1_interior_size=area // 4, erosion2_interior_size=area // 8,
        boundary_size=area // 2, environment_size=area,
        can_separate_boundary_from_interior=(
            cls in ("thick_candidate", "very_thick_candidate")
        ),
        can_classify_environment_coupled=True,
    )


def _stub_eff(*, pc=0.001, raw=0.01, n_perturbed=10):
    return RegionEffect(
        region_name="x", n_perturbed_cells_2d=n_perturbed,
        n_flipped_cells_4d=n_perturbed * 8,
        region_hidden_effect=raw, region_local_divergence=raw,
        region_global_divergence=raw, region_response_fraction=0.25,
        region_effect_per_cell=pc,
        region_effect_per_flipped_cell=raw / max(n_perturbed * 8, 1),
    )


def _stub_result(*, source="M7_HCE_optimized", cls="thick_candidate",
                 mech="interior_reservoir", area=30, valid_far=True,
                 candidate_id=1, seed=0, hce=0.05):
    region_effects = {
        "interior": _stub_eff(pc=0.005, raw=0.05),
        "boundary": _stub_eff(pc=0.001, raw=0.01),
        "environment": _stub_eff(pc=0.0001, raw=0.005),
        "whole": _stub_eff(pc=0.003, raw=hce),
    }
    far_eff = _stub_eff(pc=0.0001, raw=0.005)
    return M8CCandidateResult(
        rule_id="r0", rule_source=source, seed=seed,
        candidate_id=candidate_id, snapshot_t=10,
        candidate_area=area, candidate_lifetime=50,
        observer_score=None, near_threshold_fraction=0.05,
        morphology=_stub_morphology(cls=cls, area=area),
        far_control=_stub_far(valid=valid_far),
        region_effects=region_effects, far_effect=far_eff,
        first_visible_effect_time=2, hidden_to_visible_conversion_time=1,
        fraction_hidden_at_end=0.05, fraction_visible_at_end=0.02,
        mechanism_label=mech, mechanism_confidence=0.7,
        supporting_metrics={},
    )


def test_aggregate_per_source_thick_fraction():
    from observer_worlds.analysis.m8c_stats import aggregate_per_source
    rs = [
        _stub_result(cls="thick_candidate", candidate_id=1),
        _stub_result(cls="thick_candidate", candidate_id=2, seed=1),
        _stub_result(cls="thin_candidate", candidate_id=3, seed=2),
    ]
    a = aggregate_per_source(rs)["M7_HCE_optimized"]
    assert a["n_thick"] == 2
    assert a["thick_fraction"] == pytest.approx(2 / 3)


def test_far_control_quality_excludes_invalid():
    from observer_worlds.analysis.m8c_stats import far_control_quality_summary
    rs = [
        _stub_result(valid_far=True, candidate_id=1),
        _stub_result(valid_far=False, candidate_id=2, seed=1),
    ]
    q = far_control_quality_summary(rs)["M7_HCE_optimized"]
    assert q["n_total"] == 2
    assert q["n_valid_far"] == 1
    assert q["valid_far_fraction"] == pytest.approx(0.5)


def test_select_interpretations_strong_interior():
    from observer_worlds.analysis.m8c_stats import (
        m8c_full_summary, select_interpretations,
    )
    rs = [
        _stub_result(cls="thick_candidate",
                    mech="interior_reservoir" if i % 2 == 0
                    else "whole_body_hidden_support",
                    candidate_id=i, seed=i)
        for i in range(8)
    ]
    msgs = select_interpretations(m8c_full_summary(rs))
    assert any("interior/whole-body" in m for m in msgs)


def test_select_interpretations_global_chaos_persists():
    from observer_worlds.analysis.m8c_stats import (
        m8c_full_summary, select_interpretations,
    )
    rs = [
        _stub_result(cls="thick_candidate", mech="global_chaotic",
                    candidate_id=i, seed=i)
        for i in range(6)
    ]
    msgs = select_interpretations(m8c_full_summary(rs))
    assert any("broad dynamical instability" in m for m in msgs)


def test_select_interpretations_no_thick_baselines():
    """If M4C has 0 thick and M4A has < 5 thick, the no-baselines
    interpretation should fire."""
    from observer_worlds.analysis.m8c_stats import (
        m8c_full_summary, select_interpretations,
    )
    rs = (
        [_stub_result(source="M7_HCE_optimized", cls="thick_candidate",
                     mech="interior_reservoir", candidate_id=i, seed=i)
         for i in range(5)] +
        [_stub_result(source="M4C_observer_optimized", cls="thin_candidate",
                     mech="candidate_local_thin", candidate_id=i + 100,
                     seed=i + 100) for i in range(3)]
    )
    msgs = select_interpretations(m8c_full_summary(rs))
    assert any("uniquely produces thick" in m for m in msgs)


def test_render_m8c_summary_md_has_far_quality_section():
    from observer_worlds.analysis.m8c_stats import (
        m8c_full_summary, render_m8c_summary_md,
    )
    rs = [
        _stub_result(cls="thick_candidate", candidate_id=1, valid_far=True),
        _stub_result(cls="thin_candidate", candidate_id=2, seed=1, valid_far=False),
    ]
    md = render_m8c_summary_md(m8c_full_summary(rs))
    assert "M8C" in md
    assert "Far-control quality" in md


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_arg_parser_quick_overrides():
    from observer_worlds.experiments.run_m8c_large_grid_mechanism_validation \
        import build_arg_parser, _quick
    args = build_arg_parser().parse_args([
        "--m7-rules", "/tmp/x.json", "--quick",
    ])
    args = _quick(args)
    assert args.timesteps <= 100
    assert args.grid == [32, 32, 4, 4]
    assert args.min_far_distance_floor == 8

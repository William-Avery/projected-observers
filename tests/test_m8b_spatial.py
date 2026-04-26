"""Tests for M8B spatial mechanism disambiguation.

Combines the spec's five test files:
  test_m8b_morphology_gates
  test_m8b_region_response_maps
  test_m8b_mechanism_classifier
  test_large_candidate_search
  test_m8b_per_cell_normalization
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as ndi

from observer_worlds.detection.morphology import (
    MORPHOLOGY_CLASSES,
    classify_morphology,
    far_mask,
    shell_masks_strict,
)
from observer_worlds.experiments._m8b_spatial import (
    M8B_MECHANISM_CLASSES,
    M8BCandidateResult,
    RegionEffect,
    classify_mechanism_v2,
    measure_region_effect,
)
from observer_worlds.search.rules import FractionalRule


# ---------------------------------------------------------------------------
# Morphology gates
# ---------------------------------------------------------------------------


def test_morphology_empty_mask_is_degenerate():
    m = np.zeros((10, 10), dtype=bool)
    res = classify_morphology(m)
    assert res.morphology_class == "degenerate"
    assert not res.can_separate_boundary_from_interior
    assert not res.can_classify_environment_coupled


def test_morphology_single_cell_is_degenerate():
    m = np.zeros((10, 10), dtype=bool); m[5, 5] = True
    res = classify_morphology(m)
    assert res.morphology_class == "degenerate"


def test_morphology_thin_below_thick_threshold():
    """A 3×3 disc has area=9, below min_thick_area=25."""
    m = np.zeros((20, 20), dtype=bool); m[8:11, 8:11] = True
    res = classify_morphology(m)
    assert res.morphology_class == "thin_candidate"
    assert not res.can_separate_boundary_from_interior


def test_morphology_thick_passes_at_area_25():
    """5×5 disc: area=25, erosion gives 3×3 interior."""
    m = np.zeros((20, 20), dtype=bool); m[7:12, 7:12] = True
    res = classify_morphology(m)
    assert res.morphology_class == "thick_candidate"
    assert res.can_separate_boundary_from_interior
    assert res.can_classify_environment_coupled
    assert res.erosion1_interior_size > 0


def test_morphology_very_thick_passes_at_area_50():
    """8×8 square: area=64, erosion(2) leaves 4×4."""
    m = np.zeros((24, 24), dtype=bool); m[8:16, 8:16] = True
    res = classify_morphology(m)
    assert res.morphology_class == "very_thick_candidate"
    assert res.erosion2_interior_size > 0


def test_morphology_class_in_known_classes():
    """All four classes are known."""
    for cls in ("very_thick_candidate", "thick_candidate",
                "thin_candidate", "degenerate"):
        assert cls in MORPHOLOGY_CLASSES


# ---------------------------------------------------------------------------
# Shell masks (strict) — the function the morphology gate uses
# ---------------------------------------------------------------------------


def test_shell_masks_strict_returns_required_keys():
    m = np.zeros((10, 10), dtype=bool); m[3:7, 3:7] = True
    s = shell_masks_strict(m)
    for k in ("interior", "boundary", "environment", "whole"):
        assert k in s


def test_shell_masks_strict_disjoint():
    m = np.zeros((20, 20), dtype=bool); m[8:13, 8:13] = True
    s = shell_masks_strict(m)
    # Boundary cells must be inside the candidate.
    assert (s["boundary"] & ~m).sum() == 0
    # Environment cells must be outside the candidate.
    assert (s["environment"] & m).sum() == 0
    # Interior cells must be inside.
    assert (s["interior"] & ~m).sum() == 0


def test_shell_masks_strict_no_fallback():
    """Unlike M8's _shell_masks, the strict version does NOT alias
    boundary to mask when interior is empty."""
    # Single cell — interior after erosion is empty.
    m = np.zeros((10, 10), dtype=bool); m[5, 5] = True
    s = shell_masks_strict(m)
    # Whole = mask, interior = empty.
    assert s["whole"].sum() == 1
    assert s["interior"].sum() == 0
    # Boundary = mask & ~interior = mask.
    # That's fine — we're testing that interior is preserved as empty,
    # not that boundary is empty.
    assert (s["boundary"] == m).all()


def test_far_mask_translates_to_antipode():
    Nx, Ny = 16, 16
    m = np.zeros((Nx, Ny), dtype=bool); m[2:5, 2:5] = True
    f = far_mask(m)
    assert f.sum() == m.sum()
    # Default translation is (Nx//2, Ny//2) = (8, 8).
    assert f[10, 10]  # original (2, 2) shifted to (10, 10)
    assert not f[2, 2]


# ---------------------------------------------------------------------------
# Region effect / per-cell normalization
# ---------------------------------------------------------------------------


def _toy_state(seed: int = 0, shape=(8, 8, 4, 4)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < 0.25).astype(np.uint8)


def _toy_thick_mask(shape=(8, 8)) -> np.ndarray:
    m = np.zeros(shape, dtype=bool); m[2:6, 2:6] = True  # 4x4 = 16 cells
    return m


def test_region_effect_returns_zero_for_empty_region():
    state = _toy_state()
    rule = FractionalRule(
        birth_min=0.30, birth_max=0.45,
        survive_min=0.25, survive_max=0.50, initial_density=0.20,
    ).to_bsrule()
    empty_region = np.zeros((8, 8), dtype=bool)
    cand_mask = _toy_thick_mask()
    eff = measure_region_effect(
        snapshot_4d=state, rule=rule, region_mask_2d=empty_region,
        candidate_mask_2d=cand_mask, region_name="x",
        horizon=2, n_replicates=1, backend="numpy", rng_seed=0,
    )
    assert eff.region_hidden_effect == 0.0
    assert eff.region_effect_per_cell == 0.0


def test_region_effect_normalizes_by_perturbed_cell_count():
    """If we apply the same per-cell strength to a small region vs a
    large region, the per-cell normalized effect should be more
    comparable than the raw effect."""
    state = _toy_state()
    rule = FractionalRule(
        birth_min=0.30, birth_max=0.45,
        survive_min=0.25, survive_max=0.50, initial_density=0.20,
    ).to_bsrule()
    small_region = np.zeros((8, 8), dtype=bool); small_region[3, 3] = True
    cand = _toy_thick_mask()
    eff_small = measure_region_effect(
        snapshot_4d=state, rule=rule, region_mask_2d=small_region,
        candidate_mask_2d=cand, region_name="small", horizon=3,
        n_replicates=2, backend="numpy", rng_seed=0,
    )
    assert eff_small.n_perturbed_cells_2d == 1
    # Per-cell normalization divides hidden_effect by 1 * Nz * Nw = 1*4*4 = 16.
    assert eff_small.region_effect_per_cell == pytest.approx(
        eff_small.region_hidden_effect / 16.0, abs=1e-9
    )


def test_region_effect_per_flipped_cell_is_nonneg():
    state = _toy_state()
    rule = FractionalRule(
        birth_min=0.30, birth_max=0.45,
        survive_min=0.25, survive_max=0.50, initial_density=0.20,
    ).to_bsrule()
    cand = _toy_thick_mask()
    eff = measure_region_effect(
        snapshot_4d=state, rule=rule, region_mask_2d=cand,
        candidate_mask_2d=cand, region_name="whole", horizon=2,
        n_replicates=1, backend="numpy", rng_seed=0,
    )
    assert eff.region_effect_per_flipped_cell >= 0.0


# ---------------------------------------------------------------------------
# v2 mechanism classifier
# ---------------------------------------------------------------------------


def _stub_morphology(*, cls="thick_candidate"):
    from observer_worlds.detection.morphology import MorphologyResult
    return MorphologyResult(
        morphology_class=cls, area=25,
        erosion1_interior_size=4, erosion2_interior_size=1,
        boundary_size=20, environment_size=30,
        can_separate_boundary_from_interior=(cls != "thin_candidate" and cls != "degenerate"),
        can_classify_environment_coupled=True,
    )


def _stub_eff(*, name="x", per_cell=0.0, raw=0.0, n_perturbed=10):
    return RegionEffect(
        region_name=name, n_perturbed_cells_2d=n_perturbed,
        n_flipped_cells_4d=n_perturbed * 8,
        region_hidden_effect=raw, region_local_divergence=raw,
        region_global_divergence=raw, region_response_fraction=0.25,
        region_effect_per_cell=per_cell,
        region_effect_per_flipped_cell=raw / max(n_perturbed * 8, 1),
    )


def test_classifier_thin_returns_candidate_local_thin():
    morph = _stub_morphology(cls="thin_candidate")
    region_effects = {
        "interior": _stub_eff(per_cell=0.01, raw=0.05),
        "boundary": _stub_eff(per_cell=0.01, raw=0.05),
        "environment": _stub_eff(per_cell=0.005, raw=0.10),
        "whole": _stub_eff(per_cell=0.01, raw=0.05),
    }
    far = _stub_eff(per_cell=0.001, raw=0.01)
    label, conf, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, fraction_hidden_at_end=0.05,
        fraction_visible_at_end=0.02, near_threshold_fraction=0.0,
    )
    # Thin morphology cannot return boundary_mediated; should be
    # candidate_local_thin or environment_coupled_thin.
    assert label in ("candidate_local_thin", "environment_coupled_thin")
    assert label not in ("boundary_mediated", "interior_reservoir",
                         "whole_body_hidden_support")


def test_classifier_thick_boundary_dominant():
    morph = _stub_morphology()
    region_effects = {
        "interior": _stub_eff(per_cell=0.001, raw=0.01),
        "boundary": _stub_eff(per_cell=0.005, raw=0.05),  # 5x interior pc
        "environment": _stub_eff(per_cell=0.001, raw=0.03),
        "whole": _stub_eff(per_cell=0.003, raw=0.06),
    }
    far = _stub_eff(per_cell=0.0001, raw=0.005)
    label, _, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, fraction_hidden_at_end=0.05,
        fraction_visible_at_end=0.02, near_threshold_fraction=0.0,
    )
    assert label == "boundary_mediated"


def test_classifier_thick_interior_dominant():
    morph = _stub_morphology()
    region_effects = {
        "interior": _stub_eff(per_cell=0.005, raw=0.05),  # 5x boundary pc
        "boundary": _stub_eff(per_cell=0.001, raw=0.01),
        "environment": _stub_eff(per_cell=0.001, raw=0.03),
        "whole": _stub_eff(per_cell=0.003, raw=0.06),
    }
    far = _stub_eff(per_cell=0.0001, raw=0.005)
    label, _, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, fraction_hidden_at_end=0.05,
        fraction_visible_at_end=0.02, near_threshold_fraction=0.0,
    )
    assert label == "interior_reservoir"


def test_classifier_thick_environment_coupled():
    morph = _stub_morphology()
    region_effects = {
        "interior": _stub_eff(per_cell=0.001, raw=0.01),
        "boundary": _stub_eff(per_cell=0.001, raw=0.01),
        "environment": _stub_eff(per_cell=0.005, raw=0.10),  # 5x candidate pc
        "whole": _stub_eff(per_cell=0.001, raw=0.02),
    }
    far = _stub_eff(per_cell=0.0001, raw=0.005)
    label, _, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, fraction_hidden_at_end=0.05,
        fraction_visible_at_end=0.02, near_threshold_fraction=0.0,
    )
    assert label == "environment_coupled"


def test_classifier_thick_whole_body_when_similar():
    morph = _stub_morphology()
    region_effects = {
        "interior": _stub_eff(per_cell=0.003, raw=0.03),
        "boundary": _stub_eff(per_cell=0.003, raw=0.03),  # similar to interior
        "environment": _stub_eff(per_cell=0.0015, raw=0.03),  # not 1.5x
        "whole": _stub_eff(per_cell=0.003, raw=0.06),
    }
    far = _stub_eff(per_cell=0.0001, raw=0.01)
    label, _, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, fraction_hidden_at_end=0.05,
        fraction_visible_at_end=0.02, near_threshold_fraction=0.0,
    )
    assert label == "whole_body_hidden_support"


def test_classifier_global_chaotic_takes_priority_over_boundary():
    morph = _stub_morphology()
    region_effects = {
        "interior": _stub_eff(per_cell=0.001, raw=0.01),
        "boundary": _stub_eff(per_cell=0.005, raw=0.05),
        "environment": _stub_eff(per_cell=0.001, raw=0.01),
        "whole": _stub_eff(per_cell=0.003, raw=0.06),
    }
    far = _stub_eff(per_cell=0.001, raw=0.10)  # far >> candidate (0.05)
    label, _, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, fraction_hidden_at_end=0.05,
        fraction_visible_at_end=0.02, near_threshold_fraction=0.0,
    )
    assert label == "global_chaotic"


def test_classifier_threshold_mediated_priority():
    morph = _stub_morphology()
    region_effects = {
        "interior": _stub_eff(per_cell=0.0005, raw=0.005),  # 0.5x boundary pc
        "boundary": _stub_eff(per_cell=0.005, raw=0.05),
        "environment": _stub_eff(per_cell=0.001, raw=0.02),
        "whole": _stub_eff(per_cell=0.003, raw=0.05),
    }
    far = _stub_eff(per_cell=0.0001, raw=0.005)
    label, _, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, fraction_hidden_at_end=0.05,
        fraction_visible_at_end=0.02, near_threshold_fraction=0.80,
    )
    assert label == "threshold_mediated"


def test_classifier_label_in_known_classes():
    morph = _stub_morphology()
    region_effects = {
        "interior": _stub_eff(),
        "boundary": _stub_eff(),
        "environment": _stub_eff(),
        "whole": _stub_eff(),
    }
    far = _stub_eff()
    label, _, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, fraction_hidden_at_end=0.05,
        fraction_visible_at_end=0.02, near_threshold_fraction=0.0,
    )
    assert label in M8B_MECHANISM_CLASSES


def test_classifier_unclear_for_zero_inputs():
    morph = _stub_morphology()
    region_effects = {
        "interior": _stub_eff(),
        "boundary": _stub_eff(),
        "environment": _stub_eff(),
        "whole": _stub_eff(),
    }
    far = _stub_eff()
    label, conf, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, fraction_hidden_at_end=0.0,
        fraction_visible_at_end=0.0, near_threshold_fraction=0.0,
    )
    assert label == "unclear"
    assert conf == 0.0


# ---------------------------------------------------------------------------
# search_large_candidates CLI sanity (smoke)
# ---------------------------------------------------------------------------


def test_search_large_candidates_arg_parser_quick():
    from observer_worlds.experiments.search_large_candidates import (
        build_arg_parser, _quick,
    )
    parser = build_arg_parser()
    args = parser.parse_args([
        "--m7-rules", "/tmp/x.json", "--quick",
    ])
    args = _quick(args)
    assert args.timesteps <= 80
    assert args.grid == [16, 16, 4, 4]
    assert args.min_area == 4


# ---------------------------------------------------------------------------
# m8b_stats: thick-only proportions sum to 1 when N_thick > 0
# ---------------------------------------------------------------------------


def _stub_result(*, source="M7_HCE_optimized", morph_class="thick_candidate",
                 mech="boundary_mediated", area=30, lifetime=60,
                 hce=0.05, candidate_id=1, rule_id="r0", seed=0):
    from observer_worlds.detection.morphology import MorphologyResult
    morph = MorphologyResult(
        morphology_class=morph_class, area=area,
        erosion1_interior_size=area // 4, erosion2_interior_size=area // 8,
        boundary_size=area // 2, environment_size=area,
        can_separate_boundary_from_interior=(morph_class
            in ("thick_candidate", "very_thick_candidate")),
        can_classify_environment_coupled=True,
    )
    eff = lambda pc, raw: RegionEffect(
        region_name="x", n_perturbed_cells_2d=10, n_flipped_cells_4d=80,
        region_hidden_effect=raw, region_local_divergence=raw,
        region_global_divergence=raw, region_response_fraction=0.25,
        region_effect_per_cell=pc, region_effect_per_flipped_cell=raw / 80.0,
    )
    region_effects = {
        "interior": eff(0.001, 0.01), "boundary": eff(0.005, 0.05),
        "environment": eff(0.001, 0.01), "whole": eff(0.003, hce),
    }
    far = eff(0.0001, 0.005)
    return M8BCandidateResult(
        rule_id=rule_id, rule_source=source, seed=seed,
        candidate_id=candidate_id, snapshot_t=10,
        candidate_area=area, candidate_lifetime=lifetime,
        observer_score=0.5, near_threshold_fraction=0.05,
        morphology=morph, region_effects=region_effects, far_effect=far,
        first_visible_effect_time=2, hidden_to_visible_conversion_time=1,
        fraction_hidden_at_end=0.05, fraction_visible_at_end=0.02,
        mechanism_label=mech, mechanism_confidence=0.7,
        supporting_metrics={},
    )


def test_aggregate_per_source_thick_thin_counts():
    from observer_worlds.analysis.m8b_stats import aggregate_per_source
    rs = [
        _stub_result(morph_class="thick_candidate", candidate_id=1),
        _stub_result(morph_class="thick_candidate", candidate_id=2, seed=1),
        _stub_result(morph_class="thin_candidate", candidate_id=3, seed=2),
    ]
    a = aggregate_per_source(rs)["M7_HCE_optimized"]
    assert a["n_thick"] == 2
    assert a["n_thin"] == 1


def test_morphology_distribution_fractions_sum_to_one():
    from observer_worlds.analysis.m8b_stats import morphology_class_distribution
    rs = [
        _stub_result(morph_class="thick_candidate", candidate_id=1),
        _stub_result(morph_class="thin_candidate", candidate_id=2, seed=1),
        _stub_result(morph_class="degenerate", candidate_id=3, seed=2),
    ]
    d = morphology_class_distribution(rs)["M7_HCE_optimized"]
    total = sum(d["per_class"][c]["fraction"] for c in MORPHOLOGY_CLASSES)
    assert total == pytest.approx(1.0)


def test_thick_only_mechanism_distribution_excludes_thin():
    from observer_worlds.analysis.m8b_stats import (
        mechanism_class_distribution_thick_only,
    )
    rs = [
        _stub_result(morph_class="thick_candidate", mech="boundary_mediated",
                     candidate_id=1),
        _stub_result(morph_class="thin_candidate", mech="candidate_local_thin",
                     candidate_id=2, seed=1),
    ]
    d = mechanism_class_distribution_thick_only(rs)["M7_HCE_optimized"]
    assert d["n_thick"] == 1
    # Thin label shouldn't show up in the thick-only distribution counts.
    assert d["per_class"]["candidate_local_thin"]["count"] == 0
    assert d["per_class"]["boundary_mediated"]["count"] == 1


def test_boundary_vs_interior_paired_thick_only():
    from observer_worlds.analysis.m8b_stats import boundary_vs_interior_paired
    rs = [
        _stub_result(morph_class="thick_candidate", candidate_id=1),
        _stub_result(morph_class="thick_candidate", candidate_id=2, seed=1),
        _stub_result(morph_class="thin_candidate", candidate_id=3, seed=2),
    ]
    d = boundary_vs_interior_paired(rs)["M7_HCE_optimized"]
    assert d["n"] == 2  # excludes the thin


def test_render_m8b_summary_md_basic():
    from observer_worlds.analysis.m8b_stats import (
        m8b_full_summary, render_m8b_summary_md,
    )
    rs = [
        _stub_result(morph_class="thick_candidate", mech="boundary_mediated",
                     candidate_id=1),
        _stub_result(morph_class="thick_candidate", mech="boundary_mediated",
                     candidate_id=2, seed=1),
    ]
    md = render_m8b_summary_md(m8b_full_summary(rs))
    assert "M8B" in md
    assert "boundary_mediated" in md
    assert "Mechanism distribution among thick" in md


def test_select_interpretations_dominant_boundary():
    from observer_worlds.analysis.m8b_stats import (
        m8b_full_summary, select_interpretations,
    )
    rs = [
        _stub_result(morph_class="thick_candidate", mech="boundary_mediated",
                     candidate_id=i, seed=i) for i in range(5)
    ]
    msgs = select_interpretations(m8b_full_summary(rs))
    assert any("boundary-mediated" in m for m in msgs)


def test_select_interpretations_insufficient_thick():
    from observer_worlds.analysis.m8b_stats import (
        m8b_full_summary, select_interpretations,
    )
    rs = [
        _stub_result(morph_class="thin_candidate", mech="candidate_local_thin",
                     candidate_id=i, seed=i) for i in range(5)
    ]
    msgs = select_interpretations(m8b_full_summary(rs))
    assert any("Insufficient thick candidates" in m for m in msgs)

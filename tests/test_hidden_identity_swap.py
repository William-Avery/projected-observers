"""CLI / config / swap-mechanic / smoke tests for Follow-up Topic 2."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from observer_worlds.experiments import (
    run_followup_hidden_identity_swap as runner,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_help_runs_without_error():
    result = subprocess.run(
        [sys.executable, "-m", runner.__name__, "--help"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0
    assert "hidden identity swap" in result.stdout.lower()


def test_unsupported_match_mode_rejected_with_clear_error(tmp_path: Path):
    """Stage 5A: morphology_nearest is now supported. Use a still-
    unsupported mode (observer_score_bin) to verify the runner still
    rejects them at config-resolution time."""
    with pytest.raises(SystemExit):
        runner.main([
            "--quick",
            "--out-root", str(tmp_path),
            "--label", "bad",
            "--matching-mode", "observer_score_bin",
        ])


def test_metric_inventory_is_importable():
    from observer_worlds.analysis.identity_swap_stats import (
        IDENTITY_METRICS, aggregate_identity_results,
    )
    assert "hidden_identity_pull" in IDENTITY_METRICS
    assert "projection_preservation_error" in IDENTITY_METRICS
    s = aggregate_identity_results([], [])
    assert s["stage"] == 3
    assert s["n_pairs_attempted"] == 0


# ---------------------------------------------------------------------------
# Hybrid construction unit tests
# ---------------------------------------------------------------------------


def _zero_state(shape=(8, 8, 4, 4)):
    return np.zeros(shape, dtype=np.uint8)


def test_construct_hybrid_matched_projection_preserves_visible():
    """Two cells with identical mean_threshold projection (=1) but
    different hidden fibres can be swapped without changing the
    projection."""
    from observer_worlds.experiments._followup_identity_swap import (
        construct_hybrid,
    )
    Nx = Ny = 4; Nz = Nw = 4
    a = _zero_state(shape=(Nx, Ny, Nz, Nw))
    b = _zero_state(shape=(Nx, Ny, Nz, Nw))
    # Both cells project to 1 (mean > 0.5: at least 9 of 16 ON), but
    # the hidden patterns differ.
    a[1, 1, :, :] = 0; a[1, 1, :3, :3] = 1   # 9 ON in alpha
    b[2, 2, :, :] = 1; b[2, 2, 0, 0] = 0     # 15 ON in beta at (2,2)
    mask_a = np.zeros((Nx, Ny), dtype=bool); mask_a[1, 1] = True
    mask_b = np.zeros((Nx, Ny), dtype=bool); mask_b[2, 2] = True
    centroid_a = (1.0, 1.0); centroid_b = (2.0, 2.0)
    res = construct_hybrid(
        a, b, mask_a, mask_b, centroid_a, centroid_b,
        projection_name="mean_threshold",
    )
    assert res.accepted is True
    assert res.n_cells_swapped == 1
    assert res.projection_preservation_error <= 1e-6
    # Hybrid's hidden fibre at (1,1) now matches B's (2,2) fibre.
    np.testing.assert_array_equal(
        res.hybrid_state[1, 1], b[2, 2],
    )
    # Cells outside A's mask are untouched.
    np.testing.assert_array_equal(res.hybrid_state[0, 0], a[0, 0])


def test_construct_hybrid_mismatched_per_cell_projection_skips_swap():
    """If the per-cell projected values disagree (one is 1, the other
    is 0) we must not swap — that would change the visible projection."""
    from observer_worlds.experiments._followup_identity_swap import (
        construct_hybrid,
    )
    Nx = Ny = 4; Nz = Nw = 4
    a = _zero_state(shape=(Nx, Ny, Nz, Nw))
    b = _zero_state(shape=(Nx, Ny, Nz, Nw))
    a[1, 1, :3, :3] = 1            # mean > 0.5 -> projects to 1
    # b[2, 2] stays all zeros -> projects to 0
    mask_a = np.zeros((Nx, Ny), dtype=bool); mask_a[1, 1] = True
    mask_b = np.zeros((Nx, Ny), dtype=bool); mask_b[2, 2] = True
    res = construct_hybrid(
        a, b, mask_a, mask_b, (1.0, 1.0), (2.0, 2.0),
        projection_name="mean_threshold",
    )
    # Per-cell projection mismatch -> no swap, invalid pair.
    assert res.n_cells_swapped == 0
    assert res.accepted is False
    assert "no swap-eligible cells" in (res.invalid_reason or "")
    # State is untouched (projection trivially preserved).
    np.testing.assert_array_equal(res.hybrid_state[1, 1], a[1, 1])


def test_construct_hybrid_centroid_translation_aligns_overlap():
    """Translation maps A's centroid onto B's centroid; overlap occurs
    where both masks are aligned post-translation."""
    from observer_worlds.experiments._followup_identity_swap import (
        construct_hybrid,
    )
    Nx = Ny = 6; Nz = Nw = 4
    a = _zero_state(shape=(Nx, Ny, Nz, Nw))
    b = _zero_state(shape=(Nx, Ny, Nz, Nw))
    # Each fibre has 9 ON cells -> mean > 0.5 -> projection 1.
    for x in (1, 2):
        for y in (1, 2):
            a[x, y, :3, :3] = 1
    for x in (3, 4):
        for y in (3, 4):
            b[x, y, :3, :3] = 1   # 9 ON
            b[x, y, 0, 0] = 0     # toggle one bit off
            b[x, y, 3, 3] = 1     # add one elsewhere → still 9 ON, different pattern
    mask_a = np.zeros((Nx, Ny), dtype=bool); mask_a[1:3, 1:3] = True
    mask_b = np.zeros((Nx, Ny), dtype=bool); mask_b[3:5, 3:5] = True
    res = construct_hybrid(
        a, b, mask_a, mask_b, (1.5, 1.5), (3.5, 3.5),
        projection_name="mean_threshold",
    )
    # All four cells in A's mask map cleanly to B's mask under +(2, 2).
    assert res.n_cells_swapped == 4
    assert res.accepted is True


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_supported_modes_listed():
    from observer_worlds.experiments._followup_identity_swap import (
        SUPPORTED_MATCH_MODES, UNSUPPORTED_MATCH_MODES,
    )
    # Stage 5A: morphology_nearest is now supported.
    assert "same_area" in SUPPORTED_MATCH_MODES
    assert "feature_nearest" in SUPPORTED_MATCH_MODES
    assert "morphology_nearest" in SUPPORTED_MATCH_MODES
    # Unsupported modes raise NotImplementedError, not silently succeed.
    for mode in UNSUPPORTED_MATCH_MODES:
        with pytest.raises(NotImplementedError):
            from observer_worlds.experiments._followup_identity_swap import (
                find_candidate_pairs,
            )
            find_candidate_pairs([], match_mode=mode, max_pairs=1)


# ---------------------------------------------------------------------------
# Stage 5A: visible-similarity gating
# ---------------------------------------------------------------------------


def test_translation_aligned_iou_identical_masks_is_one():
    import numpy as np
    from observer_worlds.experiments._followup_identity_swap import (
        _translation_aligned_iou,
    )
    m = np.zeros((6, 6), dtype=bool); m[2:4, 2:4] = True
    assert _translation_aligned_iou(m, m) == pytest.approx(1.0)


def test_translation_aligned_iou_disjoint_masks_is_zero():
    import numpy as np
    from observer_worlds.experiments._followup_identity_swap import (
        _translation_aligned_iou,
    )
    a = np.zeros((6, 6), dtype=bool); a[0, 0] = True
    b = np.zeros((6, 6), dtype=bool); b[5, 5] = True
    # Translation alignment crops both to 1x1; once aligned they're
    # identical 1x1 patches -> IoU = 1. The metric is intentionally
    # translation-invariant; this test documents that.
    assert _translation_aligned_iou(a, b) == pytest.approx(1.0)


def test_translation_aligned_iou_different_shapes_below_one():
    import numpy as np
    from observer_worlds.experiments._followup_identity_swap import (
        _translation_aligned_iou,
    )
    a = np.zeros((6, 6), dtype=bool); a[1:3, 1:5] = True   # 2x4 rectangle
    b = np.zeros((6, 6), dtype=bool); b[1:5, 2:4] = True   # 4x2 rectangle
    iou = _translation_aligned_iou(a, b)
    assert 0.0 < iou < 1.0


def test_compute_visible_similarity_returns_combined_components():
    """The combined score is the average of three components and lives
    in [0, 1]."""
    from observer_worlds.experiments._followup_identity_swap import (
        compute_visible_similarity,
    )
    cic_a, _ = _tiny_cic()
    cic_b, _ = _tiny_cic()
    vs = compute_visible_similarity(cic_a, cic_b)
    assert 0.0 <= vs["translation_aligned_iou"] <= 1.0
    assert 0.0 <= vs["area_ratio"] <= 1.0
    assert 0.0 <= vs["bbox_aspect_similarity"] <= 1.0
    assert 0.0 <= vs["combined"] <= 1.0


def _tiny_cic():
    """Local helper to build a CandidateInCell quickly. Mirrors the
    helper in test_agent_tasks.py — kept local for module isolation."""
    import numpy as np
    from observer_worlds.experiments._followup_identity_swap import (
        CandidateInCell,
    )
    from observer_worlds.experiments._followup_projection import (
        CandidateRef, initial_4d_state, run_substrate,
    )
    from observer_worlds.search.rules import FractionalRule
    rule = FractionalRule(
        birth_min=0.4, birth_max=0.6,
        survive_min=0.3, survive_max=0.7,
        initial_density=0.5,
    )
    bs = rule.to_bsrule()
    grid = (10, 10, 3, 3)
    state0 = initial_4d_state(grid, 0.5, seed=42)
    stream = run_substrate(bs, state0, 8, backend="numpy")
    peak_mask = np.zeros((10, 10), dtype=np.uint8)
    peak_mask[3:6, 3:6] = 1
    peak_interior = peak_mask.copy()
    cand = CandidateRef(
        candidate_id=0, track_id=0, peak_frame=2, peak_mask=peak_mask,
        peak_interior=peak_interior, peak_bbox=(3, 3, 5, 5), lifetime=5,
    )
    return CandidateInCell(
        cell_id="r|42", rule_id="r", rule_source="src", seed=42,
        cand=cand, state_at_peak=stream[2].copy(), state_stream=stream,
    ), bs


def test_min_visible_similarity_threshold_rejects_low_quality_pair():
    """A pair whose combined visible_similarity is below the threshold
    is recorded with valid_swap=False and invalid_reason starting
    with 'visible_similarity_too_low'."""
    from observer_worlds.experiments._followup_identity_swap import (
        measure_pair,
    )
    cic_a, bs = _tiny_cic()
    cic_b, _ = _tiny_cic()
    cic_b.seed = 99   # so different_cell-like check passes
    cic_b.cell_id = "r|99"
    # Pretend the matcher reported visible_similarity = 0.05 (low).
    meta = {
        "match_distance": 0.0,
        "visible_similarity": 0.05,
        "translation_aligned_iou": 0.0,
        "area_ratio": 0.05,
        "bbox_aspect_similarity": 0.10,
        "area_a": 9.0, "area_b": 9.0,
    }
    res = measure_pair(
        pair_id=0, A=cic_a, B=cic_b, match_meta=meta,
        match_mode="morphology_nearest",
        projection_name="mean_threshold",
        horizons=(2, 5), rule_bs=bs, backend="numpy",
        min_visible_similarity=0.30,
    )
    assert res.valid_swap_a is False
    assert res.valid_swap_b is False
    assert res.invalid_reason_a is not None
    assert res.invalid_reason_a.startswith("visible_similarity_too_low")
    assert res.invalid_reason_b == res.invalid_reason_a


def test_morphology_features_computed():
    from observer_worlds.experiments._followup_identity_swap import (
        _morphology_features,
    )
    cic, _ = _tiny_cic()
    feats = _morphology_features(cic)
    # Returns 4-vector [area, perimeter, aspect, compactness].
    assert feats.shape == (4,)
    assert feats[0] > 0  # area
    assert feats[1] > 0  # perimeter
    assert 0.0 < feats[2] < 10.0  # aspect ratio (sane)
    assert 0.0 <= feats[3] <= 2.0  # compactness loose


# ---------------------------------------------------------------------------
# End-to-end tiny smoke
# ---------------------------------------------------------------------------


def test_tiny_smoke_writes_full_artifact_set(tmp_path: Path):
    """Smallest viable end-to-end: 1 rule, 2 seeds, T=20, grid 12x12x3x3,
    feature_nearest matching, mean_threshold projection."""
    rc = runner.main([
        "--quick",
        "--out-root", str(tmp_path),
        "--label", "tiny_swap",
        "--n-rules-per-source", "1",
        "--test-seeds", "6000", "6001",
        "--timesteps", "20",
        "--max-candidates", "3",
        "--max-pairs", "5",
        "--horizons", "3", "5",
        "--projection", "mean_threshold",
        "--matching-mode", "feature_nearest",
        "--n-workers", "1",
        "--grid", "12", "12", "3", "3",
    ])
    assert rc == 0
    out = next(tmp_path.iterdir())
    expected = {
        "config.json", "frozen_manifest.json",
        "candidate_pairs.csv", "swap_interventions.csv",
        "identity_scores.csv", "stats_summary.json", "summary.md",
    }
    have = {p.name for p in out.iterdir() if p.is_file()}
    assert expected.issubset(have), f"missing: {expected - have}"
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["projection"] == "mean_threshold"
    assert cfg["matching_mode"] == "feature_nearest"
    summary = json.loads(
        (out / "stats_summary.json").read_text(encoding="utf-8")
    )
    assert summary["stage"] == 3
    assert (out / "plots").is_dir()

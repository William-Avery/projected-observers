"""Tests for ``observer_worlds.search.viability`` (M4A scoring + evaluation)."""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.detection.tracking import Track
from observer_worlds.metrics.persistence import CandidateScore
from observer_worlds.search.rules import FractionalRule
from observer_worlds.search.viability import (
    ViabilityWeights,
    _trapezoid,
    _triangle,
    compute_viability_score,
    evaluate_viability,
    evaluate_viability_multi_seed,
)


# ---------------------------------------------------------------------------
# Helpers (private to this test module)
# ---------------------------------------------------------------------------


def _build_track(track_id: int, n_frames: int, area: int = 10, start_frame: int = 0) -> Track:
    """Build a minimal Track with ``n_frames`` contiguous frames, all-True
    5x5 mask in a 16x16 grid.  Used to manufacture persistent components."""
    H, W = 16, 16
    mask = np.zeros((H, W), dtype=bool)
    mask[5:10, 5:10] = True
    interior = mask.copy()
    boundary = np.zeros_like(mask)
    env = np.zeros_like(mask)
    track = Track(
        track_id=track_id,
        birth_frame=start_frame,
        last_frame=start_frame + n_frames - 1,
    )
    for t in range(n_frames):
        track.frames.append(start_frame + t)
        track.centroid_history.append((7.0, 7.0))
        track.area_history.append(area)
        track.bbox_history.append((5, 5, 10, 10))
        track.mask_history.append(mask.copy())
        track.interior_history.append(interior.copy())
        track.boundary_history.append(boundary.copy())
        track.env_history.append(env.copy())
    return track


def _candidate(
    track_id: int,
    age: int,
    *,
    mean_area: float = 25,
    internal_variation: float = 1.0,
    boundedness: float = 0.8,
    length: int | None = None,
) -> CandidateScore:
    return CandidateScore(
        track_id=track_id,
        age=age,
        length=length if length is not None else age,
        is_candidate=True,
        boundedness=boundedness,
        internal_variation=internal_variation,
        mean_area=float(mean_area),
        max_area=int(mean_area),
        reasons=[],
    )


def _flicker_frames(
    n_frames: int = 100,
    *,
    grid: tuple[int, int] = (16, 16),
    initial_density: float = 0.20,
    flip_prob: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Generate ``n_frames`` of frames where each cell flips with probability
    ``flip_prob`` between successive frames.  Seed-sensitive."""
    rng = np.random.default_rng(seed)
    H, W = grid
    frames = np.zeros((n_frames, H, W), dtype=np.uint8)
    frames[0] = (rng.random((H, W)) < initial_density).astype(np.uint8)
    for t in range(1, n_frames):
        flips = (rng.random((H, W)) < flip_prob).astype(np.uint8)
        frames[t] = np.bitwise_xor(frames[t - 1], flips)
    return frames


# ---------------------------------------------------------------------------
# compute_viability_score: penalty stacks
# ---------------------------------------------------------------------------


def test_extinction_penalty_when_frames_die() -> None:
    """100 frames of all-zeros: extinction + frozen penalties both fire."""
    frames = np.zeros((100, 16, 16), dtype=np.uint8)
    score, comp = compute_viability_score(frames, tracks=[], candidates=[])
    assert comp["extinction_penalty"] == pytest.approx(1.0)
    assert comp["frozen_world_penalty"] == pytest.approx(1.0)
    assert comp["saturation_penalty"] == pytest.approx(0.0)
    # Penalty stack: -3 (extinction) + -1 (frozen) ~ -4.
    assert score == pytest.approx(-4.0)


def test_saturation_penalty_when_frames_full() -> None:
    """100 frames of all-ones: saturation + frozen penalties both fire."""
    frames = np.ones((100, 16, 16), dtype=np.uint8)
    score, comp = compute_viability_score(frames, tracks=[], candidates=[])
    assert comp["saturation_penalty"] == pytest.approx(1.0)
    assert comp["frozen_world_penalty"] == pytest.approx(1.0)
    assert comp["extinction_penalty"] == pytest.approx(0.0)
    # Penalty stack: -3 (saturation) + -1 (frozen) ~ -4.
    assert score == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# compute_viability_score: per-component sub-scores
# ---------------------------------------------------------------------------


def test_target_activity_score_in_range() -> None:
    """100 identical frames at ~20% activity -> target plateau, zero change."""
    frames = np.zeros((100, 16, 16), dtype=np.uint8)
    flat = frames.reshape(100, -1)
    flat[:, :51] = 1  # ~19.9% active
    score, comp = compute_viability_score(frames, tracks=[], candidates=[])
    assert comp["target_activity_score"] == pytest.approx(1.0)
    # Identical frames -> no flips -> zero change score.
    assert comp["temporal_change_score"] == pytest.approx(0.0)
    # No flips also triggers frozen-world penalty.
    assert comp["frozen_world_penalty"] == pytest.approx(1.0)


def test_temporal_change_rewards_moderate_flicker() -> None:
    """Per-cell flip probability 0.05 -> mean_delta ~0.05 -> plateau reward."""
    frames = _flicker_frames(n_frames=100, flip_prob=0.05, seed=42)
    score, comp = compute_viability_score(frames, tracks=[], candidates=[])
    assert comp["temporal_change_score"] > 0.5
    # Flicker generator never freezes; verify the penalty is off.
    assert comp["frozen_world_penalty"] == pytest.approx(0.0)


def test_persistent_component_score_increases_with_count() -> None:
    """5 persistent candidates score higher than 1 (log1p-saturated reward)."""
    frames = _flicker_frames(n_frames=100, flip_prob=0.05, seed=42)
    one = [_candidate(0, age=30)]
    five = [_candidate(i, age=30) for i in range(5)]
    _, comp1 = compute_viability_score(frames, tracks=[], candidates=one)
    _, comp5 = compute_viability_score(frames, tracks=[], candidates=five)
    assert comp5["persistent_component_score"] > comp1["persistent_component_score"]


def test_boundedness_score_zero_when_areas_extreme() -> None:
    """Areas of 1 (too small) and 200 (>0.5*256) both contribute zero."""
    frames = _flicker_frames(n_frames=100, flip_prob=0.05, seed=42)
    cands = [
        _candidate(0, age=30, mean_area=1),
        _candidate(1, age=30, mean_area=200),
    ]
    _, comp = compute_viability_score(frames, tracks=[], candidates=cands)
    assert comp["boundedness_score"] == pytest.approx(0.0)


def test_boundedness_score_high_when_areas_in_range() -> None:
    """A candidate with mean_area in the trapezoid plateau scores ~1.0."""
    frames = _flicker_frames(n_frames=100, flip_prob=0.05, seed=42)
    cands = [_candidate(0, age=30, mean_area=25)]
    _, comp = compute_viability_score(frames, tracks=[], candidates=cands)
    assert comp["boundedness_score"] == pytest.approx(1.0)


def test_diversity_score_increases_with_size_buckets() -> None:
    """More distinct log2 area buckets -> higher diversity score."""
    frames = _flicker_frames(n_frames=100, flip_prob=0.05, seed=42)
    same = [_candidate(i, age=30, mean_area=25) for i in range(4)]
    varied = [
        _candidate(0, age=30, mean_area=5),
        _candidate(1, age=30, mean_area=25),
        _candidate(2, age=30, mean_area=100),
        _candidate(3, age=30, mean_area=200),
    ]
    _, comp_same = compute_viability_score(frames, tracks=[], candidates=same)
    _, comp_var = compute_viability_score(frames, tracks=[], candidates=varied)
    assert comp_var["diversity_score"] > comp_same["diversity_score"]


def test_compute_viability_score_returns_components_dict() -> None:
    """The components dict must carry all 8 documented keys."""
    frames = np.zeros((10, 16, 16), dtype=np.uint8)
    score, comp = compute_viability_score(frames, tracks=[], candidates=[])
    expected_keys = {
        "persistent_component_score",
        "target_activity_score",
        "temporal_change_score",
        "boundedness_score",
        "diversity_score",
        "extinction_penalty",
        "saturation_penalty",
        "frozen_world_penalty",
    }
    assert set(comp.keys()) == expected_keys

    # Reconstruct the score from raw components using default weights and
    # confirm it matches the returned float.
    w = ViabilityWeights()
    reconstructed = (
        w.persistent_component * comp["persistent_component_score"]
        + w.target_activity * comp["target_activity_score"]
        + w.temporal_change * comp["temporal_change_score"]
        + w.boundedness * comp["boundedness_score"]
        + w.diversity * comp["diversity_score"]
        - w.extinction_penalty * comp["extinction_penalty"]
        - w.saturation_penalty * comp["saturation_penalty"]
        - w.frozen_world_penalty * comp["frozen_world_penalty"]
    )
    assert score == pytest.approx(reconstructed, abs=1e-9)


# ---------------------------------------------------------------------------
# evaluate_viability / evaluate_viability_multi_seed
# ---------------------------------------------------------------------------


# A rule that essentially cannot give birth or survive -> die-off scout.
_DIE_OFF_RULE = FractionalRule(
    birth_min=0.95,
    birth_max=0.99,
    survive_min=0.95,
    survive_max=0.99,
    initial_density=0.05,
)


def test_evaluate_viability_die_off_yields_negative_score() -> None:
    score, _comp, _diag = evaluate_viability(
        _DIE_OFF_RULE,
        seed=0,
        grid_shape=(8, 8, 2, 2),
        timesteps=20,
    )
    assert score < 0.0


def test_evaluate_viability_returns_report_with_diagnostics() -> None:
    report = evaluate_viability_multi_seed(
        _DIE_OFF_RULE,
        n_seeds=2,
        base_seed=0,
        grid_shape=(8, 8, 2, 2),
        timesteps=20,
    )
    assert report.n_seeds == 2
    assert len(report.per_seed_scores) == 2
    assert len(report.activity_traces) == 2
    assert len(report.component_count_traces) == 2
    assert report.viability_score == pytest.approx(
        float(np.mean(report.per_seed_scores)), abs=1e-9
    )


# ---------------------------------------------------------------------------
# Helper kernels
# ---------------------------------------------------------------------------


def test_helpers_trapezoid_kernel() -> None:
    # On the plateau-low corner -> 1.0.
    assert _trapezoid(0.10, 0.05, 0.10, 0.30, 0.50) == pytest.approx(1.0)
    # On the plateau-high corner -> 1.0.
    assert _trapezoid(0.30, 0.05, 0.10, 0.30, 0.50) == pytest.approx(1.0)
    # Below ``low`` -> 0.0.
    assert _trapezoid(0.04, 0.05, 0.10, 0.30, 0.50) == pytest.approx(0.0)
    # Equal to ``low`` -> 0.0 (strict inequality).
    assert _trapezoid(0.05, 0.05, 0.10, 0.30, 0.50) == pytest.approx(0.0)
    # At ``high`` -> 0.0 (strict inequality).
    assert _trapezoid(0.50, 0.05, 0.10, 0.30, 0.50) == pytest.approx(0.0)
    # Midway up the left ramp (0.05 -> 0.10) at x=0.075 -> 0.5.
    assert _trapezoid(0.075, 0.05, 0.10, 0.30, 0.50) == pytest.approx(0.5)
    # Midway down the right ramp (0.30 -> 0.50) at x=0.40 -> 0.5.
    assert _trapezoid(0.40, 0.05, 0.10, 0.30, 0.50) == pytest.approx(0.5)
    # Triangle sanity: midpoint of [0, 1] is 1.0; endpoints are 0.
    assert _triangle(0.5, 0.0, 1.0) == pytest.approx(1.0)
    assert _triangle(0.0, 0.0, 1.0) == pytest.approx(0.0)
    assert _triangle(1.0, 0.0, 1.0) == pytest.approx(0.0)

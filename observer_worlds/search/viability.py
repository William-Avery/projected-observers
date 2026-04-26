"""4D rule viability scoring (M4A).

Before any observer-likeness metric can be computed on the 4D world, the
underlying dynamics must be **viable**: the projected 2D world must
exhibit non-trivial activity that produces persistent bounded structures.

The viability score is a weighted sum of seven components computed
directly from a scout simulation's projected frames + tracked structures:

    + 2.0 * persistent_component_score
    + 1.0 * target_activity_score
    + 1.0 * temporal_change_score
    + 1.0 * boundedness_score
    + 0.5 * diversity_score
    - 3.0 * extinction_penalty
    - 3.0 * saturation_penalty
    - 1.0 * frozen_world_penalty

Each component is normalized to roughly [0, 1] so the weights are directly
interpretable.  The score itself can be negative (penalties dominate) or
positive (rewards dominate); rules with score > 0 are the viable ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from observer_worlds.detection import GreedyTracker, extract_components
from observer_worlds.metrics import score_persistence
from observer_worlds.search.rules import FractionalRule
from observer_worlds.search.fitness import simulate_4d_in_memory
from observer_worlds.utils.config import DetectionConfig


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViabilityWeights:
    """Default weights from the M4A spec."""

    persistent_component: float = 2.0
    target_activity: float = 1.0
    temporal_change: float = 1.0
    boundedness: float = 1.0
    diversity: float = 0.5
    extinction_penalty: float = 3.0
    saturation_penalty: float = 3.0
    frozen_world_penalty: float = 1.0


DEFAULT_VIABILITY_WEIGHTS = ViabilityWeights()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class ViabilityReport:
    """Per-rule viability summary.

    Aggregates across multiple seeds when produced by
    :func:`evaluate_viability_multi_seed` (single-seed reports are
    surfaced unchanged).
    """

    rule: FractionalRule
    n_seeds: int
    viability_score: float                 # mean across seeds (or single value)

    # Per-component sub-scores (mean across seeds when n_seeds > 1).
    persistent_component_score: float
    target_activity_score: float
    temporal_change_score: float
    boundedness_score: float
    diversity_score: float
    extinction_penalty: float
    saturation_penalty: float
    frozen_world_penalty: float

    # Diagnostics (mean across seeds).
    final_active_fraction: float
    mean_late_active_fraction: float
    n_components_over_time_mean: float     # avg # of components per frame
    max_component_lifetime: int            # max age across seeds
    mean_component_lifetime: float
    n_persistent_components: float         # mean across seeds

    # Per-seed scores, kept for variance inspection.
    per_seed_scores: list[float] = field(default_factory=list)
    per_seed_aborted: list[bool] = field(default_factory=list)
    activity_traces: list[list[float]] = field(default_factory=list)
    component_count_traces: list[list[int]] = field(default_factory=list)

    sim_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _safe_clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _triangle(x: float, low: float, high: float) -> float:
    """Triangular kernel: 0 at boundaries, 1 at midpoint."""
    if x <= low or x >= high:
        return 0.0
    mid = 0.5 * (low + high)
    half = 0.5 * (high - low)
    return _safe_clip(1.0 - abs(x - mid) / half)


def _trapezoid(x: float, low: float, plateau_low: float, plateau_high: float, high: float) -> float:
    """Trapezoidal kernel: 0 below ``low`` or above ``high``, 1 between
    ``plateau_low`` and ``plateau_high``, linear ramps elsewhere."""
    if x <= low or x >= high:
        return 0.0
    if plateau_low <= x <= plateau_high:
        return 1.0
    if x < plateau_low:
        return _safe_clip((x - low) / max(plateau_low - low, 1e-12))
    return _safe_clip((high - x) / max(high - plateau_high, 1e-12))


def compute_viability_score(
    frames_2d: np.ndarray,
    tracks: list,
    candidates: list,
    *,
    min_component_age: int = 20,
    late_window_frac: float = 0.5,
    target_low: float = 0.05,
    target_high: float = 0.35,
    extinction_threshold: float = 0.005,
    saturation_threshold: float = 0.80,
    frozen_change_threshold: float = 1e-3,
    weights: ViabilityWeights | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the composite viability score.

    Returns ``(score, components)`` where ``components`` maps each
    sub-score / penalty name to its raw value (in roughly [0, 1]).  The
    composite ``score`` is the weighted sum.
    """
    weights = weights or DEFAULT_VIABILITY_WEIGHTS
    T = frames_2d.shape[0]
    if T == 0:
        return 0.0, _zero_components()

    # ---- per-frame activity --------------------------------------------------
    # mean active fraction in projected world per frame.
    active_per_frame = frames_2d.reshape(T, -1).mean(axis=1).astype(np.float64)
    late_start = int(T * (1.0 - late_window_frac))
    late = active_per_frame[late_start:] if late_start < T else active_per_frame[-1:]

    # extinction_penalty: fraction of late frames where activity < threshold.
    extinction_penalty = float((late < extinction_threshold).mean())
    # saturation_penalty: fraction of late frames where activity > threshold.
    saturation_penalty = float((late > saturation_threshold).mean())

    # target_activity_score: trapezoidal reward in [target_low, target_high].
    mean_late = float(late.mean())
    target_activity_score = _trapezoid(
        mean_late,
        low=target_low * 0.5,
        plateau_low=target_low,
        plateau_high=target_high,
        high=target_high * 1.5,
    )

    # ---- temporal change ----------------------------------------------------
    # Mean per-frame delta = mean fraction of cells that flipped between
    # consecutive frames in the late window.
    if T >= 2:
        # Vectorized: XOR successive frames, average per cell over time.
        xor = np.bitwise_xor(frames_2d[1:].astype(np.uint8), frames_2d[:-1].astype(np.uint8))
        delta_per_frame = xor.reshape(T - 1, -1).mean(axis=1)
        late_delta = delta_per_frame[late_start - 1: T - 1] if late_start - 1 >= 0 else delta_per_frame
        mean_delta = float(late_delta.mean()) if len(late_delta) else 0.0
    else:
        mean_delta = 0.0

    # frozen_world_penalty: 1 if mean delta is essentially zero, else 0.
    frozen_world_penalty = 1.0 if mean_delta < frozen_change_threshold else 0.0
    # temporal_change_score: triangular kernel — reward moderate change,
    # penalize zero and noise (>0.5).
    temporal_change_score = _trapezoid(
        mean_delta,
        low=0.001,
        plateau_low=0.01,
        plateau_high=0.20,
        high=0.50,
    )

    # ---- persistence + boundedness + diversity ------------------------------
    cand_only = [c for c in candidates if c.is_candidate]
    persistent = [c for c in cand_only if c.age >= min_component_age]
    n_persistent = len(persistent)
    # log-scaled, clipped: reward up to ~10 persistent structures.
    persistent_component_score = _safe_clip(np.log1p(n_persistent) / np.log1p(10.0))

    # boundedness_score: mean over persistent candidates of
    # 1 if mean_area in [4, 0.4 * grid] else fall-off. Use trapezoid.
    grid_cells = float(frames_2d.shape[1] * frames_2d.shape[2])
    if persistent:
        bnd_terms = []
        for c in persistent:
            area = float(c.mean_area)
            # Reward areas that aren't tiny and aren't whole-grid.
            term = _trapezoid(
                area,
                low=2.0,
                plateau_low=4.0,
                plateau_high=0.25 * grid_cells,
                high=0.50 * grid_cells,
            )
            bnd_terms.append(term)
        boundedness_score = float(np.mean(bnd_terms))
    else:
        boundedness_score = 0.0

    # diversity_score: number of distinct area buckets (log scale).
    # Bucket areas into bins of doubling width.
    if persistent:
        areas = np.array([c.mean_area for c in persistent])
        # log2 bins.
        buckets = set()
        for a in areas:
            if a > 0:
                buckets.add(int(np.floor(np.log2(a + 1))))
        diversity_score = _safe_clip(np.log1p(len(buckets)) / np.log1p(6.0))
    else:
        diversity_score = 0.0

    components_dict = {
        "persistent_component_score": persistent_component_score,
        "target_activity_score": target_activity_score,
        "temporal_change_score": temporal_change_score,
        "boundedness_score": boundedness_score,
        "diversity_score": diversity_score,
        "extinction_penalty": extinction_penalty,
        "saturation_penalty": saturation_penalty,
        "frozen_world_penalty": frozen_world_penalty,
    }

    score = (
        weights.persistent_component * persistent_component_score
        + weights.target_activity * target_activity_score
        + weights.temporal_change * temporal_change_score
        + weights.boundedness * boundedness_score
        + weights.diversity * diversity_score
        - weights.extinction_penalty * extinction_penalty
        - weights.saturation_penalty * saturation_penalty
        - weights.frozen_world_penalty * frozen_world_penalty
    )

    return float(score), components_dict


def _zero_components() -> dict[str, float]:
    return {
        "persistent_component_score": 0.0,
        "target_activity_score": 0.0,
        "temporal_change_score": 0.0,
        "boundedness_score": 0.0,
        "diversity_score": 0.0,
        "extinction_penalty": 0.0,
        "saturation_penalty": 0.0,
        "frozen_world_penalty": 0.0,
    }


# ---------------------------------------------------------------------------
# Single-seed evaluation
# ---------------------------------------------------------------------------


def evaluate_viability(
    rule: FractionalRule,
    *,
    seed: int,
    grid_shape: tuple[int, int, int, int] = (64, 64, 8, 8),
    timesteps: int = 300,
    detection_config: DetectionConfig | None = None,
    backend: str = "numba",
    weights: ViabilityWeights | None = None,
    min_component_age: int = 20,
    early_abort: bool = True,
) -> tuple[float, dict[str, float], dict[str, object]]:
    """Run a single scout for one seed and return ``(score, components, diag)``.

    ``diag`` carries diagnostics consumed by
    :func:`evaluate_viability_multi_seed` to aggregate across seeds.
    """
    detection_config = detection_config or DetectionConfig()
    bsrule = rule.to_bsrule()

    frames, active_trace, abort_reason = simulate_4d_in_memory(
        bsrule,
        grid_shape=grid_shape,
        timesteps=timesteps,
        initial_density=rule.initial_density,
        seed=seed,
        backend=backend,
        early_abort=early_abort,
    )

    aborted = abort_reason != "completed"

    # Detect + track even on aborted scouts (an early-abort due to die-off
    # still has frames; tracks will simply be empty).
    tracker = GreedyTracker(config=detection_config)
    component_counts: list[int] = []
    for t in range(frames.shape[0]):
        comps = extract_components(frames[t], frame_idx=t, config=detection_config)
        component_counts.append(len(comps))
        tracker.update(t, comps)
    tracks = tracker.finalize()
    candidates = score_persistence(
        tracks, grid_shape=(grid_shape[0], grid_shape[1]), config=detection_config
    )

    score, components = compute_viability_score(
        frames, tracks, candidates,
        min_component_age=min_component_age,
        weights=weights,
    )

    cand_only = [c for c in candidates if c.is_candidate]
    ages = [c.age for c in cand_only]
    diag: dict[str, object] = {
        "abort_reason": abort_reason,
        "aborted": aborted,
        "frames_completed": int(frames.shape[0]),
        "active_trace": active_trace.tolist(),
        "component_counts": component_counts,
        "n_tracks": len(tracks),
        "n_candidates": len(cand_only),
        "ages": ages,
        "final_active_fraction": float(active_trace[-1]) if len(active_trace) else 0.0,
        "mean_late_active_fraction": float(
            active_trace[len(active_trace) // 2:].mean()
        ) if len(active_trace) else 0.0,
    }
    return score, components, diag


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------


def evaluate_viability_multi_seed(
    rule: FractionalRule,
    *,
    n_seeds: int = 3,
    base_seed: int = 0,
    grid_shape: tuple[int, int, int, int] = (64, 64, 8, 8),
    timesteps: int = 300,
    detection_config: DetectionConfig | None = None,
    backend: str = "numba",
    weights: ViabilityWeights | None = None,
    min_component_age: int = 20,
    early_abort: bool = True,
) -> ViabilityReport:
    """Run ``n_seeds`` scouts and aggregate into a single :class:`ViabilityReport`.

    Score and component values are averaged across seeds; diagnostic time
    series are retained per seed for inspection.
    """
    import time

    t0 = time.time()
    per_seed_scores: list[float] = []
    per_seed_components: list[dict[str, float]] = []
    per_seed_diag: list[dict[str, object]] = []
    for i in range(n_seeds):
        s = base_seed + i
        score, components, diag = evaluate_viability(
            rule,
            seed=s,
            grid_shape=grid_shape,
            timesteps=timesteps,
            detection_config=detection_config,
            backend=backend,
            weights=weights,
            min_component_age=min_component_age,
            early_abort=early_abort,
        )
        per_seed_scores.append(score)
        per_seed_components.append(components)
        per_seed_diag.append(diag)

    sim_time = time.time() - t0

    def _mean_component(name: str) -> float:
        return float(np.mean([c[name] for c in per_seed_components]))

    # Aggregated diagnostics.
    finals = [d["final_active_fraction"] for d in per_seed_diag]
    means_late = [d["mean_late_active_fraction"] for d in per_seed_diag]
    comp_traces = [d["component_counts"] for d in per_seed_diag]
    n_components_mean = float(np.mean([np.mean(t) if t else 0.0 for t in comp_traces]))
    all_ages = [a for d in per_seed_diag for a in d["ages"]]
    max_age = int(max(all_ages)) if all_ages else 0
    mean_age = float(np.mean(all_ages)) if all_ages else 0.0
    n_persistent_per_seed = [
        sum(1 for a in d["ages"] if a >= min_component_age) for d in per_seed_diag
    ]

    return ViabilityReport(
        rule=rule,
        n_seeds=n_seeds,
        viability_score=float(np.mean(per_seed_scores)),
        persistent_component_score=_mean_component("persistent_component_score"),
        target_activity_score=_mean_component("target_activity_score"),
        temporal_change_score=_mean_component("temporal_change_score"),
        boundedness_score=_mean_component("boundedness_score"),
        diversity_score=_mean_component("diversity_score"),
        extinction_penalty=_mean_component("extinction_penalty"),
        saturation_penalty=_mean_component("saturation_penalty"),
        frozen_world_penalty=_mean_component("frozen_world_penalty"),
        final_active_fraction=float(np.mean(finals)),
        mean_late_active_fraction=float(np.mean(means_late)),
        n_components_over_time_mean=n_components_mean,
        max_component_lifetime=max_age,
        mean_component_lifetime=mean_age,
        n_persistent_components=float(np.mean(n_persistent_per_seed)),
        per_seed_scores=per_seed_scores,
        per_seed_aborted=[bool(d["aborted"]) for d in per_seed_diag],
        activity_traces=[list(d["active_trace"]) for d in per_seed_diag],
        component_count_traces=[list(d["component_counts"]) for d in per_seed_diag],
        sim_time_seconds=sim_time,
    )

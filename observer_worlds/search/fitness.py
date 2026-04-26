"""Cheap fitness function for 4D rule search.

Computing the full M2 metric suite (Ridge + KFold across time/memory/selfhood)
is too expensive to run on every candidate rule when the search visits
hundreds or thousands of (B, S) configurations.  This module provides a
**cheap proxy** for the eventual observer_score:

    fitness = w_count * log(1 + n_candidates)
            + w_age   * log(1 + mean_age)
            + w_bnd   * mean_boundedness
            + w_var   * mean_internal_variation
            + w_max   * (max_age >= min_age * 2)        # bonus for long-lived

The proxy is monotone in the quantities the persistence filter already
gates on, so a high proxy fitness implies the rule produces tracks the
M2 metrics will actually be able to score.

Scout simulations run in-memory (no zarr store) on a small grid (default
32x32x4x4) for a short horizon (default T=80).  Trivial dynamics
(immediate die-off or saturation) trigger an early abort so wasted scouts
don't dominate runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from observer_worlds.detection import GreedyTracker, extract_components
from observer_worlds.metrics import score_persistence
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import DetectionConfig
from observer_worlds.worlds import CA4D, BSRule, project


# Default fitness weights.  Documented in the README.  The log scaling on
# count and age is intentional: doubling either should give a sublinear
# fitness boost so a rule with a few stable structures isn't dominated by
# one with many unstable ones.
DEFAULT_FITNESS_WEIGHTS: dict[str, float] = {
    "count": 1.0,
    "age": 0.5,
    "boundedness": 0.3,
    "internal_variation": 0.2,
    "max_age_bonus": 0.5,
}


@dataclass
class FitnessReport:
    """Per-rule fitness summary."""

    rule: BSRule
    fitness: float

    # Components.
    n_tracks: int
    n_candidates: int
    max_age: int
    mean_age: float
    mean_boundedness: float
    mean_internal_variation: float
    mean_area: float
    final_active_fraction: float

    # Diagnostics.
    aborted: bool
    abort_reason: str
    sim_time_seconds: float

    # Trace of active-cell fraction in the projected 2D world over time.
    # Useful for plotting / debugging; capped at 200 entries to keep memory
    # bounded across thousands of trials.
    active_fraction_trace: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scout simulation (in-memory, no zarr)
# ---------------------------------------------------------------------------


def simulate_4d_in_memory(
    rule: BSRule,
    *,
    grid_shape: tuple[int, int, int, int] = (32, 32, 4, 4),
    timesteps: int = 80,
    initial_density: float = 0.15,
    seed: int = 0,
    backend: str = "numba",
    projection_method: str = "mean_threshold",
    projection_theta: float = 0.5,
    early_abort: bool = True,
    abort_after: int = 10,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run a 4D scout simulation entirely in memory.

    Returns ``(frames_2d, active_fraction_history, abort_reason)``:

    - ``frames_2d``: ``(T_completed, Nx, Ny)`` uint8.  Note ``T_completed``
      may be less than ``timesteps`` when ``early_abort`` triggers.
    - ``active_fraction_history``: ``(T_completed,)`` float fraction of
      active cells in the projected 2D frame at each step.
    - ``abort_reason``: one of ``"completed"``, ``"die_off"``, ``"saturated"``.
    """
    rng = seeded_rng(seed)
    ca = CA4D(shape=grid_shape, rule=rule, backend=backend)
    ca.initialize_random(density=initial_density, rng=rng)

    Nx, Ny = grid_shape[0], grid_shape[1]
    frames = np.empty((timesteps, Nx, Ny), dtype=np.uint8)
    active = np.empty(timesteps, dtype=np.float32)

    def _proj(state: np.ndarray) -> np.ndarray:
        return project(state, method=projection_method, theta=projection_theta)

    Y0 = _proj(ca.state)
    frames[0] = Y0
    active[0] = float(Y0.mean())

    abort_reason = "completed"
    completed = timesteps

    for t in range(1, timesteps):
        ca.step()
        Y = _proj(ca.state)
        frames[t] = Y
        active[t] = float(Y.mean())

        if early_abort and t >= abort_after:
            recent = active[max(0, t - 5): t + 1]
            if recent.max() < 1e-4:
                abort_reason = "die_off"
                completed = t + 1
                break
            if recent.min() > 0.99:
                abort_reason = "saturated"
                completed = t + 1
                break

    return frames[:completed], active[:completed], abort_reason


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


def evaluate_rule(
    rule: BSRule,
    *,
    grid_shape: tuple[int, int, int, int] = (32, 32, 4, 4),
    timesteps: int = 80,
    initial_density: float = 0.15,
    detection_config: DetectionConfig | None = None,
    seed: int = 0,
    backend: str = "numba",
    weights: dict[str, float] | None = None,
    early_abort: bool = True,
) -> FitnessReport:
    """Score a single (B, S) rule using the cheap-proxy fitness.

    Pipeline: scout simulation -> detect+track -> persistence filter.
    Computes the proxy fitness; does NOT run the full M2 metric suite.
    Use ``observer_worlds.experiments._pipeline.compute_full_metrics``
    on the top-K rules afterwards if you want the real observer_score.
    """
    weights = {**DEFAULT_FITNESS_WEIGHTS, **(weights or {})}
    detection_config = detection_config or DetectionConfig()

    t0 = time.time()
    frames, active, abort_reason = simulate_4d_in_memory(
        rule,
        grid_shape=grid_shape,
        timesteps=timesteps,
        initial_density=initial_density,
        seed=seed,
        backend=backend,
        early_abort=early_abort,
    )
    sim_time = time.time() - t0

    aborted = abort_reason != "completed"

    # Sub-sample the trace for memory.
    if len(active) > 200:
        idx = np.linspace(0, len(active) - 1, 200).astype(int)
        trace = active[idx].tolist()
    else:
        trace = active.tolist()

    if aborted:
        return FitnessReport(
            rule=rule,
            fitness=0.0,
            n_tracks=0,
            n_candidates=0,
            max_age=0,
            mean_age=0.0,
            mean_boundedness=0.0,
            mean_internal_variation=0.0,
            mean_area=0.0,
            final_active_fraction=float(active[-1]) if len(active) else 0.0,
            aborted=True,
            abort_reason=abort_reason,
            sim_time_seconds=sim_time,
            active_fraction_trace=trace,
        )

    # Detect + track.
    tracker = GreedyTracker(config=detection_config)
    for t in range(frames.shape[0]):
        components = extract_components(frames[t], frame_idx=t, config=detection_config)
        tracker.update(t, components)
    tracks = tracker.finalize()

    candidates = score_persistence(
        tracks, grid_shape=(grid_shape[0], grid_shape[1]), config=detection_config
    )
    cand_only = [c for c in candidates if c.is_candidate]
    n_candidates = len(cand_only)
    n_tracks = len(tracks)

    if n_candidates == 0:
        # Active dynamics but no persistent structures -- score 0.
        return FitnessReport(
            rule=rule,
            fitness=0.0,
            n_tracks=n_tracks,
            n_candidates=0,
            max_age=0,
            mean_age=0.0,
            mean_boundedness=0.0,
            mean_internal_variation=0.0,
            mean_area=0.0,
            final_active_fraction=float(active[-1]),
            aborted=False,
            abort_reason="no_candidates",
            sim_time_seconds=sim_time,
            active_fraction_trace=trace,
        )

    ages = np.array([c.age for c in cand_only], dtype=np.float64)
    bnd = np.array([c.boundedness for c in cand_only], dtype=np.float64)
    var = np.array([c.internal_variation for c in cand_only], dtype=np.float64)
    area = np.array([c.mean_area for c in cand_only], dtype=np.float64)

    max_age = int(ages.max())
    mean_age = float(ages.mean())
    mean_bnd = float(bnd.mean())
    mean_var = float(var.mean())
    mean_area = float(area.mean())

    # Composite fitness.
    fitness = (
        weights["count"] * np.log1p(n_candidates)
        + weights["age"] * np.log1p(mean_age)
        + weights["boundedness"] * mean_bnd
        + weights["internal_variation"] * mean_var
        + weights["max_age_bonus"] * (1.0 if max_age >= 2 * detection_config.min_age else 0.0)
    )

    return FitnessReport(
        rule=rule,
        fitness=float(fitness),
        n_tracks=n_tracks,
        n_candidates=n_candidates,
        max_age=max_age,
        mean_age=mean_age,
        mean_boundedness=mean_bnd,
        mean_internal_variation=mean_var,
        mean_area=mean_area,
        final_active_fraction=float(active[-1]),
        aborted=False,
        abort_reason="completed",
        sim_time_seconds=sim_time,
        active_fraction_trace=trace,
    )


# ---------------------------------------------------------------------------
# Helpers for serialization
# ---------------------------------------------------------------------------


def fitness_report_to_csv_row(r: FitnessReport) -> dict[str, str]:
    """Flatten a FitnessReport into stringly-typed CSV columns."""
    return {
        "fitness": f"{r.fitness:.6f}",
        "rule_birth": "|".join(str(b) for b in r.rule.birth),
        "rule_survival": "|".join(str(s) for s in r.rule.survival),
        "n_tracks": str(r.n_tracks),
        "n_candidates": str(r.n_candidates),
        "max_age": str(r.max_age),
        "mean_age": f"{r.mean_age:.3f}",
        "mean_boundedness": f"{r.mean_boundedness:.4f}",
        "mean_internal_variation": f"{r.mean_internal_variation:.4f}",
        "mean_area": f"{r.mean_area:.2f}",
        "final_active_fraction": f"{r.final_active_fraction:.4f}",
        "aborted": str(r.aborted),
        "abort_reason": r.abort_reason,
        "sim_time_seconds": f"{r.sim_time_seconds:.3f}",
    }


CSV_COLUMNS: tuple[str, ...] = (
    "fitness",
    "rule_birth",
    "rule_survival",
    "n_tracks",
    "n_candidates",
    "max_age",
    "mean_age",
    "mean_boundedness",
    "mean_internal_variation",
    "mean_area",
    "final_active_fraction",
    "aborted",
    "abort_reason",
    "sim_time_seconds",
)

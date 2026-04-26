"""M5 — per-candidate intervention experiments.

For each top-K observer-candidate from a 4D run, this module runs paired
forward rollouts under each of four intervention types and produces a
**time-resolved divergence trajectory** (not just an aggregate scalar
like the M2 ``causality_score`` returns).

The four intervention types match the M2 / M5 spec:

  * ``internal_flip``    -- flip cells in 4D fibers under the candidate's
                            interior footprint
  * ``boundary_flip``    -- flip cells under the candidate's boundary
  * ``environment_flip`` -- flip cells under the candidate's environment shell
  * ``hidden_shuffle``   -- permute z,w values inside the interior footprint

For each intervention we measure:

  * ``full_grid_l1[t]``       -- mean per-cell L1 between projected
                                 frames in the unperturbed vs intervened
                                 rollouts at step t
  * ``candidate_footprint_l1[t]`` -- same but masked to the candidate's
                                     interior footprint
  * ``candidate_active_orig[t]``       -- active cells in the footprint, unperturbed
  * ``candidate_active_intervened[t]`` -- active cells in the footprint, intervened
  * ``final_survival``         -- True if intervened candidate has any
                                  active cells in the footprint at end
  * ``final_area_ratio``       -- intervened/orig active count at last step
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from observer_worlds.metrics.causality_score import (
    apply_flip_intervention,
    apply_hidden_shuffle_intervention,
)
from observer_worlds.worlds import CA4D, BSRule, project


INTERVENTION_TYPES: tuple[str, ...] = (
    "internal_flip",
    "boundary_flip",
    "environment_flip",
    "hidden_shuffle",
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InterventionTrajectory:
    """Time-resolved divergence trajectory for one intervention at one snapshot."""

    intervention_type: str
    snapshot_t: int
    n_steps: int
    flip_fraction: float

    # Per-step divergence (length n_steps).
    full_grid_l1: list[float] = field(default_factory=list)
    candidate_footprint_l1: list[float] = field(default_factory=list)
    candidate_active_orig: list[int] = field(default_factory=list)
    candidate_active_intervened: list[int] = field(default_factory=list)

    # End-of-rollout summary.
    final_survival: bool = False
    final_area_ratio: float = 0.0
    mean_full_grid_l1: float = 0.0
    mean_candidate_footprint_l1: float = 0.0
    auc_full_grid_l1: float = 0.0  # sum over time, equivalent to area under divergence curve


@dataclass
class CandidateInterventionReport:
    """All intervention trajectories for a single observer-candidate."""

    track_id: int
    track_age: int
    snapshot_t: int
    observer_score: float | None
    n_steps: int
    flip_fraction: float
    interior_size: int
    boundary_size: int
    env_size: int
    trajectories: dict[str, InterventionTrajectory] = field(default_factory=dict)

    # Convenience: per-intervention final divergence summaries (also in
    # trajectories[type].mean_full_grid_l1, exposed here as a flat dict
    # for easier CSV writing).
    intervention_summary: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Paired rollout
# ---------------------------------------------------------------------------


def _project(state: np.ndarray, theta: float) -> np.ndarray:
    return project(state, method="mean_threshold", theta=theta)


def _paired_rollout(
    state_orig: np.ndarray,
    state_intervened: np.ndarray,
    rule: BSRule,
    n_steps: int,
    *,
    backend: str,
    projection_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run two CA4D rollouts in lockstep; return (frames_orig, frames_int)
    of shape ``(n_steps, Nx, Ny)`` each."""
    Nx, Ny = state_orig.shape[0], state_orig.shape[1]
    ca_orig = CA4D(shape=state_orig.shape, rule=rule, backend=backend)
    ca_int = CA4D(shape=state_intervened.shape, rule=rule, backend=backend)
    ca_orig.state = state_orig.copy()
    ca_int.state = state_intervened.copy()
    fo = np.empty((n_steps, Nx, Ny), dtype=np.uint8)
    fi = np.empty((n_steps, Nx, Ny), dtype=np.uint8)
    for t in range(n_steps):
        ca_orig.step()
        ca_int.step()
        fo[t] = _project(ca_orig.state, projection_theta)
        fi[t] = _project(ca_int.state, projection_theta)
    return fo, fi


def _summarize_trajectory(
    traj: InterventionTrajectory,
    interior_mask_2d: np.ndarray,
    frames_orig: np.ndarray,
    frames_int: np.ndarray,
) -> None:
    """Populate per-step lists + end-of-rollout summary fields. Mutates ``traj``."""
    n_steps = frames_orig.shape[0]
    grid_cells = float(frames_orig.shape[1] * frames_orig.shape[2])
    interior_size = int(interior_mask_2d.sum())

    full_l1 = np.empty(n_steps, dtype=np.float64)
    cand_l1 = np.empty(n_steps, dtype=np.float64)
    active_orig = np.empty(n_steps, dtype=np.int64)
    active_int = np.empty(n_steps, dtype=np.int64)

    interior_bool = interior_mask_2d.astype(bool)
    interior_safe = max(interior_size, 1)

    for t in range(n_steps):
        diff = np.abs(
            frames_orig[t].astype(np.int16) - frames_int[t].astype(np.int16)
        )
        full_l1[t] = float(diff.sum() / grid_cells)
        cand_l1[t] = float(diff[interior_bool].sum() / interior_safe)
        active_orig[t] = int((frames_orig[t].astype(bool) & interior_bool).sum())
        active_int[t] = int((frames_int[t].astype(bool) & interior_bool).sum())

    traj.full_grid_l1 = full_l1.tolist()
    traj.candidate_footprint_l1 = cand_l1.tolist()
    traj.candidate_active_orig = active_orig.tolist()
    traj.candidate_active_intervened = active_int.tolist()
    traj.mean_full_grid_l1 = float(full_l1.mean())
    traj.mean_candidate_footprint_l1 = float(cand_l1.mean())
    traj.auc_full_grid_l1 = float(full_l1.sum())
    traj.final_survival = bool(active_int[-1] > 0)
    if active_orig[-1] > 0:
        traj.final_area_ratio = float(active_int[-1] / max(active_orig[-1], 1))
    else:
        # Original candidate also gone -- ratio is 1 if both zero, else 0.
        traj.final_area_ratio = 1.0 if active_int[-1] == 0 else 0.0


# ---------------------------------------------------------------------------
# Per-intervention runner
# ---------------------------------------------------------------------------


def _apply_intervention(
    snapshot_4d: np.ndarray,
    intervention_type: str,
    *,
    interior_mask_2d: np.ndarray,
    boundary_mask_2d: np.ndarray,
    env_mask_2d: np.ndarray,
    flip_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if intervention_type == "internal_flip":
        return apply_flip_intervention(
            snapshot_4d, interior_mask_2d, flip_fraction, rng
        )
    if intervention_type == "boundary_flip":
        return apply_flip_intervention(
            snapshot_4d, boundary_mask_2d, flip_fraction, rng
        )
    if intervention_type == "environment_flip":
        return apply_flip_intervention(
            snapshot_4d, env_mask_2d, flip_fraction, rng
        )
    if intervention_type == "hidden_shuffle":
        return apply_hidden_shuffle_intervention(
            snapshot_4d, interior_mask_2d, rng
        )
    raise ValueError(f"unknown intervention type {intervention_type!r}")


# ---------------------------------------------------------------------------
# Top-level: one candidate, all interventions
# ---------------------------------------------------------------------------


def run_candidate_interventions(
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask_2d: np.ndarray,
    boundary_mask_2d: np.ndarray,
    env_mask_2d: np.ndarray,
    *,
    track_id: int,
    track_age: int,
    snapshot_t: int,
    observer_score: float | None = None,
    n_steps: int = 20,
    flip_fraction: float = 0.5,
    backend: str = "numpy",
    seed: int = 0,
    intervention_types: Iterable[str] = INTERVENTION_TYPES,
    projection_theta: float = 0.5,
) -> CandidateInterventionReport:
    """Run all intervention rollouts for one candidate at one snapshot.

    Returns a :class:`CandidateInterventionReport` containing per-step
    divergence trajectories for each intervention type, plus survival /
    area-ratio summaries.
    """
    report = CandidateInterventionReport(
        track_id=track_id, track_age=track_age, snapshot_t=snapshot_t,
        observer_score=observer_score, n_steps=n_steps,
        flip_fraction=flip_fraction,
        interior_size=int(interior_mask_2d.sum()),
        boundary_size=int(boundary_mask_2d.sum()),
        env_size=int(env_mask_2d.sum()),
    )
    if (interior_mask_2d.sum() == 0
            or boundary_mask_2d.sum() == 0
            or env_mask_2d.sum() == 0):
        # Cannot run interventions on a degenerate candidate; return empty
        # report (caller can detect via interior_size == 0).
        return report

    # Shared unperturbed rollout: we evaluate it ONCE and reuse across
    # all four interventions.
    ca_orig = CA4D(shape=snapshot_4d.shape, rule=rule, backend=backend)
    ca_orig.state = snapshot_4d.copy()
    Nx, Ny = snapshot_4d.shape[0], snapshot_4d.shape[1]
    frames_orig = np.empty((n_steps, Nx, Ny), dtype=np.uint8)
    for t in range(n_steps):
        ca_orig.step()
        frames_orig[t] = _project(ca_orig.state, projection_theta)

    # Independent RNG per intervention (matches causality_score.py for parity).
    parent_rng = np.random.default_rng(seed)
    sub_seeds = parent_rng.integers(0, 2**63 - 1, size=len(list(intervention_types)))
    interv_list = list(intervention_types)
    for i, kind in enumerate(interv_list):
        rng_kind = np.random.default_rng(int(sub_seeds[i]))
        intervened = _apply_intervention(
            snapshot_4d, kind,
            interior_mask_2d=interior_mask_2d,
            boundary_mask_2d=boundary_mask_2d,
            env_mask_2d=env_mask_2d,
            flip_fraction=flip_fraction,
            rng=rng_kind,
        )
        ca_int = CA4D(shape=snapshot_4d.shape, rule=rule, backend=backend)
        ca_int.state = intervened
        frames_int = np.empty((n_steps, Nx, Ny), dtype=np.uint8)
        for t in range(n_steps):
            ca_int.step()
            frames_int[t] = _project(ca_int.state, projection_theta)

        traj = InterventionTrajectory(
            intervention_type=kind, snapshot_t=snapshot_t,
            n_steps=n_steps, flip_fraction=flip_fraction,
        )
        _summarize_trajectory(traj, interior_mask_2d, frames_orig, frames_int)
        report.trajectories[kind] = traj
        report.intervention_summary[kind] = {
            "mean_full_grid_l1": traj.mean_full_grid_l1,
            "mean_candidate_footprint_l1": traj.mean_candidate_footprint_l1,
            "auc_full_grid_l1": traj.auc_full_grid_l1,
            "final_survival": float(traj.final_survival),
            "final_area_ratio": traj.final_area_ratio,
        }
    return report


# ---------------------------------------------------------------------------
# Helpers for cross-candidate aggregation (used by plots + summary)
# ---------------------------------------------------------------------------


def aggregate_intervention_summaries(
    reports: list[CandidateInterventionReport],
) -> dict[str, dict[str, float]]:
    """Compute per-intervention-type aggregates across many candidates.

    Returns a dict ``{intervention_type: {metric: mean_across_candidates}}``.
    Missing entries (degenerate candidates) are skipped.
    """
    from collections import defaultdict
    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in reports:
        for kind, summary in r.intervention_summary.items():
            for metric, val in summary.items():
                by_type[kind][metric].append(val)
    out: dict[str, dict[str, float]] = {}
    for kind, metrics in by_type.items():
        out[kind] = {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}
    return out

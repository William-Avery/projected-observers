"""Workhorse module for Follow-up Topic 3 — agent-task environments.

Implements three minimal **functional** task probes. These are not
claims about agency, intent, or consciousness; they are observable
behavioural metrics computed on candidate snapshots.

* **repair**  — knock out a candidate's hidden support; measure how
  much projected activity returns to the candidate region over the
  rollout. ``repair_score`` ∈ ``[0, ~1]`` is the recovered fraction
  relative to the control (unperturbed) rollout.

* **foraging** — passive resource region placed at a fixed offset from
  the candidate's centroid. ``resource_contact_score`` is the mean
  projected activity inside the resource region across horizons;
  ``movement_toward_resource`` is the (positive) change in distance
  from the candidate's centroid to the resource centre normalised by
  the initial distance. We do **not** couple the resource into the
  CA update — that would be a full game engine. Smoke-only signal.

* **memory** — apply two distinct transient hidden cues at the peak
  frame, one per pair condition; run forward in both conditions;
  measure projected-pattern divergence in the candidate region at
  each horizon. ``cue_memory_score`` is the mean divergence across
  horizons. High = the cue identity persisted in the candidate's
  later projection (memory-like). Note: this is a contrastive,
  cue-specific HCE; it differs from generic HCE because the two cues
  are deterministic and compared against each other rather than
  against an unperturbed control.

Stage 4 also computes a smoke-quality ``hce`` per candidate inline so
the regression module can correlate task_score with HCE without
re-routing through the Topic-1 pipeline; production work in Stage 5
will join with M8 / M8G classifier results instead of recomputing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from observer_worlds.experiments._followup_identity_swap import (
    CandidateInCell, discover_candidates_for_cell,
)
from observer_worlds.experiments._followup_projection import (
    _rollout_perturbed,
)
from observer_worlds.projection import default_suite


# ---------------------------------------------------------------------------
# Per-trial result
# ---------------------------------------------------------------------------


@dataclass
class TaskTrial:
    trial_id: int
    rule_id: str
    rule_source: str
    seed: int
    candidate_id: int
    track_id: int
    task_name: str
    horizon: int
    projection_name: str
    survived: bool
    survival_time: int
    hce: float | None
    observer_score: float | None
    repair_score: float | None
    resource_contact_score: float | None
    movement_toward_resource: float | None
    cue_memory_score: float | None
    task_score: float | None
    hidden_intervention_task_delta: float | None
    visible_intervention_task_delta: float | None
    mechanism_class: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project(name: str, X: np.ndarray) -> np.ndarray:
    return default_suite().project(name, X)


def _binarize(arr: np.ndarray) -> np.ndarray:
    """Reduce any projection's output to a 2D uint8 0/1 frame for
    survival / overlap measurements."""
    if arr.ndim == 3:  # multi-channel
        arr = arr.mean(axis=-1)
    if arr.dtype.kind in "iu":
        return (arr > 0).astype(np.uint8)
    thr = float(np.median(arr))
    return (arr > thr).astype(np.uint8)


def _candidate_mask_at_h(state_4d: np.ndarray, projection_name: str
                          ) -> np.ndarray:
    return _binarize(_project(projection_name, state_4d))


def _local_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=-1)
    return float(diff[mask].sum() / float(mask.sum()))


# ---------------------------------------------------------------------------
# Inline HCE estimate (smoke; uses simple random-flip perturbation)
# ---------------------------------------------------------------------------


def _estimate_hce(
    state_at_peak: np.ndarray, peak_mask: np.ndarray,
    rule_bs, projection_name: str, horizons: Sequence[int], backend: str,
    rng: np.random.Generator,
    n_flips: int = 8,
) -> float:
    """Mean over horizons of projected-pattern L1 in candidate mask
    after a small random hidden perturbation. Used only as a per-
    candidate predictor for the regression module; production work
    should join with the Topic-1 / M8 measurement instead."""
    Nx, Ny, Nz, Nw = state_at_peak.shape
    perturbed = state_at_peak.copy()
    rows, cols = np.where(peak_mask)
    if rows.size == 0:
        return 0.0
    n = min(n_flips, rows.size * Nz * Nw)
    pick = rng.integers(0, rows.size, size=n)
    zs = rng.integers(0, Nz, size=n)
    ws = rng.integers(0, Nw, size=n)
    for i in range(n):
        x, y = int(rows[pick[i]]), int(cols[pick[i]])
        perturbed[x, y, int(zs[i]), int(ws[i])] ^= 1
    deltas = []
    local = peak_mask.astype(bool)
    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        ctrl = _rollout_perturbed(rule_bs, state_at_peak, h, backend=backend)
        pert = _rollout_perturbed(rule_bs, perturbed, h, backend=backend)
        deltas.append(_local_l1(
            _project(projection_name, pert),
            _project(projection_name, ctrl),
            local,
        ))
    return float(np.mean(deltas)) if deltas else 0.0


# ---------------------------------------------------------------------------
# Task A: repair
# ---------------------------------------------------------------------------


def evaluate_repair(
    *, cic: CandidateInCell, rule_bs, projection_name: str,
    horizons: Sequence[int], backend: str,
) -> list[TaskTrial]:
    """Knock out the candidate's 4D support (set to zero in the (z, w)
    fibre at every (x, y) in the mask) and measure projected
    re-activity inside the original mask across horizons."""
    state = cic.state_at_peak
    mask = cic.cand.peak_mask.astype(bool)

    perturbed = state.copy()
    perturbed[mask, :, :] = 0  # zero hidden support inside the mask

    out = []
    survival_time = 0
    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        ctrl_h = _rollout_perturbed(rule_bs, state, h, backend=backend)
        pert_h = _rollout_perturbed(rule_bs, perturbed, h, backend=backend)
        ctrl_proj = _binarize(_project(projection_name, ctrl_h))
        pert_proj = _binarize(_project(projection_name, pert_h))
        # Recovered fraction: cells active in the candidate region after
        # perturbation, normalised by control activity in same region.
        ctrl_active_in_mask = float(ctrl_proj[mask].sum())
        pert_active_in_mask = float(pert_proj[mask].sum())
        if ctrl_active_in_mask < 1.0:
            repair = 0.0  # nothing to recover toward
            survived = False
        else:
            repair = float(min(1.5, pert_active_in_mask / ctrl_active_in_mask))
            survived = pert_active_in_mask >= 0.25 * ctrl_active_in_mask
        if survived:
            survival_time = h
        out.append(TaskTrial(
            trial_id=-1, rule_id=cic.rule_id, rule_source=cic.rule_source,
            seed=int(cic.seed), candidate_id=cic.cand.candidate_id,
            track_id=cic.cand.track_id, task_name="repair", horizon=h,
            projection_name=projection_name,
            survived=survived, survival_time=survival_time,
            hce=None, observer_score=None,
            repair_score=repair,
            resource_contact_score=None, movement_toward_resource=None,
            cue_memory_score=None,
            task_score=repair,
            hidden_intervention_task_delta=None,
            visible_intervention_task_delta=None,
            mechanism_class=None,
        ))
    return out


# ---------------------------------------------------------------------------
# Task B: foraging / resource contact
# ---------------------------------------------------------------------------


def _resource_region(centroid: tuple[float, float],
                      grid_shape_2d: tuple[int, int],
                      offset: tuple[int, int] = (4, 4),
                      radius: int = 2) -> np.ndarray:
    """Disc of radius ``radius`` centred at ``centroid + offset`` within
    the 2D grid; returns a uint8 mask."""
    Nx, Ny = grid_shape_2d
    cx = int(round(centroid[0] + offset[0])) % Nx
    cy = int(round(centroid[1] + offset[1])) % Ny
    yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx))
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2).astype(np.uint8)


def evaluate_foraging(
    *, cic: CandidateInCell, rule_bs, projection_name: str,
    horizons: Sequence[int], backend: str,
) -> list[TaskTrial]:
    """Place a passive disc resource at fixed offset from the
    candidate's centroid; measure projected activity inside that disc
    over horizons. Resource is non-coupling (does not affect CA
    updates) — this is a smoke-level signal of "drift / proximity",
    not a foraging claim."""
    state = cic.state_at_peak
    mask = cic.cand.peak_mask.astype(bool)
    Nx, Ny = state.shape[:2]
    if not mask.any():
        return []
    rows, cols = np.where(mask)
    cx = float(rows.mean()); cy = float(cols.mean())
    resource = _resource_region((cx, cy), (Nx, Ny)).astype(bool)

    out = []
    init_distance = None
    survival_time = 0
    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        future = _rollout_perturbed(rule_bs, state, h, backend=backend)
        future_proj = _binarize(_project(projection_name, future))
        # Resource contact: mean projected activity inside resource disc.
        contact = float(future_proj[resource].mean()) if resource.any() else 0.0
        # Movement toward resource: distance between current centroid
        # (largest active blob) and resource centre.
        active_xy = np.argwhere(future_proj > 0)
        if active_xy.size:
            now_cx = float(active_xy[:, 0].mean())
            now_cy = float(active_xy[:, 1].mean())
            res_xy = np.argwhere(resource)
            res_cx = float(res_xy[:, 0].mean())
            res_cy = float(res_xy[:, 1].mean())
            dist = float(np.hypot(now_cx - res_cx, now_cy - res_cy))
        else:
            dist = float(np.hypot(Nx, Ny))  # max possible
        if init_distance is None:
            res_xy = np.argwhere(resource)
            res_cx = float(res_xy[:, 0].mean())
            res_cy = float(res_xy[:, 1].mean())
            init_distance = float(np.hypot(cx - res_cx, cy - res_cy))
        if init_distance > 1e-6:
            movement = float(
                (init_distance - dist) / init_distance,
            )
        else:
            movement = 0.0
        # Survival: any active cells anywhere?
        survived = bool(future_proj.any())
        if survived:
            survival_time = h
        # Composite task_score: foraging combines contact + movement.
        task_score = float(contact + max(0.0, movement)) / 2.0
        out.append(TaskTrial(
            trial_id=-1, rule_id=cic.rule_id, rule_source=cic.rule_source,
            seed=int(cic.seed), candidate_id=cic.cand.candidate_id,
            track_id=cic.cand.track_id, task_name="foraging", horizon=h,
            projection_name=projection_name,
            survived=survived, survival_time=survival_time,
            hce=None, observer_score=None,
            repair_score=None,
            resource_contact_score=contact,
            movement_toward_resource=movement,
            cue_memory_score=None,
            task_score=task_score,
            hidden_intervention_task_delta=None,
            visible_intervention_task_delta=None,
            mechanism_class=None,
        ))
    return out


# ---------------------------------------------------------------------------
# Task C: memory / delayed cue
# ---------------------------------------------------------------------------


def _make_cue(
    state_4d: np.ndarray, mask_2d: np.ndarray, *,
    pattern: str, n_cells: int, rng: np.random.Generator,
) -> np.ndarray:
    """Apply one of two distinct deterministic cue patterns to the 4D
    state at cells inside ``mask_2d``. Returns a perturbed copy."""
    Nx, Ny, Nz, Nw = state_4d.shape
    out = state_4d.copy()
    rows, cols = np.where(mask_2d)
    if rows.size == 0:
        return out
    n = min(int(n_cells), int(rows.size))
    # Use deterministic seeded picks that depend on `pattern` so cue A
    # and cue B are reproducible and distinct.
    seed = (rng.integers(0, 2**31) ^ (0xA1 if pattern == "A" else 0xB2))
    local_rng = np.random.default_rng(int(seed))
    picks = local_rng.choice(rows.size, size=n, replace=False)
    if pattern == "A":
        zs = np.zeros(n, dtype=int); ws = np.zeros(n, dtype=int)
    else:
        zs = np.full(n, Nz - 1, dtype=int); ws = np.full(n, Nw - 1, dtype=int)
    for i in range(n):
        x, y = int(rows[picks[i]]), int(cols[picks[i]])
        out[x, y, int(zs[i]), int(ws[i])] ^= 1
    return out


def evaluate_memory(
    *, cic: CandidateInCell, rule_bs, projection_name: str,
    horizons: Sequence[int], backend: str,
    rng: np.random.Generator,
) -> list[TaskTrial]:
    """Apply two distinct cues (A and B) at the peak frame and run
    forward. cue_memory_score = projected-pattern divergence between
    cue-A and cue-B futures inside the candidate region. High score
    means the system retained cue identity over the rollout."""
    state = cic.state_at_peak
    mask = cic.cand.peak_mask.astype(bool)
    if not mask.any():
        return []
    cue_a = _make_cue(state, mask, pattern="A", n_cells=4, rng=rng)
    cue_b = _make_cue(state, mask, pattern="B", n_cells=4, rng=rng)

    out = []
    survival_time = 0
    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        fut_a = _rollout_perturbed(rule_bs, cue_a, h, backend=backend)
        fut_b = _rollout_perturbed(rule_bs, cue_b, h, backend=backend)
        proj_a = _project(projection_name, fut_a)
        proj_b = _project(projection_name, fut_b)
        score = _local_l1(proj_a, proj_b, mask)
        # Survival: candidate region has any activity in either future.
        proj_a_b = _binarize(proj_a)
        proj_b_b = _binarize(proj_b)
        survived = bool(proj_a_b[mask].sum() > 0 or proj_b_b[mask].sum() > 0)
        if survived:
            survival_time = h
        out.append(TaskTrial(
            trial_id=-1, rule_id=cic.rule_id, rule_source=cic.rule_source,
            seed=int(cic.seed), candidate_id=cic.cand.candidate_id,
            track_id=cic.cand.track_id, task_name="memory", horizon=h,
            projection_name=projection_name,
            survived=survived, survival_time=survival_time,
            hce=None, observer_score=None,
            repair_score=None,
            resource_contact_score=None, movement_toward_resource=None,
            cue_memory_score=score,
            task_score=score,
            hidden_intervention_task_delta=None,
            visible_intervention_task_delta=None,
            mechanism_class=None,
        ))
    return out


# ---------------------------------------------------------------------------
# Per-candidate orchestration
# ---------------------------------------------------------------------------


TASK_EVALUATORS = {
    "repair": evaluate_repair,
    "foraging": evaluate_foraging,
    "memory": evaluate_memory,
}


def run_tasks_for_candidate(
    *, cic: CandidateInCell, rule_bs, projection_name: str,
    horizons: Sequence[int], backend: str,
    tasks: Sequence[str],
    rng: np.random.Generator,
) -> list[TaskTrial]:
    """Compute HCE / observer-score proxy once per candidate; then run
    each requested task and stamp the per-candidate predictors onto
    every trial."""
    hce = _estimate_hce(
        cic.state_at_peak, cic.cand.peak_mask, rule_bs,
        projection_name=projection_name, horizons=horizons,
        backend=backend, rng=rng,
    )
    observer_proxy = float(cic.cand.lifetime)
    out: list[TaskTrial] = []
    for task in tasks:
        evaluator = TASK_EVALUATORS.get(task)
        if evaluator is None:
            continue
        if task == "memory":
            trials = evaluator(
                cic=cic, rule_bs=rule_bs, projection_name=projection_name,
                horizons=horizons, backend=backend, rng=rng,
            )
        else:
            trials = evaluator(
                cic=cic, rule_bs=rule_bs, projection_name=projection_name,
                horizons=horizons, backend=backend,
            )
        for t in trials:
            t.hce = hce
            t.observer_score = observer_proxy
            out.append(t)
    return out

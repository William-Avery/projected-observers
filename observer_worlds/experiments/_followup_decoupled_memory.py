"""Stage 5E2 decoupled-memory audit workhorse.

Stage 5E found Pearson(HCE, memory_score) up to +0.75. The cue-memory
score and the inline HCE estimate were both computed by perturbing
hidden cells in the candidate region — methodologically aligned, so
the correlation could be partially driven by shared construction.

This module implements four variants where the **cue region** and the
**HCE perturbation region** are deliberately disjoint:

* ``cue_far_boundary``      — cue at candidate's boundary cells; HCE
                               perturbation in candidate's interior.
* ``cue_environment_shell`` — cue in the dilation shell *outside* the
                               candidate; HCE perturbation in the
                               candidate mask.
* ``cue_opposite_side``     — candidate split by centroid; cue on the
                               left half, HCE perturbation on the
                               right half.
* ``cue_random_remote``     — cue at a random rectangular patch
                               nearby (not overlapping the candidate);
                               HCE perturbation in the candidate mask.

Per (candidate × variant × horizon) the evaluator produces:

* ``memory_score`` — projected divergence between cue-A and cue-B
  futures, measured **inside the candidate region**.
* ``hce`` — projected divergence between unperturbed and hidden-
  perturbed futures, measured **inside the candidate region**.
* audit fields: ``cue_region``, ``hce_region``, ``overlap_fraction``,
  ``cue_to_hce_distance``, ``coupled`` (True iff overlap exceeds the
  decoupling threshold).

The goal is to test whether the Stage 5E HCE-memory correlation
survives when the two measurements operate on disjoint hidden-state
regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.ndimage as ndi

from observer_worlds.experiments._followup_agent_tasks import (
    _local_l1, _make_cue, _project,
)
from observer_worlds.experiments._followup_identity_swap import (
    CandidateInCell,
)
from observer_worlds.experiments._followup_projection import (
    _rollout_perturbed,
)


SUPPORTED_VARIANTS = (
    "cue_far_boundary",
    "cue_environment_shell",
    "cue_opposite_side",
    "cue_random_remote",
)


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RegionPair:
    cue: np.ndarray            # 2D bool mask
    hce: np.ndarray            # 2D bool mask


def _candidate_regions(cic: CandidateInCell, *, env_dilation: int = 2) -> dict:
    mask = cic.cand.peak_mask.astype(bool)
    interior = cic.cand.peak_interior.astype(bool)
    if not interior.any():
        interior = mask
    boundary = mask & ~interior
    env_shell = ndi.binary_dilation(mask, iterations=int(env_dilation)) & ~mask
    rows, cols = np.where(mask)
    if rows.size > 0:
        cy = float(cols.mean())
        col_grid = np.broadcast_to(np.arange(mask.shape[1]),
                                    mask.shape).astype(np.float64)
        left = mask & (col_grid < cy)
        right = mask & ~left
    else:
        left = right = np.zeros_like(mask)
    return {
        "mask": mask, "interior": interior, "boundary": boundary,
        "env_shell": env_shell, "left": left, "right": right,
    }


def _random_remote_patch(
    cic: CandidateInCell, *, rng: np.random.Generator,
    patch_size: int = 4, max_attempts: int = 10,
) -> np.ndarray:
    """Random rectangular patch nearby but not overlapping the candidate."""
    mask = cic.cand.peak_mask.astype(bool)
    Nx, Ny = mask.shape
    rmin, cmin, rmax, cmax = cic.cand.peak_bbox
    h = rmax - rmin + 1; w = cmax - cmin + 1
    out = np.zeros_like(mask)
    for _ in range(int(max_attempts)):
        # Place patch centred at a random offset that puts it adjacent
        # to but not over the candidate bbox.
        side = rng.choice([0, 1, 2, 3])  # top, right, bottom, left
        if side == 0:
            r0 = max(0, rmin - patch_size - 1)
            c0 = max(0, min(Ny - patch_size, cmin + rng.integers(0, max(1, w))))
        elif side == 1:
            r0 = max(0, min(Nx - patch_size, rmin + rng.integers(0, max(1, h))))
            c0 = min(Ny - patch_size, cmax + 1)
        elif side == 2:
            r0 = min(Nx - patch_size, rmax + 1)
            c0 = max(0, min(Ny - patch_size, cmin + rng.integers(0, max(1, w))))
        else:
            r0 = max(0, min(Nx - patch_size, rmin + rng.integers(0, max(1, h))))
            c0 = max(0, cmin - patch_size - 1)
        candidate_patch = np.zeros_like(mask)
        candidate_patch[r0:r0 + patch_size, c0:c0 + patch_size] = True
        # Reject if it overlaps the candidate.
        if (candidate_patch & mask).any():
            continue
        return candidate_patch
    return out  # all-zero if attempts exhausted


def _region_pair_for_variant(
    cic: CandidateInCell, variant: str, *,
    rng: np.random.Generator,
) -> _RegionPair:
    """Pick the (cue_region, hce_region) pair for ``variant`` so the
    two regions are disjoint by construction."""
    r = _candidate_regions(cic)
    if variant == "cue_far_boundary":
        # cue at boundary; HCE in interior. Disjoint by definition
        # (boundary = mask AND NOT interior).
        return _RegionPair(cue=r["boundary"], hce=r["interior"])
    if variant == "cue_environment_shell":
        # cue OUTSIDE candidate; HCE inside candidate.
        return _RegionPair(cue=r["env_shell"], hce=r["mask"])
    if variant == "cue_opposite_side":
        # cue on left half; HCE on right half.
        return _RegionPair(cue=r["left"], hce=r["right"])
    if variant == "cue_random_remote":
        cue = _random_remote_patch(cic, rng=rng)
        return _RegionPair(cue=cue, hce=r["mask"])
    raise ValueError(f"unknown variant {variant!r}")


def _centroid(m: np.ndarray) -> tuple[float, float]:
    rs, cs = np.where(m)
    if rs.size == 0:
        return (0.0, 0.0)
    return float(rs.mean()), float(cs.mean())


def _overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any():
        return 0.0
    return float((a & b).sum() / a.sum())


# ---------------------------------------------------------------------------
# Per-trial result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DecoupledMemoryTrial:
    trial_id: int
    rule_id: str
    rule_source: str
    seed: int
    candidate_id: int
    track_id: int
    variant: str
    horizon: int
    projection_name: str
    # Audit fields
    cue_region_size: int
    hce_region_size: int
    overlap_fraction: float
    cue_to_hce_distance: float
    coupled: bool
    # Metrics
    memory_score: float | None
    hce: float | None
    observer_score: float | None
    # Per-candidate (constant across variants).
    candidate_lifetime: int


# ---------------------------------------------------------------------------
# Per-(candidate, variant) evaluator
# ---------------------------------------------------------------------------


def _measure_memory_in_region(
    state_at_peak: np.ndarray, cue_region: np.ndarray,
    candidate_region: np.ndarray,
    rule_bs, projection_name: str, horizons: Sequence[int],
    backend: str, rng: np.random.Generator,
    n_cells: int = 4,
) -> dict[int, float]:
    """Apply two distinct cues at ``cue_region``; run forward; return
    per-horizon projected divergence between cue-A and cue-B futures
    measured *inside* ``candidate_region``."""
    if not cue_region.any():
        return {int(h): 0.0 for h in horizons}
    cue_a = _make_cue(state_at_peak, cue_region.astype(np.uint8),
                       pattern="A", n_cells=n_cells, rng=rng)
    cue_b = _make_cue(state_at_peak, cue_region.astype(np.uint8),
                       pattern="B", n_cells=n_cells, rng=rng)
    out: dict[int, float] = {}
    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        fut_a = _rollout_perturbed(rule_bs, cue_a, h, backend=backend)
        fut_b = _rollout_perturbed(rule_bs, cue_b, h, backend=backend)
        proj_a = _project(projection_name, fut_a)
        proj_b = _project(projection_name, fut_b)
        out[h] = _local_l1(proj_a, proj_b, candidate_region)
    return out


def _measure_hce_in_region(
    state_at_peak: np.ndarray, hce_region: np.ndarray,
    candidate_region: np.ndarray,
    rule_bs, projection_name: str, horizons: Sequence[int],
    backend: str, rng: np.random.Generator,
    n_flips: int = 8,
) -> dict[int, float]:
    """Random hidden-bit flip in ``hce_region``; per-horizon projected
    divergence between unperturbed and perturbed futures, measured
    inside ``candidate_region``. Uses inline-style HCE (no projection
    invariance enforcement) to keep the audit comparable to Stage 5E's
    inline HCE estimator."""
    Nx, Ny, Nz, Nw = state_at_peak.shape
    perturbed = state_at_peak.copy()
    rows, cols = np.where(hce_region)
    if rows.size == 0:
        return {int(h): 0.0 for h in horizons}
    n = min(int(n_flips), rows.size * Nz * Nw)
    pick = rng.integers(0, rows.size, size=n)
    zs = rng.integers(0, Nz, size=n)
    ws = rng.integers(0, Nw, size=n)
    for i in range(n):
        x, y = int(rows[pick[i]]), int(cols[pick[i]])
        perturbed[x, y, int(zs[i]), int(ws[i])] ^= 1
    out: dict[int, float] = {}
    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        ctrl = _rollout_perturbed(rule_bs, state_at_peak, h, backend=backend)
        pert = _rollout_perturbed(rule_bs, perturbed, h, backend=backend)
        out[h] = _local_l1(
            _project(projection_name, pert),
            _project(projection_name, ctrl),
            candidate_region,
        )
    return out


def evaluate_decoupled_memory_for_candidate(
    *, cic: CandidateInCell, rule_bs, projection_name: str,
    horizons: Sequence[int], backend: str,
    variants: Sequence[str],
    rng: np.random.Generator,
    overlap_threshold: float = 0.05,
) -> list[DecoupledMemoryTrial]:
    """For one candidate, run each variant's decoupled memory + HCE
    measurement and emit one trial per (variant, horizon)."""
    out: list[DecoupledMemoryTrial] = []
    candidate_region = cic.cand.peak_mask.astype(bool)
    if not candidate_region.any():
        return out
    observer_proxy = float(cic.cand.lifetime)
    for variant in variants:
        if variant not in SUPPORTED_VARIANTS:
            continue
        regions = _region_pair_for_variant(cic, variant, rng=rng)
        ovl = _overlap_fraction(regions.cue, regions.hce)
        coupled = ovl > float(overlap_threshold)
        cue_c = _centroid(regions.cue); hce_c = _centroid(regions.hce)
        dist = float(np.hypot(cue_c[0] - hce_c[0], cue_c[1] - hce_c[1]))
        # Empty cue region (e.g. all-zero candidate or no env shell):
        # still record as a trial with metric None and flag.
        if int(regions.cue.sum()) == 0 or int(regions.hce.sum()) == 0:
            for h in horizons:
                out.append(DecoupledMemoryTrial(
                    trial_id=-1, rule_id=cic.rule_id,
                    rule_source=cic.rule_source, seed=int(cic.seed),
                    candidate_id=cic.cand.candidate_id,
                    track_id=cic.cand.track_id, variant=variant,
                    horizon=int(h), projection_name=projection_name,
                    cue_region_size=int(regions.cue.sum()),
                    hce_region_size=int(regions.hce.sum()),
                    overlap_fraction=float(ovl),
                    cue_to_hce_distance=dist,
                    coupled=bool(coupled),
                    memory_score=None, hce=None,
                    observer_score=observer_proxy,
                    candidate_lifetime=int(cic.cand.lifetime),
                ))
            continue
        memory_per_h = _measure_memory_in_region(
            cic.state_at_peak, regions.cue, candidate_region,
            rule_bs, projection_name, horizons, backend, rng,
        )
        hce_per_h = _measure_hce_in_region(
            cic.state_at_peak, regions.hce, candidate_region,
            rule_bs, projection_name, horizons, backend, rng,
        )
        for h in horizons:
            out.append(DecoupledMemoryTrial(
                trial_id=-1, rule_id=cic.rule_id,
                rule_source=cic.rule_source, seed=int(cic.seed),
                candidate_id=cic.cand.candidate_id,
                track_id=cic.cand.track_id, variant=variant,
                horizon=int(h), projection_name=projection_name,
                cue_region_size=int(regions.cue.sum()),
                hce_region_size=int(regions.hce.sum()),
                overlap_fraction=float(ovl),
                cue_to_hce_distance=dist,
                coupled=bool(coupled),
                memory_score=float(memory_per_h.get(int(h), 0.0)),
                hce=float(hce_per_h.get(int(h), 0.0)),
                observer_score=observer_proxy,
                candidate_lifetime=int(cic.cand.lifetime),
            ))
    return out

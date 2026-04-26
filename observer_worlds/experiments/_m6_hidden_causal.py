"""M6 — Hidden Causal Dependence.

Tests for an effect that *only* a 4D-projected world can have:
**downstream projected divergence under a perturbation that is
invisible in the 2D projection at t=0**.

The hidden_invisible perturbation permutes (z, w) values within each
(x, y) column in the candidate's interior. Because mean-threshold
projection depends only on the per-column count of active cells, this
permutation leaves ``project(state)`` byte-identical to the unperturbed
state at t=0. In a pure 2D system, *no* such perturbation exists —
there are no hidden cells to permute — so any 2D analogue would have
HCE ≡ 0 by construction.

Operational definitions:

  * ``HCE`` — Hidden Causal Effect. Mean projected-frame divergence
    (full-grid L1) at the **final** rollout step under
    ``hidden_invisible`` perturbation, averaged over replicates.
    Higher = more causal weight on hidden structure.
  * ``visible_match_count`` — control perturbation that flips the
    same number of 4D bits in the interior, but uniformly at random,
    so it (almost certainly) changes ``project(state)`` immediately.
    Provides a divergence ceiling at matched perturbation magnitude.
  * ``immediate_l1`` — divergence at step 1; should be ~0 for
    hidden_invisible (regression test) and >0 for visible.

The framework runs paired rollouts and aggregates divergence
trajectories across replicates and candidates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from observer_worlds.metrics.causality_score import (
    apply_flip_intervention,
    apply_hidden_shuffle_intervention,
)
from observer_worlds.worlds import CA4D, BSRule, project


PERTURBATION_TYPES: tuple[str, ...] = ("hidden_invisible", "visible_match_count")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HiddenCausalTrajectory:
    """Aggregated trajectory across replicates for one perturbation type."""

    perturbation_type: str
    n_steps: int
    n_replicates: int
    # Mean / std across replicates of the projected divergence per step.
    full_grid_l1_mean: list[float] = field(default_factory=list)
    full_grid_l1_std: list[float] = field(default_factory=list)
    candidate_footprint_l1_mean: list[float] = field(default_factory=list)
    candidate_footprint_l1_std: list[float] = field(default_factory=list)
    # Aggregate scalars.
    mean_immediate_l1: float = 0.0       # at step 1
    mean_final_l1: float = 0.0           # at last step
    mean_auc: float = 0.0
    # Number of bit-flips applied per replicate (matched between perturbations).
    mean_n_flips: float = 0.0
    # Per-replicate raw values (for further analysis).
    per_replicate_immediate: list[float] = field(default_factory=list)
    per_replicate_final: list[float] = field(default_factory=list)
    per_replicate_auc: list[float] = field(default_factory=list)


@dataclass
class HiddenCausalReport:
    """Result for one (candidate, snapshot) under M6.

    The HCE field is the headline scalar: mean projected divergence at
    final rollout step under hidden_invisible perturbation. In a pure 2D
    system this is identically 0; in 4D it can be positive iff hidden
    structure has causal weight on the projected future.
    """

    track_id: int
    snapshot_t: int
    track_age: int
    observer_score: float | None
    interior_size: int
    n_steps: int
    n_replicates: int
    flip_fraction_for_visible: float

    hidden_invisible: HiddenCausalTrajectory
    visible_match_count: HiddenCausalTrajectory

    # Headline scalar.
    HCE: float = 0.0
    # Visible-perturbation final divergence (control).
    visible_final_l1: float = 0.0
    # Ratio HCE / visible_final_l1; in [0, 1] generally. 1.0 means hidden
    # perturbation produces just as much downstream divergence as a
    # bit-matched visible one.
    hce_to_visible_ratio: float = 0.0
    # Sanity: immediate divergence under hidden_invisible. Should be ~0.
    hce_immediate_check: float = 0.0


# ---------------------------------------------------------------------------
# Paired rollout (project after each step)
# ---------------------------------------------------------------------------


def _project(state: np.ndarray, theta: float) -> np.ndarray:
    return project(state, method="mean_threshold", theta=theta)


def _rollout_to_frames(
    state: np.ndarray, rule: BSRule, n_steps: int, *,
    backend: str, projection_theta: float,
) -> np.ndarray:
    """Run a CA4D rollout from ``state`` for n_steps, return projected frames
    of shape (n_steps, Nx, Ny)."""
    Nx, Ny = state.shape[0], state.shape[1]
    ca = CA4D(shape=state.shape, rule=rule, backend=backend)
    ca.state = state.copy()
    out = np.empty((n_steps, Nx, Ny), dtype=np.uint8)
    for t in range(n_steps):
        ca.step()
        out[t] = _project(ca.state, projection_theta)
    return out


def _trajectory_l1(
    frames_orig: np.ndarray, frames_int: np.ndarray,
    interior_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step (full_grid_l1, candidate_footprint_l1)."""
    n_steps = frames_orig.shape[0]
    grid_cells = float(frames_orig.shape[1] * frames_orig.shape[2])
    interior_bool = interior_mask.astype(bool)
    interior_size = max(int(interior_bool.sum()), 1)
    full = np.empty(n_steps, dtype=np.float64)
    cand = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        diff = np.abs(
            frames_orig[t].astype(np.int16) - frames_int[t].astype(np.int16)
        )
        full[t] = float(diff.sum() / grid_cells)
        cand[t] = float(diff[interior_bool].sum() / interior_size)
    return full, cand


# ---------------------------------------------------------------------------
# Per-candidate experiment
# ---------------------------------------------------------------------------


def run_hidden_causal_experiment(
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask_2d: np.ndarray,
    *,
    track_id: int,
    track_age: int,
    snapshot_t: int,
    observer_score: float | None = None,
    n_steps: int = 20,
    n_replicates: int = 5,
    backend: str = "numpy",
    seed: int = 0,
    projection_theta: float = 0.5,
) -> HiddenCausalReport:
    """Run the M6 paired-perturbation experiment on one candidate.

    Per replicate r:
      1. Apply hidden_invisible perturbation (z,w shuffle inside interior).
         Verify project(perturbed) == project(snapshot) at t=0 (assertion).
         Run paired rollout; record divergence per step.
      2. Count bit-flips n_flips made by the shuffle.
      3. Build a visible_match_count perturbation that flips n_flips random
         bits inside the interior fibers. Run paired rollout; record.

    Aggregates trajectories across replicates and computes HCE.
    """
    if interior_mask_2d.sum() == 0:
        # Degenerate: HCE is 0 because there are no hidden cells to permute.
        empty = HiddenCausalTrajectory(
            perturbation_type="hidden_invisible",
            n_steps=n_steps, n_replicates=0,
        )
        empty2 = HiddenCausalTrajectory(
            perturbation_type="visible_match_count",
            n_steps=n_steps, n_replicates=0,
        )
        return HiddenCausalReport(
            track_id=track_id, snapshot_t=snapshot_t, track_age=track_age,
            observer_score=observer_score, interior_size=0,
            n_steps=n_steps, n_replicates=0,
            flip_fraction_for_visible=0.0,
            hidden_invisible=empty, visible_match_count=empty2,
        )

    # Unperturbed rollout (computed once, reused for all replicates).
    frames_orig = _rollout_to_frames(
        snapshot_4d, rule, n_steps,
        backend=backend, projection_theta=projection_theta,
    )

    # Sanity assertion: hidden_invisible must match projection at t=0.
    # We check it on the first replicate then trust the construction.
    parent_rng = np.random.default_rng(seed)

    hi_full = np.empty((n_replicates, n_steps), dtype=np.float64)
    hi_cand = np.empty((n_replicates, n_steps), dtype=np.float64)
    vis_full = np.empty((n_replicates, n_steps), dtype=np.float64)
    vis_cand = np.empty((n_replicates, n_steps), dtype=np.float64)
    n_flips_per_rep = np.empty(n_replicates, dtype=np.int64)
    proj0 = _project(snapshot_4d, projection_theta)

    interior_size = int(interior_mask_2d.sum())

    for r in range(n_replicates):
        sub_seeds = parent_rng.integers(0, 2**63 - 1, size=2)
        rng_hi = np.random.default_rng(int(sub_seeds[0]))
        rng_vis = np.random.default_rng(int(sub_seeds[1]))

        # ---- hidden_invisible
        state_hi = apply_hidden_shuffle_intervention(
            snapshot_4d, interior_mask_2d, rng_hi
        )
        # Sanity at t=0: projections must match.
        proj_hi_0 = _project(state_hi, projection_theta)
        assert np.array_equal(proj0, proj_hi_0), (
            "hidden_invisible perturbation did not preserve projection; "
            "this is a bug or a non-mean-threshold projection."
        )
        # Count actual bit-flips in interior fibers.
        diff_4d = (state_hi != snapshot_4d)
        # Restrict to the interior columns (broadcast 2D mask).
        interior_4d_mask = interior_mask_2d[:, :, None, None].astype(bool)
        n_flips = int((diff_4d & interior_4d_mask).sum())
        n_flips_per_rep[r] = n_flips

        frames_hi = _rollout_to_frames(
            state_hi, rule, n_steps,
            backend=backend, projection_theta=projection_theta,
        )
        full_hi, cand_hi = _trajectory_l1(frames_orig, frames_hi, interior_mask_2d)
        hi_full[r] = full_hi
        hi_cand[r] = cand_hi

        # ---- visible_match_count: flip exactly n_flips random cells in
        # the interior fibers. Total interior cells = interior_size *
        # Nz * Nw. Use flip_fraction = n_flips / total_interior_cells.
        Nz, Nw = snapshot_4d.shape[2], snapshot_4d.shape[3]
        total_interior_cells = interior_size * Nz * Nw
        flip_fraction = (
            float(n_flips) / total_interior_cells
            if total_interior_cells > 0 else 0.0
        )
        # apply_flip_intervention already does this exactly via
        # int(round(N * fraction)) cells.
        state_vis = apply_flip_intervention(
            snapshot_4d, interior_mask_2d, flip_fraction, rng_vis
        )
        frames_vis = _rollout_to_frames(
            state_vis, rule, n_steps,
            backend=backend, projection_theta=projection_theta,
        )
        full_vis, cand_vis = _trajectory_l1(frames_orig, frames_vis, interior_mask_2d)
        vis_full[r] = full_vis
        vis_cand[r] = cand_vis

    # Aggregate.
    hi_traj = _aggregate_trajectory(
        "hidden_invisible", n_steps, n_replicates,
        hi_full, hi_cand, n_flips_per_rep,
    )
    vis_traj = _aggregate_trajectory(
        "visible_match_count", n_steps, n_replicates,
        vis_full, vis_cand, n_flips_per_rep,
    )

    HCE = float(hi_traj.mean_final_l1)
    visible_final = float(vis_traj.mean_final_l1)
    ratio = HCE / visible_final if visible_final > 1e-12 else 0.0

    return HiddenCausalReport(
        track_id=track_id, snapshot_t=snapshot_t, track_age=track_age,
        observer_score=observer_score,
        interior_size=interior_size,
        n_steps=n_steps, n_replicates=n_replicates,
        flip_fraction_for_visible=float(np.mean(n_flips_per_rep)) / max(
            interior_size * snapshot_4d.shape[2] * snapshot_4d.shape[3], 1
        ),
        hidden_invisible=hi_traj,
        visible_match_count=vis_traj,
        HCE=HCE,
        visible_final_l1=visible_final,
        hce_to_visible_ratio=float(ratio),
        hce_immediate_check=float(hi_traj.mean_immediate_l1),
    )


def _aggregate_trajectory(
    name: str, n_steps: int, n_replicates: int,
    full: np.ndarray, cand: np.ndarray, n_flips: np.ndarray,
) -> HiddenCausalTrajectory:
    full_mean = full.mean(axis=0)
    full_std = full.std(axis=0)
    cand_mean = cand.mean(axis=0)
    cand_std = cand.std(axis=0)
    return HiddenCausalTrajectory(
        perturbation_type=name,
        n_steps=n_steps, n_replicates=n_replicates,
        full_grid_l1_mean=full_mean.tolist(),
        full_grid_l1_std=full_std.tolist(),
        candidate_footprint_l1_mean=cand_mean.tolist(),
        candidate_footprint_l1_std=cand_std.tolist(),
        mean_immediate_l1=float(full_mean[0]) if n_steps > 0 else 0.0,
        mean_final_l1=float(full_mean[-1]) if n_steps > 0 else 0.0,
        mean_auc=float(full_mean.sum()),
        mean_n_flips=float(n_flips.mean()) if n_flips.size else 0.0,
        per_replicate_immediate=full[:, 0].tolist() if full.size else [],
        per_replicate_final=full[:, -1].tolist() if full.size else [],
        per_replicate_auc=full.sum(axis=1).tolist() if full.size else [],
    )


# ---------------------------------------------------------------------------
# Aggregation across candidates
# ---------------------------------------------------------------------------


def aggregate_hce_stats(reports: list[HiddenCausalReport]) -> dict:
    """Compute population-level HCE statistics across many candidates.

    Returns a dict with:
      n_candidates, mean_HCE, median_HCE, std_HCE,
      mean_visible_final_l1, mean_ratio,
      one_sample_t_hce_gt_zero (test that mean HCE > 0),
      paired_p_hce_lt_visible (test HCE < visible_final, sign test on diffs)
    """
    if not reports:
        return {
            "n_candidates": 0, "mean_HCE": 0.0, "median_HCE": 0.0,
            "std_HCE": 0.0, "mean_visible_final_l1": 0.0, "mean_ratio": 0.0,
            "one_sample_p_hce_gt_zero": 1.0,
            "paired_p_hce_lt_visible": 1.0,
        }
    hce = np.array([r.HCE for r in reports], dtype=np.float64)
    vis = np.array([r.visible_final_l1 for r in reports], dtype=np.float64)
    ratios = np.array([r.hce_to_visible_ratio for r in reports], dtype=np.float64)

    # One-sample test that mean HCE > 0: simple sign test.
    n_pos = int((hce > 0).sum())
    n = int(hce.size)
    # Two-sided p approximation using binomial under H0=0.5.
    from math import comb
    one_sample_p = 0.0
    if n > 0:
        # P(X >= n_pos | H0=0.5)
        upper_tail = sum(comb(n, k) for k in range(n_pos, n + 1)) / (2 ** n)
        one_sample_p = min(1.0, 2.0 * min(upper_tail, 1 - upper_tail + (1 / 2 ** n)))

    # Paired sign test for HCE < visible.
    diffs = vis - hce
    n_diff_pos = int((diffs > 0).sum())
    paired_p = 1.0
    if n > 0:
        upper_tail = sum(comb(n, k) for k in range(n_diff_pos, n + 1)) / (2 ** n)
        paired_p = min(1.0, 2.0 * min(upper_tail, 1 - upper_tail + (1 / 2 ** n)))

    return {
        "n_candidates": int(n),
        "mean_HCE": float(hce.mean()),
        "median_HCE": float(np.median(hce)),
        "std_HCE": float(hce.std()),
        "min_HCE": float(hce.min()),
        "max_HCE": float(hce.max()),
        "mean_visible_final_l1": float(vis.mean()),
        "median_visible_final_l1": float(np.median(vis)),
        "mean_ratio": float(ratios.mean()),
        "fraction_hce_positive": float(n_pos / n) if n > 0 else 0.0,
        "one_sample_p_hce_gt_zero": float(one_sample_p),
        "paired_p_hce_lt_visible": float(paired_p),
        "mean_immediate_check": float(np.mean([r.hce_immediate_check for r in reports])),
    }


def _empty_compare_result(strategy: str) -> dict:
    return {
        "n_paired": 0, "n_coherent": 0, "n_shuffled": 0,
        "mean_diff_coh_minus_shuf": 0.0,
        "median_diff": 0.0, "sign_test_p": 1.0,
        "bootstrap_ci_low": 0.0, "bootstrap_ci_high": 0.0,
        "n_coherent_wins": 0, "n_shuffled_wins": 0,
        "comparison_strategy": strategy,
    }


def _bootstrap_diff_ci(diffs: np.ndarray, *, n_boot: int = 2000, seed: int = 0
                       ) -> tuple[float, float]:
    if diffs.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = diffs.size
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def _sign_test_p(diffs: np.ndarray) -> float:
    from math import comb
    n = int(diffs.size)
    if n == 0:
        return 1.0
    n_pos = int((diffs > 0).sum())
    upper_tail = sum(comb(n, k) for k in range(n_pos, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * min(upper_tail, 1 - upper_tail + (1 / 2 ** n)))


def compare_hce_paired(
    coherent_reports: list[HiddenCausalReport],
    shuffled_reports: list[HiddenCausalReport],
) -> dict:
    """Compare HCE between coherent and shuffled-4D candidates.

    Tries three strategies in order:
      1. **id_pairing**: match by track_id (works only if the same simulation
         seed + rule produced both runs and IDs happen to overlap, which is
         rare; coherent/shuffled simulations diverge from t=1 onwards).
      2. **rank_pairing**: sort both lists by descending HCE and pair
         position-wise up to min(N_coh, N_shuf). This is a pseudo-paired
         comparison treating "the kth-best candidate" as a matched unit
         across conditions.
      3. **unpaired**: full distributions; report mean diff, CI from
         independent-bootstrap, sign-test on rank pairs, but flag as unpaired.

    Falls back through the strategies. Returns a dict that always has the
    same keys (Part B-friendly).
    """
    if not coherent_reports or not shuffled_reports:
        return _empty_compare_result("none")

    # Strategy 1: track-id pairing.
    sh_by_id = {r.track_id: r for r in shuffled_reports}
    id_pairs = [(c, sh_by_id[c.track_id]) for c in coherent_reports
                if c.track_id in sh_by_id]
    if id_pairs:
        diffs = np.array([c.HCE - s.HCE for c, s in id_pairs], dtype=np.float64)
        ci_low, ci_high = _bootstrap_diff_ci(diffs)
        return {
            "n_paired": int(len(id_pairs)),
            "n_coherent": len(coherent_reports),
            "n_shuffled": len(shuffled_reports),
            "mean_diff_coh_minus_shuf": float(diffs.mean()),
            "median_diff": float(np.median(diffs)),
            "sign_test_p": _sign_test_p(diffs),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "n_coherent_wins": int((diffs > 0).sum()),
            "n_shuffled_wins": int((diffs < 0).sum()),
            "comparison_strategy": "id_pairing",
        }

    # Strategy 2: rank pairing.
    coh_sorted = sorted(coherent_reports, key=lambda r: -r.HCE)
    sh_sorted = sorted(shuffled_reports, key=lambda r: -r.HCE)
    n_pairs = min(len(coh_sorted), len(sh_sorted))
    if n_pairs > 0:
        diffs = np.array([coh_sorted[i].HCE - sh_sorted[i].HCE
                          for i in range(n_pairs)], dtype=np.float64)
        ci_low, ci_high = _bootstrap_diff_ci(diffs)
        return {
            "n_paired": int(n_pairs),
            "n_coherent": len(coherent_reports),
            "n_shuffled": len(shuffled_reports),
            "mean_diff_coh_minus_shuf": float(diffs.mean()),
            "median_diff": float(np.median(diffs)),
            "sign_test_p": _sign_test_p(diffs),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "n_coherent_wins": int((diffs > 0).sum()),
            "n_shuffled_wins": int((diffs < 0).sum()),
            "comparison_strategy": "rank_pairing",
        }

    return _empty_compare_result("none")

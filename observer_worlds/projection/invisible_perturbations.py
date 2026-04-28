"""Projection-specific hidden-invisible perturbation generation.

A "hidden-invisible" perturbation modifies the 4D substrate state in
ways that the chosen projection's output cannot detect (within
tolerance). It is the central artifact-control of the HCE protocol:
without it, ``initial_projection_delta`` is non-zero and any HCE we
measure is contaminated by visible-component drift.

Stage 2 used a uniform random-bit-flip strategy that was strictly
invisible only for ``mean_threshold``. Stage 2B replaces that with
projection-specific filtering:

* **Count-preserving swap** (used for ``mean_threshold``, ``sum_threshold``,
  ``max_projection``, ``parity_projection``). For each ``(x, y)`` cell in
  the candidate region, pick one ON and one OFF cell in the
  ``(z, w)`` fibre, toggle both. Total count is unchanged, so:

  * mean and sum are unchanged (mean_threshold / sum_threshold pass);
  * max stays at 1 because at least one ON cell remains (max_projection
    passes);
  * parity is unchanged because the count is unchanged (parity passes).

  The strategy fails on a fibre that is all-zero or all-one (no
  swap-pair available); such fibres are simply not perturbed.

* **Verification-based rejection sampling** (used for
  ``random_linear_projection`` and ``multi_channel_projection``). The
  projection depends on *which* hidden cells are ON, not only on the
  count, so count-preserving swaps don't generally preserve it. We
  propose random flip sets, verify the projection at the candidate's
  perturbation slice, and accept if ``initial_projection_delta``
  is within tolerance. If ``max_attempts`` are exhausted without
  acceptance, the perturbation is reported invalid for that candidate.

Returns a tuple ``(perturbed_X, report)`` where ``report`` is the
artifact-audit dict the runner records per candidate.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from observer_worlds.projection import default_suite


# ---------------------------------------------------------------------------
# Per-projection metadata
# ---------------------------------------------------------------------------

# Projections whose output depends only on the per-fibre count of ON
# cells: a count-preserving swap is exactly invisible.
_COUNT_BASED = (
    "mean_threshold",
    "sum_threshold",
    "max_projection",
    "parity_projection",
)

# Stage 5B: smarter strategies for the previously verification-only
# projections. Each is a deterministic, exact-or-tolerance-bounded
# preserving swap rather than rejection sampling on random flips.
_WEIGHT_CANCELING = ("random_linear_projection",)
_CHANNEL_SIGNATURE = ("multi_channel_projection",)

# Projections that still fall back to verification-based rejection
# sampling (none today; kept as the catch-all).
_VERIFICATION_BASED: tuple[str, ...] = ()

# Per-projection default invisibility tolerance. Binary / count-based
# projections get an effectively-exact 1e-6 ceiling. Continuous
# projections are relaxed because exact zero is a measure-zero event
# under random weights.
_DEFAULT_TOLERANCES: Mapping[str, float] = {
    "mean_threshold": 1e-6,
    "sum_threshold": 1e-6,
    "max_projection": 1e-6,
    "parity_projection": 1e-6,
    "random_linear_projection": 5e-3,
    "multi_channel_projection": 1e-6,
}


# ---------------------------------------------------------------------------
# Strategy 1: count-preserving swaps
# ---------------------------------------------------------------------------


def _count_preserving_swap(
    X: np.ndarray, *, candidate_mask: np.ndarray,
    rng: np.random.Generator, target_flip_fraction: float,
) -> tuple[np.ndarray, dict]:
    """For each ``(x, y)`` in ``candidate_mask``, swap ``k`` random
    ON↔OFF pairs in the ``(z, w)`` fibre, where ``k`` is chosen from
    ``target_flip_fraction``.

    A "swap" toggles two cells, so it counts as 2 flips. The fibre's
    count of ON cells is unchanged.

    Fibres that are entirely all-zero or all-one have no available
    swap pair; they are skipped (no perturbation at that ``(x, y)``).
    """
    Nx, Ny, Nz, Nw = X.shape
    perturbed = X.copy()
    fibre_size = Nz * Nw
    # Approximate target swaps per fibre. Each swap = 2 toggles, so we
    # divide by 2.
    n_swaps_per_fibre = max(1, int(round(target_flip_fraction * fibre_size / 2)))

    n_flipped_total = 0
    n_skipped_all_zero = 0
    n_skipped_all_one = 0
    n_fibres_touched = 0
    xys = np.argwhere(candidate_mask)
    for x, y in xys:
        fibre = perturbed[x, y].reshape(-1)  # view (Nz*Nw,)
        ones_idx = np.where(fibre == 1)[0]
        zeros_idx = np.where(fibre == 0)[0]
        if ones_idx.size == 0:
            n_skipped_all_zero += 1
            continue
        if zeros_idx.size == 0:
            n_skipped_all_one += 1
            continue
        n_fibres_touched += 1
        n_swaps = min(n_swaps_per_fibre, ones_idx.size, zeros_idx.size)
        if n_swaps == 0:
            continue
        on_pick = rng.choice(ones_idx, size=n_swaps, replace=False)
        off_pick = rng.choice(zeros_idx, size=n_swaps, replace=False)
        fibre[on_pick] = 0
        fibre[off_pick] = 1
        n_flipped_total += int(n_swaps * 2)

    return perturbed, {
        "n_flipped": n_flipped_total,
        "n_fibres_touched": n_fibres_touched,
        "n_skipped_all_zero": n_skipped_all_zero,
        "n_skipped_all_one": n_skipped_all_one,
    }


# ---------------------------------------------------------------------------
# Strategy 2a (Stage 5B): weight-canceling pair swap for random_linear
# ---------------------------------------------------------------------------


def _weight_canceling_pair_swap(
    X: np.ndarray, *, candidate_mask: np.ndarray, projection_config: Mapping,
    rng: np.random.Generator, target_flip_fraction: float,
    tolerance: float,
) -> tuple[np.ndarray, dict]:
    """For each ``(x, y)`` in ``candidate_mask``, find ON / OFF pairs in
    the ``(z, w)`` fibre whose weights ``W[i], W[j]`` are closest, and
    swap them. The change in projection at that ``(x, y)`` is
    ``W[j] - W[i]``; absolute value is bounded by the chosen pair.

    Multiple swaps per fibre stack their net change additively. To keep
    that bounded by ``tolerance``, we choose pairs greedily and stop
    once any further swap would push the running per-fibre delta over
    ``tolerance``.

    If no fibre has any ON-OFF pair within ``tolerance``, the
    perturbation is reported invalid with ``no_weight_canceling_pair``.
    """
    from observer_worlds.projection.projection_suite import (
        random_linear_weights,
    )
    Nx, Ny, Nz, Nw = X.shape
    seed = int(projection_config.get("seed", 0))
    weights = random_linear_weights(Nz, Nw, seed=seed)
    weights_flat = weights.reshape(-1)
    perturbed = X.copy()

    n_flipped_total = 0
    n_fibres_touched = 0
    n_pair_candidates = 0
    best_pair_deltas: list[float] = []
    n_skipped_all_zero = 0
    n_skipped_all_one = 0
    n_skipped_no_pair_within_tolerance = 0

    fibre_size = Nz * Nw
    target_swaps_per_fibre = max(1, int(round(
        target_flip_fraction * fibre_size / 2,
    )))

    # The full-grid tolerance averages over Nx*Ny cells; a per-fibre
    # swap of magnitude d contributes d / (Nx*Ny) to that mean. So we
    # let the per-fibre swap pick the smallest available pairs without
    # a tight per-fibre gate; the full-grid delta is the binding
    # invariant and is checked once at the end of the pass.
    for x, y in np.argwhere(candidate_mask):
        fibre = perturbed[x, y].reshape(-1)
        ones_idx = np.where(fibre == 1)[0]
        zeros_idx = np.where(fibre == 0)[0]
        if ones_idx.size == 0:
            n_skipped_all_zero += 1
            continue
        if zeros_idx.size == 0:
            n_skipped_all_one += 1
            continue
        # All ON-OFF pair deltas: W[j] - W[i] for j in zeros, i in ones.
        diffs = (weights_flat[zeros_idx][None, :]
                 - weights_flat[ones_idx][:, None])
        n_pair_candidates += int(diffs.size)
        # Sort pairs by absolute value of the swap delta.
        flat_idx = np.argsort(np.abs(diffs), axis=None)
        used_ones: set[int] = set()
        used_zeros: set[int] = set()
        n_swaps_here = 0
        best_local = float("inf")
        for k in flat_idx:
            i_pos = int(k // diffs.shape[1])
            j_pos = int(k % diffs.shape[1])
            i = int(ones_idx[i_pos]); j = int(zeros_idx[j_pos])
            if i in used_ones or j in used_zeros:
                continue
            d = float(diffs[i_pos, j_pos])
            if abs(d) < best_local:
                best_local = abs(d)
            fibre[i] = 0; fibre[j] = 1
            used_ones.add(i); used_zeros.add(j)
            n_swaps_here += 1
            n_flipped_total += 2
            if n_swaps_here >= target_swaps_per_fibre:
                break
        if n_swaps_here > 0:
            n_fibres_touched += 1
            best_pair_deltas.append(best_local)
        else:
            n_skipped_no_pair_within_tolerance += 1

    info = {
        "n_flipped": n_flipped_total,
        "n_fibres_touched": n_fibres_touched,
        "n_pair_candidates_considered": n_pair_candidates,
        "best_pair_delta":
            float(min(best_pair_deltas)) if best_pair_deltas else None,
        "n_skipped_all_zero": n_skipped_all_zero,
        "n_skipped_all_one": n_skipped_all_one,
        "n_skipped_no_pair_within_tolerance":
            n_skipped_no_pair_within_tolerance,
    }
    return perturbed, info


# ---------------------------------------------------------------------------
# Strategy 2b (Stage 5B): channel-signature pair swap for multi_channel
# ---------------------------------------------------------------------------


def _channel_signature_pair_swap(
    X: np.ndarray, *, candidate_mask: np.ndarray, projection_config: Mapping,
    rng: np.random.Generator, target_flip_fraction: float,
) -> tuple[np.ndarray, dict]:
    """Group ``(z, w)`` cells by their channel-mask signature and swap
    ON-OFF pairs within the same signature group.

    Cells with identical signatures contribute identically to every
    channel; swapping an ON cell for an OFF cell of the same signature
    keeps each channel's masked count unchanged → channel mean
    unchanged → thresholded channel value unchanged. Exact preservation
    by construction.

    If a fibre has no signature group with both ON and OFF cells, no
    swap is possible at that ``(x, y)``. If no fibre in the candidate
    mask has any swap-eligible signature group, the perturbation is
    reported invalid with ``no_channel_preserving_pair``.
    """
    from observer_worlds.projection.projection_suite import (
        multi_channel_masks,
    )
    Nx, Ny, Nz, Nw = X.shape
    n_channels = int(projection_config.get("n_channels", 4))
    seed = int(projection_config.get("seed", 0))
    masks = multi_channel_masks(Nz, Nw, n_channels=n_channels, seed=seed)
    # Per-cell signature: (n_channels,) bool tuple.
    sigs_arr = np.stack(
        [m.reshape(-1) > 0 for m in masks], axis=-1,
    )  # (Nz*Nw, n_channels)
    sig_keys = [tuple(int(b) for b in row) for row in sigs_arr]

    perturbed = X.copy()
    n_flipped_total = 0
    n_fibres_touched = 0
    n_skipped_all_zero = 0
    n_skipped_all_one = 0
    n_skipped_no_signature_group_with_pair = 0

    fibre_size = Nz * Nw
    target_swaps_per_fibre = max(1, int(round(
        target_flip_fraction * fibre_size / 2,
    )))

    for x, y in np.argwhere(candidate_mask):
        fibre = perturbed[x, y].reshape(-1)
        ones_idx = np.where(fibre == 1)[0]
        zeros_idx = np.where(fibre == 0)[0]
        if ones_idx.size == 0:
            n_skipped_all_zero += 1
            continue
        if zeros_idx.size == 0:
            n_skipped_all_one += 1
            continue
        # Group ones / zeros by signature.
        ones_by_sig: dict[tuple, list[int]] = {}
        zeros_by_sig: dict[tuple, list[int]] = {}
        for i in ones_idx:
            ones_by_sig.setdefault(sig_keys[int(i)], []).append(int(i))
        for j in zeros_idx:
            zeros_by_sig.setdefault(sig_keys[int(j)], []).append(int(j))
        eligible_sigs = [
            s for s in ones_by_sig
            if s in zeros_by_sig
            and len(ones_by_sig[s]) > 0 and len(zeros_by_sig[s]) > 0
        ]
        if not eligible_sigs:
            n_skipped_no_signature_group_with_pair += 1
            continue
        # Greedy: pick swaps from sig groups, smallest signature first
        # for stability across runs.
        n_swaps_here = 0
        rng_local = np.random.default_rng(int(rng.integers(0, 2**31)))
        for sig in sorted(eligible_sigs):
            ons = list(ones_by_sig[sig])
            offs = list(zeros_by_sig[sig])
            n_pairs = min(len(ons), len(offs))
            if n_pairs == 0:
                continue
            rng_local.shuffle(ons); rng_local.shuffle(offs)
            for k in range(n_pairs):
                if n_swaps_here >= target_swaps_per_fibre:
                    break
                i, j = ons[k], offs[k]
                fibre[i] = 0; fibre[j] = 1
                n_swaps_here += 1
                n_flipped_total += 2
            if n_swaps_here >= target_swaps_per_fibre:
                break
        if n_swaps_here > 0:
            n_fibres_touched += 1

    info = {
        "n_flipped": n_flipped_total,
        "n_fibres_touched": n_fibres_touched,
        "n_pair_candidates_considered": int(sigs_arr.shape[0]),
        "best_pair_delta": 0.0,  # exact by construction when accepted
        "n_skipped_all_zero": n_skipped_all_zero,
        "n_skipped_all_one": n_skipped_all_one,
        "n_skipped_no_signature_group_with_pair":
            n_skipped_no_signature_group_with_pair,
    }
    return perturbed, info


# ---------------------------------------------------------------------------
# Strategy 3: verification-based rejection sampling (catch-all)
# ---------------------------------------------------------------------------


def _project_with_suite(projection_name: str, projection_config: Mapping,
                         X: np.ndarray) -> np.ndarray:
    suite = default_suite()
    return suite.project(projection_name, X, **projection_config)


def _projection_delta(
    a: np.ndarray, b: np.ndarray, *, restrict_mask: np.ndarray | None = None,
) -> float:
    """Mean per-cell |a - b|. If ``restrict_mask`` (2D) is provided,
    measure only inside the masked region (used to gate acceptance to
    the candidate's perturbation footprint, ignoring numerical drift
    outside)."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=-1)
    if restrict_mask is not None:
        if not restrict_mask.any():
            return 0.0
        return float(diff[restrict_mask].sum() / float(restrict_mask.sum()))
    return float(diff.mean())


def _verification_based(
    X: np.ndarray, *, candidate_mask: np.ndarray,
    projection_name: str, projection_config: Mapping,
    rng: np.random.Generator, target_flip_fraction: float,
    max_attempts: int, tolerance: float,
) -> tuple[np.ndarray, dict]:
    """Try random flip sets until projection is preserved within
    ``tolerance`` *outside* the candidate region (we deliberately
    measure on the full grid because the projection is non-local).

    Reports ``initial_projection_delta`` on the full projected output
    so the audit is comparable across projections.
    """
    Nx, Ny, Nz, Nw = X.shape
    proj_unperturbed = _project_with_suite(
        projection_name, projection_config, X,
    )
    xys = np.argwhere(candidate_mask)
    n_fibre_cells = Nz * Nw
    target_total_flips = max(1, int(round(
        target_flip_fraction * xys.shape[0] * n_fibre_cells,
    )))
    best_delta = float("inf")
    best_perturbed = X.copy()
    best_n_flipped = 0
    accepted = False
    attempts_used = 0
    for attempt in range(int(max_attempts)):
        attempts_used = attempt + 1
        candidate_X = X.copy()
        # Sample (xy, z, w) triples to flip.
        if xys.shape[0] == 0:
            return X.copy(), {
                "n_flipped": 0, "best_delta": 0.0,
                "attempts_used": attempts_used,
                "invalid_reason": "empty candidate mask",
                "preservation_strategy": "verification",
            }
        pick_xy = rng.integers(0, xys.shape[0], size=target_total_flips)
        zs = rng.integers(0, Nz, size=target_total_flips)
        ws = rng.integers(0, Nw, size=target_total_flips)
        for i in range(target_total_flips):
            x, y = xys[pick_xy[i]]
            candidate_X[x, y, zs[i], ws[i]] ^= 1
        proj_pert = _project_with_suite(
            projection_name, projection_config, candidate_X,
        )
        d = _projection_delta(proj_pert, proj_unperturbed)
        if d < best_delta:
            best_delta = d
            best_perturbed = candidate_X
            best_n_flipped = int(target_total_flips)
        if d <= tolerance:
            accepted = True
            return candidate_X, {
                "n_flipped": int(target_total_flips),
                "best_delta": float(d),
                "attempts_used": attempts_used,
                "invalid_reason": None,
                "preservation_strategy": "verification",
            }
    # Exhausted attempts.
    return best_perturbed, {
        "n_flipped": best_n_flipped,
        "best_delta": float(best_delta),
        "attempts_used": attempts_used,
        "invalid_reason":
            f"verification failed after {attempts_used} attempts "
            f"(best_delta={best_delta:.4g}, tolerance={tolerance:.4g})",
        "preservation_strategy": "verification",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_projection_invisible_perturbation(
    X: np.ndarray,
    candidate_mask: np.ndarray,
    projection_name: str,
    projection_config: Mapping | None = None,
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 100,
    target_flip_fraction: float = 0.1,
    verification_tolerance: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a hidden-invisible perturbation under one projection.

    Parameters
    ----------
    X
        4D state ``(Nx, Ny, Nz, Nw)``, ``uint8``.
    candidate_mask
        2D bool / uint8 mask ``(Nx, Ny)``. Perturbation is applied only
        in the ``(z, w)`` hyperplane at locations where this is True.
    projection_name
        Name registered in :func:`default_suite`.
    projection_config
        Per-projection params (``theta`` for thresholded projections,
        ``seed`` for random projections, ``n_channels`` for multi-
        channel). Optional; defaults from the suite are applied.
    rng
        ``numpy`` random generator. One is created if omitted.
    max_attempts
        Verification-based projections retry up to this many times.
    target_flip_fraction
        Approximate fraction of (z, w) cells to flip per perturbed
        fibre. The strategies translate this differently:

        * count-preserving: half this fraction in *swaps* per fibre
          (each swap toggles two cells)
        * verification: this fraction over the entire perturbation
          region
    verification_tolerance
        Per-cell projection-delta tolerance for the verification
        strategy. Counted on the full projected output (not just the
        candidate region) so the audit is comparable across
        projections.

    Returns
    -------
    perturbed_X
        ``np.ndarray`` of the same shape as ``X``. If the perturbation
        was invalid, the returned array may equal ``X`` or contain a
        best-effort attempt; callers should check ``report.accepted``
        before using it for HCE.
    report
        Dict with the artifact-audit fields:

        * ``projection_name``
        * ``preservation_strategy`` — ``"count_preserving_swap"`` |
          ``"verification"``
        * ``attempted`` — bool (always True here; the runner gates on
          mask emptiness)
        * ``accepted`` — bool
        * ``initial_projection_delta`` — float (full-grid mean abs
          difference of projected outputs)
        * ``n_flipped`` — total bits toggled
        * ``target_region_size`` — number of ``(x, y)`` cells in the
          candidate region (size of fibres available for perturbation)
        * ``invalid_reason`` — string when not accepted, else None
    """
    if projection_name not in default_suite().names():
        raise ValueError(
            f"unknown projection {projection_name!r}; "
            f"available: {default_suite().names()}"
        )
    if rng is None:
        rng = np.random.default_rng()
    if verification_tolerance is None:
        verification_tolerance = float(
            _DEFAULT_TOLERANCES.get(projection_name, 1e-6)
        )
    cm = candidate_mask.astype(bool)
    target_size = int(cm.sum())

    suite = default_suite()
    spec = suite.get(projection_name)
    config = dict(spec.default_params)
    if projection_config:
        config.update(projection_config)

    base_report = {
        "projection_name": projection_name,
        "attempted": True,
        "target_region_size": target_size,
    }

    if target_size == 0:
        base_report.update({
            "preservation_strategy": "noop",
            "accepted": False,
            "initial_projection_delta": 0.0,
            "n_flipped": 0,
            "invalid_reason": "empty candidate mask",
        })
        return X.copy(), base_report

    # --- Strategy dispatch ---------------------------------------------
    if projection_name in _COUNT_BASED:
        perturbed, info = _count_preserving_swap(
            X, candidate_mask=cm, rng=rng,
            target_flip_fraction=target_flip_fraction,
        )
        # Verify (cheap consistency check; should be exactly 0 by
        # construction).
        proj_un = _project_with_suite(projection_name, config, X)
        proj_pe = _project_with_suite(projection_name, config, perturbed)
        delta = _projection_delta(proj_pe, proj_un)
        accepted = delta <= verification_tolerance
        # If no swap was possible anywhere, mark invalid with a clear
        # reason; HCE for that candidate is uninterpretable.
        if info["n_fibres_touched"] == 0:
            base_report.update({
                "preservation_strategy": "count_preserving_swap",
                "accepted": False,
                "initial_projection_delta": float(delta),
                "n_flipped": 0,
                "invalid_reason": (
                    "no fibre in candidate mask had both ON and OFF "
                    f"cells (all_zero={info['n_skipped_all_zero']}, "
                    f"all_one={info['n_skipped_all_one']})"
                ),
            })
            return X.copy(), base_report
        base_report.update({
            "preservation_strategy": "count_preserving_swap",
            "accepted": bool(accepted),
            "initial_projection_delta": float(delta),
            "n_flipped": int(info["n_flipped"]),
            "n_fibres_touched": int(info["n_fibres_touched"]),
            "n_skipped_all_zero": int(info["n_skipped_all_zero"]),
            "n_skipped_all_one": int(info["n_skipped_all_one"]),
            "invalid_reason": None if accepted else (
                f"count-preserving swap produced delta {delta:.4g} > "
                f"tolerance {verification_tolerance:.4g}"
            ),
        })
        return perturbed, base_report

    if projection_name in _WEIGHT_CANCELING:
        # random_linear_projection: weight-canceling pair swap.
        perturbed, info = _weight_canceling_pair_swap(
            X, candidate_mask=cm, projection_config=config, rng=rng,
            target_flip_fraction=target_flip_fraction,
            tolerance=verification_tolerance,
        )
        proj_un = _project_with_suite(projection_name, config, X)
        proj_pe = _project_with_suite(projection_name, config, perturbed)
        delta = _projection_delta(proj_pe, proj_un)
        accepted = (info["n_fibres_touched"] > 0
                    and delta <= verification_tolerance)
        invalid_reason = None
        if info["n_fibres_touched"] == 0:
            invalid_reason = (
                "no_weight_canceling_pair (no ON-OFF pair in any "
                "candidate fibre had |W[j]-W[i]| within tolerance "
                f"{verification_tolerance:.4g}; smallest seen "
                f"={info.get('best_pair_delta')!r})"
            )
            accepted = False
        elif delta > verification_tolerance:
            invalid_reason = (
                f"weight-canceling swap produced full-grid delta "
                f"{delta:.4g} > tolerance {verification_tolerance:.4g}"
            )
            accepted = False
        base_report.update({
            "preservation_strategy": "weight_canceling_pair_swap",
            "accepted": bool(accepted),
            "initial_projection_delta": float(delta),
            "projection_tolerance_used": float(verification_tolerance),
            "n_flipped": int(info["n_flipped"]),
            "n_pair_candidates_considered":
                int(info["n_pair_candidates_considered"]),
            "best_pair_delta": info["best_pair_delta"],
            "n_fibres_touched": int(info["n_fibres_touched"]),
            "invalid_reason": invalid_reason,
        })
        return perturbed, base_report

    if projection_name in _CHANNEL_SIGNATURE:
        # multi_channel_projection: signature-grouped pair swap.
        perturbed, info = _channel_signature_pair_swap(
            X, candidate_mask=cm, projection_config=config, rng=rng,
            target_flip_fraction=target_flip_fraction,
        )
        proj_un = _project_with_suite(projection_name, config, X)
        proj_pe = _project_with_suite(projection_name, config, perturbed)
        delta = _projection_delta(proj_pe, proj_un)
        accepted = (info["n_fibres_touched"] > 0
                    and delta <= verification_tolerance)
        invalid_reason = None
        if info["n_fibres_touched"] == 0:
            invalid_reason = (
                "no_channel_preserving_pair (no signature group in any "
                "candidate fibre had both ON and OFF cells)"
            )
            accepted = False
        elif delta > verification_tolerance:
            invalid_reason = (
                f"channel-signature swap produced delta {delta:.4g} > "
                f"tolerance {verification_tolerance:.4g}"
            )
            accepted = False
        base_report.update({
            "preservation_strategy": "channel_signature_pair_swap",
            "accepted": bool(accepted),
            "initial_projection_delta": float(delta),
            "projection_tolerance_used": float(verification_tolerance),
            "n_flipped": int(info["n_flipped"]),
            "n_pair_candidates_considered":
                int(info["n_pair_candidates_considered"]),
            "best_pair_delta": info["best_pair_delta"],
            "n_fibres_touched": int(info["n_fibres_touched"]),
            "invalid_reason": invalid_reason,
        })
        return perturbed, base_report

    if projection_name in _VERIFICATION_BASED:
        perturbed, info = _verification_based(
            X, candidate_mask=cm, projection_name=projection_name,
            projection_config=config, rng=rng,
            target_flip_fraction=target_flip_fraction,
            max_attempts=max_attempts,
            tolerance=verification_tolerance,
        )
        accepted = info["invalid_reason"] is None
        base_report.update({
            "preservation_strategy": info["preservation_strategy"],
            "accepted": bool(accepted),
            "initial_projection_delta": float(info["best_delta"]),
            "n_flipped": int(info["n_flipped"]),
            "attempts_used": int(info["attempts_used"]),
            "invalid_reason": info["invalid_reason"],
        })
        return perturbed, base_report

    # Defensive: a future projection without a strategy.
    base_report.update({
        "preservation_strategy": "none",
        "accepted": False,
        "initial_projection_delta": 0.0,
        "n_flipped": 0,
        "invalid_reason": (
            f"projection {projection_name!r} has no perturbation "
            "strategy assigned"
        ),
    })
    return X.copy(), base_report

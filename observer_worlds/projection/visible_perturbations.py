"""Projection-changing (visible) perturbation generation.

Stage 5B counterpart to :mod:`invisible_perturbations`. Where the
invisible generator preserves the projection within tolerance, this
module **intentionally** produces a perturbation whose projection at
or near the candidate region differs from the unperturbed projection
by at least ``min_projection_delta``.

Used by the agent-task layer to compute
``visible_intervention_task_delta`` alongside
``hidden_intervention_task_delta``: a candidate's task performance is
re-measured starting from a state whose visible appearance has been
deliberately altered, so we can compare how much the task is sensitive
to *visible* state vs *hidden* state.

Strategy
--------

Smoke-quality but universal across the suite's six projections: for
each ``(x, y)`` in the candidate mask, flip the ENTIRE ``(z, w)`` fibre
(``XOR`` with 1). This maximally changes any cell-independent
projection at that location. We sample ``target_visible_fraction`` of
the candidate cells; if the resulting full-grid projection delta is
below ``min_projection_delta`` we add more cells until the delta
threshold is met or the candidate mask is exhausted.

If the candidate region cannot achieve the requested visible delta
(e.g., the projection is degenerate at every candidate cell), the
perturbation is reported invalid with a clear reason.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from observer_worlds.projection.projection_suite import default_suite


def _project_with_suite(
    projection_name: str, projection_config: Mapping, X: np.ndarray,
) -> np.ndarray:
    return default_suite().project(projection_name, X, **projection_config)


def _projection_delta(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=-1)
    return float(diff.mean())


def make_projection_visible_perturbation(
    X: np.ndarray,
    candidate_mask: np.ndarray,
    projection_name: str,
    projection_config: Mapping | None = None,
    rng: np.random.Generator | None = None,
    *,
    target_visible_fraction: float = 0.1,
    min_projection_delta: float = 1e-3,
    max_attempts: int = 100,
) -> tuple[np.ndarray, dict]:
    """Return a perturbation of ``X`` whose projection differs from the
    original by at least ``min_projection_delta`` (full-grid mean abs
    diff).

    Parameters
    ----------
    X
        4D state ``(Nx, Ny, Nz, Nw)``.
    candidate_mask
        2D bool mask ``(Nx, Ny)`` defining the perturbation region.
    projection_name
        Registered projection name.
    projection_config
        Per-projection params; defaults from the suite are applied.
    target_visible_fraction
        Initial fraction of candidate cells to flip. We start by
        flipping this fraction's worth of fibres entirely
        (``XOR 1``); if the projection delta is still below
        ``min_projection_delta`` we add more cells up to ``max_attempts``
        rounds.
    min_projection_delta
        Required minimum full-grid mean abs projection delta. If we
        cannot reach it within ``max_attempts`` even after flipping
        every candidate cell's fibre, the perturbation is invalid.
    max_attempts
        Cap on rounds of "add more flipped cells".

    Returns
    -------
    perturbed_X, report
        ``report`` includes ``visible_projection_delta``, ``n_flipped``,
        ``strategy``, ``valid``, ``invalid_reason``.
    """
    if projection_name not in default_suite().names():
        raise ValueError(
            f"unknown projection {projection_name!r}; "
            f"available: {default_suite().names()}"
        )
    if rng is None:
        rng = np.random.default_rng()

    suite = default_suite()
    spec = suite.get(projection_name)
    config = dict(spec.default_params)
    if projection_config:
        config.update(projection_config)

    cm = candidate_mask.astype(bool)
    target_size = int(cm.sum())
    base = {
        "projection_name": projection_name,
        "strategy": "fibre_xor_visible",
        "target_region_size": target_size,
    }

    if target_size == 0:
        base.update({
            "visible_projection_delta": 0.0,
            "n_flipped": 0, "valid": False,
            "invalid_reason": "empty candidate mask",
        })
        return X.copy(), base

    proj_unperturbed = _project_with_suite(projection_name, config, X)

    # Order candidate cells by a deterministic shuffle so the result is
    # reproducible per ``rng``.
    coords = np.argwhere(cm)
    perm = rng.permutation(coords.shape[0])
    coords = coords[perm]

    initial_n = max(1, int(round(
        float(target_visible_fraction) * coords.shape[0],
    )))
    perturbed = X.copy()
    n_used = 0
    n_flipped_bits = 0

    def _apply_fibre_flip(idx: int) -> int:
        x, y = int(coords[idx, 0]), int(coords[idx, 1])
        before = int(perturbed[x, y].sum())
        perturbed[x, y] ^= 1
        after = int(perturbed[x, y].sum())
        return abs(after - before) + (perturbed.shape[2] * perturbed.shape[3])

    # Apply the initial batch.
    for k in range(min(initial_n, coords.shape[0])):
        n_flipped_bits += _apply_fibre_flip(k)
        n_used += 1

    proj_pe = _project_with_suite(projection_name, config, perturbed)
    delta = _projection_delta(proj_pe, proj_unperturbed)
    attempts = 1
    # Add more fibres if delta is still below threshold.
    while (delta < float(min_projection_delta)
            and n_used < coords.shape[0]
            and attempts < int(max_attempts)):
        # Add roughly another batch of cells (same size as the initial
        # batch, capped at remaining).
        next_batch_end = min(coords.shape[0], n_used + initial_n)
        for k in range(n_used, next_batch_end):
            n_flipped_bits += _apply_fibre_flip(k)
            n_used += 1
        proj_pe = _project_with_suite(projection_name, config, perturbed)
        delta = _projection_delta(proj_pe, proj_unperturbed)
        attempts += 1

    valid = bool(delta >= float(min_projection_delta))
    invalid_reason = None
    if not valid:
        invalid_reason = (
            f"could not reach min_projection_delta "
            f"({float(min_projection_delta):.4g}); achieved "
            f"{delta:.4g} after flipping {n_used} of {coords.shape[0]} "
            f"candidate fibres"
        )
    base.update({
        "visible_projection_delta": float(delta),
        "n_flipped": int(n_flipped_bits),
        "n_fibres_flipped": int(n_used),
        "attempts_used": int(attempts),
        "valid": valid,
        "invalid_reason": invalid_reason,
        "min_projection_delta_requested": float(min_projection_delta),
    })
    return perturbed, base

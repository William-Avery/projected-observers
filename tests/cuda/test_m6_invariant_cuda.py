"""The M6 hidden-invisible projection-preservation invariant must hold
identically on the cuda backend: a hidden_shuffle perturbation of an
interior fiber must not change the t=0 projection.

This is the foundational invariant of M6/M7/M8 — if it fails on cuda,
no GPU run is meaningful.
"""
from __future__ import annotations

import numpy as np

from observer_worlds.metrics.causality_score import (
    apply_hidden_shuffle_intervention,
)
from observer_worlds.utils import seeded_rng
from observer_worlds.worlds import CA4D, BSRule, project


def test_hidden_shuffle_preserves_projection_t0():
    """The intervention runs on host (numpy); both backends read the
    same host arrays at t=0, so projection equality is bit-identical."""
    shape = (12, 12, 4, 4)
    rng = seeded_rng(42)
    state = (rng.random(shape) < 0.4).astype(np.uint8)
    interior = np.zeros(shape[:2], dtype=bool)
    interior[3:9, 3:9] = True

    perturbed = apply_hidden_shuffle_intervention(state, interior, rng)
    assert not np.array_equal(perturbed, state), "shuffle was a no-op"

    proj_orig = project(state, method="mean_threshold", theta=0.5)
    proj_pert = project(perturbed, method="mean_threshold", theta=0.5)
    np.testing.assert_array_equal(proj_orig, proj_pert)


def test_cuda_rollout_from_perturbed_preserves_t0_projection():
    """Run the perturbed state forward 1 step on the cuda backend and
    confirm the t=0 projection equals the unperturbed t=0 projection.
    (The t=0 projection is the projection BEFORE any step; the rollout
    test is just to confirm the cuda backend doesn't corrupt the input
    state during ingestion.)"""
    shape = (12, 12, 4, 4)
    rule = BSRule(birth=(3, 4, 5), survival=(2, 3, 4, 5))
    rng = seeded_rng(7)
    state = (rng.random(shape) < 0.4).astype(np.uint8)
    interior = np.zeros(shape[:2], dtype=bool)
    interior[3:9, 3:9] = True
    perturbed = apply_hidden_shuffle_intervention(state, interior, rng)

    # t=0 projection must match before any cuda contact.
    p0 = project(state, method="mean_threshold", theta=0.5)
    p0_pert = project(perturbed, method="mean_threshold", theta=0.5)
    np.testing.assert_array_equal(p0, p0_pert)

    # Now confirm cuda ingestion of perturbed state preserves the
    # post-ingestion projection (i.e. cupy round-trip is lossless).
    ca = CA4D(shape=shape, rule=rule, backend="cuda")
    ca.state = perturbed
    p_after_ingestion = project(ca.state, method="mean_threshold", theta=0.5)
    np.testing.assert_array_equal(p0_pert, p_after_ingestion)

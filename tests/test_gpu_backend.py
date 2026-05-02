"""Stage G1D — CPU/GPU equivalence tests for the new backend layer.

These tests skip cleanly when cupy is not available so the standard
CPU CI pipeline is unaffected. When cupy is available, every numerical
primitive is checked for bit-exact (binary) or tolerance (continuous)
agreement on tiny deterministic inputs.

Scope is the API in :mod:`observer_worlds.backends`. Production
runners are not invoked.
"""
from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.backends import (
    estimate_max_safe_batch_size,
    get_backend,
    is_cupy_available,
)
from observer_worlds.worlds.rules import BSRule


HAS_CUPY = is_cupy_available()
needs_cupy = pytest.mark.skipif(
    not HAS_CUPY, reason="cupy + CUDA device not available; GPU tests skipped"
)


# ---------------------------------------------------------------------------
# Deterministic mini-fixture
# ---------------------------------------------------------------------------


def _tiny_states(B: int = 3, Nx: int = 6, Ny: int = 6, Nz: int = 4, Nw: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((B, Nx, Ny, Nz, Nw)) < 0.3).astype(np.uint8)


def _tiny_masks(B: int, Nx: int, Ny: int, frac: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(1)
    return (rng.random((B, Nx, Ny)) < frac).astype(np.uint8)


def _tiny_luts(B: int) -> tuple[np.ndarray, np.ndarray]:
    rule = BSRule(birth=(3,), survival=(2, 3))
    bl, sl = rule.to_lookup_tables(80)
    bl = bl.astype(np.uint8)
    sl = sl.astype(np.uint8)
    return (
        np.broadcast_to(bl, (B, 81)).copy(),
        np.broadcast_to(sl, (B, 81)).copy(),
    )


# ---------------------------------------------------------------------------
# 1. Backend availability and skip behavior
# ---------------------------------------------------------------------------


def test_numpy_backend_always_available():
    backend = get_backend("numpy")
    assert backend.name == "numpy"
    assert backend.is_gpu is False
    a = backend.asarray(np.array([1, 2, 3]))
    assert isinstance(a, np.ndarray)
    assert backend.asnumpy(a).tolist() == [1, 2, 3]


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("opencl")


def test_cupy_backend_unavailable_raises_runtime_error_not_import_error():
    """If cupy is missing, ``get_backend('cupy')`` must raise
    ``RuntimeError`` (not ``ImportError``) so callers can downgrade to
    numpy without an import-failure traceback."""
    if HAS_CUPY:
        pytest.skip("cupy is installed; this guard only fires when it isn't")
    with pytest.raises(RuntimeError, match="cupy backend requested"):
        get_backend("cupy")


def test_cpu_path_does_not_require_cupy_import():
    """Importing the backends package and using NumpyBackend must not
    drag in cupy as a hard dep (the project must remain CPU-runnable
    on machines without CUDA)."""
    import observer_worlds.backends as bk
    backend = bk.get_backend("numpy")
    states = _tiny_states()
    bl, sl = _tiny_luts(states.shape[0])
    out = backend.step_4d_batch(states, bl, sl)
    assert out.shape == states.shape


# ---------------------------------------------------------------------------
# 2. Projection equivalence (numpy vs cupy)
# ---------------------------------------------------------------------------


PROJECTION_CASES = [
    ("mean_threshold", {}),
    ("sum_threshold", {"theta": 2}),
    ("max_projection", {}),
    ("parity_projection", {}),
    ("multi_channel_projection", {"n_channels": 4, "seed": 0}),
]


@needs_cupy
@pytest.mark.parametrize("method,params", PROJECTION_CASES)
def test_projection_equivalence_binary(method, params):
    """Binary projections must be bit-exact across backends."""
    states = _tiny_states()
    cpu = get_backend("numpy").project_batch(states, method=method, **params)
    gpu = get_backend("cupy").project_batch(
        get_backend("cupy").asarray(states), method=method, **params,
    )
    gpu_h = get_backend("cupy").asnumpy(gpu)
    assert cpu.shape == gpu_h.shape
    np.testing.assert_array_equal(cpu, gpu_h)


@needs_cupy
def test_projection_equivalence_random_linear():
    """Continuous projection — within fp32 tolerance.

    The reduction order on GPU may differ from numpy's; agreement is
    checked at fp32 ulp scale.
    """
    states = _tiny_states()
    cpu = get_backend("numpy").project_batch(
        states, method="random_linear_projection", seed=0,
    )
    gpu = get_backend("cupy").project_batch(
        get_backend("cupy").asarray(states),
        method="random_linear_projection", seed=0,
    )
    gpu_h = get_backend("cupy").asnumpy(gpu)
    np.testing.assert_allclose(cpu, gpu_h, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. One-step CA equivalence
# ---------------------------------------------------------------------------


@needs_cupy
def test_step_4d_batch_one_step_equivalence():
    states = _tiny_states()
    bl, sl = _tiny_luts(states.shape[0])
    cpu = get_backend("numpy").step_4d_batch(states, bl, sl)
    gpu_backend = get_backend("cupy")
    gpu = gpu_backend.step_4d_batch(
        gpu_backend.asarray(states),
        gpu_backend.asarray(bl),
        gpu_backend.asarray(sl),
    )
    np.testing.assert_array_equal(cpu, gpu_backend.asnumpy(gpu))


@needs_cupy
def test_step_4d_batch_multi_step_equivalence():
    """Five sequential steps must remain bit-identical."""
    states = _tiny_states()
    bl, sl = _tiny_luts(states.shape[0])
    cpu_backend = get_backend("numpy")
    gpu_backend = get_backend("cupy")
    cur_c = states
    cur_g = gpu_backend.asarray(states)
    bl_g = gpu_backend.asarray(bl)
    sl_g = gpu_backend.asarray(sl)
    for _ in range(5):
        cur_c = cpu_backend.step_4d_batch(cur_c, bl, sl)
        cur_g = gpu_backend.step_4d_batch(cur_g, bl_g, sl_g)
    np.testing.assert_array_equal(cur_c, gpu_backend.asnumpy(cur_g))


# ---------------------------------------------------------------------------
# 4. Hidden / visible perturbation equivalence
# ---------------------------------------------------------------------------


@needs_cupy
def test_hidden_perturbation_equivalence():
    states = _tiny_states()
    masks = _tiny_masks(states.shape[0], states.shape[1], states.shape[2])
    cpu = get_backend("numpy").apply_hidden_perturbations_batch(
        states, masks, n_swaps_per_fibre=2,
    )
    gpu_backend = get_backend("cupy")
    gpu = gpu_backend.apply_hidden_perturbations_batch(
        gpu_backend.asarray(states),
        gpu_backend.asarray(masks),
        n_swaps_per_fibre=2,
    )
    np.testing.assert_array_equal(cpu, gpu_backend.asnumpy(gpu))


@needs_cupy
def test_hidden_perturbation_preserves_mean_threshold_projection():
    """The contract of hidden swap: mean_threshold projection of the
    perturbed state must equal that of the original."""
    states = _tiny_states()
    masks = _tiny_masks(states.shape[0], states.shape[1], states.shape[2])
    pert = get_backend("numpy").apply_hidden_perturbations_batch(
        states, masks, n_swaps_per_fibre=2,
    )
    proj_orig = get_backend("numpy").project_batch(states, method="mean_threshold")
    proj_pert = get_backend("numpy").project_batch(pert, method="mean_threshold")
    np.testing.assert_array_equal(proj_orig, proj_pert)


@needs_cupy
def test_visible_perturbation_equivalence():
    states = _tiny_states()
    masks = _tiny_masks(states.shape[0], states.shape[1], states.shape[2])
    cpu = get_backend("numpy").apply_visible_perturbations_batch(
        states, masks, n_flips_per_fibre=1,
    )
    gpu_backend = get_backend("cupy")
    gpu = gpu_backend.apply_visible_perturbations_batch(
        gpu_backend.asarray(states),
        gpu_backend.asarray(masks),
        n_flips_per_fibre=1,
    )
    np.testing.assert_array_equal(cpu, gpu_backend.asnumpy(gpu))


# ---------------------------------------------------------------------------
# 5. HCE rollout equivalence (compact metric within tolerance)
# ---------------------------------------------------------------------------


@needs_cupy
def test_rollout_hce_equivalence_binary_projection():
    """Full rollout HCE metric must be bit-identical for binary
    projection (every step is uint8, so no fp tolerance applies)."""
    states = _tiny_states()
    bl, sl = _tiny_luts(states.shape[0])
    masks = _tiny_masks(states.shape[0], states.shape[1], states.shape[2])
    pert_cpu = get_backend("numpy").apply_hidden_perturbations_batch(
        states, masks, n_swaps_per_fibre=2,
    )
    cpu_metric = get_backend("numpy").rollout_hce_batch(
        states, pert_cpu, bl, sl, masks,
        horizons=[1, 3, 5], projection="mean_threshold",
    )
    gpu_backend = get_backend("cupy")
    states_g = gpu_backend.asarray(states)
    pert_g = gpu_backend.asarray(pert_cpu)
    masks_g = gpu_backend.asarray(masks)
    bl_g = gpu_backend.asarray(bl)
    sl_g = gpu_backend.asarray(sl)
    gpu_metric = gpu_backend.rollout_hce_batch(
        states_g, pert_g, bl_g, sl_g, masks_g,
        horizons=[1, 3, 5], projection="mean_threshold",
    )
    np.testing.assert_array_equal(cpu_metric, gpu_backend.asnumpy(gpu_metric))


@needs_cupy
def test_rollout_hce_equivalence_continuous_projection():
    states = _tiny_states()
    bl, sl = _tiny_luts(states.shape[0])
    masks = _tiny_masks(states.shape[0], states.shape[1], states.shape[2])
    pert = get_backend("numpy").apply_visible_perturbations_batch(
        states, masks, n_flips_per_fibre=2,
    )
    cpu_metric = get_backend("numpy").rollout_hce_batch(
        states, pert, bl, sl, masks,
        horizons=[1, 3], projection="random_linear_projection",
        projection_params={"seed": 0},
    )
    gpu_backend = get_backend("cupy")
    gpu_metric = gpu_backend.rollout_hce_batch(
        gpu_backend.asarray(states),
        gpu_backend.asarray(pert),
        gpu_backend.asarray(bl),
        gpu_backend.asarray(sl),
        gpu_backend.asarray(masks),
        horizons=[1, 3], projection="random_linear_projection",
        projection_params={"seed": 0},
    )
    np.testing.assert_allclose(
        cpu_metric, gpu_backend.asnumpy(gpu_metric), rtol=1e-4, atol=1e-3,
    )


# ---------------------------------------------------------------------------
# 6. Memory planner
# ---------------------------------------------------------------------------


def test_memory_planner_basic():
    """Standard production grid 64x64x8x8 with 5 perturbation conditions
    and 8 horizons must fit comfortably inside a 9.5 GB budget at
    batch sizes around the 256-2048 range."""
    b = estimate_max_safe_batch_size(
        grid=(64, 64, 8, 8),
        n_perturbation_conditions=5,
        n_projection_frames_per_state=8,
        target_gb=9.5,
    )
    # Sanity: > 1, sane order of magnitude.
    assert b >= 64
    assert b <= 200_000


def test_memory_planner_respects_target():
    """Smaller GB target -> smaller batch."""
    big = estimate_max_safe_batch_size(
        grid=(64, 64, 8, 8),
        n_perturbation_conditions=5,
        n_projection_frames_per_state=8,
        target_gb=9.5,
    )
    small = estimate_max_safe_batch_size(
        grid=(64, 64, 8, 8),
        n_perturbation_conditions=5,
        n_projection_frames_per_state=8,
        target_gb=1.0,
    )
    assert big > small
    assert big >= 1
    assert small >= 1


def test_memory_planner_minimum_one():
    """Even with absurd parameters, planner returns >= 1."""
    b = estimate_max_safe_batch_size(
        grid=(128, 128, 16, 16),
        n_perturbation_conditions=10,
        n_projection_frames_per_state=20,
        target_gb=0.001,
    )
    assert b >= 1

"""Workhorse module for Follow-up Topic 1 — projection robustness.

Per ``(rule, seed)``, runs the 4D substrate **once**, then evaluates each
requested projection from the same in-memory state stream. For each
projection it:

1. Projects every frame to 2D (binarising continuous / multi-channel
   projections through a documented rule).
2. Detects connected components per frame and tracks them across frames
   with the existing :class:`GreedyTracker`.
3. Picks the top-``max_candidates`` longest-lived tracks.
4. For each candidate, measures three rollout-based HCE-style numbers:
   * **HCE**: hidden 4D perturbation **inside** the candidate's
     interior bounding box at the peak frame. Future divergence is
     measured *in the candidate's local 2D region* over the requested
     horizons, then averaged over horizons.
   * **far_HCE**: same shape, but the perturbation is applied **outside**
     the candidate's bbox. If far_HCE ≈ HCE, the candidate is not
     locally responsible and the contrast is what M8 calls
     ``global_chaotic`` territory.
   * **sham_HCE**: identity perturbation (no change). Always ~0 by
     construction, included for symmetry with the M6 / M8 reporting
     surface.
5. Records ``initial_projection_delta`` — how much the perturbed and
   unperturbed t = 0 projection differ. If this is non-zero the
   intervention was not actually hidden-invisible under that
   projection; downstream interpretation must flag the candidate.

Stage 2 deliberately does **not** compute the full M8 mechanism
classifier (response maps, mediation, pathway traces). Adding that
across projections is a Stage-5+ refactor; the immediate Stage-2
question is whether HCE itself survives projection changes, which the
metrics above answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from observer_worlds.detection import GreedyTracker
from observer_worlds.detection.components import extract_components
from observer_worlds.projection import ProjectionSpec, ProjectionSuite, default_suite
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import DetectionConfig
from observer_worlds.worlds import CA4D


# ---------------------------------------------------------------------------
# 1. Substrate rollout (shared across projections within one (rule, seed))
# ---------------------------------------------------------------------------


def initial_4d_state(grid_shape, density: float, seed: int) -> np.ndarray:
    """Bernoulli(density) initial 4D state, deterministic in ``seed``."""
    rng = seeded_rng(seed)
    return (rng.random(grid_shape) < float(density)).astype(np.uint8)


def run_substrate(rule_bs, state0: np.ndarray, n_steps: int, *, backend: str
                  ) -> np.ndarray:
    """Run the 4D CA forward ``n_steps`` from ``state0``.

    Returns a ``(T+1, Nx, Ny, Nz, Nw)`` array; index 0 is ``state0``.
    """
    out = np.empty((n_steps + 1, *state0.shape), dtype=np.uint8)
    out[0] = state0
    ca = CA4D(shape=state0.shape, rule=rule_bs, backend=backend)
    ca.state = state0.copy()
    for t in range(n_steps):
        ca.step()
        out[t + 1] = ca.state.copy()
    return out


# ---------------------------------------------------------------------------
# 2. Projection -> 2D binarization for the detector
# ---------------------------------------------------------------------------


def binarize_for_detection(
    projected: np.ndarray, output_kind: str,
) -> np.ndarray:
    """Reduce any projection's output to a ``(Nx, Ny)`` ``uint8`` 0/1
    frame so the existing :func:`extract_components` can consume it.

    * ``binary`` — already ``uint8 0/1``; pass through.
    * ``count`` — threshold at ``> 0``.
    * ``continuous`` — threshold at the per-frame median (deterministic,
      keeps roughly half the cells active; documented as a smoke-level
      heuristic).
    * ``multi_channel`` — take channel ``0`` of the multi-channel
      output. Smoke-level; production work can do better.
    """
    if output_kind == "binary":
        return (projected > 0).astype(np.uint8)
    if output_kind == "count":
        return (projected > 0).astype(np.uint8)
    if output_kind == "continuous":
        thr = float(np.median(projected))
        return (projected > thr).astype(np.uint8)
    if output_kind == "multi_channel":
        return (projected[..., 0] > 0).astype(np.uint8)
    raise ValueError(f"unknown output_kind {output_kind!r}")


def project_stream(
    suite: ProjectionSuite, name: str, state_stream_4d: np.ndarray,
) -> np.ndarray:
    """Project every timestep of a 4D state stream through ``name``.

    Returns the projection's native shape per-frame, stacked along the
    leading axis.
    """
    spec = suite.get(name)
    sample = suite.project(name, state_stream_4d[0])
    out = np.empty((state_stream_4d.shape[0], *sample.shape), dtype=sample.dtype)
    out[0] = sample
    for t in range(1, state_stream_4d.shape[0]):
        out[t] = suite.project(name, state_stream_4d[t])
    return out


# ---------------------------------------------------------------------------
# 3. Candidate selection
# ---------------------------------------------------------------------------


@dataclass
class CandidateRef:
    """The minimal handle the HCE measurement needs."""
    candidate_id: int
    track_id: int
    peak_frame: int                    # global timestep, 0-indexed
    peak_mask: np.ndarray              # (Nx, Ny) uint8
    peak_interior: np.ndarray          # (Nx, Ny) uint8
    peak_bbox: tuple[int, int, int, int]  # rmin, cmin, rmax, cmax (incl)
    lifetime: int


def detect_candidates(
    binary_frames: np.ndarray,
    *,
    det_cfg: DetectionConfig,
    max_candidates: int,
    min_lifetime: int = 3,
) -> list[CandidateRef]:
    """Run extract_components + GreedyTracker over ``binary_frames`` and
    return the top-``max_candidates`` longest-lived tracks (each represented
    by its peak — largest-area — frame)."""
    tracker = GreedyTracker(det_cfg)
    for t, frame in enumerate(binary_frames):
        components = extract_components(frame, t, det_cfg)
        tracker.update(t, components)
    tracks = tracker.finalize()

    # Rank tracks by (lifetime, max-area-of-track), descending.
    scored = []
    for tr in tracks:
        if tr.length < min_lifetime:
            continue
        areas = tr.area_history
        if not areas: continue
        scored.append((tr.length, max(areas), tr))
    scored.sort(key=lambda x: (-x[0], -x[1]))

    cands: list[CandidateRef] = []
    for cid, (life, _, tr) in enumerate(scored[:max_candidates]):
        peak_idx_in_track = int(np.argmax(tr.area_history))
        peak_frame = tr.frames[peak_idx_in_track]
        peak_mask = tr.mask_history[peak_idx_in_track].astype(np.uint8)
        peak_interior = tr.interior_history[peak_idx_in_track].astype(np.uint8)
        if peak_interior.sum() == 0:
            peak_interior = peak_mask.copy()
        rows, cols = np.where(peak_mask)
        if rows.size == 0:
            continue
        bbox = (int(rows.min()), int(cols.min()),
                int(rows.max()), int(cols.max()))
        cands.append(CandidateRef(
            candidate_id=cid, track_id=tr.track_id,
            peak_frame=peak_frame, peak_mask=peak_mask,
            peak_interior=peak_interior, peak_bbox=bbox,
            lifetime=life,
        ))
    return cands


# ---------------------------------------------------------------------------
# 4. HCE measurement under arbitrary projection
# ---------------------------------------------------------------------------


def _flip_random_hidden_cells(
    state_4d: np.ndarray, *, mask_2d: np.ndarray,
    n_flips: int, rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of ``state_4d`` with up to ``n_flips`` cells flipped
    inside the (z, w) hyperplane at locations whose 2D mask is True.

    Used both for the candidate-local perturbation (mask_2d = candidate
    bbox) and for the far perturbation (mask_2d = ~candidate bbox).
    """
    perturbed = state_4d.copy()
    Nx, Ny, Nz, Nw = state_4d.shape
    if not mask_2d.any():
        return perturbed
    valid_xy = np.argwhere(mask_2d)
    if valid_xy.size == 0:
        return perturbed
    n = min(int(n_flips), valid_xy.shape[0] * Nz * Nw)
    pick = rng.integers(0, valid_xy.shape[0], size=n)
    zs = rng.integers(0, Nz, size=n)
    ws = rng.integers(0, Nw, size=n)
    for i in range(n):
        x, y = valid_xy[pick[i]]
        z, w = int(zs[i]), int(ws[i])
        perturbed[x, y, z, w] ^= 1
    return perturbed


def _project(suite: ProjectionSuite, name: str, state_4d: np.ndarray) -> np.ndarray:
    return suite.project(name, state_4d)


def _l1_local(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-cell |a - b| over cells where ``mask`` is True. Works for
    binary, continuous, and multi-channel projections."""
    if not mask.any():
        return 0.0
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if diff.ndim == 3:  # multi-channel: average across channels first
        diff = diff.mean(axis=-1)
    return float(diff[mask].sum() / float(mask.sum()))


def _l1_global(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=-1)
    return float(diff.mean())


def _bbox_mask(bbox: tuple[int, int, int, int], shape_2d: tuple[int, int]
               ) -> np.ndarray:
    rmin, cmin, rmax, cmax = bbox
    m = np.zeros(shape_2d, dtype=bool)
    m[rmin:rmax + 1, cmin:cmax + 1] = True
    return m


def _far_mask(bbox: tuple[int, int, int, int], shape_2d: tuple[int, int]
              ) -> np.ndarray:
    return ~_bbox_mask(bbox, shape_2d)


def _rollout_perturbed(
    rule_bs, state0: np.ndarray, n_steps: int, *, backend: str,
) -> np.ndarray:
    """Like ``run_substrate`` but only returns the final state. Used for
    a single-horizon HCE measurement."""
    ca = CA4D(shape=state0.shape, rule=rule_bs, backend=backend)
    ca.state = state0.copy()
    for _ in range(n_steps):
        ca.step()
    return ca.state.copy()


@dataclass
class CandidateMetrics:
    """Per-candidate result for one projection."""
    candidate_id: int
    track_id: int
    peak_frame: int
    lifetime: int
    HCE: float                          # mean over horizons of local divergence under hidden perturbation
    far_HCE: float
    sham_HCE: float                     # always 0.0 by construction; preserved for surface symmetry
    hidden_vs_far_delta: float          # HCE - far_HCE
    hidden_vs_sham_delta: float         # HCE - sham_HCE
    initial_projection_delta: float     # |P(perturbed) - P(unperturbed)| at t = peak_frame
    far_initial_projection_delta: float


def measure_candidate_under_projection(
    *, candidate: CandidateRef,
    state_stream_4d: np.ndarray,
    rule_bs,
    suite: ProjectionSuite, projection_name: str,
    horizons: Sequence[int],
    hce_replicates: int,
    backend: str,
    rng: np.random.Generator,
    n_flips_hidden: int = 8,
) -> CandidateMetrics:
    """Compute HCE / far_HCE / sham_HCE for one candidate under one projection."""
    Nx, Ny = state_stream_4d.shape[1:3]
    spec = suite.get(projection_name)

    state_at_peak = state_stream_4d[candidate.peak_frame]
    bbox_mask = _bbox_mask(candidate.peak_bbox, (Nx, Ny))
    far_mask = _far_mask(candidate.peak_bbox, (Nx, Ny))

    # The candidate-local mask: actual interior; that is the region where
    # divergence is measured.
    local_mask = candidate.peak_interior.astype(bool)
    if not local_mask.any():
        local_mask = candidate.peak_mask.astype(bool)

    proj_unperturbed_at_peak = _project(suite, projection_name, state_at_peak)

    # Run unperturbed forwards from peak for each horizon (max horizon
    # determines the longest rollout). We can reuse the global state
    # stream when peak_frame + horizon <= len(stream)-1.
    max_horizon = max(int(h) for h in horizons)
    avail_steps = state_stream_4d.shape[0] - 1 - candidate.peak_frame

    # Future divergence accumulators per condition.
    hce_horizon_means: list[float] = []
    far_horizon_means: list[float] = []
    sham_horizon_means: list[float] = []
    init_deltas: list[float] = []
    far_init_deltas: list[float] = []

    for _ in range(int(hce_replicates)):
        # 4a. Hidden perturbation INSIDE candidate bbox.
        s_hidden = _flip_random_hidden_cells(
            state_at_peak, mask_2d=bbox_mask,
            n_flips=n_flips_hidden, rng=rng,
        )
        proj_hidden_at_peak = _project(suite, projection_name, s_hidden)
        init_deltas.append(_l1_global(
            proj_hidden_at_peak, proj_unperturbed_at_peak,
        ))

        # 4b. Far perturbation OUTSIDE candidate bbox.
        s_far = _flip_random_hidden_cells(
            state_at_peak, mask_2d=far_mask,
            n_flips=n_flips_hidden, rng=rng,
        )
        proj_far_at_peak = _project(suite, projection_name, s_far)
        far_init_deltas.append(_l1_global(
            proj_far_at_peak, proj_unperturbed_at_peak,
        ))

        # Roll forward and project at each horizon, computing local
        # divergence for hidden / far / sham (sham == 0 by construction
        # since identity-perturbation produces identical futures).
        for h in horizons:
            h = int(h)
            if h > avail_steps:
                # Run extra steps just for this horizon.
                # (Smoke run rarely hits this; skip if uneconomical.)
                continue
            # Unperturbed future at horizon h.
            future_unperturbed = state_stream_4d[candidate.peak_frame + h]
            # Hidden-perturbed future at horizon h.
            future_hidden = _rollout_perturbed(
                rule_bs, s_hidden, h, backend=backend,
            )
            # Far-perturbed future at horizon h.
            future_far = _rollout_perturbed(
                rule_bs, s_far, h, backend=backend,
            )
            proj_unperturbed = _project(suite, projection_name, future_unperturbed)
            proj_hidden_future = _project(suite, projection_name, future_hidden)
            proj_far_future = _project(suite, projection_name, future_far)

            hce_horizon_means.append(
                _l1_local(proj_hidden_future, proj_unperturbed, local_mask),
            )
            far_horizon_means.append(
                _l1_local(proj_far_future, proj_unperturbed, local_mask),
            )
            # Sham == identity perturbation; future is identical to
            # unperturbed; divergence is zero by construction.
            sham_horizon_means.append(0.0)

    hce = float(np.mean(hce_horizon_means)) if hce_horizon_means else 0.0
    far_hce = float(np.mean(far_horizon_means)) if far_horizon_means else 0.0
    sham_hce = float(np.mean(sham_horizon_means)) if sham_horizon_means else 0.0
    return CandidateMetrics(
        candidate_id=candidate.candidate_id,
        track_id=candidate.track_id,
        peak_frame=candidate.peak_frame,
        lifetime=candidate.lifetime,
        HCE=hce,
        far_HCE=far_hce,
        sham_HCE=sham_hce,
        hidden_vs_far_delta=hce - far_hce,
        hidden_vs_sham_delta=hce - sham_hce,
        initial_projection_delta=float(np.mean(init_deltas)) if init_deltas else 0.0,
        far_initial_projection_delta=float(np.mean(far_init_deltas))
            if far_init_deltas else 0.0,
    )


# ---------------------------------------------------------------------------
# 5. Top-level (rule, seed) cell runner
# ---------------------------------------------------------------------------


def run_one_cell(
    *,
    rule_bs, rule_id: str, rule_source: str, seed: int,
    grid_shape, timesteps: int, backend: str,
    projections: Sequence[str], suite: ProjectionSuite | None,
    max_candidates: int, horizons: Sequence[int], hce_replicates: int,
    detection_config: DetectionConfig | None = None,
    initial_density: float = 0.5,
) -> dict:
    """Evaluate one (rule, seed) cell across all requested projections.

    Returns a dict keyed by projection name; each value is a dict with
    ``candidates`` (list of :class:`CandidateMetrics`), ``n_components``,
    and ``n_candidates``.
    """
    suite = suite or default_suite()
    det_cfg = detection_config or DetectionConfig()
    rng = np.random.default_rng(int(seed) ^ 0xA51C0DE)

    state0 = initial_4d_state(grid_shape, initial_density, seed=seed)
    stream = run_substrate(rule_bs, state0, timesteps, backend=backend)

    out: dict = {}
    for name in projections:
        spec = suite.get(name)
        proj_stream = project_stream(suite, name, stream)
        binary_frames = np.stack([
            binarize_for_detection(proj_stream[t], spec.output_kind)
            for t in range(proj_stream.shape[0])
        ], axis=0)
        cands = detect_candidates(
            binary_frames, det_cfg=det_cfg, max_candidates=max_candidates,
        )
        per_cand: list[CandidateMetrics] = []
        for c in cands:
            try:
                m = measure_candidate_under_projection(
                    candidate=c, state_stream_4d=stream, rule_bs=rule_bs,
                    suite=suite, projection_name=name,
                    horizons=horizons, hce_replicates=hce_replicates,
                    backend=backend, rng=rng,
                )
            except Exception as e:  # noqa: BLE001
                # Smoke-level safety net: log and continue.
                print(f"  [warn] candidate measurement failed "
                      f"({rule_id}, seed={seed}, proj={name}, "
                      f"cid={c.candidate_id}): {e!r}")
                continue
            per_cand.append(m)
        out[name] = {
            "candidates": per_cand,
            "n_candidates": len(per_cand),
            "n_components_first_frame":
                int((binary_frames[0] > 0).sum()),
            "rule_id": rule_id,
            "rule_source": rule_source,
            "seed": int(seed),
            "projection_supports_threshold_margin":
                bool(spec.threshold_margin_supported),
            "projection_output_kind": spec.output_kind,
        }
    return out

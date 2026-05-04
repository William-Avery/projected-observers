"""Phase-instrumented copy of ``discover_one_cell`` for G5B profiling.

Mirrors the structure of
:func:`observer_worlds.experiments._parallel_discovery.discover_one_cell`
exactly, but wraps each logical phase in a per-thread timer so the
profiler can attribute time to substrate / projection_stream /
candidate_detection / perturbation_construction / npz_write
breakouts. The same primitives (substrate, project_stream,
detect_candidates, make_projection_invisible_perturbation) are
called, so any timing here reflects the production code path.

This module is **not** used by the GPU runner. Only G5B's
``profile_candidate_discovery`` invokes it.
"""
from __future__ import annotations

import os as _os

# Match _parallel_discovery's BLAS pin so workers behave identically.
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_k, "1")

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from observer_worlds.experiments._followup_projection import (
    _bbox_mask,
    _far_mask,
    binarize_for_detection,
    detect_candidates,
    initial_4d_state,
    project_stream,
    run_substrate,
)
from observer_worlds.experiments._parallel_discovery import (
    CandidateScaffold,
    _safe,
)
from observer_worlds.projection import (
    default_suite,
    make_projection_invisible_perturbation,
)
from observer_worlds.utils.config import DetectionConfig


# Phases reported in the profile output.
_PHASES = (
    "substrate_rollout",
    "projection_stream",
    "candidate_detection",
    "perturbation_construction",
    "npz_write",
    "total",
)


@dataclass
class CellProfile:
    """Per-cell timing breakdown for one (rule, seed)."""
    rule_id: str
    rule_source: str
    seed: int
    per_phase_seconds: dict[str, float]               # phase -> seconds
    per_phase_per_projection: dict[str, dict[str, float]]
    n_candidates_total: int
    work_files_written: int
    payload_mb: float


def discover_one_cell_profiled(
    *,
    rule_record: dict,
    seed: int,
    cfg: dict,
    scratch_dir: str,
) -> tuple[CellProfile, dict[tuple, CandidateScaffold]]:
    """One-cell discovery with per-phase timing. Returns the profile
    and the per-cell scaffolds (the latter unused by the profiler but
    matches the production return contract for bit-equivalence)."""
    suite = default_suite()
    det_cfg = DetectionConfig()
    rule_bs = rule_record["rule"].to_bsrule()
    bl_lut, sl_lut = rule_bs.to_lookup_tables(80)
    bl_lut = bl_lut.astype(np.uint8)
    sl_lut = sl_lut.astype(np.uint8)

    rid = rule_record["rule_id"]
    rsrc = rule_record["rule_source"]

    per_phase: dict[str, float] = {p: 0.0 for p in _PHASES}
    per_phase_proj: dict[str, dict[str, float]] = {
        p: {} for p in _PHASES if p not in ("total",)
    }

    def _add(phase: str, dt: float, projection: str | None = None) -> None:
        per_phase[phase] += dt
        if projection is not None and phase in per_phase_proj:
            per_phase_proj[phase][projection] = (
                per_phase_proj[phase].get(projection, 0.0) + dt
            )

    t_total_0 = time.perf_counter()

    # ---- Phase: substrate rollout ----
    t0 = time.perf_counter()
    state0 = initial_4d_state(
        tuple(cfg["grid"]),
        float(rule_record["rule"].initial_density),
        seed=int(seed),
    )
    stream = run_substrate(
        rule_bs, state0, int(cfg["timesteps"]),
        backend=cfg["cpu_discovery_backend"],
    )
    _add("substrate_rollout", time.perf_counter() - t0)

    rng = np.random.default_rng(int(seed) ^ 0xA51C0DE)

    n_candidates_total = 0
    work_files_written = 0
    payload_bytes = 0
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    scaffolds: dict[tuple, CandidateScaffold] = {}

    for projection in cfg["projections"]:
        spec = suite.get(projection)

        # ---- Phase: projection stream + binarisation ----
        t0 = time.perf_counter()
        proj_stream = project_stream(suite, projection, stream)
        binary_frames = np.stack([
            binarize_for_detection(proj_stream[t], spec.output_kind)
            for t in range(proj_stream.shape[0])
        ], axis=0)
        _add("projection_stream", time.perf_counter() - t0, projection)

        # ---- Phase: candidate detection (CC + tracker) ----
        t0 = time.perf_counter()
        cands = detect_candidates(
            binary_frames, det_cfg=det_cfg,
            max_candidates=int(cfg["max_candidates"]),
        )
        _add("candidate_detection", time.perf_counter() - t0, projection)

        # ---- Phase: perturbation construction + per-cand scaffold ----
        states_o: list[np.ndarray] = []
        states_h: list[np.ndarray] = []
        states_f: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        cand_ids: list[int] = []
        replicates: list[int] = []
        avail_steps: list[int] = []

        t0 = time.perf_counter()
        for c in cands:
            state_at_peak = stream[c.peak_frame]
            bbox = _bbox_mask(c.peak_bbox, stream.shape[1:3])
            far = _far_mask(c.peak_bbox, stream.shape[1:3])
            local_mask = c.peak_interior.astype(bool)
            if not local_mask.any():
                local_mask = c.peak_mask.astype(bool)
            avail = stream.shape[0] - 1 - c.peak_frame

            s_h0, hidden_rep = make_projection_invisible_perturbation(
                state_at_peak, candidate_mask=bbox,
                projection_name=projection, rng=rng,
            )
            s_f0, far_rep = make_projection_invisible_perturbation(
                state_at_peak, candidate_mask=far,
                projection_name=projection, rng=rng,
            )
            accepted = bool(hidden_rep["accepted"])

            scaffolds[(rid, int(seed), projection, int(c.candidate_id))] = (
                CandidateScaffold(
                    rule_id=rid, rule_source=rsrc, seed=int(seed),
                    projection=projection,
                    candidate_id=int(c.candidate_id),
                    track_id=int(c.track_id),
                    peak_frame=int(c.peak_frame),
                    lifetime=int(c.lifetime),
                    accepted_first_replicate=accepted,
                    invalid_reason=(
                        None if accepted
                        else str(hidden_rep.get("invalid_reason"))
                    ),
                    preservation_strategy=str(hidden_rep["preservation_strategy"]),
                    initial_projection_delta=float(
                        hidden_rep["initial_projection_delta"]
                    ),
                    far_initial_projection_delta=float(
                        far_rep["initial_projection_delta"]
                    ),
                    n_flipped_hidden_first=int(hidden_rep.get("n_flipped", 0)),
                    n_flipped_far_first=int(far_rep.get("n_flipped", 0)),
                    avail_steps=int(avail),
                    hce_per_replicate_per_horizon=[],
                    far_per_replicate_per_horizon=[],
                )
            )
            n_candidates_total += 1
            if not accepted:
                continue

            states_o.append(state_at_peak)
            states_h.append(s_h0)
            states_f.append(s_f0)
            masks.append(local_mask.astype(np.uint8))
            cand_ids.append(int(c.candidate_id))
            replicates.append(0)
            avail_steps.append(int(avail))

            for r in range(1, int(cfg["hce_replicates"])):
                s_hr, _ = make_projection_invisible_perturbation(
                    state_at_peak, candidate_mask=bbox,
                    projection_name=projection, rng=rng,
                )
                s_fr, _ = make_projection_invisible_perturbation(
                    state_at_peak, candidate_mask=far,
                    projection_name=projection, rng=rng,
                )
                states_o.append(state_at_peak)
                states_h.append(s_hr)
                states_f.append(s_fr)
                masks.append(local_mask.astype(np.uint8))
                cand_ids.append(int(c.candidate_id))
                replicates.append(r)
                avail_steps.append(int(avail))
        _add("perturbation_construction", time.perf_counter() - t0, projection)

        # ---- Phase: npz write ----
        if states_o:
            t0 = time.perf_counter()
            so = np.stack(states_o, axis=0)
            sh = np.stack(states_h, axis=0)
            sf = np.stack(states_f, axis=0)
            mk = np.stack(masks, axis=0)
            cids = np.asarray(cand_ids, dtype=np.int32)
            reps = np.asarray(replicates, dtype=np.int32)
            avs = np.asarray(avail_steps, dtype=np.int32)
            n = int(so.shape[0])
            blR = np.broadcast_to(bl_lut, (n, 81)).copy()
            slR = np.broadcast_to(sl_lut, (n, 81)).copy()
            path = scratch / (
                f"profcell_{_safe(rid)}_{int(seed)}_{projection}.npz"
            )
            np.savez(
                path,
                states_orig=so, states_hidden=sh, states_far=sf,
                candidate_local_masks=mk,
                birth_luts=blR, surv_luts=slR,
                avail_steps=avs,
                candidate_ids=cids, replicates=reps,
            )
            payload_bytes += so.nbytes + sh.nbytes + sf.nbytes + mk.nbytes
            work_files_written += 1
            _add("npz_write", time.perf_counter() - t0, projection)

    per_phase["total"] = time.perf_counter() - t_total_0
    return (
        CellProfile(
            rule_id=rid,
            rule_source=rsrc,
            seed=int(seed),
            per_phase_seconds=per_phase,
            per_phase_per_projection=per_phase_proj,
            n_candidates_total=n_candidates_total,
            work_files_written=work_files_written,
            payload_mb=round(payload_bytes / (1024 ** 2), 2),
        ),
        scaffolds,
    )

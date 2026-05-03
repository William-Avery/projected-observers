"""Stage G3 — parallel CPU discovery for the GPU-aware projection-
robustness runner.

Thread-control note: this module sets ``OMP_NUM_THREADS`` and friends
to ``"1"`` at import time. Without that, OpenBLAS / MKL inside each
joblib loky worker spawn dozens of internal threads (default ≈ all
cores). With ``n_workers=30`` that produces ~720 threads on a 32-core
box and causes intermittent Windows ``access violation`` crashes in
numpy / scipy reductions (observed in three out of three G3 smoke
runs before this fix; Stage 6D first attempt was the same root cause).
We set the env vars *before* any BLAS-using import so worker
processes (which spawn fresh Python on Windows) inherit them.

Background: G2B showed CPU candidate discovery dominated the GPU
runner wall (5470 / 5868 s = 93%). Each (rule, seed) cell is
independent and can be discovered in parallel; per-cell determinism
is preserved because each worker constructs its own per-cell RNG via
``default_rng(seed ^ 0xA51C0DE)``. Joblib ``loky`` workers run the
substrate, project each frame, detect candidates, build perturbations,
and write the GPU work items to a per-(cell, projection) ``.npz``
file under ``scratch_dir``. Workers return only compact metadata —
file paths, scaffolds (no numpy arrays), cell_meta dicts.

This avoids putting multi-GB state arrays through joblib IPC: peak
per-cell payload is ~50 MB on disk and is mmap-cheap to load on
demand by the single GPU controller process.

Determinism contract:

* Per-cell results are bit-identical to single-worker discovery —
  the worker uses the same RNG seed and the same projection iteration
  order as
  :func:`observer_worlds.experiments._followup_projection.run_one_cell`.
* Across cells, the candidate-row CSV order can drift if iteration
  is not sorted; the runner's ``_finalize_metrics`` sorts by
  ``(rule_id, seed, projection, candidate_id)`` so worker count does
  not change row order.
"""
from __future__ import annotations

import os as _os

# Pin BLAS threading to 1 per worker BEFORE numpy / scipy import.
# See module docstring for the why.
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_k, "1")

import concurrent.futures as _cf
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
from observer_worlds.projection import (
    default_suite,
    make_projection_invisible_perturbation,
)
from observer_worlds.utils.config import DetectionConfig


# ---------------------------------------------------------------------------
# Scaffold + worker payload dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CandidateScaffold:
    """Per-candidate book-keeping built during discovery; finalized
    by the runner after the GPU rollout populates HCE numbers.

    Fields are plain Python so the dict is cheap to ship through joblib.
    """
    rule_id: str
    rule_source: str
    seed: int
    projection: str
    candidate_id: int
    track_id: int
    peak_frame: int
    lifetime: int
    accepted_first_replicate: bool
    invalid_reason: str | None
    preservation_strategy: str
    initial_projection_delta: float
    far_initial_projection_delta: float
    n_flipped_hidden_first: int
    n_flipped_far_first: int
    avail_steps: int
    # Filled in after GPU pass:
    hce_per_replicate_per_horizon: list[list[float]]   # [replicate][h_idx]
    far_per_replicate_per_horizon: list[list[float]]


@dataclass
class CellDiscoveryResult:
    """Worker return payload — compact, no numpy arrays."""
    rule_id: str
    rule_source: str
    seed: int
    cell_meta: dict[str, dict]                  # projection -> meta
    scaffolds: dict[tuple, CandidateScaffold]   # (rid, seed, proj, cand_id) -> scaffold
    work_files: dict[str, str]                  # projection -> npz path
    payload_mb_estimate: float


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def discover_one_cell(
    *,
    rule_record: dict,
    seed: int,
    cfg: dict,
    scratch_dir: str,
) -> CellDiscoveryResult:
    """Process one (rule, seed) cell end-to-end on CPU.

    Per-cell pipeline:
    1. Substrate rollout (single 4D stream).
    2. For each requested projection, in ``cfg["projections"]`` order:
       project + binarize + tracker + detect candidates.
    3. For each candidate, in detection order:
       construct hidden + far perturbations for replicate 0; if accepted,
       construct perturbations for replicates 1..hce_replicates-1.
    4. Save per-(cell, projection) work items to a ``.npz`` file.
    5. Return scaffolds + cell_meta + file paths (no large arrays).
    """
    suite = default_suite()
    det_cfg = DetectionConfig()
    horizons = [int(h) for h in cfg["horizons"]]
    rule_bs = rule_record["rule"].to_bsrule()
    bl_lut, sl_lut = rule_bs.to_lookup_tables(80)
    bl_lut = bl_lut.astype(np.uint8)
    sl_lut = sl_lut.astype(np.uint8)

    state0 = initial_4d_state(
        tuple(cfg["grid"]),
        float(rule_record["rule"].initial_density),
        seed=int(seed),
    )
    stream = run_substrate(
        rule_bs, state0, int(cfg["timesteps"]),
        backend=cfg["cpu_discovery_backend"],
    )

    rng = np.random.default_rng(int(seed) ^ 0xA51C0DE)

    cell_meta: dict[str, dict] = {}
    scaffolds: dict[tuple, CandidateScaffold] = {}
    work_files: dict[str, str] = {}
    payload_bytes = 0
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    rid = rule_record["rule_id"]
    rsrc = rule_record["rule_source"]

    for projection in cfg["projections"]:
        spec = suite.get(projection)
        proj_stream = project_stream(suite, projection, stream)
        binary_frames = np.stack([
            binarize_for_detection(proj_stream[t], spec.output_kind)
            for t in range(proj_stream.shape[0])
        ], axis=0)
        cands = detect_candidates(
            binary_frames, det_cfg=det_cfg,
            max_candidates=int(cfg["max_candidates"]),
        )
        cell_meta[projection] = {
            "rule_id": rid, "rule_source": rsrc,
            "seed": int(seed), "projection": projection,
            "n_candidates": len(cands),
            "projection_supports_threshold_margin":
                bool(spec.threshold_margin_supported),
            "projection_output_kind": spec.output_kind,
        }

        # Accumulators for this (cell, projection)'s npz file.
        states_o: list[np.ndarray] = []
        states_h: list[np.ndarray] = []
        states_f: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        cand_ids: list[int] = []
        replicates: list[int] = []
        avail_steps: list[int] = []

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
                    invalid_reason=(None if accepted
                                    else str(hidden_rep.get("invalid_reason"))),
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

            if not accepted:
                # CPU semantics: invalid candidate is not rolled out and
                # the per-cell RNG is NOT consumed for further replicates.
                continue

            # Replicate 0
            states_o.append(state_at_peak)
            states_h.append(s_h0)
            states_f.append(s_f0)
            masks.append(local_mask.astype(np.uint8))
            cand_ids.append(int(c.candidate_id))
            replicates.append(0)
            avail_steps.append(int(avail))

            # Replicates 1..hce_replicates-1
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

        if states_o:
            so = np.stack(states_o, axis=0)
            sh = np.stack(states_h, axis=0)
            sf = np.stack(states_f, axis=0)
            mk = np.stack(masks, axis=0)
            cids = np.asarray(cand_ids, dtype=np.int32)
            reps = np.asarray(replicates, dtype=np.int32)
            avs = np.asarray(avail_steps, dtype=np.int32)
            # Per-row LUTs (constant across this cell since same rule).
            n = int(so.shape[0])
            blR = np.broadcast_to(bl_lut, (n, 81)).copy()
            slR = np.broadcast_to(sl_lut, (n, 81)).copy()

            path = scratch / f"cell_{_safe(rid)}_{int(seed)}_{projection}.npz"
            np.savez(
                path,
                states_orig=so, states_hidden=sh, states_far=sf,
                candidate_local_masks=mk,
                birth_luts=blR, surv_luts=slR,
                avail_steps=avs,
                candidate_ids=cids, replicates=reps,
            )
            work_files[projection] = str(path)
            payload_bytes += so.nbytes + sh.nbytes + sf.nbytes + mk.nbytes

    return CellDiscoveryResult(
        rule_id=rid, rule_source=rsrc, seed=int(seed),
        cell_meta=cell_meta, scaffolds=scaffolds,
        work_files=work_files,
        payload_mb_estimate=round(payload_bytes / (1024 ** 2), 2),
    )


def _safe(s: str) -> str:
    """Filename-safe slug (rule_id contains underscores already; this
    just guards against future weirdness)."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(s))


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def parallel_discover_all_cells(
    *,
    rule_records: list[dict],
    cfg: dict,
    scratch_dir: str,
    n_workers: int,
    max_pool_restarts: int = 6,
) -> tuple[dict, dict, dict, dict]:
    """Discover candidates for every (rule, seed) cell in parallel.

    Uses :class:`concurrent.futures.ProcessPoolExecutor` + ``as_completed``
    so per-cell results are kept even when a sibling worker crashes —
    the prior joblib-loky implementation aborted the *entire* batch on
    the first ``TerminatedWorkerError``, wasting all in-flight work.

    Robustness model:

    1. Submit all pending cells to a pool of ``n_workers`` workers.
    2. Drain via ``as_completed``; collect successful results in
       ``results``. Catch per-future exceptions (e.g. propagated
       worker faults) and queue those cells for retry.
    3. On ``BrokenProcessPool`` / unhandled pool death: shut down,
       restart with the same workers, and resubmit only the cells
       that have not produced a result yet.
    4. After ``max_pool_restarts`` attempts, fall back to in-process
       serial execution for any still-pending cells. Serial can't be
       killed by sibling-worker faults.

    Returns
    -------
    cell_meta
        ``(rule_id, seed, projection) -> meta_dict``.
    scaffolds
        ``(rule_id, seed, projection, candidate_id) -> CandidateScaffold``.
    work_files_per_proj
        ``projection -> [npz_path, ...]`` (sorted).
    discovery_stats
        ``{n_cells, n_workers, n_pool_restarts, payload_mb_total}``.
    """
    cells = [(rec, seed) for rec in rule_records for seed in cfg["test_seeds"]]
    nw = max(1, int(n_workers))
    results: list[CellDiscoveryResult | None] = [None] * len(cells)

    def _serial_fill(indices: list[int]) -> None:
        for i in indices:
            rec, seed = cells[i]
            try:
                results[i] = discover_one_cell(
                    rule_record=rec, seed=int(seed),
                    cfg=cfg, scratch_dir=scratch_dir,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[discovery] cell ({rec['rule_id']}, {seed}) "
                    f"failed in serial fallback: {e!r}"
                )

    pending = list(range(len(cells)))
    n_pool_restarts = 0

    if nw <= 1 or len(pending) <= 1:
        _serial_fill(pending)
        pending = [i for i in pending if results[i] is None]
    else:
        attempt = 0
        while pending and attempt < max_pool_restarts:
            attempt += 1
            print(
                f"[discovery] pool attempt {attempt} "
                f"with workers={nw} for {len(pending)} pending cells."
            )
            this_pending = list(pending)
            try:
                with _cf.ProcessPoolExecutor(max_workers=nw) as ex:
                    fut_to_idx = {
                        ex.submit(
                            discover_one_cell,
                            rule_record=cells[i][0], seed=int(cells[i][1]),
                            cfg=cfg, scratch_dir=scratch_dir,
                        ): i
                        for i in this_pending
                    }
                    try:
                        for fut in _cf.as_completed(fut_to_idx):
                            i = fut_to_idx[fut]
                            try:
                                results[i] = fut.result()
                            except (
                                _cf.process.BrokenProcessPool,
                                _cf.CancelledError,
                            ) as e:
                                # Pool died; remaining futures will also
                                # raise. Break out and let the next
                                # attempt rebuild the pool.
                                print(
                                    f"[discovery] pool died while "
                                    f"draining (cell idx={i}): "
                                    f"{type(e).__name__}; will rebuild."
                                )
                                break
                            except Exception as e:  # noqa: BLE001
                                rec, seed = cells[i]
                                print(
                                    f"[discovery] cell ({rec['rule_id']}, "
                                    f"{seed}) raised in worker: {e!r}; "
                                    f"will retry."
                                )
                    except _cf.process.BrokenProcessPool as e:
                        print(
                            f"[discovery] BrokenProcessPool draining: "
                            f"{e!r}; will rebuild pool."
                        )
            except _cf.process.BrokenProcessPool as e:
                # Pool's __exit__ raised — odd, but tolerable.
                print(
                    f"[discovery] BrokenProcessPool on shutdown: {e!r}; "
                    f"continuing."
                )
            n_pool_restarts += 1
            pending = [i for i in pending if results[i] is None]
        if pending:
            print(
                f"[discovery] serial fallback for {len(pending)} "
                f"cells that never completed in parallel after "
                f"{n_pool_restarts} pool attempt(s)."
            )
            _serial_fill(pending)
            pending = [i for i in pending if results[i] is None]
    if pending:
        raise RuntimeError(
            f"Discovery failed for {len(pending)} cells even in serial: "
            f"{[(cells[i][0]['rule_id'], cells[i][1]) for i in pending]}"
        )

    cell_meta: dict[tuple, dict] = {}
    scaffolds: dict[tuple, CandidateScaffold] = {}
    work_files_per_proj: dict[str, list[str]] = {}
    payload_total_mb = 0.0
    for r in results:
        for proj, meta in r.cell_meta.items():
            cell_meta[(r.rule_id, r.seed, proj)] = meta
        scaffolds.update(r.scaffolds)
        for proj, path in r.work_files.items():
            work_files_per_proj.setdefault(proj, []).append(path)
        payload_total_mb += r.payload_mb_estimate

    # Stable file order so GPU controller iteration is deterministic.
    for proj in work_files_per_proj:
        work_files_per_proj[proj].sort()

    discovery_stats = {
        "n_cells": len(cells),
        "n_workers": nw,
        "n_pool_restarts": n_pool_restarts,
        "payload_mb_total": round(payload_total_mb, 1),
    }
    return cell_meta, scaffolds, work_files_per_proj, discovery_stats

"""M6C — taxonomy of hidden organization properties that drive HCE.

Combines M6B's intervention-and-rollout machinery with hidden-feature
extraction at the snapshot, plus a small ablation battery designed to
separate which kind of hidden property matters.

Each output row corresponds to one (rule, seed, candidate, snapshot)
and contains:

  * candidate metadata (id, area, lifetime, observer_score)
  * all hidden features at the snapshot (column-aggregate + temporal)
  * the headline HCE quantities at one or more horizons:
      - future_projection_divergence under hidden_invisible
      - hidden_vs_sham_delta
      - local_future_divergence under hidden_invisible
      - local_future_divergence under hidden_invisible_far
      - hidden_vs_far_delta
      - per-ablation HCE outcomes

Joining lets us correlate features ↔ HCE and run grouped-CV regression
to ask "which hidden features predict large HCE?"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from observer_worlds.analysis.hidden_features import (
    HIDDEN_FEATURE_NAMES,
    candidate_hidden_features,
    temporal_hidden_features,
)
from observer_worlds.detection import GreedyTracker
from observer_worlds.experiments._m6b_interventions import (
    apply_far_hidden_intervention,
    apply_fiber_replacement_intervention,
    apply_one_time_scramble_intervention,
    apply_sham_intervention,
)
from observer_worlds.experiments._pipeline import (
    compute_full_metrics,
    detect_and_track,
    simulate_4d_to_zarr,
)
from observer_worlds.experiments._m4b_sweep import hidden_shuffle_mutator
from observer_worlds.metrics import score_persistence
from observer_worlds.metrics.causality_score import (
    apply_flip_intervention,
    apply_hidden_shuffle_intervention,
)
from observer_worlds.search import FractionalRule
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils import RunConfig, seeded_rng
from observer_worlds.utils.config import (
    DetectionConfig,
    OutputConfig,
    ProjectionConfig,
    WorldConfig,
)
from observer_worlds.worlds import CA4D, BSRule, project


# ---------------------------------------------------------------------------
# Ablation interventions
# ---------------------------------------------------------------------------


# The ablation set the spec asks for. Every ablation preserves projection
# at t=0 (mean-threshold theta=0.5) by construction.
ABLATION_TYPES: tuple[str, ...] = (
    "random_hidden_shuffle",         # M6B hidden_invisible_local
    "count_preserving_shuffle",      # alias of the above (named for clarity)
    "spatial_destroying_scramble",   # M6B one_time_scramble_local
    "fiber_replacement",             # M6B fiber_replacement_local
    "temporal_history_swap",         # NEW — swap fiber with same column k snapshots ago
    "sham",
)


def apply_temporal_history_swap_intervention(
    state_4d_now: np.ndarray,
    state_4d_past: np.ndarray | None,
    interior_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Swap each candidate (x,y) fiber with the same column from a
    prior snapshot of the same simulation.

    If ``state_4d_past`` is None (no history available) or has different
    shape, fall back to identity (returns ``state_4d_now.copy()``). The
    caller is responsible for providing a past snapshot whose per-column
    counts match (so projection is preserved); we **enforce** it here
    by only swapping columns where past and current have the same
    active count.
    """
    out = state_4d_now.copy()
    if state_4d_past is None or state_4d_past.shape != state_4d_now.shape:
        return out
    coords = np.argwhere(interior_mask)
    n_swapped = 0
    for x, y in coords:
        n_now = int(state_4d_now[x, y].sum())
        n_past = int(state_4d_past[x, y].sum())
        if n_now == n_past:
            out[x, y] = state_4d_past[x, y].copy()
            n_swapped += 1
    return out


# ---------------------------------------------------------------------------
# Per-row dataclass (matches the spec column list)
# ---------------------------------------------------------------------------


@dataclass
class M6CRow:
    rule_id: str
    rule_source: str
    seed: int
    candidate_id: int
    snapshot_t: int
    horizon: int

    # Candidate metadata.
    candidate_area: float = 0.0
    candidate_lifetime: int = 0
    observer_score: float | None = None

    # Hidden features (flattened — keys match HIDDEN_FEATURE_NAMES).
    features: dict = field(default_factory=dict)

    # HCE outcomes at this horizon.
    future_div_hidden_invisible: float = 0.0
    local_div_hidden_invisible: float = 0.0
    future_div_sham: float = 0.0
    local_div_far_hidden: float = 0.0
    hidden_vs_sham_delta: float = 0.0
    hidden_vs_far_delta: float = 0.0
    future_div_visible: float = 0.0
    hidden_vs_visible_ratio: float = 0.0
    survival_delta: float = 0.0
    HCE: float = 0.0

    # Ablation outcomes (per ABLATION_TYPES, future_div under each).
    ablation_future_div: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_2d(state: np.ndarray, theta: float = 0.5) -> np.ndarray:
    return project(state, method="mean_threshold", theta=theta)


def _l1_full(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum() / a.size)


def _l1_local(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any(): return 0.0
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff[mask].sum() / mask.sum())


def _rollout_proj(state, rule, n_steps, *, backend, theta=0.5):
    Nx, Ny = state.shape[0], state.shape[1]
    ca = CA4D(shape=state.shape, rule=rule, backend=backend)
    ca.state = state.copy()
    out = np.empty((n_steps, Nx, Ny), dtype=np.uint8)
    for t in range(n_steps):
        ca.step()
        out[t] = _project_2d(ca.state, theta)
    return out


# ---------------------------------------------------------------------------
# Per-candidate runner
# ---------------------------------------------------------------------------


def measure_candidate(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    rule_id: str,
    rule_source: str,
    seed: int,
    candidate_id: int,
    snapshot_t: int,
    candidate_area: float,
    candidate_lifetime: int,
    observer_score: float | None,
    horizons: list[int],
    n_replicates: int,
    backend: str,
    rng_seed: int,
    history_snapshots: list[tuple[int, np.ndarray]] | None = None,
) -> list[M6CRow]:
    """For one candidate at one snapshot, compute hidden features, run
    M6B-style HCE measurements at each horizon, and run the ablation
    battery.

    Returns one M6CRow per horizon. Ablation_future_div is filled per
    ABLATION_TYPES at the *median* horizon.
    """
    if not interior_mask.any():
        return []

    # Hidden features at the snapshot.
    feats = candidate_hidden_features(snapshot_4d, interior_mask)
    # Temporal features (if history available).
    if history_snapshots:
        snaps = [s for _, s in history_snapshots] + [snapshot_4d]
        times = [t for t, _ in history_snapshots] + [snapshot_t]
        tfeats = temporal_hidden_features(snaps, interior_mask, snapshot_times=times)
    else:
        tfeats = {"hidden_temporal_persistence": 0.0,
                  "hidden_temporal_volatility": 0.0,
                  "n_snapshots_used": 0}
    feats = {**feats, **tfeats}

    parent_rng = np.random.default_rng(rng_seed)
    H_max = max(horizons)

    # Unperturbed rollout (used as reference).
    frames_orig = _rollout_proj(snapshot_4d, rule, H_max, backend=backend)

    # Per replicate: hidden_invisible, sham, far, visible.
    hi_full = np.zeros((n_replicates, H_max))
    hi_local = np.zeros((n_replicates, H_max))
    sham_full = np.zeros((n_replicates, H_max))
    far_local = np.zeros((n_replicates, H_max))
    vis_full = np.zeros((n_replicates, H_max))
    surv_orig = np.zeros((n_replicates, H_max))
    surv_int = np.zeros((n_replicates, H_max))
    for rep in range(n_replicates):
        rng_h = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))
        rng_s = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))
        rng_far = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))
        rng_vis = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))

        s_hi = apply_hidden_shuffle_intervention(snapshot_4d, interior_mask, rng_h)
        n_flips = int(((s_hi != snapshot_4d) & interior_mask[:,:,None,None]).sum())
        s_sham = apply_sham_intervention(snapshot_4d, interior_mask, rng_s)
        s_far, _ = apply_far_hidden_intervention(snapshot_4d, interior_mask, rng_far)
        Nz, Nw = snapshot_4d.shape[2], snapshot_4d.shape[3]
        flip_frac = n_flips / max(int(interior_mask.sum()) * Nz * Nw, 1)
        s_vis = apply_flip_intervention(snapshot_4d, interior_mask, flip_frac, rng_vis)

        f_hi = _rollout_proj(s_hi, rule, H_max, backend=backend)
        f_sh = _rollout_proj(s_sham, rule, H_max, backend=backend)
        f_far = _rollout_proj(s_far, rule, H_max, backend=backend)
        f_vis = _rollout_proj(s_vis, rule, H_max, backend=backend)

        for h_idx in range(H_max):
            hi_full[rep, h_idx] = _l1_full(frames_orig[h_idx], f_hi[h_idx])
            hi_local[rep, h_idx] = _l1_local(frames_orig[h_idx], f_hi[h_idx], interior_mask)
            sham_full[rep, h_idx] = _l1_full(frames_orig[h_idx], f_sh[h_idx])
            far_local[rep, h_idx] = _l1_local(frames_orig[h_idx], f_far[h_idx], interior_mask)
            vis_full[rep, h_idx] = _l1_full(frames_orig[h_idx], f_vis[h_idx])
            surv_orig[rep, h_idx] = int((frames_orig[h_idx].astype(bool) & interior_mask).sum())
            surv_int[rep, h_idx] = int((f_hi[h_idx].astype(bool) & interior_mask).sum())

    # Build per-horizon rows.
    rows: list[M6CRow] = []
    median_h = horizons[len(horizons) // 2]
    for h in horizons:
        idx = h - 1
        future = float(hi_full[:, idx].mean())
        local = float(hi_local[:, idx].mean())
        sham_v = float(sham_full[:, idx].mean())
        far_v = float(far_local[:, idx].mean())
        vis_v = float(vis_full[:, idx].mean())
        ratio = future / vis_v if vis_v > 1e-12 else 0.0
        surv_d = float(surv_int[:, idx].mean() - surv_orig[:, idx].mean())

        ablations: dict[str, float] = {}
        if h == median_h:
            ablations = _run_ablations(
                snapshot_4d=snapshot_4d, rule=rule, interior_mask=interior_mask,
                horizon=h, n_replicates=n_replicates, backend=backend,
                rng=np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1))),
                history_snapshots=history_snapshots,
                frames_orig=frames_orig,
            )

        rows.append(M6CRow(
            rule_id=rule_id, rule_source=rule_source, seed=seed,
            candidate_id=candidate_id, snapshot_t=snapshot_t, horizon=h,
            candidate_area=candidate_area, candidate_lifetime=candidate_lifetime,
            observer_score=observer_score,
            features=feats,
            future_div_hidden_invisible=future,
            local_div_hidden_invisible=local,
            future_div_sham=sham_v,
            local_div_far_hidden=far_v,
            hidden_vs_sham_delta=future - sham_v,
            hidden_vs_far_delta=local - far_v,
            future_div_visible=vis_v,
            hidden_vs_visible_ratio=ratio,
            survival_delta=surv_d,
            HCE=future,
            ablation_future_div=ablations,
        ))
    return rows


# ---------------------------------------------------------------------------
# Ablation battery
# ---------------------------------------------------------------------------


def _run_ablations(
    *,
    snapshot_4d, rule, interior_mask, horizon, n_replicates, backend, rng,
    history_snapshots, frames_orig,
) -> dict[str, float]:
    """Run each ablation type and return mean future_div at the horizon."""
    out = {}
    sub_seeds = rng.integers(0, 2**63 - 1, size=len(ABLATION_TYPES) * n_replicates)
    si = 0

    def _rollout_and_div(perturbed_state):
        f = _rollout_proj(perturbed_state, rule, horizon, backend=backend)
        return _l1_full(frames_orig[horizon - 1], f[-1])

    past_state = history_snapshots[-1][1] if history_snapshots else None

    for kind in ABLATION_TYPES:
        divs = []
        for _ in range(n_replicates):
            rng_k = np.random.default_rng(int(sub_seeds[si])); si += 1
            if kind in ("random_hidden_shuffle", "count_preserving_shuffle"):
                s = apply_hidden_shuffle_intervention(snapshot_4d, interior_mask, rng_k)
            elif kind == "spatial_destroying_scramble":
                s = apply_one_time_scramble_intervention(snapshot_4d, interior_mask, rng_k)
            elif kind == "fiber_replacement":
                s = apply_fiber_replacement_intervention(snapshot_4d, interior_mask, rng_k)
            elif kind == "temporal_history_swap":
                s = apply_temporal_history_swap_intervention(
                    snapshot_4d, past_state, interior_mask, rng_k
                )
            elif kind == "sham":
                s = apply_sham_intervention(snapshot_4d, interior_mask, rng_k)
            else:
                continue
            divs.append(_rollout_and_div(s))
        out[kind] = float(np.mean(divs)) if divs else 0.0
    return out


# ---------------------------------------------------------------------------
# Per-(rule, seed) runner
# ---------------------------------------------------------------------------


def _build_run_config(rule, *, grid_shape, timesteps, backend, seed, label):
    bs = rule.to_bsrule()
    return RunConfig(
        world=WorldConfig(
            nx=grid_shape[0], ny=grid_shape[1], nz=grid_shape[2], nw=grid_shape[3],
            timesteps=timesteps, initial_density=rule.initial_density,
            rule_birth=tuple(int(x) for x in bs.birth),
            rule_survival=tuple(int(x) for x in bs.survival),
            backend=backend,
        ),
        projection=ProjectionConfig(method="mean_threshold", theta=0.5),
        detection=DetectionConfig(),
        output=OutputConfig(save_4d_snapshots=True,
                            snapshot_interval=max(1, timesteps // 6)),
        seed=seed, label=label,
    )


def _select_top_candidates(tracks, observer_scores, max_k):
    """Top-K by observer_score, requires non-empty interior at some snapshot."""
    score_by_id = {o.track_id: o for o in observer_scores}
    ranked = sorted(observer_scores, key=lambda o: -o.combined)
    track_by_id = {t.track_id: t for t in tracks}
    out = []
    for o in ranked:
        if len(out) >= max_k: break
        tr = track_by_id.get(o.track_id)
        if tr is None: continue
        out.append((tr, float(o.combined)))
    return out


def run_taxonomy_for_rule_seed(
    *,
    rule: FractionalRule, rule_id: str, rule_source: str,
    seed: int, grid_shape, timesteps,
    max_candidates: int, horizons: list[int], n_replicates: int,
    backend: str, workdir, progress=None,
) -> list[M6CRow]:
    cfg = _build_run_config(
        rule, grid_shape=grid_shape, timesteps=timesteps,
        backend=backend, seed=seed,
        label=f"m6c_{rule_id}_seed{seed}",
    )
    rundir = workdir / f"{rule_id}_seed{seed}"
    rundir.mkdir(parents=True, exist_ok=True)
    store = ZarrRunStore(
        rundir, timesteps=timesteps,
        shape_2d=(grid_shape[0], grid_shape[1]),
        save_4d_snapshots=True, shape_4d=grid_shape,
    )
    rng_init = seeded_rng(seed)
    simulate_4d_to_zarr(cfg, store, rng_init)
    frames = store.read_frames_2d()
    tracks = detect_and_track(cfg, frames)
    candidates = score_persistence(
        tracks, grid_shape=(grid_shape[0], grid_shape[1]), config=cfg.detection
    )
    observer_scores, _ = compute_full_metrics(
        cfg, tracks, candidates, store, rollout_steps=4, world_kind="4d",
    )
    snap_times = store.list_snapshots()
    rule_4d_bs = BSRule(birth=cfg.world.rule_birth, survival=cfg.world.rule_survival)
    selected = _select_top_candidates(tracks, observer_scores, max_candidates)

    rows: list[M6CRow] = []
    for cand_idx, (tr, obs_score) in enumerate(selected):
        # Find the latest snapshot in track lifetime.
        snap_t = None
        for st in reversed(snap_times):
            if tr.birth_frame <= st <= tr.last_frame:
                snap_t = st; break
        if snap_t is None: continue
        # Get interior mask at the snapshot (M6's lenient fallback).
        if snap_t in tr.frames:
            i = tr.frames.index(snap_t)
        else:
            nearest = min(tr.frames, key=lambda f: abs(f - snap_t))
            i = tr.frames.index(nearest)
        interior = tr.interior_history[i]
        if not interior.any(): interior = tr.mask_history[i]
        if not interior.any(): continue

        snapshot_4d = store.read_snapshot_4d(snap_t)
        # History: previous snapshot (if available).
        history = []
        prev_t = None
        for st in snap_times:
            if st < snap_t and tr.birth_frame <= st:
                prev_t = st
        if prev_t is not None:
            history = [(prev_t, store.read_snapshot_4d(prev_t))]

        if progress:
            progress(f"    cand {cand_idx+1}/{len(selected)} track={tr.track_id} "
                     f"snap_t={snap_t} interior={int(interior.sum())}")

        cand_rows = measure_candidate(
            snapshot_4d=snapshot_4d, rule=rule_4d_bs,
            interior_mask=interior,
            rule_id=rule_id, rule_source=rule_source, seed=seed,
            candidate_id=tr.track_id, snapshot_t=snap_t,
            candidate_area=float(np.mean(tr.area_history)) if tr.area_history else 0.0,
            candidate_lifetime=int(tr.age),
            observer_score=obs_score,
            horizons=horizons, n_replicates=n_replicates,
            backend=backend,
            rng_seed=seed * 1009 + tr.track_id * 31 + cand_idx,
            history_snapshots=history,
        )
        rows.extend(cand_rows)
    return rows


def _run_taxonomy_for_parallel(
    item: tuple[FractionalRule, str, str, int], shared: dict
) -> list[M6CRow]:
    rule, rule_id, rule_source, seed = item
    return run_taxonomy_for_rule_seed(
        rule=rule, rule_id=rule_id, rule_source=rule_source,
        seed=seed,
        grid_shape=shared["grid_shape"],
        timesteps=shared["timesteps"],
        max_candidates=shared["max_candidates"],
        horizons=shared["horizons"],
        n_replicates=shared["n_replicates"],
        backend=shared["backend"],
        workdir=shared["workdir"],
        progress=None,
    )


class _M6CParallelTask:
    """Picklable task dispatcher for parallel_sweep.

    joblib loky pickles the callable by reference (qualified name) and
    re-imports it in the worker; nested closures aren't picklable. This
    class binds the per-sweep ``shared`` dict to a top-level callable.
    """

    def __init__(self, shared: dict) -> None:
        self.shared = shared

    def __call__(
        self, item: tuple[FractionalRule, str, str, int]
    ) -> list[M6CRow]:
        return _run_taxonomy_for_parallel(item, self.shared)


def run_m6c_taxonomy(
    *,
    rules: list[tuple[FractionalRule, str, str]],
    seeds: list[int],
    grid_shape, timesteps,
    max_candidates: int,
    horizons: list[int], n_replicates: int,
    backend: str, workdir,
    progress: Callable[[str], None] | None = None,
    n_workers: int | None = None,
) -> list[M6CRow]:
    """Driver. Returns a flat list of M6CRow across all
    (rule, seed, candidate, horizon).

    Work is parallelized over (rule, seed) tuples via ``parallel_sweep``.
    Default ``n_workers`` is ``cpu_count - 2``; on machines with <= 2
    cores this resolves to 1 and the serial path is used. Pass
    ``n_workers=1`` explicitly to force serial.
    """
    from observer_worlds.parallel import parallel_sweep

    items: list[tuple[FractionalRule, str, str, int]] = [
        (rule, rule_id, rule_source, seed)
        for rule, rule_id, rule_source in rules
        for seed in seeds
    ]

    shared = {
        "grid_shape": grid_shape,
        "timesteps": timesteps,
        "max_candidates": max_candidates,
        "horizons": horizons,
        "n_replicates": n_replicates,
        "backend": backend,
        # Normalize to Path so workers (and the serial path) can do
        # ``workdir / f"{rule_id}_seed{seed}"`` even if a string was
        # passed in.
        "workdir": Path(workdir),
    }

    # Forward live per-task lines to stderr when a progress callback is
    # provided -- otherwise long sweeps go silent for hours.
    verbose = 10 if progress is not None else 0

    t0 = time.time()
    flat_results = parallel_sweep(
        items,
        _M6CParallelTask(shared),
        n_workers=n_workers,
        progress=progress,
        verbose=verbose,
    )
    elapsed = time.time() - t0
    if progress is not None:
        progress(
            f"  m6c sweep wall time {elapsed:.0f}s "
            f"({len(items)} runs across {len(rules)} rules x "
            f"{len(seeds)} seeds)"
        )

    rows: list[M6CRow] = [row for sublist in flat_results for row in sublist]
    return rows

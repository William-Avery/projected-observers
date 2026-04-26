"""Shared experiment pipeline.

Each experiment script (4D, 2D baseline, shuffled-hidden baseline) plugs
into a common pipeline:

    simulate -> detect+track -> persistence-filter -> M2 metric suite ->
    write CSVs/plots/GIF/summary.

The simulate stage is experiment-specific; everything downstream is
shared.  The shuffled-hidden baseline reuses :func:`simulate_4d_to_zarr`
with a state-mutator hook that permutes z,w fibers before projection.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np

from observer_worlds.analysis import plot_area_vs_time, plot_lifetimes, write_projected_gif
from observer_worlds.detection import GreedyTracker, classify_boundary, extract_components
from observer_worlds.metrics import (
    DEFAULT_WEIGHTS,
    collect_raw_scores,
    compute_causality_score,
    compute_memory_score,
    compute_observer_scores,
    compute_resilience_score,
    compute_selfhood_score,
    compute_time_score,
    extract_track_features,
    score_persistence,
)
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils import RunConfig
from observer_worlds.worlds import CA2D, CA4D, BSRule, project


# ---------------------------------------------------------------------------
# Simulation stages
# ---------------------------------------------------------------------------


# A state-mutator transforms a 4D state in place (or by returning a new array)
# *after* the CA step but *before* projection.  Used by the shuffled-hidden
# baseline to permute z,w fibers.  Signature: (state, t, rng) -> state.
StateMutator = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def simulate_4d_to_zarr(
    cfg: RunConfig,
    store: ZarrRunStore,
    rng: np.random.Generator,
    *,
    state_mutator: StateMutator | None = None,
) -> None:
    """Run the 4D CA, project each step, write to the run store.

    If ``state_mutator`` is provided, it is called on the state *before*
    projection at each timestep; the mutated state is also what gets saved
    as a 4D snapshot, so that downstream causality/resilience operate on
    the same dynamics that produced the projected frames.
    """
    rule = BSRule(birth=cfg.world.rule_birth, survival=cfg.world.rule_survival)
    ca = CA4D(shape=cfg.world.shape, rule=rule, backend=cfg.world.backend)
    ca.initialize_random(density=cfg.world.initial_density, rng=rng)

    T = cfg.world.timesteps
    snapshot_interval = cfg.output.snapshot_interval
    save_snapshots = cfg.output.save_4d_snapshots

    def _project_and_store(t: int) -> np.ndarray:
        state = ca.state
        if state_mutator is not None:
            mutated = state_mutator(state.copy(), t, rng)
            # Inject the mutated state back into the CA so subsequent steps
            # evolve from it.  Without this, the mean-threshold projection
            # makes the shuffled-hidden baseline a no-op (the projection
            # only depends on per-column active counts, which the shuffle
            # preserves -- so without feedback into ca.state, the projected
            # frames would be byte-identical to the coherent run).
            ca.state = mutated
            state = mutated
        Y = project(state, method=cfg.projection.method, theta=cfg.projection.theta)
        store.write_frame_2d(t, Y.astype(np.uint8))
        if save_snapshots and snapshot_interval > 0 and (t == 0 or t % snapshot_interval == 0):
            store.write_snapshot_4d(t, state)
        return Y

    Y0 = _project_and_store(0)
    t0 = time.time()
    for t in range(1, T):
        ca.step()
        Y = _project_and_store(t)
        if t % max(1, T // 10) == 0:
            print(f"  step {t}/{T} active_2d_frac={Y.mean():.3f}")
    print(f"  simulation done in {time.time()-t0:.1f}s")


def simulate_2d_to_zarr(
    cfg: RunConfig,
    rule_2d: BSRule,
    store: ZarrRunStore,
    rng: np.random.Generator,
) -> None:
    """Run a 2D CA (e.g. Conway's Life) and write each frame to the store.

    No 4D state exists, so causality and resilience scores are skipped
    downstream (handled by ``compute_full_metrics(world_kind="2d")``).
    """
    ca = CA2D(shape=(cfg.world.nx, cfg.world.ny), rule=rule_2d)
    ca.initialize_random(density=cfg.world.initial_density, rng=rng)

    T = cfg.world.timesteps
    store.write_frame_2d(0, ca.state.astype(np.uint8))
    t0 = time.time()
    for t in range(1, T):
        ca.step()
        store.write_frame_2d(t, ca.state.astype(np.uint8))
        if t % max(1, T // 10) == 0:
            frac = float(ca.state.mean())
            print(f"  step {t}/{T} active_2d_frac={frac:.3f}")
    print(f"  simulation done in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Detection + tracking
# ---------------------------------------------------------------------------


def detect_and_track(cfg: RunConfig, frames_2d: np.ndarray) -> list:
    tracker = GreedyTracker(config=cfg.detection)
    T = frames_2d.shape[0]
    t0 = time.time()
    for t in range(T):
        components = extract_components(frames_2d[t], frame_idx=t, config=cfg.detection)
        tracker.update(t, components)
    tracks = tracker.finalize()
    print(f"  tracking done in {time.time()-t0:.1f}s -> {len(tracks)} total tracks")
    return tracks


# ---------------------------------------------------------------------------
# Metric stage
# ---------------------------------------------------------------------------


def _pick_snapshot_for_track(track, available_snapshots: list[int]) -> int | None:
    if not available_snapshots:
        return None
    for t in reversed(available_snapshots):
        if track.birth_frame <= t <= track.last_frame:
            return t
    return None


def _mask_at_frame(track, frame_idx: int, kind: str) -> np.ndarray | None:
    try:
        i = track.frames.index(frame_idx)
    except ValueError:
        return None
    if kind == "interior":
        return track.interior_history[i]
    if kind == "boundary":
        return track.boundary_history[i]
    if kind == "env":
        return track.env_history[i]
    raise ValueError(kind)


def compute_full_metrics(
    cfg: RunConfig,
    tracks: list,
    candidates: list,
    store: ZarrRunStore,
    *,
    rollout_steps: int = 10,
    world_kind: str = "4d",
):
    """Compute the full M2 metric suite for every observer-candidate.

    ``world_kind == "4d"``: causality + resilience are computed when 4D
    snapshots are available.

    ``world_kind == "2d"``: causality + resilience are skipped (the 2D
    baseline has no 4D fibers to perturb in the same sense; intervention
    semantics differ across dimensionalities).  Candidates still get
    time / memory / selfhood / observer_score.

    Returns ``(observer_scores, per_candidate_results)`` -- the second is a
    dict keyed by track_id with the raw result objects (for CSV writing).
    """
    candidate_ids = {c.track_id for c in candidates if c.is_candidate}
    track_by_id = {t.track_id: t for t in tracks}
    candidate_tracks = [track_by_id[i] for i in candidate_ids if i in track_by_id]

    available_snapshots: list[int] = []
    if world_kind == "4d" and cfg.output.save_4d_snapshots:
        available_snapshots = store.list_snapshots()
        if available_snapshots:
            print(
                f"  found {len(available_snapshots)} 4D snapshots — will compute "
                f"causality + resilience where applicable"
            )
        else:
            print("  no 4D snapshots available — causality + resilience will be skipped")
    elif world_kind == "2d":
        print("  world_kind=2d — causality + resilience skipped (no 4D fibers)")

    rule = (
        BSRule(birth=cfg.world.rule_birth, survival=cfg.world.rule_survival)
        if world_kind == "4d"
        else None
    )

    raw_per_track = []
    per_candidate: dict[int, dict] = {}
    for tr in candidate_tracks:
        feats = extract_track_features(tr)
        time_res = compute_time_score(feats, seed=cfg.seed)
        mem_res = compute_memory_score(feats, seed=cfg.seed)
        self_res = compute_selfhood_score(feats, seed=cfg.seed)
        bnd_res = classify_boundary(feats)

        causal_res = None
        resil_res = None
        if world_kind == "4d":
            snap_t = _pick_snapshot_for_track(tr, available_snapshots)
            if snap_t is not None and rule is not None:
                interior = _mask_at_frame(tr, snap_t, "interior")
                boundary = _mask_at_frame(tr, snap_t, "boundary")
                env = _mask_at_frame(tr, snap_t, "env")
                if interior is None:
                    nearest = min(tr.frames, key=lambda f: abs(f - snap_t))
                    interior = _mask_at_frame(tr, nearest, "interior")
                    boundary = _mask_at_frame(tr, nearest, "boundary")
                    env = _mask_at_frame(tr, nearest, "env")
                try:
                    snapshot_4d = store.read_snapshot_4d(snap_t)
                    if (
                        interior is not None
                        and boundary is not None
                        and env is not None
                        and bool(interior.any())
                        and bool(boundary.any())
                        and bool(env.any())
                    ):
                        causal_res = compute_causality_score(
                            snapshot_4d, rule, interior, boundary, env,
                            n_steps=rollout_steps,
                            backend=cfg.world.backend,
                            seed=cfg.seed,
                            track_id=tr.track_id,
                        )
                        resil_res = compute_resilience_score(
                            snapshot_4d, rule, interior,
                            n_steps=rollout_steps,
                            backend=cfg.world.backend,
                            seed=cfg.seed,
                            track_id=tr.track_id,
                        )
                except Exception as e:
                    print(f"  causality/resilience failed for track {tr.track_id}: {e}")

        per_candidate[tr.track_id] = {
            "time": time_res,
            "memory": mem_res,
            "selfhood": self_res,
            "boundary": bnd_res,
            "causality": causal_res,
            "resilience": resil_res,
        }
        raw_per_track.append(
            collect_raw_scores(
                track_id=tr.track_id,
                time=time_res,
                memory=mem_res,
                selfhood=self_res,
                causality=causal_res,
                resilience=resil_res,
            )
        )

    observer_scores = compute_observer_scores(raw_per_track)
    return observer_scores, per_candidate


# ---------------------------------------------------------------------------
# Output stage
# ---------------------------------------------------------------------------


def write_observer_scores_csv(
    out_path: Path, observer_scores: list, per_candidate: dict
) -> None:
    columns = [
        "track_id", "combined", "n_components_used",
        "time_raw", "memory_raw", "selfhood_raw", "causality_raw", "resilience_raw",
        "time_z", "memory_z", "selfhood_z", "causality_z", "resilience_z",
        "boundary_predictability", "extra_env_given_boundary", "persistence",
        "boundedness", "sensory_fraction", "active_fraction",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for o in observer_scores:
            extras = per_candidate.get(o.track_id, {})
            self_res = extras.get("selfhood")
            bnd_res = extras.get("boundary")
            w.writerow([
                o.track_id, f"{o.combined:.6f}", o.n_components_used,
                "" if o.time_raw is None else f"{o.time_raw:.6f}",
                "" if o.memory_raw is None else f"{o.memory_raw:.6f}",
                "" if o.selfhood_raw is None else f"{o.selfhood_raw:.6f}",
                "" if o.causality_raw is None else f"{o.causality_raw:.6f}",
                "" if o.resilience_raw is None else f"{o.resilience_raw:.6f}",
                "" if o.time_normalized is None else f"{o.time_normalized:.6f}",
                "" if o.memory_normalized is None else f"{o.memory_normalized:.6f}",
                "" if o.selfhood_normalized is None else f"{o.selfhood_normalized:.6f}",
                "" if o.causality_normalized is None else f"{o.causality_normalized:.6f}",
                "" if o.resilience_normalized is None else f"{o.resilience_normalized:.6f}",
                f"{self_res.boundary_predictability:.4f}" if (self_res and self_res.valid) else "",
                f"{self_res.extra_env_given_boundary:.4f}" if (self_res and self_res.valid) else "",
                f"{self_res.persistence:.4f}" if (self_res and self_res.valid) else "",
                f"{self_res.boundedness:.4f}" if (self_res and self_res.valid) else "",
                f"{bnd_res.sensory_fraction:.4f}" if (bnd_res and bnd_res.valid) else "",
                f"{bnd_res.active_fraction:.4f}" if (bnd_res and bnd_res.valid) else "",
            ])


def build_summary(
    cfg: RunConfig,
    tracks: list,
    candidates: list,
    observer_scores: list,
    run_dir: Path,
    *,
    world_kind: str,
    extra_lines: list[str] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Run summary — {cfg.label} ({world_kind})")
    lines.append("")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- World kind: **{world_kind}**")
    lines.append(f"- Grid: {cfg.world.shape}")
    lines.append(f"- Timesteps: {cfg.world.timesteps}")
    lines.append(f"- Seed: {cfg.seed}")
    if world_kind in ("4d", "shuffled_4d"):
        lines.append(f"- 4D rule: B={cfg.world.rule_birth}, S={cfg.world.rule_survival}")
        lines.append(f"- Projection: {cfg.projection.method} (theta={cfg.projection.theta})")
        lines.append(
            f"- 4D snapshots: {'on' if cfg.output.save_4d_snapshots else 'off'} "
            f"(interval={cfg.output.snapshot_interval})"
        )
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("")
    lines.append("## Tracking")
    lines.append("")
    lines.append(f"- Total tracks: {len(tracks)}")
    if tracks:
        ages = [t.age for t in tracks]
        lengths = [t.length for t in tracks]
        lines.append(f"- Mean age: {np.mean(ages):.1f}, max age: {int(np.max(ages))}")
        lines.append(f"- Mean length: {np.mean(lengths):.1f}, max length: {int(np.max(lengths))}")
    lines.append("")
    lines.append("## Persistence filter")
    lines.append("")
    n_cand = sum(1 for c in candidates if c.is_candidate)
    lines.append(f"- Candidates: {n_cand} / {len(candidates)}")
    if n_cand == 0:
        lines.append("")
        lines.append(
            "**No observer-candidates found.** Expected for the default heuristic "
            "rule. Use rule search (M4) to find rules that produce persistent "
            "structures."
        )
    lines.append("")
    lines.append("## Observer-likeness scores (M2)")
    lines.append("")
    if not observer_scores:
        lines.append("No candidates to score.")
    else:
        n_with_time = sum(1 for o in observer_scores if o.time_raw is not None)
        n_with_memory = sum(1 for o in observer_scores if o.memory_raw is not None)
        n_with_selfhood = sum(1 for o in observer_scores if o.selfhood_raw is not None)
        n_with_causal = sum(1 for o in observer_scores if o.causality_raw is not None)
        n_with_resil = sum(1 for o in observer_scores if o.resilience_raw is not None)
        lines.append(
            f"- Component coverage: time={n_with_time}, memory={n_with_memory}, "
            f"selfhood={n_with_selfhood}, causality={n_with_causal}, "
            f"resilience={n_with_resil}"
        )
        lines.append(f"- Default weights: {DEFAULT_WEIGHTS}")
        lines.append("")
        top = sorted(observer_scores, key=lambda o: -o.combined)[:10]
        lines.append("Top candidates by combined observer_score:")
        lines.append("")
        lines.append("| track_id | combined | time | memory | selfhood | causality | resilience |")
        lines.append("|---|---|---|---|---|---|---|")
        for o in top:
            def fmt(x):
                return "—" if x is None else f"{x:+.3f}"
            lines.append(
                f"| {o.track_id} | {o.combined:+.3f} | {fmt(o.time_raw)} | "
                f"{fmt(o.memory_raw)} | {fmt(o.selfhood_raw)} | "
                f"{fmt(o.causality_raw)} | {fmt(o.resilience_raw)} |"
            )
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append("- `config.json`")
    lines.append("- `data/states.zarr/frames_2d`")
    if world_kind in ("4d", "shuffled_4d"):
        lines.append("- `data/states.zarr/snapshots_4d/` (if enabled)")
    lines.append("- `data/tracks.csv`, `data/candidates.csv`, `data/observer_scores.csv`")
    lines.append("- `frames/projected_world.gif`")
    lines.append("- `plots/lifetimes.png`, `plots/area_vs_time.png`")
    return "\n".join(lines)


def write_plots_and_gif(
    cfg: RunConfig,
    store: ZarrRunStore,
    frames: np.ndarray,
    tracks: list,
    candidates: list,
) -> None:
    plot_lifetimes(
        tracks,
        store.plots_dir / "lifetimes.png",
        min_age_threshold=cfg.detection.min_age,
    )
    plot_area_vs_time(tracks, store.plots_dir / "area_vs_time.png")
    if cfg.output.save_gif:
        candidate_ids = {c.track_id for c in candidates if c.is_candidate}
        write_projected_gif(
            frames,
            tracks,
            out_path=store.frames_dir / "projected_world.gif",
            fps=cfg.output.gif_fps,
            max_frames=cfg.output.gif_max_frames,
            candidate_track_ids=candidate_ids,
        )

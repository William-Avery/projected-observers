"""M7 — HCE-guided rule search with anti-artifact safeguards.

Combines observer-likeness, hidden causal dependence, candidate
persistence, and explicit penalties for projection-threshold artifacts,
global chaos, and fragile candidates into a single composite fitness.

Crucial design choice: M7 does **not** optimize raw HCE alone. The
spec lists eight failure modes M6/M6C surfaced — projection-threshold
artifacts, global chaos that swamps local effects, candidates that die
from any perturbation, swarms of degenerate near-zero-area tracks.
The composite fitness penalizes each so the search can't trivially
exploit them.

Two search loops are exposed:
  * ``random_search_hce`` — uniform sampling of fractional rules
  * ``evolutionary_search_hce`` — (μ+λ) loop with Gaussian mutation,
    optionally seeded from an M4A or M4C leaderboard
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from observer_worlds.analysis.hidden_features import candidate_hidden_features
from observer_worlds.detection import GreedyTracker
from observer_worlds.experiments._m6b_interventions import (
    apply_far_hidden_intervention,
    apply_sham_intervention,
)
from observer_worlds.experiments._pipeline import (
    compute_full_metrics,
    detect_and_track,
    simulate_4d_to_zarr,
)
from observer_worlds.metrics import score_persistence
from observer_worlds.metrics.causality_score import (
    apply_hidden_shuffle_intervention,
)
from observer_worlds.search import FractionalRule, sample_random_fractional_rule
from observer_worlds.search.observer_evolve import mutate_fractional_rule
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import (
    DetectionConfig,
    OutputConfig,
    ProjectionConfig,
    RunConfig,
    WorldConfig,
)
from observer_worlds.worlds import CA4D, BSRule, project


# ---------------------------------------------------------------------------
# Default fitness weights
# ---------------------------------------------------------------------------


# Documented in the M7 spec. Easy to override per-run.
DEFAULT_M7_WEIGHTS: dict[str, float] = {
    "obs":        1.0,
    "hce":        2.0,
    "local":      2.0,
    "life":       0.75,
    "recovery":   0.75,
    "thresh":     1.5,
    "global":     1.0,
    "fragile":    1.0,
    "degenerate": 1.0,
}


# Quantity scales used to map raw quantities to roughly [0, 1] before
# applying the weights. Calibrated from M4C / M6B / M6C real runs.
DEFAULT_M7_SCALES: dict[str, float] = {
    "observer_score":       0.20,   # M4C top rules ~0.15
    "hidden_vs_sham_delta": 0.05,   # M6C all-cand mean +0.040
    "hidden_vs_far_delta":  0.25,   # M6C local-far mean +0.24
    "lifetime":             100.0,  # ~T/3 for typical T=300
    "recovery":             1.0,    # already in [0,1]
}


@dataclass
class M7Fitness:
    """All quantities that go into the composite fitness, plus the score."""

    rule: FractionalRule
    fitness: float
    n_seeds: int

    # Raw means across seeds × candidates (NOT normalized).
    mean_observer_score: float = 0.0
    mean_hidden_vs_sham_delta: float = 0.0
    mean_hidden_vs_far_delta: float = 0.0
    mean_local_hidden_effect: float = 0.0   # alias for hidden_vs_far on local div
    mean_candidate_lifetime: float = 0.0
    mean_recovery: float = 0.0

    # Penalty quantities (all in [0, 1] — higher = worse).
    mean_near_threshold_fraction: float = 0.0
    mean_excess_global_divergence: float = 0.0
    mean_fragility_penalty: float = 0.0
    mean_degenerate_candidate_penalty: float = 0.0
    mean_initial_projection_delta: float = 0.0  # should be ~0 by construction

    n_candidates_total: int = 0
    aborted_seeds: int = 0
    sim_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Cheap per-(rule, seed) HCE estimator
# ---------------------------------------------------------------------------


def _project_2d(state, theta=0.5):
    return project(state, method="mean_threshold", theta=theta)


def _l1_full(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum() / a.size)


def _l1_local(a, b, mask):
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
                            snapshot_interval=max(1, timesteps // 5)),
        seed=seed, label=label,
    )


@dataclass
class _PerSeedAggregate:
    """Per-(rule, seed) intermediate aggregate fed into the composite fitness."""
    n_candidates: int = 0
    observer_scores: list = field(default_factory=list)
    lifetimes: list = field(default_factory=list)
    hidden_vs_sham_deltas: list = field(default_factory=list)
    hidden_vs_far_deltas: list = field(default_factory=list)
    near_threshold_fracs: list = field(default_factory=list)
    excess_global_divs: list = field(default_factory=list)
    fragilities: list = field(default_factory=list)
    initial_proj_deltas: list = field(default_factory=list)
    n_degenerate: int = 0
    n_total_candidates_pre_filter: int = 0


def _evaluate_rule_seed_cheap(
    *,
    rule: FractionalRule,
    seed: int,
    grid_shape: tuple[int, int, int, int],
    timesteps: int,
    max_candidates: int,
    horizons: list[int],
    n_replicates: int,
    backend: str,
    workdir,
) -> _PerSeedAggregate:
    """Cheap HCE-aware per-(rule, seed) evaluation.

    For top-K observer-candidates, run paired rollouts under sham,
    hidden_invisible, and hidden_invisible_far (skipping the visible
    and one_time/fiber controls used in M6B/M6C — those are recovered
    in holdout validation). Returns a tidy aggregate.
    """
    cfg = _build_run_config(
        rule, grid_shape=grid_shape, timesteps=timesteps,
        backend=backend, seed=seed,
        label=f"m7eval_seed{seed}",
    )
    rundir = workdir / f"seed{seed}_{int(time.time()*1000)%100000}"
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

    agg = _PerSeedAggregate()
    agg.n_total_candidates_pre_filter = sum(1 for c in candidates if c.is_candidate)
    H_max = max(horizons)
    median_h = horizons[len(horizons) // 2]
    score_by_id = {o.track_id: o for o in observer_scores}
    track_by_id = {t.track_id: t for t in tracks}

    # Rank by observer score descending.
    ranked = sorted(observer_scores, key=lambda o: -o.combined)
    n_eligible = 0
    for obs in ranked:
        if n_eligible >= max_candidates:
            break
        tr = track_by_id.get(obs.track_id)
        if tr is None: continue
        snap_t = None
        for st in reversed(snap_times):
            if tr.birth_frame <= st <= tr.last_frame:
                snap_t = st; break
        if snap_t is None: continue
        if snap_t in tr.frames: i = tr.frames.index(snap_t)
        else:
            nearest = min(tr.frames, key=lambda f: abs(f - snap_t))
            i = tr.frames.index(nearest)
        interior = tr.interior_history[i]
        if not interior.any(): interior = tr.mask_history[i]
        if not interior.any():
            continue

        # Degenerate candidate penalty: very small interior.
        if int(interior.sum()) < 4:
            agg.n_degenerate += 1
            continue
        n_eligible += 1

        snapshot_4d = store.read_snapshot_4d(snap_t)
        feats = candidate_hidden_features(snapshot_4d, interior)
        near_thresh = float(feats.get("near_threshold_fraction", 0.0))

        # Run replicates: sham, hidden_invisible, far.
        rng = np.random.default_rng(seed * 1009 + tr.track_id * 31)
        frames_orig = _rollout_proj(snapshot_4d, rule_4d_bs, H_max, backend=backend)

        rep_full_hi, rep_local_hi = [], []
        rep_full_sham = []
        rep_local_far = []
        rep_init_hi = []
        rep_survival = []
        surv_orig_at_h = int((frames_orig[median_h - 1].astype(bool) & interior).sum())

        for rep in range(n_replicates):
            r1 = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            r2 = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            r3 = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            s_hi = apply_hidden_shuffle_intervention(snapshot_4d, interior, r1)
            s_sham = apply_sham_intervention(snapshot_4d, interior, r2)
            s_far, _ = apply_far_hidden_intervention(snapshot_4d, interior, r3)
            f_hi = _rollout_proj(s_hi, rule_4d_bs, H_max, backend=backend)
            f_sh = _rollout_proj(s_sham, rule_4d_bs, H_max, backend=backend)
            f_far = _rollout_proj(s_far, rule_4d_bs, H_max, backend=backend)
            init_d = _l1_full(_project_2d(snapshot_4d), _project_2d(s_hi))
            rep_init_hi.append(init_d)
            for h in horizons:
                idx = h - 1
                rep_full_hi.append(_l1_full(frames_orig[idx], f_hi[idx]))
                rep_local_hi.append(_l1_local(frames_orig[idx], f_hi[idx], interior))
                rep_full_sham.append(_l1_full(frames_orig[idx], f_sh[idx]))
                rep_local_far.append(_l1_local(frames_orig[idx], f_far[idx], interior))
            # Survival at median horizon.
            surv_int = int((f_hi[median_h - 1].astype(bool) & interior).sum())
            rep_survival.append(surv_int)

        full_hi = float(np.mean(rep_full_hi))
        local_hi = float(np.mean(rep_local_hi))
        full_sham = float(np.mean(rep_full_sham))
        local_far = float(np.mean(rep_local_far))
        vs_sham = full_hi - full_sham
        vs_far = local_hi - local_far
        excess_global = max(0.0, full_hi - local_hi)
        fragility = 0.0 if surv_orig_at_h == 0 else max(
            0.0, 1.0 - float(np.mean(rep_survival)) / surv_orig_at_h
        )

        agg.n_candidates += 1
        agg.observer_scores.append(float(obs.combined))
        agg.lifetimes.append(int(tr.age))
        agg.hidden_vs_sham_deltas.append(vs_sham)
        agg.hidden_vs_far_deltas.append(vs_far)
        agg.near_threshold_fracs.append(near_thresh)
        agg.excess_global_divs.append(excess_global)
        agg.fragilities.append(fragility)
        agg.initial_proj_deltas.append(float(np.mean(rep_init_hi)))
    return agg


# ---------------------------------------------------------------------------
# Composite fitness
# ---------------------------------------------------------------------------


def _normalize(value: float, scale: float) -> float:
    """Linear rescaling, clipped to a moderate range so a single huge
    candidate can't dominate the fitness."""
    if scale <= 0: return 0.0
    return float(np.clip(value / scale, -3.0, 3.0))


def _aggregate_to_fitness(
    rule: FractionalRule,
    aggs: list[_PerSeedAggregate],
    *,
    weights: dict[str, float],
    scales: dict[str, float],
    sim_time_seconds: float = 0.0,
) -> M7Fitness:
    """Combine per-seed aggregates into one M7Fitness."""
    if not aggs:
        return M7Fitness(rule=rule, fitness=0.0, n_seeds=0)

    def _flat(attr):
        out = []
        for a in aggs:
            out.extend(getattr(a, attr))
        return np.array(out, dtype=np.float64) if out else np.array([0.0])

    obs = _flat("observer_scores")
    life = _flat("lifetimes")
    vs_sham = _flat("hidden_vs_sham_deltas")
    vs_far = _flat("hidden_vs_far_deltas")
    near_th = _flat("near_threshold_fracs")
    excess_g = _flat("excess_global_divs")
    frag = _flat("fragilities")
    init_d = _flat("initial_proj_deltas")

    n_total = sum(a.n_candidates for a in aggs)
    n_pre = sum(a.n_total_candidates_pre_filter for a in aggs)
    n_deg = sum(a.n_degenerate for a in aggs)
    deg_penalty = (
        float(n_deg / max(n_pre, 1)) if n_pre > 0 else 0.0
    )

    # Recovery proxy: 1 - mean(fragility).
    recovery = max(0.0, 1.0 - float(frag.mean()))

    # Composite.
    score = (
        weights["obs"]        * _normalize(float(obs.mean()), scales["observer_score"])
        + weights["hce"]      * _normalize(float(vs_sham.mean()), scales["hidden_vs_sham_delta"])
        + weights["local"]    * _normalize(float(vs_far.mean()), scales["hidden_vs_far_delta"])
        + weights["life"]     * _normalize(float(life.mean()), scales["lifetime"])
        + weights["recovery"] * recovery
        - weights["thresh"]   * float(near_th.mean())
        - weights["global"]   * _normalize(float(excess_g.mean()), scales["hidden_vs_sham_delta"])
        - weights["fragile"]  * float(frag.mean())
        - weights["degenerate"] * deg_penalty
    )
    # Hard penalty: any non-zero initial projection delta indicates a bug;
    # drag fitness toward zero.
    score -= 5.0 * float(init_d.mean())

    aborted = sum(1 for a in aggs if a.n_candidates == 0)

    return M7Fitness(
        rule=rule, fitness=float(score), n_seeds=len(aggs),
        mean_observer_score=float(obs.mean()),
        mean_hidden_vs_sham_delta=float(vs_sham.mean()),
        mean_hidden_vs_far_delta=float(vs_far.mean()),
        mean_local_hidden_effect=float(vs_far.mean()),
        mean_candidate_lifetime=float(life.mean()),
        mean_recovery=recovery,
        mean_near_threshold_fraction=float(near_th.mean()),
        mean_excess_global_divergence=float(excess_g.mean()),
        mean_fragility_penalty=float(frag.mean()),
        mean_degenerate_candidate_penalty=deg_penalty,
        mean_initial_projection_delta=float(init_d.mean()),
        n_candidates_total=n_total,
        aborted_seeds=aborted,
        sim_time_seconds=sim_time_seconds,
    )


# ---------------------------------------------------------------------------
# Public: evaluate one rule on a list of seeds
# ---------------------------------------------------------------------------


def evaluate_rule_m7(
    rule: FractionalRule, *,
    seeds: list[int],
    grid_shape: tuple[int, int, int, int] = (32, 32, 4, 4),
    timesteps: int = 100,
    max_candidates: int = 5,
    horizons: list[int] | None = None,
    n_replicates: int = 1,
    backend: str = "numpy",
    weights: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    workdir = None,
) -> M7Fitness:
    """Cheap M7 fitness evaluation across seeds. Use during evolution."""
    horizons = horizons or [10, 20]
    weights = {**DEFAULT_M7_WEIGHTS, **(weights or {})}
    scales = {**DEFAULT_M7_SCALES, **(scales or {})}
    if workdir is None:
        import tempfile
        from pathlib import Path
        workdir = Path(tempfile.mkdtemp(prefix="m7eval_"))

    t0 = time.time()
    aggs = []
    for s in seeds:
        try:
            agg = _evaluate_rule_seed_cheap(
                rule=rule, seed=s, grid_shape=grid_shape,
                timesteps=timesteps, max_candidates=max_candidates,
                horizons=horizons, n_replicates=n_replicates,
                backend=backend, workdir=workdir,
            )
            aggs.append(agg)
        except Exception as e:
            # Truly broken rule (e.g. die-off) — record empty aggregate.
            aggs.append(_PerSeedAggregate())
    return _aggregate_to_fitness(rule, aggs, weights=weights, scales=scales,
                                sim_time_seconds=time.time() - t0)


# ---------------------------------------------------------------------------
# Random + evolutionary search
# ---------------------------------------------------------------------------


def random_search_hce(
    *,
    n_rules: int,
    train_seeds: list[int],
    grid_shape, timesteps,
    max_candidates: int, horizons: list[int], n_replicates: int,
    backend: str,
    sampler_seed: int = 0,
    weights: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    workdir = None,
    progress: Callable[[str], None] | None = None,
) -> list[M7Fitness]:
    rng = seeded_rng(sampler_seed)
    out = []
    t0 = time.time()
    for i in range(n_rules):
        rule = sample_random_fractional_rule(rng)
        fit = evaluate_rule_m7(
            rule, seeds=train_seeds, grid_shape=grid_shape,
            timesteps=timesteps, max_candidates=max_candidates,
            horizons=horizons, n_replicates=n_replicates,
            backend=backend, weights=weights, scales=scales,
            workdir=workdir,
        )
        out.append(fit)
        if progress:
            best = max(out, key=lambda f: f.fitness)
            progress(f"  [{i+1}/{n_rules}] elapsed={time.time()-t0:.0f}s "
                     f"fitness={fit.fitness:+.3f} best={best.fitness:+.3f}")
    out.sort(key=lambda f: -f.fitness)
    return out


def evolutionary_search_hce(
    *,
    n_generations: int,
    mu: int, lam: int,
    train_seeds: list[int],
    grid_shape, timesteps,
    max_candidates: int, horizons: list[int], n_replicates: int,
    backend: str,
    sampler_seed: int = 0,
    weights: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    sigmas: dict[str, float] | None = None,
    initial_population: list[FractionalRule] | None = None,
    workdir = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[M7Fitness], list[dict]]:
    """(μ+λ) loop. Mutates fractional rules with Gaussian noise on each
    of the 5 floats, evaluates, truncates to top-mu by fitness."""
    rng = seeded_rng(sampler_seed)

    if initial_population is not None:
        pop_rules = list(initial_population)[:mu]
        while len(pop_rules) < mu:
            pop_rules.append(sample_random_fractional_rule(rng))
    else:
        pop_rules = [sample_random_fractional_rule(rng) for _ in range(mu)]

    def _eval(r): return evaluate_rule_m7(
        r, seeds=train_seeds, grid_shape=grid_shape,
        timesteps=timesteps, max_candidates=max_candidates,
        horizons=horizons, n_replicates=n_replicates,
        backend=backend, weights=weights, scales=scales,
        workdir=workdir,
    )
    population = [_eval(r) for r in pop_rules]
    history = []

    def _record(g):
        fits = [r.fitness for r in population]
        h = {
            "generation": g,
            "best_fitness": float(max(fits)),
            "mean_fitness": float(np.mean(fits)),
            "median_fitness": float(np.median(fits)),
            "population_size": len(population),
            "best_observer_score": float(max(r.mean_observer_score for r in population)),
            "best_hidden_vs_sham": float(max(r.mean_hidden_vs_sham_delta for r in population)),
            "best_hidden_vs_far": float(max(r.mean_hidden_vs_far_delta for r in population)),
            "best_near_threshold_fraction": float(min(r.mean_near_threshold_fraction
                                                     for r in population)),
        }
        history.append(h)

    _record(0)
    if progress:
        progress(f"  [gen 0] best={history[-1]['best_fitness']:+.3f}  "
                 f"mean={history[-1]['mean_fitness']:+.3f}")

    for gen in range(1, n_generations + 1):
        parent_idx = rng.integers(0, len(population), size=lam)
        offspring_rules = [
            mutate_fractional_rule(population[i].rule, rng, sigmas=sigmas)
            for i in parent_idx
        ]
        offspring = [_eval(r) for r in offspring_rules]
        combined = population + offspring
        combined.sort(key=lambda r: -r.fitness)
        population = combined[:mu]
        _record(gen)
        if progress:
            progress(f"  [gen {gen}] best={history[-1]['best_fitness']:+.3f}  "
                     f"mean={history[-1]['mean_fitness']:+.3f}  "
                     f"obs={history[-1]['best_observer_score']:+.3f}  "
                     f"hce={history[-1]['best_hidden_vs_sham']:+.4f}")
    population.sort(key=lambda r: -r.fitness)
    return population, history

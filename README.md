# observer_worlds

A research framework that tests whether higher-dimensional dynamics can produce
lower-dimensional projected structures with **functional signatures of
observerhood** — persistent boundary, predictive memory, temporal asymmetry,
intervention-sensitive causal structure, self-maintenance under perturbation.

> **This project does not claim to simulate consciousness.** It operationalizes
> observer-likeness as a battery of measurable functional properties of
> tracked persistent structures in a projected lower-dimensional world.

— *William Lawrence Avery, 2026. MIT-licensed.*

> **New here?** Read [TUTORIAL.md](TUTORIAL.md) for a runnable
> walkthrough — every CLI, in narrative order, with time budgets and
> expected outputs.

## Objectives

The framework operationalizes a sequence of increasingly stringent
empirical questions about higher-dimensional dynamics and projected
observers:

1. **Build the substrate** — a 4D binary cellular automaton, a 4D-to-2D
   projection, and connected-component tracking on the 2D projected
   frames so that "observer-candidates" can be defined and persisted
   over time. *(M1)*
2. **Operationalize observerhood as five orthogonal functional
   metrics** — temporal asymmetry, predictive memory, Markov-blanket
   selfhood, intervention-sensitive causality, perturbation
   resilience — and combine them into a single track-count-aware
   `observer_score`. *(M2)*
3. **Establish controls** — Conway's Life as a 2D baseline, and a
   hidden-shuffled 4D variant (z,w fibers permuted but the simulation
   evolves coherently in x,y) as the per-condition negative control.
   *(M3)*
4. **Discover non-trivial 4D rules** via a viability search and an
   observer-fitness-guided rule search. *(M4A, M4C)*
5. **Test the hypothesis statistically** — paired sweeps over many
   (rule, seed) pairs with bootstrap CIs, permutation tests, effect
   sizes, win rates. *(M4B)*
6. **Hold the test out** — re-run on held-out seeds and against an
   *equally optimized* 2D rule baseline to remove optimization-effort
   confounds. *(M4D)*
7. **Look for an effect 2D cannot have by construction** — Hidden
   Causal Dependence (M6): perturbations that are invisible in the 2D
   projection at t=0 but produce measurable downstream divergence.
   This is the first quantity in the framework that is identically
   zero in 2D systems by construction.

## Key findings

| Question | Result |
|---|---|
| Do 4D-projected candidates with M4A viability rules score higher than 2D Life on `observer_score`? | **No** — not significant at N=25 (p ≥ 0.07 on normalized metrics). |
| Do 4D rules **selected for observer-fitness** beat *fixed* 2D Life on normalized metrics? | **Yes** — M4B-on-M4C: `score_per_track` p=0.003, Cliff's δ=+0.76, 88% win rate. *Replicates on held-out seeds (M4D Pass A).* |
| Do those same 4D rules beat an **equally optimized** 2D rule? | **No** — M4D Pass B: optimized 2D wins on `score_per_track` (p=0.030, 32% coh win rate). The "4D advantage" was *optimization-effort*, not dimensionality. |
| Do coherent 4D candidates show **hidden causal dependence** (HCE) — downstream divergence under perturbations invisible to the 2D projection at t=0 — that hidden-shuffled-4D candidates lack? | **Yes (suggestive, N=8 candidates)** — coherent mean HCE = 0.064, shuffled mean HCE = 0.000, paired diff +0.064 with 95% bootstrap CI [+0.036, +0.092] excluding zero, coh wins 7/0. **The first signature unique to coherent 4D dynamics that the framework has surfaced.** Sign-test p=0.07 (small-N artifact); replication at scale is the obvious next step. |

In one sentence: **on the generic observer-likeness battery, optimized 2D
rules match or beat 4D — but on a dimension-specific causal probe that
2D systems cannot exhibit by construction (M6 Hidden Causal Effect),
coherent 4D shows a positive, directional advantage over its hidden-
shuffled control.**

## The hypothesis

Let

- `X_tau ∈ {0,1}^{Nx × Ny × Nz × Nw}` be a 4D binary cellular automaton
- `Y_tau(x,y) = 1 if mean_{z,w}(X_tau(x,y,z,w)) > theta else 0` be its projection
  to a 2D world

The central question is: **can `Y_tau` contain persistent connected structures
that score highly on observer-likeness metrics, and do they score higher than
matched 2D-native structures and matched hidden-shuffled controls?**

The strongest positive result would be a tracked structure in `Y_tau` that:

1. persists for many timesteps,
2. is spatially bounded,
3. has internal state variation,
4. has a measurable boundary,
5. carries information about past sensory states,
6. predicts future sensory states better than chance,
7. reacts differently to self / boundary / environment / hidden-fiber
   perturbations,
8. recovers from mild perturbation,
9. has a higher `observer_score` than matched 2D-baseline structures.

## Current evidence status

The central hypothesis evolved across milestones. **For generic
observer_score, the original directional claim does not survive a fair
2D-baseline comparison.** **For a hidden-dimensional causal effect that
2D systems cannot have by construction, coherent 4D shows a
measurable, directional advantage over hidden-shuffled-4D.**

| Experiment | Comparison | Verdict |
|---|---|---|
| M3 (M4A rules) | coh 4D vs 2D Life, normalized | not significant |
| M4B (M4A rules) | coh 4D vs 2D Life, normalized | not significant (p ≥ 0.07) |
| M4B (M4C-optimized rules) | coh 4D vs **fixed** 2D Life, normalized | **significant** (p = 0.003 on score_per_track) |
| M4D Pass A (held-out seeds) | coh 4D vs **fixed** 2D Life, normalized | **replicates**: significant (p = 0.0005, 96% win) |
| **M4D Pass B (held-out seeds)** | **coh 4D vs optimized 2D, normalized** | **optimized 2D wins** (p = 0.030 on score_per_track) |
| **M6 (Hidden Causal Dependence)** | **coh-4D HCE vs shuffled-4D HCE** | **coh wins**: mean diff +0.064, 95% CI [+0.036, +0.092], coh 7/0 wins |

**Headline finding (generic observer_score)**: when the 2D baseline is
also observer-fitness-optimized (matched compute budget, same fitness,
same metric suite), the 4D advantage **disappears** and slightly
reverses. The previous positive result against fixed Conway's Life was
an artifact of comparing optimized 4D rules against an unoptimized 2D
rule.

**Headline finding (M6 — hidden causal dependence)**: every M6
candidate is subjected to a `hidden_invisible` perturbation that
permutes z,w cells in the candidate's interior. By construction,
mean-threshold projection at t=0 is byte-identical (count preserved).
Any downstream projected divergence is therefore **causally attributable
to hidden-dimensional structure** — a quantity 2D systems can't have.
Coherent-4D candidates have **mean HCE = 0.064** (88% positive,
HCE/visible-control ratio = 0.99). Hidden-shuffled-4D candidates have
**mean HCE = 0.000** (the framework correctly detects "no hidden
structure to disrupt"). Coherent beats shuffled with a 95% bootstrap CI
excluding zero. **This is the first signature unique to coherent 4D
dynamics that the framework has surfaced.**

**What is established**:
- Both M4A viability rules and M4C-optimized rules produce projected
  structures that survive the persistence filter and yield finite
  observer scores under the full M2 metric suite.
- The shuffled-4D no-op bug discovered in M4A is fixed and
  regression-tested.
- Track-count-resistant metrics (`score_per_track`,
  `lifetime_weighted_mean_score`) reverse the apparent verdict on
  extreme-score metrics, which are confounded by the fact that
  shuffled-4D often produces 2-3× more candidates than coherent-4D.
- Held-out seeds (M4D Pass A) confirm the M4B-on-M4C finding *against
  fixed Life*. So the optimization wasn't overfitting to the training
  seed set.
- **Coherent 4D has a hidden causal effect (M6) that shuffled-4D
  doesn't** — perturbing only the hidden (z,w) arrangement, while
  preserving the 2D projection at t=0, produces measurable downstream
  projected divergence. This effect is by-construction impossible in
  2D systems and is observed only in coherent (not shuffled) 4D
  dynamics.

**What is not established**:
- That coherent hidden-dimensional structure beats shuffled 4D dynamics
  on the **generic observer_score**: directional positive across runs
  but never reaches significance with N=25.
- That 4D dimensionality per se yields any observer-likeness advantage
  beyond what an equally-optimized 2D rule class delivers, on the
  generic observer_score.
- The M6 hidden causal effect at scale: N=8 candidates with bootstrap
  CI excluding zero is suggestive but small-N. Replication on more
  rules and more candidates is the next step.

**Main scientific question for follow-on work**: *Is there any
constraint-satisfying rule family or metric formulation under which
coherent 4D dynamics genuinely beat both the shuffled-4D control AND an
equally-optimized 2D baseline?* The current toolchain — M4A viability,
M4C observer search, M4B paired sweep, M4D held-out validation with
optimized 2D baseline — supports asking that question repeatedly with
new rule families, new fitness modes, or new candidate definitions.

---

## Status

**M1 + M2 + M3 + M4A + M4B + M4C + M4D + M5 + M6 are complete.** The framework now implements:

- 4D Moore-r1 CA (numpy reference + numba kernel) with periodic boundaries
- 4D → 2D mean-threshold projection
- Connected-component extraction (interior / boundary / environment shells,
  including active-cell counts in each shell)
- Greedy IoU + centroid tracker
- Persistence-based observer-candidate filter
- **Per-track feature extraction** (`TrackFeatures`)
- **Five observer-likeness scores:**
  - `time_score` — forward vs backward predictive error (Ridge + KFold)
  - `memory_score` — does I_t add predictive power for S_{t+k} beyond S_t?
  - `selfhood_score` — Markov-blanket-style boundary mediation
    (boundary_predictability − extra_env_given_boundary), plus persistence
    and boundedness
  - `causality_score` — paired forward rollouts under flip / boundary /
    environment / hidden-shuffle interventions; computed when 4D snapshots
    are saved
  - `resilience_score` — interior-perturbation rollouts measuring survival /
    area recovery / centroid continuity / shape similarity
- **`observer_score`** — z-normalized weighted combination of the five
- **Sensory/active boundary classification** via directional information flow
- Zarr storage (projected frames + 4D checkpoints + snapshot reader)
- GIF visualization with track-ID overlays
- Reproducible runs via seeded RNG and JSON-serialized configs
- **Three matched experiment scripts** that share a common pipeline:
  - `run_4d_projection` — the 4D-to-2D experiment
  - `run_2d_baseline` — Conway's Life (or any 2D B/S rule) on a flat 2D grid
  - `run_shuffled_hidden_baseline` — same 4D dynamics, but with z,w fibers
    permuted before each projection (preserves per-column counts, destroys
    coherent hidden structure)
- **Cross-run analysis** (`analysis/summarize_results.py`): loads N runs,
  groups by world_kind, produces `observer_score_histogram.png`,
  `score_vs_age.png`, `baseline_comparison.png`, and a markdown summary table.
- **M4A — viability search** (`search/`, `experiments/search_viable_4d_rules.py`):
  - **Fractional totalistic rules** (`FractionalRule`): birth/survival as
    continuous threshold ranges over neighbor *fraction*; smooth, easily
    sampled, reduces to a `BSRule` for the existing 4D engine.
  - **Multi-component viability score**: `extinction_penalty`,
    `saturation_penalty`, `frozen_world_penalty` (penalties);
    `target_activity_score`, `temporal_change_score`,
    `persistent_component_score`, `boundedness_score`, `diversity_score`
    (rewards). Composite score = weighted sum (penalties heavy).
  - **Multi-seed evaluation**: each rule scored across S=3 seeds.
  - **Random search** over the documented sample ranges; ranked CSV +
    JSON leaderboard; per-rule artifact directory for top-K with
    `config.json` (re-runnable in the 4D experiment), `video.gif`,
    `activity_trace.png`, `components_over_time.png`,
    `viability_report.json`.

Full intervention experiments (M5) and observer-metric search on top of
viable rules (M4B) are scoped for follow-on milestones. See the plan file.

> **Honest caveat (the M4A motivation).** With the heuristic default 4D
> rule, both 4D and shuffled-4D runs produce zero observer-candidates.
> The original M3 hypothesis — "observer_score(4D) > observer_score(2D)"
> — is therefore **not directly testable**: there's nothing in the 4D
> conditions to compare against. M4A addresses this by first searching
> for 4D rules with non-trivial dynamics (the **viability** layer).
> Observer-likeness metrics should only run on rules that pass viability.

## Reformulated hypothesis (M4A onwards)

The hypothesis test now has two stages:

1. **Viability search** (M4A — random search over fractional rules):
   discover 4D rules that produce projected dynamics with persistent
   bounded structures.
2. **Observer-likeness comparison** (M3 once viable rules exist): for the
   top-K viable rules, run the three matched conditions and compare:
   - **A**. coherent 4D projected world
   - **B**. hidden-shuffled 4D projected world
   - **C**. matched 2D baseline (Conway's Life or similar)

### First end-to-end M4A → M3 result (N=1, suggestive only)

A 50-rule × 2-seed × T=150 search on a 32×32×4×4 grid (≈45 sec on CPU)
produced the top rule `B[0.15, 0.26]_S[0.09, 0.38]_d=0.15` with viability
score +5.31 (21 persistent components, max age 96).  Re-running the three
matched conditions with this rule (single seed each):

| Condition | candidates | max observer_score |
|---|---|---|
| **4D coherent** | 45 | **+1.987** |
| Shuffled-4D | 182 | +1.676 |
| 2D Life | 67 | +1.530 |

This **N=1 result was misleading** — see M4B.

### M4B paired sweep (N=25 paired triples) — the real test

Running the proper paired sweep (5 viable rules × 5 seeds × 3 conditions
= 75 runs, T=200 on 32×32×4×4) reverses the picture:

| Comparison | metric | mean diff | 95% CI | perm p | win-rate |
|---|---|---|---|---|---|
| coherent vs shuffled | top5_mean | **−0.32** | [−0.48, −0.16] | **0.001** | 16% / 84% |
| coherent vs 2D Life | top5_mean | **−0.20** | [−0.35, −0.04] | **0.020** | 32% / 68% |
| coherent vs shuffled | score_per_track | +0.0038 | [−0.0004, 0.008] | 0.08 | 60% / 40% |
| coherent vs 2D Life | score_per_track | +0.0043 | [−0.0003, 0.009] | 0.07 | 76% / 24% |

**Key findings:**

1. **Shuffled-4D wins on extreme-score metrics** (max, top-5, p95) with a
   statistically significant Cliff's δ = -0.68 on top-5. But shuffled-4D
   *also* produces 2-7× more candidates than coherent-4D on the same
   rule + seed — more shots, higher max.
2. **On track-count-normalized score** (`score_per_track`) coherent-4D
   has a directional edge over both baselines (76% win-rate vs 2D, 60%
   vs shuffled), but neither difference reaches significance at p < 0.05
   with N=25 pairs.
3. **The original directional hypothesis (coherent > shuffled > 2D on
   observer_score) is NOT supported** under most distributional summary
   metrics. The M4B summary correctly fires its
   "shuffling-confound" interpretation: *"max-score comparisons are
   confounded by track count."*
4. **0/25 byte-identical hashes** between coherent and shuffled — the M3
   shuffled-baseline-no-op bug stays fixed.

This is the kind of result the framework was built to surface: an
infrastructure that lets us see when an N=1 directional finding doesn't
survive a paired statistical test.  Whether the hypothesis can be
rescued by (a) more rules, (b) longer runs, (c) per-track-normalized
scoring, or (d) is genuinely refuted, is the question M4C / M5 should
address.

### M4B sweep on M4C-evolve rules (the real headline)

Re-running the same M4B paired sweep but with the M4C **observer-fitness-optimized** rules
(5 rules × 5 seeds × T=200, 32×32×4×4, 836 sec) gives a statistically
distinct picture:

| Comparison | metric | M4B-on-M4A | M4B-on-M4C |
|---|---|---|---|
| coh vs **2D** | `score_per_track` | +0.004, p=0.07 | **+0.0076, p=0.003, Cliff δ +0.76, 88%/12%** |
| coh vs **2D** | `lifetime_weighted` | -0.002, p=0.94 | **+0.052, p=0.010, CI [0.017, 0.087]** |
| coh vs **2D** | `top5_mean` | -0.20, p=0.02 | -0.37, p=0.0005 |
| coh vs shuf | `score_per_track` | +0.004, p=0.08 | +0.001, p=0.59 |
| coh vs shuf | `top5_mean` | -0.32, p=0.001 | -0.26, p=0.0005 |

**Two real findings:**

1. **Coherent 4D significantly outperforms matched 2D Life on
   track-count-resistant metrics** (`score_per_track`,
   `lifetime_weighted_mean_score`) when the rules have been
   observer-fitness-optimized. The `score_per_track` effect is large:
   Cliff's δ = +0.76, 88% win rate. **This was not true for the M4A
   viability-search rules** — selection on observer fitness matters.

2. **Coherent vs shuffled is still inconclusive on normalized metrics**.
   Directionally positive (60–72% win rate) but neither difference
   reaches significance at N=25. Shuffled-4D also beats 2D Life on
   normalized metrics, suggesting it's "4D dynamics" (coherent or not)
   that produce higher per-candidate-quality observers vs 2D, with
   coherence giving a small but undetectable-at-N=25 additional bump.

3. **The track-count confound persists on extreme-score metrics.**
   Coherent loses to shuffled and 2D on max/top5/p95 because shuffled
   produces 2-3× more candidates per run.

The framework's automatic interpretation engine still outputs the
"shuffling confound" canonical paragraph because it gates on
`top5_mean`; the actual headline finding is on the *normalized* metrics
the engine doesn't currently surface. A natural next iteration is to
make the interpretation engine fire on `score_per_track` /
`lifetime_weighted` when those clear significance.

### M4C result (30-rule random + 3-gen × (5+5) evolve)

Random search over 30 fractional rules (T=150, 32×32×4×4, 2 seeds each)
in 54 sec produced a top rule with `lifetime_weighted` fitness +0.137.
Evolutionary search (3 generations × μ=5 + λ=5, seeded from the M4A
leaderboard) in 79 sec found a top rule with fitness +0.155 — slightly
better, with **60 candidates** vs the random-search winner's 4.  This
suggests evolutionary search finds rules that produce *both* many
candidates *and* high-quality observers, while random search stumbled
onto a rule with very few but very persistent observers.  The fitness
choice (`lifetime_weighted`) is doing what we wanted: it doesn't reward
chaotic high-track-count rules.

### M5 intervention findings — two regimes

**Run A** (M4C-evolve top rule, 3 candidates with `interior_size=1`,
12-step rollouts):

| intervention | mean_full_l1 | survival |
|---|---|---|
| `internal_flip` | 0.026 | 0.33 |
| `boundary_flip` | 0.055 | 0.33 |
| `environment_flip` | **0.112** | **1.00** |
| `hidden_shuffle` | 0.021 | 0.33 |

**Run B** (M4A-viability rule, 8 candidates with `interior_size=4–7`,
20-step rollouts):

| intervention | mean_full_l1 | survival |
|---|---|---|
| `internal_flip` | 0.054 | **0.75** |
| `boundary_flip` | 0.055 | 0.50 |
| `environment_flip` | **0.086** | **0.25** |
| `hidden_shuffle` | 0.043 | 0.62 |

Two **different** patterns emerge depending on candidate size and
rollout horizon:

- **Run A** (tiny candidates, short rollouts): environment perturbations
  spread widely but **don't kill** the candidate. Candidate-targeted
  perturbations kill 67% of the time. This is the "agency-like"
  asymmetry: self/boundary perturbations matter more than environment
  for the candidate's survival.
- **Run B** (substantial candidates, longer rollouts): environment
  perturbations both **spread widest AND kill 75% of candidates**.
  Internal perturbations are now the *least* lethal (75% survival).
  The "agency" asymmetry inverts.

Honest read: the directional finding is **regime-dependent**. Whether
candidates exhibit the agency pattern depends on (a) candidate size
relative to flip-fraction, and (b) how long perturbations have to
propagate. The framework now lets us measure this dependence; the next
empirical question is whether **any** rule + size combination produces
a regime where the agency-like asymmetry holds robustly across many
candidates and seeds.

### M3 shuffled-baseline bug discovered during M4A

The pre-M4A `simulate_4d_to_zarr` applied the `state_mutator` to a copy
before projection but never fed the result back into the CA state, which
made the hidden-shuffled baseline a no-op under mean-threshold projection
(both depend only on per-column counts, which the shuffle preserves).
Fix: the mutated state is now also written back into `ca.state` so
subsequent CA steps evolve from the shuffled state.  The bug only became
visible once viable rules existed to actually compare against.

## Install

```bash
cd observer_worlds
pip install -e .
```

## Run the first experiment

```bash
python -m observer_worlds.experiments.run_4d_projection
```

To enable causality + resilience scores (which need 4D state for paired rollouts):

```bash
python -m observer_worlds.experiments.run_4d_projection \
    --save-4d-snapshots --snapshot-interval 20 --rollout-steps 10
```

Output layout:

```
config.json                      # exact config used
data/states.zarr/frames_2d       # projected 2D frames (T, Nx, Ny) uint8
data/states.zarr/snapshots_4d/   # 4D state at checkpoints (if enabled)
data/tracks.csv                  # one row per (track_id, frame)
data/candidates.csv              # persistence-filter results
data/observer_scores.csv         # full M2 metric breakdown per candidate
frames/projected_world.gif       # animated 2D world with track overlays
plots/lifetimes.png
plots/area_vs_time.png
summary.md                       # human-readable run summary
```

## Sensible defaults (and caveats)

| Knob | Default | Notes |
|---|---|---|
| Grid | 64 × 64 × 8 × 8 | per spec |
| Timesteps | 200 | enough for tracking; CPU-friendly |
| Initial density | 0.15 | sparse; avoids immediate chaos |
| Birth set (4D) | `{30, 31, 32, 33}` | **heuristic**; first runs may be trivial |
| Survival set (4D) | `{28, ..., 34}` | **heuristic**; first runs may be trivial |
| 2D baseline rule | Conway's Life (B3/S23) | canonical reference (used in M3) |
| Projection | `mean_threshold(theta=0.5)` | per spec |
| Tracker | greedy IoU ≥ 0.3, fallback centroid ≤ 5 | simple, good-enough |
| Min observer age | 10 | per spec |
| Storage | zarr v2, time-chunked | streaming-friendly |

The default 4D rule was chosen by analogy to Life's "alive" fraction of its
neighbourhood, but high-dimensional CAs are notoriously sensitive: the first
run will very likely show rapid die-off, a frozen lattice, or noise.  This is
expected.  The rule-search experiment (M4) is the principled way to find
rules that produce persistent bounded structures.

## CLI flags

```bash
python -m observer_worlds.experiments.run_4d_projection \
    --grid 32 32 4 4 \
    --timesteps 100 \
    --seed 0 \
    --backend numba \
    --label tiny
```

You can also pass `--config path/to/config.json` to load a full `RunConfig`.

## Tests

```bash
pytest tests/ -v
```

Unit tests cover:

- `tests/test_projection.py` — mean-threshold projection edge cases
- `tests/test_components.py` — boundary/environment shell construction
- `tests/test_tracking.py` — frame-to-frame track linkage
- `tests/test_ca4d_kernels.py` — numpy and numba kernels agree (skipped if
  numba unavailable)

## Repo layout

```
observer_worlds/
  worlds/        4D + 2D CA dynamics, projection, rules
  detection/     connected components and tracking
  metrics/       observer-likeness scores (M1: persistence only)
  experiments/   orchestration scripts
  storage/       zarr-backed run store
  analysis/      plots and GIFs
  utils/         configs, RNG
tests/           unit tests
outputs/         run artifacts (gitignored)
```

## Roadmap

- **M1** (shipped): minimal runnable pipeline + persistence filter.
- **M2** (shipped): full metric suite — time / memory / selfhood / causality
  / resilience, plus the combined `observer_score`.
- **M3** (shipped): 2D baseline + hidden-shuffled baseline + cross-run
  comparison plots.
- **M4A** (shipped): **viability search** — fractional totalistic 4D rules
  + 7-component viability score + multi-seed random search. Discovers 4D
  rules with non-trivial dynamics so the hypothesis comparison can run on
  something other than die-off.
- **M4B** (shipped): **multi-rule, multi-seed observer-metric sweep** —
  paired (rule, seed) evaluations across all 3 conditions, with bootstrap
  CIs, permutation tests, Cohen's d / Cliff's δ effect sizes, win rates,
  and 9 diagnostic plots. Surfaces the track-count confound that made
  the N=1 result misleading.
- **M4C** (shipped): **observer-metric-guided rule search** — random
  search and `(μ+λ)` evolutionary search using the **full M2 metric
  suite** as fitness. Default fitness is `lifetime_weighted_mean_score`
  (track-count-resistant by construction). Configurable via
  `--fitness-mode {lifetime_weighted, top5_mean, score_per_track,
  composite}`. Supports seeding the initial population from an M4A
  leaderboard.
- **M4D** (shipped): **held-out validation + optimized 2D baseline**.
  Runs M4B-style sweep twice — once against fixed Conway's Life, once
  against an observer-fitness-optimized 2D rule from
  `evolve_2d_observer_rules.py`. Combined summary applies M4D
  interpretation rules. Aborts if held-out seeds overlap training
  seeds (unless `--allow-overlap`). Provenance metadata propagates
  through stats and summary.md.
- **M6** (shipped): **Hidden Causal Dependence**. For each top
  observer-candidate, applies a `hidden_invisible` perturbation
  (permutes z,w within the candidate's interior; projection at t=0 is
  byte-identical, so any downstream projected divergence comes purely
  from hidden state). The HCE scalar measures that downstream
  divergence; in 2D systems it is zero by construction. Optional
  `--shuffled-config` runs the same experiment on a hidden-shuffled
  baseline and reports the paired (rank-paired or id-paired)
  difference. First framework component that produces a positive,
  uniquely-4D directional finding.
- **M5** (shipped): **per-candidate intervention experiments** — for top
  observer-candidates, applies all 4 intervention types
  (`internal_flip`, `boundary_flip`, `environment_flip`,
  `hidden_shuffle`) at a snapshot inside each candidate's lifetime, runs
  paired forward rollouts, captures **per-step divergence trajectories**
  (not just aggregates), produces per-candidate divergence + resilience
  plots, aggregate plots, and an intervention heatmap.

## Run M6 Hidden Causal Dependence experiment

```bash
# Compare coherent-4D HCE against hidden-shuffled-4D HCE on the same rule.
# Both runs use the rule's RunConfig; the shuffled run threads the
# hidden_shuffle_mutator into the simulator (writes back into ca.state).
# ~30s on the M4A viable rule at the moderate config.
python -m observer_worlds.experiments.run_m6_hidden_causal \
    --config outputs/rule_search/m4a_search/top_k/rule_001/config.json \
    --shuffled-config outputs/rule_search/m4a_search/top_k/rule_001/config.json \
    --top-k 8 --n-steps 15 --n-replicates 5 \
    --backend numpy --label m6
```

Outputs `summary.md` with the headline HCE numbers, plus
`hidden_causal_summary.csv`, `hidden_causal_trajectories.json`,
`stats_summary.json`, and 8 plots (per-condition aggregate divergence,
HCE distribution, HCE-vs-visible scatter, plus paired and boxplot
comparisons when shuffled-config is provided).

## Run M4D held-out validation (with optimized 2D baseline)

```bash
# 1. First evolve an optimized 2D rule (~5 min for the moderate config)
python -m observer_worlds.experiments.evolve_2d_observer_rules \
    --strategy evolve --population 12 --generations 5 --lam 12 --n-seeds 2 \
    --grid 32 32 --timesteps 150 --top-k 5 \
    --fitness-mode lifetime_weighted \
    --base-eval-seed 1000 \
    --out-dir outputs/m4d_2d_evolve

# 2. Held-out validation against both Life and the optimized 2D rule
#    (~9 min for the moderate config; held-out seeds = 2000..2004)
python -m observer_worlds.experiments.run_m4d_holdout_validation \
    --rules-from outputs/observer_search/m4c_evolve/leaderboard.json \
    --optimized-2d-rules outputs/m4d_2d_evolve/top_2d_rules.json \
    --n-rules 5 --seeds 5 --timesteps 200 \
    --grid 32 32 4 4 --rollout-steps 6 \
    --base-eval-seed 2000 \
    --training-seeds 1000 1001 1002 1003 1004 \
    --label m4d
```

The combined `summary.md` applies one of four canonical interpretations:
- "...advantage was due to optimization, not dimensional projection." — optimized-2D wins
- "...advantage beyond optimization alone." — coh wins both 2D baselines
- "...beat standard 2D Life, but not an equally optimized 2D rule class." — coh wins fixed only
- Mixed result.

## Run M4C observer-metric search

```bash
# Random search (~minutes): 30 rules × 2 seeds × T=150
python -m observer_worlds.experiments.run_search_observer_rules \
    --strategy random --n-rules 30 --n-seeds 2 \
    --grid 32 32 4 4 --timesteps 150 \
    --fitness-mode lifetime_weighted --top-k 5

# Evolutionary (μ+λ) seeded from M4A:
python -m observer_worlds.experiments.run_search_observer_rules \
    --strategy evolve --n-generations 5 --mu 8 --lam 8 --n-seeds 2 \
    --grid 32 32 4 4 --timesteps 150 \
    --seed-population outputs/rule_search/m4a_search/leaderboard.json
```

Each run produces `leaderboard.csv`, `leaderboard.json`, `top_k/rule_<rank>/{config,rule}.json`,
and `history.json` (evolve only).

## Run M5 intervention experiment

```bash
# From an M4C top rule (re-runs the 4D experiment with snapshots):
python -m observer_worlds.experiments.run_intervention_experiment \
    --config outputs/observer_search/m4c_evolve/top_k/rule_001/config.json \
    --top-k 5 --n-steps 12 --flip-fraction 0.5 --backend numpy

# Or from an existing run dir that already saved snapshots:
python -m observer_worlds.experiments.run_intervention_experiment \
    --from-run outputs/<some_4d_run> \
    --top-k 5 --n-steps 12
```

Outputs `intervention_summary.csv`, `intervention_trajectories.json`,
per-candidate divergence + resilience plots in `plots/per_candidate/`,
and aggregate plots: `aggregate_divergence_*.png`,
`intervention_heatmap_*.png`, `intervention_summary_bars.png`.

## Run M4B paired sweep

```bash
# Smoke test (~25 seconds): 2 rules × 2 seeds × T=100 on 32x32x4x4
python -m observer_worlds.experiments.run_m4b_observer_sweep \
    --rules-from outputs/rule_search/m4a_search/leaderboard.json \
    --quick --label m4b_smoke

# Real sweep (~4 minutes): 5 rules × 5 seeds × T=200 on 32x32x4x4
python -m observer_worlds.experiments.run_m4b_observer_sweep \
    --rules-from outputs/rule_search/m4a_search/leaderboard.json \
    --n-rules 5 --seeds 5 --timesteps 200 \
    --grid 32 32 4 4 --top-k-videos 3 --label m4b

# Production (~hours): 10 rules × 10 seeds × T=500 on 64x64x8x8
python -m observer_worlds.experiments.run_m4b_observer_sweep \
    --rules-from outputs/rule_search/m4a_search/leaderboard.json \
    --n-rules 10 --seeds 10 --timesteps 500 --label m4b_full
```

Outputs land in `outputs/m4b_<UTC>/`: `paired_runs.csv`,
`condition_summary.csv`, `candidate_metrics.csv`,
`paired_differences.csv`, `stats_summary.json`, 9 PNGs in `plots/`,
top-K videos in `videos/top_candidates/<condition>/`, and `summary.md`
with one of five canonical interpretation paragraphs.

## Run viability search (M4A)

```bash
# Tiny smoke (verify wiring): ~1 minute
python -m observer_worlds.experiments.search_viable_4d_rules \
    --n-rules 20 --seeds 2 --timesteps 80 \
    --grid 32 32 4 4 --top-k 5 \
    --out-dir outputs/rule_search/viability_smoke

# Production search (per the M4A spec): ~tens of minutes on CPU
python -m observer_worlds.experiments.search_viable_4d_rules \
    --n-rules 200 --seeds 3 --timesteps 300 \
    --grid 64 64 8 8 --top-k 10 \
    --out-dir outputs/rule_search/viability
```

Inspect `outputs/rule_search/viability/leaderboard.csv` and the top-K
artifact directories, then re-run M3 with a viable rule's config:

```bash
python -m observer_worlds.experiments.run_4d_projection \
    --config outputs/rule_search/viability/top_k/rule_001/config.json \
    --label viable_rule_001 --save-4d-snapshots --snapshot-interval 25
```

## Run the full M3 hypothesis comparison

```bash
# 1. 4D experiment with snapshots
python -m observer_worlds.experiments.run_4d_projection \
    --grid 48 48 4 4 --timesteps 200 --seed 0 --label m3_4d \
    --save-4d-snapshots --snapshot-interval 25 --rollout-steps 6

# 2. 2D Life baseline
python -m observer_worlds.experiments.run_2d_baseline \
    --grid 48 48 --timesteps 200 --seed 0 --initial-density 0.3 --label m3_2d

# 3. Shuffled-hidden baseline (4D, but z,w permuted before projection)
python -m observer_worlds.experiments.run_shuffled_hidden_baseline \
    --grid 48 48 4 4 --timesteps 200 --seed 0 --label m3_shuffled \
    --save-4d-snapshots --snapshot-interval 25 --rollout-steps 6

# 4. Cross-run summary + comparison plots
python -m observer_worlds.analysis.summarize_results \
    --output-root outputs --out-dir outputs/_summary
```

Inspect `outputs/_summary/baseline_comparison.png` and `summary.md`.

## Glossary of operationalizations

- **observer-candidate**: a tracked structure that persists ≥ `min_age`,
  remains bounded (area never exceeds `max_area_fraction × grid`), and has
  non-trivial internal state variation.
- **interior `I`**: the cells inside a candidate's mask, eroded by 1.
- **boundary `B`**: `mask XOR erosion(mask, 1)`.
- **environment `E`**: `dilation(mask, env_radius) XOR dilation(mask, 1)`.
- **observer_score** (M2+): weighted sum of normalized time / memory / selfhood
  / causality / resilience scores.  Equal weights by default.

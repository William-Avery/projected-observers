# Tutorial — running `projected-observers` end to end

This walks through every experiment in the framework, in the order you'd
run them to reproduce (and extend) the headline findings. Read top to
bottom on a first pass; later you can jump straight to the milestone
you care about.

> **Time budgets** below assume a modern macOS / Linux laptop with
> numba available. Numba JIT-compiles the 4D CA kernel on first use
> (~5–10 s), then runs at full speed. All times exclude that first
> compile.

---

## 0. Setup

```bash
git clone https://github.com/William-Avery/projected-observers.git
cd projected-observers/observer_worlds
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v   # should print "153 passed"
```

If `numba` install gives you trouble, you can fall back to
`--backend numpy` on every CLI flag below — slower, but no JIT.

---

## 1. Smallest end-to-end run (1 minute)

The simplest 4D-to-2D experiment. The default heuristic rule produces
trivial dynamics (die-off), which is **expected** — this run validates
that the pipeline is wired correctly.

```bash
python -m observer_worlds.experiments.run_4d_projection \
    --grid 32 32 4 4 --timesteps 60 --seed 0 --label first
```

Look in `outputs/first_<UTC>/`:

```
config.json                          # exact config used
data/states.zarr/frames_2d/...       # projected 2D frames (T, Nx, Ny) uint8
data/tracks.csv                      # one row per (track_id, frame)
data/candidates.csv                  # persistence-filter results
data/observer_scores.csv             # full M2 metric breakdown per candidate
frames/projected_world.gif           # animated projected world with overlays
plots/lifetimes.png, area_vs_time.png
summary.md
```

Open `summary.md`. With the default rule you should see "0 candidates"
and a note that this is expected — the next step (M4A) is what
discovers rules that actually produce persistent structures.

---

## 2. Find non-trivial 4D rules (M4A — viability search) — 1 minute

Random search over **fractional totalistic 4D rules** (5 continuous
floats: `birth_min`, `birth_max`, `survive_min`, `survive_max`,
`initial_density`). Scores each rule on a multi-component viability
metric (extinction / saturation / frozen-world penalties; target-
activity / temporal-change / persistence / boundedness / diversity
rewards).

```bash
python -m observer_worlds.experiments.search_viable_4d_rules \
    --n-rules 50 --seeds 2 --timesteps 150 \
    --grid 32 32 4 4 --top-k 10 \
    --out-dir outputs/m4a_search
```

After ~45 s you should see a top-K table with viability scores > 0.
The top-K artifact directories under `outputs/m4a_search/top_k/rule_NNN/`
each contain a re-runnable `config.json`. Try one:

```bash
python -m observer_worlds.experiments.run_4d_projection \
    --config outputs/m4a_search/top_k/rule_001/config.json \
    --label viable_4d --save-4d-snapshots --snapshot-interval 25
```

`outputs/viable_4d_<UTC>/summary.md` should now show **dozens of
candidates** with finite observer scores. The rest of the tutorial uses
this rule.

---

## 3. The simplest hypothesis test (M3 baselines) — 5 minutes

Runs the same rule under three matched conditions, then a cross-run
analysis.

```bash
# A. Coherent 4D (already done above as 'viable_4d').
# B. Shuffled-4D (z,w fibers permuted into ca.state every step):
python -m observer_worlds.experiments.run_shuffled_hidden_baseline \
    --config outputs/m4a_search/top_k/rule_001/config.json \
    --label viable_shuffled --save-4d-snapshots --snapshot-interval 25

# C. 2D Conway's Life baseline:
python -m observer_worlds.experiments.run_2d_baseline \
    --grid 32 32 --timesteps 150 --seed 1000 \
    --initial-density 0.30 --label viable_2d

# Cross-run summary:
python -m observer_worlds.analysis.summarize_results \
    --output-root outputs --out-dir outputs/_summary --glob "viable_*"
```

`outputs/_summary/summary.md` shows per-condition stats. **N=1 per
condition**, so this is suggestive only — for a real test, see step 5.

---

## 4. Search for rules that maximize observer-likeness (M4C) — 1–10 min

Random or evolutionary `(μ+λ)` search using the **full M2 metric
suite** as fitness. Default fitness is `lifetime_weighted_mean_score`
(track-count-resistant; high score requires both many candidates *and*
long-lived high-quality observers).

```bash
# Random search:
python -m observer_worlds.experiments.run_search_observer_rules \
    --strategy random --n-rules 30 --n-seeds 2 \
    --grid 32 32 4 4 --timesteps 150 \
    --fitness-mode lifetime_weighted --top-k 5 \
    --out-dir outputs/m4c_random

# Evolutionary, seeded from M4A leaderboard:
python -m observer_worlds.experiments.run_search_observer_rules \
    --strategy evolve --n-generations 5 --mu 8 --lam 8 --n-seeds 2 \
    --grid 32 32 4 4 --timesteps 150 \
    --fitness-mode lifetime_weighted \
    --seed-population outputs/m4a_search/leaderboard.json \
    --out-dir outputs/m4c_evolve
```

Each writes `leaderboard.{csv,json}`, `top_k/rule_NNN/{config,rule}.json`,
plus `history.json` and `plots/fitness_vs_generation.png` for evolve.

**Other fitness modes** for ablation: `--fitness-mode {top5_mean, score_per_track, composite}`.

---

## 5. The actual statistical hypothesis test (M4B) — ~5 minutes

Paired (rule × seed) sweep across all three conditions with bootstrap
CIs, permutation tests, Cohen's d / Cliff's δ effect sizes, and win
rates. **This is the main reportable test.**

```bash
python -m observer_worlds.experiments.run_m4b_observer_sweep \
    --rules-from outputs/m4c_evolve/leaderboard.json \
    --n-rules 5 --seeds 5 --timesteps 200 \
    --grid 32 32 4 4 --top-k-videos 3 \
    --rule-source M4C_observer_optimized \
    --optimization-objective lifetime_weighted_mean_score \
    --label m4b_on_m4c
```

`outputs/m4b_on_m4c_<UTC>/summary.md` headlines:
- per-condition track / candidate counts
- paired-difference table for each headline metric (max, top5, p95,
  lifetime-weighted, score_per_track) with mean diff, 95% bootstrap
  CI, permutation p-value, Cliff's δ, win rate
- automatic interpretation that selects from one of five canonical
  paragraphs and appends caveats (track-count-confound,
  optimized-rules-not-fair, seed-overlap)

The interpretation engine **prioritizes normalized metrics**
(`score_per_track`, `lifetime_weighted_mean_score`) because extreme-
score metrics (max, top5) are confounded by track count: shuffled
runs typically produce 2–3× more candidates → more shots at high max.

---

## 6. Held-out validation with optimized 2D baseline (M4D) — ~15 minutes

Two passes:
- **Pass A**: against fixed Conway's Life (the standard cheap 2D baseline)
- **Pass B** *(if `--optimized-2d-rules` provided)*: against an observer-
  fitness-optimized 2D rule from `evolve_2d_observer_rules.py`.

Held-out seeds are checked for overlap with training seeds; the run
**aborts** if they overlap unless `--allow-overlap` is set.

```bash
# 1. Evolve an optimized 2D rule (~5 min):
python -m observer_worlds.experiments.evolve_2d_observer_rules \
    --strategy evolve --population 12 --generations 5 --lam 12 --n-seeds 2 \
    --grid 32 32 --timesteps 150 --top-k 5 \
    --fitness-mode lifetime_weighted \
    --base-eval-seed 1000 \
    --out-dir outputs/m4d_2d_evolve

# 2. Held-out M4D run (~9 min):
python -m observer_worlds.experiments.run_m4d_holdout_validation \
    --rules-from outputs/m4c_evolve/leaderboard.json \
    --optimized-2d-rules outputs/m4d_2d_evolve/top_2d_rules.json \
    --n-rules 5 --seeds 5 --timesteps 200 \
    --grid 32 32 4 4 --rollout-steps 6 \
    --base-eval-seed 2000 \
    --training-seeds 1000 1001 1002 1003 1004 \
    --label m4d
```

`outputs/m4d_<UTC>/summary.md` selects from four interpretation
paragraphs:

- "advantage was due to optimization, not dimensional projection" —
  optimized-2D wins
- "advantage beyond optimization alone" — coherent wins both 2D
  baselines
- "beat standard 2D Life, but not an equally optimized 2D rule class" —
  coherent wins fixed only
- mixed result.

You'll also find per-pass full M4B-style summaries under
`vs_fixed_2d/summary.md` and `vs_optimized_2d/summary.md`.

---

## 7. Per-candidate intervention experiment (M5) — ~30 seconds

For top observer-candidates, applies four intervention types
(`internal_flip`, `boundary_flip`, `environment_flip`,
`hidden_shuffle`) at a snapshot inside the candidate's lifetime, runs
paired forward rollouts, and captures **per-step divergence
trajectories** (not just aggregates).

```bash
python -m observer_worlds.experiments.run_intervention_experiment \
    --config outputs/m4a_search/top_k/rule_001/config.json \
    --top-k 8 --n-steps 20 --flip-fraction 0.5 \
    --backend numpy --label m5
```

The CLI walks down candidates by observer score until it finds enough
with non-degenerate masks, runs all four interventions on each, and
writes:

- `intervention_summary.csv` — one row per (track_id, intervention_type)
- `intervention_trajectories.json` — full per-step lists per replicate
- `plots/aggregate_divergence_*.png`,
  `plots/intervention_heatmap_*.png`,
  `plots/intervention_summary_bars.png`,
  `plots/per_candidate/track_<id>_{divergence,resilience}.png`
- `summary.md` ranking interventions by divergence and survival impact

---

## 8. Hidden Causal Effect (M6) — the headline experiment — ~30 seconds

The framework's positive finding. Tests for an effect that is
**identically zero in 2D systems by construction**: downstream
projected divergence under a perturbation that preserves the 2D
projection at t=0.

```bash
python -m observer_worlds.experiments.run_m6_hidden_causal \
    --config outputs/m4a_search/top_k/rule_001/config.json \
    --shuffled-config outputs/m4a_search/top_k/rule_001/config.json \
    --top-k 8 --n-steps 15 --n-replicates 5 \
    --backend numpy --label m6
```

`outputs/m6_<UTC>/summary.md` reports:
- **Coherent 4D**: mean HCE across candidates, fraction > 0,
  HCE/visible-control ratio, sign-test p-value
- **Hidden-shuffled 4D**: same metrics on the shuffled control
- **Headline**: paired comparison (id-pairing → rank-pairing fallback)
  with bootstrap CI

A representative result on the M4A viable rule (8 candidates × 5
replicates × 15 rollout steps): coherent mean HCE = 0.064 (88%
positive), shuffled mean HCE = 0.000, paired diff +0.064 with 95%
bootstrap CI [+0.036, +0.092] excluding zero. Coherent wins 7/0 (1
tie at 0).

---

## What "summary.md" looks like across the experiments

Every CLI writes a self-contained `summary.md` to its run dir. Open
the one for the experiment you just ran — it has:

- the exact rule(s) and seeds used
- the per-condition / per-perturbation aggregate metrics
- a paragraph of automatic interpretation (selected from a small set
  of canonical sentences) plus any caveats that triggered

The interpretation engine is in `analysis/m4b_stats.py` and
`experiments/run_m4d_holdout_validation.py`. Each canonical sentence is
exposed as a module-level constant so tests can substring-match — they
won't drift over time.

---

## Cross-run analysis

To compare multiple runs after the fact:

```bash
python -m observer_worlds.analysis.summarize_results \
    --output-root outputs --out-dir outputs/_my_summary \
    --glob "m4b_*"   # or any glob matching run dir names
```

Produces `observer_score_histogram.png`, `score_vs_age.png`,
`baseline_comparison.png`, and a markdown summary table grouping runs
by their auto-detected `world_kind`.

---

## Tests

```bash
pytest tests/ -v
```

153 tests covering: CA kernels, projection variants, component
extraction, tracking, all five observer-likeness metrics, persistence
filter, M4B paired statistics + interpretation rules, M4D held-out
combined-interpretation branches, M5 interventions, **the M6 hidden-
invisible projection-preservation invariant**, and shuffled-baseline
no-op regression test.

---

## Performance notes

- The 4D CA Moore-r1 update has 80 neighbours per cell. The numba
  kernel handles a 64×64×8×8 grid at ~10–30 steps/s on a laptop CPU.
- M2 metric scoring (Ridge + KFold across time/memory/selfhood per
  candidate) is the slowest stage downstream — expect ~0.1–1 s per
  candidate.
- M4B / M4D sweeps are embarrassingly parallel across (rule, seed)
  pairs but currently run sequentially. The default sweep config in
  this tutorial is sized for a few minutes of wall time. The
  production defaults (10 rules × 10 seeds × T=500 on 64×64×8×8)
  take 1–3 hours.

---

## Where to look in the code

| Question | File |
|---|---|
| What's the 4D CA update rule? | `observer_worlds/worlds/ca4d.py` |
| How is "observer-candidate" defined? | `observer_worlds/metrics/persistence.py` |
| What's the M2 metric suite? | `observer_worlds/metrics/{time,memory,selfhood,causality,resilience,observer}_score.py` |
| How is the shuffled baseline implemented? | `observer_worlds/experiments/_pipeline.py` (`simulate_4d_to_zarr` with state_mutator hook) and `observer_worlds/experiments/_m4b_sweep.py` (`hidden_shuffle_mutator`) |
| Where do the interpretation rules live? | `observer_worlds/analysis/m4b_stats.py` (`_INTERP_*` constants) and `observer_worlds/experiments/run_m4d_holdout_validation.py` (`COMBINED_*` constants) |
| What exactly is HCE? | `observer_worlds/experiments/_m6_hidden_causal.py` (module docstring + `HiddenCausalReport`) |

---

## Suggested next experiments

1. **M6 at scale**: replicate the M6 finding across all top-K M4C rules
   (10+ rules, 8+ candidates each) for tighter CIs.
2. **HCE-as-fitness**: run M4C-style search with HCE replacing
   `lifetime_weighted_mean_score` as the fitness function. Discover
   rules whose hidden structure has *maximal* causal weight.
3. **HCE → refined selfhood**: a high-HCE candidate has a measurable
   hidden-state correlate of its persistent identity. Use it as a
   sharper operationalization of "boundary-mediated selfhood" than
   the current M2 selfhood_score.
4. **Stochastic rule family**: extend `FractionalRule` with a sigmoid-
   stochastic variant `p_on = sigmoid(α·rho + β·X + bias)` (mentioned
   in the original M4A spec but deferred). New search space, possibly
   different HCE regime.

# Tutorial — running `projected-observers` end to end

This walks through every experiment in the framework, in the order you'd
run them to reproduce (and extend) the headline findings. Read top to
bottom on a first pass; later you can jump straight to the milestone
you care about.

**Each section starts with the question the experiment is designed to
answer.** The milestones are not "more code piled on" — each one
exists because the previous one left a specific loophole, and this
section makes the loophole explicit before showing how to run the test.

> **Time budgets** below assume a modern macOS / Linux laptop with
> numba available. Numba JIT-compiles the 4D CA kernel on first use
> (~5–10 s), then runs at full speed. All times exclude that first
> compile.

## Quick glossary

(See `README.md` for the full glossary table.)

| Term | Quick meaning |
|---|---|
| **HCE** | Hidden Causal Effect — downstream projected divergence under a perturbation invisible at t=0. Zero in 2D by construction |
| **observer_score** | Z-normalized combined metric: time + memory + selfhood + causality + resilience |
| **score_per_track** | `sum(observer_scores) / num_tracks`. Track-count-resistant cross-condition comparison metric |
| **lifetime_weighted_mean_score** | Default M4C fitness — weights observer_scores by candidate age. Track-count-resistant |
| **fractional rule** | 4D CA rule defined by 5 floats: `birth_min`, `birth_max`, `survive_min`, `survive_max` (all over neighbor *fraction* in [0,1]), `initial_density` |
| **coherent_4d** | Standard 4D CA — natural hidden-dimensional dynamics |
| **per-step-shuffled-4d** | Negative-control simulation: z,w fibers randomized at every step |
| **candidate / observer-candidate** | Connected component in 2D projection that persists across multiple frames + passes the persistence filter |
| **hidden_invisible** | Perturbation that permutes z,w cells in a candidate's interior — preserves projection at t=0 (unique to 4D) |
| **track-count confound** | More candidates → more shots at extreme max scores. Why `score_per_track` is preferred over max/top5 |
| **cluster-bootstrap by rule** | Bootstrap that resamples whole rules (not rows) so CIs respect within-rule correlation |
| **held-out seeds** | Seeds disjoint from those used during optimization. M4D enforces this |

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

> **Why run this?** Before any hypothesis test, confirm the pipeline
> is wired correctly: 4D CA → projection → tracking → metric scoring →
> CSVs/plots/GIF/summary. The default rule produces trivial dynamics
> on purpose, so a "0 candidates" outcome here is a successful
> integration test, not a research result.

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

> **Why run this?** The default 4D rule produces zero candidates. The
> hypothesis is unanswerable when the 4D side is empty, so we need a
> systematic search of the rule space for rules that produce
> *anything* persistent and bounded. Without this step, every
> downstream comparison is "0 vs Conway's Life" which tells us
> nothing.

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

> **Why run this?** A high observer_score in 4D alone tells us
> nothing. We need to know whether 4D-projected candidates score
> higher than (a) a fair 2D system (Conway's Life) and (b) a
> "shuffle-the-hidden-dimensions" 4D control. The 2D baseline rules
> out "any complex dynamics produces high scores"; the shuffled-4D
> control rules out "having 4D state at all is enough — coherent
> hidden organization doesn't matter."

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

> **Why run this?** M4A's viability score rewards rules that produce
> *any* persistent candidates. But viability and observer-likeness
> are different things — a rule could produce many persistent blobs
> that all score badly on time/memory/selfhood. To give the 4D side
> the strongest possible chance against the 2D baseline, search rule
> space using the actual `observer_score` as fitness. If 4D *can*
> outperform 2D, this is where it should show up.

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

> **Why run this?** A single-run "max observer_score" comparison is
> easy to fool: shuffled-4D often produces 2-3× more candidates per
> run than coherent-4D, so it has more shots at extreme high scores.
> We need **paired** evaluation across many rules and seeds, with
> rigorous statistical tools that don't conflate "more shots" with
> "higher quality." This is the test that, if positive, would let
> us claim a real effect — and if negative, would let us rule out a
> false-positive from M3-style single-condition comparisons.

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

> **Why run this?** If M4B-on-M4C-rules shows "coherent 4D beats 2D
> Life on score_per_track," there are still two confounds: (a) the
> M4C-optimized 4D rules might overfit to the training seeds, and
> (b) the comparison is fundamentally unfair — optimized 4D rules
> vs unoptimized 2D Life. The "advantage" might just be optimization
> effort. Two fixes: re-run on **held-out seeds** (disjoint from
> training), and re-run vs an **equally-optimized 2D rule** with
> matched compute budget. If 4D still wins, the result is real. If
> not, the apparent advantage was optimization, not dimensionality.

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

> **Why run this?** A persistent track that scores high on
> `observer_score` might still be passive matter. The functional
> definition of "agency" requires that perturbing the structure
> produces *different* consequences than perturbing its environment.
> M5 actually intervenes on each candidate and measures
> per-step divergence trajectories under four intervention types. The
> distinctive signature of agency is asymmetry — internal/boundary
> perturbations should affect the candidate's future *differently*
> than equivalent-magnitude environment perturbations.

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

## 8. Hidden Causal Effect (M6) — ~30 seconds

> **Why run this?** M4D falsified the simple "4D > 2D on generic
> observer_score" claim under a fair comparison. But that doesn't
> rule out **dimension-specific** effects — properties that 4D
> systems *can* have and 2D systems *cannot have by construction*.
> The cleanest test: a perturbation that is *invisible* in the 2D
> projection at t=0 (mean-threshold projection unchanged because
> the per-column count is preserved) but changes the underlying 4D
> state. Any downstream projected divergence must be causally
> attributable to **hidden state** — and 2D systems have no hidden
> state to perturb, so this quantity is identically zero in 2D.
> If positive in 4D, it's the framework's first dimension-specific
> finding.

Tests for an effect that is **identically zero in 2D systems by
construction**: downstream projected divergence under a perturbation
that preserves the 2D projection at t=0.

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

## 9. M6 replication at scale + stronger controls (M6B) — ~7 minutes

> **Why run this?** M6 was N=8 candidates from a *single* rule. Three
> things needed checking before reporting M6 as a real finding:
>
> 1. **Replication** — does the effect survive across many rules
>    from both M4A (viability) and M4C (observer-optimized) sources,
>    many seeds, many candidates per run?
> 2. **Localization** — is the effect candidate-*local*, or does
>    perturbing the hidden state *anywhere* on the grid produce
>    similar downstream divergence? If the latter, the effect is a
>    global instability of the dynamics, not a "self"-specific
>    causal dependence.
> 3. **The shuffled-control claim** — M6 reported coh-HCE = 0.064
>    while shuf-HCE = 0.000. Is this gap real, or an artifact of
>    snapshot mechanics in a single-run setup?
>
> M6B answers all three with proper cluster-bootstrap-by-rule
> statistics, three additional control interventions
> (`one_time_scramble_local`, `fiber_replacement_local`, `hidden_invisible_far`,
> `sham`), and three candidate selection modes (top observer-score,
> top lifetime, random eligible) so the result doesn't depend on
> hand-picked extreme winners.

```bash
# ~7 minutes: 3 M4C + 3 M4A rules × 3 seeds × ~5 candidates per mode ×
# 3 selection modes × 6 interventions × 3 replicates × 3 horizons
python -m observer_worlds.experiments.run_m6b_hidden_causal_replication \
    --rules-from outputs/m4c_evolve/leaderboard.json \
    --also-rules-from outputs/m4a_search/leaderboard.json \
    --n-rules 3 --seeds 3 --timesteps 150 \
    --grid 32 32 4 4 --max-candidates 5 --replicates 3 \
    --horizons 5 10 20 --backend numpy \
    --base-seed 2000 --label m6b
```

Outputs `summary.md` with cluster-bootstrap CIs for every
(condition, intervention, horizon) cell, paired comparisons across
five intervention pairs, win rates, horizon trends, and an automatic
interpretation that selects from seven canonical paragraphs based on
which controls survive significance.

The framework's primary M6B finding from a real run of this exact
config:

| Comparison | mean diff | 95% CI | wins | sign-test p |
|---|---|---|---|---|
| coh local hidden vs sham | **+0.039** | [+0.013, +0.070] | 210/244 (86%) | **<0.0001** |
| coh local hidden vs far hidden (local div) | **+0.239** | [+0.152, +0.335] | 183/244 (75%) | **<0.0001** |

→ HCE is **robust and candidate-local**. Pearson r between HCE and
observer_score is +0.22 (weak) — the framework's interpretation
engine flags HCE as "a distinct dimension-specific property not
captured by the current observer_score."

The M6 *secondary* claim (coh > shuffled) **does NOT replicate** at
scale: only 1/6 rules show coh > shuf. The single-rule M6 result was
a snapshot-mechanics artifact at short horizons. **What's special
about 4D is having the hidden state at all** — not specifically
what kind of hidden organization it has.

---

## 11. M7 HCE-guided rule search with anti-artifact safeguards (~10 min)

> **Why run this?** M6/M6B/M6C established that HCE is real, replicable,
> candidate-local, and not just a projection-threshold artifact. But
> the rules being measured were *not* selected for HCE — they came
> from M4A (viability) and M4C (generic observer_score). Can we evolve
> rules whose projected candidates are *simultaneously*
> observer-like AND hidden-causally-dependent — without exploiting
> any of the failure modes M6/M6C surfaced (threshold artifacts,
> global chaos that swamps local effects, fragile candidates that
> die from any perturbation, swarms of degenerate near-zero-area
> tracks, train-seed overfit)?
>
> M7 doesn't try to prove "4D > 2D" on generic observer_score —
> M4D already showed that's false under fair comparison. M7 asks the
> narrower, well-defined question: under the dimension-specific HCE
> quantity, can we find rules with both observer-likeness and HCE?

```bash
# Step 1: evolve (~3-4 min on moderate config; full defaults are ~1 hour)
python -m observer_worlds.experiments.evolve_4d_hce_rules \
    --strategy evolve --population 12 --generations 4 --lam 12 \
    --train-seeds 2 --validation-seeds 2 \
    --train-base-seed 1000 --validation-base-seed 4000 --test-base-seed 3000 \
    --timesteps 100 --grid 32 32 4 4 \
    --max-candidates 5 --hce-replicates 2 --horizons 10 20 \
    --top-k 5 --backend numpy \
    --seed-population outputs/observer_search/m4c_evolve/leaderboard.json \
    --label m7_evolve
```

The CLI **enforces train/validation/test seed disjointness** at startup
and aborts if any overlap is detected. Train seeds are used during
evolution; validation seeds re-rank the top-K after evolution; test
seeds are reserved for the holdout step below.

```bash
# Step 2: holdout validation against M4A, M4C, optimized 2D (~3 min)
python -m observer_worlds.experiments.run_m7_hce_holdout_validation \
    --m7-rules outputs/m7_evolve_<UTC>/top_hce_rules.json \
    --m4c-rules outputs/observer_search/m4c_evolve/leaderboard.json \
    --m4a-rules outputs/rule_search/m4a_search/leaderboard.json \
    --optimized-2d-rules outputs/m4d_2d_evolve/top_2d_rules.json \
    --n-rules 3 --test-seeds 3000 3001 3002 \
    --timesteps 120 --grid 32 32 4 4 \
    --max-candidates 6 --hce-replicates 2 --horizons 10 20 \
    --backend numpy --label m7_holdout
```

**Real-run result** of this exact config (44 M7 candidates / 49 M4C
/ 46 M4A on test seeds disjoint from training):

| Source | mean_observer | mean_HCE | non-threshold HCE retention |
|---|---|---|---|
| **M7_HCE_optimized** | **+1.03** | **+0.297** | **86%** |
| M4C_observer_optimized | +0.80 | +0.133 | 57% |
| M4A_viability | +0.87 | +0.147 | 68% |

M7 achieves higher observer score AND 2.2× higher HCE AND better
non-threshold HCE retention than either M4A or M4C, on held-out
test seeds. Both interpretation rules fire:
> *M7 found non-threshold-mediated hidden causal dependence.*
> *HCE-guided search found candidates with both observer-like
> projected structure and hidden causal dependence.*

This is the strongest defensible positive finding in the framework.

---

## 12. M7B — production-scale validation (~10–60 min depending on scale)

> **Why run this?** M7's promising holdout used only 3 test seeds.
> Before treating the M7 result as established, we want to see it
> replicate at substantially larger scale, with frozen code/configs
> (so the result is bit-reproducible from a git commit + input rule
> file hashes), hard invariant enforcement (any candidate whose
> hidden-invisible perturbation accidentally produced a non-zero
> initial-projection-delta is flagged INVALID and excluded — this
> is the contract that makes HCE meaningful in the first place), and
> multi-level cluster bootstrap (resampling by rule, by seed, and by
> rule+seed jointly) so the CIs respect both kinds of correlation.
>
> M7B can either **confirm** the M7 result at scale, **show it
> doesn't replicate**, or surface one of four canonical failure modes
> (threshold artifact, global chaos, observer collapse, train→test
> overfit). Each outcome is useful; what's not useful is asserting
> M7 without testing it at scale.

```bash
# Smoke (~1 minute): all four sources at minimal scale
python -m observer_worlds.experiments.run_m7b_production_holdout \
    --m7-rules outputs/m7_evolve_<UTC>/top_hce_rules.json \
    --m4c-rules outputs/observer_search/m4c_evolve/leaderboard.json \
    --m4a-rules outputs/rule_search/m4a_search/leaderboard.json \
    --optimized-2d-rules outputs/m4d_2d_evolve/top_2d_rules.json \
    --quick --label m7b_smoke

# Production-scale per spec (~hours): 10 rules per source × 50 test seeds × T=500
python -m observer_worlds.experiments.run_m7b_production_holdout \
    --m7-rules outputs/m7_evolve_<UTC>/top_hce_rules.json \
    --m4c-rules outputs/observer_search/m4c_evolve/leaderboard.json \
    --m4a-rules outputs/rule_search/m4a_search/leaderboard.json \
    --optimized-2d-rules outputs/m4d_2d_evolve/top_2d_rules.json \
    --n-rules-per-source 10 \
    --test-seeds $(seq 5000 5049) \
    --timesteps 500 --max-candidates 40 --hce-replicates 5 \
    --horizons 5 10 20 40 80
```

The CLI **autodetects** the M7 evolve run's train/validation seeds
(walks up from `--m7-rules` looking for `config.json`) and **aborts
at startup** if any test seed overlaps train or validation. The
frozen manifest at `frozen_manifest.json` records the git commit +
dirty flag, command line, Python and package versions, SHA-256 hash
of every input rule file, and all seed sets — so a reader can
exactly reproduce the run.

The summary.md selects from 7 canonical paragraphs:
- *"M7 production validation supports the core claim..."* (strong success)
- *"M7 optimized hidden dependence but partially traded off generic observer-likeness."* (partial success)
- *"HCE and observer_score remain distinct objectives."*
- *"The M7 result did not replicate at production scale."* (failure: HCE collapse)
- *"M7 exploited projection-threshold sensitivity."* (failure: threshold artifact)
- *"The effect is candidate-local, not merely global hidden chaos."* (when local-vs-far positive)
- *"This does not invalidate the HCE result; it confirms that HCE is the dimension-specific contribution..."* (when 2D beats on observer)

---

## 10. M6C hidden-organization taxonomy (~2 minutes)

> **Why run this?** M6B confirmed HCE is real and replicable, but the
> mechanism is unclear. One critical alternative explanation is that
> mean-threshold projection creates artifacts: if a candidate's
> columns sit near the projection threshold (active_fraction ≈ 0.5),
> any hidden bit-flip easily flips the projection downstream, and
> "HCE" would reduce to "projection mechanics on threshold-marginal
> columns." We need to either show HCE depends mostly on threshold
> margin (in which case it's a projection-specific finding) or that
> HCE persists in candidates whose columns are *far* from threshold
> (in which case it's a stronger hidden-causal-substrate finding).
> M6C extracts 24 hidden features per candidate, runs a threshold
> audit, fits grouped-CV regression to identify which features
> predict HCE, and runs an ablation battery to separate which kind
> of hidden organization matters.

```bash
# ~2 minutes: 3 M4C + 3 M4A rules × 2 seeds × 10 candidates × 3 replicates × 3 horizons
python -m observer_worlds.experiments.run_m6c_hidden_organization_taxonomy \
    --rules-from outputs/m4c_evolve/leaderboard.json \
    --also-rules-from outputs/m4a_search/leaderboard.json \
    --n-rules 3 --seeds 2 --timesteps 150 \
    --grid 32 32 4 4 --max-candidates 10 --replicates 3 \
    --horizons 5 10 20 --backend numpy \
    --base-seed 3000 --label m6c
```

Outputs:
- `hidden_features.csv` — 24 features × candidate
- `hce_joined_features.csv` — features ⨯ HCE outcomes (the analysis-ready table)
- `correlation_table.csv`, `feature_importances.csv`, `threshold_audit.csv`,
  `ablation_results.csv`, `model_scores.json`
- 12 plots: scatter HCE vs each top feature, threshold-audit boxplot,
  predicted-vs-actual HCE, ablation effects, feature correlation heatmap
- `summary.md` with the canonical interpretation paragraphs

**Real-run result (this exact config, 107 candidates):**

Threshold audit:
| Subset | n | mean future_div | fraction > 0 |
|---|---|---|---|
| all | 107 | +0.040 | 90% |
| **near-threshold-fraction < 0.10** | 45 | **+0.016** | **80%** |
| mean-threshold-margin > 0.10 | 91 | +0.035 | 88% |

Top RF feature importances for HCE:
1. `hidden_temporal_persistence` (0.24)
2. `mean_hidden_spatial_autocorrelation` (0.11)
3. `mean_threshold_margin` (0.08)
4. `mean_hidden_variance` (0.08)
5. `near_threshold_fraction` (0.05)

Interpretation that fired:
> *HCE persists away from projection thresholds, supporting a stronger
> hidden-causal-substrate interpretation. Hidden causal dependence is
> associated with temporally coherent hidden structure. HCE is
> associated with hidden microstate complexity. Current feature set
> does not fully explain HCE; hidden dependence may be rule-specific
> or require richer descriptors.*

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

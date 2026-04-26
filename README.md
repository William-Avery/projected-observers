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

## Why each milestone exists

Every milestone responds to a specific concern raised by the previous
one. The narrative arc isn't "more code" — it's "the previous result
left a loophole; here is the experiment that closes it."

### M1 — Build the substrate

> **Why**: nothing else matters without a working pipeline. We need a
> 4D CA, a projection to 2D, and a way to identify and track
> "candidate observers" across frames before any of the metric
> machinery is meaningful.

Implements: 4D Moore-r1 binary CA (numpy + numba kernels), 4D→2D
mean-threshold projection, connected-component extraction with
interior/boundary/environment shells, greedy IoU+centroid tracker,
zarr storage for projected frames + 4D snapshots.

### M2 — Operationalize "observerhood" as measurable quantities

> **Why**: without numbers, the whole project is hand-waving. The
> spec lists five functional signatures of observerhood (temporal
> asymmetry, predictive memory, Markov-blanket selfhood,
> intervention-sensitive causality, perturbation resilience). Each
> needs a concrete computable estimator on tracked structures.

Implements: `time_score`, `memory_score`, `selfhood_score`,
`causality_score`, `resilience_score`, combined `observer_score`
with z-normalization across the candidate population so per-run
results are comparable.

### M3 — Establish controls

> **Why**: a 4D-only result means nothing without baselines.
> We need both (a) a fair 2D baseline (Conway's Life) and (b) a
> negative-control 4D simulation where coherent hidden-dimensional
> structure has been destroyed. Without these, any "high observer
> score" in 4D might just reflect "more dynamical complexity" rather
> than anything dimension-specific.

Implements: `run_2d_baseline.py` (Conway's Life), `run_shuffled_hidden_baseline.py`
(4D CA with z,w fibers permuted into ca.state every step), and
`analysis/summarize_results.py` for cross-condition comparison plots.

### M4A — Find rules that produce structure at all

> **Why**: M3 ran end-to-end but produced **0 candidates** in the 4D
> conditions because the heuristic default rule kills everything
> immediately. The hypothesis is unanswerable when the 4D side is
> empty. We need a search over the rule space to find rules that
> produce *anything* persistent and bounded.

Implements: `FractionalRule` family (5 continuous floats — easier to
search than discrete birth/survival sets), 7-component **viability
score** (extinction/saturation/frozen-world penalties; target-
activity/temporal-change/persistent-component/boundedness/diversity
rewards), random search with multi-seed evaluation. **Surfaced rules
that produce 30+ persistent candidates.**

### M4B — Test statistically, not from a single max score

> **Why**: an M3 single-run "max observer_score(4D) > max(2D)" finding
> would be misleading because more candidates → more shots at a high
> max. Need **paired** evaluation across many (rule, seed) pairs,
> with **bootstrap CIs**, **permutation tests**, **Cohen's d /
> Cliff's δ effect sizes**, and **win rates** on multiple summary
> metrics — not just max.

Implements: 75-run paired sweep (5 rules × 5 seeds × 3 conditions),
all 16 SUMMARY_METRICS, three pairwise comparisons (coh vs shuf, coh
vs 2D, shuf vs 2D), automatic interpretation engine that **flags the
track-count confound** when extreme-score metrics disagree with
normalized ones (`score_per_track`, `lifetime_weighted_mean_score`).

### M4C — Maybe the M4A rules just weren't good enough

> **Why**: M4A's viability score is a *cheap proxy*. It rewards
> "rules that produce candidates" but not "rules that produce
> *high-observer-score* candidates." If we want the strongest 4D
> result possible, we should search rule space using the actual
> observer_score as fitness.

Implements: `(μ+λ)` evolutionary search with Gaussian mutation on
the 5 fractional-rule floats, full M2 metric suite as fitness,
default `lifetime_weighted_mean_score` (track-count-resistant).
Found rules with positive lifetime-weighted observer fitness.

### M4D — Maybe the M4C "win" was just optimization effort

> **Why**: M4B-on-M4C-rules showed coherent 4D significantly beats
> Conway's Life on `score_per_track` (Cliff's δ = +0.76, 88% win
> rate). But this comparison is **unfair**: M4C-optimized 4D rules
> vs unoptimized 2D Life. The "4D advantage" might just be
> "optimization-effort advantage." We need (a) **held-out
> evaluation seeds** disjoint from training, and (b) an
> **equally-optimized 2D baseline** with matched compute budget.

Implements: `evolve_2d_observer_rules.py` (same M4C machinery on
2D fractional rules), `run_m4d_holdout_validation.py` (two passes:
vs fixed Life + vs optimized 2D, with held-out seed protocol that
**aborts** if eval seeds overlap training).
**Result: optimized 2D wins.** M4C's apparent 4D advantage was
optimization effort, not dimensionality.

### M5 — Are observer-candidates "agents" by their response to perturbation?

> **Why**: scoring tracks isn't enough. A persistent structure with
> high observer_score might still be passive matter. The functional
> definition of "agency" requires that **perturbing the structure**
> (vs perturbing its environment) has different consequences for
> the structure's future. We need to actually intervene and
> measure.

Implements: per-candidate intervention runner with 4 types
(`internal_flip`, `boundary_flip`, `environment_flip`,
`hidden_shuffle`); paired forward rollouts capture **per-step
divergence trajectories**; aggregate plots + per-candidate plots
+ resilience curves.

### M6 — Even if 4D doesn't beat 2D on generic observer_score, look for an effect 2D *cannot have*

> **Why**: M4D falsified the simple "4D > 2D on observer_score"
> claim under a fair comparison. But that doesn't rule out
> **dimension-specific** effects that are *by construction*
> impossible in 2D systems. The cleanest such test: a perturbation
> that is *invisible* in the 2D projection at t=0 (preserves the
> projected frame exactly) but changes the underlying 4D state.
> If the future projection diverges, that divergence is **causally
> attributable** to hidden-dimensional structure — and 2D systems
> have no hidden state to perturb, so this quantity is identically
> zero in 2D by construction.

Implements: `apply_hidden_shuffle_intervention` (z,w permutation,
preserves count → preserves mean-threshold projection at t=0),
paired rollouts measuring downstream divergence, the **HCE scalar**
as the headline metric. Single-rule, N=8 result was suggestive (coh
HCE +0.064, shuf HCE 0.000, coh wins 7/0).

### M6B — Does M6's effect replicate, and is it candidate-local?

> **Why**: M6 was N=8 candidates, single rule, single run. Three
> things needed checking: (a) does the effect replicate across
> many rules + seeds, (b) is it really *candidate-local* (or does
> any hidden perturbation anywhere have similar effect — in which
> case it's a global instability, not "self"-specific causal
> dependence), (c) does the apparent advantage of *coherent* over
> *shuffled* 4D dynamics survive at scale?

Implements: 4 new interventions (`one_time_scramble_local`,
`fiber_replacement_local`, `hidden_invisible_far`, `sham`) all
verified to preserve projection at t=0; **cluster-bootstrap by
rule** (not row-level — respects within-rule correlation);
3 candidate selection modes (top observer-score, top lifetime,
random eligible) so the result doesn't depend on extreme winners.

**Replicates the primary HCE finding** at N=244 (sign-test p < 0.0001,
86% wins). **Confirms candidate-locality** (local effect 75% larger
than far-mask control). **Falsifies the M6 secondary claim** that
coherent specifically > shuffled — both 4D conditions show
comparable hidden-perturbation responses; M6's apparent gap was a
snapshot-mechanics artifact.

### M6C — What hidden organization produces HCE? Is it just a threshold artifact?

> **Why**: M6B confirmed HCE is real, replicable, and candidate-local.
> But the mechanism is unclear, and one critical alternative
> explanation needs ruling out: **mean-threshold projection might
> create artifacts**. If most candidate columns have active_fraction
> ≈ 0.5 (right at the projection threshold), then any hidden bit-flip
> easily flips the projection downstream — and "HCE" would just be
> "projection mechanics applied to threshold-marginal columns."
> If we filter to candidates whose columns sit far from threshold
> and HCE survives, the finding is much stronger.

Implements: 24 hidden-state features per candidate (per-column +
candidate-aggregate + temporal); a **threshold-artifact audit** that
re-runs the HCE analysis on candidate subsets filtered by
`near_threshold_fraction`; **grouped-CV regression** (`Ridge` and
`RandomForest`) with `GroupKFold` by rule_id so models can't memorize
rule identity; and an ablation battery (`random_hidden_shuffle`,
`count_preserving_shuffle`, `spatial_destroying_scramble`,
`fiber_replacement`, `temporal_history_swap`, `sham`).

**Result**: HCE is **partially threshold-mediated** — strict filter
to candidates with no near-threshold columns drops mean future_div
by ~60% — but **not entirely**. 80% of far-from-threshold candidates
still show positive HCE (mean 0.016 vs 0.040 unfiltered). The
strongest RF feature importance is `hidden_temporal_persistence`
(0.24), with threshold features second-tier (0.05–0.08). The
interpretation engine fires *"HCE persists away from projection
thresholds, supporting a stronger hidden-causal-substrate
interpretation"* and *"Hidden causal dependence is strongly related
to hidden temporal dynamics; in this run, lower hidden temporal
persistence / higher volatility gave hidden perturbations more
leverage on the future."*

### M7 — Searching for hidden-supported projected observers

> **Why**: M6/M6B/M6C established that HCE is real, replicable,
> candidate-local, and not just a projection artifact. But the rules
> we'd been measuring were *not* selected for HCE — they came from
> M4A (viability) and M4C (generic observer_score). Can we evolve
> rules whose projected candidates are *simultaneously*
> observer-like AND hidden-causally-dependent? And critically —
> without the search exploiting any of the failure modes M6/M6C
> surfaced (threshold artifacts, global chaos, fragile candidates,
> degenerate near-zero-area swarms, train-seed overfit)?
>
> M7 does **not** try to prove 4D has a higher generic
> `observer_score` than optimized 2D — M4D already showed that's
> false under fair comparison. Instead, M7 asks: under the
> dimension-specific HCE quantity that 2D systems can't have, can we
> find rules that produce candidates with both functional
> observer-likeness AND robust local hidden-causal dependence?

Implements: composite fitness combining +observer_score, +HCE,
+local-hidden-effect, +lifetime, +recovery with **explicit penalties**
for −near_threshold_fraction, −excess_global_divergence, −fragility,
−degenerate-candidate-fraction, and a hard penalty on
non-zero-initial-projection-delta (regression guard); cheap multi-seed
HCE estimator during evolution; **train/validation/test seed split
protocol** with disjointness check; `(μ+λ)` evolutionary loop
optionally seeded from M4A or M4C leaderboard;
`run_m7_hce_holdout_validation.py` that compares M7 against M4A, M4C,
and an optimized-2D baseline on **test seeds disjoint from training**.

**Real-run result** (12 rules × 4 generations, seeded from M4C, then
held out on 3 disjoint test seeds against M4A/M4C/optimized-2D):

| Source | n_cand | mean_observer | mean_HCE | mean_near_thresh | non-threshold HCE retention |
|---|---|---|---|---|---|
| **M7_HCE_optimized** | 44 | **+1.03** | **+0.297** | 0.18 | **86%** at strictest filter |
| M4C_observer_optimized | 49 | +0.80 | +0.133 | 0.20 | 57% |
| M4A_viability | 46 | +0.87 | +0.147 | 0.18 | 68% |

M7 candidates show **higher observer score AND 2.2× higher HCE AND
better non-threshold HCE retention** than either M4A or M4C rules,
all on held-out seeds. The interpretation engine fires both
*"M7 found non-threshold-mediated hidden causal dependence"* and
*"HCE-guided search found candidates with both observer-like
projected structure and hidden causal dependence"*. This is the
strongest defensible positive finding the framework has produced.

### M7B — Production-scale validation

> **Why**: M7's promising holdout used only 3 test seeds, 3 rules per
> source, T=120, and 8 candidates each. Before treating the M7
> result as established, we need it to replicate at substantially
> larger scale, with **frozen code/configs** so the result is
> bit-reproducible, **hard invariant enforcement** on the
> initial-projection-delta = 0 contract that makes HCE meaningful,
> **multi-level cluster bootstrap** (rule, seed, rule+seed)
> respecting both rule and seed heterogeneity, and a
> **train→validation→test generalization gap** report so we can see
> whether the M7 fitness translated cleanly to test-seed performance.
>
> M7B is the **claim-hardening milestone**. It either confirms M7
> at production scale (a strong positive result) or surfaces one of
> four canonical failure modes (not replicated, threshold artifact,
> global chaos, fragility/observer collapse). Either outcome is
> useful; what's not useful is reporting M7 as established without
> running it.

Implements: hard initial-projection-delta invariant check (any row
that violates it is excluded; CLI aborts unless
`--allow-invariant-violation`); frozen manifest with git commit +
SHA-256 file hashes + seed sets; auto-detected M7 train/validation
seeds from the M7 evolve config; three-level cluster bootstrap;
Cliff's δ + rank-biserial + Cohen's d effect sizes;
generalization-gap reporting; 12 plots; 7 canonical interpretation
paragraphs covering every success/failure mode.

**Real-run result** (15 rules × 8 test seeds × 8 candidates × 2 replicates,
N=1086 measurements, 0 invariant violations):

| metric | M7 | M4C | mean diff | 95% CI (rule+seed) | perm p | Cliff's δ |
|---|---|---|---|---|---|---|
| **observer_score** | +0.99 | +0.65 | **+0.34** | [+0.24, +0.45] | **0.0005** | **+0.51** |
| **HCE (= future_div)** | +0.31 | +0.17 | **+0.14** | [+0.08, +0.19] | **0.0005** | **+0.45** |
| local_div | +0.38 | +0.32 | +0.06 | [+0.01, +0.12] | 0.003 | +0.16 |
| hidden_vs_far_delta | +0.33 | +0.28 | +0.05 | [+0.00, +0.10] | 0.019 | +0.14 |
| candidate_lifetime | 37 | 57 | -20 | [-29, -12] | 0.0005 | -0.30 |

vs M4A: M7 wins HCE +0.17 (p=0.0005, Cliff's δ +0.48), observer +0.29
(p=0.0005, Cliff's δ +0.42). **Threshold audit on M7**: at strict
filter (near_threshold_fraction < 0.10, n=155), M7 retains +0.274
future_div — **89% retention**, 81% positive. The advantage is
genuine, not threshold-mediated.

The interpretation engine fired both
*"M7 production validation supports the core claim: HCE-guided
evolution produces projected candidates that are both observer-like
and locally dependent on hidden state"* and *"The effect is
candidate-local, not merely global hidden chaos."*

**The single trade-off**: M7 candidates are shorter-lived (mean
lifetime 37 vs M4C's 57). This is exactly what the M7 fitness
expected — `w_hce=2.0, w_local=2.0` vs `w_life=0.75`, so the search
appropriately traded some lifetime for substantially more HCE. The
trade is recorded honestly in the headline table.

This is the **strongest defensible positive finding** the framework
has produced — production-scale, statistically significant on every
primary metric, frozen-manifest reproducible, zero invariant
violations, and resilient to the strictest threshold filter.

### M8 — How does hidden 4D state actually support projected candidates?

> **Why**: M7B established that M7 candidates show robustly larger HCE
> than baselines and that the effect survives the strictest threshold
> filter. But "HCE is real" is a behavioural claim — it tells us hidden
> state matters, not *how*. M8 decomposes the mechanism: which spatial
> region of the hidden fibers (interior, boundary, environment, far)
> actually carries the causal signal, when does that signal first
> surface in the 2D projection, what does the propagation look like in
> 4D XOR-mass over time, and which hidden features lead the visible
> divergence. Without that decomposition, we can defend "hidden state
> matters" but not "*here's the mechanism*."

Implements: six per-candidate analyses — (1) **per-column response
maps** (perturb one (x,y) column at a time, measure local divergence
at a fixed horizon, build a heatmap over the candidate footprint);
(2) **emergence timing** (dense short horizons to find the first
horizon at which local divergence exceeds ε); (3) **pathway tracing**
(per-step 4D and 2D XOR-mass between original and intervened
rollouts, plus spread radii from the candidate centroid);
(4) **mediation analysis** (paired hidden-shuffle interventions on
interior, boundary, environment, far masks plus visible-flip controls
on boundary/environment); (5) **feature dynamics** (lead-lag
correlation between candidate-aggregate hidden features and the
visible divergence trajectory); (6) **rule-based mechanism
classifier** assigning one of 7 classes (`boundary_mediated`,
`interior_reservoir`, `environment_coupled`, `global_chaotic`,
`threshold_mediated`, `delayed_hidden_channel`, `unclear`) per
candidate.

The CLI reuses M7B's frozen-manifest machinery (git commit + SHA-256
of input rule files + autodetected M7 train/validation seed sets +
disjointness check); writes 7 CSVs (`mechanism_labels.csv`,
`mediation_summary.csv`, `pathway_traces.csv`, `lifetime_tradeoff.csv`,
`response_maps.csv`, `feature_dynamics.csv`,
`mechanism_candidates.csv`); per-candidate response-grid `.npy`
arrays under `arrays/`; 12 plots; and a `summary.md` that selects
from 7 canonical interpretation paragraphs covering every
mechanism-class outcome.

The success criterion: **identify at least one non-artifact mechanism
by which hidden 4D state locally supports future projected 2D
candidate dynamics.**

> **Status**: M8 code is implemented; the 29 M8 tests pass and the
> 252-test full suite has no regressions. **Production-scale empirical
> conclusions are still pending.** A moderate-scale run has landed and
> is summarized below — it replicates the M7B HCE advantage and rules
> out threshold-mediated and global-chaotic explanations, but cannot
> cleanly disentangle boundary-mediated from interior-reservoir
> mechanisms at the candidate sizes this run produced.

**Moderate-scale M8 result** (3 sources × 3 rules × 5 test seeds × T=200 × grid 32×32×4×4, N=264 candidates, frozen-manifest reproducible):

| Source | n | mean obs | mean life | mean HCE | locality_idx | boundary-mediated | global_chaotic | threshold-mediated |
|---|---|---|---|---|---|---|---|---|
| **M7_HCE_optimized** | 84 | **+0.94** | 44 | **+0.082** | **+0.090** | **88%** | 10% | 1% |
| M4C_observer_optimized | 90 | +0.64 | 92 | +0.040 | +0.044 | 73% | 24% | 1% |
| M4A_viability | 90 | +0.60 | 75 | +0.048 | +0.051 | 70% | 22% | 0% |

Paired M7 vs M4C: HCE +0.042 (p=0.001, Cliff's δ +0.59), candidate_locality_index +0.047 (p=0.001, Cliff's δ +0.52), candidate_lifetime −48 (p=0.001). Paired M7 vs M4A: HCE +0.034 (p=0.001, Cliff's δ +0.34), locality +0.039 (p=0.001).

What this rules out and what it shows:
- ✅ **HCE is not threshold-mediated** in M7. M7's HCE↔near_threshold_fraction correlation is +0.13, vs M4C's +0.49 and M4A's +0.64. M7 evolution successfully decoupled HCE from threshold sensitivity.
- ✅ **HCE is not predominantly global chaos** in M7. Only 10% of M7 candidates classify as `global_chaotic` (vs M4C 24%, M4A 22%), and M7's `candidate_locality_index` is ~2× the baselines' (p=0.001).
- ✅ **M7 broke the HCE↔lifetime tradeoff** that holds in baselines. M4C and M4A show HCE↔lifetime correlations of −0.55 and −0.58 respectively (high-HCE candidates die faster), but M7 shows +0.07 — essentially decoupled.
- ✅ **HCE is candidate-localized** in M7: HCE↔boundary_response_fraction = +0.65 (vs M4C +0.27, M4A +0.43).
- ⚠️ **Boundary vs interior mediation cannot be distinguished at this candidate size**. Most M7 candidates are <10 cells (median lifetime 39 frames, median area small). For thin candidates `_shell_masks` falls back to using the entire interior_mask as the boundary, which structurally pegs `boundary_mediation_index ≈ 0.50` across all sources and mechanism classes. The 88% `boundary_mediated` label reflects "response is on the candidate (interior or boundary), not on environment or far cells" — not a clean boundary-vs-interior decomposition.
- ⚠️ **Environment-coupling cannot be ruled out**. The mediation analysis finds `environment_hidden_effect = +0.168` for M7, ~2× larger than `interior_hidden_effect = +0.083`. The current response-map only probes (interior ∪ boundary), so this wouldn't trigger the `environment_coupled` classifier — this is a real limitation flagged for the production run.

The moderate-scale run is therefore **positive on HCE replication, candidate-locality, and absence-of-artifact**, but **inconclusive on which spatial sub-region of the candidate carries the mechanism**. A production-scale run (larger grid, longer T, longer-lived candidates so erosion produces a non-empty interior) is needed to cleanly classify the mechanism. The M8 code milestone stands; the specific mechanism class label is pending production validation.

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

## Glossary

### Acronyms

| Term | Meaning |
|---|---|
| **HCE** | **Hidden Causal Effect** — downstream projected divergence under a perturbation that leaves the 2D projection byte-identical at t=0. Identically zero in 2D systems by construction; positive in 4D iff hidden state has causal weight on the projected future. The framework's primary dimension-specific quantity. |
| **CA** | Cellular Automaton |
| **CI** | Confidence Interval (bootstrap, 95% by default) |
| **CSV** | Comma-Separated Values (run output format) |
| **L1** | L1-norm distance, equivalent to mean per-cell absolute difference between two 2D projected frames |
| **AUC** | Area-Under-Curve of a divergence trajectory over rollout steps |
| **MSE** | Mean Squared Error (used by Ridge regression in M2 metrics) |
| **IoU** | Intersection-over-Union (used by the tracker to match components frame-to-frame) |

### Core technical terms

| Term | Meaning |
|---|---|
| **4D Moore-r1 neighborhood** | The 80 neighbors of a cell in the 4D grid (3⁴ - 1 = 80) |
| **Mean-threshold projection** | `Y(x,y) = 1 if mean_{z,w}(X(x,y,z,w)) > 0.5 else 0`. Maps 4D state to 2D frame; depends only on per-(x,y) active count |
| **Fractional totalistic rule** | A 4D CA rule defined by 5 floats: `birth_min`, `birth_max`, `survive_min`, `survive_max` (all in [0,1] over neighbor *fraction*), and `initial_density`. Cell births if `birth_min ≤ rho ≤ birth_max` where `rho = active_neighbors / 80`; survives if `survive_min ≤ rho ≤ survive_max` |
| **Track / candidate / observer-candidate** | A connected component in the 2D projected world that persists across multiple frames. *Observer-candidate* = a track passing the persistence filter (age ≥ 10 frames, bounded area, non-trivial internal variation) |
| **Interior / boundary / environment shells** | For each candidate's 2D mask: interior = mask eroded by 1, boundary = mask XOR interior, environment = dilation of mask minus mask. Used by the M2 selfhood score and the intervention experiments |
| **Coherent 4D** | The standard 4D CA — natural dynamics with hidden (z,w) state evolving according to the rule |
| **Per-step-shuffled 4D** | Negative control: same rule, but z,w fibers are randomly permuted into `ca.state` at every step. Per-(x,y) counts preserved, hidden geometry destroyed |

### M2 metric vocabulary

| Term | Meaning |
|---|---|
| **time_score** | Backward predictive error − forward predictive error (Ridge + KFold). High score = future is more predictable from current state than past is. Tests **temporal asymmetry** |
| **memory_score** | `error(predict S_{t+k} \| S_t) − error(predict S_{t+k} \| S_t, I_t)`. High = candidate's internal state adds predictive power for the future. Tests **predictive memory** |
| **selfhood_score** | `R²(B → I) − max(0, R²((B,E) → I) − R²(B → I))` — high = boundary mediates environment-internal interaction (Markov-blanket style). Tests **boundary mediation** |
| **causality_score** | Aggregate paired-rollout divergence under 4 intervention types (`internal_flip`, `boundary_flip`, `environment_flip`, `hidden_shuffle`). High = candidate-targeted perturbations matter more than environment perturbations |
| **resilience_score** | Survival + area_recovery + centroid_continuity + shape_similarity after a candidate-interior perturbation. High = candidate recovers from damage |
| **observer_score** | Z-normalized weighted sum of the five components above, computed across the run's candidate population |
| **score_per_track** | `sum(observer_scores) / max(num_tracks, 1)` — track-count-resistant aggregate. Used as the primary cross-condition comparison metric |
| **lifetime_weighted_mean_score** | `sum(observer_score × age) / sum(age)` — long-lived candidates dominate. Default fitness for M4C |
| **top5_mean_score / p95_score** | Top-5-mean and 95th-percentile of observer_scores in a run. *Confounded* by candidate count (more candidates → more shots at extremes) |

### Intervention types (M5 + M6 + M6B)

| Term | Effect on 2D projection at t=0 | Effect on 4D state |
|---|---|---|
| **sham** | unchanged | unchanged (identity) |
| **internal_flip** *(M5)* | usually changes | random bit-flips in the candidate's interior 4D fibers |
| **boundary_flip** *(M5)* | usually changes | random bit-flips in the candidate's boundary 4D fibers |
| **environment_flip** *(M5)* | usually changes | random bit-flips in the candidate's environment shell 4D fibers |
| **hidden_shuffle** / **hidden_invisible_local** *(M2/M5/M6/M6B)* | **byte-identical** | z,w cells permuted within each (x,y) column under the candidate's interior |
| **one_time_scramble_local** *(M6B)* | **byte-identical** | z,w fibers replaced with fresh uniform-random arrangement at matched count |
| **fiber_replacement_local** *(M6B)* | **byte-identical** | z,w fibers swapped with bucket-matched fibers from elsewhere on the grid |
| **hidden_invisible_far** *(M6B)* | **byte-identical** | hidden_invisible applied to a translated mask at the antipodal grid location (localization control) |
| **visible_match_count** *(M6/M6B)* | small but non-zero | random bit-flips in interior 4D fibers, count matched to whatever hidden_invisible flipped (visible control) |

### Statistical vocabulary

| Term | Meaning |
|---|---|
| **Cluster-bootstrap by rule** | Bootstrap resampling done at the *rule* level (not row level) so within-rule correlation doesn't artificially tighten CIs |
| **Cliff's δ** | Effect size = P(A > B) − P(A < B). Range [-1, +1]; +1 means A always beats B |
| **Cohen's d (paired)** | Effect size = mean(diffs) / std(diffs). Standard small/medium/large = 0.2/0.5/0.8 |
| **Permutation p-value** | Two-sided p computed by randomly flipping condition labels per pair and counting permutations as extreme as observed |
| **Sign-test p-value** | Two-sided binomial test on the count of paired differences > 0 vs ≤ 0 |
| **Win rate** | Fraction of paired comparisons where condition A wins |
| **Track-count confound** | Bigger N → more shots at high max scores. Why we report `score_per_track` and `lifetime_weighted_mean_score` as primary metrics rather than max/top5 |
| **Held-out seeds** | Evaluation seeds disjoint from those used during rule optimization (M4D enforces this; aborts on overlap) |
| **Optimized 2D baseline** | A 2D fractional rule found by running the same M4C-style observer-fitness search on the 2D rule family. Used by M4D to remove the "optimization-effort confound" from coh-vs-2D comparisons |

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
| **M6 (Hidden Causal Dependence)** | coh-4D HCE vs shuffled-4D HCE *(N=8, single rule)* | suggestive: coh +0.064, 95% CI [+0.036, +0.092], coh 7/0 |
| **M6B (HCE replication, N=244 across 6 rules from both M4A + M4C)** | **coh local hidden vs sham, on raw future divergence** | **robust**: +0.039, 95% CI [+0.013, +0.070], **210/244 wins, sign p < 0.0001** |
| M6B | **coh local hidden vs far hidden, on candidate-local divergence** | **localized**: +0.239, 95% CI [+0.152, +0.335], 183/244 wins, sign p < 0.0001 |
| M6B | coh-4D HCE vs per-step-shuffled-4D HCE *(does the M6 effect depend on coherent dynamics?)* | **does NOT replicate**: only 1/6 rules show coh > shuf, diff −0.006. M6's single-rule "shuffled HCE ≈ 0" was a snapshot-mechanics artifact. |
| **M6C (Hidden Organization Taxonomy)** | **Is HCE primarily a projection-threshold artifact?** Run candidate-level threshold audit with feature regressions | **Partially threshold-mediated, but not entirely**: HCE drops ~40-60% under strict near-threshold filters but **80% of far-from-threshold candidates still show positive HCE**. RF importances rank `hidden_temporal_persistence` (0.24) > threshold features (0.05-0.08). HCE *is* a real hidden-causal property; threshold sensitivity is one channel through which it manifests, not the whole story. |
| **M7 (HCE-guided rule search + holdout validation)** | **Can we evolve 4D rules where projected candidates are simultaneously observer-like AND hidden-causally-dependent, without exploiting threshold artifacts?** | **Yes.** M7 evolution (composite fitness with anti-artifact penalties) found rules where on **held-out test seeds**: M7 candidates have **higher observer_score** (+1.03 vs M4C +0.80, M4A +0.87) AND **2.2× higher HCE** (+0.297 vs +0.133). Threshold audit on M7 keeps **86% of its HCE under the strictest threshold filter** (vs M4C's 57%). Both interpretation rules fired: *"M7 found non-threshold-mediated hidden causal dependence"* and *"HCE-guided search found candidates with both observer-like projected structure and hidden causal dependence"*. |
| **M7B (production-scale validation)** | **Does the M7 result replicate at substantially larger scale, with frozen code/configs, hard invariant enforcement on initial_projection_delta, and three-level cluster bootstrap?** | **Yes — strongly.** N=1086 measured rows across 15 rules × 8 test seeds × 8 candidates × 2 replicates × 2 horizons, **0 invalid rows** under the hard invariant. M7 vs M4C: HCE +0.135 (p=0.0005, Cliff's δ +0.45, 72% win), observer +0.341 (p=0.0005, Cliff's δ +0.51, 76% win). M7 vs M4A: HCE +0.174 (p=0.0005). M7's 89% non-threshold HCE retention. **Strong-success interpretation paragraph fired.** |

**Headline finding (generic observer_score)**: when the 2D baseline is
also observer-fitness-optimized (matched compute budget, same fitness,
same metric suite), the 4D advantage **disappears** and slightly
reverses. The previous positive result against fixed Conway's Life was
an artifact of comparing optimized 4D rules against an unoptimized 2D
rule.

**Headline finding (M6+M6B — hidden causal dependence)**:

Every candidate gets a `hidden_invisible` perturbation: z,w cells in the
candidate's interior are permuted, leaving the mean-threshold 2D projection
**byte-identical at t=0**. Any downstream projected divergence is therefore
causally attributable to hidden-dimensional structure — a quantity 2D
systems cannot have by construction.

M6 (single rule, N=8 candidates) found a directional positive effect with
the 4D-coherent vs hidden-shuffled comparison statistically suggestive.
**M6B replicated the experiment at scale** (N=244 paired measurements
across 6 rules from both M4A *viability* and M4C *observer-fitness*
sources, 3 seeds each, 5 candidates per selection mode × 3 modes,
multiple horizons, with proper cluster-bootstrap by rule):

* **HCE survives replication**: coh local hidden vs sham mean diff
  **+0.039 [+0.013, +0.070], 86% win rate, sign-test p < 0.0001**.
* **Effect is candidate-local**: coh local hidden vs FAR hidden on
  candidate-region divergence: **+0.239, 75% win rate, p < 0.0001**.
* **HCE / observer_score correlation = +0.22** — *positive but weak*.
  The interpretation engine selects the canonical sentence "Hidden-causal
  dependence is a distinct dimension-specific property not captured by
  the current observer_score."
* **The M6 *secondary* claim (coh > shuffled on HCE) does NOT replicate**.
  Coherent vs per-step-shuffled on `hidden_invisible_local`: only 1/6
  rules show coh > shuf. M6's apparent "shuffled HCE ≈ 0" was a
  snapshot-mechanics artifact (the per-step shuffle drives columns to
  uniform-random, so further shuffling at the snapshot is near no-op
  *at short horizons*; at h ≥ 10 with proper rollout, both conditions
  produce comparable hidden-perturbation responses).

**Net result**: hidden causal dependence is **robust and candidate-local**
across rules and seeds; the effect is real and reproducible. But it
**does not specifically require coherent 4D dynamics** — both coherent
and per-step-shuffled 4D dynamics show comparable hidden-causal responses.
The framework's primary novelty stands: a quantity that is *zero by
construction in 2D* and *positive by direct measurement in 4D*. The
question of which kinds of hidden organization are responsible remains
open — a natural M6C / M7 follow-up.

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

**M1 + M2 + M3 + M4A + M4B + M4C + M4D + M5 + M6 + M6B + M6C + M7 + M7B are complete.** The framework now implements:

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
- **M6B** (shipped): **HCE replication at scale + stronger controls**.
  Adds three new interventions (`one_time_scramble_local`,
  `fiber_replacement_local`, `hidden_invisible_far`) plus a sham
  baseline. Runs across multiple rules + multiple seeds + multiple
  candidates × 3 selection modes (top observer-score, top lifetime,
  random eligible) × multiple horizons × multiple replicates. Reports
  raw future divergence, sham-subtracted, far-control-subtracted,
  hidden-vs-visible ratio — never letting the HCE-as-ratio dominate
  interpretation. Cluster-bootstrap by rule. Replicates the M6 primary
  finding and falsifies the M6 secondary claim.
- **M6C** (shipped): **hidden-organization taxonomy**. Now that we
  know HCE is real and replicates, what hidden-state properties
  produce it? The framework extracts 24 hidden features per candidate
  (per-column: active fraction, distance to threshold, entropy,
  spatial autocorrelation, connected components in z,w; cross-column:
  heterogeneity, fiber-fiber correlation; temporal: hidden
  persistence and volatility), runs a **threshold-artifact audit**
  on filtered candidate subsets, fits Ridge + RandomForest with
  **GroupKFold by rule_id** for cross-validated feature importances,
  and runs an ablation battery (random shuffle / count-preserving /
  spatial-destroying / fiber-replacement / temporal-history-swap /
  sham). Surfaces which hidden properties drive HCE.
- **M7** (shipped): **HCE-guided rule search with anti-artifact
  safeguards + held-out validation**. Composite fitness with
  positive terms (observer_score, hidden_vs_sham_delta,
  hidden_vs_far_delta, lifetime, recovery) and explicit penalties
  for the failure modes M6/M6C surfaced (threshold artifacts,
  excess global divergence, fragility, degenerate candidates,
  non-zero initial projection delta). Train/validation/test seed
  split protocol enforced by a disjointness check.
  `run_m7_hce_holdout_validation.py` compares M7 vs M4A vs M4C vs
  optimized 2D on test seeds.
- **M7B** (shipped): **production-scale claim-hardening**.
  Re-runs the M7 holdout at larger scale with hard invariant
  enforcement on `initial_projection_delta` (any non-zero value
  flags rows INVALID and excludes them from interpretation),
  **frozen manifest** (git commit + dirty flag + SHA-256 of input
  rule files + Python and package versions + auto-detected M7
  train/validation seeds), **three-level cluster bootstrap** (by
  rule, by seed, by rule+seed), multiple effect-size measures
  (Cliff's δ, rank-biserial, Cohen's d), generalization-gap
  reporting (M7 train → validation → production-test fitness), 12
  diagnostic plots, and 7 canonical interpretation paragraphs that
  cover every success/failure mode in the spec.
- **M5** (shipped): **per-candidate intervention experiments** — for top
  observer-candidates, applies all 4 intervention types
  (`internal_flip`, `boundary_flip`, `environment_flip`,
  `hidden_shuffle`) at a snapshot inside each candidate's lifetime, runs
  paired forward rollouts, captures **per-step divergence trajectories**
  (not just aggregates), produces per-candidate divergence + resilience
  plots, aggregate plots, and an intervention heatmap.
- **M8** (shipped): **mechanism discovery**. Decomposes the M7B HCE
  finding into six per-candidate analyses — per-column response maps
  (which spatial region of the hidden fibers carries the causal
  signal), emergence timing (when does hidden state first surface in
  the projection), pathway tracing (per-step 4D and 2D XOR-mass and
  spread radii), mediation analysis (interior/boundary/environment/far
  paired hidden-shuffle), feature dynamics (5 hidden features →
  visible divergence lead-lag), and a rule-based mechanism classifier
  with 7 classes (boundary_mediated, interior_reservoir,
  environment_coupled, global_chaotic, threshold_mediated,
  delayed_hidden_channel, unclear). Reuses M7B's frozen-manifest
  machinery (git commit + SHA-256 + auto-detected M7 seed-split
  disjointness check). 12 plots, 7 CSVs, per-candidate response
  `.npy` arrays, and 7 canonical interpretation paragraphs.

## Run M7B production-scale validation

```bash
# Smoke (~1 minute): all four sources at quick scale
python -m observer_worlds.experiments.run_m7b_production_holdout \
    --m7-rules outputs/m7_evolve_<UTC>/top_hce_rules.json \
    --m4c-rules outputs/observer_search/m4c_evolve/leaderboard.json \
    --m4a-rules outputs/rule_search/m4a_search/leaderboard.json \
    --optimized-2d-rules outputs/m4d_2d_evolve/top_2d_rules.json \
    --quick --label m7b_smoke

# Production (~hours, per spec): 10 rules per source × 50 test seeds × T=500
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

The CLI **autodetects** M7's training and validation seeds from the
M7 evolve run's `config.json` (walks up from `--m7-rules`) and
**aborts** if test seeds overlap. The frozen manifest records the
git commit, dirty flag, command-line invocation, Python and package
versions, SHA-256 of every input rule file, and all seed sets — so
the result is reproducible bit-for-bit.

The summary.md selects from seven canonical interpretation paragraphs
covering every success/failure mode (strong-success, partial-success,
distinct-objectives, not-replicated, threshold-artifact,
local-not-global, 2D-beats-on-observer).

## Run M8 mechanism discovery

```bash
# Smoke (~1 minute): one M7 rule, two test seeds, T=80, grid 16x16x4x4
python -m observer_worlds.experiments.run_m8_mechanism_discovery \
    --m7-rules outputs/m7_evolve_<UTC>/top_hce_rules.json \
    --quick --label m8_smoke

# Moderate (~10 minutes): all three sources, 5 seeds, T=200, grid 32x32x4x4
python -m observer_worlds.experiments.run_m8_mechanism_discovery \
    --m7-rules outputs/m7_evolve_<UTC>/top_hce_rules.json \
    --m4c-rules outputs/observer_search/m4c_evolve/leaderboard.json \
    --m4a-rules outputs/rule_search/m4a_search/leaderboard.json \
    --n-rules-per-source 3 \
    --test-seeds 6000 6001 6002 6003 6004 \
    --timesteps 200 --grid 32 32 4 4 \
    --max-candidates 6 --hce-replicates 2 \
    --horizons 1 2 5 10 20 40 \
    --backend numpy --label m8_real

# Production (~hours, per spec): 5 rules per source × 20 test seeds × T=500
python -m observer_worlds.experiments.run_m8_mechanism_discovery \
    --m7-rules outputs/m7_evolve_<UTC>/top_hce_rules.json \
    --m4c-rules outputs/observer_search/m4c_evolve/leaderboard.json \
    --m4a-rules outputs/rule_search/m4a_search/leaderboard.json \
    --n-rules-per-source 5 \
    --test-seeds $(seq 6000 6019) \
    --timesteps 500 --max-candidates 20 --hce-replicates 3 \
    --horizons 1 2 3 5 10 20 40 80
```

Default test seeds (6000–6019) are disjoint from M7 train (1000–1004),
M7 validation (4000–4001), and M7B test (5000–5049). The CLI
autodetects M7's seed splits from the M7 evolve `config.json` and
**aborts** if test seeds overlap. The frozen manifest captures git
commit, dirty flag, command-line invocation, Python and package
versions, SHA-256 of every input rule file, and all seed sets.

Mechanism-class outputs are interpreted by 7 canonical paragraphs:
- *"M7's hidden causal dependence is primarily mediated by hidden state under projected candidate boundaries."* (boundary_mediated dominates)
- *"M7 candidates appear to use hidden fibers as latent internal state reservoirs."* (interior_reservoir dominates)
- *"Hidden perturbations propagate invisibly before becoming visible, supporting a hidden-channel mechanism."* (delayed_hidden_channel dominates)
- *"M7's HCE may reflect broad instability rather than candidate-local hidden support."* (global_chaotic dominates — bad outcome)
- *"M8 confirms M7's HCE is not primarily threshold-mediated."* (threshold_mediated < 15%)
- *"Hidden causal sensitivity increases observer-like dependence but reduces persistence."* (HCE↔lifetime negative correlation)
- *"M8 identifies a promising subpopulation of stable hidden-supported projected observers."* (M7 mean lifetime > 30 AND mean HCE > 0.10)

## Run M7 HCE-guided rule search + held-out validation

```bash
# 1. Evolve HCE-aware rules (~3-4 minutes for the moderate config below;
#    the spec's defaults of pop=40, gens=20, T=300 are 1+ hour)
python -m observer_worlds.experiments.evolve_4d_hce_rules \
    --strategy evolve --population 12 --generations 4 --lam 12 \
    --train-seeds 2 --validation-seeds 2 \
    --train-base-seed 1000 --validation-base-seed 4000 --test-base-seed 3000 \
    --timesteps 100 --grid 32 32 4 4 \
    --max-candidates 5 --hce-replicates 2 --horizons 10 20 \
    --top-k 5 --backend numpy \
    --seed-population outputs/observer_search/m4c_evolve/leaderboard.json \
    --label m7_evolve

# 2. Holdout validation on test seeds 3000–3002 (disjoint from training)
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

The holdout `summary.md` selects from canonical interpretation paragraphs:
- *"M7 found non-threshold-mediated hidden causal dependence"* (when far-from-threshold HCE remains positive)
- *"HCE-guided search found candidates with both observer-like projected structure and hidden causal dependence"* (when M7 ≥ M4C on observer)
- *"HCE optimization exploited projection-threshold sensitivity"* (when far-from-threshold HCE collapses)
- *"The search found globally chaotic hidden sensitivity, not candidate-local hidden support"* (when global ≫ local)

## Run M6C hidden-organization taxonomy (recommended after M6B)

```bash
# ~2 minutes: 3 M4C + 3 M4A rules × 2 seeds × 10 candidates × 3 replicates × 3 horizons.
# Computes 24 hidden features per candidate, runs threshold audit,
# grouped-CV regression, and ablation battery.
python -m observer_worlds.experiments.run_m6c_hidden_organization_taxonomy \
    --rules-from outputs/observer_search/m4c_evolve/leaderboard.json \
    --also-rules-from outputs/rule_search/m4a_search/leaderboard.json \
    --n-rules 3 --seeds 2 --timesteps 150 \
    --grid 32 32 4 4 --max-candidates 10 --replicates 3 \
    --horizons 5 10 20 --backend numpy \
    --base-seed 3000 --label m6c
```

Outputs `summary.md` with the threshold audit, top-correlated features,
GroupKFold-CV regression scores, RF feature importances, and one of
six canonical interpretation paragraphs (or a combination) selected
based on which features survive significance.

## Run M6B replication (recommended over M6)

```bash
# ~7 minutes: 3 M4C + 3 M4A rules × 3 seeds × ~5 candidates each ×
# 3 selection modes × 6 interventions × 3 replicates × 3 horizons
python -m observer_worlds.experiments.run_m6b_hidden_causal_replication \
    --rules-from outputs/observer_search/m4c_evolve/leaderboard.json \
    --also-rules-from outputs/rule_search/m4a_search/leaderboard.json \
    --n-rules 3 --seeds 3 --timesteps 150 \
    --grid 32 32 4 4 --max-candidates 5 --replicates 3 \
    --horizons 5 10 20 --backend numpy \
    --base-seed 2000 --label m6b
```

Outputs `summary.md` with cluster-bootstrap CIs and an interpretation
that selects from a small set of canonical paragraphs based on which
controls survive significance. Per the M6B spec, *raw* future
divergence + sham-subtracted + far-control-subtracted are the primary
reportable quantities; the HCE-as-ratio is exposed but not headlined.

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

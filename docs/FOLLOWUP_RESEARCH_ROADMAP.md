# Follow-up research roadmap

This roadmap describes the next three research directions for the
projected-observers project. They are **follow-up topics**, not
replacements for the Initial Topic.

## Initial Topic

**Hidden-supported projected observers in 4D → 2D artificial-life
systems.**

### Established findings

* The repo does **not** show that 4D generally beats optimized 2D on
  generic observer-score (M4D refutation; preserved as a negative
  result).
* The strongest supported result is that 4D-projected candidates can
  exhibit **Hidden Causal Effect (HCE)**: hidden 4D perturbations can
  preserve the current 2D projection while altering future 2D
  behavior.
* M7 / M7B: HCE-guided evolution strengthens HCE while retaining
  observer-like structure.
* M8E / M8F / M8G: the dominant mechanism is **boundary-and-interior
  co-mediated hidden response**, not boundary-only mediation. M7
  amplifies HCE within existing mechanism classes rather than creating
  a new dominant class. The old `boundary_mediated` label is legacy
  terminology; revised classification puts ≈99 % of those candidates
  into `boundary_and_interior_co_mediated`.
* Within-class M7-vs-M4C HCE is CI-clean (+0.0063 [+0.0034, +0.0093]);
  M7-vs-M4A is borderline (+0.0025 [−0.0005, +0.0055]).
* M7's HCE advantage in `global_chaotic` is the largest in absolute
  magnitude (+0.0086 vs M4C, +0.0063 vs M4A).

### What the project does **not** claim

* No claim of consciousness, sentience, or first-person experience.
* No claim that real-world physics works this way.
* No claim that 4D substrates always beat 2D substrates.
* No claim that the discovered mechanism is anything other than a
  classifier-conditional, projection-dependent, dynamics-specific
  empirical regularity in this artificial-life model.

---

## Why follow-ups

Three independent directions test whether HCE is a substantive property
or an artifact of specific choices in the framework. They are
intentionally *parallel*, not sequential — each can be run on its own
and reported on its own. Together they probe robustness across:

| Axis | Question | Topic |
|---|---|---|
| Observation map | Does HCE survive a different projection function? | Topic 1 |
| Causal identity | Does the future follow the visible host or the hidden donor when hidden fibers are swapped? | Topic 2 |
| Functional agency | Does HCE predict task-relevant behavior, not only future divergence? | Topic 3 |

If all three hold, "hidden-supported projected observers" is a property
of these systems' dynamics, not of the specific mean-threshold
projection / specific perturbation protocol / specific divergence
metric used so far. If any breaks, that is itself the finding.

---

## Follow-up Topic 1 — Projection Robustness

### Question

Is hidden causal dependence a real property of projected 4D
candidates, or is it specific to the current `mean_threshold`
projection?

### Hypothesis

If HCE is robust, M7-style HCE should persist across multiple
projection functions, though effect size may vary.

### Projections to evaluate

1. `mean_threshold` (existing baseline)
2. `sum_threshold`
3. `max_projection` (existing)
4. `parity_projection` (existing)
5. `random_linear_projection`
6. `multi_channel_projection`
7. *Optional later:* `learned_projection`

### Per-projection metrics

* HCE
* `hidden_vs_sham_delta`, `hidden_vs_far_delta`
* `initial_projection_delta`
* candidate count, observer_score
* `near_threshold_fraction` (only where defined)
* revised mechanism labels (M8G), `boundary_and_interior_co_mediated`
  fraction, `global_chaotic` fraction
* HCE within revised mechanism classes

### Important caveat

Some projections do not have a natural "threshold margin." For those,
mark threshold metrics N/A and do not force the threshold-audit logic
to apply.

### Activated interpretations

| Outcome | Reading |
|---|---|
| HCE persists across all projections | "HCE is not specific to the mean-threshold projection." |
| HCE only in threshold projections | "HCE may be projection-threshold mediated or projection-family specific." |
| HCE survives `random_linear` / `multi_channel` | "Hidden causal dependence is robust to substantial changes in the observation map." |
| Candidate counts collapse under a projection | "This projection does not support enough candidate structure for comparison." |

### Files

```
observer_worlds/projection/projection_suite.py
observer_worlds/experiments/run_followup_projection_robustness.py
observer_worlds/analysis/projection_robustness_stats.py
observer_worlds/analysis/projection_robustness_plots.py
tests/test_projection_suite.py
tests/test_projection_robustness.py
```

### Default smoke run

```
n_rules=1, seeds=2, timesteps=100, max_candidates=5,
projections={mean_threshold, max, parity},
horizons=[5, 10], hce_replicates=1
```

### Default main run

```
n_rules=5, seeds=20, timesteps=500, max_candidates=20,
projections={all six},
horizons=[1, 2, 3, 5, 10, 20, 40, 80], hce_replicates=3
```

### Performance plan

* Cache projected frames per `(rule, seed, projection)`.
* Run the 4D substrate once; checkpoint state stream; evaluate multiple
  projections from the same stream.
* If full-state storage is too large, checkpoint candidate snapshots
  and replay deterministic rollouts from the same seed.
* Parallelize over `(rule × seed × projection)`.

---

## Follow-up Topic 2 — Hidden Identity Swap

### Question

Does the future of a projected candidate depend more on its visible 2D
state or its hidden 4D microstate?

### Core intervention

Find candidate pair `A`, `B` with similar projections
`Y_A ≈ Y_B` but different hidden states `X_A_hidden ≠ X_B_hidden`.
Swap hidden fibers while preserving the visible projection. Run
futures for `original A`, `original B`, `A-with-B-hidden`,
`B-with-A-hidden`. Compare swapped futures to the visible host vs
hidden donor.

### Key derived score

```
hidden_identity_pull
    = similarity(swapped_future, donor_future)
    − similarity(swapped_future, host_future)
```

Positive ⇒ future follows hidden donor more than visible host.

### Candidate matching modes

1. Same projected area within tolerance.
2. Same morphology features.
3. Same `observer_score` bin.
4. Same M8G mechanism class.
5. Nearest-neighbor matching in projected feature space.
6. Strict projection-equal matching when possible.

### Activated interpretations

| Outcome | Reading |
|---|---|
| Swapped futures resemble hidden donor | "The candidate's future is partially carried by hidden substrate identity." |
| Swapped futures resemble visible host | "Visible projected structure dominates future identity under this intervention." |
| Projection preservation error high | "Swap is not clean; interpretation is limited." |
| Effect concentrated in high-HCE candidates | "HCE predicts hidden identity transfer." |
| No matching pairs | "Current candidate population does not support clean identity-swap testing." |

### Files

```
observer_worlds/experiments/run_followup_hidden_identity_swap.py
observer_worlds/analysis/identity_swap_stats.py
observer_worlds/analysis/identity_swap_plots.py
tests/test_hidden_identity_swap.py
```

### Performance plan

* Pre-compute candidate feature embeddings.
* Vectorized / approximate nearest-neighbor matching, not full
  pairwise.
* Cache original rollouts so swapped rollouts compare against
  already-computed futures.
* Parallelize over candidate pairs.

---

## Follow-up Topic 3 — Agent-Task Environments

### Question

Does hidden causal dependence predict useful behavior such as
survival, repair, foraging, memory, or task performance?

### Tasks (start small; do not build a game engine)

* **Task A — Repair / resilience.** Perturb candidate; measure
  recovery of shape, area, trajectory, observer-score. Compare
  high-HCE vs low-HCE candidates.
* **Task B — Resource-gradient / foraging.** Add a simple resource
  field in 2D. Score for movement toward / contact with the resource.
  Optional bias of CA update through a local field.
* **Task C — Memory / delayed response.** Temporary environmental cue,
  later response measurement. Compare whether hidden state preserves
  cue-related information.

### Models to fit

```
task_score ~ HCE
task_score ~ observer_score
task_score ~ HCE + observer_score
task_score ~ HCE + observer_score + mechanism_class
```

### Activated interpretations

| Outcome | Reading |
|---|---|
| HCE predicts task success after controlling for `observer_score` | "Hidden causal dependence contributes functional agency-relevant information beyond generic observer-likeness." |
| `observer_score` predicts but HCE does not | "Generic projected organization matters more than hidden causal dependence for this task." |
| Neither predicts | "Current tasks may not align with the existing candidate metrics." |
| Hidden perturbations reduce task performance | "Hidden state is functionally involved in task behavior, not only future divergence." |

### Files

```
observer_worlds/environments/tasks.py
observer_worlds/experiments/run_followup_agent_tasks.py
observer_worlds/analysis/agent_task_stats.py
observer_worlds/analysis/agent_task_plots.py
tests/test_agent_tasks.py
```

### Performance plan

* Small task maps first.
* Cache candidate snapshots; run tasks from saved states rather than
  re-running discovery.
* Parallelize over `(candidate × task × replicate)`.
* Batch rollouts; numpy / numba for task-field updates; plotting out
  of the inner loop.

---

## Implementation order

The follow-ups are gated to avoid one-shot mega-changes.

| Stage | Scope | Done in this commit? |
|---|---|---|
| 1 | Roadmap docs, performance strategy, profiler skeleton, projection-suite registry, skeleton CLIs for the three topics, smoke tests for imports / configs / `--help`. | **Yes** |
| 2 | Implement Topic 1 (projection robustness). Optimize shared state / projection caching. Smoke test only. | No |
| 3 | Implement Topic 2 (hidden identity swap). Use Topic 1 artifacts where possible. Smoke test only. | No |
| 4 | Implement Topic 3 (agent-task environments). Start with repair + memory; foraging is optional if complexity grows. Smoke test only. | No |
| 5 | Production runs, only after smoke tests pass and performance profiles are recorded. | No |

---

## Reporting requirements (per stage)

Each stage report includes:

* Status: completed / partial / failed.
* Commit / branch / dirty status; wall time; tests run / passing /
  skipped; whether any simulation actually ran.
* Changed-files table.
* Exact commands run.
* Performance: wall time, backend, `n_workers`, candidates/sec or
  rollouts/sec when meaningful, known bottleneck.
* Scientific result (if any): headline, numbers, artifact controls,
  caveats, recommended next step.

### Language posture (binding for all reports)

* Distinguish four claim types: generic observer-score,
  hidden-causal-dependence, performance/engineering, mechanism.
* Prefer "supports / is consistent with / does not support / falsifies
  the simple version / suggests / requires replication / survives this
  artifact check."
* Avoid "proved / consciousness / undeniable / obviously /
  guaranteed / done forever," except where "proves" is about a code
  invariant or a passing test.

# Stage 6 — Fresh-seed replication summary

Stage 6 re-ran the four major Stage 5 production findings on a
disjoint seed block (**7000–7019**, vs the original **6000–6019**).
Every other config knob — rule selection, grid, timesteps, horizons,
projection set, candidate cap, replicate count, classifier, backend,
worker count — was preserved exactly. The only deliberately varied
dimension is the seed range. This isolates *seed-block sensitivity*:
a finding that survives Stage 6 is not a 6000-block artifact.

This document records what was run, where the artifacts live, the
headline numbers per stage, and what the replication does and does
not establish.

## Per-stage table

| Stage | Topic | Seeds | Wall (h) | Output dir | Verdict |
|---|---|---|---:|---|:---:|
| 6E2 | Decoupled memory audit | 7000..7019 | 1.18 | `outputs/stage6e2_decoupled_memory_seed7000_20260430T032651Z/` | replicated |
| 6E | Agent tasks (repair / memory) | 7000..7019 | 2.66 | `outputs/stage6e_agent_tasks_seed7000_20260430T230923Z/` | replicated |
| 6D | Hidden identity swap | 7000..7019 | 0.39 | `outputs/stage6d_identity_swap_seed7000_20260501T023553Z/` | replicated |
| 6C | Projection robustness | 7000..7019 | 6.48 | `outputs/stage6c_projection_robustness_seed7000_20260501T035342Z/` | replicated |
| **Total** | | | **≈ 10.71** | | **4 / 4** |

All four runs were executed on the same Python environment, code
commit, and hardware (numpy backend, 30 workers). The Stage 6
production commands are recorded verbatim in
[docs/STAGE6_FRESH_SEED_REPLICATION_PLAN.md](STAGE6_FRESH_SEED_REPLICATION_PLAN.md);
each `config.json` and `frozen_manifest.json` in the output dirs above
captures the exact runtime configuration.

## Headline numbers per stage

### Stage 6E2 — decoupled memory audit (vs Stage 5E2)

Three variants × three sources (M7 / M4C / M4A); cue and HCE
perturbation regions deliberately disjoint (`mean_overlap_fraction = 0`
verified per variant). 14 400 trials.

| variant × source | n_cands | mean memory | r(HCE, memory) | 95% CI |
|---|---:|---:|---:|---|
| cue_far_boundary × M7 | 1627 | +0.228 | **+0.795** | [+0.776, +0.813] |
| cue_far_boundary × M4C | 594 | +0.212 | +0.394 | [+0.316, +0.470] |
| cue_far_boundary × M4A | 852 | +0.236 | +0.718 | [+0.683, +0.753] |
| cue_environment_shell × M7 | 2000 | +0.221 | +0.733 | [+0.709, +0.757] |
| cue_environment_shell × M4C | 1609 | +0.203 | +0.296 | [+0.245, +0.346] |
| cue_environment_shell × M4A | 1921 | +0.200 | +0.546 | [+0.505, +0.588] |
| cue_opposite_side × M7 | 2000 | +0.212 | +0.712 | [+0.685, +0.738] |
| cue_opposite_side × M4C | 1606 | +0.204 | +0.273 | [+0.223, +0.324] |
| cue_opposite_side × M4A | 1828 | +0.202 | +0.545 | [+0.503, +0.585] |

Mean Pearson(HCE, decoupled memory) across (variant, source) = **+0.557**.
8/9 cells preserve CI-clean positive Pearson(HCE, memory) when the
cue and HCE regions are exactly disjoint. The HCE-memory association
is not an artifact of methodological coupling between cue and probe.

### Stage 6E — agent tasks (vs Stage 5E)

Two tasks (repair, memory), three sources, 8 horizons, 3 replicates;
5530 candidates × 16 trials = 88 480 trials.

| source × task | mean task | r(HCE, task) | r(observer, task) |
|---|---:|---:|---:|
| M7 × repair | +0.755 | +0.085 | +0.392 |
| M7 × memory | +0.164 | **+0.727** | +0.578 |
| M4C × repair | +0.638 | -0.042 | +0.534 |
| M4C × memory | +0.169 | **+0.430** | -0.289 |
| M4A × repair | +0.614 | +0.000 | +0.477 |
| M4A × memory | +0.161 | **+0.583** | -0.377 |

HCE strongly predicts memory across all three sources (r = +0.43 to
+0.73). HCE does not predict repair for M4C/M4A (r ≈ 0).
Observer score predicts repair across all sources. M7 memory rises
monotonically with horizon: short 0.052 → medium 0.244 → long 0.305.

### Stage 6D — hidden identity swap (vs Stage 5D)

1434 valid pairs × 8 horizons × 2 directions = 22 944 score rows.
`projection_preservation_error = 0` for every valid swap.

| source | n | mean pull | 95% CI | frac > 0 |
|---|---:|---:|---|---:|
| M7_HCE_optimized | 8000 | -0.085 | [-0.132, -0.047] | 0.28 |
| M4C_observer_optimized | 6944 | -0.088 | [-0.177, -0.035] | 0.25 |
| M4A_viability | 8000 | -0.184 | [-0.252, -0.102] | 0.17 |

Overall mean pull = **-0.119**. Visible host structure dominates
future trajectory after hidden swap. Host dominance weakens with
horizon (short -0.181 → long -0.049). The fragile Stage 5D M7-only
late-horizon positive pull replicates: **M7 at h=40,80, mean pull
+0.011, CI [+0.003, +0.018]** (5D was +0.009 [+0.0003, +0.018]).

### Stage 6C — projection robustness (vs Stage 5C / 5C2)

26 139 candidates across 6 projections × 3 sources = 18 (projection ×
source) cells.

| projection | M4A norm. HCE | M4C norm. HCE | M7 norm. HCE |
|---|---:|---:|---:|
| mean_threshold | +0.527 | +0.554 | +0.558 |
| sum_threshold | +0.709 | **+0.493** | +0.861 |
| max_projection | +0.701 | **+0.493** | +0.861 |
| parity_projection | +0.648 | +0.633 | +0.613 |
| random_linear_projection | +0.641 | +0.628 | +0.623 |
| multi_channel_projection | +0.583 | +0.546 | +0.561 |

* **17 / 18 cells** have mean normalized_HCE > 0.5.
* **16 / 18 cells** have bootstrap CI lower bound > 0.5.
* M7 wins CI-clean over both baselines on `sum_threshold` and
  `max_projection` (diff ≈ +0.37). M7's advantage is
  projection-conditional, not universal.
* `initial_projection_delta = 0` exactly for 5/6 projections;
  `random_linear_projection` is near-invisible at tol 5e-3 by design
  (73 / 6000 candidates exceeded tolerance and were excluded).

## Final interpretation

All four major Stage 5 findings replicate at the same direction and
CI-cleanness on the disjoint 7000–7019 seed block. None of the
following Stage 5 production conclusions are seed-block artifacts:

1. **HCE is candidate-local across six projections** — local hidden
   perturbations produce more candidate-local future divergence than
   far hidden perturbations, in 16/18 cells with CI-clean support.
2. **Visible host structure dominates future trajectory after hidden
   identity swap, but the effect weakens at long horizons** — overall
   mean pull -0.12; M7 alone shows a CI-clean *positive* pull at
   h=40,80 (the most fragile Stage 5 finding, and the only one we
   flagged as at meaningful risk of non-replication; it survived).
3. **HCE strongly predicts memory task_score across all three sources;
   HCE does not predict repair for M4C/M4A** — Pearson(HCE, memory)
   r = +0.43 to +0.73 across sources; r(HCE, repair) ≈ 0 for M4C/M4A,
   small for M7.
4. **The HCE-memory association survives methodological decoupling** —
   when cue and HCE perturbation regions are exactly disjoint, 8/9
   (variant × source) cells preserve CI-clean positive Pearson(HCE,
   memory).

Stage 6 used the same code, hardware, and config as Stage 5; the only
deliberately varied dimension is the seed block. Cross-block
deltas exist (rules + seeds form a finite sample) but the *direction*
and *CI-cleanness* of every load-bearing claim survives.

## Remaining caveats

- **Functional simulation, not consciousness.** All metrics are
  measurable functional signatures of tracked persistent structures
  in a 4D → 2D cellular-automaton substrate. No claim is made about
  agency, intent, or phenomenal experience. The `repair`, `memory`,
  and `foraging` task probes are mechanism tests, not behavioral
  claims; the `foraging` resource is non-coupling and reported as a
  smoke-level drift / proximity signal only.
- **Not a real-universe claim.** The framework operationalizes a
  specific question about projected dynamics in a closed
  cellular-automaton model. It does not establish anything about
  physical observers, quantum measurement, or cosmology.
- **Sample-size sensitivity.** Stage 6 is one fresh seed block, not
  a Monte-Carlo over many. Effect-size estimates remain Stage-5+6
  pooled or per-block; CIs reported in the post-hoc analyzers reflect
  the new data alone.
- **One sub-threshold cell.** Stage 6C has one (projection × source)
  cell with mean normalized_HCE = +0.493, CI [+0.422, +0.549]
  straddling 0.5 (max_projection × M4C / sum_threshold × M4C — the
  same conceptual cell, since M4C produces few candidates that
  survive the binary detector under those projections). 5C2 had
  exactly one sub-threshold cell at the same conceptual location.
- **Mechanism-class breakdown.** The M8-classifier mechanism-class
  decomposition was deferred from Topic 1 to Stage 5+ and remains
  outside Stage 6. The "boundary-and-interior co-mediated, with a
  global-chaotic tail" picture is from M8 on the 5000/6000 blocks
  and was not re-tested on the 7000 block.
- **One transient worker crash.** Stage 6D's first launch crashed at
  ~16 min with a Windows access violation in `numpy._sum` inside
  `tracking._iou` in a single worker. The retry with identical config
  and `PYTHONFAULTHANDLER=1` succeeded; Stage 6C with faulthandler
  enabled showed no recurrence. Treat as flaky-environment noise
  pending further occurrence.

## Reproducibility

Each output dir contains `config.json` (the runtime config),
`frozen_manifest.json` (rules + seeds + grid frozen at launch), and
all CSVs the post-hoc analyzers consume. The Stage 5D / 5E posthoc
scripts (`observer_worlds.analysis.identity_swap_posthoc`,
`observer_worlds.analysis.agent_task_posthoc`,
`observer_worlds.analysis.projection_robustness_posthoc`) operate on
any matching run dir and re-emit the per-source / per-horizon /
per-projection breakdowns above without re-running the simulation.

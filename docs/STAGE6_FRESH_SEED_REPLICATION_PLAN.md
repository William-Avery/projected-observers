# Stage 6 — fresh-seed replication plan

All Stage 5 production results rest on **a single seed block** (6000–6019).
Stage 6 replicates the four major Stage-5 findings on a disjoint seed
block (**7000–7019**), preserving every other config knob, so we can
distinguish robust effects from seed-block artifacts.

This document records:
1. The Stage-5 findings to be replicated (with their original numbers).
2. The Stage-6 production commands (one per finding, seeds 7000–7019).
3. The Stage-6A preflight commands (tiny scale, seeds 7000–7002), used
   to verify the runners don't crash and the configs are valid before
   committing to the long sweeps.
4. The interpretation rules for the replication outcome.

## Findings being replicated

### Stage 5C — HCE projection robustness
Topic 1. 6.43 h on numpy / 30 workers. 26 105 candidates × 6 projections.
* **Headline:** 16/18 (projection × source) cells have normalized_HCE
  > 0.5 with CI lower bound > 0.5; the candidate-local HCE > far_HCE
  pattern is universal across the six tested projections.
* **M7-vs-baselines:** projection-conditional. M7 wins CI-clean on
  `sum_threshold` and `max_projection`; ties or marginal losses on the
  other four.

### Stage 5D — hidden identity swap
Topic 2. 33.5 min on numpy / 30 workers. 1500 pairs (500 / source).
* **Headline:** All three sources show CI-clean *negative*
  hidden_identity_pull (mean −0.08 to −0.15). Visible host structure
  dominates future trajectory.
* **M7 long-horizon detail:** M7 alone shows a CI-clean *positive*
  pull at h ∈ {40, 80} (+0.009, CI [+0.0003, +0.018]).

### Stage 5E — agent tasks
Topic 3. 2.44 h on numpy / 30 workers. 89 312 trials.
* **Headline:** Pearson(HCE, memory_score) = +0.748 (M7), +0.419
  (M4C), +0.619 (M4A). HCE strongly predicts memory across all three
  sources. HCE does *not* predict repair (r ≈ 0 for M4C/M4A).

### Stage 5E2 — decoupled memory audit
12.5 min on numpy / 30 workers. 14 400 trials, 3 variants × 3 sources.
* **Headline:** 8 of 9 (variant × source) cells preserve CI-clean
  positive Pearson(HCE, decoupled memory) when cue and HCE regions
  are exactly disjoint. M4A's correlation is essentially preserved
  under decoupling (~+0.60).

## Stage 6 production commands (seeds 7000–7019)

Each command preserves the Stage-5 production config except for the
seed block and the output label. **Run with `PYTHONIOENCODING=utf-8`**
to avoid the cp1252 print-crash hazard observed in Stages 5D/5E (the
runner banners contain Unicode em-dashes in some places; the env var
is the safety belt). Use the `--profile` flag to capture wall-time
breakdown.

### Stage 6C — projection robustness replication

```bash
PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_projection_robustness \
    --rules-from release/rules/m7_top_hce_rules.json \
    --m4c-rules release/rules/m4c_evolve_leaderboard.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --n-rules-per-source 5 --seeds 7000..7019 \
    --timesteps 500 --grid 64 64 8 8 \
    --max-candidates 20 --hce-replicates 3 \
    --horizons 1 2 3 5 10 20 40 80 \
    --projections mean_threshold sum_threshold max_projection \
                  parity_projection random_linear_projection \
                  multi_channel_projection \
    --backend numpy --n-workers 30 \
    --label stage6c_projection_robustness_seed7000 --profile
```

Expected wall: ≈ 6–7 hours (matches Stage 5C).

### Stage 6D — identity-swap replication

```bash
PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_hidden_identity_swap \
    --rules-from release/rules/m7_top_hce_rules.json \
    --m4c-rules release/rules/m4c_evolve_leaderboard.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --n-rules-per-source 5 --seeds 7000..7019 \
    --timesteps 500 --grid 64 64 8 8 \
    --max-candidates 20 --max-pairs 100 --hce-replicates 3 \
    --horizons 1 2 3 5 10 20 40 80 \
    --projection mean_threshold \
    --matching-mode morphology_nearest \
    --min-visible-similarity 0.3 \
    --backend numpy --n-workers 30 \
    --label stage6d_identity_swap_seed7000 --profile
```

Expected wall: ≈ 30–60 minutes (matches Stage 5D's 33 min).

### Stage 6E — agent-task replication

```bash
PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_agent_tasks \
    --rules-from release/rules/m7_top_hce_rules.json \
    --m4c-rules release/rules/m4c_evolve_leaderboard.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --n-rules-per-source 5 --seeds 7000..7019 \
    --timesteps 500 --grid 64 64 8 8 \
    --max-candidates 20 --tasks repair memory \
    --projection mean_threshold \
    --horizons 1 2 3 5 10 20 40 80 \
    --replicates 3 \
    --backend numpy --n-workers 30 \
    --label stage6e_agent_tasks_seed7000 --profile
```

Expected wall: ≈ 2.5–3 hours (matches Stage 5E's 2.44 h).

### Stage 6E2 — decoupled memory replication

Note: the Stage 5E2 medium-scale run was 2 rules × 10 seeds. Stage 6E2
uses the full 5 rules × 20 fresh seeds to harden the audit.

```bash
PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_agent_tasks \
    --rules-from release/rules/m7_top_hce_rules.json \
    --m4c-rules release/rules/m4c_evolve_leaderboard.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --n-rules-per-source 5 --seeds 7000..7019 \
    --timesteps 500 --grid 64 64 8 8 \
    --max-candidates 20 --tasks memory \
    --memory-variants cue_far_boundary cue_environment_shell cue_opposite_side \
    --projection mean_threshold \
    --horizons 1 2 3 5 10 20 40 80 \
    --replicates 3 \
    --backend numpy --n-workers 30 \
    --label stage6e2_decoupled_memory_seed7000 --profile
```

Expected wall: ≈ 1.5–2.5 hours (proportionally more than 5E2 medium
since 5 × 20 vs 2 × 10).

## Stage 6A preflight commands (seeds 7000–7002, ~1–2 min each)

Same configs as full runs, but reduced to 1 rule per source × 3 seeds
× T=150 × max_candidates=5 × reduced horizons. The point is to
confirm the runners do not crash and the documented artifacts are
written; numerical output of preflights is **not** replication
evidence.

```bash
# 6A-Topic1 preflight
PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_projection_robustness \
    --rules-from release/rules/m7_top_hce_rules.json \
    --m4c-rules release/rules/m4c_evolve_leaderboard.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --n-rules-per-source 1 --seeds 7000..7002 \
    --timesteps 150 --grid 64 64 8 8 \
    --max-candidates 5 --hce-replicates 1 \
    --horizons 5 10 20 \
    --projections mean_threshold sum_threshold max_projection \
                  parity_projection random_linear_projection \
                  multi_channel_projection \
    --backend numpy --n-workers 8 \
    --label preflight_stage6c

# 6A-Topic2 preflight
PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_hidden_identity_swap \
    --rules-from release/rules/m7_top_hce_rules.json \
    --m4c-rules release/rules/m4c_evolve_leaderboard.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --n-rules-per-source 1 --seeds 7000..7002 \
    --timesteps 150 --grid 64 64 8 8 \
    --max-candidates 5 --max-pairs 10 --hce-replicates 1 \
    --horizons 5 10 20 \
    --projection mean_threshold \
    --matching-mode morphology_nearest \
    --min-visible-similarity 0.3 \
    --backend numpy --n-workers 8 \
    --label preflight_stage6d

# 6A-Topic3 preflight (standard agent tasks)
PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_agent_tasks \
    --rules-from release/rules/m7_top_hce_rules.json \
    --m4c-rules release/rules/m4c_evolve_leaderboard.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --n-rules-per-source 1 --seeds 7000..7002 \
    --timesteps 150 --grid 64 64 8 8 \
    --max-candidates 5 --tasks repair memory \
    --projection mean_threshold \
    --horizons 5 10 20 \
    --replicates 1 \
    --backend numpy --n-workers 8 \
    --label preflight_stage6e

# 6A-Topic3 decoupled-memory preflight
PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_agent_tasks \
    --rules-from release/rules/m7_top_hce_rules.json \
    --m4c-rules release/rules/m4c_evolve_leaderboard.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --n-rules-per-source 1 --seeds 7000..7002 \
    --timesteps 150 --grid 64 64 8 8 \
    --max-candidates 5 --tasks memory \
    --memory-variants cue_far_boundary cue_environment_shell cue_opposite_side \
    --projection mean_threshold \
    --horizons 5 10 20 \
    --replicates 1 \
    --backend numpy --n-workers 8 \
    --label preflight_stage6e2
```

### Preflight success criteria

For each preflight, the run is OK iff:
* `returncode == 0`.
* All documented artifact files are written under the run dir.
* At least one cell produced > 0 candidates.
* No Unicode / cp1252 stdout crash (would manifest as 12-min exit-0 with no CSVs, as in the first Stage 5D attempt).
* Topic-1 projection-preservation invariant: `mean_initial_projection_delta = 0` for the count/order projections; ≤ 5e-3 for `random_linear`; 0 for `multi_channel`.
* Topic-2 swap invariant: `valid_swap_a / valid_swap_b > 0`, projection_preservation_error == 0 for valid pairs.
* Topic-3 standard: trial counts > 0; HCE and observer_score populated.
* Topic-3 decoupled: `mean_overlap_fraction == 0` per variant (decoupling holds).
* `pytest tests/ -q` still passes.

If any preflight fails, fix and re-preflight before launching the full
sweep.

## Recommended order for Stage 6 full replication

After preflight pass:

1. **Stage 6E2 (decoupled memory) — highest priority.** Cheapest of the
   four (≈ 1.5–2.5 h) and addresses the most novel finding (HCE-memory
   association beyond methodological coupling). If 5E2 doesn't survive
   on a fresh seed block, the headline conclusion changes immediately.
2. **Stage 6E (agent tasks) — second.** Replicates the underlying HCE-
   memory r = +0.42 to +0.75 numbers from 5E. If 6E2 holds, 6E should
   too; running 6E confirms cross-task structure (repair vs memory)
   replicates.
3. **Stage 6D (identity swap) — third.** Cheapest absolute (≈ 30–60
   min); replicates the visible-host-dominance result and the small
   M7 long-horizon positive pull. The M7 long pull (+0.009, CI just
   touches 0) is the most fragile finding, so seed-block replication
   is most informative here.
4. **Stage 6C (projection robustness) — last.** Most expensive (≈ 6 h);
   the underlying claim (HCE candidate-local across projections) was
   the most CI-clean and least likely to flip on a fresh seed block.

Total wall budget for full Stage 6: ≈ 11–14 hours. Each runs against
a fresh seed block disjoint from Stage 5; CIs reported in the
post-hoc analyzers will reflect the new data alone.

## Activated interpretation rules for replication

* **All four Stage-5 findings replicate at the same direction and CI-cleanness:**
  > "All four major Stage 5 findings replicate on the fresh seed block; the production results are not seed-block artifacts."
* **HCE-memory survives 6E2:**
  > "The decoupled HCE-memory association replicates; the Stage 5 conclusion that HCE predicts memory beyond methodological coupling stands."
* **HCE-memory does not survive 6E2:**
  > "The 5E2 decoupled HCE-memory association did not replicate; the Stage 5 association may be seed-specific or contingent on the original rule selection."
* **M7 long-horizon positive pull does not replicate:**
  > "The Stage 5D M7-only late-horizon positive pull was likely seed-specific; treat as noise unless multi-replication confirms."
* **Mixed:**
  > Report which findings replicated and which did not. Do not promote any non-replicating result to a production claim.

## Notes / caveats

* All four runs use the same Python environment, code commit, and
  hardware as Stage 5. The only deliberately varied dimension is
  the seed block.
* Stage 6 is **not a different experiment** — it is the same
  experiment with disjoint seeds. Cross-block deltas are expected
  (small CIs would not include all of the Stage-5 numbers because
  rules + seeds form a finite sample).
* Output dirs are timestamped; the labels above are stable
  prefixes. Post-hoc analyzers (`projection_robustness_posthoc`,
  `identity_swap_posthoc`, `agent_task_posthoc`) work on any
  matching run dir.
* Per the standing CLAUDE preference, full Stage 6 commits go
  directly to `origin/main`; no PR.

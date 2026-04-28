# M8E — cross-source mechanism comparison (specification)

> **Naming note.** The user originally called this "M8B." That name is
> already taken by `run_m8b_spatial_mechanism_disambiguation.py`
> (per-candidate region-aware perturbations on filtered-morphology
> candidates from a separate large-search). To avoid breaking the
> existing M8B/M8C/M8D experiments, plots, stats modules, and tests,
> this specification uses **M8E**. Renaming the existing M8B is a
> separate decision and is not in scope here.

## Status

**Specified, not yet run.** Wall time at production scale is expected to
be roughly 3× the M7-only `m8_m7b_class_numpy` baseline (≈90 minutes on
the captured hardware on the numpy backend, considerably less on
cuda-batched). It must be invoked manually; it is not part of normal
pytest.

## Research question

Is M7's `boundary_mediated` fraction (≈0.79 in
`outputs/m8_20260427T214903Z`) higher than M4C and M4A's, or is
boundary-organized response geometry common to all viable projected
4D candidates?

The same question applies to the `global_chaotic` fraction, the
`boundary_mediation_index` distribution, and the within-class HCE
distributions.

## Hypothesis

Support:
* M7 has a measurably higher `boundary_mediated` fraction than both
  M4C and M4A under the same classifier and same config; or
* M7 has equal `boundary_mediated` fraction but higher HCE *within*
  the `boundary_mediated` class.

Failure / partial:
* All three sources have similar `boundary_mediated` fractions (within
  bootstrap CI overlap). Under that outcome, the mechanism shift claim
  is **not** supported and the M7 advantage is in *strength*, not
  *kind*, of mechanism.
* M7's gain is concentrated in the `global_chaotic` class. Under that
  outcome, part of M7's HCE advantage may reflect increased instability
  rather than structured boundary-organized mediation.

## Design

| Field | Value |
|---|---|
| Sources | `M7_HCE_optimized`, `M4C_observer_optimized`, `M4A_viability` |
| Rule files | `release/rules/m7_top_hce_rules.json`, `release/rules/m4c_evolve_leaderboard.json`, `release/rules/m4a_search_leaderboard.json` |
| `n_rules_per_source` | 5 (matches the M7-only baseline) |
| Test seeds | 6000..6019 (20 seeds, identical block) |
| Grid | 64 × 64 × 8 × 8 |
| Timesteps | 500 |
| Max candidates per cell | 20 |
| HCE replicates | 3 |
| Horizons | 1, 2, 3, 5, 10, 20, 40, 80 |
| Backend | `numpy` (matches M7-only baseline; cuda-batched is a separate variant) |
| Workers | 30 |
| Sweep cells | 5 × 20 × 3 = 300 |
| Candidate selection mode | identical across sources (default M8 selection) |
| Train/validation/test split | sources are evaluated independently on the same test seed block; no per-source seed split needed for this comparison (the comparison is in-distribution) |
| Classifier | unchanged — see `docs/M8_classifier_audit.md` for current rule order |

## Command

The existing M8 driver already supports `--m4c-rules` and `--m4a-rules`,
so no new code is required. Invocation:

```bash
python -m observer_worlds.experiments.run_m8_mechanism_discovery \
    --m7-rules  release/rules/m7_top_hce_rules.json \
    --m4c-rules release/rules/m4c_evolve_leaderboard.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --n-rules-per-source 5 \
    --test-seeds 6000 6001 6002 6003 6004 6005 6006 6007 6008 6009 \
                 6010 6011 6012 6013 6014 6015 6016 6017 6018 6019 \
    --timesteps 500 \
    --grid 64 64 8 8 \
    --max-candidates 20 \
    --hce-replicates 3 \
    --horizons 1 2 3 5 10 20 40 80 \
    --backend numpy \
    --label m8e_cross_source \
    --n-workers 30
```

If cuda-batched is also captured, repeat with `--backend cuda-batched`
and `--label m8e_cross_source_cuda`.

## Primary outputs

By rule source:

* mechanism distribution (counts, fractions, 95% CIs)
* `boundary_mediated` fraction
* `global_chaotic` fraction
* `threshold_mediated` fraction
* mean `boundary_response_fraction`, `interior_response_fraction`,
  `boundary_mediation_index`
* HCE within each mechanism class
* HCE/lifetime tradeoff curve

`comparison_grid` in `stats_summary.json` will contain the cross-source
deltas; this field is currently empty in `m8_20260427T214903Z` because
that run had only one source.

## Primary comparisons

Each row reports M7 vs. baseline with grouped-bootstrap CI (group =
`(rule_id, seed)`) and permutation p-value.

| Comparison | Statistic |
|---|---|
| M7 vs M4C | difference in `boundary_mediated` fraction |
| M7 vs M4A | difference in `boundary_mediated` fraction |
| M7 vs M4C | difference in `global_chaotic` fraction |
| M7 vs M4C | difference in mean `boundary_mediation_index` |
| M7 vs M4C | difference in mean HCE within `boundary_mediated` class |
| M7 vs M4C | difference in mean HCE within `global_chaotic` class |
| M7 vs M4A | same set, against M4A |

## Success criteria

A primary result is meaningful if:

* The grouped-bootstrap 95% CI on the M7-vs-baseline difference excludes
  zero, AND
* The within-class HCE comparison is reported alongside the cross-class
  fraction comparison (so a "shift" claim is not confounded with a
  "strength within class" claim), AND
* The audit table from `docs/M8_classifier_audit.md` is regenerated for
  each source so the per-class sanity check is repeatable.

## Interpretation rules (commit to these before running)

| Outcome | Interpretation |
|---|---|
| M7 `boundary_mediated` fraction > M4C and M4A by CI-clean margin | "M7 evolution shifted candidates toward boundary-organized hidden causal mechanisms under the current classifier." |
| All three sources have CI-overlapping `boundary_mediated` fractions | "Boundary-organized response geometry appears to be a general feature of viable projected 4D candidates, not specific to M7. M7's advantage must be in strength rather than kind." |
| M7 has CI-clean higher `global_chaotic` fraction | "Part of M7's HCE gain may come from increased global instability, not only from structured boundary-organized mediation. Re-check candidate-locality controls before claiming structured mechanism." |
| M7 has higher HCE within the `boundary_mediated` class | "M7 strengthens the boundary-organized mechanism rather than merely increasing its frequency." |
| M7 has higher HCE within the `global_chaotic` class but not within `boundary_mediated` | "M7's HCE advantage is concentrated in unstable candidates; the structured mechanism story is not supported." |

## Invariants and confound checks

Before accepting any result, the M8E run must satisfy:

* `returncode == 0`
* `stats_summary.json` has non-empty `comparison_grid`
* `n_candidates ≥ ` (M7's per-source candidate count); plausibly ~3×
  the M7-only run total, but no per-source count should be near zero
* Each source has at least one candidate in `boundary_mediated`
* Per-source `near_threshold_fraction` recorded for the threshold audit
* Per-class sanity table (the table in `M8_classifier_audit.md`)
  regenerated per source

Performance/perf-gate is **not** a concern here; this run is opt-in only.

## Caveats fixed in advance

* Any difference in `boundary_mediated` fraction is conditional on the
  current classifier rule order. The audit doc shows this rule is a
  geometry test, not a mediation contrast. A future classifier revision
  (e.g. adding a `boundary_and_interior_co_mediated` class) will produce
  a different distribution from the same data.
* M4A is unoptimized for observer-likeness; M4C is observer-optimized
  but not HCE-optimized; M7 is HCE-optimized. The three-way comparison
  is therefore "are mechanism distributions different across optimizers
  with different fitness pressure," not "are these matched baselines."
  The M4D fairness caveat (optimized vs unoptimized) does not apply
  here because M8E reports mechanism distribution, not observer-score.
* Within-class HCE comparisons are subject to the same per-candidate
  noise as the M7B comparison; grouped bootstrap by `(rule_id, seed)`
  is required.

## Recommended next step after M8E

* If `boundary_mediated` fractions overlap across sources: revisit the
  classifier (add the co-mediated class) and re-label all three sources
  with the extended classifier before drawing any cross-source
  conclusion.
* If M7's gain concentrates in `global_chaotic`: tighten M7 fitness to
  penalize global chaos and re-evolve.
* If M7 is CI-clean higher in `boundary_mediated` and within-class HCE:
  proceed to mechanism *strength* studies (M8C / M8D / spatial M8B) on
  the M7 boundary-mediated subset.

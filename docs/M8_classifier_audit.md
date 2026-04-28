# M8 mechanism-classifier audit

This document audits the rule-based mechanism classifier used by M8 mechanism
discovery (`observer_worlds.experiments._m8_mechanism.classify_mechanism`).
It was prompted by the apparent contradiction in the
`outputs/m8_20260427T214903Z` run between

* the per-source aggregate (`boundary_response_fraction = 0.94`,
  `interior_response_fraction = 0.95`, `boundary_mediation_index = 0.498`),
  which says boundary and interior are both highly responsive and equally
  mediating, and
* the per-class distribution (~79% labeled `boundary_mediated`), which the
  prior summary read as evidence of boundary-localization.

The audit shows the two are consistent because the classifier is
**priority-ordered** and the `boundary_mediated` rule is a response-map
geometry check, not a boundary-vs-interior contrast.

## Classifier rules (exact, in firing order)

Source: `observer_worlds/experiments/_m8_mechanism.py`,
`classify_mechanism()`.

The rules are checked in this fixed order; the first matching rule assigns
the label. Once a rule fires, later rules are not evaluated.

| Order | Label | Condition |
|---:|---|---|
| 1 | `threshold_mediated` | `near_threshold_fraction > 0.5` AND `interior_hidden_effect ≤ 0.5 × boundary_hidden_effect + 1e-6` |
| 2 | `global_chaotic` | `far_hidden_effect > 0.7 × (interior_hidden_effect + boundary_hidden_effect)` |
| 3 | `boundary_mediated` | `boundary_response_fraction > 0.6` |
| 4 | `interior_reservoir` | `interior_response_fraction > 0.6` |
| 5 | `environment_coupled` | `environment_response_fraction > 0.4` |
| 6 | `delayed_hidden_channel` | `first_visible_effect_time ≥ 5` AND `fraction_hidden_at_end > 0.5 × fraction_visible_at_end` |
| 7 | `unclear` | (fallthrough) |

Two structural observations follow:

1. Rule 3 (`boundary_mediated`) is a **response-map geometry** test
   (does the response field show activity in boundary cells), not a
   **mediation contrast** (does boundary mediate more than interior).
   Whenever both `boundary_response_fraction > 0.6` and
   `interior_response_fraction > 0.6`, rule 3 fires before rule 4 is
   evaluated, even if `boundary_mediation_index ≈ 0.5` (boundary and
   interior contribute equally).
2. Rule 4 (`interior_reservoir`) is reachable only when
   `boundary_response_fraction ≤ 0.6`. The 0.14% interior-reservoir rate
   in the M7 run is therefore a *floor* on interior-reservoir-like
   structure, not a measurement of how often interiors mediate.

## Per-class sanity table (run `m8_20260427T214903Z`, M7_HCE_optimized, N=1387)

Means within each predicted class. Pulled from
`mechanism_labels.csv`, joined with `lifetime_tradeoff.csv` for HCE.
Per-candidate `local_div` is not stored in the CSVs; only the per-source
aggregate (0.290) is available without re-running.

| label | n | boundary_resp | interior_resp | BMI | CLI | far_hid | HCE | near_thr | first_vis_t | h2v_t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boundary_mediated | 1093 | 0.993 | 1.000 | 0.500 | 0.022 | 0.020 | 0.021 | 0.480 | 1.77 | 1.26 |
| interior_reservoir | 2 | 0.517 | 1.000 | 0.541 | 0.012 | 0.011 | 0.012 | 0.310 | 2.00 | 1.50 |
| environment_coupled | 0 | — | — | — | — | — | — | — | — | — |
| global_chaotic | 257 | 0.848 | 0.864 | 0.486 | 0.001 | 0.019 | 0.010 | 0.239 | 2.47 | 1.45 |
| threshold_mediated | 0 | — | — | — | — | — | — | — | — | — |
| delayed_hidden_channel | 7 | 0.000 | 0.000 | 0.493 | 0.007 | 0.010 | 0.008 | 0.000 | 16.43 | 2.71 |
| unclear | 28 | 0.000 | 0.000 | 0.500 | 0.012 | 0.009 | 0.010 | 0.261 | 1.75 | 1.36 |

Key reads:

* In the **`boundary_mediated`** class, `boundary_mediation_index = 0.500`
  almost exactly. The label is firing on `boundary_response_fraction = 0.99`
  while interior response is also 1.00 and the actual boundary/interior
  hidden-effect split is ~50/50. This is "boundary-organized" geometry,
  not boundary-localized mediation.
* The `near_threshold_fraction` for the boundary-labeled class is 0.480 —
  just under the 0.5 cutoff for rule 1. ~48% of these candidates have
  most of their hidden activity in near-threshold cells; rule 1 narrowly
  misses many of them because of its second clause (the interior must
  be ≤ half of boundary). With both close to 1.0, rule 1 cannot fire.
* The **`global_chaotic`** class still has high boundary and interior
  response (~0.85). What separates it from `boundary_mediated` is the
  `far_hidden_effect`-to-candidate-effect ratio (rule 2 fires before
  rule 3). In other words, ~19% of "boundary-active" candidates are
  reclassified as global because hidden perturbations far from the
  candidate produce comparable effects.
* **`delayed_hidden_channel`** (n=7) has zero boundary and interior
  response; it is reached only via the timing rule (rule 6).
* **`unclear`** (n=28) also has zero boundary/interior response and
  intermediate near-threshold; everything misses every threshold.
* **`threshold_mediated`** never fires in this run because rule 1's
  second clause requires interior ≪ boundary, and in M7 candidates the
  two are roughly equal.

## Revised interpretation language

The audit does **not** support the strong wording that HCE is
"localized at boundaries, not in interiors." It supports a weaker,
classifier-conditional wording.

**Use:**

> Under the current mechanism classifier, M7 candidates are predominantly
> labeled `boundary_mediated`. However, aggregate response fractions show
> both boundary and interior regions are highly responsive
> (means 0.94 and 0.95), and the boundary-mediation index averages
> 0.498 — i.e. boundary and interior contribute roughly equally to the
> mediated hidden effect. The label should therefore be read as
> "boundary-organized response geometry under the classifier's priority
> ordering," not as "boundary-localized mediation." The 18.5%
> `global_chaotic` rate further indicates that not all M7 candidates pass
> the local-vs-far locality contrast.

**Do not use:**

* "HCE is predominantly localized at projected-candidate boundaries,
  not in interiors."
* "Interior responses are negligible."
* "M7 confines hidden causal effect to the boundary."

## What the audit does and does not change

This audit changes the **interpretation** of the M8 run. It does not
change any number in the run. The previously-recorded findings remain:

* HCE in M7 candidates is **not classified as threshold-mediated**
  (0/1387 under rule 1). This survives the audit because rule 1's
  failure mode here is interior ≈ boundary, not threshold-margin
  proximity, and that is itself informative.
* HCE in M7 candidates is **not classified as environment-coupled**
  (0/1387 under rule 5).
* The classifier does not currently produce a distinct
  "boundary-and-interior co-mediated" class. That is a gap in the
  taxonomy, not in the data.

## What this audit recommends

1. Treat any single-source M8 run as a *labeling distribution under the
   current classifier*, not as a direct measurement of where HCE lives.
2. Run a cross-source mechanism comparison (see
   `docs/M8E_cross_source_mechanism_spec.md`) before making any claim
   that M7 *shifts* mechanism distribution relative to M4C/M4A. The
   per-class sanity table above only quantifies what M7 looks like in
   isolation.
3. Consider an extension to the classifier (separate ticket) that adds
   an explicit `boundary_and_interior_co_mediated` class for cases where
   `boundary_response_fraction > 0.6` AND
   `interior_response_fraction > 0.6` AND
   `|boundary_mediation_index − 0.5| < 0.1`. Under that classifier the
   1093 currently-`boundary_mediated` candidates would split, and the
   resulting distribution would more accurately reflect the
   geometry-vs-mediation distinction.

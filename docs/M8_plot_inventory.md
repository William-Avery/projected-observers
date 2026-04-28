# M8E / M8F / M8G plot inventory

A curated guide to the figures generated for the M8E cross-source run and
its post-hoc analyses (M8F within-class bootstrap, M8G revised
classifier). Use this when reading the run summary or writing reports
about it.

The plots themselves are local-only — `outputs/` is gitignored. To
regenerate, re-run M8E (long) or the analysis modules
(`observer_worlds.analysis.m8f_within_class`,
`observer_worlds.analysis.m8g_revised_mechanisms`) on the existing
M8E artifacts.

**Reference run dir:** `outputs/m8e_cross_source_20260428T150640Z/`
(commit `9486c17`, captured 2026-04-28; 3 888 candidates, M7 / M4C / M4A,
20 seeds, numpy backend.)

Read every "boundary"-related plot as **classifier-conditional**: see
`docs/M8_classifier_audit.md` for why the old `boundary_mediated` label
is response-map geometry, not boundary-localized mediation.

---

## Primary evidence — within-class HCE comparisons

These are the figures that should be referenced first when describing
the M7-vs-baseline mechanism story.

### `plots/m8f_within_class_hce_ci.png`
- **Shows:** mean per-candidate HCE within `boundary_mediated` (left)
  and `global_chaotic` (right) for each source (M7, M4C, M4A), with
  95 % grouped-bootstrap CI bars (groups = `(rule_id, seed)`).
- **Supports:** "M7 has CI-clean higher HCE than both baselines within
  `global_chaotic`; CI-clean against M4C but visually overlapping
  against M4A within `boundary_mediated`."
- **Status:** **primary evidence.**
- **Caveat:** the boundary-mediated panel makes the borderline M7 vs
  M4A case visible — the upper M4A error bar reaches into the lower
  M7 error bar. The figure is honest about the uncertainty and should
  not be cropped.

### `plots/m8f_within_class_effect_sizes.png`
- **Shows:** four bars of M7-minus-baseline mean HCE differences with
  95 % grouped-bootstrap CI; one bar each for `(M7 vs M4C, M7 vs M4A) ×
  (boundary_mediated, global_chaotic)`.
- **Supports:** "3 of 4 primary HCE comparisons exclude zero in CI; the
  M7 vs M4A boundary_mediated bar's lower whisker just touches zero."
- **Status:** **primary evidence.** Most direct visual of the M8F
  headline.
- **Caveat:** none — the borderline result is shown as-is and not
  hidden. The y-axis crosses zero, making "CI excludes zero" visually
  unambiguous.

### `plots/m8g_hce_by_revised_mechanism.png`
- **Shows:** mean per-candidate HCE for each revised M8G mechanism
  class, grouped bars per source.
- **Supports:** "M7 has the highest mean HCE in the dominant
  `boundary_and_interior_co_mediated` class and in `global_chaotic`;
  M7's bars in `boundary_only_or_boundary_dominant`,
  `interior_only_or_interior_dominant`, and `threshold_mediated` are
  zero — M7 produces no candidates in those classes at all."
- **Status:** **primary evidence** for both M8G's mechanism re-labelling
  and M7's within-class strength advantage.
- **Caveat:** no error bars on this plot (point estimates only). For
  CIs use `m8f_within_class_hce_ci.png` for the equivalent classes
  (M8F's `boundary_mediated` ≈ M8G's `boundary_and_interior_co_mediated`
  to ≈99 %).

---

## Primary evidence — cross-source mechanism distribution

### `plots/mechanism_class_distribution.png` *(M8 standard, written by `m8_plots`)*
- **Shows:** old-classifier mechanism class fractions per source, with
  three side-by-side bars for M7 / M4C / M4A.
- **Supports:** "Boundary-organized response geometry is general across
  all three rule sources, not specific to M7. ~76 – 79 %
  `boundary_mediated` everywhere; ~16 – 19 % `global_chaotic`
  everywhere."
- **Status:** **primary evidence** for the M8E headline that M7 does
  *not* shift mechanism distribution relative to baselines.
- **Caveat:** the old classifier compresses co-mediated and
  boundary-only into a single class. See `m8g_revised_mechanism_distribution.png`
  for the revised view.

### `plots/m8g_revised_mechanism_distribution.png`
- **Shows:** stacked bars per source, "old (M8 classifier)" vs
  "revised (M8G)", primary BMI band 0.35–0.65.
- **Supports:** "The relabelling moves ~99 % of old `boundary_mediated`
  to new `boundary_and_interior_co_mediated`; the high-level shape of
  the distribution is preserved."
- **Status:** **diagnostic.**
- **Caveat:** *the relabelling story is in the data, not the visual.*
  The two bars per source look nearly identical because the dominant
  ~78 % chunk is just relabelled in place. The plot is honest but not
  illustrative of the relabelling. To see the actual relabelling, refer
  to the transition matrix in `m8g_revised_mechanism_summary.md` or in
  the report text.

### `plots/boundary_vs_interior_response_by_source.png` *(M8 standard)*
- **Shows:** mean `boundary_response_fraction` and
  `interior_response_fraction` per source as paired bars.
- **Supports:** the M8 classifier audit's central observation:
  boundary and interior response are essentially equal in every source
  (M7: 0.94 / 0.95; M4C: 0.94 / 0.94; M4A: 0.90 / 0.91). The
  "boundary-organized" reading is required.
- **Status:** **primary evidence** for the audit.
- **Caveat:** none. The pattern is universal across sources, which is
  itself the finding.

---

## Diagnostic — mechanism-classifier validity

### `plots/local_vs_far_effect_by_mechanism.png` *(M8 standard)*
- **Shows:** mean `candidate_locality_index` vs mean `far_hidden_effect`
  per old-classifier mechanism class, aggregated across all sources.
- **Supports:** "The classifier's `global_chaotic` rule is doing real
  work — in that class, far_hidden_effect dominates locality
  (~0.011 vs ~0.001), the opposite of `boundary_mediated`
  (locality 0.019 > far 0.017) and `interior_reservoir`
  (locality 0.012 > far 0.011)."
- **Status:** **diagnostic** for the classifier; secondary for any
  M7-vs-baseline claim.
- **Caveat:** aggregated across sources, not per source. Cannot be used
  to claim "M7's boundary_mediated has stronger locality than M4C's" —
  that comparison is in the M8F effect-size plot and the per-source
  per-class table.

---

## Descriptive — HCE / lifetime tradeoff

### `plots/hce_vs_lifetime_tradeoff.png` *(M8 standard)*
- **Shows:** scatter of per-candidate HCE vs `candidate_lifetime`,
  colored by source.
- **Supports:** "M7 (blue) is concentrated at low lifetime (≤ 100) with
  a wide HCE spread; M4A (green) populates the long-lifetime tail
  (200 – 500) with HCE near zero; the visual shape is consistent with
  the cross-source HCE-vs-lifetime sign-flip
  (M7: r = +0.30; M4C: r = −0.88; M4A: r = −0.72)."
- **Status:** **descriptive** — supports the qualitative cross-source
  observation, not a hypothesis-tested claim.
- **Caveat:** **M4C (orange) is largely visually obscured.** M4C
  candidates cluster tightly around lifetime ≈ 50 and HCE ≈ 0.01, so
  they are stacked under M7 dots in that region. The M4C strong
  negative correlation (r = −0.88) is not obvious from the scatter
  alone; it is in the data because M4C has both a few outliers and a
  tight short-life / high-HCE corner that flips sign relative to its
  long-life / low-HCE bulk. Do not conclude from this scatter alone
  that M4C lacks a correlation.

### `plots/hidden_volatility_vs_lifetime.png` *(M8 standard)*
- **Shows:** scatter of `boundary_response_fraction` (used as a hidden-
  volatility proxy) vs `candidate_lifetime`, by source.
- **Supports:** boundary response is bimodal (≈ 0 or ≈ 1) at every
  lifetime; not a clean predictor of lifetime.
- **Status:** **descriptive / diagnostic.** Useful for showing that
  the response-map geometry is bimodal (you are in or out of the
  response field, with little intermediate).
- **Caveat:** **M4C (orange) is again largely invisible in this scatter
  for the same overplot reason as `hce_vs_lifetime_tradeoff.png`.** Do
  not interpret the absence of orange as the absence of M4C
  candidates — they are present in the data but visually obscured.

### `plots/hce_vs_boundary_response_fraction.png` *(M8 standard)*
- **Shows:** scatter of per-candidate HCE vs
  `boundary_response_fraction`, by source.
- **Supports:** HCE is concentrated at `boundary_response_fraction = 1`
  with a long upper tail; high HCE generally requires high boundary
  response.
- **Status:** **descriptive.**
- **Caveat:** The visible high-HCE outliers (HCE > 0.10) are mostly
  M4A (green), not M7 (blue). M7 has the highest *mean* HCE but does
  not contain the tallest individual peaks. This is consistent with
  M7's narrower per-candidate HCE distribution and with the M8F
  result (M7 wins in mean and in ranks, not in extreme tails). M4C
  is again largely visually obscured.

---

## Per-class / per-source side panels

These supplement the headline plots; they are useful for sanity-checks
but rarely the primary citation.

### `plots/first_visible_effect_time_by_source.png`
- **Shows:** boxplot of `first_visible_effect_time` (the first horizon
  at which `local_div > epsilon`) per source.
- **Supports:** most candidates show visible effect at the first
  horizon (median ≈ 1 in all sources); the outliers ≥ 5 are the
  `delayed_hidden_channel` population.
- **Status:** **diagnostic.**

### `plots/hidden_to_visible_conversion_time_by_source.png`
- **Shows:** boxplot of `hidden_to_visible_conversion_time` per source.
- **Supports:** the typical lag from hidden activity to visible response
  is ~1 step in all sources; not a discriminating feature.
- **Status:** **diagnostic / secondary.**

### `plots/feature_lead_lag_heatmap.png`
- **Shows:** lead/lag correlations between hidden-state and visible-mass
  features over rollout horizons.
- **Supports:** specific lead/lag patterns inside the pathway trace.
- **Status:** **secondary / diagnostic** — useful for mechanism studies
  but not load-bearing for the cross-source comparison.

---

## Qualitative — single-candidate visualizations

### `plots/example_candidate_pathway_trace.png`
- **Shows:** one example candidate's hidden vs visible mass trajectory
  over the rollout.
- **Status:** **illustrative.** Not load-bearing for any cross-source
  claim.

### `plots/hidden_mass_vs_visible_mass_over_time.png`
- **Shows:** hidden vs visible XOR mass over the 80-step rollout for
  the top-6 HCE candidates (across sources).
- **Supports:** hidden mass dwarfs visible mass throughout the rollout
  (~120 k vs ~5 k by step 30). Confirms the hidden state is
  quantitatively much larger than the visible projection.
- **Status:** **illustrative.** Useful for explaining "what HCE is
  measuring" to a reader unfamiliar with the project, not for any
  comparison claim.
- **Caveat:** these are top-6 HCE candidates pooled across sources,
  not source-stratified.

### `plots/response_map_examples_top_hce.png`
- **Shows:** spatial response maps for the top-HCE candidates.
- **Status:** **illustrative.**

---

## Quick reference — which plot for which claim

| Claim in a report | Cite this plot |
|---|---|
| "M7 has CI-clean higher HCE than M4C within boundary_mediated / co-mediated." | `m8f_within_class_hce_ci.png` and/or `m8f_within_class_effect_sizes.png` |
| "M7 vs M4A within boundary_mediated is borderline (CI just touches zero)." | `m8f_within_class_effect_sizes.png` (the CI bar is visible) |
| "M7 has the largest within-class HCE advantage in `global_chaotic`." | `m8f_within_class_hce_ci.png` and `m8g_hce_by_revised_mechanism.png` |
| "Boundary-organized response geometry is universal across rule sources." | `boundary_vs_interior_response_by_source.png` and `mechanism_class_distribution.png` |
| "The old `boundary_mediated` class compresses co-mediated and boundary-only." | Reference the M8G transition tables in the summary text — the distribution plot is too compressed visually. |
| "The classifier's `global_chaotic` rule discriminates locality vs far-effect correctly." | `local_vs_far_effect_by_mechanism.png` (diagnostic) |
| "HCE / lifetime correlation is sign-flipped between M7 and baselines." | Cite the table from `m8f_within_class_summary.md`; `hce_vs_lifetime_tradeoff.png` is illustrative but obscures M4C. |

---

## Plots that complicate or do not fit the headline narrative

Honest list, not hidden:

- `m8g_revised_mechanism_distribution.png` does not visually demonstrate
  the relabelling. The plot is correct; the relabelling is invisible at
  the high level because the dominant chunk just changes color.
  **Do not** use it as the primary visual evidence for "old
  `boundary_mediated` becomes co-mediated"; use the transition table.
- `hce_vs_boundary_response_fraction.png` shows the highest individual
  HCE peaks belong to **M4A**, not M7. M7's mean HCE is higher; M7's
  per-candidate maximum is lower. Acknowledge this if a reader asks
  "where are the high-HCE outliers?"
- `hce_vs_lifetime_tradeoff.png` and `hidden_volatility_vs_lifetime.png`
  visually obscure M4C (orange) under M7 (blue) and M4A (green). The
  M4C population is in the data but not in the picture. The strong
  negative HCE-vs-lifetime correlation for M4C (r = −0.88) is not
  visually obvious from the scatter; rely on the table.
- `local_vs_far_effect_by_mechanism.png` is aggregated across sources;
  it cannot answer "is M7's boundary_mediated more candidate-local
  than M4C's?" Use the M8F effect-size table for that.

If a reader points to one of these plots and disputes a claim that
relies on a different artifact, the right move is to cite the table
or the relevant primary plot, not to re-fit the narrative to the
visual.

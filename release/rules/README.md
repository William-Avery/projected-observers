# Pinned rule artifacts

These are the three rule files referenced by every M5+ experiment in
this repo. They are committed here (instead of left under `outputs/`,
which is gitignored) so a fresh clone has everything needed to run
M5–M8D end to end without re-evolving rules from scratch.

| File | Source | Format | Use |
|---|---|---|---|
| `m7_top_hce_rules.json` | `outputs/m7_real_20260426T104724Z/top_hce_rules.json` | List of `FractionalRule.to_dict()` plus `_metadata` | M7 / M7B / M8 / M8B / M8C / M8D `--m7-rules` |
| `m4c_evolve_leaderboard.json` | `outputs/observer_search/m4c_evolve/leaderboard.json` | M4C evolutionary leaderboard (entries with `rule` + `fitness`) | All experiments `--m4c-rules` |
| `m4a_search_leaderboard.json` | `outputs/rule_search/m4a_search/leaderboard.json` | M4A viability random-search leaderboard (entries with `rule` + `viability_score`) | All experiments `--m4a-rules` |

## Usage

Substitute these paths into any CLI example in the top-level README. E.g.
the M8D moderate run becomes:

```bash
python -m observer_worlds.experiments.run_m8d_global_chaos_decomposition \
    --m7-rules release/rules/m7_top_hce_rules.json \
    --m4a-rules release/rules/m4a_search_leaderboard.json \
    --backend numba \
    --grid 96 96 8 8 --timesteps 600 --max-candidates-per-run 50 \
    --hce-replicates 3 --horizons 1 2 3 5 10 20 40 80
```

## Why these specific rules?

- **`m7_top_hce_rules.json`** is the output of the M7 HCE-guided evolution
  in `outputs/m7_real_20260426T104724Z/`. Top-5 rules by composite
  fitness (observer + HCE + locality − fragility − threshold). M7B
  validated these at production scale; M8 / M8B / M8C used them as the
  primary M7 source.
- **`m4c_evolve_leaderboard.json`** is the M4C evolutionary observer
  search (`(μ+λ)` over fractional rules). Top entries are the
  observer-fitness baseline.
- **`m4a_search_leaderboard.json`** is the M4A random-search viability
  leaderboard. Used as the "observer-agnostic but viable" baseline.

## Reproducibility

The frozen manifests in any M7B / M8 / M8B / M8C / M8D output directory
record the SHA-256 of the rule file passed via `--m7-rules`. If you
re-run with `release/rules/m7_top_hce_rules.json`, the manifest's
`input_rule_files["m7_rules"].sha256` should match the file's hash —
that's the bit-for-bit reproducibility guarantee.

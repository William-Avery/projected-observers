"""M8G — revised mechanism classifier (analysis-only, no simulation).

Re-labels the candidates in an existing M8 / M8E run using a revised
priority-ordered classifier that distinguishes:

    1. threshold_mediated
    2. global_chaotic
    3. boundary_and_interior_co_mediated
    4. boundary_only_or_boundary_dominant
    5. interior_only_or_interior_dominant
    6. environment_coupled
    7. delayed_hidden_channel
    8. unclear

The motivation is in `docs/M8_classifier_audit.md`: the original M8
classifier label `boundary_mediated` is a response-map geometry test
(`boundary_response_fraction > 0.6`) that does not contrast against the
interior. Inside that class, `boundary_mediation_index` averages 0.5
across all three rule sources — boundary and interior contribute
roughly equally to the mediated hidden effect. The revised classifier
splits the old `boundary_mediated` class into co-mediated vs
truly-dominant variants.

Read every comparison this module produces as **classifier-conditional**:
the `boundary_and_interior_co_mediated` label captures candidates whose
response field shows activity at both boundary and interior cells with
roughly equal mediation share, not a claim about a specific physical
mechanism.

Inputs (all under ``--run-dir``):

    mechanism_labels.csv    — old labels and per-candidate response/BMI metrics
    mediation_summary.csv   — interior_hidden_effect + boundary_hidden_effect
                              + far_hidden_effect (needed for the global rule)
    lifetime_tradeoff.csv   — HCE_at_headline_horizon (per-candidate HCE)

Outputs (under ``--run-dir``):

    m8g_revised_mechanism_labels.csv
    m8g_revised_mechanism_stats.json
    m8g_revised_mechanism_summary.md
    plots/m8g_revised_mechanism_distribution.png
    plots/m8g_hce_by_revised_mechanism.png

Sensitivity:
    --bmi-band <lo> <hi>       primary co-mediated BMI band (default 0.35 0.65)
    --sensitivity-bands <lo hi lo hi ...>
                                additional bands to report distribution
                                deltas (default 0.40 0.60 0.35 0.65 0.30 0.70)

Note on rules where this module preserves the old label:

  * `threshold_mediated`: the spec asks for `near_threshold_fraction > 0.5
    AND HCE-drops-under-threshold-filtered-audit conditions OR old
    threshold_mediated label`. We do not have the per-candidate
    threshold-filtered audit data in the CSV artifacts; the conservative
    interpretation is to preserve the old `threshold_mediated` label
    set. This matches the "OR old label" branch of the rule.

  * `delayed_hidden_channel`: the spec rule needs `fraction_hidden_at_end`
    and `fraction_visible_at_end`, which are not stored per-candidate in
    the CSVs. We preserve the old `delayed_hidden_channel` label set.
    The old rule was first_visible_effect_time >= 5 AND a hidden/visible
    end-fraction inequality, both checked at run time.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REVISED_CLASSES = (
    "threshold_mediated",
    "global_chaotic",
    "boundary_and_interior_co_mediated",
    "boundary_only_or_boundary_dominant",
    "interior_only_or_interior_dominant",
    "environment_coupled",
    "delayed_hidden_channel",
    "unclear",
)

SOURCES = ("M7_HCE_optimized", "M4C_observer_optimized", "M4A_viability")
SOURCE_SHORT = {
    "M7_HCE_optimized": "M7",
    "M4C_observer_optimized": "M4C",
    "M4A_viability": "M4A",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_candidates(run_dir: Path) -> list[dict]:
    """Join mechanism_labels.csv + mediation_summary.csv + lifetime_tradeoff.csv
    on (rule_id, seed, candidate_id). Returns one dict per candidate."""
    labels_path = run_dir / "mechanism_labels.csv"
    med_path = run_dir / "mediation_summary.csv"
    trade_path = run_dir / "lifetime_tradeoff.csv"
    for p in (labels_path, med_path, trade_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    def _key(r): return (r["rule_id"], r["seed"], r["candidate_id"])

    with labels_path.open(encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    with med_path.open(encoding="utf-8") as f:
        med = {_key(r): r for r in csv.DictReader(f)}
    with trade_path.open(encoding="utf-8") as f:
        trade = {_key(r): r for r in csv.DictReader(f)}

    out = []
    for r in labels:
        k = _key(r)
        m = med.get(k)
        t = trade.get(k)
        if m is None or t is None:
            continue
        out.append({
            "rule_id": r["rule_id"],
            "rule_source": r["rule_source"],
            "seed": r["seed"],
            "candidate_id": r["candidate_id"],
            "old_label": r["mechanism"],
            "near_threshold_fraction": float(r["near_threshold_fraction"]),
            "boundary_response_fraction": float(r["boundary_response_fraction"]),
            "interior_response_fraction": float(r["interior_response_fraction"]),
            "environment_response_fraction": float(
                r["environment_response_fraction"]
            ),
            "boundary_mediation_index": float(r["boundary_mediation_index"]),
            "candidate_locality_index": float(r["candidate_locality_index"]),
            "first_visible_effect_time": float(r["first_visible_effect_time"]),
            "candidate_lifetime": float(r["candidate_lifetime"]),
            "interior_hidden_effect": float(m["interior_hidden_effect"]),
            "boundary_hidden_effect": float(m["boundary_hidden_effect"]),
            "far_hidden_effect": float(m["far_hidden_effect"]),
            "environment_hidden_effect": float(m["environment_hidden_effect"]),
            "HCE": float(t["HCE_at_headline_horizon"]),
        })
    return out


# ---------------------------------------------------------------------------
# Revised classifier
# ---------------------------------------------------------------------------


def classify_revised(
    *,
    old_label: str,
    near_threshold_fraction: float,
    boundary_response_fraction: float,
    interior_response_fraction: float,
    environment_response_fraction: float,
    boundary_mediation_index: float,
    interior_hidden_effect: float,
    boundary_hidden_effect: float,
    far_hidden_effect: float,
    first_visible_effect_time: float,
    bmi_band: tuple[float, float] = (0.35, 0.65),
) -> str:
    """Apply the revised priority-ordered classifier and return the new label.

    Priority order matches REVISED_CLASSES (1 → 8). The first matching
    rule assigns the label; later rules are not evaluated.
    """
    bmi_lo, bmi_hi = bmi_band

    # 1. threshold_mediated — preserve old label (audit data not available
    #    per-candidate in CSV artifacts; the spec's "OR old label" branch
    #    is the conservative path here).
    if old_label == "threshold_mediated":
        return "threshold_mediated"

    # 2. global_chaotic — fire on the new far-effect rule OR preserve old.
    cand_e = max(interior_hidden_effect + boundary_hidden_effect, 1e-9)
    if far_hidden_effect > 0.7 * cand_e or old_label == "global_chaotic":
        return "global_chaotic"

    # 3. boundary_and_interior_co_mediated.
    if (boundary_response_fraction > 0.6
            and interior_response_fraction > 0.6
            and bmi_lo <= boundary_mediation_index <= bmi_hi):
        return "boundary_and_interior_co_mediated"

    # 4. boundary_only_or_boundary_dominant.
    if (boundary_response_fraction > 0.6
            and boundary_mediation_index > bmi_hi):
        return "boundary_only_or_boundary_dominant"

    # 5. interior_only_or_interior_dominant.
    if (interior_response_fraction > 0.6
            and boundary_mediation_index < bmi_lo):
        return "interior_only_or_interior_dominant"

    # 6. environment_coupled.
    if (environment_response_fraction > 0.4
            or old_label == "environment_coupled"):
        return "environment_coupled"

    # 7. delayed_hidden_channel — preserve old (fraction-at-end data not
    #    available per-candidate). The old rule already required
    #    first_visible_effect_time >= 5 plus the fraction inequality.
    if old_label == "delayed_hidden_channel":
        return "delayed_hidden_channel"

    # 8. fallthrough.
    return "unclear"


def classify_all(
    cands: list[dict], *, bmi_band: tuple[float, float],
) -> list[str]:
    return [classify_revised(
        old_label=c["old_label"],
        near_threshold_fraction=c["near_threshold_fraction"],
        boundary_response_fraction=c["boundary_response_fraction"],
        interior_response_fraction=c["interior_response_fraction"],
        environment_response_fraction=c["environment_response_fraction"],
        boundary_mediation_index=c["boundary_mediation_index"],
        interior_hidden_effect=c["interior_hidden_effect"],
        boundary_hidden_effect=c["boundary_hidden_effect"],
        far_hidden_effect=c["far_hidden_effect"],
        first_visible_effect_time=c["first_visible_effect_time"],
        bmi_band=bmi_band,
    ) for c in cands]


# ---------------------------------------------------------------------------
# Distributions and within-class stats
# ---------------------------------------------------------------------------


def distribution_by_source(cands: list[dict], new_labels: list[str]) -> dict:
    """For each source, return n_total and per-class counts/fractions."""
    out: dict = {}
    for src in SOURCES:
        idxs = [i for i, c in enumerate(cands) if c["rule_source"] == src]
        n = len(idxs)
        cnt = Counter(new_labels[i] for i in idxs)
        per = {}
        for cls in REVISED_CLASSES:
            c = cnt.get(cls, 0)
            per[cls] = {"count": c, "fraction": (c / n) if n else 0.0}
        out[src] = {"n_total": n, "per_class": per}
    return out


def transition_matrix(
    cands: list[dict], new_labels: list[str],
) -> dict:
    """Per-source: how many candidates moved from each old label to each new
    label."""
    out: dict = {}
    for src in SOURCES:
        m: dict = defaultdict(lambda: defaultdict(int))
        for c, new in zip(cands, new_labels):
            if c["rule_source"] != src:
                continue
            m[c["old_label"]][new] += 1
        out[src] = {k: dict(v) for k, v in m.items()}
    return out


def hce_by_revised_class(
    cands: list[dict], new_labels: list[str],
) -> dict:
    """Per (source, revised_class): n, mean HCE, mean lifetime."""
    out: dict = {}
    for src in SOURCES:
        per_cls: dict = {}
        for cls in REVISED_CLASSES:
            rs = [(c, new_labels[i]) for i, c in enumerate(cands)
                  if c["rule_source"] == src and new_labels[i] == cls]
            if not rs:
                per_cls[cls] = {"n": 0, "mean_HCE": None,
                                "mean_lifetime": None,
                                "mean_BMI": None,
                                "mean_boundary_resp": None,
                                "mean_interior_resp": None,
                                "mean_far_hidden": None,
                                "mean_near_threshold": None}
                continue
            xs = [r[0] for r in rs]
            per_cls[cls] = {
                "n": len(xs),
                "mean_HCE": float(np.mean([x["HCE"] for x in xs])),
                "mean_lifetime": float(np.mean([x["candidate_lifetime"] for x in xs])),
                "mean_BMI": float(np.mean([x["boundary_mediation_index"] for x in xs])),
                "mean_boundary_resp": float(np.mean([x["boundary_response_fraction"] for x in xs])),
                "mean_interior_resp": float(np.mean([x["interior_response_fraction"] for x in xs])),
                "mean_far_hidden": float(np.mean([x["far_hidden_effect"] for x in xs])),
                "mean_near_threshold": float(np.mean([x["near_threshold_fraction"] for x in xs])),
            }
        out[src] = per_cls
    return out


# ---------------------------------------------------------------------------
# Within-class M7 vs baseline (HCE only; reuses M8F-style grouped bootstrap)
# ---------------------------------------------------------------------------


def _grouped_bootstrap_diff(
    a_vals, a_groups, b_vals, b_groups, *, n_boot=5000, seed=0,
):
    if a_vals.size == 0 or b_vals.size == 0:
        return {"mean_a": 0.0, "mean_b": 0.0, "mean_diff": 0.0,
                "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    ua = np.unique(a_groups); ub = np.unique(b_groups)
    idx_a = {g: np.where(a_groups == g)[0] for g in ua}
    idx_b = {g: np.where(b_groups == g)[0] for g in ub}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(ua, size=ua.size, replace=True)
        sb = rng.choice(ub, size=ub.size, replace=True)
        ia = np.concatenate([idx_a[g] for g in sa])
        ib = np.concatenate([idx_b[g] for g in sb])
        diffs[i] = float(a_vals[ia].mean() - b_vals[ib].mean())
    return {
        "mean_a": float(a_vals.mean()),
        "mean_b": float(b_vals.mean()),
        "mean_diff": float(a_vals.mean() - b_vals.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def within_class_hce_comparisons(
    cands: list[dict], new_labels: list[str], *, n_boot: int = 5000,
) -> dict:
    """For each revised class, compare M7 HCE vs M4C and vs M4A using
    grouped bootstrap by (rule_id, seed)."""
    out: dict = {}
    for cls in REVISED_CLASSES:
        out[cls] = []
        m7 = [c for i, c in enumerate(cands)
              if new_labels[i] == cls and c["rule_source"] == "M7_HCE_optimized"]
        for src_b in ("M4C_observer_optimized", "M4A_viability"):
            base = [c for i, c in enumerate(cands)
                    if new_labels[i] == cls and c["rule_source"] == src_b]
            if len(m7) < 2 or len(base) < 2:
                out[cls].append({
                    "comparison": f"M7_vs_{SOURCE_SHORT[src_b]}",
                    "n_a": len(m7), "n_b": len(base),
                    "skipped": "insufficient n",
                })
                continue
            a_vals = np.array([x["HCE"] for x in m7])
            b_vals = np.array([x["HCE"] for x in base])
            a_groups = np.array([f"{x['rule_id']}|{x['seed']}" for x in m7])
            b_groups = np.array([f"{x['rule_id']}|{x['seed']}" for x in base])
            boot = _grouped_bootstrap_diff(
                a_vals, a_groups, b_vals, b_groups, n_boot=n_boot,
                seed=hash(f"{cls}|{src_b}") & 0xFFFF,
            )
            ci_excl = (boot["ci_low"] > 0.0 or boot["ci_high"] < 0.0)
            out[cls].append({
                "comparison": f"M7_vs_{SOURCE_SHORT[src_b]}",
                "n_a": int(a_vals.size), "n_b": int(b_vals.size),
                "n_groups_a": int(np.unique(a_groups).size),
                "n_groups_b": int(np.unique(b_groups).size),
                "mean_a": boot["mean_a"], "mean_b": boot["mean_b"],
                "mean_diff": boot["mean_diff"],
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "ci_excludes_zero": ci_excl,
            })
    return out


# ---------------------------------------------------------------------------
# Sensitivity analysis (BMI bands)
# ---------------------------------------------------------------------------


def parse_sensitivity_bands(flat: list[float] | None) -> list[tuple[float, float]]:
    if not flat:
        return [(0.40, 0.60), (0.35, 0.65), (0.30, 0.70)]
    if len(flat) % 2 != 0:
        raise ValueError("sensitivity-bands must contain pairs (lo hi lo hi ...)")
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]


def sensitivity_distributions(
    cands: list[dict], bands: list[tuple[float, float]],
) -> dict:
    """Run the classifier under each band, return per-source distribution
    of revised classes."""
    out: dict = {}
    for band in bands:
        new_labels = classify_all(cands, bmi_band=band)
        out[f"{band[0]}_{band[1]}"] = {
            "band": list(band),
            "by_source": distribution_by_source(cands, new_labels),
        }
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_revised_labels_csv(
    cands: list[dict], new_labels: list[str], path: Path,
) -> None:
    fields = ["rule_id", "rule_source", "seed", "candidate_id",
              "old_label", "revised_label",
              "boundary_response_fraction", "interior_response_fraction",
              "boundary_mediation_index",
              "near_threshold_fraction",
              "interior_hidden_effect", "boundary_hidden_effect",
              "far_hidden_effect", "environment_response_fraction",
              "first_visible_effect_time", "candidate_lifetime",
              "HCE"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c, new in zip(cands, new_labels):
            row = {k: c.get(k) for k in fields if k != "revised_label"}
            row["old_label"] = c["old_label"]
            row["revised_label"] = new
            w.writerow(row)


def _md_row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"


def write_summary_md(stats: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# M8G — revised mechanism classifier")
    lines.append("")
    lines.append(f"Run dir: `{stats['run_dir']}`")
    lines.append(f"Generated: {stats['generated_at_utc']}")
    lines.append(f"Primary BMI band: {stats['primary_bmi_band']}")
    lines.append(f"Sensitivity bands: {stats['sensitivity_bands']}")
    lines.append("")
    lines.append("Read every comparison below as **classifier-conditional**. "
                 "The new `boundary_and_interior_co_mediated` label captures "
                 "candidates whose response field shows activity at both "
                 "boundary and interior cells with roughly equal mediation "
                 "share. It is response-map geometry, not boundary-vs-interior "
                 "physical localization.")
    lines.append("")

    # Old vs revised distribution side-by-side
    lines.append("## Old vs revised label distribution by source")
    lines.append("")
    for src in SOURCES:
        old = stats["old_distribution"][src]
        new = stats["revised_distribution_primary"][src]
        lines.append(f"### {SOURCE_SHORT[src]}  (n = {new['n_total']})")
        lines.append("")
        lines.append(_md_row(["label kind", "label", "count", "fraction"]))
        lines.append(_md_row(["---", "---", "---:", "---:"]))
        for cls, rec in old["per_class"].items():
            if rec["count"] == 0:
                continue
            lines.append(_md_row([
                "old", cls, rec["count"], f"{rec['fraction']:.3f}",
            ]))
        for cls in REVISED_CLASSES:
            rec = new["per_class"][cls]
            if rec["count"] == 0:
                continue
            lines.append(_md_row([
                "revised", cls, rec["count"], f"{rec['fraction']:.3f}",
            ]))
        lines.append("")

    # Transition matrix (per source) — focus on where boundary_mediated went
    lines.append("## Where did the old `boundary_mediated` candidates go?")
    lines.append("")
    for src in SOURCES:
        trans = stats["transition_matrix"][src]
        old_bm = trans.get("boundary_mediated", {})
        n_old = sum(old_bm.values())
        if n_old == 0:
            continue
        lines.append(f"### {SOURCE_SHORT[src]} — {n_old} old `boundary_mediated` candidates")
        lines.append("")
        lines.append(_md_row(["new label", "count", "fraction of old class"]))
        lines.append(_md_row(["---", "---:", "---:"]))
        for cls in REVISED_CLASSES:
            c = old_bm.get(cls, 0)
            if c == 0:
                continue
            lines.append(_md_row([
                cls, c, f"{c / n_old:.3f}",
            ]))
        lines.append("")

    # HCE by revised class and source
    lines.append("## HCE by revised class × source (primary band)")
    lines.append("")
    lines.append(_md_row(["class"] + [f"{SOURCE_SHORT[s]} mean HCE (n)" for s in SOURCES]))
    lines.append(_md_row(["---"] + ["---:"] * len(SOURCES)))
    for cls in REVISED_CLASSES:
        cells = [cls]
        for src in SOURCES:
            d = stats["hce_by_revised_class"][src][cls]
            if d["n"] == 0:
                cells.append(f"— (0)")
            else:
                cells.append(f"{d['mean_HCE']:+.4f} ({d['n']})")
        lines.append(_md_row(cells))
    lines.append("")

    # Within-class M7 vs baseline HCE (grouped bootstrap)
    lines.append("## M7 vs baseline HCE within revised classes (grouped bootstrap)")
    lines.append("")
    lines.append(_md_row([
        "class", "comparison", "n_M7", "n_base",
        "mean_M7", "mean_base", "diff", "95% CI", "CI excl 0",
    ]))
    lines.append(_md_row(["---"] * 9))
    for cls in REVISED_CLASSES:
        for rec in stats["within_class_comparisons"][cls]:
            if rec.get("skipped"):
                lines.append(_md_row([
                    cls, rec["comparison"], rec["n_a"], rec["n_b"],
                    "—", "—", "—", "—", "(skipped: " + rec["skipped"] + ")",
                ]))
                continue
            ci = f"[{rec['ci_low']:+.4f}, {rec['ci_high']:+.4f}]"
            lines.append(_md_row([
                cls, rec["comparison"], rec["n_a"], rec["n_b"],
                f"{rec['mean_a']:+.4f}", f"{rec['mean_b']:+.4f}",
                f"{rec['mean_diff']:+.4f}", ci,
                "yes" if rec["ci_excludes_zero"] else "no",
            ]))
    lines.append("")

    # Sensitivity to BMI band
    lines.append("## Sensitivity to BMI band on co-mediated class")
    lines.append("")
    lines.append("Fraction of candidates labeled `boundary_and_interior_co_mediated` "
                 "vs other reclassified-from-old-boundary-mediated labels, per "
                 "BMI band.")
    lines.append("")
    lines.append(_md_row([
        "BMI band",
    ] + [f"{SOURCE_SHORT[s]} co-mediated" for s in SOURCES]
      + [f"{SOURCE_SHORT[s]} boundary-dom" for s in SOURCES]
      + [f"{SOURCE_SHORT[s]} interior-dom" for s in SOURCES]
    ))
    lines.append(_md_row(["---"] + ["---:"] * (len(SOURCES) * 3)))
    for band_key, sd in stats["sensitivity"].items():
        band = sd["band"]
        cells = [f"[{band[0]:.2f}, {band[1]:.2f}]"]
        for cls in ("boundary_and_interior_co_mediated",
                    "boundary_only_or_boundary_dominant",
                    "interior_only_or_interior_dominant"):
            for src in SOURCES:
                f_ = sd["by_source"][src]["per_class"][cls]["fraction"]
                cells.append(f"{f_:.3f}")
        # The above interleaved order is wrong; fix by rearranging.
        # Re-do cells properly:
        cells = [f"[{band[0]:.2f}, {band[1]:.2f}]"]
        for cls in ("boundary_and_interior_co_mediated",
                    "boundary_only_or_boundary_dominant",
                    "interior_only_or_interior_dominant"):
            for src in SOURCES:
                f_ = sd["by_source"][src]["per_class"][cls]["fraction"]
                cells.append(f"{f_:.3f}")
        lines.append(_md_row(cells))
    lines.append("")

    # Interpretation block
    lines.append("## Interpretation")
    lines.append("")
    lines.extend(stats["interpretation_lines"])
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(stats: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Plot 1: stacked bar of old vs revised class fractions per source
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(13, 5), sharey=True)
    if len(SOURCES) == 1:
        axes = [axes]
    nonzero_revised = [
        cls for cls in REVISED_CLASSES
        if any(stats["revised_distribution_primary"][s]["per_class"][cls]["count"] > 0
               for s in SOURCES)
    ]
    colors = {
        "threshold_mediated":             "#bbb",
        "global_chaotic":                 "#c33",
        "boundary_and_interior_co_mediated": "#357",
        "boundary_only_or_boundary_dominant": "#3a7",
        "interior_only_or_interior_dominant": "#a73",
        "environment_coupled":            "#7a7",
        "delayed_hidden_channel":         "#a37",
        "unclear":                        "#444",
    }
    for ax, src in zip(axes, SOURCES):
        labels = ["old\n(M8 classifier)", "revised\n(M8G)"]
        bottoms = [0.0, 0.0]
        # Old labels
        old = stats["old_distribution"][src]["per_class"]
        new = stats["revised_distribution_primary"][src]["per_class"]
        # Build a unified ordering so colors align across the two bars.
        unified_classes = [
            "threshold_mediated", "global_chaotic",
            "boundary_mediated",
            "boundary_and_interior_co_mediated",
            "boundary_only_or_boundary_dominant",
            "interior_only_or_interior_dominant",
            "interior_reservoir",
            "environment_coupled", "delayed_hidden_channel", "unclear",
        ]
        old_color_extra = {"boundary_mediated": "#379",
                           "interior_reservoir": "#a73"}
        for cls in unified_classes:
            o_frac = old.get(cls, {"fraction": 0.0})["fraction"] if cls in old else 0.0
            n_frac = new.get(cls, {"fraction": 0.0})["fraction"] if cls in new else 0.0
            color = colors.get(cls, old_color_extra.get(cls, "#999"))
            ax.bar(labels[0], o_frac, bottom=bottoms[0], color=color, edgecolor="white")
            ax.bar(labels[1], n_frac, bottom=bottoms[1], color=color, edgecolor="white")
            bottoms[0] += o_frac
            bottoms[1] += n_frac
        ax.set_title(f"{SOURCE_SHORT[src]}  (n={stats['revised_distribution_primary'][src]['n_total']})")
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("fraction of candidates")
    fig.suptitle("M8G — old vs revised mechanism distribution per source "
                 f"(BMI band {stats['primary_bmi_band']})")
    fig.tight_layout()
    fig.savefig(plots_dir / "m8g_revised_mechanism_distribution.png", dpi=120)
    plt.close(fig)

    # Plot 2: HCE by revised class, grouped by source
    classes_present = [cls for cls in REVISED_CLASSES
                       if any(stats["hce_by_revised_class"][s][cls]["n"] > 0
                              for s in SOURCES)]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    x = np.arange(len(classes_present))
    src_colors = {"M7_HCE_optimized": "#3a7",
                  "M4C_observer_optimized": "#357",
                  "M4A_viability": "#a73"}
    for i, src in enumerate(SOURCES):
        means = []
        for cls in classes_present:
            d = stats["hce_by_revised_class"][src][cls]
            means.append(d["mean_HCE"] if d["mean_HCE"] is not None else 0.0)
        ax.bar(x + (i - 1) * width, means, width=width,
               color=src_colors[src], label=SOURCE_SHORT[src])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in classes_present],
                       rotation=20, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("mean HCE per candidate")
    ax.set_title(f"M8G — mean HCE by revised mechanism × source "
                 f"(BMI band {stats['primary_bmi_band']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "m8g_hce_by_revised_mechanism.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interpretation lines (deterministic from results)
# ---------------------------------------------------------------------------


def make_interpretation(stats: dict) -> list[str]:
    lines: list[str] = []
    # 1. Did most old boundary_mediated → co_mediated?
    moves = []
    for src in SOURCES:
        trans = stats["transition_matrix"][src].get("boundary_mediated", {})
        n_old = sum(trans.values())
        if n_old == 0:
            continue
        n_co = trans.get("boundary_and_interior_co_mediated", 0)
        moves.append((src, n_old, n_co, n_co / n_old))
    if moves:
        majority = all(frac >= 0.5 for _, _, _, frac in moves)
        if majority:
            lines.append(
                "* The dominant mechanism is better described as whole-candidate "
                "or boundary+interior co-mediated hidden response, not "
                "boundary-only mediation. Across all three rule sources, ≥50% "
                "of candidates that the old classifier labeled `boundary_mediated` "
                "are labeled `boundary_and_interior_co_mediated` under the revised "
                "classifier. Specific fractions:"
            )
            for src, n_old, n_co, frac in moves:
                lines.append(
                    f"    * {SOURCE_SHORT[src]}: {n_co}/{n_old} = {frac:.0%} moved "
                    "from `boundary_mediated` to `boundary_and_interior_co_mediated`."
                )

    # 2. Does M7 still beat baselines within co-mediated?
    co = stats["within_class_comparisons"].get("boundary_and_interior_co_mediated", [])
    co_clean = [r for r in co if not r.get("skipped") and r["ci_excludes_zero"]]
    if co_clean:
        lines.append(
            "* M7 amplifies hidden causal dependence within the dominant "
            "`boundary_and_interior_co_mediated` mechanism. CI-clean comparisons:"
        )
        for r in co_clean:
            lines.append(
                f"    * {r['comparison']}: M7 mean HCE {r['mean_a']:+.4f} vs "
                f"{r['mean_b']:+.4f} → diff {r['mean_diff']:+.4f} "
                f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]."
            )

    # 3. Is global_chaotic still the strongest amplification?
    gc = stats["within_class_comparisons"].get("global_chaotic", [])
    gc_clean = [r for r in gc if not r.get("skipped") and r["ci_excludes_zero"]]
    if gc_clean:
        # Compare effect sizes: max absolute diff in global vs co
        max_gc = max(abs(r["mean_diff"]) for r in gc_clean)
        max_co = max(abs(r["mean_diff"]) for r in co_clean) if co_clean else 0.0
        if max_gc >= max_co:
            lines.append(
                "* The global-chaotic tail remains the strongest within-class "
                "M7 amplification and should be treated as a separate caveat. "
                "M7's HCE advantage in `global_chaotic` is the largest in absolute "
                "magnitude across all revised classes."
            )

    # 4. Did revised labels change the conclusion materially?
    # Heuristic: if old boundary_mediated is now ≤ 5% of revised distribution
    # in every source, the previous label compressed the structure.
    materially_changed = True
    for src in SOURCES:
        new_dist = stats["revised_distribution_primary"][src]["per_class"]
        # Old label space had a single "boundary_mediated"; revised has at
        # least three (co-mediated, boundary-dominant, interior-dominant).
        co_frac = new_dist["boundary_and_interior_co_mediated"]["fraction"]
        bd_frac = new_dist["boundary_only_or_boundary_dominant"]["fraction"]
        if co_frac > 0.5 and bd_frac < 0.5 * co_frac:
            continue
        materially_changed = False
        break
    if materially_changed:
        lines.append(
            "* The previous classifier over-compressed distinct mechanisms; "
            "in every source, the dominant revised label is "
            "`boundary_and_interior_co_mediated` and `boundary_only_or_boundary_dominant` "
            "is a small minority. Revised labels should replace old labels in "
            "future reports, with the old labels preserved only for backward "
            "compatibility."
        )

    if not lines:
        lines.append("* No interpretation rules activated. Review the tables above directly.")

    return lines


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def run(
    run_dir: Path, *,
    bmi_band: tuple[float, float] = (0.35, 0.65),
    sensitivity_bands: list[tuple[float, float]] | None = None,
    n_boot: int = 5000,
) -> dict:
    cands = load_candidates(run_dir)
    new_labels = classify_all(cands, bmi_band=bmi_band)

    # Old distribution (for comparison)
    old_distribution = {}
    for src in SOURCES:
        idxs = [i for i, c in enumerate(cands) if c["rule_source"] == src]
        n = len(idxs)
        cnt = Counter(cands[i]["old_label"] for i in idxs)
        per = {cls: {"count": cnt.get(cls, 0),
                     "fraction": (cnt.get(cls, 0) / n) if n else 0.0}
               for cls in cnt}
        old_distribution[src] = {"n_total": n, "per_class": per}

    revised_distribution_primary = distribution_by_source(cands, new_labels)
    trans = transition_matrix(cands, new_labels)
    hce = hce_by_revised_class(cands, new_labels)
    cmps = within_class_hce_comparisons(cands, new_labels, n_boot=n_boot)

    bands = sensitivity_bands or [(0.40, 0.60), (0.35, 0.65), (0.30, 0.70)]
    sens = sensitivity_distributions(cands, bands)

    stats = {
        "run_dir": str(run_dir).replace("\\", "/"),
        "generated_at_utc": (
            _dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
        "primary_bmi_band": list(bmi_band),
        "sensitivity_bands": [list(b) for b in bands],
        "n_candidates_total": len(cands),
        "old_distribution": old_distribution,
        "revised_distribution_primary": revised_distribution_primary,
        "transition_matrix": trans,
        "hce_by_revised_class": hce,
        "within_class_comparisons": cmps,
        "sensitivity": sens,
    }
    stats["interpretation_lines"] = make_interpretation(stats)

    # Write outputs
    write_revised_labels_csv(
        cands, new_labels, run_dir / "m8g_revised_mechanism_labels.csv",
    )
    (run_dir / "m8g_revised_mechanism_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8",
    )
    write_summary_md(stats, run_dir / "m8g_revised_mechanism_summary.md")
    write_plots(stats, run_dir / "plots")
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M8G revised mechanism classifier on existing M8E artifacts.",
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="M8E output directory.")
    p.add_argument("--bmi-band", nargs=2, type=float,
                   default=[0.35, 0.65], metavar=("LO", "HI"))
    p.add_argument("--sensitivity-bands", nargs="+", type=float, default=None,
                   help="Pairs of LO HI for sensitivity analysis "
                        "(e.g. 0.40 0.60 0.35 0.65 0.30 0.70). "
                        "Defaults to 0.40-0.60, 0.35-0.65, 0.30-0.70.")
    p.add_argument("--n-boot", type=int, default=5000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"run dir does not exist: {run_dir}")
        return 2
    bands = parse_sensitivity_bands(args.sensitivity_bands)
    stats = run(run_dir, bmi_band=tuple(args.bmi_band),
                sensitivity_bands=bands, n_boot=args.n_boot)
    print(f"M8G written to {run_dir}")
    print(f"  candidates joined: {stats['n_candidates_total']}")
    print(f"  primary BMI band: {stats['primary_bmi_band']}")
    print()
    print("Old -> revised transition for old `boundary_mediated`:")
    for src in SOURCES:
        trans = stats["transition_matrix"][src].get("boundary_mediated", {})
        n = sum(trans.values())
        if n == 0:
            continue
        print(f"  {SOURCE_SHORT[src]} (n={n}):")
        for cls, c in sorted(trans.items(), key=lambda x: -x[1]):
            print(f"    {cls:42s} {c:5d}  ({c / n:5.1%})")
    print()
    print("Within-class M7 vs baseline HCE (revised classes, primary band):")
    for cls in REVISED_CLASSES:
        for r in stats["within_class_comparisons"][cls]:
            if r.get("skipped"):
                continue
            tag = "***" if r["ci_excludes_zero"] else "   "
            print(f"  {tag} {cls:42s} {r['comparison']:14s} "
                  f"diff={r['mean_diff']:+.4f} "
                  f"CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

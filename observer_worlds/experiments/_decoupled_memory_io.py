"""Output writers + per-variant statistics for the Stage 5E2 audit.

Separated from the runner so the runner stays small and so tests can
import the helpers without launching a full sweep.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def _safe_mean(xs):
    xs = [float(x) for x in xs if x not in (None, "", "None")]
    return float(np.mean(xs)) if xs else None


def _pearson(xs, ys):
    pairs = [(float(x), float(y))
             for x, y in zip(xs, ys)
             if x not in (None, "", "None") and y not in (None, "", "None")]
    if len(pairs) < 3:
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _bootstrap_pearson_ci(
    xs, ys, *, n_boot=2000, seed=0,
) -> tuple[float | None, float | None, float | None]:
    """Bootstrap CI for Pearson r, paired resampling. Returns
    (point, lo, hi) or (None, None, None)."""
    pairs = [(float(x), float(y))
             for x, y in zip(xs, ys)
             if x not in (None, "", "None") and y not in (None, "", "None")]
    if len(pairs) < 5:
        r = _pearson(xs, ys)
        return (r, None, None)
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    rng = np.random.default_rng(int(seed))
    rs = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        idx = rng.integers(0, a.size, size=a.size)
        ai, bi = a[idx], b[idx]
        if ai.std() < 1e-12 or bi.std() < 1e-12:
            rs[i] = 0.0
        else:
            rs[i] = float(np.corrcoef(ai, bi)[0, 1])
    return (
        float(np.corrcoef(a, b)[0, 1]),
        float(np.quantile(rs, 0.025)),
        float(np.quantile(rs, 0.975)),
    )


# ---------------------------------------------------------------------------
# Per-(variant × source) summary
# ---------------------------------------------------------------------------


def _per_variant_source_summary(trials, *, n_boot: int = 2000) -> dict:
    by_key: dict[tuple, list] = defaultdict(list)
    for t in trials:
        by_key[(t.variant, t.rule_source)].append(t)
    out: dict = {}
    seed_offset = 0
    for (variant, src), ts in by_key.items():
        valid_ts = [t for t in ts if not t.coupled
                     and t.memory_score is not None and t.hce is not None]
        all_ts = [t for t in ts if t.memory_score is not None
                   and t.hce is not None]
        # Per-candidate aggregation: mean memory_score and mean HCE per
        # candidate within this (variant, source).
        by_cand: dict[tuple, list] = defaultdict(list)
        for t in valid_ts:
            by_cand[(t.rule_id, t.seed, t.candidate_id)].append(t)
        cand_rows = []
        for ts_in_cand in by_cand.values():
            mems = [t.memory_score for t in ts_in_cand]
            hces = [t.hce for t in ts_in_cand]
            obs = [t.observer_score for t in ts_in_cand]
            cand_rows.append({
                "mean_memory_score": _safe_mean(mems),
                "mean_hce": _safe_mean(hces),
                "mean_observer": _safe_mean(obs),
                "rule_id": ts_in_cand[0].rule_id,
            })
        hce_xs = [r["mean_hce"] for r in cand_rows]
        mem_ys = [r["mean_memory_score"] for r in cand_rows]
        obs_xs = [r["mean_observer"] for r in cand_rows]
        r_hce_mem, lo_h, hi_h = _bootstrap_pearson_ci(
            hce_xs, mem_ys, n_boot=n_boot, seed=seed_offset,
        )
        seed_offset += 1
        r_obs_mem, lo_o, hi_o = _bootstrap_pearson_ci(
            obs_xs, mem_ys, n_boot=n_boot, seed=seed_offset,
        )
        seed_offset += 1
        out.setdefault(variant, {})[src] = {
            "n_trials_total": len(ts),
            "n_trials_valid_decoupled": len(valid_ts),
            "n_trials_coupled": len(ts) - len(valid_ts),
            "n_candidates": len(cand_rows),
            "mean_memory_score":
                _safe_mean([r["mean_memory_score"] for r in cand_rows]),
            "mean_hce":
                _safe_mean([r["mean_hce"] for r in cand_rows]),
            "mean_observer_score":
                _safe_mean([r["mean_observer"] for r in cand_rows]),
            "pearson_hce_memory": r_hce_mem,
            "pearson_hce_memory_ci_low": lo_h,
            "pearson_hce_memory_ci_high": hi_h,
            "pearson_observer_memory": r_obs_mem,
            "pearson_observer_memory_ci_low": lo_o,
            "pearson_observer_memory_ci_high": hi_o,
            "mean_overlap_fraction":
                _safe_mean([t.overlap_fraction for t in all_ts]),
            "mean_cue_to_hce_distance":
                _safe_mean([t.cue_to_hce_distance for t in all_ts]),
        }
    return out


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def write_decoupled_outputs(trials, out: Path, *, cfg: dict, t_total: float) -> dict:
    # decoupled_memory_trials.csv: full per-trial dump.
    fields = [
        "trial_id", "rule_id", "rule_source", "seed", "candidate_id",
        "track_id", "variant", "horizon", "projection_name",
        "cue_region_size", "hce_region_size", "overlap_fraction",
        "cue_to_hce_distance", "coupled",
        "memory_score", "hce", "observer_score", "candidate_lifetime",
    ]
    with (out / "decoupled_memory_trials.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trials:
            w.writerow({k: getattr(t, k) for k in fields})

    # decoupled_memory_scores.csv: per (variant, source, candidate) aggregates.
    by_cand: dict[tuple, list] = defaultdict(list)
    for t in trials:
        if t.memory_score is None or t.hce is None:
            continue
        by_cand[(t.variant, t.rule_source, t.rule_id, t.seed,
                 t.candidate_id)].append(t)
    score_rows = []
    for (variant, src, rid, seed, cid), ts in by_cand.items():
        any_coupled = any(t.coupled for t in ts)
        score_rows.append({
            "variant": variant, "rule_source": src, "rule_id": rid,
            "seed": int(seed), "candidate_id": cid,
            "n_horizons": len(ts),
            "mean_memory_score":
                _safe_mean([t.memory_score for t in ts]),
            "mean_hce": _safe_mean([t.hce for t in ts]),
            "observer_score": ts[0].observer_score,
            "any_coupled": bool(any_coupled),
            "mean_overlap_fraction":
                _safe_mean([t.overlap_fraction for t in ts]),
        })
    with (out / "decoupled_memory_scores.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "rule_source", "rule_id", "seed", "candidate_id",
            "n_horizons", "mean_memory_score", "mean_hce",
            "observer_score", "any_coupled", "mean_overlap_fraction",
        ])
        w.writeheader()
        for r in score_rows:
            w.writerow(r)

    # decoupling_audit.csv: per (variant, source) audit row.
    summary = _per_variant_source_summary(trials, n_boot=2000)
    audit_rows = []
    for variant, by_src in summary.items():
        for src, agg in by_src.items():
            audit_rows.append({
                "variant": variant, "rule_source": src,
                **{k: agg.get(k) for k in (
                    "n_trials_total", "n_trials_valid_decoupled",
                    "n_trials_coupled", "n_candidates",
                    "mean_overlap_fraction",
                    "mean_cue_to_hce_distance",
                    "mean_memory_score", "mean_hce",
                    "mean_observer_score",
                    "pearson_hce_memory",
                    "pearson_hce_memory_ci_low",
                    "pearson_hce_memory_ci_high",
                    "pearson_observer_memory",
                    "pearson_observer_memory_ci_low",
                    "pearson_observer_memory_ci_high",
                )},
            })
    with (out / "decoupling_audit.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        if audit_rows:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
            w.writeheader()
            for r in audit_rows:
                w.writerow(r)

    # stats_summary.json
    payload = {
        "stage": "5E2",
        "wall_time_seconds": float(time.time() - t_total),
        "n_trials": len(trials),
        "config": cfg,
        "per_variant_source": summary,
    }
    (out / "stats_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )

    # summary.md
    write_summary_md(payload, out / "summary.md")

    # Plots.
    try:
        write_plots(payload, trials, out / "plots")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] plot generation failed: {e!r}")
    return payload


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------


def _md_row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"


def write_summary_md(payload: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Stage 5E2 - decoupled memory-task audit")
    lines.append("")
    lines.append(
        "Test of whether the Stage 5E HCE-memory correlation survives "
        "when the cue and HCE perturbation regions are deliberately "
        "disjoint."
    )
    lines.append("")
    lines.append("Variants tested:")
    lines.append("")
    lines.append("* ``cue_far_boundary`` - cue at boundary; HCE in interior")
    lines.append("* ``cue_environment_shell`` - cue in env shell; HCE in mask")
    lines.append("* ``cue_opposite_side`` - cue left half; HCE right half")
    lines.append("* ``cue_random_remote`` - cue random nearby patch; HCE in mask")
    lines.append("")
    lines.append(
        "**Functional probe only.** Not a consciousness or agency claim."
    )
    lines.append("")

    # Per (variant × source) main table.
    lines.append("## Per (variant x source) summary")
    lines.append("")
    lines.append(_md_row([
        "variant", "source", "n_cands",
        "mean memory", "mean HCE", "mean obs",
        "Pearson(HCE, memory)", "95% CI",
        "Pearson(obs, memory)", "95% CI",
        "n trials valid",
        "mean overlap",
    ]))
    lines.append(_md_row(["---"] * 12))
    for variant, by_src in payload["per_variant_source"].items():
        for src, agg in by_src.items():
            ci_h = (agg.get("pearson_hce_memory_ci_low"),
                    agg.get("pearson_hce_memory_ci_high"))
            ci_o = (agg.get("pearson_observer_memory_ci_low"),
                    agg.get("pearson_observer_memory_ci_high"))
            ci_h_s = (f"[{ci_h[0]:+.3f}, {ci_h[1]:+.3f}]"
                      if ci_h[0] is not None else "-")
            ci_o_s = (f"[{ci_o[0]:+.3f}, {ci_o[1]:+.3f}]"
                      if ci_o[0] is not None else "-")
            r_h = agg.get("pearson_hce_memory")
            r_o = agg.get("pearson_observer_memory")
            lines.append(_md_row([
                variant, src.split("_")[0], agg.get("n_candidates"),
                f"{agg.get('mean_memory_score'):+.4f}"
                    if agg.get("mean_memory_score") is not None else "-",
                f"{agg.get('mean_hce'):+.4f}"
                    if agg.get("mean_hce") is not None else "-",
                f"{agg.get('mean_observer_score'):.2f}"
                    if agg.get("mean_observer_score") is not None else "-",
                f"{r_h:+.3f}" if r_h is not None else "-",
                ci_h_s,
                f"{r_o:+.3f}" if r_o is not None else "-",
                ci_o_s,
                agg.get("n_trials_valid_decoupled"),
                f"{agg.get('mean_overlap_fraction'):.3f}"
                    if agg.get("mean_overlap_fraction") is not None else "-",
            ]))
    lines.append("")

    # Activated interpretation.
    lines.append("## Activated interpretation")
    lines.append("")
    rs_hce = []
    rs_obs = []
    rs_hce_clean = []  # CI excludes 0
    for variant, by_src in payload["per_variant_source"].items():
        for src, agg in by_src.items():
            r = agg.get("pearson_hce_memory")
            if r is not None:
                rs_hce.append(r)
                lo = agg.get("pearson_hce_memory_ci_low")
                hi = agg.get("pearson_hce_memory_ci_high")
                if lo is not None and hi is not None:
                    if lo > 0 or hi < 0:
                        rs_hce_clean.append((variant, src, r))
            ro = agg.get("pearson_observer_memory")
            if ro is not None: rs_obs.append(ro)
    if rs_hce:
        avg = float(np.mean(rs_hce))
        if avg > 0.20:
            lines.append(
                f"* Mean Pearson(HCE, decoupled memory) across "
                f"(variant, source) = {avg:+.3f}. "
                "**HCE predicts memory-like behavior beyond methodological "
                "overlap with the HCE perturbation region.**"
            )
        elif avg < 0.05 and rs_obs and float(np.mean(rs_obs)) > avg:
            lines.append(
                f"* Mean Pearson(HCE, decoupled memory) "
                f"= {avg:+.3f} (small); mean observer-memory "
                f"correlation = {float(np.mean(rs_obs)):+.3f}. "
                "**The Stage 5E HCE-memory association was partly or "
                "mostly driven by shared measurement construction.**"
            )
        else:
            lines.append(
                f"* Mean Pearson(HCE, decoupled memory) = {avg:+.3f}. "
                "**The Stage 5E HCE-memory association is partially "
                "preserved under decoupling but weaker than the "
                "methodologically-coupled measurement.**"
            )
        if rs_hce_clean:
            survived = ", ".join(
                f"{v}/{s.split('_')[0]} (r={r:+.2f})"
                for v, s, r in rs_hce_clean
            )
            lines.append(
                f"* CI-clean HCE-memory correlations survived in: {survived}."
            )
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def write_plots(payload: dict, trials, plots_dir: Path) -> None:
    try:
        plt = _import_plt()
    except ImportError:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    src_colors = {"M7_HCE_optimized": "#3a7",
                  "M4C_observer_optimized": "#357",
                  "M4A_viability": "#a73"}

    # 1. HCE vs decoupled memory by variant (scatter; one panel per variant)
    try:
        path = plots_dir / "hce_vs_decoupled_memory_by_variant.png"
        variants = list(payload["per_variant_source"])
        n = max(1, len(variants))
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]
        for ax, variant in zip(axes, variants):
            for src in payload["per_variant_source"][variant]:
                rs = [t for t in trials
                      if t.variant == variant and t.rule_source == src
                      and not t.coupled and t.memory_score is not None
                      and t.hce is not None]
                if not rs:
                    continue
                xs = [t.hce for t in rs]
                ys = [t.memory_score for t in rs]
                ax.scatter(xs, ys, alpha=0.4,
                            color=src_colors.get(src, "#666"),
                            s=10, label=src.split("_")[0])
            ax.set_title(variant.replace("_", " "), fontsize=10)
            ax.set_xlabel("HCE (in hce_region)")
            if ax is axes[0]:
                ax.set_ylabel("memory_score (cue A vs B in cue_region)")
        axes[-1].legend(fontsize=8)
        fig.suptitle("Stage 5E2 - HCE vs decoupled memory by variant",
                      fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] hce_vs_decoupled_memory_by_variant.png: {e!r}")

    # 2. memory_correlation_by_variant: bar plot of Pearson r per (variant, source)
    try:
        path = plots_dir / "memory_correlation_by_variant.png"
        variants = list(payload["per_variant_source"])
        sources = sorted(set(s for v in variants
                              for s in payload["per_variant_source"][v]))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(variants))
        width = 0.25
        for i, src in enumerate(sources):
            rs = []
            for v in variants:
                d = payload["per_variant_source"][v].get(src) or {}
                r = d.get("pearson_hce_memory")
                rs.append(0.0 if r is None else float(r))
            ax.bar(x + (i - 1) * width, rs, width=width,
                    color=src_colors.get(src, "#666"),
                    label=src.split("_")[0])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(
            [v.replace("_", "\n") for v in variants], fontsize=9,
        )
        ax.set_ylabel("Pearson(HCE, decoupled memory_score)")
        ax.set_title("HCE-memory correlation by variant x source")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] memory_correlation_by_variant.png: {e!r}")

    # 3. overlap fraction histogram
    try:
        path = plots_dir / "overlap_fraction_histogram.png"
        ovs = [t.overlap_fraction for t in trials]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ovs, bins=30, color="#357", edgecolor="white")
        ax.set_xlabel("overlap_fraction (cue ∩ hce / cue)")
        ax.set_ylabel("count of trials")
        ax.axvline(payload["config"].get("decoupling_overlap_threshold", 0.05),
                    color="black", linestyle="--",
                    label=f"threshold = {payload['config'].get('decoupling_overlap_threshold', 0.05)}")
        ax.set_title("overlap fraction distribution across all trials")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] overlap_fraction_histogram.png: {e!r}")

    # 4. cue_to_hce_distance vs memory
    try:
        path = plots_dir / "cue_to_hce_distance_vs_memory.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        for src in sorted({t.rule_source for t in trials}):
            xs = [t.cue_to_hce_distance for t in trials
                  if t.rule_source == src and t.memory_score is not None]
            ys = [t.memory_score for t in trials
                  if t.rule_source == src and t.memory_score is not None]
            if not xs:
                continue
            ax.scatter(xs, ys, alpha=0.4, s=10,
                        color=src_colors.get(src, "#666"),
                        label=src.split("_")[0])
        ax.set_xlabel("cue-to-hce distance (2D centroid distance)")
        ax.set_ylabel("memory_score")
        ax.set_title("cue-to-hce distance vs memory_score")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] cue_to_hce_distance_vs_memory.png: {e!r}")

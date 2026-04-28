"""Per-projection statistical aggregation for Follow-up Topic 1.

The runner produces one row per (rule, seed, projection, candidate)
under ``candidate_metrics.csv``; this module aggregates that flat list
into per-projection means, standard deviations, and counts that the
summary report and plots consume.

Stage 2: smoke-quality means + per-projection candidate counts. No
grouped-bootstrap CIs yet — those are Stage 5+ work and require enough
groups to be meaningful.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# Metric inventory for documentation / tests.
PROJECTION_METRICS: tuple[str, ...] = (
    "n_candidates",
    "mean_HCE",
    "mean_far_HCE",
    "mean_hidden_vs_far_delta",
    "mean_hidden_vs_sham_delta",
    "mean_initial_projection_delta",
    "fraction_clean_initial_projection",  # init delta < 1e-6
    "mean_lifetime",
)


def _safe_mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _safe_std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def project_metrics_template(projections: Iterable[str]) -> dict:
    """Empty-result template; preserved from Stage 1 for tests."""
    return {
        proj: {m: None for m in PROJECTION_METRICS}
        for proj in projections
    }


def aggregate_per_projection(
    candidate_rows: list[dict],
    projections: Iterable[str],
) -> dict:
    """Compute per-projection summary stats from ``candidate_metrics.csv``
    rows.

    Returns a dict shaped like::

        {
            "per_projection": {
                "<projection>": {
                    "n_candidates": int,
                    "n_clean_initial_projection": int,
                    "mean_HCE": float | None,
                    ...
                },
            },
            "metrics_recorded": [...],
            "candidate_count_by_projection": {<projection>: int, ...},
        }
    """
    by_proj: dict[str, list[dict]] = defaultdict(list)
    for r in candidate_rows:
        by_proj[r["projection"]].append(r)
    per_projection: dict[str, dict] = {}
    for proj in projections:
        rs = by_proj.get(proj, [])
        n = len(rs)
        if n == 0:
            per_projection[proj] = {
                "n_candidates": 0,
                "n_clean_initial_projection": 0,
                "fraction_clean_initial_projection": None,
                "mean_HCE": None,
                "std_HCE": None,
                "mean_far_HCE": None,
                "mean_hidden_vs_far_delta": None,
                "mean_hidden_vs_sham_delta": None,
                "mean_initial_projection_delta": None,
                "mean_lifetime": None,
                "_status": "no candidates measured",
            }
            continue

        def col(k):
            return [float(r[k]) for r in rs
                    if r.get(k) not in (None, "", "None")]

        hce_values = col("HCE")
        far_values = col("far_HCE")
        delta_far = col("hidden_vs_far_delta")
        delta_sham = col("hidden_vs_sham_delta")
        init_delta = col("initial_projection_delta")
        lifetimes = col("lifetime")

        n_clean = sum(1 for v in init_delta if v < 1e-6)

        per_projection[proj] = {
            "n_candidates": n,
            "n_clean_initial_projection": n_clean,
            "fraction_clean_initial_projection":
                (n_clean / n) if n else None,
            "mean_HCE": _safe_mean(hce_values),
            "std_HCE": _safe_std(hce_values),
            "mean_far_HCE": _safe_mean(far_values),
            "mean_hidden_vs_far_delta": _safe_mean(delta_far),
            "mean_hidden_vs_sham_delta": _safe_mean(delta_sham),
            "mean_initial_projection_delta": _safe_mean(init_delta),
            "mean_lifetime": _safe_mean(lifetimes),
            "_status": "ok",
        }
    return {
        "stage": 2,
        "metrics_recorded": list(PROJECTION_METRICS),
        "per_projection": per_projection,
        "candidate_count_by_projection": {
            p: per_projection[p]["n_candidates"] for p in projections
        },
    }


def summarize(per_projection: dict) -> dict:
    """Stage-1 compatibility shim used by ``test_projection_robustness``.

    Wraps a per-projection metrics template so the existing import
    surface keeps working while Stage 2 adds the real
    :func:`aggregate_per_projection` for the runner.
    """
    return {
        "stage": 1 if (
            isinstance(per_projection, dict)
            and any(isinstance(v, dict) and all(vv is None for vv in v.values())
                    for v in per_projection.values())
        ) else 2,
        "metrics_recorded": list(PROJECTION_METRICS),
        "per_projection": per_projection,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _md_row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"


def write_summary_md(summary: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Follow-up Topic 1 — projection robustness (Stage 2)")
    lines.append("")
    lines.append(
        "Reuses the Stage-1 scaffold (`docs/FOLLOWUP_RESEARCH_ROADMAP.md`). "
        "Each (rule, seed) cell runs the 4D substrate **once**; every "
        "requested projection consumes the same in-memory state stream. "
        "Per-candidate HCE is measured by candidate-local hidden "
        "perturbations; the `far` control varies the perturbation "
        "location; the `sham` value is identically zero by construction.\n"
    )
    lines.append("")
    lines.append("**Read every comparison below as classifier-conditional and "
                 "projection-conditional.** Mechanism-class fractions are "
                 "deferred to Stage 5+ (full M8 classifier integration "
                 "across projections is non-trivial).\n")
    lines.append("")

    # Headline table
    lines.append("## Per-projection summary")
    lines.append("")
    lines.append(_md_row([
        "projection", "n", "mean HCE", "mean far_HCE",
        "mean (HCE - far)", "mean init_proj_delta",
        "frac_clean_init",
    ]))
    lines.append(_md_row(["---"] + ["---:"] * 6))
    for proj, agg in summary["per_projection"].items():
        if agg["n_candidates"] == 0:
            lines.append(_md_row([proj, "0", "—", "—", "—", "—", "—"]))
            continue
        cells = [
            proj,
            str(agg["n_candidates"]),
            f"{agg['mean_HCE']:+.4f}" if agg["mean_HCE"] is not None else "—",
            f"{agg['mean_far_HCE']:+.4f}" if agg["mean_far_HCE"] is not None else "—",
            f"{agg['mean_hidden_vs_far_delta']:+.4f}"
                if agg["mean_hidden_vs_far_delta"] is not None else "—",
            f"{agg['mean_initial_projection_delta']:+.4f}"
                if agg["mean_initial_projection_delta"] is not None else "—",
            f"{agg['fraction_clean_initial_projection']:.2f}"
                if agg["fraction_clean_initial_projection"] is not None else "—",
        ]
        lines.append(_md_row(cells))
    lines.append("")

    # Caveats per projection.
    lines.append("## Per-projection caveats")
    lines.append("")
    lines.append("* **mean_threshold / sum_threshold** — natural threshold "
                 "margin; standard hidden-invisible perturbation logic "
                 "applies.")
    lines.append("* **max_projection / parity_projection** — no natural "
                 "threshold margin; threshold-audit metrics are N/A. The "
                 "hidden-invisible perturbation here is empirical (we "
                 "verify post-hoc that t = peak projection is unchanged "
                 "by checking `initial_projection_delta`).")
    lines.append("* **random_linear_projection** — continuous output; the "
                 "binary detector consumes a per-frame median threshold. "
                 "Smoke-level binarisation; production refinement needed.")
    lines.append("* **multi_channel_projection** — channel 0 is consumed by "
                 "the binary detector. Smoke-level reduction; production "
                 "should treat all channels.")
    lines.append("")
    if summary.get("wall_time_seconds_sweep") is not None:
        lines.append(
            f"Wall time (sweep): "
            f"{summary['wall_time_seconds_sweep']:.1f}s, "
            f"{summary.get('n_cells', 0)} cells, "
            f"{summary.get('n_candidate_rows', 0)} candidate rows.\n"
        )

    path.write_text("\n".join(lines), encoding="utf-8")

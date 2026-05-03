#!/usr/bin/env python3
"""Compare a G4 (or any GPU runner) candidate_metrics.csv against the
Stage 6C CPU baseline.

Lifted from the inline snippets the G3 commit messages used; factored
out so the G4 Linux bootstrap script can call it directly.

Equivalence policy (from docs/GPU_BACKEND_PLAN.md and the G2 audit):
  * Binary projections (mean_threshold, sum_threshold, max_projection,
    parity_projection, multi_channel_projection): bit-identical HCE.
  * Continuous projection (random_linear_projection): max abs delta
    must be <= --rtl-tol (default 1e-6, matches G3 result).
  * initial_projection_delta: bit-identical (swap invariant exact).

Exit codes:
  0  equivalence passes (matches the policy above)
  2  one or more cells outside tolerance, or row-count mismatch,
     or matched-key mismatch
  3  posthoc summary verdict differs from baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def _f(x):
    if x is None:
        return None
    s = (str(x) or "").strip()
    if s in ("", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _index(path: Path) -> dict[tuple, dict]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return {
        (r["rule_id"], int(r["seed"]), r["projection"], int(r["candidate_id"])): r
        for r in rows
    }


def compare_runs(g4_dir: Path, baseline_dir: Path, *, rtl_tol: float = 1e-6) -> dict:
    """Row-by-row comparison. Returns a dict suitable for JSON dump."""
    g4_csv = g4_dir / "candidate_metrics.csv"
    bl_csv = baseline_dir / "candidate_metrics.csv"
    if not g4_csv.exists():
        raise FileNotFoundError(f"missing {g4_csv}")
    if not bl_csv.exists():
        raise FileNotFoundError(f"missing {bl_csv}")

    a = _index(bl_csv)
    b = _index(g4_csv)
    matched = sorted(set(a) & set(b))
    bl_only = sorted(set(a) - set(b))
    g4_only = sorted(set(b) - set(a))

    hce_max = far_max = init_max = 0.0
    binary_n = 0
    binary_match = 0
    cont_n = 0
    cont_max = 0.0
    out_of_tol = []
    for k in matched:
        proj = k[2]
        is_continuous = ("random_linear" in proj)
        for fn in ("HCE", "far_HCE", "initial_projection_delta"):
            va = _f(a[k][fn])
            vb = _f(b[k][fn])
            if va is None or vb is None:
                continue
            if math.isnan(va) or math.isnan(vb):
                continue
            d = abs(va - vb)
            if fn == "HCE":
                hce_max = max(hce_max, d)
            elif fn == "far_HCE":
                far_max = max(far_max, d)
            else:
                init_max = max(init_max, d)
        va = _f(a[k]["HCE"])
        vb = _f(b[k]["HCE"])
        if is_continuous:
            cont_n += 1
            if va is not None and vb is not None:
                d = abs(va - vb)
                cont_max = max(cont_max, d)
                if d > rtl_tol:
                    out_of_tol.append({
                        "key": list(k), "field": "HCE",
                        "baseline": va, "g4": vb, "abs_delta": d,
                    })
        else:
            binary_n += 1
            if va == vb:
                binary_match += 1
            else:
                out_of_tol.append({
                    "key": list(k), "field": "HCE_binary",
                    "baseline": va, "g4": vb,
                    "abs_delta": (abs(va - vb) if (va is not None and vb is not None) else None),
                })

    init_pass = (init_max == 0.0)
    binary_pass = (binary_match == binary_n)
    continuous_pass = (cont_max <= rtl_tol)
    overall_pass = bool(
        len(matched) == len(a) == len(b)
        and binary_pass and continuous_pass and init_pass
    )

    return {
        "baseline_csv": str(bl_csv),
        "g4_csv": str(g4_csv),
        "rows_baseline": len(a),
        "rows_g4": len(b),
        "matched_keys": len(matched),
        "baseline_only_keys": len(bl_only),
        "g4_only_keys": len(g4_only),
        "binary": {
            "n": binary_n,
            "bit_identical": binary_match,
            "pass": binary_pass,
        },
        "continuous": {
            "n": cont_n,
            "max_abs_HCE_delta": cont_max,
            "tolerance": rtl_tol,
            "pass": continuous_pass,
        },
        "max_abs_HCE_delta_overall": hce_max,
        "max_abs_far_HCE_delta_overall": far_max,
        "max_abs_init_projection_delta_overall": init_max,
        "init_projection_delta_pass": init_pass,
        "overall_pass": overall_pass,
        "examples_outside_tolerance": out_of_tol[:10],
    }


def run_posthoc(run_dir: Path) -> dict:
    """Invoke the projection_robustness_posthoc analyzer; parse the
    'cells with normalized_HCE > 0.5' counters from its stdout."""
    cmd = [
        sys.executable, "-m",
        "observer_worlds.analysis.projection_robustness_posthoc",
        "--run-dir", str(run_dir),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
    )
    out = (result.stdout or "") + (result.stderr or "")
    n_above = n_ci_clean = None
    for line in out.splitlines():
        ls = line.strip()
        if "cells with normalized_HCE > 0.5" in ls:
            try:
                # form: "  cells with normalized_HCE > 0.5: 17/18 (94%)"
                tail = ls.split(":", 1)[1].strip()
                n_above = int(tail.split("/", 1)[0].strip())
            except Exception:
                pass
        if "CI lower bound > 0.5" in ls:
            try:
                tail = ls.split(":", 1)[1].strip()
                n_ci_clean = int(tail.split("/", 1)[0].strip())
            except Exception:
                pass
    return {
        "returncode": result.returncode,
        "n_cells_normalized_HCE_above_0_5": n_above,
        "n_cells_CI_lower_above_0_5": n_ci_clean,
        "stdout_tail": out.strip().splitlines()[-15:],
    }


def write_md_summary(report: dict, posthoc_g4: dict, posthoc_bl: dict | None,
                     out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# G4 vs Stage 6C equivalence report")
    lines.append("")
    lines.append(f"* baseline: `{report['baseline_csv']}`")
    lines.append(f"* g4:       `{report['g4_csv']}`")
    lines.append("")
    lines.append("## Row-by-row")
    lines.append("")
    lines.append(
        f"| metric | value |\n|---|---:|\n"
        f"| rows (baseline) | {report['rows_baseline']} |\n"
        f"| rows (G4) | {report['rows_g4']} |\n"
        f"| matched keys | {report['matched_keys']} |\n"
        f"| baseline-only keys | {report['baseline_only_keys']} |\n"
        f"| g4-only keys | {report['g4_only_keys']} |\n"
        f"| binary HCE bit-identical | {report['binary']['bit_identical']} / {report['binary']['n']} |\n"
        f"| continuous HCE within tol ({report['continuous']['tolerance']}) | "
        f"{'yes' if report['continuous']['pass'] else 'NO'} "
        f"(max abs delta {report['continuous']['max_abs_HCE_delta']:.3e}) |\n"
        f"| max abs far_HCE delta | {report['max_abs_far_HCE_delta_overall']:.3e} |\n"
        f"| max abs init_proj_delta | {report['max_abs_init_projection_delta_overall']:.3e} |\n"
        f"| **overall row-by-row pass** | **{'PASS' if report['overall_pass'] else 'FAIL'}** |"
    )
    lines.append("")
    lines.append("## Posthoc verdict")
    lines.append("")

    def _row(name: str, p: dict | None) -> str:
        if p is None:
            return f"| {name} | (not computed) |"
        return (
            f"| {name} | n>{0.5}: "
            f"{p['n_cells_normalized_HCE_above_0_5']}/18; "
            f"CI lower>{0.5}: {p['n_cells_CI_lower_above_0_5']}/18 |"
        )

    lines.append("| run | counters |")
    lines.append("|---|---|")
    lines.append(_row("Stage 6C CPU baseline", posthoc_bl))
    lines.append(_row("G4 GPU run", posthoc_g4))
    lines.append("")

    posthoc_pass = (
        posthoc_g4 is not None
        and posthoc_bl is not None
        and posthoc_g4["n_cells_normalized_HCE_above_0_5"]
            == posthoc_bl["n_cells_normalized_HCE_above_0_5"]
        and posthoc_g4["n_cells_CI_lower_above_0_5"]
            == posthoc_bl["n_cells_CI_lower_above_0_5"]
    )
    if posthoc_g4 is not None and posthoc_bl is not None:
        lines.append(
            f"**Posthoc verdict pass:** "
            f"{'PASS' if posthoc_pass else 'FAIL'}"
        )
    if report["examples_outside_tolerance"]:
        lines.append("")
        lines.append("## First-10 rows outside tolerance")
        lines.append("")
        for ex in report["examples_outside_tolerance"]:
            lines.append(
                f"* `{ex['key']}` field={ex['field']} "
                f"baseline={ex['baseline']} g4={ex['g4']} "
                f"|delta|={ex.get('abs_delta')}"
            )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g4-run-dir", type=Path, required=True,
                   help="G4 GPU runner output dir.")
    p.add_argument("--stage6c-baseline-dir", type=Path, required=True,
                   help="Stage 6C CPU baseline output dir.")
    p.add_argument("--rtl-tol", type=float, default=1e-6,
                   help="Tolerance for random_linear_projection HCE.")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--skip-posthoc", action="store_true")
    args = p.parse_args(argv)

    g4 = args.g4_run_dir
    bl = args.stage6c_baseline_dir
    if not g4.is_dir():
        print(f"[g4-compare] ERROR: --g4-run-dir is not a dir: {g4}",
              file=sys.stderr)
        return 1
    if not bl.is_dir():
        print(f"[g4-compare] ERROR: --stage6c-baseline-dir is not a dir: {bl}",
              file=sys.stderr)
        return 1

    report = compare_runs(g4, bl, rtl_tol=args.rtl_tol)

    posthoc_g4 = posthoc_bl = None
    if not args.skip_posthoc:
        print("[g4-compare] running posthoc on G4 run dir ...", file=sys.stderr)
        posthoc_g4 = run_posthoc(g4)
        print("[g4-compare] running posthoc on baseline dir ...", file=sys.stderr)
        posthoc_bl = run_posthoc(bl)

    posthoc_pass = (
        posthoc_g4 is not None and posthoc_bl is not None
        and posthoc_g4["n_cells_normalized_HCE_above_0_5"]
            == posthoc_bl["n_cells_normalized_HCE_above_0_5"]
        and posthoc_g4["n_cells_CI_lower_above_0_5"]
            == posthoc_bl["n_cells_CI_lower_above_0_5"]
    )

    bundle = {
        "row_by_row": report,
        "posthoc_g4": posthoc_g4,
        "posthoc_baseline": posthoc_bl,
        "posthoc_pass": posthoc_pass,
        "overall_pass": bool(report["overall_pass"] and (
            posthoc_pass or args.skip_posthoc
        )),
    }

    if args.out_json:
        args.out_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    if args.out_md:
        write_md_summary(report, posthoc_g4, posthoc_bl, args.out_md)

    print(json.dumps(bundle, indent=2))

    if not report["overall_pass"]:
        return 2
    if not args.skip_posthoc and not posthoc_pass:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

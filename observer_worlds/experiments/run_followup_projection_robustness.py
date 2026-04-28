"""Follow-up Topic 1 — projection-robustness experiment runner (Stage 1 skeleton).

Stage 1: this module exposes the documented CLI surface, validates
arguments, prints what would run, and exits 0 without simulating
anything. Stage 2 will plug in the actual evaluation pipeline.

CLI is the contract for Stage 2; see
``docs/PERFORMANCE_STRATEGY_PYTHON.md`` for the standard CLI surface
that every follow-up runner must expose.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from observer_worlds.projection import default_suite


REPO = Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Follow-up Topic 1: projection-robustness experiment.",
    )
    # Standard CLI surface.
    p.add_argument("--quick", action="store_true",
                   help="Smoke run with reduced defaults.")
    p.add_argument("--n-workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--backend", type=str, default="numpy",
                   choices=["numpy", "numba", "cupy", "cuda-batched"])
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 20, 40, 80])
    p.add_argument("--out-root", type=Path, default=REPO / "outputs")
    p.add_argument("--label", type=str, default="followup_projection_robustness")
    # Topic-specific flags.
    p.add_argument("--n-rules-per-source", type=int, default=5)
    p.add_argument("--test-seeds", type=int, nargs="+",
                   default=list(range(6000, 6020)))
    p.add_argument("--projections", nargs="+", default=None,
                   help="Subset of projections to evaluate. "
                        "Default: all six in the default suite.")
    p.add_argument("--hce-replicates", type=int, default=3)
    p.add_argument("--profile", action="store_true",
                   help="Wrap the run with the M-perf profiler.")
    return p


def _smoke_defaults() -> dict:
    return {
        "n_rules_per_source": 1,
        "test_seeds": [6000, 6001],
        "timesteps": 100,
        "max_candidates": 5,
        "projections": ["mean_threshold", "max_projection",
                        "parity_projection"],
        "horizons": [5, 10],
        "hce_replicates": 1,
    }


def _resolve_config(args: argparse.Namespace) -> dict:
    cfg = {
        "n_rules_per_source": args.n_rules_per_source,
        "test_seeds": list(args.test_seeds),
        "timesteps": args.timesteps,
        "max_candidates": args.max_candidates,
        "projections": list(args.projections) if args.projections
                       else default_suite().names(),
        "horizons": list(args.horizons),
        "hce_replicates": args.hce_replicates,
        "backend": args.backend,
        "n_workers": args.n_workers,
    }
    if args.quick:
        cfg.update(_smoke_defaults())
    # Validate projections.
    suite = default_suite()
    for name in cfg["projections"]:
        if name not in suite.names():
            raise SystemExit(
                f"unknown projection {name!r}; "
                f"available: {suite.names()}"
            )
    return cfg


def _make_out_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_root / f"{args.label}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = _resolve_config(args)
    out = _make_out_dir(args)
    (out / "config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8",
    )

    # Stage 1 placeholder. Print plan, write a stub summary.md and exit 0.
    print("=" * 72)
    print("Follow-up Topic 1: projection robustness — Stage 1 skeleton")
    print("=" * 72)
    print(f"  out         = {out}")
    print(f"  backend     = {cfg['backend']}")
    print(f"  n_workers   = {cfg['n_workers']}")
    print(f"  rules/src   = {cfg['n_rules_per_source']}")
    print(f"  seeds       = {len(cfg['test_seeds'])} ({cfg['test_seeds'][0]}..{cfg['test_seeds'][-1]})")
    print(f"  timesteps   = {cfg['timesteps']}")
    print(f"  horizons    = {cfg['horizons']}")
    print(f"  projections = {cfg['projections']}")
    print(f"  replicates  = {cfg['hce_replicates']}")
    print(f"  profile     = {bool(args.profile)}")
    print("Stage 1: no simulation will run.")

    summary = (
        "# Follow-up Topic 1 — projection robustness (Stage 1 skeleton)\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        "Stage 1 only validates the CLI and writes config.json. "
        "Stage 2 will run the full evaluation. See "
        "`docs/FOLLOWUP_RESEARCH_ROADMAP.md` for the topic spec.\n"
    )
    (out / "summary.md").write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

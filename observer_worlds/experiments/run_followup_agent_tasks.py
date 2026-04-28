"""Follow-up Topic 3 — agent-task experiment runner (Stage 1 skeleton).

Validates the CLI surface and writes ``config.json``. Stage 4 will
implement the three task environments (repair, foraging, memory).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

KNOWN_TASKS = ("repair", "foraging", "memory")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Follow-up Topic 3: agent-task experiment.",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--backend", type=str, default="numpy",
                   choices=["numpy", "numba", "cupy", "cuda-batched"])
    p.add_argument("--max-candidates", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[5, 10, 20, 40, 80])
    p.add_argument("--out-root", type=Path, default=REPO / "outputs")
    p.add_argument("--label", type=str, default="followup_agent_tasks")

    p.add_argument("--n-rules-per-source", type=int, default=5)
    p.add_argument("--test-seeds", type=int, nargs="+",
                   default=list(range(6000, 6020)))
    p.add_argument("--tasks", nargs="+", default=list(KNOWN_TASKS),
                   choices=KNOWN_TASKS,
                   help="Which tasks to run.")
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    return p


def _smoke_defaults() -> dict:
    return {
        "n_rules_per_source": 1,
        "test_seeds": [6000, 6001],
        "timesteps": 100,
        "max_candidates": 10,
        "tasks": ["repair", "memory"],
        "horizons": [5, 10],
        "replicates": 1,
    }


def _resolve_config(args: argparse.Namespace) -> dict:
    cfg = {
        "n_rules_per_source": args.n_rules_per_source,
        "test_seeds": list(args.test_seeds),
        "timesteps": args.timesteps,
        "max_candidates": args.max_candidates,
        "tasks": list(args.tasks),
        "horizons": list(args.horizons),
        "replicates": args.replicates,
        "backend": args.backend,
        "n_workers": args.n_workers,
    }
    if args.quick:
        cfg.update(_smoke_defaults())
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

    print("=" * 72)
    print("Follow-up Topic 3: agent-task environments — Stage 1 skeleton")
    print("=" * 72)
    print(f"  out         = {out}")
    print(f"  backend     = {cfg['backend']}")
    print(f"  n_workers   = {cfg['n_workers']}")
    print(f"  tasks       = {cfg['tasks']}")
    print(f"  rules/src   = {cfg['n_rules_per_source']}")
    print(f"  seeds       = {len(cfg['test_seeds'])} ({cfg['test_seeds'][0]}..{cfg['test_seeds'][-1]})")
    print(f"  timesteps   = {cfg['timesteps']}")
    print(f"  horizons    = {cfg['horizons']}")
    print(f"  replicates  = {cfg['replicates']}")
    print(f"  profile     = {bool(args.profile)}")
    print("Stage 1: no simulation or task evaluation will run.")
    print("These are *functional agency* tasks. They are not claims about")
    print("consciousness; see docs/FOLLOWUP_RESEARCH_ROADMAP.md.")

    summary = (
        "# Follow-up Topic 3 — agent-task environments (Stage 1 skeleton)\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        "Stage 1 only validates the CLI and writes config.json. "
        "Stage 4 will implement the repair / foraging / memory task "
        "environments.\n\nNote: these are *functional agency* tasks; the "
        "framework does not claim consciousness or sentience.\n"
    )
    (out / "summary.md").write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

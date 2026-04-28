"""Capture an M7B-class production baseline into ``baselines.json``.

Opt-in script. Not part of pytest. Runs the M8 mechanism-discovery
driver at production scale, captures wall time, candidate count, and
full machine/software metadata, and merges the result into
``tests/perf/baselines.json`` under a named variant.

Typical invocation (from the repository root, with the .venv active)::

    python tests/perf/capture_m7b_baseline.py \\
        --backend numpy \\
        --variant m8_m7b_class_numpy \\
        --update-baseline

    python tests/perf/capture_m7b_baseline.py \\
        --backend cuda-batched \\
        --variant m8_m7b_class_cuda_batched \\
        --update-baseline

The default config matches the captured ``m8_m7b_class_numpy`` baseline:
5 M7 rules x 20 test seeds (6000-6019) x T=500 x 64x64x8x8, n_workers=30.
Override individual fields with the corresponding flags.

This script intentionally does *not* import pytest; it is a stand-alone
runner so it can be invoked outside the test environment if needed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BASELINES_PATH = REPO / "tests" / "perf" / "baselines.json"


# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------


def _detect_cpu_name() -> str:
    """Best-effort CPU name. PowerShell on Windows; /proc/cpuinfo on Linux;
    sysctl on macOS. Returns ``platform.processor()`` as a fallback."""
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            name = out.stdout.strip()
            if name:
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if sys.platform.startswith("linux"):
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            for line in cpuinfo.splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            name = out.stdout.strip()
            if name:
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or "unknown"


def _detect_ram_bytes() -> int | None:
    """Total physical RAM in bytes, or None if undetectable."""
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            v = out.stdout.strip()
            if v.isdigit():
                return int(v)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if sys.platform.startswith("linux"):
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
        except (OSError, ValueError):
            pass
    return None


def _detect_gpu() -> dict:
    """Best-effort NVIDIA GPU info via nvidia-smi."""
    info: dict = {"name": None, "vram_mib": None, "driver": None}
    if shutil.which("nvidia-smi") is None:
        return info
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        line = out.stdout.strip().splitlines()[0] if out.stdout else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            info["name"] = parts[0]
            try:
                info["vram_mib"] = int(parts[1])
            except ValueError:
                pass
            info["driver"] = parts[2]
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        pass
    return info


def _detect_package_versions() -> dict:
    """Versions of the perf-relevant Python packages."""
    out: dict = {}
    for pkg in ("numpy", "scipy", "numba", "cupy"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", None)
        except ImportError:
            out[pkg] = None
    out["python"] = platform.python_version()
    return out


def _detect_cuda_runtime() -> int | None:
    """CUDA runtime version (e.g. 12090) or None."""
    try:
        import cupy
        return int(cupy.cuda.runtime.runtimeGetVersion())
    except Exception:
        return None


def _git_revparse() -> dict:
    """Current git commit SHA, dirty flag, and tag (if HEAD points at one)."""
    info: dict = {"commit": None, "dirty": None, "tag": None}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO), check=False,
        ).stdout.strip()
        info["commit"] = sha or None
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(REPO), check=False,
        ).stdout.strip()
        info["dirty"] = bool(dirty)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    try:
        tag = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO), check=False,
        ).stdout.strip().splitlines()
        info["tag"] = tag[0] if tag else None
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return info


def collect_machine_metadata() -> dict:
    """Build the ``machine`` block recorded in baselines.json."""
    pkgs = _detect_package_versions()
    gpu = _detect_gpu()
    return {
        "cpu": _detect_cpu_name(),
        "cpu_logical_threads": __import__("os").cpu_count(),
        "ram_bytes": _detect_ram_bytes(),
        "ram_gib_approx": (
            round(_detect_ram_bytes() / (1024**3), 2)
            if _detect_ram_bytes() else None
        ),
        "gpu": gpu["name"],
        "vram_mib": gpu["vram_mib"],
        "nvidia_driver": gpu["driver"],
        "os": platform.platform(),
        "python": pkgs["python"],
        "numpy": pkgs["numpy"],
        "scipy": pkgs["scipy"],
        "numba": pkgs["numba"],
        "cupy": pkgs["cupy"],
        "cuda_runtime_version": _detect_cuda_runtime(),
    }


# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "-m",
        "observer_worlds.experiments.run_m8_mechanism_discovery",
        "--m7-rules", str(args.m7_rules),
        "--n-rules-per-source", str(args.n_rules_per_source),
        "--test-seeds", *[str(s) for s in args.test_seeds],
        "--timesteps", str(args.timesteps),
        "--grid", *[str(g) for g in args.grid],
        "--max-candidates", str(args.max_candidates),
        "--hce-replicates", str(args.hce_replicates),
        "--horizons", *[str(h) for h in args.horizons],
        "--backend", args.backend,
        "--label", args.label,
        "--out-root", str(args.out_root),
    ]
    if args.n_workers is not None:
        cmd += ["--n-workers", str(args.n_workers)]
    if args.m4c_rules:
        cmd += ["--m4c-rules", str(args.m4c_rules)]
    if args.m4a_rules:
        cmd += ["--m4a-rules", str(args.m4a_rules)]
    return cmd


_RE_SWEEP = re.compile(r"m8 sweep wall time (\d+(?:\.\d+)?)s")
_RE_CANDS = re.compile(r"Measured (\d+) candidates")


def parse_run_output(stdout: str) -> tuple[int | None, int | None]:
    """Pull (sweep_seconds, n_candidates) out of run stdout."""
    sweep = None
    cands = None
    m = _RE_SWEEP.search(stdout)
    if m:
        sweep = int(round(float(m.group(1))))
    m = _RE_CANDS.search(stdout)
    if m:
        cands = int(m.group(1))
    return sweep, cands


# ---------------------------------------------------------------------------
# baselines.json merge
# ---------------------------------------------------------------------------


def update_baselines_file(variant: str, payload: dict) -> None:
    """Merge a baseline payload into ``baselines.json`` under
    ``production_baselines[variant]``."""
    data = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    data.setdefault("production_baselines", {})[variant] = payload
    BASELINES_PATH.write_text(
        json.dumps(data, indent=4) + "\n", encoding="utf-8",
    )


def load_baseline(variant: str) -> dict | None:
    data = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    return data.get("production_baselines", {}).get(variant)


def get_tolerance() -> dict:
    data = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    return data.get("tolerance_policy", {})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEFAULT_TEST_SEEDS = list(range(6000, 6020))
_DEFAULT_HORIZONS = [1, 2, 3, 5, 10, 20, 40, 80]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--variant", type=str, required=True,
                   help="Key under production_baselines/ in baselines.json. "
                        "Convention: m8_m7b_class_<backend>.")
    p.add_argument("--backend",
                   choices=["numba", "numpy", "cuda", "cuda-batched"],
                   default="cuda-batched")
    p.add_argument("--m7-rules", type=Path,
                   default=REPO / "release" / "rules" / "m7_top_hce_rules.json")
    p.add_argument("--m4c-rules", type=Path, default=None)
    p.add_argument("--m4a-rules", type=Path, default=None)
    p.add_argument("--n-rules-per-source", type=int, default=5)
    p.add_argument("--test-seeds", type=int, nargs="+",
                   default=_DEFAULT_TEST_SEEDS)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8],
                   metavar=("NX", "NY", "NZ", "NW"))
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--hce-replicates", type=int, default=3)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=_DEFAULT_HORIZONS)
    p.add_argument("--n-workers", type=int, default=None,
                   help="Default: cpu_count - 2.")
    p.add_argument("--label", type=str, default="perf_baseline_capture")
    p.add_argument("--out-root", type=Path, default=REPO / "outputs")
    p.add_argument("--update-baseline", action="store_true",
                   help="Merge the captured run into baselines.json.")
    p.add_argument("--timeout-seconds", type=int, default=4 * 3600,
                   help="Subprocess timeout. Default 4 hours.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    machine = collect_machine_metadata()
    git_info = _git_revparse()

    cmd = build_command(args)
    print("=" * 72)
    print(f"Capturing baseline variant={args.variant!r}")
    print(f"  backend={args.backend}  grid={args.grid}  T={args.timesteps}")
    print(f"  n_seeds={len(args.test_seeds)}  rules_per_source={args.n_rules_per_source}")
    print(f"  n_workers={args.n_workers if args.n_workers is not None else 'auto (cpu_count-2)'}")
    print()
    print("Command:")
    print("  " + " ".join(cmd))
    print()
    print("Machine:")
    for k, v in machine.items():
        print(f"  {k:>22} = {v}")
    print()

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=args.timeout_seconds, cwd=str(REPO),
    )
    elapsed = time.time() - t0

    sweep_seconds, n_candidates = parse_run_output(result.stdout)
    print()
    print(f"Run finished in {elapsed:.1f}s wall (rc={result.returncode}).")
    print(f"  m8 sweep wall time (parsed): {sweep_seconds}s")
    print(f"  candidates measured        : {n_candidates}")

    if result.returncode != 0:
        tail = (result.stdout + "\n" + result.stderr).splitlines()[-30:]
        print("\n--- last 30 lines of combined output ---")
        for line in tail:
            print(line)
        return result.returncode

    if n_candidates is None or n_candidates == 0:
        print("WARNING: no candidates measured. Refusing to update baseline.")
        return 2

    payload = {
        "_description": (
            f"M8 mechanism discovery, backend={args.backend}, captured "
            f"by tests/perf/capture_m7b_baseline.py."
        ),
        "captured_at_utc": _dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": git_info["commit"],
        "git_commit_tag": git_info["tag"],
        "git_dirty": git_info["dirty"],
        "machine": machine,
        "command": " ".join(cmd),
        "config": {
            "n_rules_per_source": args.n_rules_per_source,
            "n_test_seeds": len(args.test_seeds),
            "test_seeds_range": f"{args.test_seeds[0]}..{args.test_seeds[-1]}",
            "grid": list(args.grid),
            "timesteps": args.timesteps,
            "max_candidates": args.max_candidates,
            "hce_replicates": args.hce_replicates,
            "horizons": list(args.horizons),
            "backend": args.backend,
            "n_workers": args.n_workers,
        },
        "results": {
            "wall_time_seconds_sweep": sweep_seconds,
            "wall_time_seconds_total_approx": int(round(elapsed)),
            "n_sweep_cells": (
                args.n_rules_per_source * len(args.test_seeds)
            ),
            "candidates_measured": n_candidates,
        },
    }

    if args.update_baseline:
        update_baselines_file(args.variant, payload)
        print(f"\nbaselines.json updated: production_baselines.{args.variant}")
    else:
        print("\n--update-baseline not passed; baseline NOT written. Payload preview:")
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

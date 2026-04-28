# Test performance policy

This repository has two layers of performance regression protection:

1. **Fast automatic gate** (~10 seconds) — runs in normal `pytest`. Catches
   broken-pipeline regressions: a kernel bug, a stale validator, a CLI
   plumbing fail, an encoding fail in a writer. Cheap enough that every
   developer pays the cost on every test run.

2. **Long-form production baseline** (tens of minutes to ~1 hour) —
   opt-in only. Compares a real production-scale run against a captured
   baseline. Catches *speed* regressions, not just correctness ones.
   Skipped by default; the user explicitly opts in with `--perf-long`.

Both layers read from `tests/perf/baselines.json`, the single source of
truth for thresholds, captured baselines, and tolerance policy.

---

## Layer 1 — fast automatic gate

| Aspect | Value |
|---|---|
| Test file | `tests/perf/test_m8_quick_perf.py` |
| Trigger | `pytest tests/` (any default invocation) |
| Skip when | `cupy` is unavailable (no NVIDIA GPU) |
| Wall-time gate | `fast_perf_gates.m8_quick_cuda_batched.wall_time_seconds_max` (currently 15.0s) |
| Observed | ~8.5s on the captured machine |
| Config | `M8 --quick --backend cuda-batched` (3 rules per source × 2 seeds × T=80 × 16×16×4×4) |

The gate asserts five independent conditions, each of which would
catch a real regression class:

1. Process exits clean (`returncode == 0`).
2. Wall time under `wall_time_seconds_max`.
3. ≥ `min_candidates` (10) actually measured (catches silent
   per-cell errors that pass the wall-time check).
4. ≤ 1 sweep-cell error in stderr (out of 6 cells; allows a single
   rule flake; >1 is a regression).
5. `summary.md` exists, ≥ 200 bytes, contains the section header
   "Mechanism Discovery" (catches encoding bugs in summary writers).

**Refresh command (after a real perf change):**

```bash
# 1. Re-run the gate to see current numbers.
pytest tests/perf/test_m8_quick_perf.py -v

# 2. If a legitimate baseline shift, edit
#    fast_perf_gates.m8_quick_cuda_batched.wall_time_seconds_observed
#    and (if the gate moved) wall_time_seconds_max in baselines.json.
```

---

## Layer 2 — long-form M7B-class production baseline

| Aspect | Value |
|---|---|
| Test file | `tests/perf/test_m7b_reference.py` |
| Trigger | `pytest tests/perf/test_m7b_reference.py --perf-long` |
| Skip when | flag absent, baseline missing, or required backend unavailable |
| Hard-fail mode | add `--perf-gate` to promote tolerance warnings to failures |

### Captured baseline: `m8_m7b_class_numpy`

| Field | Value |
|---|---|
| Captured | 2026-04-27 21:49 UTC |
| Git commit | `251aeba` (tag `perf-refactor-v1`) |
| Wall time (sweep) | **1843 s** |
| Wall time (total approx) | ~1875 s |
| Candidates measured | 1387 |
| Sweep cells | 100 (5 rules × 20 seeds) |
| Backend | `numpy` |
| Grid | 64 × 64 × 8 × 8 |
| Timesteps | 500 |
| `n_workers` | 30 |
| Hardware | Intel i9-14900K · 64 GiB RAM · RTX 3080 Ti · Windows 11 |
| Software | Python 3.12.10 · numba 0.65.1 · cupy 14.0.1 · CUDA 12.9 |

### Why numpy and not cuda-batched?

The captured baseline was an opportunistic real-world run that the user
performed during the refactor; it happened to be on the slow `numpy`
backend (the default). It is therefore the *worst-case* reference —
anything that's slower than this on the same config is a real
regression. The cuda-batched run on the same config is expected to be
much faster; capturing it explicitly is straightforward (see "Refresh
or add a baseline" below) and is recorded in `baselines.json` as a
null slot waiting to be populated.

### Tolerance policy

Defined in `tolerance_policy` of `baselines.json`:

* `warn_regression_fraction = 0.20` — re-runs are allowed to be up to
  20% slower than the baseline before any signal is emitted.
* If a re-run exceeds that, the test issues a `UserWarning` describing
  the delta. The test still passes.
* When invoked with `--perf-gate`, that warning becomes an
  `assert False` and the test fails.

This split means routine `--perf-long` runs surface regressions
without blocking work, while `--perf-gate` provides a strict mode for
a release cut or a CI run that should not let a 20%+ regression
through.

### Refresh or add a baseline

Always run from the repo root with `.venv` active. The capture script
is opt-in and not wired into pytest collection.

```bash
# Re-capture the numpy baseline (~30 min on the captured machine):
python tests/perf/capture_m7b_baseline.py \
    --backend numpy \
    --variant m8_m7b_class_numpy \
    --update-baseline

# Capture a fresh cuda-batched baseline on the same config:
python tests/perf/capture_m7b_baseline.py \
    --backend cuda-batched \
    --variant m8_m7b_class_cuda_batched \
    --update-baseline
```

The script auto-detects CPU, GPU, RAM, OS, Python, package versions,
CUDA runtime, git commit, and git tag. It refuses to update the
baseline if `n_candidates == 0` (catching the silent-failure mode the
fast gate already guards against).

### Inspect the baseline without re-running

```bash
python -c "
import json
from pathlib import Path
data = json.loads(Path('tests/perf/baselines.json').read_text(encoding='utf-8'))
print(json.dumps(data['production_baselines'], indent=2))
"
```

---

## What is automatic vs manual

| Action | Trigger | Frequency | Cost |
|---|---|---|---|
| Fast gate | every `pytest tests/` | every test run | ~10s |
| Long perf check (warn) | `pytest tests/perf/ --perf-long` | manual, before merging perf-sensitive changes | ~baseline duration |
| Long perf check (strict) | `pytest tests/perf/ --perf-long --perf-gate` | manual, release cuts | ~baseline duration |
| Refresh baseline | `python tests/perf/capture_m7b_baseline.py ... --update-baseline` | when intentionally changing perf characteristics | ~baseline duration |

The 30-minute production benchmark **never runs in normal pytest**.
The contract: `pytest tests/` should always finish in seconds-to-tens-of-seconds
on a developer machine.

---

## Hardware sensitivity

The captured baseline encodes specific hardware. Running
`--perf-long --perf-gate` on a different machine (slower CPU, less
RAM, different GPU) will fail not because of a regression but because
of hardware asymmetry. The intended workflow:

1. On the reference machine (the one in `production_baselines.<variant>.machine`):
   re-run with `--perf-gate` to enforce no regression vs. the captured
   number.
2. On a different machine: re-run without `--perf-gate` to surface
   the hardware delta as a warning, not a failure.
3. To establish a new reference: run `capture_m7b_baseline.py` on the
   target machine and commit the updated `baselines.json`. The
   `machine` block records the hardware so future readers can tell
   why the number is what it is.

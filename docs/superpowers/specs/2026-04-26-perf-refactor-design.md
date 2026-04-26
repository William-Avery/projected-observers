# Performance refactor — Windows + 14900k + 3080 Ti

**Author:** William Avery
**Date:** 2026-04-26
**Status:** Draft, awaiting user review

## Goal

Make the `projected-observers` framework substantially faster on a Windows
PC with a 14900k (24 cores / 32 threads) and an NVIDIA RTX 3080 Ti (12 GB
VRAM) by adding (a) process-level parallelism over independent sweep cells
and (b) a CUDA backend for the 4D CA core, including a batched API for the
hot inner loops in M8 mechanism analysis and rule-search fitness
evaluation.

The framework was originally developed on an M1 MacBook Pro. The Mac is no
longer a target; CI/dev on Windows is the only supported platform from
this refactor onward.

## Context

The current codebase has these performance characteristics:

- **CA core**: `observer_worlds/worlds/ca4d.py` provides `update_4d_numba`
  (4-axis nested loops with `prange` on the outermost axis, 80-neighbor
  Moore-r1 sum, periodic BCs via modulo, birth/survival LUTs of size 81)
  and `update_4d_numpy` (scipy `convolve` reference). `CA4D` is a stateful
  wrapper that picks a backend and steps in place.
- **Rollouts**: `metrics/causality_score.py:rollout`,
  `experiments/_m6_hidden_causal.py`, `experiments/_m8_mechanism.py` all
  spin up a fresh `CA4D` per call and step it serially. Each rollout is
  short (10–80 steps on small grids) but called thousands of times.
- **Sweeps**: `_m4b_sweep.py:run_sweep`, `_m6b_replication.py`,
  `run_m7b_production_holdout.py`, `_m8_mechanism.py:run_m8_mechanism_discovery`
  all run a sequential `for rule: for seed: for condition` triple loop.
  No multiprocessing, no joblib, no concurrent.futures anywhere in the
  codebase.
- **GPU**: zero current use. No cupy, jax, or torch dependencies.

This design decisions in this document are settled answers to five
brainstorming questions:

| # | Question | Decision |
|---|---|---|
| 1 | Refactor scope | Core + parallelism + experiment migration |
| 2 | GPU framework | Cupy on Windows; numba CPU stays as fallback. M1 Mac dropped. |
| 3 | Reproducibility | CPU is canonical for published runs; GPU is for exploratory runs and rule-search fitness. No bit-identity required. |
| 4 | Wall-time priorities | Rule search, validation sweeps, mechanism analysis all matter — implies a batched GPU primitive at the bottom of the stack. |
| 5 | Rollout + targets | Incremental, four shippable stages, with quantitative T1 gates. |

## Success criteria (T1 quantitative gates)

These targets fail Stage 4 if not met:

- **M8 moderate config** (3 rules × 3 seeds × T=200 × 32×32×4×4) end-to-end
  in **under 60 seconds**. Currently ~10 minutes.
- **M7B production reference config** (5 rules per source × 50 test seeds
  × T=500 × 32×32×4×4) end-to-end in **under 30 minutes**. Currently a few
  hours.
- **Rule-search throughput** (sims/sec on `evolve_4d_hce_rules.py`
  reference config) **≥20× the current numba CPU baseline**.

Stages 1–3 do not have hard gates; they only need to demonstrably
contribute to meeting Stage 4's gates.

## Architecture

Four layers, bottom to top:

```
┌──────────────────────────────────────────────────────────────┐
│ Stage 4: Experiment migration + perf gates                   │
│   Each experiment script gains --backend {numba,cuda,         │
│   cuda-batched}. Default unchanged. Drivers fall back        │
│   gracefully if cuda unavailable.                            │
│   tests/perf/ benchmarks enforce T1.                         │
├──────────────────────────────────────────────────────────────┤
│ Stage 3: Batched-grid GPU API                                │
│   observer_worlds/worlds/ca4d_batch.py:CA4DBatch             │
│   State (B, Nx, Ny, Nz, Nw) on device, per-batch LUTs        │
│   (B, 81). One kernel launch per step.                       │
│   Migrate M8 response map and evolve_*_rules fitness         │
│   evaluation to use it.                                      │
├──────────────────────────────────────────────────────────────┤
│ Stage 2: Single-grid GPU CA                                  │
│   observer_worlds/worlds/ca4d_cuda.py: cupy RawKernel        │
│   that ports _update_4d_numba_core to CUDA.                  │
│   CA4D backend extends to {"numba","numpy","cuda"}.          │
├──────────────────────────────────────────────────────────────┤
│ Stage 1: CPU process parallelism                             │
│   observer_worlds/parallel/sweep.py: joblib loky over        │
│   (rule × seed × condition). 22 workers on the 14900k        │
│   (cpu_count - 2). NUMBA_NUM_THREADS=1 in workers.           │
└──────────────────────────────────────────────────────────────┘
```

Each stage is independently shippable. Stop-points after Stage 1 / 2 / 3
all leave a faster, working repo.

## Stage 0 — Environment

Discovered during brainstorming: the Windows machine has only Python 3.14.2
installed, which has no numba support yet (numba currently supports up
through 3.13).

Stage 0 deliverable:

- Install Python 3.12 alongside 3.14 via the Python launcher (`py -3.12`).
- Create `.venv` at the repo root using Python 3.12.
- Install the project plus dev extras: `pip install -e ".[dev]"`.
- Install cupy with CUDA 12.x bindings: `pip install cupy-cuda12x`.
- Run `pytest tests/ -v` and confirm all 153 existing tests pass.
- Run a smoke import: `python -c "import cupy; cupy.cuda.runtime.runtimeGetVersion()"`.
- Document the setup in TUTORIAL.md (replace the macOS/Linux assumption
  with a Windows section; keep the existing instructions as a note for
  reproducibility on the original platform).

This stage is not optional and lands first.

## Stage 1 — CPU process parallelism

### Module

`observer_worlds/parallel/sweep.py`:

```python
def parallel_sweep(
    items: Sequence,
    fn: Callable,
    *,
    n_workers: int | None = None,
    backend: str = "loky",
    progress: Callable[[str], None] | None = None,
) -> list:
    """Map fn over items, in parallel, preserving input order.

    Each worker sets NUMBA_NUM_THREADS=1 in its initializer so that
    numba's intra-step prange does not oversubscribe physical cores.
    Default n_workers is max(1, cpu_count - 2).
    """
```

### Migration sites

All sweep drivers currently follow the same `for rule: for seed:
[for condition]` pattern. Migration recipe:

1. Flatten the loops to a list of work items, each item carrying enough
   info to do its task standalone (rule, seed, condition_name, plus any
   shared config).
2. Call `parallel_sweep(items, _do_one_task)`.
3. Regroup the returned flat list back into the original output shape
   (`PairedRecord` for M4B, etc.).

Migration sites:

- `experiments/_m4b_sweep.py:run_sweep` — flatten to `(rule_idx, seed,
  condition)` triples; regroup into `PairedRecord` per (rule_idx, seed).
- `experiments/_m6b_replication.py` outer driver — `(rule, seed,
  intervention, horizon)` → flat tasks.
- `experiments/run_m7b_production_holdout.py` — `(source, rule, seed)` →
  flat tasks.
- `experiments/_m8_mechanism.py:run_m8_mechanism_discovery` — `(rule,
  seed)` → flat tasks (per-seed work stays sequential to keep the inner
  candidate selection deterministic).
- `experiments/_m8b_spatial.py`, `experiments/_m8c_validation.py`,
  `experiments/_m8d_decomposition.py` — same `(rule, seed)` flattening
  pattern as M8.

Stage 1 only adds the parallel driver and changes the *outer loop*. The
inner per-task work (one sim + tracking + metrics) stays on the existing
numba CPU backend in this stage; GPU acceleration arrives in Stage 4.

### Determinism

Worker RNG must derive deterministically from the work item. Existing
code already uses `seeded_rng(seed)` derivations; we keep that contract.
No worker-shared mutable state.

### Failure modes & mitigations

- Pickling failures on `FractionalRule` or `RunConfig`: it's all
  dataclasses with primitive fields, but verify with a smoke test as the
  first task in this stage.
- Loky workers crash silently on Windows: joblib reports the exception
  with `verbose=10`; we surface it via the `progress` callback.
- Memory blow-up if all workers hold a 4D snapshot dict at once: cap
  workers at `min(cpu_count-2, 32)` and document the per-task memory
  budget in the docstring.

### Tests

- `tests/parallel/test_sweep_parity.py` — 4 rules × 3 seeds × 3
  conditions sweep on tiny grid: parallel and serial outputs must match
  field-for-field across all `PairedRecord`s for fixed seeds.
- `tests/parallel/test_oversubscription.py` — confirm
  `os.environ["NUMBA_NUM_THREADS"]` is "1" inside a worker.

## Stage 2 — Single-grid GPU CA

### Module

`observer_worlds/worlds/ca4d_cuda.py`:

```python
_KERNEL = cupy.RawKernel(r"""
extern "C" __global__
void update_4d(const unsigned char* __restrict__ in,
               unsigned char* __restrict__ out,
               const bool* __restrict__ birth_lut,
               const bool* __restrict__ surv_lut,
               int Nx, int Ny, int Nz, int Nw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx*Ny*Nz*Nw;
    if (idx >= total) return;
    // unflatten idx -> (x, y, z, w)
    // 80-neighbor sum with modulo periodic BCs
    // out[idx] = in[idx] ? surv_lut[count] : birth_lut[count]
}
""", "update_4d")

def update_4d_cuda(state: cupy.ndarray, rule: BSRule) -> cupy.ndarray:
    """One CA step on device. Allocates `out`, launches kernel, returns."""
```

### CA4D integration

`CA4D` accepts `backend="cuda"`. State storage:

- `numba` / `numpy` backends: state is `np.ndarray` (unchanged).
- `cuda` backend: state is `cupy.ndarray` on device. The `state` getter
  returns a host numpy copy (only called at snapshot / projection-out
  boundaries). The setter accepts numpy or cupy and coerces to device.

### Why cupy RawKernel (not numba.cuda or ElementwiseKernel)

- Lower kernel-launch overhead than numba.cuda (no Python-side dispatch
  on each call).
- Easier than `ElementwiseKernel` for a stencil with 80 neighbors.
- Single-file, ~50 lines of CUDA C, easy to read and maintain.

### Parity validation

`tests/cuda/test_ca4d_parity.py`:

- Fixed seed, 16×16×4×4 grid, 50 steps under both numba and cuda
  backends. Compare per-step `mean(active)`: must match within 1%.
- M6 hidden-invisible projection-preservation invariant test must pass
  on the cuda backend.

Auto-skip via `pytest.mark.cuda` when `cupy.cuda.is_available()` is False.

## Stage 3 — Batched-grid GPU API

### Module

`observer_worlds/worlds/ca4d_batch.py`:

```python
class CA4DBatch:
    """K independent 4D CAs on device, stepped together.

    State shape: (B, Nx, Ny, Nz, Nw), uint8.
    LUTs shape: (B, 81), bool.

    All grids must share the same spatial shape (Nx, Ny, Nz, Nw).
    Rules can differ per batch element.
    """

    def __init__(self, shape, rules: list[BSRule], seeds: list[int],
                 initial_density: list[float]) -> None: ...

    def step(self) -> None:
        """Advance all B grids one timestep with one kernel launch."""

    @property
    def state(self) -> cupy.ndarray:
        """Device-resident (B, Nx, Ny, Nz, Nw)."""

    def state_at(self, b: int) -> np.ndarray:
        """Host copy of one batch element."""
```

### Killer migration sites

1. **M8 response map** (`_m8_mechanism.py:compute_response_map`).
   Currently iterates `for x, y in coords: for rep in range(n_replicates):
   rollout(...)`. For a 32×32 candidate with ~200 interior cells and 3
   replicates × horizon 10, that's 6000 sequential rollouts.
   Reframe: build a `CA4DBatch` of B = (n_columns × n_replicates)
   identical-rule grids, each with the per-column shuffle applied to its
   own copy of the snapshot. Step them together for `horizon` steps.

2. **Rule-search fitness** (`evolve_4d_hce_rules.py`,
   `run_search_observer_rules.py`). A generation evaluates K rules × M
   seeds → K·M independent grids. Batch them.

### VRAM budget

| Grid shape | Per-grid bytes | Max B at 8 GB usable |
|---|---|---|
| 32×32×4×4   | 4 KB        | ~2,000,000 |
| 64×64×8×8   | 256 KB      | ~32,000 |
| 128×128×16×16 | 4 MB      | ~2,000 |

Practical caps are far lower (kernel launch overhead, intermediate
buffers, snapshot retention). Initial cap: B ≤ 2048 for small grids,
B ≤ 128 for M8C-scale. Auto-shrink on `cupy.cuda.OutOfMemoryError`.

### Tests

- `tests/cuda/test_batch_parity.py` — K=8 batched evolution must produce
  the same per-grid trajectory as K independent single-grid `CA4D` runs
  with matching rules and seeds.
- `tests/cuda/test_batch_oom_recovery.py` — request a batch known to
  OOM, verify auto-shrink fallback completes.

## Stage 4 — Experiment migration + perf gates

### CLI changes

Each experiment script gains `--backend {numba,cuda,cuda-batched}`.
Default remains `numba` (canonical CPU path). When `cuda-batched` is
selected and the experiment has an inner loop that can be batched, it
uses `CA4DBatch`; otherwise it falls back to single-grid `cuda`.

Drivers detect cuda availability at startup; if `--backend cuda*` was
requested but cuda is unavailable, abort with a clear error rather than
silently falling back.

### Migration order within Stage 4

1. `experiments/_m4b_sweep.py` — `--backend cuda` for the per-condition
   simulation; sweep already process-parallel from Stage 1.
2. `experiments/run_m6b_hidden_causal_replication.py` — same.
3. `experiments/run_m7b_production_holdout.py` — same.
4. `experiments/run_m8_mechanism_discovery.py` — `--backend cuda-batched`
   wires `compute_response_map` to `CA4DBatch`.
5. `experiments/evolve_4d_hce_rules.py` and `run_search_observer_rules.py`
   — fitness evaluation uses `CA4DBatch`.
6. M8B / M8C / M8D follow the same pattern as M8.

### Perf gates

`tests/perf/` with pytest-benchmark fixtures:

- `test_m8_moderate_under_60s` — runs M8 moderate config end-to-end,
  fails if wall time >60s.
- `test_m7b_reference_under_30min` — gated behind `--perf-long` mark
  so it doesn't run in the default test suite.
- `test_rule_search_throughput_ge_20x` — measures sims/sec vs a
  frozen numba baseline checked into the repo.

Baselines for comparison (Stage 0 deliverable): record current numba
wall times for these three configs into `tests/perf/baselines.json`.

## Testing strategy

| Layer | Test | Where |
|---|---|---|
| Existing | 153 numba CPU tests | `tests/` (unchanged) |
| Stage 1 | Parallel sweep parity | `tests/parallel/` |
| Stage 2 | CPU↔CUDA parity (single grid) | `tests/cuda/` |
| Stage 2 | M6 hidden-invisible invariant on CUDA | `tests/cuda/` |
| Stage 3 | Batched↔serial parity | `tests/cuda/` |
| Stage 3 | OOM auto-shrink | `tests/cuda/` |
| Stage 4 | T1 perf gates | `tests/perf/` |

CUDA-conditional tests use `pytest.mark.cuda` and auto-skip when
`cupy.cuda.is_available()` returns False.

Existing tests stay green throughout. The CPU canonical path is the
falsifier — any GPU bug that causes drift surfaces in parity tests
before it pollutes published runs.

## Risks & mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| cupy has no Python 3.14 wheel yet | High | Blocks Stage 0 | Pin venv to Python 3.12 |
| cupy RawKernel parity drift hides bugs | Medium | Silent corruption of GPU runs | CPU stays canonical; parity tests guard regressions |
| VRAM exhaustion on large batches | Medium | Crashes M8C-scale runs | Runtime cap + auto-chunk on OOM |
| joblib pickling fails on Windows for some object | Medium | Stage 1 blocks | Smoke test as first deliverable in Stage 1 |
| Process workers interfere with numba JIT cache | Low | First-run slowness | Numba cache=True is process-shared; pre-warm in worker init |
| Test suite slows past acceptable bound | Low | Dev friction | Keep CPU + CUDA + perf suites distinct via pytest marks |

## Out of scope

- M1 / macOS support beyond what already works in CPU-only mode.
- AMD GPUs / ROCm.
- Distributed / multi-node parallelism.
- Replacing zarr storage. The M4B sweep already keeps frames in memory;
  zarr stays as the durable archive format for `run_4d_projection`.
- Refactoring the metric code itself (M2 suite). Out of scope unless a
  metric becomes the bottleneck after Stage 3.
- A new fitness function or experiment design. This is purely a
  performance refactor; semantic behavior is unchanged.

#!/usr/bin/env bash
# scripts/g4_linux_bootstrap.sh
#
# Stage G4-Linux one-shot bootstrap. Intended to run on a fresh
# Ubuntu/Debian GPU host with NVIDIA drivers + CUDA-compatible card.
#
# Drives the full G4 experiment end-to-end:
#   1. Environment + GPU verification
#   2. venv + dependency setup
#   3. pytest (full + GPU-equivalence subset)
#   4. n_workers=30 discovery stress smoke (GATE — stops on failure)
#   5. Full Stage 6C-equivalent GPU run
#   6. Row-by-row + posthoc equivalence vs the Stage 6C CPU baseline
#      (only if --stage6c-baseline-dir is supplied)
#   7. Markdown report under outputs/g4_linux_report_<timestamp>.md
#
# All output (stdout + stderr) is teed to outputs/g4_linux_bootstrap_<ts>.log.
# Idempotent; safe to re-run after a partial failure (skips already-done
# steps via flag files under .g4_state/).
#
# Exit codes:
#   0  full success (smoke + full + equivalence all pass)
#   1  generic / setup error
#   2  smoke gate failed; full run not started
#   3  full run failed
#   4  equivalence comparison failed (row-by-row or posthoc verdict)
#
# Usage:
#   chmod +x scripts/g4_linux_bootstrap.sh
#   ./scripts/g4_linux_bootstrap.sh \
#       [--stage6c-baseline-dir /path/to/stage6c_baseline_run/] \
#       [--n-workers 30] [--gpu-batch-size 64] \
#       [--skip-smoke] [--skip-full] [--skip-tests] \
#       [--smoke-only] [--help]
#
# Do not run from Windows.

set -euo pipefail

# ---- Pretty logging ---------------------------------------------------------

if [[ -t 1 ]]; then
    _BOLD=$(printf '\033[1m'); _DIM=$(printf '\033[2m')
    _RED=$(printf '\033[31m'); _GRN=$(printf '\033[32m')
    _YLW=$(printf '\033[33m'); _RST=$(printf '\033[0m')
else
    _BOLD=""; _DIM=""; _RED=""; _GRN=""; _YLW=""; _RST=""
fi

_now()  { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
_ts()   { date -u +"%Y%m%dT%H%M%SZ"; }
_say()  { printf '%s[%s]%s %s\n' "$_BOLD" "$(_now)" "$_RST" "$*"; }
_warn() { printf '%s[%s WARN]%s %s\n' "$_YLW" "$(_now)" "$_RST" "$*" >&2; }
_err()  { printf '%s[%s ERR]%s %s\n'  "$_RED" "$(_now)" "$_RST" "$*" >&2; }
_ok()   { printf '%s[%s OK]%s %s\n'   "$_GRN" "$(_now)" "$_RST" "$*"; }

usage() {
    sed -n '1,60p' "$0" | sed 's/^# \{0,1\}//'
}

# ---- Arg parsing ------------------------------------------------------------

STAGE6C_BASELINE=""
N_WORKERS=30
GPU_BATCH_SIZE=64
GPU_MEM_TARGET_GB=9.5
SKIP_SMOKE=0
SKIP_FULL=0
SKIP_TESTS=0
SMOKE_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage6c-baseline-dir) STAGE6C_BASELINE="${2:-}"; shift 2 ;;
        --n-workers) N_WORKERS="${2:-}"; shift 2 ;;
        --gpu-batch-size) GPU_BATCH_SIZE="${2:-}"; shift 2 ;;
        --gpu-memory-target-gb) GPU_MEM_TARGET_GB="${2:-}"; shift 2 ;;
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        --skip-full)  SKIP_FULL=1;  shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --smoke-only) SMOKE_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) _err "unknown argument: $1"; usage; exit 1 ;;
    esac
done

# ---- Repo root --------------------------------------------------------------

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"
cd "$REPO_ROOT"

if [[ ! -f "pyproject.toml" ]] || [[ ! -d "observer_worlds" ]]; then
    _err "not in repo root (no pyproject.toml / observer_worlds/); cwd=$REPO_ROOT"
    exit 1
fi

# ---- OS sanity --------------------------------------------------------------

OS_KERNEL="$(uname -srm 2>/dev/null || true)"
if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
    _err "this script is for Linux. uname -s = $(uname -s 2>/dev/null || echo unknown)"
    _err "see docs/G4_LINUX_GPU_RUNBOOK.md and run on a Linux GPU host."
    exit 1
fi

# ---- Output / log paths -----------------------------------------------------

RUN_TS="$(_ts)"
mkdir -p outputs
LOG_FILE="outputs/g4_linux_bootstrap_${RUN_TS}.log"
REPORT_FILE="outputs/g4_linux_report_${RUN_TS}.md"
STATE_DIR=".g4_state"
mkdir -p "$STATE_DIR"

# Tee everything to the log. Background subshell pattern keeps stdin
# usable for sub-processes that might be interactive.
exec > >(tee -a "$LOG_FILE") 2>&1

_say "Stage G4-Linux bootstrap"
_say "log:    $LOG_FILE"
_say "report: $REPORT_FILE"
_say "n_workers=$N_WORKERS gpu_batch_size=$GPU_BATCH_SIZE gpu_mem_target_gb=$GPU_MEM_TARGET_GB"
if [[ -n "$STAGE6C_BASELINE" ]]; then
    _say "stage6c-baseline: $STAGE6C_BASELINE"
else
    _warn "no --stage6c-baseline-dir; equivalence comparison will be SKIPPED."
fi

# ---- 1. Environment + GPU verification --------------------------------------

_say "==== 1. Environment verification ===="

PY_BIN="${PYTHON:-python3.12}"
if ! command -v "$PY_BIN" &>/dev/null; then
    _warn "$PY_BIN not found; falling back to python3"
    PY_BIN="python3"
fi
PY_VERSION="$("$PY_BIN" --version 2>&1 || true)"

if ! command -v git &>/dev/null; then
    _err "git not found"
    exit 1
fi
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
GIT_COMMIT="$(git rev-parse --short HEAD)"
GIT_DIRTY="$(git status --porcelain | head -1)"

TAG_PRESENT="no"
if git tag --list 'stage6-replication-complete' | grep -q stage6-replication-complete; then
    TAG_PRESENT="yes"
fi

NVIDIA_OUT=""
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_OUT="$(nvidia-smi 2>&1 || true)"
else
    _err "nvidia-smi not found. Install the NVIDIA driver before re-running."
    NVIDIA_OUT="(nvidia-smi missing)"
    exit 1
fi

_say "OS kernel: $OS_KERNEL"
_say "Python: $PY_VERSION"
_say "git: branch=$GIT_BRANCH commit=$GIT_COMMIT dirty=${GIT_DIRTY:+yes}"
_say "tag stage6-replication-complete present: $TAG_PRESENT"
_say "nvidia-smi:"
echo "$NVIDIA_OUT" | head -25 | sed 's/^/  /'

# ---- 2. venv + deps ---------------------------------------------------------

_say "==== 2. venv + dependencies ===="

if [[ ! -d ".venv" ]]; then
    _say "creating venv at .venv with $PY_BIN"
    "$PY_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ ! -f "$STATE_DIR/pip_done" ]]; then
    _say "pip install --upgrade pip"
    pip install --upgrade pip >/dev/null
    _say 'pip install -e ".[dev,gpu]"'
    pip install -e ".[dev,gpu]" >/dev/null
    touch "$STATE_DIR/pip_done"
else
    _ok "pip install already done; skipping (rm $STATE_DIR/pip_done to redo)"
fi

# ---- CuPy GPU sanity check --------------------------------------------------

_say "verifying CuPy + GPU..."
CUPY_INFO_FILE="outputs/g4_linux_cupy_info_${RUN_TS}.txt"
if ! python - <<'PY' > "$CUPY_INFO_FILE" 2>&1
import cupy as cp
print("cupy_version", cp.__version__)
nd = cp.cuda.runtime.getDeviceCount()
print("device_count", nd)
if nd <= 0:
    raise SystemExit("no CUDA device visible to cupy")
props = cp.cuda.runtime.getDeviceProperties(0)
print("gpu_name", props["name"].decode("utf-8", errors="replace"))
free, total = cp.cuda.Device(0).mem_info
print(f"gpu_total_mb {total/1024**2:.0f}")
print(f"gpu_free_mb  {free/1024**2:.0f}")
print("cuda_runtime_version", cp.cuda.runtime.runtimeGetVersion())
print("driver_version", cp.cuda.runtime.driverGetVersion())
PY
then
    _err "CuPy could not see the GPU. Output:"
    cat "$CUPY_INFO_FILE" | sed 's/^/  /'
    _err "Aborting; fix the CUDA / cupy install before re-running."
    exit 1
fi
sed 's/^/  /' "$CUPY_INFO_FILE"
GPU_NAME="$(grep '^gpu_name ' "$CUPY_INFO_FILE" | cut -d' ' -f2- || echo 'unknown')"
GPU_TOTAL_MB="$(grep '^gpu_total_mb ' "$CUPY_INFO_FILE" | awk '{print $2}' || echo '0')"
CUPY_VERSION="$(grep '^cupy_version ' "$CUPY_INFO_FILE" | awk '{print $2}' || echo 'unknown')"
CUDA_RT="$(grep '^cuda_runtime_version ' "$CUPY_INFO_FILE" | awk '{print $2}' || echo 'unknown')"
NV_DRIVER="$(grep '^driver_version ' "$CUPY_INFO_FILE" | awk '{print $2}' || echo 'unknown')"
_ok "CuPy $CUPY_VERSION sees $GPU_NAME (${GPU_TOTAL_MB} MB total). CUDA rt=$CUDA_RT driver=$NV_DRIVER."

# ---- 3. Tests ---------------------------------------------------------------

TESTS_GPU_RESULT="skipped"
TESTS_FULL_RESULT="skipped"
if [[ "$SKIP_TESTS" -eq 1 ]]; then
    _warn "--skip-tests passed; tests not run."
else
    _say "==== 3. Tests ===="
    PYTHONIOENCODING=utf-8 python -m pytest \
        tests/test_gpu_backend.py tests/test_projection_robustness_gpu.py -q \
        | tee "outputs/g4_linux_pytest_gpu_${RUN_TS}.log"
    TESTS_GPU_RESULT="$(tail -3 "outputs/g4_linux_pytest_gpu_${RUN_TS}.log" \
        | grep -E 'passed|failed' | tail -1 | tr -s ' ')"
    PYTHONIOENCODING=utf-8 python -m pytest tests/ -q \
        | tee "outputs/g4_linux_pytest_full_${RUN_TS}.log"
    TESTS_FULL_RESULT="$(tail -3 "outputs/g4_linux_pytest_full_${RUN_TS}.log" \
        | grep -E 'passed|failed' | tail -1 | tr -s ' ')"
    _ok "tests gpu-subset: $TESTS_GPU_RESULT"
    _ok "tests full:        $TESTS_FULL_RESULT"
fi

# ---- 4. n_workers=30 stress smoke (GATE) ------------------------------------

SMOKE_DIR=""
SMOKE_PASS="skipped"
SMOKE_WALL_S=""
SMOKE_DISCOVERY_WALL_S=""
SMOKE_GPU_ROLLOUT_WALL_S=""
SMOKE_POOL_RESTARTS=""
SMOKE_N_CANDIDATES=""

run_smoke() {
    _say "==== 4. n_workers=$N_WORKERS Linux discovery stress smoke (GATE) ===="
    PYTHONIOENCODING=utf-8 PYTHONFAULTHANDLER=1 python -m \
        observer_worlds.experiments.run_followup_projection_robustness_gpu \
        --rules-from release/rules/m7_top_hce_rules.json \
        --m4c-rules release/rules/m4c_evolve_leaderboard.json \
        --m4a-rules release/rules/m4a_search_leaderboard.json \
        --n-rules-per-source 1 \
        --seeds 7000..7002 \
        --timesteps 500 \
        --grid 64 64 8 8 \
        --max-candidates 20 \
        --hce-replicates 3 \
        --horizons 1 2 3 5 10 20 40 80 \
        --projections mean_threshold sum_threshold max_projection \
                      parity_projection random_linear_projection \
                      multi_channel_projection \
        --backend cupy \
        --n-workers "$N_WORKERS" \
        --gpu-batch-size "$GPU_BATCH_SIZE" \
        --gpu-memory-target-gb "$GPU_MEM_TARGET_GB" \
        --label g4_linux_discovery_stress_smoke \
        --profile
    SMOKE_DIR="$(ls -td outputs/g4_linux_discovery_stress_smoke_*/ 2>/dev/null | head -1)"
    if [[ -z "$SMOKE_DIR" ]] || [[ ! -d "$SMOKE_DIR" ]]; then
        _err "smoke produced no output dir."
        SMOKE_PASS="failed"
        return 1
    fi
    _say "smoke output: $SMOKE_DIR"

    # Parse stats_summary.json for the G3 perf block.
    if [[ -f "$SMOKE_DIR/stats_summary.json" ]]; then
        SMOKE_WALL_S="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$SMOKE_DIR/stats_summary.json').read_text())
print(s.get('g3_perf', {}).get('total_wall_s', ''))
" 2>/dev/null || true)"
        SMOKE_DISCOVERY_WALL_S="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$SMOKE_DIR/stats_summary.json').read_text())
print(s.get('g3_perf', {}).get('cpu_discovery_wall_s', ''))
" 2>/dev/null || true)"
        SMOKE_GPU_ROLLOUT_WALL_S="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$SMOKE_DIR/stats_summary.json').read_text())
print(s.get('g3_perf', {}).get('gpu_rollout_wall_s', ''))
" 2>/dev/null || true)"
        SMOKE_N_CANDIDATES="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$SMOKE_DIR/stats_summary.json').read_text())
print(s.get('n_candidate_rows', ''))
" 2>/dev/null || true)"
    fi
    # Pool restarts: scan stdout / stats_summary if available.
    SMOKE_POOL_RESTARTS="$(grep -c '\[discovery\] pool died' "$LOG_FILE" 2>/dev/null || true)"

    _say "smoke metrics: wall=${SMOKE_WALL_S}s disc=${SMOKE_DISCOVERY_WALL_S}s gpu=${SMOKE_GPU_ROLLOUT_WALL_S}s candidates=${SMOKE_N_CANDIDATES} pool_restarts=${SMOKE_POOL_RESTARTS}"

    # Gate: any pool restart fails the gate.
    if [[ "${SMOKE_POOL_RESTARTS:-0}" -gt 0 ]]; then
        _err "smoke gate FAILED: ${SMOKE_POOL_RESTARTS} pool restart(s) detected."
        SMOKE_PASS="failed"
        return 1
    fi
    SMOKE_PASS="passed"
    _ok "smoke gate PASSED."
    return 0
}

if [[ "$SKIP_SMOKE" -eq 1 ]]; then
    _warn "--skip-smoke passed; skipping smoke."
else
    if ! run_smoke; then
        _err "smoke gate failed; not running full Stage 6C-equivalent run."
        # Write report and exit 2 (gate failure path).
        FULL_PASS="not_run"
        EQUIV_PASS="not_run"
        FULL_DIR=""
    fi
fi

# ---- 5. Full Stage 6C-equivalent run ---------------------------------------

FULL_DIR="${FULL_DIR-}"
FULL_PASS="${FULL_PASS-skipped}"
FULL_WALL_S=""
FULL_DISCOVERY_WALL_S=""
FULL_GPU_ROLLOUT_WALL_S=""
FULL_GPU_COMPUTE_S=""
FULL_GPU_MEM_PEAK_MB=""
FULL_N_CANDIDATES=""
FULL_POOL_RESTARTS=""

run_full() {
    _say "==== 5. Full Stage 6C-equivalent Linux GPU run ===="
    PYTHONIOENCODING=utf-8 PYTHONFAULTHANDLER=1 python -m \
        observer_worlds.experiments.run_followup_projection_robustness_gpu \
        --rules-from release/rules/m7_top_hce_rules.json \
        --m4c-rules release/rules/m4c_evolve_leaderboard.json \
        --m4a-rules release/rules/m4a_search_leaderboard.json \
        --n-rules-per-source 5 \
        --seeds 7000..7019 \
        --timesteps 500 \
        --grid 64 64 8 8 \
        --max-candidates 20 \
        --hce-replicates 3 \
        --horizons 1 2 3 5 10 20 40 80 \
        --projections mean_threshold sum_threshold max_projection \
                      parity_projection random_linear_projection \
                      multi_channel_projection \
        --backend cupy \
        --n-workers "$N_WORKERS" \
        --gpu-batch-size "$GPU_BATCH_SIZE" \
        --gpu-memory-target-gb "$GPU_MEM_TARGET_GB" \
        --label g4_linux_projection_robustness_gpu_stage6c_equivalent \
        --profile
    FULL_DIR="$(ls -td outputs/g4_linux_projection_robustness_gpu_stage6c_equivalent_*/ \
                | head -1)"
    if [[ -z "$FULL_DIR" ]] || [[ ! -d "$FULL_DIR" ]]; then
        _err "full run produced no output dir."
        FULL_PASS="failed"
        return 1
    fi
    if [[ -f "$FULL_DIR/stats_summary.json" ]]; then
        FULL_WALL_S="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$FULL_DIR/stats_summary.json').read_text())
print(s.get('g3_perf', {}).get('total_wall_s', ''))
" 2>/dev/null || true)"
        FULL_DISCOVERY_WALL_S="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$FULL_DIR/stats_summary.json').read_text())
print(s.get('g3_perf', {}).get('cpu_discovery_wall_s', ''))
" 2>/dev/null || true)"
        FULL_GPU_ROLLOUT_WALL_S="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$FULL_DIR/stats_summary.json').read_text())
print(s.get('g3_perf', {}).get('gpu_rollout_wall_s', ''))
" 2>/dev/null || true)"
        FULL_GPU_COMPUTE_S="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$FULL_DIR/stats_summary.json').read_text())
print(s.get('g3_perf', {}).get('gpu_compute_s', ''))
" 2>/dev/null || true)"
        FULL_GPU_MEM_PEAK_MB="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$FULL_DIR/stats_summary.json').read_text())
print(s.get('g3_perf', {}).get('gpu_memory_peak_mb', ''))
" 2>/dev/null || true)"
        FULL_N_CANDIDATES="$(python -c "
import json, pathlib
s = json.loads(pathlib.Path('$FULL_DIR/stats_summary.json').read_text())
print(s.get('n_candidate_rows', ''))
" 2>/dev/null || true)"
    fi
    FULL_POOL_RESTARTS="$(grep -c '\[discovery\] pool died' "$LOG_FILE" 2>/dev/null || true)"
    # subtract smoke restarts (which were 0 if we got past the gate)
    FULL_POOL_RESTARTS=$(( FULL_POOL_RESTARTS - ${SMOKE_POOL_RESTARTS:-0} ))
    if [[ "$FULL_POOL_RESTARTS" -lt 0 ]]; then FULL_POOL_RESTARTS=0; fi
    FULL_PASS="passed"
    _ok "full run done. wall=${FULL_WALL_S}s disc=${FULL_DISCOVERY_WALL_S}s gpu=${FULL_GPU_ROLLOUT_WALL_S}s n=${FULL_N_CANDIDATES} pool_restarts=${FULL_POOL_RESTARTS}"
    return 0
}

if [[ "$SKIP_FULL" -eq 1 ]] || [[ "$SMOKE_ONLY" -eq 1 ]] \
   || [[ "$SMOKE_PASS" == "failed" ]]; then
    _warn "skipping full run (skip-full=$SKIP_FULL smoke-only=$SMOKE_ONLY smoke=$SMOKE_PASS)."
else
    if ! run_full; then
        _err "full run failed; will still write report."
    fi
fi

# ---- 6. Equivalence comparison ---------------------------------------------

EQUIV_PASS="${EQUIV_PASS-skipped}"
EQUIV_REPORT_JSON="outputs/g4_linux_equivalence_${RUN_TS}.json"
EQUIV_REPORT_MD="outputs/g4_linux_equivalence_${RUN_TS}.md"
EQUIV_SUMMARY=""

if [[ -n "$STAGE6C_BASELINE" ]] && [[ -d "$FULL_DIR" ]]; then
    _say "==== 6. Equivalence vs Stage 6C CPU baseline ===="
    if [[ ! -d "$STAGE6C_BASELINE" ]]; then
        _err "--stage6c-baseline-dir not a directory: $STAGE6C_BASELINE"
        EQUIV_PASS="failed"
    else
        set +e
        python scripts/g4_compare_to_stage6c.py \
            --g4-run-dir "$FULL_DIR" \
            --stage6c-baseline-dir "$STAGE6C_BASELINE" \
            --out-json "$EQUIV_REPORT_JSON" \
            --out-md "$EQUIV_REPORT_MD"
        EQUIV_RC=$?
        set -e
        case "$EQUIV_RC" in
            0) EQUIV_PASS="passed"; _ok "equivalence PASS." ;;
            2) EQUIV_PASS="failed_row_by_row";
               _err "equivalence FAIL: row-by-row mismatch" ;;
            3) EQUIV_PASS="failed_posthoc";
               _err "equivalence FAIL: posthoc verdict differs from baseline" ;;
            *) EQUIV_PASS="failed_other";
               _err "equivalence comparator returned rc=$EQUIV_RC" ;;
        esac
        if [[ -f "$EQUIV_REPORT_MD" ]]; then
            EQUIV_SUMMARY="$(head -25 "$EQUIV_REPORT_MD" | sed 's/^/    /')"
        fi
    fi
elif [[ -z "$STAGE6C_BASELINE" ]]; then
    _warn "equivalence not run: --stage6c-baseline-dir not provided."
    EQUIV_PASS="skipped_no_baseline"
fi

# ---- 7. Final report --------------------------------------------------------

_say "==== 7. Writing final report -> $REPORT_FILE ===="

# Constants for headline comparison.
WIN_G3_WALL_S=23870       # G3 Windows total wall (s)
S6C_WALL_S=23318          # Stage 6C CPU baseline (s)

speedup() {
    local base="$1"; local cur="$2"
    if [[ -z "$cur" ]] || [[ "$cur" == "0" ]] || [[ ! "$cur" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "—"
    else
        python -c "print(f'{$base / $cur:.2f}x')"
    fi
}

cat > "$REPORT_FILE" <<EOF
# Stage G4-Linux report

Generated: $(_now)

## System

* OS / kernel:                $OS_KERNEL
* Python:                     $PY_VERSION
* GPU:                        $GPU_NAME
* GPU total mem (MB):         $GPU_TOTAL_MB
* CuPy version:               $CUPY_VERSION
* CUDA runtime (cupy):        $CUDA_RT
* NVIDIA driver (cupy):       $NV_DRIVER
* git branch:                 $GIT_BRANCH
* git commit:                 $GIT_COMMIT
* git dirty:                  ${GIT_DIRTY:+yes}
* tag stage6-replication-complete present: $TAG_PRESENT

## Tests

* GPU subset:                 $TESTS_GPU_RESULT
* Full suite:                 $TESTS_FULL_RESULT

## Stress smoke (n_workers=$N_WORKERS, GATE)

* result:                     $SMOKE_PASS
* output dir:                 ${SMOKE_DIR:-(none)}
* wall (s):                   $SMOKE_WALL_S
* cpu_discovery_wall_s:       $SMOKE_DISCOVERY_WALL_S
* gpu_rollout_wall_s:         $SMOKE_GPU_ROLLOUT_WALL_S
* candidate rows:             $SMOKE_N_CANDIDATES
* pool restarts:              ${SMOKE_POOL_RESTARTS:-0}

## Full Stage 6C-equivalent run

* result:                     $FULL_PASS
* output dir:                 ${FULL_DIR:-(none)}
* total_wall_s:               $FULL_WALL_S
* cpu_discovery_wall_s:       $FULL_DISCOVERY_WALL_S
* gpu_rollout_wall_s:         $FULL_GPU_ROLLOUT_WALL_S
* gpu_compute_s:              $FULL_GPU_COMPUTE_S
* gpu_memory_peak_mb:         $FULL_GPU_MEM_PEAK_MB
* candidate rows:             $FULL_N_CANDIDATES
* pool restarts (full only):  ${FULL_POOL_RESTARTS:-0}
* speedup vs Windows G3 (6.63h / ${WIN_G3_WALL_S}s):  $(speedup $WIN_G3_WALL_S "${FULL_WALL_S:-0}")
* speedup vs Stage 6C CPU (6.48h / ${S6C_WALL_S}s):   $(speedup $S6C_WALL_S  "${FULL_WALL_S:-0}")

## Equivalence vs Stage 6C CPU baseline

* result:                     $EQUIV_PASS
* json:                       ${EQUIV_REPORT_JSON}
* markdown:                   ${EQUIV_REPORT_MD}
* note:                       \
$( [[ -z "$STAGE6C_BASELINE" ]] \
   && echo "equivalence not run: baseline dir missing — pass --stage6c-baseline-dir to enable" \
   || echo "compared against ${STAGE6C_BASELINE}" )

EOF

if [[ -n "$EQUIV_SUMMARY" ]]; then
    cat >> "$REPORT_FILE" <<EOF
### Equivalence summary (head of generated md)

\`\`\`
$EQUIV_SUMMARY
\`\`\`

EOF
fi

# ---- Recommendation ---------------------------------------------------------

REC=""
DECISION_RC=0
if [[ "$SMOKE_PASS" == "failed" ]]; then
    REC="Linux smoke crashed at n_workers=$N_WORKERS — Linux is NOT free of the worker-count ceiling on this host. The CPU discovery bottleneck is not OS-specific; recommend planning the **G5 GPU-discovery refactor** (port per-frame projection / tracker hot loops to GPU). Re-running this script with a smaller --n-workers may still succeed but won't change the recommendation."
    DECISION_RC=2
elif [[ "$FULL_PASS" == "passed" ]] && [[ -n "$FULL_WALL_S" ]]; then
    # Numeric thresholds: < 3h preferred, < 4h acceptable, otherwise modest.
    THREE_H=10800; FOUR_H=14400
    if python -c "import sys; sys.exit(0 if float('$FULL_WALL_S') < $THREE_H else 1)" 2>/dev/null; then
        REC="Linux full run completed in <3h with n_workers=$N_WORKERS. Recommend **adopting Linux as the GPU production target**. The G5 GPU-discovery refactor can be shelved for now; revisit only if discovery becomes a bottleneck again under larger workloads."
    elif python -c "import sys; sys.exit(0 if float('$FULL_WALL_S') < $FOUR_H else 1)" 2>/dev/null; then
        REC="Linux full run completed in 3–4h with n_workers=$N_WORKERS. **Acceptable but modest** speedup vs Stage 6C CPU baseline. Document this as the safe Linux ceiling; G5 GPU-discovery refactor remains worth scoping for the next workload size increase."
    else
        REC="Linux full run completed but wall >4h. **Linux removes the Windows worker crashes but does not deliver the expected speedup**. The CPU discovery is genuinely heavy on this host. Plan **G5 GPU-discovery refactor** as the next step."
    fi
    if [[ "$EQUIV_PASS" != "passed" ]] && [[ "$EQUIV_PASS" != "skipped_no_baseline" ]]; then
        REC="$REC\n\n**Caveat:** equivalence vs Stage 6C CPU baseline did NOT pass ($EQUIV_PASS). Do not promote any G4 number to a scientific claim until equivalence is resolved."
        DECISION_RC=4
    fi
elif [[ "$FULL_PASS" == "failed" ]]; then
    REC="Full Linux run did not complete successfully. See ${FULL_DIR:-(no dir)} and the log $LOG_FILE for details."
    DECISION_RC=3
else
    REC="Full Linux run was not executed (smoke-only or skipped). Re-run without --smoke-only/--skip-full to get the production timing."
fi

cat >> "$REPORT_FILE" <<EOF

## Recommendation

$REC
EOF

_say "report written: $REPORT_FILE"
echo
echo "------------------------- Recommendation -------------------------"
echo -e "$REC"
echo "------------------------------------------------------------------"

if [[ "$SMOKE_PASS" == "failed" ]]; then
    exit 2
fi
if [[ "$FULL_PASS" == "failed" ]]; then
    exit 3
fi
if [[ "$EQUIV_PASS" == "failed_row_by_row" ]] || [[ "$EQUIV_PASS" == "failed_posthoc" ]] || [[ "$EQUIV_PASS" == "failed_other" ]]; then
    exit 4
fi
exit 0

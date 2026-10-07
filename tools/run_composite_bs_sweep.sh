#!/bin/bash
# Batch size sweep for COMPOSITE DDP on Aurora
# Runs BS=1,2,3,4 sequentially with hang detection and automatic cleanup.
#
# Usage: ssh <node> bash /path/to/tools/run_composite_bs_sweep.sh
#
# Each test runs with a 5-minute hang-detection timeout. If a test hangs,
# it is killed and the next batch size is tried.
#
# Results summary is printed at the end.

set -uo pipefail

PROJ_DIR="/lus/flare/projects/ModCon/ngetty/BaseMM_PRISM"
SCRIPT="${PROJ_DIR}/tools/run_composite_interactive.sh"
SWEEP_LOG="/tmp/composite_bs_sweep_$(date +%Y%m%d_%H%M%S).log"

# Configuration for sweep
MAX_STEPS=10
TIMEOUT_SECS=300  # 5 minutes per test
declare -a BATCH_SIZES=(1 2 3 4)
declare -a GRAD_ACCUM=(6 3 2 1)  # Keep effective_batch ~ 36 (BS*accum*6_ranks=36)

echo "=============================================="
echo "  COMPOSITE DDP Batch Size Sweep"
echo "=============================================="
echo "Node: $(hostname)"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Max steps per test: ${MAX_STEPS}"
echo "Hang timeout: ${TIMEOUT_SECS}s"
echo "Sweep log: ${SWEEP_LOG}"
echo "=============================================="
echo ""

# Results arrays
declare -a RESULTS_STATUS=()
declare -a RESULTS_THROUGHPUT=()
declare -a RESULTS_PEAK_MEM=()
declare -a RESULTS_STEP_TIME=()

for i in "${!BATCH_SIZES[@]}"; do
  BS="${BATCH_SIZES[$i]}"
  GA="${GRAD_ACCUM[$i]}"
  EFF_BS=$((BS * GA * 6))

  echo ""
  echo "======================================================"
  echo "  TEST: BS=${BS}, grad_accum=${GA}, effective_batch=${EFF_BS}"
  echo "======================================================"
  echo ""

  # Step 1: Clean up any lingering processes
  echo "[SWEEP] Cleaning up stale processes..."
  pkill -9 python3 2>/dev/null || true
  pkill -9 -f mpiexec 2>/dev/null || true
  sleep 5
  echo "[SWEEP] Cleanup complete."

  # Step 2: Run the test with hang detection
  COMPOSITE_BS="${BS}" \
  COMPOSITE_MAX_STEPS="${MAX_STEPS}" \
  COMPOSITE_GRAD_ACCUM="${GA}" \
  COMPOSITE_TIMEOUT_SECS="${TIMEOUT_SECS}" \
  DDP_DEBUG=1 \
  DDP_DEBUG_STEPS=2 \
  LOG_EVERY_N_STEPS=1 \
  bash "${SCRIPT}" 2>&1
  TEST_EXIT=$?

  # Step 3: Extract results from log
  # Find the most recent log file
  LATEST_LOG=$(find "${PROJ_DIR}"/outputs/composite-interactive/ -name run.log -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

  if [ "${TEST_EXIT}" -eq 124 ]; then
    STATUS="HANG (timeout ${TIMEOUT_SECS}s)"
    THROUGHPUT="N/A"
    PEAK_MEM="N/A"
    STEP_TIME="N/A"
  elif [ "${TEST_EXIT}" -ne 0 ]; then
    STATUS="FAIL (exit ${TEST_EXIT})"
    THROUGHPUT="N/A"
    PEAK_MEM="N/A"
    STEP_TIME="N/A"
    # Check for OOM
    if [ -n "${LATEST_LOG}" ] && grep -q "signal 9\|SIGKILL\|OutOfMemory\|OOM" "${LATEST_LOG}" 2>/dev/null; then
      STATUS="OOM"
    fi
  else
    STATUS="OK"
    # Extract throughput from last step
    if [ -n "${LATEST_LOG}" ]; then
      THROUGHPUT=$(grep -oP '\d+\.\d+ samples/sec' "${LATEST_LOG}" 2>/dev/null | tail -1 || echo "N/A")
      PEAK_MEM=$(grep -oP 'Peak: \K[\d.]+' "${LATEST_LOG}" 2>/dev/null | tail -1 || echo "N/A")
      if [ -n "${PEAK_MEM}" ]; then
        PEAK_MEM="${PEAK_MEM} GB"
      else
        PEAK_MEM="N/A"
      fi
      # Try to get per-step time from timing lines
      STEP_TIME=$(grep -oP 'Data: [\d.]+s \| Fwd: [\d.]+s \| Bwd: [\d.]+s' "${LATEST_LOG}" 2>/dev/null | tail -1 || echo "N/A")
    else
      THROUGHPUT="N/A"
      PEAK_MEM="N/A"
      STEP_TIME="N/A"
    fi
  fi

  RESULTS_STATUS+=("${STATUS}")
  RESULTS_THROUGHPUT+=("${THROUGHPUT}")
  RESULTS_PEAK_MEM+=("${PEAK_MEM}")
  RESULTS_STEP_TIME+=("${STEP_TIME}")

  echo ""
  echo "[SWEEP] BS=${BS}: ${STATUS} | Throughput: ${THROUGHPUT} | Peak: ${PEAK_MEM}"
  echo ""

  # If OOM, skip larger batch sizes
  if [ "${STATUS}" = "OOM" ]; then
    echo "[SWEEP] OOM at BS=${BS} — skipping remaining batch sizes."
    for j in $(seq $((i + 1)) $((${#BATCH_SIZES[@]} - 1))); do
      RESULTS_STATUS+=("SKIPPED (OOM at BS=${BS})")
      RESULTS_THROUGHPUT+=("N/A")
      RESULTS_PEAK_MEM+=("N/A")
      RESULTS_STEP_TIME+=("N/A")
    done
    break
  fi

  # Post-test cleanup
  pkill -9 python3 2>/dev/null || true
  sleep 3
done

# Final summary
echo ""
echo "=============================================="
echo "  BATCH SIZE SWEEP RESULTS"
echo "=============================================="
printf "%-6s %-8s %-10s %-20s %-15s\n" "BS" "EffBS" "Status" "Throughput" "Peak Memory"
printf "%-6s %-8s %-10s %-20s %-15s\n" "---" "-----" "-------" "-----------" "------------"
for i in "${!BATCH_SIZES[@]}"; do
  BS="${BATCH_SIZES[$i]}"
  GA="${GRAD_ACCUM[$i]}"
  EFF_BS=$((BS * GA * 6))
  printf "%-6s %-8s %-10s %-20s %-15s\n" \
    "${BS}" "${EFF_BS}" "${RESULTS_STATUS[$i]}" "${RESULTS_THROUGHPUT[$i]}" "${RESULTS_PEAK_MEM[$i]}"
done
echo "=============================================="
echo "Detailed logs: ${PROJ_DIR}/outputs/composite-interactive/"
echo "Sweep log: ${SWEEP_LOG}"

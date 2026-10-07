#!/bin/bash
# Automated AGPT2B test sweep — runs multiple configs sequentially with cleanup
# Usage: ssh <node> bash /path/to/run_agpt2b_sweep.sh
#
# Writes results to SUMMARY_FILE for polling.
# Each test gets a timeout; if it hangs, it's killed and the next test starts.

set -uo pipefail

PROJ_DIR="/lus/flare/projects/ModCon/ngetty/BaseMM_PRISM"
SWEEP_DIR="${PROJ_DIR}/outputs/agpt2b-sweep/$(date +%Y-%m-%d/%H-%M-%S)"
SUMMARY_FILE="${SWEEP_DIR}/summary.txt"
SCRIPT="${PROJ_DIR}/tools/run_agpt2b_interactive.sh"

mkdir -p "${SWEEP_DIR}"

echo "=== AGPT2B Sweep Started: $(date) ===" | tee "${SUMMARY_FILE}"
echo "Node: $(hostname)" | tee -a "${SUMMARY_FILE}"
echo "Sweep dir: ${SWEEP_DIR}" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"

# Define tests: NAME|BS|ACCUM|STEPS|ATTN|TIMEOUT
# Sweep 3: find exact crash boundary + BS=6 stability
TESTS=(
  "sdpa-bs7-accum3-20steps|7|3|20|sdpa|300"
  "sdpa-bs6-accum3-100steps|6|3|100|sdpa|600"
)

run_test() {
  local NAME="$1" BS="$2" ACCUM="$3" STEPS="$4" ATTN="$5" TIMEOUT="$6"
  local TEST_START=$(date +%s)

  echo "----------------------------------------" | tee -a "${SUMMARY_FILE}"
  echo "[TEST] ${NAME} — BS=${BS} accum=${ACCUM} steps=${STEPS} attn=${ATTN} timeout=${TIMEOUT}s" | tee -a "${SUMMARY_FILE}"
  echo "[TEST] Started: $(date)" | tee -a "${SUMMARY_FILE}"

  # Clean up from previous test
  pkill -9 python3 2>/dev/null || true
  pkill -9 mpiexec 2>/dev/null || true
  sleep 10

  # Verify clean
  local PCOUNT=$(ps aux | grep python3 | grep -v grep | grep -v geopmd | wc -l)
  if [ "$PCOUNT" -gt 0 ]; then
    echo "[TEST] WARNING: ${PCOUNT} python3 processes still running after cleanup" | tee -a "${SUMMARY_FILE}"
    pkill -9 python3 2>/dev/null || true
    sleep 5
  fi

  # Run test
  AGPT2B_BS="${BS}" \
  AGPT2B_MAX_STEPS="${STEPS}" \
  AGPT2B_GRAD_ACCUM="${ACCUM}" \
  AGPT2B_ATTN_IMPL="${ATTN}" \
  AGPT2B_TIMEOUT_SECS="${TIMEOUT}" \
  bash "${SCRIPT}" 2>&1

  local EXIT_CODE=$?
  local TEST_END=$(date +%s)
  local DURATION=$((TEST_END - TEST_START))

  # Find the output dir (most recent — two levels: date/timestamp)
  local LATEST_OUT=$(ls -td ${PROJ_DIR}/outputs/agpt2b-interactive/*/*/ 2>/dev/null | head -1)
  local LOG_FILE="${LATEST_OUT}run.log"

  # Extract key metrics from log
  local LAST_STEP=$(grep -oP '^Step \K\d+' "${LOG_FILE}" 2>/dev/null | tail -1)
  local LAST_LOSS=$(grep -oP '^Step \d+: Loss \K[0-9.]+' "${LOG_FILE}" 2>/dev/null | tail -1)
  local THROUGHPUT=$(grep -oP '\[THROUGHPUT\] \K[0-9.]+' "${LOG_FILE}" 2>/dev/null | tail -1)
  local PEAK_MEM=$(grep -oP '\[MEMORY\] Peak: \K[0-9.]+GB' "${LOG_FILE}" 2>/dev/null | tail -1)
  local UR_ERROR=$(grep -c "UR_RESULT_ERROR_OUT_OF_RESOURCES" "${LOG_FILE}" 2>/dev/null)
  local OOM_ERROR=$(grep -c "OutOfMemoryError" "${LOG_FILE}" 2>/dev/null)

  # Determine result — check UR/OOM even on exit 0 (mpiexec may mask child errors)
  local RESULT="UNKNOWN"
  if [ "${UR_ERROR}" -gt 0 ]; then
    RESULT="UR_CRASH"
  elif [ "${OOM_ERROR}" -gt 0 ]; then
    RESULT="OOM"
  elif [ "${EXIT_CODE}" -eq 0 ]; then
    RESULT="PASS"
  elif [ "${EXIT_CODE}" -eq 124 ]; then
    RESULT="HANG"
  else
    RESULT="FAIL(${EXIT_CODE})"
  fi

  echo "[RESULT] ${NAME}: ${RESULT} | steps=${LAST_STEP:-0}/${STEPS} | loss=${LAST_LOSS:-N/A} | throughput=${THROUGHPUT:-N/A} samp/s | peak_mem=${PEAK_MEM:-N/A} | duration=${DURATION}s" | tee -a "${SUMMARY_FILE}"
  echo "[RESULT] Log: ${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
  echo "" | tee -a "${SUMMARY_FILE}"

  return ${EXIT_CODE}
}

# Run all tests
PASS_COUNT=0
FAIL_COUNT=0

for TEST_SPEC in "${TESTS[@]}"; do
  IFS='|' read -r NAME BS ACCUM STEPS ATTN TIMEOUT <<< "${TEST_SPEC}"
  run_test "${NAME}" "${BS}" "${ACCUM}" "${STEPS}" "${ATTN}" "${TIMEOUT}"
  if [ $? -eq 0 ]; then
    ((PASS_COUNT++))
  else
    ((FAIL_COUNT++))
  fi
done

echo "========================================" | tee -a "${SUMMARY_FILE}"
echo "=== SWEEP COMPLETE: $(date) ===" | tee -a "${SUMMARY_FILE}"
echo "=== PASS: ${PASS_COUNT} / FAIL: ${FAIL_COUNT} / TOTAL: ${#TESTS[@]} ===" | tee -a "${SUMMARY_FILE}"
echo "DONE" >> "${SUMMARY_FILE}"

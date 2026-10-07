#!/bin/bash
# Spawn-worker smoke gate. Exercises the OpenAI API server boot path so we
# catch plugin-registration regressions that ONLY surface under the spawn
# multiprocessing executor (each worker re-imports our module fresh).
#
# Required for VLLM-3+ which moves registration from the imperative
# `register()` call to a setuptools `vllm.general_plugins` entry point.
#
# Usage:
#   bash tools/_vllm_serve_smoke_runner.sh \
#     --vllm-model exported/prism-olmo1b-image-step500 \
#     [--port 8000] [--timeout 180]
#
# Note: `set -u` (nounset) is intentionally omitted — Aurora's Lmod
# `module load` reads unset shell vars and crashes under nounset.
set -eo pipefail

PROJ="${PRISM_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJ"

VLLM_MODEL=""
PORT=8000
TIMEOUT=180
# Set --ts-route only when the exported model has time_series in
# active_modalities. Otherwise we'd hit the placeholder-not-in-prompt
# error trying to send a ts payload to an image-only model.
TS_ROUTE=0
EXTRA_SERVE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-model) VLLM_MODEL="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --timeout)    TIMEOUT="$2"; shift 2 ;;
        --ts-route)   TS_ROUTE=1; shift ;;
        --)           shift; EXTRA_SERVE_ARGS=("$@"); break ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$VLLM_MODEL" ]]; then
    echo "ERROR: --vllm-model is required" >&2
    exit 2
fi

module load frameworks/2025.3.1 2>&1 | tail -1
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python

LOG="vllm_serve_smoke_${PORT}.log"
echo "[serve-smoke] launching tools/vllm_serve.py on :$PORT (log: $LOG)"

"$PY" tools/vllm_serve.py \
    --model "$VLLM_MODEL" \
    --port "$PORT" \
    --enforce-eager \
    --trust-remote-code \
    "${EXTRA_SERVE_ARGS[@]}" \
    > "$LOG" 2>&1 &
SERVE_PID=$!

cleanup() {
    if kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[serve-smoke] killing server pid=$SERVE_PID"
        kill "$SERVE_PID" 2>/dev/null || true
        # Give it a couple seconds to drain, then SIGKILL.
        for _ in 1 2 3 4 5; do
            kill -0 "$SERVE_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$SERVE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Poll /v1/models until 200 or timeout.
# Aurora compute nodes set http_proxy=proxy.alcf.anl.gov:3128; the proxy can't
# reach localhost so curl must be told to bypass it.
DEADLINE=$((SECONDS + TIMEOUT))
URL="http://127.0.0.1:${PORT}/v1/models"
echo "[serve-smoke] polling $URL (timeout ${TIMEOUT}s)"
while true; do
    if curl -fsS -m 2 --noproxy '*' "$URL" >/dev/null 2>&1; then
        echo "[serve-smoke] /v1/models is up"
        break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[serve-smoke] FAILED: server pid=$SERVE_PID died before becoming ready"
        echo "--- last 40 log lines ---"
        tail -40 "$LOG" || true
        exit 1
    fi
    if (( SECONDS > DEADLINE )); then
        echo "[serve-smoke] FAILED: timeout after ${TIMEOUT}s"
        echo "--- last 40 log lines ---"
        tail -40 "$LOG" || true
        exit 1
    fi
    sleep 2
done

echo "[serve-smoke] verifying /v1/models lists $VLLM_MODEL"
"$PY" tools/vllm_serve_smoke_client.py --port "$PORT" --vllm-model "$VLLM_MODEL"

if [[ "$TS_ROUTE" -eq 1 ]]; then
    echo "[serve-smoke] POST /v1/prism/ts with synthetic tensor"
    "$PY" tools/vllm_ts_serve_smoke.py --port "$PORT"
fi

echo "[serve-smoke] PASSED"

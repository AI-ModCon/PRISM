#!/bin/bash
# Compute-node smoke gate for the vLLM plugin (PR Pre-flight).
#
# Runs three checks in one PBS job:
#   1. Plugin registration + LLM boot + 4-token generate (tools/vllm_smoke.py)
#   2. Token-level parity assertion (tools/vllm_parity_assert.py, default 18/20)
#   3. (Optional) Time-series checkpoint existence + key presence
#      (tools/vllm_check_ts_checkpoint.py), only when --ts-checkpoint is passed
#
# Mirrors tools/_vllm_parity_runner.sh: load frameworks/2025.3.1, invoke its
# python directly so we use the in-module vllm install (0.15.0+xpu).
#
# Usage:
#   bash tools/_vllm_smoke_runner.sh \
#     --vllm-model exported/prism-olmo1b-image-step500 \
#     --checkpoint outputs/<run>/checkpoints/step_500 \
#     --image test_images/cat.jpg \
#     [--ts-checkpoint outputs/<ts-run>/checkpoints/step_N]
#
# Override $PRISM_DIR if not in a git working tree.
#
# Note: we deliberately do NOT use `set -u` (nounset) — Aurora's Lmod
# `module load` reads $ZSH_EVAL_CONTEXT unconditionally and that variable
# is unset in non-interactive bash sessions, killing the script before any
# work runs.
set -eo pipefail

PROJ="${PRISM_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJ"

VLLM_MODEL=""
CHECKPOINT=""
IMAGE=""
TS_CHECKPOINT=""
WINDOW=20
MIN_MATCH=18
PROMPT_FORM="<image>The image shows"
# Reference vLLM token ids captured from PR #41 main on:
#   image=test_images/cat.jpg, prompt="<image>The image shows"
#   vllm-model=exported/prism-olmo1b-image-step500
# The vLLM path is deterministic on XPU; the HF demo path is not, so we
# compare against the frozen vLLM reference, not the demo. See
# tools/vllm_parity_assert.py for the rationale.
REF_TOKENS="${REF_TOKENS:-247,3168,13,26305,14,13824,2829,342,247,3168,17848,10985,253,1755,15,380,17848,310,20618,275}"
# Parity is keyed to PR #41's exported/prism-olmo1b-image-step500. Synthetic
# or random-init checkpoints won't match it — set --no-parity to skip.
RUN_PARITY=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-model)   VLLM_MODEL="$2"; shift 2 ;;
        --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --ts-checkpoint) TS_CHECKPOINT="$2"; shift 2 ;;
        --window)       WINDOW="$2"; shift 2 ;;
        --min-match)    MIN_MATCH="$2"; shift 2 ;;
        --prompt-form)  PROMPT_FORM="$2"; shift 2 ;;
        --no-parity)    RUN_PARITY=0; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$VLLM_MODEL" ]]; then
    echo "ERROR: --vllm-model is required" >&2
    exit 2
fi

module load frameworks/2025.3.1 2>&1 | tail -1
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python

echo
echo "=================================================="
echo "Smoke 1/3: registration + boot + generate"
echo "=================================================="
"$PY" tools/vllm_smoke.py --vllm-model "$VLLM_MODEL"

if [[ "$RUN_PARITY" -eq 0 ]]; then
    echo
    echo "[smoke] --no-parity: skipping image parity check"
elif [[ -n "$IMAGE" ]]; then
    echo
    echo "=================================================="
    echo "Smoke 2/3: image parity (>= $MIN_MATCH/$WINDOW first tokens vs PR #41 reference)"
    echo "=================================================="
    # Demo path is optional and informational only (non-deterministic on XPU).
    DEMO_ARGS=()
    if [[ -n "$CHECKPOINT" ]]; then
        DEMO_ARGS=(--checkpoint "$CHECKPOINT")
    fi
    "$PY" tools/vllm_parity_assert.py \
        "${DEMO_ARGS[@]}" \
        --vllm-model "$VLLM_MODEL" \
        --image "$IMAGE" \
        --window "$WINDOW" \
        --min-match "$MIN_MATCH" \
        --prompt-form "$PROMPT_FORM" \
        --ref-tokens "$REF_TOKENS"
else
    echo
    echo "[smoke] skipping parity (need --image; --checkpoint optional)"
fi

if [[ -n "$TS_CHECKPOINT" ]]; then
    echo
    echo "=================================================="
    echo "Smoke 3/3: time-series checkpoint presence"
    echo "=================================================="
    "$PY" tools/vllm_check_ts_checkpoint.py --checkpoint "$TS_CHECKPOINT"
else
    echo
    echo "[smoke] skipping ts-checkpoint check (no --ts-checkpoint)"
fi

echo
echo "[smoke] DONE"

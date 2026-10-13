#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="PRISM-QWEN3-0-6B-SCITS-FORECAST"
RUN_DESIGN="PRISM-QWEN3-0-6B-SCITS-FORECAST"
file="experiments/LLM_timeseries_scaling.yaml"
QUEUE="capacity"
Walltime="06:00:00"
NODES=1
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

cd "$REPO_ROOT"

python_at_least_310() {
    "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1
}

if [[ -n "${PRISM_LAUNCH_PYTHON:-}" ]]; then
    PYTHON_BIN="$PRISM_LAUNCH_PYTHON"
elif [[ -x "$REPO_ROOT/.venv-deepspeed/bin/python" ]] && \
    python_at_least_310 "$REPO_ROOT/.venv-deepspeed/bin/python"; then
    PYTHON_BIN="$REPO_ROOT/.venv-deepspeed/bin/python"
elif command -v python3.12 >/dev/null 2>&1 && python_at_least_310 python3.12; then
    PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1 && python_at_least_310 python3; then
    PYTHON_BIN="python3"
else
    echo "Python >=3.10 is required. Set PRISM_LAUNCH_PYTHON or create .venv-deepspeed." >&2
    exit 1
fi

if ! python_at_least_310 "$PYTHON_BIN"; then
    echo "PRISM_LAUNCH_PYTHON must point to Python >=3.10: $PYTHON_BIN" >&2
    exit 1
fi

if ! LAUNCH_OUTPUT=$("$PYTHON_BIN" tools/launch_aurora_web.py \
    --id "$RUN_ID" \
    --file "$file" \
    --design "$RUN_DESIGN" \
    --nodes "$NODES" \
    --batch \
    --queue "$QUEUE" \
    --walltime "$Walltime" \
    --dist-strategy hsdp \
    --fsdp-sharding shard_grad_op \
    --webdataset-dir /flare/ModCon/pemami/data/SciTS-forecast \
    --webdataset-modality time_series \
    --wandb-project=goose \
    wandb.entity=pemami \
    +training.data_num_workers=1 \
    --dry-run 2>&1); then
    printf '%s\n' "$LAUNCH_OUTPUT" >&2
    exit 1
fi
printf '%s\n' "$LAUNCH_OUTPUT"

RUN_SCRIPT=$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^Generated Run Script: //p' | tail -n 1)
if [[ -z "$RUN_SCRIPT" || ! -f "$RUN_SCRIPT" ]]; then
    echo "Could not locate the generated Aurora run script." >&2
    exit 1
fi

TRAIN_OUTPUT_DIR=$(sed -n 's/^echo "Output Dir: \(.*\)"$/\1/p' "$RUN_SCRIPT" | head -n 1)
if [[ -z "$TRAIN_OUTPUT_DIR" ]]; then
    echo "Could not determine the training output directory from $RUN_SCRIPT." >&2
    exit 1
fi

cat >> "$RUN_SCRIPT" <<EOF

# --- Post-training SciTS validation ---
TRAIN_EXIT=\$?
if [[ \$TRAIN_EXIT -ne 0 ]]; then
    echo "Training failed with exit code \$TRAIN_EXIT; skipping evaluation." >&2
    exit "\$TRAIN_EXIT"
fi

CHECKPOINT=\$(find "$TRAIN_OUTPUT_DIR/checkpoints" -type f -name model.safetensors -path '*/step_*/*' -print \
    | sort -V | tail -n 1)
if [[ -z "\$CHECKPOINT" ]]; then
    echo "No model.safetensors checkpoint found under $TRAIN_OUTPUT_DIR/checkpoints." >&2
    exit 1
fi

echo "Training complete. Evaluating all SciTS validation samples from \$CHECKPOINT"
export CHECKPOINT
mpiexec -n 1 -ppn 1 bash -lc '
set -eo pipefail
module purge || exit
module use /opt/aurora/26.26.0/modulefiles || exit
module load oneapi/release/2025.3.1 || exit
module load mpich/prd/5.0.0.aurora_test.51a9474 || exit
module load frameworks/2025.3.1 || exit

if [[ -f /tmp/deepspeed_env/.venv-deepspeed/bin/activate ]]; then
    source /tmp/deepspeed_env/.venv-deepspeed/bin/activate
else
    echo "Packed training environment is missing on the evaluation worker." >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/huggingface/hub
export HF_HUB_CACHE=/tmp/huggingface/hub
export PRISM_VAL_SHARDS_DIR=/flare/ModCon/pemami/data/SciTS-forecast/val_shards

python tools/universal_evaluator.py \
    --checkpoint "\$CHECKPOINT" \
    --mode verify_timeseries_scits \
    --validation \
    --limit 0 \
    --backbone Qwen/Qwen3-0.6B
'
EOF

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Augmented run script (not submitted): $RUN_SCRIPT"
    exit 0
fi

echo "Submitting training and evaluation job: $RUN_SCRIPT"
qsub "$RUN_SCRIPT"
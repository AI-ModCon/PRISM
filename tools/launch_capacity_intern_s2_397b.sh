#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="PRISM-QWEN3-0-6B-INTERN-S2-397B-UNFROZEN-PADDING"
RUN_DESIGN="PRISM-QWEN3-0-6B-INTERN-S2-397B-UNFROZEN"
file="experiments/LLM_timeseries_scaling.yaml"
NODES=1
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

cd "$REPO_ROOT"

if ! LAUNCH_OUTPUT=$(python3 tools/launch_aurora_web.py \
    --id "$RUN_ID" \
    --file "$file" \
    --design "$RUN_DESIGN" \
    --nodes "$NODES" \
    --batch \
    --queue capacity \
    --walltime 06:00:00 \
    --dist-strategy hsdp \
    --fsdp-sharding shard_grad_op \
    --webdataset-dir /flare/ModCon/pemami/data/SciTS-processed \
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

if [[ -f /tmp/deepspeed_env/.venv-deepspeed/bin/activate ]]; then
    source /tmp/deepspeed_env/.venv-deepspeed/bin/activate
fi
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/huggingface/hub
export HF_HUB_CACHE=/tmp/huggingface/hub
export PRISM_VAL_SHARDS_DIR=/flare/ModCon/pemami/data/SciTS-processed/val_shards

echo "Training complete. Evaluating all SciTS validation samples from \$CHECKPOINT"
python tools/universal_evaluator.py \
    --checkpoint "\$CHECKPOINT" \
    --mode verify_timeseries_scits \
    --validation \
    --limit 0 \
    --backbone Qwen/Qwen3-0.6B
EOF

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Augmented run script (not submitted): $RUN_SCRIPT"
    exit 0
fi

echo "Submitting training and evaluation job: $RUN_SCRIPT"
qsub "$RUN_SCRIPT"
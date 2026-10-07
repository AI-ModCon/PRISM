#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="PRISM-QWEN3-0-6B-INTERN-S2-397B-EVAL"
QUEUE="capacity"
NODES=1
WALLTIME="02:00:00"
DRY_RUN=0
TRAIN_OUTPUT_DIR=""

usage() {
    echo "Usage: $0 TRAIN_OUTPUT_DIR [--queue QUEUE] [--nodes N] [--walltime HH:MM:SS] [--dry-run]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            QUEUE="$2"
            shift 2
            ;;
        --nodes)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            NODES="$2"
            shift 2
            ;;
        --walltime)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            WALLTIME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --*)
            usage
            exit 2
            ;;
        *)
            if [[ -n "$TRAIN_OUTPUT_DIR" ]]; then
                usage
                exit 2
            fi
            TRAIN_OUTPUT_DIR="$1"
            shift
            ;;
    esac
done

if [[ -z "$TRAIN_OUTPUT_DIR" ]]; then
    usage
    exit 2
fi

if [[ ! -d "$TRAIN_OUTPUT_DIR" ]]; then
    echo "Training output directory does not exist: $TRAIN_OUTPUT_DIR" >&2
    exit 1
fi
TRAIN_OUTPUT_DIR="$(cd "$TRAIN_OUTPUT_DIR" && pwd)"

CHECKPOINT=$(find "$TRAIN_OUTPUT_DIR/checkpoints" -type f -name model.safetensors -path '*/step_*/*' -print \
    | sort -V | tail -n 1)
if [[ -z "$CHECKPOINT" ]]; then
    echo "No model.safetensors checkpoint found under $TRAIN_OUTPUT_DIR/checkpoints." >&2
    exit 1
fi

cd "$REPO_ROOT"
LOGS_DIR="$REPO_ROOT/logs/$RUN_ID"
mkdir -p "$LOGS_DIR" "$REPO_ROOT/jobs"
DATE_STR=$(date +%Y-%m-%d)
TIME_STR=$(date +%H-%M-%S)
RUN_SCRIPT="$REPO_ROOT/jobs/run_aurora_${RUN_ID}_${DATE_STR}_${TIME_STR}_batch.sh"

cat > "$RUN_SCRIPT" <<EOF
#!/bin/bash -l
#PBS -l select=$NODES
#PBS -l walltime=$WALLTIME
#PBS -l filesystems=home:flare
#PBS -q $QUEUE
#PBS -A ModCon
#PBS -k doe
#PBS -j oe
#PBS -N $RUN_ID
#PBS -o $LOGS_DIR/
#PBS -e $LOGS_DIR/

set -eo pipefail

cd "$REPO_ROOT"
module load frameworks/2025.3.1
module load hdf5
set -u

if [[ -f /tmp/deepspeed_env/.venv-deepspeed/bin/activate ]]; then
    source /tmp/deepspeed_env/.venv-deepspeed/bin/activate
elif [[ -f .venv-deepspeed/bin/activate ]]; then
    source .venv-deepspeed/bin/activate
fi
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HF_HOME=/flare/ModCon/pemami
export TRANSFORMERS_CACHE=/flare/ModCon/pemami/hub
export HF_HUB_CACHE=/flare/ModCon/pemami/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PRISM_VAL_SHARDS_DIR=/flare/ModCon/pemami/data/SciTS-processed/val_shards

echo "Evaluating all SciTS validation samples from $CHECKPOINT"
python tools/universal_evaluator.py \\
    --checkpoint "$CHECKPOINT" \\
    --mode verify_timeseries_scits \\
    --validation \\
    --limit 0 \\
    --backbone Qwen/Qwen3-0.6B
EOF

chmod +x "$RUN_SCRIPT"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Evaluation run script (not submitted): $RUN_SCRIPT"
    exit 0
fi

echo "Submitting SciTS validation job: $RUN_SCRIPT"
qsub "$RUN_SCRIPT"
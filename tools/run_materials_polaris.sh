#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A ModCon
#PBS -k doe
#PBS -j oe
#PBS -N MAT-TEXT-GRAPH

# Submit with:
#   bash tools/run_materials_polaris.sh
#
# Optional submission-time overrides:
#   MODE=text QUEUE=debug WALLTIME=01:00:00 EPOCHS=3 MAX_SAMPLES=1000 RUN_NOTE=smoke \
#     bash tools/run_materials_polaris.sh
# This launcher uses Qwen and PRISM token-level fusion.
# MODE may be joint (default), text, or graph, and is used for both training
# and evaluation.
# MAX_SAMPLES=full (the default) uses every complete text/CIF/target triplet.

set -euo pipefail

TOOLS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$TOOLS_DIR/.." && pwd)
# PBS runs a spooled copy of this file, so preserve the original project path
# explicitly rather than deriving it from BASH_SOURCE inside the job.
if [[ -n "${PBS_JOBID:-}" ]]; then
    PROJECT_DIR=${MATERIALS_PROJECT_DIR:-${PBS_O_WORKDIR:-$PROJECT_DIR}}
fi

MODE=${MODE:-joint}
case "$MODE" in
    joint|text|graph) ;;
    *)
        echo "ERROR: MODE must be one of: joint, text, graph (got: $MODE)" >&2
        exit 2
        ;;
esac
MATERIALS_RUN_ID=${MATERIALS_RUN_ID:-materials-prism-bandgap-$MODE}
#QUEUE=${QUEUE:-capacity}
#QUEUE=${QUEUE:-preemptable}
QUEUE=${QUEUE:-debug}
WALLTIME=${WALLTIME:-00:40:00}
TARGET=${TARGET:-band_gap}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
HIDDEN_DIM=${HIDDEN_DIM:-128}
GRAPH_LAYERS=${GRAPH_LAYERS:-3}
GRAPH_TOKENS=${GRAPH_TOKENS:-8}
CUTOFF=${CUTOFF:-5.0}
MAX_NEIGHBORS=${MAX_NEIGHBORS:-16}
MAX_SAMPLES=${MAX_SAMPLES:-full}
TEXT_MODEL=${TEXT_MODEL:-Qwen/Qwen3-0.6B}
TRAIN_TEXT_BACKBONE=${TRAIN_TEXT_BACKBONE:-0}
TEXT_LEARNING_RATE=${TEXT_LEARNING_RATE:-1e-5}
MATERIALS_HF_HOME=${MATERIALS_HF_HOME:-/eagle/projects/ModCon/$USER/huggingface}
MATERIALS_HF_HUB_CACHE=${MATERIALS_HF_HUB_CACHE:-$MATERIALS_HF_HOME/hub}
if [[ -z "${MATERIALS_PYTHON:-}" ]]; then
    # Resolve Python from the caller's inherited shell environment. The
    # absolute path is passed through qsub so the compute job uses the same
    # interpreter even if PBS reconstructs PATH differently.
    MATERIALS_PYTHON=$(command -v python || true)
fi
SEED=${SEED:-17}
RUN_NOTE=${RUN_NOTE:-full}
case "$TRAIN_TEXT_BACKBONE" in
    0|1) ;;
    *)
        echo "ERROR: TRAIN_TEXT_BACKBONE must be 0 or 1 (got: $TRAIN_TEXT_BACKBONE)" >&2
        exit 2
        ;;
esac

# When invoked on a login node, submit this same file. Command-line -o/-e
# values are absolute because PBS resolves relative paths outside the repo.
if [[ -z "${PBS_JOBID:-}" ]]; then
    if [[ "$QUEUE" == "prod" ]]; then
        echo "ERROR: Polaris prod requires at least 10 nodes; this trainer requests one." >&2
        echo "Use QUEUE=capacity for a long run or QUEUE=debug for a <=1 hour smoke." >&2
        exit 2
    fi
    if [[ ! -x "$MATERIALS_PYTHON" ]]; then
        echo "ERROR: Selected Python is not executable: $MATERIALS_PYTHON" >&2
        exit 1
    fi
    echo "Submitting with Python: $MATERIALS_PYTHON"
    "$MATERIALS_PYTHON" -c 'import sys; print("Submitting environment:", sys.prefix)'
    LOG_DIR="$PROJECT_DIR/logs/materials-prism-$MODE"
    mkdir -p "$LOG_DIR"
    JOB_VARS="MATERIALS_PROJECT_DIR=$PROJECT_DIR,MATERIALS_RUN_ID=$MATERIALS_RUN_ID,MODE=$MODE"
    JOB_VARS+=",TARGET=$TARGET,EPOCHS=$EPOCHS"
    JOB_VARS+=",BATCH_SIZE=$BATCH_SIZE,NUM_WORKERS=$NUM_WORKERS"
    JOB_VARS+=",LEARNING_RATE=$LEARNING_RATE,HIDDEN_DIM=$HIDDEN_DIM"
    JOB_VARS+=",GRAPH_LAYERS=$GRAPH_LAYERS,GRAPH_TOKENS=$GRAPH_TOKENS,CUTOFF=$CUTOFF,MAX_NEIGHBORS=$MAX_NEIGHBORS"
    JOB_VARS+=",MAX_SAMPLES=$MAX_SAMPLES,SEED=$SEED,RUN_NOTE=$RUN_NOTE"
    JOB_VARS+=",TEXT_MODEL=$TEXT_MODEL,TRAIN_TEXT_BACKBONE=$TRAIN_TEXT_BACKBONE"
    JOB_VARS+=",TEXT_LEARNING_RATE=$TEXT_LEARNING_RATE"
    JOB_VARS+=",MATERIALS_HF_HOME=$MATERIALS_HF_HOME,MATERIALS_HF_HUB_CACHE=$MATERIALS_HF_HUB_CACHE"
    JOB_VARS+=",MATERIALS_PYTHON=$MATERIALS_PYTHON"
    qsub -V -q "$QUEUE" -l "walltime=$WALLTIME" -v "$JOB_VARS" \
        -o "$LOG_DIR/" -e "$LOG_DIR/" \
        "$PROJECT_DIR/tools/run_materials_polaris.sh"
    exit 0
fi

cd "$PROJECT_DIR"

# conda/2025-09-28 currently names retired CPE modules
# (gcc-native/14.2 and cray-hdf5-parallel/1.14.3.5), so loading that
# modulefile fails under the current Polaris CPE. Supply the two current
# runtime library locations needed by the inherited Python's PyTorch build.
CUDA_RUNTIME_LIB=/soft/compilers/cudatoolkit/cuda-13.0.1/lib64
MPI_RUNTIME_LIB=/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3/lib
if [[ ! -d "$CUDA_RUNTIME_LIB" || ! -e "$MPI_RUNTIME_LIB/libmpi_gnu.so.12" ]]; then
    echo "ERROR: Current Polaris CUDA/MPI compatibility libraries were not found." >&2
    echo "CUDA: $CUDA_RUNTIME_LIB" >&2
    echo "MPI : $MPI_RUNTIME_LIB/libmpi_gnu.so.12" >&2
    exit 1
fi
COMPAT_LIB_DIR=$(mktemp -d "/tmp/$USER-prism-libs.XXXXXX")
trap 'rm -r "$COMPAT_LIB_DIR"' EXIT
ln -s "$MPI_RUNTIME_LIB/libmpi_gnu.so.12" "$COMPAT_LIB_DIR/libmpi_gnu_123.so.12"
export LD_LIBRARY_PATH="$COMPAT_LIB_DIR:$CUDA_RUNTIME_LIB:$MPI_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Compute nodes should only read models staged on the shared filesystem.
export HF_HOME="$MATERIALS_HF_HOME"
export HF_HUB_CACHE="$MATERIALS_HF_HUB_CACHE"
export HUGGINGFACE_HUB_CACHE="$MATERIALS_HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON_BIN=$MATERIALS_PYTHON
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Materials Python is not executable: $PYTHON_BIN" >&2
    echo "Set MATERIALS_PYTHON to the absolute path of the desired environment's Python." >&2
    exit 1
fi
echo "Python executable: $PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c 'import sys; print("Python environment :", sys.prefix)'

if ! "$PYTHON_BIN" -c "import torch"; then
    echo "ERROR: torch is unavailable to the selected Python: $PYTHON_BIN" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c "import pymatgen"; then
    echo "ERROR: pymatgen is unavailable to the selected Python: $PYTHON_BIN" >&2
    echo "Install the materials add-on in that Python environment before resubmitting:" >&2
    echo "  $PYTHON_BIN -m pip install -r $PROJECT_DIR/requirements/materials.txt" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c "import transformers"; then
    echo "ERROR: transformers is required for PRISM materials regression." >&2
    echo "Install the base requirements in the selected Python environment." >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c \
    'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_files_only=True)' \
    "$TEXT_MODEL"; then
    echo "ERROR: $TEXT_MODEL is not complete in $HF_HUB_CACHE." >&2
    echo "Stage it from a login node, then resubmit:" >&2
    echo "  $PYTHON_BIN $PROJECT_DIR/tools/stage_materials_text_model.py --model $TEXT_MODEL --cache-dir $HF_HUB_CACHE" >&2
    exit 1
fi

# This trainer is single-process. Pin it to one A100 and give CIF parsing
# several CPU workers. The remaining GPUs on the allocated node are unused.
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

RUN_TAG=${PBS_JOBID%%.*}
OUTPUT_DIR="$PROJECT_DIR/outputs/$MATERIALS_RUN_ID/$RUN_TAG-$RUN_NOTE"
GRAPH_CACHE_DIR="/tmp/$USER/prism-material-graphs-$RUN_TAG"
mkdir -p "$OUTPUT_DIR" "$GRAPH_CACHE_DIR"

echo "Host             : $(hostname)"
echo "PBS job          : $PBS_JOBID"
echo "Output directory : $OUTPUT_DIR"
echo "Graph cache      : $GRAPH_CACHE_DIR"
echo "Trainer          : PRISM"
echo "Target/mode      : $TARGET / $MODE"
echo "Epochs/batch     : $EPOCHS / $BATCH_SIZE"
echo "Hidden/layers    : $HIDDEN_DIM / $GRAPH_LAYERS"
echo "Graph tokens     : $GRAPH_TOKENS"
echo "Cutoff/neighbors : $CUTOFF / $MAX_NEIGHBORS"
echo "Max samples      : $MAX_SAMPLES"
echo "PRISM backbone   : $TEXT_MODEL (train_backbone=$TRAIN_TEXT_BACKBONE)"
echo "HF hub cache     : $HF_HUB_CACHE (offline)"
if [[ "$TRAIN_TEXT_BACKBONE" == "1" ]]; then
    echo "Text backbone LR : $TEXT_LEARNING_RATE"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0

SAMPLE_ARGS=()
if [[ "$MAX_SAMPLES" != "full" ]]; then
    SAMPLE_ARGS=(--max-samples "$MAX_SAMPLES")
fi

TEXT_ARGS=(--text-model "$TEXT_MODEL")
if [[ "$TRAIN_TEXT_BACKBONE" == "1" ]]; then
    TEXT_ARGS+=(--train-text-backbone --text-learning-rate "$TEXT_LEARNING_RATE")
fi

"$PYTHON_BIN" -u tools/train_materials_prism.py \
    --materials-dir Materials \
    --target "$TARGET" \
    --mode "$MODE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --learning-rate "$LEARNING_RATE" \
    --hidden-dim "$HIDDEN_DIM" \
    --graph-layers "$GRAPH_LAYERS" \
    --graph-tokens "$GRAPH_TOKENS" \
    --cutoff "$CUTOFF" \
    --max-neighbors "$MAX_NEIGHBORS" \
    --seed "$SEED" \
    --device cuda \
    --graph-cache-dir "$GRAPH_CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "${TEXT_ARGS[@]}" \
    "${SAMPLE_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"

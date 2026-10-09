#!/bin/bash
# run_evaluator.sh - Launch PRISM Universal Evaluator on Aurora Compute Node
#
# Usage:
#   bash tools/run_evaluator.sh [--mode MODE] [--checkpoint PATH] [--limit N]
#
# Borrowed from: tools/launch_aurora_web.py environment setup

set -e

# --- Defaults ---
PRISM_DIR="/flare/ModCon/ngetty/BaseMM_PRISM"
CHECKPOINT="${PRISM_DIR}/exported/prism-pr3native-step50"
BACKBONE="allenai/OLMo-7B-0724-hf"
MODE="inspect_train"
LIMIT=5
ENV_TARBALL="${PRISM_DIR}/deepspeed_env.tar.gz"
LOCAL_ENV="/tmp/deepspeed_env"

# --- Parse Args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --backbone) BACKBONE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --data-only) DATA_ONLY="--data-only"; shift ;;
        --visualize) VISUALIZE="--visualize"; shift ;;
        --exhaustive) EXHAUSTIVE="--exhaustive"; shift ;;
        --viz_dir) VIZ_DIR="--viz_dir $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== PRISM Universal Evaluator ==="
echo "Checkpoint: $CHECKPOINT"
echo "Backbone:   $BACKBONE"
echo "Mode:       $MODE"
echo "Limit:      $LIMIT"
echo ""

# --- Module Load ---
echo "Loading modules..."
module load frameworks/2025.2.0 2>/dev/null || true
module load hdf5 2>/dev/null || true

# --- Proxy Settings (on compute nodes) - SET EARLY for pip ---
if [[ ! "${HOSTNAME}" =~ aurora-uan ]]; then
    export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
    export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
    export http_proxy="http://proxy.alcf.anl.gov:3128"
    export https_proxy="http://proxy.alcf.anl.gov:3128"
    export ftp_proxy="http://proxy.alcf.anl.gov:3128"
    export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
    echo "Proxy configured for compute node"
fi

# --- HuggingFace Token ---
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN: set"
else
    echo "Warning: HF_TOKEN not set. Some models may fail to load."
fi

# --- Intel XPU Settings ---
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export NUMEXPR_MAX_THREADS=64

# --- Unpack Packed Environment (matches launch_aurora_web.py) ---
ENV_MARKER="${LOCAL_ENV}/env_ready"

# Re-extract if: no marker exists, OR tarball is newer than the marker (env was repacked)
if [ ! -f "$ENV_MARKER" ] || [ "$ENV_TARBALL" -nt "$ENV_MARKER" ]; then
    if [ -f "$ENV_MARKER" ]; then
        echo "Tarball is newer than cached env, re-extracting..."
    fi
    echo "Unpacking environment from $ENV_TARBALL to $LOCAL_ENV..."
    rm -rf "$LOCAL_ENV"
    mkdir -p "$LOCAL_ENV"
    tar -xzf "$ENV_TARBALL" -C "$LOCAL_ENV"
    touch "$ENV_MARKER"
    echo "Environment unpacked."
else
    echo "Environment already unpacked at $LOCAL_ENV"
fi

# --- Model Staging to /tmp (matches launch_aurora_web.py lines 230-249) ---
SHARED_HF_HOME="/flare/ModCon/sandeep/hub"
LOCAL_HF_HOME="/tmp/huggingface/hub"
# Convert backbone ID to HF cache directory format: allenai/OLMo-7B -> models--allenai--OLMo-7B
MODEL_DIR="models--$(echo $BACKBONE | sed 's|/|--|g')"
MODEL_MARKER="${LOCAL_HF_HOME}/model_ready"

mkdir -p "$LOCAL_HF_HOME"

if [ ! -f "$MODEL_MARKER" ]; then
    if [ -d "${SHARED_HF_HOME}/${MODEL_DIR}" ]; then
        echo "Staging model ${MODEL_DIR} to /tmp..."
        cp -r "${SHARED_HF_HOME}/${MODEL_DIR}" "${LOCAL_HF_HOME}/"
        touch "$MODEL_MARKER"
        echo "Model staged."
    else
        echo "Warning: Shared model not found at ${SHARED_HF_HOME}/${MODEL_DIR}"
        echo "Will attempt to download (may be slow)..."
        touch "$MODEL_MARKER"  # Mark as attempted
    fi
else
    echo "Model already staged at ${LOCAL_HF_HOME}/${MODEL_DIR}"
fi

# Set HF_HOME to use staged model
export HF_HOME="/tmp/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="/tmp/hf_datasets"
mkdir -p "$HF_DATASETS_CACHE"
echo "HF_HOME: $HF_HOME"

# --- Activate Packed Environment ---
echo "Activating packed environment..."
export PYTHONNOUSERSITE=1
source "${LOCAL_ENV}/bin/activate"

# --- Verify webdataset is in the packed env ---
python -c "import webdataset" 2>/dev/null || {
    echo "ERROR: webdataset missing from packed env. Rebuild it."
    exit 1
}

# --- Run Evaluator ---
cd "$PRISM_DIR"
echo ""
echo "Running evaluator..."
python tools/universal_evaluator.py \
    --checkpoint "$CHECKPOINT" \
    --backbone "$BACKBONE" \
    --mode "$MODE" \
    --limit "$LIMIT" \
    $DATA_ONLY $VISUALIZE $EXHAUSTIVE $VIZ_DIR

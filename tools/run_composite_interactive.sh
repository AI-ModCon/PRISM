#!/bin/bash
# Interactive test script for COMPOSITE DDP mode on Aurora
# Usage: ssh <node> bash /path/to/run_composite_interactive.sh
#
# This script is meant to be run directly on an interactive compute node.
# It sets up all environment variables and launches 6 ranks with COMPOSITE mode.
#
# Hang detection: Set COMPOSITE_TIMEOUT_SECS to enable automatic hang detection.
# The script monitors log output for training progress. If no new output appears
# within the timeout, mpiexec is killed and cleanup is performed automatically.
# Default: 300 seconds (5 minutes). Set to 0 to disable.
#
# Environment variables:
#   COMPOSITE_BS           - Batch size per rank (default: 1)
#   COMPOSITE_MAX_STEPS    - Max training steps (default: 5)
#   COMPOSITE_GRAD_ACCUM   - Gradient accumulation steps (default: 6)
#   COMPOSITE_TIMEOUT_SECS - Hang detection timeout in seconds (default: 300, 0=disable)
#   DDP_DEBUG              - Enable per-microbatch debug prints (default: 0)
#   LOG_EVERY_N_STEPS      - Log frequency (default: 10)

set -euo pipefail

PROJ_DIR="/lus/flare/projects/ModCon/ngetty/BaseMM_PRISM"
DAOS_MOUNT="/tmp/ngetty/AuroraGPT/prism_training_data"
DAOS_MODELS_MOUNT="/tmp/ngetty/AuroraGPT/prism_models"
HOSTNAME_HSN="$(hostname).hsn.cm.aurora.alcf.anl.gov"
OUTPUT_DIR="${PROJ_DIR}/outputs/composite-interactive/$(date +%Y-%m-%d/%H-%M-%S)"

mkdir -p "${OUTPUT_DIR}"

echo "=== COMPOSITE DDP Interactive Test ==="
echo "Node: $(hostname)"
echo "Output: ${OUTPUT_DIR}"
echo "DAOS data: ${DAOS_MOUNT}"
echo ""

# --- CCL / Network ---
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export MPICH_GPU_SUPPORT_ENABLED=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=4
export CCL_ALLREDUCE=ring
export CCL_REDUCE_SCATTER=ring
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072

# --- General ---
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS=16
export OMP_NUM_THREADS=16
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export USE_NATIVE_DDP=1
export TMPDIR=/tmp

# --- Bucketing ---
export USE_BUCKETED_COLLATOR=1
export MAX_SEQ_LENGTH=1024

# --- DDP config ---
export DIST_STRATEGY=ddp
export DDP_BUCKET_CAP_MB=${DDP_BUCKET_CAP_MB:-50}
export GRAD_CKPT_FREQ=0

# --- DAOS ---
export DAOS_POOL=AuroraGPT
export DAOS_CONT=prism_training_data
export DAOS_MODELS_CONT=prism_models
export DAOS_MOUNT="${DAOS_MOUNT}"
export DAOS_MODELS_MOUNT="${DAOS_MODELS_MOUNT}"
export DAOS_MODELS_AVAILABLE=1
export NUM_NODES=1
export USE_MULTI_DATASET=1
export DATASET_GROUPS=projector
export DATASET_CONFIG="${PROJ_DIR}/src/conf/data/daos_datasets.yaml"
export DATASET_PROPORTIONS=""

# --- DAOS mounting (ensure containers are mounted) ---
export D_AGENT_DRPC_DIR=/run/daos_agent_oneScratch
module use /soft/modulefiles 2>/dev/null
module load daos 2>/dev/null

# Mount models container if not already mounted
if [ ! -d "${DAOS_MODELS_MOUNT}/hub" ]; then
    echo "Mounting DAOS models container..."
    mkdir -p "${DAOS_MODELS_MOUNT}"
    dfuse -m "${DAOS_MODELS_MOUNT}" --pool "${DAOS_POOL}" --cont "${DAOS_MODELS_CONT}" --disable-wb-cache 2>&1 || true
    sleep 2
fi
if [ -d "${DAOS_MODELS_MOUNT}/hub" ]; then
    echo "DAOS models mount OK: $(ls ${DAOS_MODELS_MOUNT}/hub/ | wc -l) models cached"
else
    echo "WARNING: DAOS models mount FAILED — HF model loading may fail"
fi

# Mount data container if not already mounted
if [ ! -d "${DAOS_MOUNT}/pixmo" ]; then
    echo "Mounting DAOS data container..."
    mkdir -p "${DAOS_MOUNT}"
    dfuse -m "${DAOS_MOUNT}" --pool "${DAOS_POOL}" --cont "${DAOS_CONT}" --disable-wb-cache 2>&1 || true
    sleep 2
fi
if [ -d "${DAOS_MOUNT}/pixmo" ]; then
    echo "DAOS data mount OK: $(ls ${DAOS_MOUNT}/ | head -5 | tr '\n' ' ')"
else
    echo "WARNING: DAOS data mount FAILED — training data unavailable"
fi

# Create HF cache symlink: HF_HOME/hub -> DAOS models hub
# Some HF code resolves models via HF_HOME/hub/ rather than HF_HUB_CACHE
mkdir -p /tmp/huggingface
ln -sfn "${DAOS_MODELS_MOUNT}/hub" /tmp/huggingface/hub
echo "HF cache symlink: /tmp/huggingface/hub -> ${DAOS_MODELS_MOUNT}/hub"

# --- Master addr ---
export MASTER_ADDR="${HOSTNAME_HSN}"
export MASTER_PORT=$((20000 + RANDOM % 20000))

# --- DDP diagnostic flag ---
# Leave PRISM_DDP_FIND_UNUSED unset to let train.py auto-detect based on
# ZE_FLAT_DEVICE_HIERARCHY. Set to 0 or 1 to override auto-detection.
# export PRISM_DDP_FIND_UNUSED=1  # Uncomment to force find_unused_parameters=True

# --- Proxy (needed for HF downloads if cache miss) ---
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"

echo "PRISM_DDP_FIND_UNUSED=${PRISM_DDP_FIND_UNUSED:-<auto-detect>}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo ""

# --- Venv staging (extract to /tmp once) ---
ENV_TARBALL="${PROJ_DIR}/deepspeed_env.tar.gz"
LOCAL_ENV="/tmp/deepspeed_env"
if [ ! -f "${LOCAL_ENV}/bin/activate" ]; then
    echo "Extracting venv to ${LOCAL_ENV}..."
    rm -rf "${LOCAL_ENV}"
    mkdir -p "${LOCAL_ENV}"
    # --strip-components=1 removes the .venv-deepspeed/ prefix from the tarball
    tar --strip-components=1 -xzf "${ENV_TARBALL}" -C "${LOCAL_ENV}"
    # Fix VIRTUAL_ENV to point to /tmp (avoid Lustre metadata lookups at runtime)
    sed -i 's|/lus/flare/projects/ModCon/ngetty/BaseMM_PRISM/.venv-deepspeed|/tmp/deepspeed_env|g' "${LOCAL_ENV}/bin/activate"
    echo "Venv extraction complete"
else
    echo "Venv already extracted at ${LOCAL_ENV}"
fi

# Ensure transformers 4.57.6+ is available (needed for olmo3 model type)
if [ ! -d "${LOCAL_ENV}/lib/python3.10/site-packages/transformers-4.57.6.dist-info" ]; then
    echo "Installing transformers 4.57.6 into local venv..."
    export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
    export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
    module load frameworks 2>/dev/null
    source "${LOCAL_ENV}/bin/activate"
    pip install --no-deps "transformers==4.57.6" --target "${LOCAL_ENV}/lib/python3.10/site-packages/" --quiet 2>&1
    echo "transformers 4.57.6 installed"
fi

# --- Hang detection config ---
TIMEOUT_SECS=${COMPOSITE_TIMEOUT_SECS:-300}
LOG_FILE="${OUTPUT_DIR}/run.log"

# --- Build the mpiexec command ---
MPIEXEC_CMD='mpiexec -n 6 -ppn 6 --no-vni --cpu-bind depth --depth 16 bash -lc '"'"'
  module use /soft/modulefiles
  module load frameworks 2>/dev/null
  module load daos 2>/dev/null

  # CRITICAL: Set COMPOSITE hierarchy INSIDE the mpiexec worker.
  # bash -lc resets env, so outer shell exports are lost.
  export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

  # Forward diagnostic/tuning env vars from outer shell into the worker.
  # Only export if set in outer env (unset vars use code defaults).
  [ -n "${PRISM_DDP_FIND_UNUSED+x}" ] && export PRISM_DDP_FIND_UNUSED="${PRISM_DDP_FIND_UNUSED}"
  [ -n "${PRISM_DDP_GRAD_BUCKET_VIEW+x}" ] && export PRISM_DDP_GRAD_BUCKET_VIEW="${PRISM_DDP_GRAD_BUCKET_VIEW}"
  export DDP_DEBUG="${DDP_DEBUG:-0}"
  export DDP_DEBUG_STEPS="${DDP_DEBUG_STEPS:-3}"
  export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
  export DEBUG_SYNC="${DEBUG_SYNC:-0}"
  export DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-50}"

  # Rank variables
  export LOCAL_WORLD_SIZE=${PMI_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-6}}
  export WORLD_SIZE=${PMI_SIZE:-${PALS_SIZE:-6}}
  export RANK=${PMI_RANK:-${PALS_RANKID:-0}}
  export LOCAL_RANK=${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}
  export ZE_AFFINITY_MASK=$LOCAL_RANK

  # Activate venv (user site-packages needed for torch_geometric, etc.)
  # Note: torch loads from frameworks (2.8.0a0), not user site-packages,
  # because the venv site-packages take priority and the user site-packages
  # only contain torch metadata (dist-info) without the actual module.
  source /tmp/deepspeed_env/bin/activate

  # HF cache — resolve via /tmp/huggingface/hub symlink (matches launch_aurora_daos.py)
  export HF_HOME="/tmp/huggingface"
  export TRANSFORMERS_CACHE="/tmp/huggingface/hub"
  export HF_HUB_CACHE="/tmp/huggingface/hub"
  export HF_DATASETS_CACHE="/tmp/huggingface/datasets"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  cd '"${PROJ_DIR}"'
  python src/train.py \
    model=prism_image_only \
    model.backbone_id=allenai/Olmo-3-1025-7B \
    model.d_img=768 \
    model.projector_norm_mode=none \
    model.projector_modality_embed_pos=none \
    training=molmo_stage1 \
    training.batch_size=${COMPOSITE_BS:-1} \
    training.gradient_accumulation_steps=${COMPOSITE_GRAD_ACCUM:-6} \
    training.max_steps=${COMPOSITE_MAX_STEPS:-5} \
    training.eval_every_n_steps=${COMPOSITE_MAX_STEPS:-5} \
    training.save_every_n_steps=0 \
    training.viz_every_n_steps=0 \
    training.learning_rate=2e-5 \
    training.weight_decay=0.01 \
    training.device=xpu \
    exp.id=COMPOSITE-INTERACTIVE-TEST \
    hydra.run.dir='"${OUTPUT_DIR}"'
'"'"

# --- Launch with hang detection ---
if [ "${TIMEOUT_SECS}" -eq 0 ]; then
  echo "[HANG-DETECT] Disabled (COMPOSITE_TIMEOUT_SECS=0)"
  eval "${MPIEXEC_CMD}" 2>&1 | tee "${LOG_FILE}"
  EXIT_CODE=$?
else
  echo "[HANG-DETECT] Enabled: ${TIMEOUT_SECS}s timeout for log activity"
  echo "[HANG-DETECT] Watching for new output in ${LOG_FILE}"
  echo ""

  # Launch mpiexec in background, piping output to log file
  eval "${MPIEXEC_CMD}" > "${LOG_FILE}" 2>&1 &
  MPIEXEC_PID=$!

  # Monitor loop: tail the log in the background for user visibility,
  # and check log file modification time for hang detection.
  tail -f "${LOG_FILE}" 2>/dev/null &
  TAIL_PID=$!

  LAST_SIZE=0
  STALL_START=""
  EXIT_CODE=0

  while kill -0 "${MPIEXEC_PID}" 2>/dev/null; do
    sleep 5

    # Check if log file has grown
    if [ -f "${LOG_FILE}" ]; then
      CURRENT_SIZE=$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo "0")
    else
      CURRENT_SIZE=0
    fi

    if [ "${CURRENT_SIZE}" != "${LAST_SIZE}" ]; then
      # Log is growing — reset stall timer
      LAST_SIZE="${CURRENT_SIZE}"
      STALL_START=""
    else
      # Log is not growing — start or check stall timer
      if [ -z "${STALL_START}" ]; then
        STALL_START=$(date +%s)
      else
        NOW=$(date +%s)
        STALL_DURATION=$((NOW - STALL_START))
        if [ "${STALL_DURATION}" -ge "${TIMEOUT_SECS}" ]; then
          echo ""
          echo "=============================================="
          echo "[HANG-DETECT] TIMEOUT: No log output for ${STALL_DURATION}s (limit: ${TIMEOUT_SECS}s)"
          echo "[HANG-DETECT] Killing mpiexec (PID ${MPIEXEC_PID}) and all python3 processes..."
          echo "=============================================="

          # Kill mpiexec and all python workers
          kill -9 "${MPIEXEC_PID}" 2>/dev/null || true
          pkill -9 -f "python.*src/train.py" 2>/dev/null || true
          sleep 3

          # Double-check cleanup
          pkill -9 python3 2>/dev/null || true
          sleep 2

          EXIT_CODE=124  # Same as timeout(1) exit code
          break
        fi
      fi
    fi
  done

  # Wait for mpiexec to fully exit (if it finished naturally)
  wait "${MPIEXEC_PID}" 2>/dev/null || true

  # Stop the tail follower
  kill "${TAIL_PID}" 2>/dev/null || true
  wait "${TAIL_PID}" 2>/dev/null || true
fi

echo ""
if [ "${EXIT_CODE}" -eq 124 ]; then
  echo "=== HANG DETECTED — Run killed after ${TIMEOUT_SECS}s stall. Log: ${LOG_FILE} ==="
elif [ "${EXIT_CODE}" -eq 0 ]; then
  echo "=== Run complete. Log: ${LOG_FILE} ==="
else
  echo "=== Run FAILED (exit code ${EXIT_CODE}). Log: ${LOG_FILE} ==="
fi

exit "${EXIT_CODE}"

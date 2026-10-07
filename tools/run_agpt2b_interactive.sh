#!/bin/bash
# Interactive test script for AuroraGPT-2B projector training on Aurora
# Usage: ssh <node> bash /path/to/run_agpt2b_interactive.sh
#
# This script runs FLAT mode (12 ranks/node, 1 per tile, ~64GB each).
# Used for debugging the UR_RESULT_ERROR_OUT_OF_RESOURCES crash.
#
# Environment variables:
#   AGPT2B_BS              - Batch size per rank (default: 16)
#   AGPT2B_MAX_STEPS       - Max training steps (default: 100)
#   AGPT2B_GRAD_ACCUM      - Gradient accumulation steps (default: 2)
#   AGPT2B_ATTN_IMPL       - Attention implementation: eager, sdpa (default: eager)
#   AGPT2B_TIMEOUT_SECS    - Hang detection timeout (default: 300, 0=disable)
#   AGPT2B_VIZ_INTERVAL    - Viz interval (default: 0 = disabled, avoids extra fwd passes)

set -euo pipefail

PROJ_DIR="/lus/flare/projects/ModCon/ngetty/BaseMM_PRISM"
DAOS_MOUNT="/tmp/ngetty/AuroraGPT/prism_training_data"
DAOS_MODELS_MOUNT="/tmp/ngetty/AuroraGPT/prism_models"
HOSTNAME_HSN="$(hostname).hsn.cm.aurora.alcf.anl.gov"
OUTPUT_DIR="${PROJ_DIR}/outputs/agpt2b-interactive/$(date +%Y-%m-%d/%H-%M-%S)"

# Config from env vars
BS=${AGPT2B_BS:-16}
MAX_STEPS=${AGPT2B_MAX_STEPS:-100}
GRAD_ACCUM=${AGPT2B_GRAD_ACCUM:-2}
ATTN_IMPL=${AGPT2B_ATTN_IMPL:-eager}
TIMEOUT_SECS=${AGPT2B_TIMEOUT_SECS:-300}
VIZ_INTERVAL=${AGPT2B_VIZ_INTERVAL:-0}

mkdir -p "${OUTPUT_DIR}"

echo "=== AuroraGPT-2B Interactive Test ==="
echo "Node: $(hostname)"
echo "Output: ${OUTPUT_DIR}"
echo "BS=${BS}, grad_accum=${GRAD_ACCUM}, max_steps=${MAX_STEPS}"
echo "attn_implementation=${ATTN_IMPL}"
echo "viz_interval=${VIZ_INTERVAL}"
echo ""

# --- CCL / Network ---
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
export DDP_BUCKET_CAP_MB=50
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

# --- DAOS mounting ---
export D_AGENT_DRPC_DIR=/run/daos_agent_oneScratch
module use /soft/modulefiles 2>/dev/null
module load daos 2>/dev/null

if [ ! -d "${DAOS_MODELS_MOUNT}/hub" ]; then
    echo "Mounting DAOS models container..."
    mkdir -p "${DAOS_MODELS_MOUNT}"
    dfuse -m "${DAOS_MODELS_MOUNT}" --pool "${DAOS_POOL}" --cont "${DAOS_MODELS_CONT}" --disable-wb-cache 2>&1 || true
    sleep 2
fi
if [ -d "${DAOS_MODELS_MOUNT}/hub" ]; then
    echo "DAOS models mount OK: $(ls ${DAOS_MODELS_MOUNT}/hub/ | wc -l) models cached"
else
    echo "WARNING: DAOS models mount FAILED"
fi

if [ ! -d "${DAOS_MOUNT}/pixmo" ]; then
    echo "Mounting DAOS data container..."
    mkdir -p "${DAOS_MOUNT}"
    dfuse -m "${DAOS_MOUNT}" --pool "${DAOS_POOL}" --cont "${DAOS_CONT}" --disable-wb-cache 2>&1 || true
    sleep 2
fi
if [ -d "${DAOS_MOUNT}/pixmo" ]; then
    echo "DAOS data mount OK: $(ls ${DAOS_MOUNT}/ | head -5 | tr '\n' ' ')"
else
    echo "WARNING: DAOS data mount FAILED"
fi

mkdir -p /tmp/huggingface
ln -sfn "${DAOS_MODELS_MOUNT}/hub" /tmp/huggingface/hub
echo "HF cache symlink: /tmp/huggingface/hub -> ${DAOS_MODELS_MOUNT}/hub"

# --- Master addr ---
export MASTER_ADDR="${HOSTNAME_HSN}"
export MASTER_PORT=$((20000 + RANDOM % 20000))

# --- Proxy ---
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo ""

# --- Venv staging ---
ENV_TARBALL="${PROJ_DIR}/deepspeed_env.tar.gz"
LOCAL_ENV="/tmp/deepspeed_env"
if [ ! -f "${LOCAL_ENV}/bin/activate" ]; then
    echo "Extracting venv to ${LOCAL_ENV}..."
    rm -rf "${LOCAL_ENV}"
    mkdir -p "${LOCAL_ENV}"
    tar --strip-components=1 -xzf "${ENV_TARBALL}" -C "${LOCAL_ENV}"
    sed -i 's|/lus/flare/projects/ModCon/ngetty/BaseMM_PRISM/.venv-deepspeed|/tmp/deepspeed_env|g' "${LOCAL_ENV}/bin/activate"
    echo "Venv extraction complete"
else
    echo "Venv already extracted at ${LOCAL_ENV}"
fi

if [ ! -d "${LOCAL_ENV}/lib/python3.10/site-packages/transformers-4.57.6.dist-info" ]; then
    echo "Installing transformers 4.57.6 into local venv..."
    module load frameworks 2>/dev/null
    source "${LOCAL_ENV}/bin/activate"
    pip install --no-deps "transformers==4.57.6" --target "${LOCAL_ENV}/lib/python3.10/site-packages/" --quiet 2>&1
    echo "transformers 4.57.6 installed"
fi

# --- Backbone path ---
BACKBONE_PATH="/lus/flare/projects/AuroraGPT/evaluation/models/safetensors/AuroraGPT-2B-ws3072-ds-stage0-nl12-hs2048-mb1-seq8192-gb6144-sp1-pp1-tp1-bf16-optsophiag-lr2.17e-5-lwf0.05_ntok7064B_tokHF_tmgoogle_gemma-7b_flash/global_step140300"

LOG_FILE="${OUTPUT_DIR}/run.log"

# --- Build mpiexec command (FLAT mode: 12 ranks, 1 per tile) ---
# CPU binding: pin each rank to its optimal core set
CPU_BIND="list:4:9:14:19:20:25:56:61:66:71:74:79"

MPIEXEC_CMD='mpiexec -n 12 -ppn 12 --cpu-bind '"${CPU_BIND}"' bash -lc '"'"'
  module use /soft/modulefiles
  module load frameworks 2>/dev/null
  module load daos 2>/dev/null

  # Rank variables
  export LOCAL_WORLD_SIZE=${PMI_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-12}}
  export WORLD_SIZE=${PMI_SIZE:-${PALS_SIZE:-12}}
  export RANK=${PMI_RANK:-${PALS_RANKID:-0}}
  export LOCAL_RANK=${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}

  # FLAT mode: each tile is a separate device (xpu:0 through xpu:11)
  # CRITICAL: Set ZE_AFFINITY_MASK so each rank sees ONLY its assigned tile.
  # Without this, all 12 ranks initialize Level Zero contexts for all 12 tiles,
  # creating 144 L0 contexts vs 12. This exhausts kernel dispatch resources
  # causing UR_RESULT_ERROR_OUT_OF_RESOURCES.
  export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
  export ZE_AFFINITY_MASK=$LOCAL_RANK

  source /tmp/deepspeed_env/bin/activate

  export HF_HOME="/tmp/huggingface"
  export TRANSFORMERS_CACHE="/tmp/huggingface/hub"
  export HF_HUB_CACHE="/tmp/huggingface/hub"
  export HF_DATASETS_CACHE="/tmp/huggingface/datasets"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  cd '"${PROJ_DIR}"'
  python src/train.py \
    model=prism_auroragpt_2b \
    model.backbone_id='"${BACKBONE_PATH}"' \
    model.d_img=768 \
    model.attn_implementation='"${ATTN_IMPL}"' \
    training=projector_only \
    training.batch_size='"${BS}"' \
    training.gradient_accumulation_steps='"${GRAD_ACCUM}"' \
    training.max_steps='"${MAX_STEPS}"' \
    training.eval_every_n_steps='"${MAX_STEPS}"' \
    training.save_every_n_steps=0 \
    training.viz_every_n_steps='"${VIZ_INTERVAL}"' \
    training.learning_rate=1e-3 \
    training.weight_decay=0.0 \
    training.warmup_steps=200 \
    training.device=xpu \
    exp.id=AGPT2B-INTERACTIVE-TEST \
    hydra.run.dir='"${OUTPUT_DIR}"'
'"'"

# --- Launch with hang detection ---
if [ "${TIMEOUT_SECS}" -eq 0 ]; then
  echo "[HANG-DETECT] Disabled"
  eval "${MPIEXEC_CMD}" 2>&1 | tee "${LOG_FILE}"
  EXIT_CODE=$?
else
  echo "[HANG-DETECT] Enabled: ${TIMEOUT_SECS}s timeout"
  echo ""

  eval "${MPIEXEC_CMD}" > "${LOG_FILE}" 2>&1 &
  MPIEXEC_PID=$!

  tail -f "${LOG_FILE}" 2>/dev/null &
  TAIL_PID=$!

  LAST_SIZE=0
  STALL_START=""
  EXIT_CODE=0

  while kill -0 "${MPIEXEC_PID}" 2>/dev/null; do
    sleep 5

    if [ -f "${LOG_FILE}" ]; then
      CURRENT_SIZE=$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo "0")
    else
      CURRENT_SIZE=0
    fi

    if [ "${CURRENT_SIZE}" != "${LAST_SIZE}" ]; then
      LAST_SIZE="${CURRENT_SIZE}"
      STALL_START=""
    else
      if [ -z "${STALL_START}" ]; then
        STALL_START=$(date +%s)
      else
        NOW=$(date +%s)
        STALL_DURATION=$((NOW - STALL_START))
        if [ "${STALL_DURATION}" -ge "${TIMEOUT_SECS}" ]; then
          echo ""
          echo "=============================================="
          echo "[HANG-DETECT] TIMEOUT: No output for ${STALL_DURATION}s"
          echo "[HANG-DETECT] Killing mpiexec and python3..."
          echo "=============================================="

          kill -9 "${MPIEXEC_PID}" 2>/dev/null || true
          pkill -9 -f "python.*src/train.py" 2>/dev/null || true
          sleep 3
          pkill -9 python3 2>/dev/null || true
          sleep 5

          EXIT_CODE=124
          break
        fi
      fi
    fi
  done

  wait "${MPIEXEC_PID}" 2>/dev/null || true
  kill "${TAIL_PID}" 2>/dev/null || true
  wait "${TAIL_PID}" 2>/dev/null || true
fi

echo ""
STATUS_FILE="${OUTPUT_DIR}/status"
if [ "${EXIT_CODE}" -eq 124 ]; then
  MSG="HANG_DETECTED after ${TIMEOUT_SECS}s stall"
  echo "=== HANG DETECTED — killed after ${TIMEOUT_SECS}s stall. Log: ${LOG_FILE} ==="
elif [ "${EXIT_CODE}" -eq 0 ]; then
  MSG="SUCCESS ${MAX_STEPS} steps"
  echo "=== Run complete (${MAX_STEPS} steps). Log: ${LOG_FILE} ==="
else
  MSG="FAILED exit_code=${EXIT_CODE}"
  echo "=== Run FAILED (exit code ${EXIT_CODE}). Log: ${LOG_FILE} ==="
fi

# Write machine-readable status for polling
echo "${MSG}" > "${STATUS_FILE}"
echo "LOG=${LOG_FILE}" >> "${STATUS_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}" >> "${STATUS_FILE}"

exit "${EXIT_CODE}"

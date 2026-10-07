#!/usr/bin/env bash
# Polaris-specific launcher (PBS/mpiexec/cudatoolkit-standalone/argonne_tpc
# allocation) -- unlike this repo's other, Aurora-first launchers.
#PBS -N bioreason_projector_kegg
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -l filesystems=home:eagle
#PBS -A argonne_tpc
#PBS -q preemptable

set -euo pipefail

PROJ_DIR="/home/abalaji/projects/modcon/genome/BaseMM_PRISM"
OUTPUT_BASE="/lus/eagle/projects/argonne_tpc/abalaji/modcon/genome/output"
RUN_ID="GENOME-PROJECTOR-KEGG"
RUN_VARIANT="1B"

# Set to a checkpoint directory to resume this same stage (optimizer/scheduler/step
# included), e.g. after preemption. Leave empty to start a fresh run.
#   RESUME_CKPT="${OUTPUT_BASE}/GENOME-PROJECTOR-1B/2026-07-01/12-00-00/checkpoints/step_2500"
RESUME_CKPT=""

cd "${PROJ_DIR}"

# --- Cuda and NCCL (Polaris) ---
ml use /soft/modulefiles
ml --ignore_cache load cudatoolkit-standalone/12.6.1

# --- Conda env ---
ml --ignore_cache load conda
conda activate /lus/eagle/projects/argonne_tpc/abalaji/conda_env/prism

# --- Proxy ---
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="127.0.0.1,admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

# --- HuggingFace cache ---
export HF_HOME="/lus/eagle/projects/argonne_tpc/abalaji/model_weights"
export HF_DATASETS_CACHE="/lus/eagle/projects/argonne_tpc/abalaji/model_weights"
export TRANSFORMERS_CACHE="/lus/eagle/projects/argonne_tpc/abalaji/model_weights/hub"
export HF_HUB_CACHE="/lus/eagle/projects/argonne_tpc/abalaji/model_weights/hub"

# --- W&B ---
# No WANDB_API_KEY here -- a plaintext key was committed here previously
# and has been removed; rotate that key on wandb.ai if it hasn't been
# already. wandb picks up credentials from ~/.netrc (machine api.wandb.ai)
# or an already-exported WANDB_API_KEY in the calling shell automatically
# -- nothing to set here.

# --- NCCL (Polaris) ---
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:${LD_LIBRARY_PATH:-}

# --- General ---
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=64

# --- Distributed ---
# Falls back to the current host when run directly on an already-allocated
# interactive node (no PBS_NODEFILE, e.g. after `qsub -I`).
if [ -n "${PBS_NODEFILE:-}" ]; then
  NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
  MASTER_ADDR=$(sort -u "$PBS_NODEFILE" | head -n 1)
else
  NNODES=1
  MASTER_ADDR=$(hostname)
fi
NPERNODE=4  # 4 GPUs per Polaris node
NPROCS=$((NNODES * NPERNODE))

export MASTER_ADDR
export MASTER_PORT=$((20000 + RANDOM % 20000))
export WORLD_SIZE=${NPROCS}

echo "=== BioReason Projector Training (Stage 1) ==="
echo "Node: $(hostname)"
echo "PBS_NODEFILE: ${PBS_NODEFILE:-<none, interactive node>}"
echo "NNODES=${NNODES} NPERNODE=${NPERNODE} NPROCS=${NPROCS}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Output base: ${OUTPUT_BASE}"
echo "Resume ckpt: ${RESUME_CKPT:-<none, fresh run>}"
echo ""

# Unquoted splice below: when empty, this contributes zero words to the command
# line (not an empty-string arg), so train.py/Hydra sees a normal 3-override
# invocation when not resuming.
RESUME_OVERRIDE=""
if [ -n "${RESUME_CKPT}" ]; then
  RESUME_OVERRIDE="training.resume_from_checkpoint=${RESUME_CKPT}"
fi

mpiexec -n "${NPROCS}" -ppn "${NPERNODE}" \
  --envall \
  bash -lc '
    export RANK=${PMI_RANK:-${PALS_RANKID:-0}}
    export WORLD_SIZE=${PMI_SIZE:-${PALS_SIZE:-'"${NPROCS}"'}}
    export LOCAL_RANK=${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}
    export LOCAL_WORLD_SIZE=${PMI_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-'"${NPERNODE}"'}}
    export MASTER_ADDR="'"${MASTER_ADDR}"'"
    export MASTER_PORT="'"${MASTER_PORT}"'"

    cd '"${PROJ_DIR}"'
    python -u src/train.py \
      training=bioreason_projector \
      exp.id='"${RUN_ID}"' \
      exp.variant='"${RUN_VARIANT}"' \
      '"${RESUME_OVERRIDE}"'
  '

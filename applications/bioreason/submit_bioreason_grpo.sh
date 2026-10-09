#!/usr/bin/env bash
# Polaris-specific launcher (PBS/mpiexec/cudatoolkit-standalone/argonne_tpc
# allocation) -- unlike this repo's other, Aurora-first launchers.
#PBS -N prism-grpo
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -l filesystems=home:eagle
#PBS -A argonne_tpc
#PBS -q preemptable

set -euo pipefail

PROJ_DIR="/home/abalaji/projects/modcon/genome/BaseMM_PRISM"
OUTPUT_BASE="/lus/eagle/projects/argonne_tpc/abalaji/modcon/genome/output"
RUN_ID="GENOME-GRPO"
RUN_VARIANT="1B"
STAGE1_CKPT="/lus/eagle/projects/argonne_tpc/abalaji/modcon/genome/output/GENOME-SFT-1B/2026-06-29/07-41-10/checkpoints/step_8000"

cd "${PROJ_DIR}"

# --- Modules ---
ml use /soft/modulefiles
ml --ignore_cache load cudatoolkit-standalone/12.6.1
module --ignore_cache load conda
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
echo "=== BioReason GRPO Training (Zone D / Stage 3) ==="
echo "Node:           $(hostname)"
echo "Stage 1 ckpt:   ${STAGE1_CKPT}"
echo "Output base:    ${OUTPUT_BASE}"
echo ""

# GRPO is single-process: sequential generation + policy update, no AllReduce needed.
# Running 4 MPI ranks caused each rank to deepcopy a reference model onto the same GPU,
# exhausting VRAM (4× model footprint on one GPU). Use a single process on GPU 0.
export CUDA_VISIBLE_DEVICES=0
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

cd "${PROJ_DIR}"
python -u src/train.py \
  training=bioreason_grpo \
  exp.id="${RUN_ID}" \
  exp.variant="${RUN_VARIANT}" \
  training.resume_weights_only="${STAGE1_CKPT}"

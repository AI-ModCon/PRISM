#!/usr/bin/env bash
#PBS -N prism-grpo-smoke
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -l filesystems=home:eagle
#PBS -A argonne_tpc
#PBS -q debug
#
# Smoke test for the GRPO LoRA rank-mismatch fix (SFT rank 32 -> GRPO rank 16).
# Runs 5 steps with G=2 generations to verify:
#   1. SFT LoRA adapter (rank 32) merges cleanly into base backbone weights
#   2. Fresh GRPO LoRA layers (rank 16) attach on top and pass the rank assertion
#      added in ZoneDTrainer.__init__ (src/training/trainer_zone_d_grpo.py)
#   3. Forward/backward pass completes without OOM or shape errors
#   4. Reward functions fire and produce non-NaN scores for all 5 steps
#
# Usage:
#   qsub scripts/smoke_test_grpo.sh          # submit as a PBS job
#   bash scripts/smoke_test_grpo.sh          # run directly on an already-allocated compute node

set -euo pipefail

PROJ_DIR="/home/abalaji/projects/modcon/genome/BaseMM_PRISM"
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

# --- W&B offline for smoke test (no network needed) ---
export WANDB_MODE=offline
export WANDB_SILENT=true

# --- NCCL (Polaris) ---
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLENT_ENABLE=1
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:${LD_LIBRARY_PATH:-}

# --- General ---
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=8

# --- Single GPU (GRPO is single-process) ---
export CUDA_VISIBLE_DEVICES=0
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

echo "================================================================"
echo " GRPO SMOKE TEST"
echo " Node:        $(hostname)"
echo " GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo " Checkpoint:  ${STAGE1_CKPT}"
echo " Steps:       5   (max_steps override)"
echo " Generations: 2   (grpo_num_generations override)"
echo "================================================================"
echo ""

SMOKE_LOG="/tmp/prism_grpo_smoke_${PBS_JOBID:-$$}.log"

python -u train.py \
  training=bioreason_grpo \
  exp.id="GRPO-SMOKE-TEST" \
  exp.variant="1B" \
  training.resume_weights_only="${STAGE1_CKPT}" \
  training.max_steps=5 \
  training.grpo_num_generations=2 \
  training.grpo_max_completion_length=128 \
  training.save_every_n_steps=9999 \
  training.eval_every_n_steps=9999 \
  +training.log_every_n_steps=1 \
  wandb.mode=offline \
  2>&1 | tee "${SMOKE_LOG}"

echo ""
echo "================================================================"
echo " LORA LOAD / MERGE / VERIFY"
echo "================================================================"
grep -E "\[ZoneD\] (Merged|Verified|WARNING)" "${SMOKE_LOG}" || echo "(no [ZoneD] setup lines found — check log above for errors)"

echo ""
echo "================================================================"
echo " REWARD SCORES (5 iterations)"
echo "================================================================"
grep -oE "\[ZoneD\] step=[0-9]+ loss=[-0-9.]+ avg_reward=[0-9.]+" "${SMOKE_LOG}" || echo "(no per-step reward lines found — training may have failed before step 1)"

echo ""
echo "================================================================"
echo " SMOKE TEST COMPLETE — full log: ${SMOKE_LOG}"
echo "================================================================"

#!/usr/bin/env bash
#PBS -N prism-eval-kegg
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -l filesystems=home:eagle
#PBS -A argonne_tpc
#PBS -q preemptable

set -euo pipefail

PROJ_DIR="/home/abalaji/projects/modcon/genome/BaseMM_PRISM"
CHECKPOINT_DIR="/lus/eagle/projects/argonne_tpc/abalaji/modcon/genome/output/GENOME-SFT-1B/2026-06-29/07-41-10/checkpoints/step_8000"
# SFT with sample weigting and answer structure <answer>: /lus/eagle/projects/argonne_tpc/abalaji/modcon/genome/output/GENOME-SFT-1B/2026-06-29/07-41-10/checkpoints/step_3000
# Set to "true" to run interleaved mode, "false" for prefix mode.
# Must be "true" for checkpoints trained with is_interleaved_qa=True (SFT/GRPO stages).
INTERLEAVE_QA="true"

# --- W&B ---
WANDB_PROJECT="prism-kegg-eval"
WANDB_ENTITY=""          # leave empty to use default entity
WANDB_RUN_NAME=""        # leave empty to auto-generate from model_type + timestamp
WANDB_MODE="online"      # online | offline | disabled

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

# --- General ---
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=8

if [[ "${INTERLEAVE_QA}" == "true" ]]; then
    MODE="interleaved"
    INTERLEAVE_FLAG="--interleave_qa"
else
    MODE="prefix"
    INTERLEAVE_FLAG=""
fi

echo "=== PRISM KEGG Evaluation ==="
echo "Node:           $(hostname)"
echo "Checkpoint:     ${CHECKPOINT_DIR}"
echo "Mode:           ${MODE}"
echo "W&B project:    ${WANDB_PROJECT} (${WANDB_MODE})"
echo "PBS_JOBID:      ${PBS_JOBID:-n/a}"
echo ""

# Build optional W&B flags
WANDB_FLAGS="--wandb_project ${WANDB_PROJECT} --wandb_mode ${WANDB_MODE}"
[[ -n "${WANDB_ENTITY}"   ]] && WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
[[ -n "${WANDB_RUN_NAME}" ]] && WANDB_FLAGS="${WANDB_FLAGS} --wandb_run_name ${WANDB_RUN_NAME}"

python -u eval_kegg.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --model_type auto \
  --backbone_id allenai/OLMo-1B-0724-hf \
  --max_new_tokens 256 \
  --truncate_per_side 1024 \
  --max_dna_tokens 1024 \
  ${INTERLEAVE_FLAG} \
  ${WANDB_FLAGS}

echo "Done."

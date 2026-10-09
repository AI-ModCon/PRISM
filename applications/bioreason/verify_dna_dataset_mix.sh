#!/usr/bin/env bash
#PBS -N prism-dna-mix-verify
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -l filesystems=home:eagle
#PBS -A argonne_tpc
#PBS -q debug
#
# Step 6 end-to-end verification for the variant_effect dataset integration.
# Runs a real GRPO training loop for enough steps that, at the configured
# 0.5/0.25/0.25 sampling weights (datasets_config.json), all three DNA
# datasets (dna_bioreason=KEGG, dna_bioreason_variant_effect_coding,
# dna_bioreason_variant_effect_non_snv) are overwhelmingly likely to be drawn
# at least once, confirming:
#   1. All three datasets stream successfully from HF (no load/schema errors)
#   2. The sampler actually mixes all three, not just KEGG
#   3. Training doesn't crash on the new answer formats/handlers
#
# Usage:
#   qsub applications/bioreason/verify_dna_dataset_mix.sh
#   bash applications/bioreason/verify_dna_dataset_mix.sh   # on an already-allocated compute node

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

# --- W&B offline (no network needed) ---
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
echo " DNA DATASET MIX VERIFICATION"
echo " Node:        $(hostname)"
echo " GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo " Checkpoint:  ${STAGE1_CKPT}"
echo " Steps:       30 (batch_size=8 -> 240 samples; P(a given dataset never"
echo "              picked) is astronomically small at weights 0.5/0.25/0.25)"
echo "================================================================"
echo ""

VERIFY_LOG="/tmp/prism_dna_mix_verify_${PBS_JOBID:-$$}.log"

python -u train.py \
  training=bioreason_grpo \
  exp.id="DNA-MIX-VERIFY" \
  exp.variant="1B" \
  training.resume_weights_only="${STAGE1_CKPT}" \
  training.max_steps=30 \
  training.grpo_num_generations=2 \
  training.grpo_max_completion_length=128 \
  training.save_every_n_steps=9999 \
  training.eval_every_n_steps=9999 \
  +training.log_every_n_steps=1 \
  +training.verbosity=DEBUG \
  wandb.mode=offline \
  2>&1 | tee "${VERIFY_LOG}"

echo ""
echo "================================================================"
echo " DATASET SELECTION COUNTS (dna modality group)"
echo "================================================================"
grep -oE "\[Sampler\] modality=dna selected dataset=[a-z_]+" "${VERIFY_LOG}" \
  | sed 's/.*dataset=//' | sort | uniq -c \
  || echo "(no [Sampler] lines found — check log above for errors)"

echo ""
echo "================================================================"
echo " REWARD SCORES"
echo "================================================================"
grep -oE "\[ZoneD\] step=[0-9]+ loss=[-0-9.]+ avg_reward=[0-9.]+" "${VERIFY_LOG}" \
  || echo "(no per-step reward lines found — training may have failed before step 1)"

echo ""
echo "================================================================"
echo " VERIFICATION COMPLETE — full log: ${VERIFY_LOG}"
echo "================================================================"

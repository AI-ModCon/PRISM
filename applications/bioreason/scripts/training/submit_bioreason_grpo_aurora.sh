#!/bin/bash -l
#PBS -N bioreason-grpo-kegg-1b
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -A ModCon
#PBS -k doe
#PBS -j oe
#
# Aurora XPU launcher: BioReason Stage 3 GRPO on KEGG (wanglab/kegg) only.
# Aurora port of submit_bioreason_grpo.sh (Polaris/CUDA).
#   base model : OLMo-1B (model=prism_olmo_1b) -- training=bioreason_grpo alone
#                would fall back to config.yaml's default prism_olmo3_7b
#   start from : Stage 2 SFT checkpoint (model.safetensors + lora_adapter.pt);
#                the SFT adapter (r=32, alpha=64) is merged into the base, then
#                a fresh r=16 GRPO adapter trains on top
#   frozen     : NT-2.5B DNA encoder, DNA projector, base LLM
#   saves      : checkpoints/step_*/lora_adapter.pt only (no model.safetensors),
#                so eval needs the SFT checkpoint + this adapter
#
# ZoneDTrainer is single-process by design (no DDP/vLLM), so this runs ONE rank
# on ONE tile; the other 11 tiles of the node sit idle.
#
# Batch job:
#   qsub -v STAGE1_CKPT=<sft run>/checkpoints/step_final applications/bioreason/scripts/training/submit_bioreason_grpo_aurora.sh
#   qsub -v STAGE1_CKPT=...,MAX_STEPS=50,RUN_TAG=demo applications/bioreason/scripts/training/submit_bioreason_grpo_aurora.sh
# Interactive node:
#   STAGE1_CKPT=... MAX_STEPS=5 bash applications/bioreason/scripts/training/submit_bioreason_grpo_aurora.sh
#
# Outputs land in ${PRISM_CKPT_ROOT}/${RUN_ID}/${RUN_TAG} (override with
# HYDRA_RUN_DIR); EXTRA_HYDRA is appended verbatim to the train.py command.

set -eo pipefail

PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." 2>/dev/null && pwd)}"
# Under qsub the script runs from a spool copy, so BASH_SOURCE points there.
if [ -n "${PBS_O_WORKDIR:-}" ] && [ ! -f "${PROJ_DIR}/src/train.py" ]; then
  PROJ_DIR="${PBS_O_WORKDIR}"
fi
RUN_ID="${RUN_ID:-BIOREASON-GRPO-KEGG}"
RUN_VARIANT="1B"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
MAX_STEPS="${MAX_STEPS:-1000}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_COMPLETION="${MAX_COMPLETION:-800}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EXTRA_HYDRA="${EXTRA_HYDRA:-}"

export PRISM_DATA_ROOT="${PRISM_DATA_ROOT:-/lus/flare/projects/ModCon/${USER}/cache}"
export PRISM_OUTPUT_ROOT="${PRISM_OUTPUT_ROOT:-/lus/flare/projects/ModCon/${USER}/output}"
export PRISM_CKPT_ROOT="${PRISM_CKPT_ROOT:-/lus/flare/projects/ModCon/${USER}/modcon/checkpoints}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${PRISM_CKPT_ROOT}/${RUN_ID}/${RUN_TAG}}"

# Stage 2 SFT checkpoint. Default: newest SFT step_final under PRISM_CKPT_ROOT.
STAGE1_CKPT="${STAGE1_CKPT:-$(ls -td "${PRISM_CKPT_ROOT}/BIOREASON-SFT-KEGG"/*/checkpoints/step_final 2>/dev/null | head -n 1)}"
if [ -z "${STAGE1_CKPT}" ]; then
  echo "ERROR: no SFT checkpoint found -- run scripts/training/submit_bioreason_sft.sh or set STAGE1_CKPT."
  exit 1
fi

cd "${PROJ_DIR}"

export PRISM_VENV="${PRISM_VENV:-/flare/ModCon/${USER}/prism-envs/bioreason-fw2026.1}"
module use /soft/modulefiles
module load frameworks/2026.1.0
module load hdf5
if [ ! -f "${PRISM_VENV}/bin/activate" ]; then
  echo "ERROR: PRISM_VENV not found: ${PRISM_VENV} (bash applications/bioreason/helpers/setup_jupyter_kernel.sh builds it)"
  exit 1
fi
source "${PRISM_VENV}/bin/activate"

if [[ ! "${HOSTNAME}" =~ aurora-uan ]]; then
  export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
  export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
  export http_proxy="http://proxy.alcf.anl.gov:3128"
  export https_proxy="http://proxy.alcf.anl.gov:3128"
fi

export HF_HOME="${HF_HOME:-/flare/ModCon/${USER}/cache/model_weights}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT=true
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS=ignore
export NUMEXPR_MAX_THREADS=512
export NUMEXPR_NUM_THREADS=128
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
# Text + 2 x max_dna_length (1024) DNA slots is ~2.8-3.6k tokens for KEGG; the
# default 2048 merged-length guard (model.py, issue #123) would reject it.
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

for p in "${STAGE1_CKPT}/model.safetensors" "${STAGE1_CKPT}/lora_adapter.pt" \
         "${PRISM_DATA_ROOT}/dna_local_shards/kegg/dataset_info.json" \
         "${HF_HOME}/hub/models--allenai--OLMo-1B-0724-hf" \
         "${HF_HOME}/hub/models--InstaDeepAI--nucleotide-transformer-2.5b-multi-species"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: missing required path: $p"
    exit 1
  fi
done

echo "================================================================"
echo " BIOREASON GRPO (KEGG only, OLMo-1B + LoRA r16, Aurora, 1 tile)"
echo " Node:          $(hostname)"
echo " SFT ckpt:      ${STAGE1_CKPT}"
echo " Run dir:       ${HYDRA_RUN_DIR}"
echo " Max steps:     ${MAX_STEPS} (G=${NUM_GENERATIONS}, max completion ${MAX_COMPLETION})"
echo "================================================================"

mpiexec -n 1 -ppn 1 --cpu-bind depth --depth 8 bash -lc '
  module use /soft/modulefiles
  module load frameworks/2026.1.0
  source '"${PRISM_VENV}"'/bin/activate
  export ZE_AFFINITY_MASK=0
  export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1

  cd '"${PROJ_DIR}"'
  python -u src/train.py \
    training=bioreason_grpo \
    model=prism_olmo_1b \
    exp.id='"${RUN_ID}"' \
    exp.variant='"${RUN_VARIANT}"' \
    training.resume_weights_only='"${STAGE1_CKPT}"' \
    training.bioreason_dataset=wanglab/kegg \
    training.max_steps='"${MAX_STEPS}"' \
    training.grpo_num_generations='"${NUM_GENERATIONS}"' \
    training.grpo_max_completion_length='"${MAX_COMPLETION}"' \
    training.save_every_n_steps='"${SAVE_EVERY}"' \
    wandb.mode='"${WANDB_MODE}"' \
    hydra.run.dir='"${HYDRA_RUN_DIR}"' '"${EXTRA_HYDRA}"'
'

echo ""
echo " GRPO adapter: ${HYDRA_RUN_DIR}/checkpoints/step_final/lora_adapter.pt"

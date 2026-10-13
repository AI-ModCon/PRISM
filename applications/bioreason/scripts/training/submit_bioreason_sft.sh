#!/bin/bash -l
#PBS -N bioreason-sft-kegg-1b
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -A ModCon
#PBS -k doe
#PBS -j oe
#
# Aurora XPU launcher: BioReason Stage 2 SFT on KEGG (wanglab/kegg) only.
#   base model : OLMo-1B (model=prism_olmo_1b), backbone frozen
#   trainable  : DNA projector (random init, no Stage 1) + LoRA adapters
#   frozen     : NT-2.5B DNA encoder
#   loss       : assistant span only
#                (<think>\n{reasoning}\n</think>\n\nAnswer: <answer>{answer}</answer><|im_end|>)
#
# Batch job:
#   qsub applications/bioreason/scripts/training/submit_bioreason_sft.sh
#   qsub -v MAX_STEPS=2000 applications/bioreason/scripts/training/submit_bioreason_sft.sh
#   # optional: start from a pretrained projector instead of random init
#   qsub -v STAGE1_CKPT=/path/to/checkpoints/step_N applications/bioreason/scripts/training/submit_bioreason_sft.sh
#
# Interactive node (after `qsub -I -l select=1,walltime=01:00:00,filesystems=home:flare -q debug -A ModCon`):
#   bash applications/bioreason/scripts/training/submit_bioreason_sft.sh
# Single-tile debug run:
#   NPERNODE=1 MAX_STEPS=20 bash applications/bioreason/scripts/training/submit_bioreason_sft.sh

set -eo pipefail

PROJ_DIR="/home/abalaji/projects/modcon/release/BaseMM_PRISM"
RUN_ID="${RUN_ID:-BIOREASON-SFT-KEGG}"
RUN_VARIANT="1B"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
MAX_STEPS="${MAX_STEPS:-500}"
NPERNODE="${NPERNODE:-12}"  # 6 PVC GPUs x 2 tiles per Aurora node (FLAT mode)

# Site roots: datasets_config.json resolves dna_bioreason's local_path as
# ${PRISM_DATA_ROOT}/dna_local_shards/kegg.
export PRISM_DATA_ROOT="${PRISM_DATA_ROOT:-/lus/flare/projects/ModCon/abalaji/cache}"
export PRISM_OUTPUT_ROOT="${PRISM_OUTPUT_ROOT:-/lus/flare/projects/ModCon/abalaji/output}"
# Checkpoints (~13 GB each) go to flare, not Hydra's default outputs/ under the
# repo in /home, which overflows its quota.
export PRISM_CKPT_ROOT="${PRISM_CKPT_ROOT:-/lus/flare/projects/ModCon/abalaji/modcon/checkpoints}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${PRISM_CKPT_ROOT}/${RUN_ID}/${RUN_TAG}}"

# Optional: Stage 1 (bioreason_projector, model=prism_olmo_1b) checkpoint. Unset
# (default) = projector trains from random init, as in BioReason. If set, its
# output width must match OLMo-1B's d_model (2048).
STAGE1_CKPT="${STAGE1_CKPT:-}"
RESUME_ARG=""
if [ -n "${STAGE1_CKPT}" ]; then
  RESUME_ARG="training.resume_weights_only=${STAGE1_CKPT}"
fi

cd "${PROJ_DIR}"

# frameworks module + a --system-site-packages venv adding only what the module
# lacks (wandb, torchtune, webdataset, braceexpand). Not the repo's .venv,
# which carries a CUDA torch build. Build it once (from a UAN) with:
#   module load frameworks/2026.1.0
#   uv venv --system-site-packages -p "$(which python)" $PRISM_VENV && source $PRISM_VENV/bin/activate
#   uv pip install wandb braceexpand==0.1.7 && uv pip install --no-deps webdataset==1.0.2 torchtune==0.6.1
export PRISM_VENV="${PRISM_VENV:-/flare/ModCon/${USER}/prism-envs/bioreason-fw2026.1}"
module use /soft/modulefiles
module load frameworks/2026.1.0
module load hdf5
if [ ! -f "${PRISM_VENV}/bin/activate" ]; then
  echo "ERROR: PRISM_VENV not found: ${PRISM_VENV} (see build steps above)"
  exit 1
fi
source "${PRISM_VENV}/bin/activate"

if [[ ! "${HOSTNAME}" =~ aurora-uan ]]; then
  export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
  export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
  export http_proxy="http://proxy.alcf.anl.gov:3128"
  export https_proxy="http://proxy.alcf.anl.gov:3128"
fi

# OLMo-1B-0724-hf, nucleotide-transformer-2.5b-multi-species and
# datasets--wanglab--kegg are all pre-cached here; run fully offline.
export HF_HOME="/flare/ModCon/${USER}/cache/model_weights"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT=true
# PBS's default TMPDIR overflows the 108-byte AF_UNIX socket path limit that
# torch's DataLoader worker IPC relies on -- force a short one.
export TMPDIR=/tmp
# DDP side-copy streams index torch.accelerator.device_count(), which is 0
# under ZE_AFFINITY_MASK on Aurora -> IndexError on first forward.
export PYTORCH_DDP_USE_SIDE_STREAM=0
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

# Falls back to the current host on an interactive node without PBS_NODEFILE.
if [ -n "${PBS_NODEFILE:-}" ]; then
  MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=$((20000 + RANDOM % 20000))

REQUIRED=("${PRISM_DATA_ROOT}/dna_local_shards/kegg/dataset_info.json" \
          "${HF_HOME}/hub/models--allenai--OLMo-1B-0724-hf" \
          "${HF_HOME}/hub/models--InstaDeepAI--nucleotide-transformer-2.5b-multi-species")
[ -n "${STAGE1_CKPT}" ] && REQUIRED+=("${STAGE1_CKPT}/model.safetensors")
for p in "${REQUIRED[@]}"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: missing required path: $p"
    exit 1
  fi
done

echo "================================================================"
echo " BIOREASON SFT (KEGG only, OLMo-1B + LoRA + projector, Aurora)"
echo " Node:          $(hostname)"
echo " Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo " Ranks:         ${NPERNODE}"
echo " Projector init: ${STAGE1_CKPT:-random (no Stage 1)}"
echo " KEGG shard:    ${PRISM_DATA_ROOT}/dna_local_shards/kegg"
echo " Max steps:     ${MAX_STEPS}"
echo " Run dir:       ${HYDRA_RUN_DIR}"
echo "================================================================"

mpiexec -n "${NPERNODE}" -ppn "${NPERNODE}" --cpu-bind depth --depth 8 bash -lc '
  module use /soft/modulefiles
  module load frameworks/2026.1.0
  source '"${PRISM_VENV}"'/bin/activate

  export CCL_PROCESS_LAUNCHER=none
  export CCL_ATL_TRANSPORT=ofi
  export CCL_OP_SYNC=1
  export CCL_WORKER_COUNT=1
  export CCL_ALLREDUCE=ring
  export CCL_OFI_ENABLE_HOSTNAME_SHARING=0
  export FI_CXI_RX_MATCH_MODE=hybrid

  export WORLD_SIZE=${PMI_SIZE:-${PALS_SIZE:-'"${NPERNODE}"'}}
  export RANK=${PMI_RANK:-${PALS_RANKID:-0}}
  export LOCAL_RANK=${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}
  export LOCAL_WORLD_SIZE=${PMI_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-'"${NPERNODE}"'}}
  export ZE_AFFINITY_MASK=$LOCAL_RANK
  # Each rank now sees exactly one tile, as xpu:0. Accelerate/DDP build the
  # device from LOCAL_RANK, so leaving it at 0..11 targets xpu:<LOCAL_RANK> ->
  # "device index is out of range". Same normalization as tools/launch_aurora_web.py.
  export ACCELERATE_ORIG_LOCAL_RANK=$LOCAL_RANK
  export LOCAL_RANK=0 PMI_LOCAL_RANK=0 PMIX_LOCAL_RANK=0 PALS_LOCAL_RANKID=0 OMPI_COMM_WORLD_LOCAL_RANK=0
  export MASTER_ADDR='"${MASTER_ADDR}"'
  export MASTER_PORT='"${MASTER_PORT}"'

  cd '"${PROJ_DIR}"'
  python -u src/train.py \
    training=bioreason_sft \
    model=prism_olmo_1b \
    exp.id='"${RUN_ID}"' \
    exp.variant='"${RUN_VARIANT}"' \
    '"${RESUME_ARG}"' \
    training.bioreason_dataset=wanglab/kegg \
    training.max_steps='"${MAX_STEPS}"' \
    wandb.mode='"${WANDB_MODE}"' \
    hydra.run.dir='"${HYDRA_RUN_DIR}"'
'

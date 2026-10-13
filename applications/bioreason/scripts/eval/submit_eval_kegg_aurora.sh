#!/bin/bash -l
#PBS -N bioreason-eval-kegg-1b
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
# Allocation is not hardcoded: submit with `qsub -A <project> ...`.
#PBS -k doe
#PBS -j oe
#
# Aurora XPU launcher: KEGG accuracy eval for BioReason checkpoints (1 tile).
# Aurora port of submit_eval_kegg.sh (Polaris); runs eval_kegg_aurora.py
# because eval_kegg.py (PR #133) is not in this repo yet.
#
#   CKPT          SFT checkpoint dir (model.safetensors [+ lora_adapter.pt]). Required.
#   GRPO_ADAPTER  optional GRPO lora_adapter.pt merged on top of CKPT.
#   SPLIT         KEGG split to score (default test; saved by the notebook /
#                 helpers/download_bioreason_dna_datasets.py under dna_local_shards_<split>/kegg).
#   LIMIT         optional cap on examples.
#   DNA_MODE      real (default) | shuffled (another item's DNA) | blank (constant N): DNA ablation.
#   OUTPUT        metrics JSON (default ${PRISM_OUTPUT_ROOT}/BIOREASON-EVAL-KEGG/${RUN_TAG}/results.json,
#                 or results_<DNA_MODE>.json when DNA_MODE != real).
#
#   qsub -v CKPT=<sft>/checkpoints/step_final applications/bioreason/scripts/eval/submit_eval_kegg_aurora.sh
#   qsub -v CKPT=...,GRPO_ADAPTER=<grpo>/checkpoints/step_final/lora_adapter.pt,LIMIT=50 ...

set -eo pipefail

PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." 2>/dev/null && pwd)}"
if [ -n "${PBS_O_WORKDIR:-}" ] && [ ! -f "${PROJ_DIR}/src/train.py" ]; then
  PROJ_DIR="${PBS_O_WORKDIR}"
fi
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DNA_MODE="${DNA_MODE:-real}"
case "${DNA_MODE}" in real|shuffled|blank) ;; *) echo "ERROR: DNA_MODE must be real|shuffled|blank"; exit 1;; esac

export PRISM_DATA_ROOT="${PRISM_DATA_ROOT:-/lus/flare/projects/<project>/${USER}/cache}"
export PRISM_OUTPUT_ROOT="${PRISM_OUTPUT_ROOT:-/lus/flare/projects/<project>/${USER}/output}"
RESULT_NAME="results.json"; [ "${DNA_MODE}" != "real" ] && RESULT_NAME="results_${DNA_MODE}.json"
OUTPUT="${OUTPUT:-${PRISM_OUTPUT_ROOT}/BIOREASON-EVAL-KEGG/${RUN_TAG}/${RESULT_NAME}}"
DATA="${DATA:-${PRISM_DATA_ROOT}/dna_local_shards_${SPLIT}/kegg}"

if [ -z "${CKPT:-}" ]; then
  echo "ERROR: set CKPT=<sft run>/checkpoints/step_final"
  exit 1
fi

cd "${PROJ_DIR}"

export PRISM_VENV="${PRISM_VENV:-/flare/<project>/${USER}/prism-envs/bioreason-fw2026.1}"
module use /soft/modulefiles
module load frameworks/2026.1.0
module load hdf5
source "${PRISM_VENV}/bin/activate"

export HF_HOME="${HF_HOME:-/flare/<project>/${USER}/cache/model_weights}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0
# Text + 2 x max_dna_length (1024) DNA slots is ~2.8-3.6k tokens for KEGG; the
# default 2048 merged-length guard (model.py, issue #123) would reject it.
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

for p in "${CKPT}/model.safetensors" "${DATA}/dataset_info.json" ${GRPO_ADAPTER:+"${GRPO_ADAPTER}"}; do
  if [ ! -e "$p" ]; then
    echo "ERROR: missing required path: $p"
    exit 1
  fi
done

echo "================================================================"
echo " BIOREASON KEGG EVAL (Aurora, 1 tile)"
echo " Checkpoint:    ${CKPT}"
echo " GRPO adapter:  ${GRPO_ADAPTER:-<none>}"
echo " Data:          ${DATA} (limit ${LIMIT:-all})"
echo " DNA mode:      ${DNA_MODE}"
echo " Output:        ${OUTPUT}"
echo "================================================================"

python -u applications/bioreason/eval_kegg_aurora.py \
  --ckpt "${CKPT}" \
  ${GRPO_ADAPTER:+--grpo-adapter "${GRPO_ADAPTER}"} \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --dna-mode "${DNA_MODE}" \
  ${LIMIT:+--limit "${LIMIT}"}

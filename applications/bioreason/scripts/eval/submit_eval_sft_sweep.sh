#!/bin/bash
# Submit the BioReason SFT validation sweep: one scripts/eval/submit_eval_kegg_aurora.sh job per config,
# run in parallel (login node; this script only calls qsub).
#
#   real DNA     on every checkpoint   -> accuracy / F1 vs. training step (overfit check)
#   shuffled DNA on step_final         -> does the model actually use the DNA?
#   blank DNA    on step_final
#
# Usage:
#   bash applications/bioreason/scripts/eval/submit_eval_sft_sweep.sh <sft run>/checkpoints
#   SPLIT=test LIMIT=20 WALLTIME=00:30:00 bash applications/bioreason/scripts/eval/submit_eval_sft_sweep.sh <dir>
# Select checkpoints on SPLIT=val (default); run SPLIT=test once on the chosen one.
# Then:  python applications/bioreason/helpers/summarize_eval_sweep.py <PRISM_OUTPUT_ROOT>/BIOREASON-EVAL-KEGG/<TAG>

set -eo pipefail
CKPT_DIR="${1:?usage: $0 <sft run>/checkpoints}"
SPLIT="${SPLIT:-val}"
WALLTIME="${WALLTIME:-01:00:00}"
TAG="${TAG:-sweep-$(date +%Y%m%d-%H%M%S)}"
PRISM_OUTPUT_ROOT="${PRISM_OUTPUT_ROOT:-/lus/flare/projects/<project>/${USER}/output}"
OUT_DIR="${PRISM_OUTPUT_ROOT}/BIOREASON-EVAL-KEGG/${TAG}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUT_DIR}"
echo "Sweep tag: ${TAG}  split: ${SPLIT}  outputs: ${OUT_DIR}"

submit() {  # <ckpt> <mode> <name>
  local vars="CKPT=$1,SPLIT=${SPLIT},DNA_MODE=$2,OUTPUT=${OUT_DIR}/$3.json,RUN_TAG=${TAG}"
  [ -n "${LIMIT:-}" ] && vars+=",LIMIT=${LIMIT}"
  qsub -l walltime="${WALLTIME}" -v "${vars}" "${HERE}/submit_eval_kegg_aurora.sh"
}

shopt -s nullglob
found=0
for c in "${CKPT_DIR}"/step_[0-9]* ; do
  [ -f "${c}/model.safetensors" ] || continue
  submit "${c}" real "$(basename "${c}")_real"; found=1
done
FINAL="${CKPT_DIR}/step_final"
if [ -f "${FINAL}/model.safetensors" ]; then
  submit "${FINAL}" real step_final_real
  submit "${FINAL}" shuffled step_final_shuffled
  submit "${FINAL}" blank step_final_blank
  found=1
fi
[ "${found}" = 1 ] || { echo "ERROR: no checkpoints with model.safetensors under ${CKPT_DIR}"; exit 1; }
echo "Submitted. Summarize with: python applications/bioreason/helpers/summarize_eval_sweep.py ${OUT_DIR}"

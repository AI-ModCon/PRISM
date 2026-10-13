#!/bin/bash -l
# One-time PRISM environment for the BioReason application on Aurora (no Jupyter here).
#
#   Builds (or reuses) the BioReason venv on top of frameworks/2026.1.0 -- the same venv the
#   scripts/training/submit_bioreason_*.sh scripts activate -- and installs the training extras
#   the frameworks module lacks plus PRISM itself (--no-deps, so the module's XPU torch stays).
#
# Run once from a login node (UAN; it has direct internet access):
#   export PRISM_VENV=/flare/<project>/$USER/prism-envs/bioreason-fw2026.1
#   bash applications/bioreason/helpers/setup_prism_env.sh
# For the notebook, then run helpers/setup_jupyter_kernel.sh.

set -eo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PRISM_DIR="$( cd "${HERE}/../../.." && pwd )"
FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks/2026.1.0}"
PRISM_VENV="${PRISM_VENV:-/flare/<project>/${USER}/prism-envs/bioreason-fw2026.1}"
case "${PRISM_VENV}" in *"<project>"*) echo "ERROR: set PRISM_VENV to your venv path, e.g. /flare/\$PROJECT/\$USER/prism-envs/bioreason-fw2026.1"; exit 1;; esac

module use /soft/modulefiles
module load "${FRAMEWORKS_MODULE}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [ ! -f "${PRISM_VENV}/bin/activate" ]; then
  echo "Creating ${PRISM_VENV} (--system-site-packages over ${FRAMEWORKS_MODULE})"
  uv venv --system-site-packages -p "$(which python)" "${PRISM_VENV}"
fi
source "${PRISM_VENV}/bin/activate"

# Same extras as the recipe in submit_bioreason_sft.sh; --no-deps keeps the
# module's XPU torch from being replaced by a PyPI/CUDA wheel.
uv pip install wandb braceexpand==0.1.7
uv pip install --no-deps webdataset==1.0.2 torchtune==0.6.1
uv pip install --no-deps -e "${PRISM_DIR}"

echo "PRISM environment ready:"
echo "  venv:    ${PRISM_VENV}"
echo "  module:  ${FRAMEWORKS_MODULE}"

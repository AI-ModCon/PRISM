#!/bin/bash -l
# One-time setup for applications/bioreason/bioreason_demo.ipynb on Aurora.
#
#   1. Builds (or reuses) the BioReason venv on top of frameworks/2026.1.0 --
#      the same venv the scripts/training/submit_bioreason_*.sh scripts activate.
#   2. Installs the notebook-only extras (ipykernel, ipywidgets) plus the
#      training extras the frameworks module lacks.
#   3. Registers a Jupyter kernel `prism-bioreason` whose argv is
#      kernel_wrapper.sh, so the kernel always starts with the module + venv
#      loaded (VS Code Remote and jupyter lab both pick it up).
#
# Run once from a login node (UAN; it has direct internet access):
#   bash applications/bioreason/helpers/setup_jupyter_kernel.sh
# Then, from VS Code connected to the interactive compute node, open the
# notebook and pick the "PRISM BioReason (Aurora)" kernel.

set -eo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PRISM_DIR="$( cd "${HERE}/../../.." && pwd )"
FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks/2026.1.0}"
PRISM_VENV="${PRISM_VENV:-/flare/ModCon/${USER}/prism-envs/bioreason-fw2026.1}"
KERNEL_NAME="prism-bioreason"

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
# jupyter_server: fallback when VS Code doesn't list the kernel -- run
# `jupyter server --no-browser --ip=$(hostname) --port 8888` on the compute node
# and connect via "Select Kernel -> Existing Jupyter Server".
uv pip install ipykernel ipywidgets jupyter_server
uv pip install --no-deps -e "${PRISM_DIR}"

chmod +x "${HERE}/kernel_wrapper.sh"
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}"
mkdir -p "${KERNEL_DIR}"
cat > "${KERNEL_DIR}/kernel.json" <<EOF
{
  "argv": ["${HERE}/kernel_wrapper.sh", "-f", "{connection_file}"],
  "display_name": "PRISM BioReason (Aurora)",
  "language": "python",
  "env": {"PRISM_VENV": "${PRISM_VENV}", "FRAMEWORKS_MODULE": "${FRAMEWORKS_MODULE}"},
  "metadata": {"debugger": false}
}
EOF

echo "Registered kernel '${KERNEL_NAME}' -> ${KERNEL_DIR}/kernel.json"
echo "  venv:    ${PRISM_VENV}"
echo "  module:  ${FRAMEWORKS_MODULE}"

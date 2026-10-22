#!/bin/bash -l
# One-time setup for applications/bioreason/bioreason_demo.ipynb on Aurora.
#
#   Installs Jupyter into the PRISM venv and registers a kernel `prism-bioreason` whose argv is
#   kernel_wrapper.sh, so the kernel always starts with the module + venv loaded (VS Code
#   Remote and jupyter lab both pick it up). Jupyter is not part of PRISM's requirements.
#
#   Requires the venv from helpers/setup_prism_env.sh (run that first).
#
# Run once from a login node (UAN; it has direct internet access):
#   export PRISM_VENV=/flare/<project>/$USER/prism-envs/bioreason-fw2026.1
#   bash applications/bioreason/helpers/setup_jupyter_kernel.sh
# Then, from VS Code connected to the interactive compute node, open the
# notebook and pick the "PRISM BioReason (Aurora)" kernel.

set -eo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks/2026.1.0}"
PRISM_VENV="${PRISM_VENV:-/flare/<project>/${USER}/prism-envs/bioreason-fw2026.1}"
KERNEL_NAME="prism-bioreason"

case "${PRISM_VENV}" in *"<project>"*) echo "ERROR: set PRISM_VENV to your venv path, e.g. /flare/\$PROJECT/\$USER/prism-envs/bioreason-fw2026.1"; exit 1;; esac

module use /soft/modulefiles
module load "${FRAMEWORKS_MODULE}"

if [ ! -f "${PRISM_VENV}/bin/activate" ]; then
  echo "ERROR: ${PRISM_VENV} not found -- run helpers/setup_prism_env.sh first."
  exit 1
fi
source "${PRISM_VENV}/bin/activate"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is not on PATH (see setup_prism_env.sh)"; exit 1; }

# jupyter_server: fallback when VS Code doesn't list the kernel -- run
# `jupyter server --no-browser --ip=$(hostname) --port 8888` on the compute node
# and connect via "Select Kernel -> Existing Jupyter Server".
uv pip install ipykernel ipywidgets jupyter_server

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

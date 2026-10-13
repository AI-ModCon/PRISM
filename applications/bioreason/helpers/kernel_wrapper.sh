#!/bin/bash -l
# Jupyter kernel launcher for the BioReason demo notebook (Aurora).
#
# Registered by setup_jupyter_kernel.sh as the argv of the `prism-bioreason`
# kernel. Jupyter/VS Code start kernels from a bare environment, so the
# frameworks module and the PRISM venv have to be loaded here, in the same
# order the scripts/training/submit_bioreason_*.sh scripts use (module first, then venv --
# reversing it breaks torch/IPEX imports).
#
# Do not call directly; Jupyter passes `-f <connection_file>` as "$@".

FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks/2026.1.0}"
PRISM_VENV="${PRISM_VENV:-/flare/ModCon/${USER}/prism-envs/bioreason-fw2026.1}"

module use /soft/modulefiles
module load "${FRAMEWORKS_MODULE}"
module load hdf5
source "${PRISM_VENV}/bin/activate"

# One PVC tile per device (12 per node), matching the submit scripts.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
# PBS's default TMPDIR overflows the AF_UNIX socket path limit used by
# DataLoader worker IPC (see submit_bioreason_sft.sh).
export TMPDIR=/tmp
export PYTHONWARNINGS=ignore

exec python -m ipykernel_launcher "$@"

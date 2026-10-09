#!/bin/bash -l
#PBS -N hold-1n-timeseries-debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A ModCon
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/pemami/BaseMM_PRISM/logs/hold_1n_timeseries_debug.log

set -euo pipefail

REPO_ROOT="/lus/flare/projects/ModCon/pemami/BaseMM_PRISM"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
NODEFILE_OUT="${LOG_DIR}/hold_1n_timeseries_debug_nodefile.txt"
DEFAULT_DESIGN="${DESIGN:-PRISM-QWEN3-0-6B-TIMEOMNI}"

echo "Hold-1N TimeSeries debug job started: $(date)"
echo "Requested design: ${DEFAULT_DESIGN}"

if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    HEAD_NODE="$(head -n 1 "${PBS_NODEFILE}")"
    sort -u "${PBS_NODEFILE}" > "${NODEFILE_OUT}"
    echo "Nodes:"
    cat "${NODEFILE_OUT}"
else
    HEAD_NODE="$(hostname)"
    echo "${HEAD_NODE}" > "${NODEFILE_OUT}"
    echo "Nodes (hostname fallback):"
    echo "${HEAD_NODE}"
fi

echo "Node file written to ${NODEFILE_OUT}"
echo "Head node: ${HEAD_NODE}"

echo ""
echo "Use a second UAN terminal to launch a debug run against this reserved node:"
echo "  module load frameworks && source .venv-deepspeed/bin/activate && python tools/timeseries_launch.py --yaml experiments/prism_designs.yaml --head-node ${HEAD_NODE} --run-via-ssh --dry-run"
echo "  module load frameworks && source .venv-deepspeed/bin/activate && python tools/timeseries_launch.py --design PRISM-QWEN3-0-6B-INTERN-S2-397B --head-node ${HEAD_NODE} --run-via-ssh"
echo ""
echo "Compatible designs: PRISM-QWEN3-0-6B-TIMEOMNI, PRISM-QWEN3-0-6B-INTERN-S2, PRISM-QWEN3-0-6B-INTERN-S2-397B, PRISM-OLMO-1B-TIMEOMNI, PRISM-OLMO-1B-INTERN-S2-397B"

echo "Sleeping to hold the node for 1 hour..."
sleep 3500

echo "Hold-1N TimeSeries debug job finished: $(date)"

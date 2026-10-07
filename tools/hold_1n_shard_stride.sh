#!/bin/bash -l
#PBS -N pr169-hold
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A ModCon
#PBS -k doe
#PBS -j oe

# Hold one node so shard-stride checks can be iterated interactively from a
# UAN terminal without paying a queue cycle per script fix.
#
# The job does nothing but publish its node name and sleep. Drive it with:
#   bash tools/shard_stride_run.sh <node>
# which ssh's in and runs the actual check. Edit the check and re-run as many
# times as you like while the hold is alive.
#
# Deliberately does NOT use `set -u`: Lmod's init dereferences
# ZSH_EVAL_CONTEXT unguarded and aborts the job (cost us job 8834494).
set -eo pipefail

PRISM_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PRISM_DIR"

NODEFILE_OUT="${PRISM_DIR}/logs/pr169_hold_nodefile.txt"
mkdir -p "$(dirname "$NODEFILE_OUT")"

if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    sort -u "${PBS_NODEFILE}" > "${NODEFILE_OUT}"
    HEAD_NODE="$(head -n 1 "${NODEFILE_OUT}")"
else
    HEAD_NODE="$(hostname)"
    echo "${HEAD_NODE}" > "${NODEFILE_OUT}"
fi

echo "=== PR169 hold node ==="
echo "job    : ${PBS_JOBID:-interactive}"
echo "node   : ${HEAD_NODE}"
echo "prism  : ${PRISM_DIR}"
echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Drive it from a UAN terminal with:"
echo "  bash tools/shard_stride_run.sh ${HEAD_NODE}"
echo ""
echo "NOTE: ${NODEFILE_OUT} is shared and persists across jobs. Always derive"
echo "the node from 'qstat -f <jobid> | grep exec_host', never from that file."

# Hold just under walltime so PBS reaps us cleanly rather than killing mid-run.
sleep 3480
echo "Hold finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

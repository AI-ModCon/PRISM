#!/bin/bash -l
#PBS -N hold_vllm_test
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -q debug
#PBS -A AuroraGPT
#PBS -k doe
#PBS -j oe

# 1-node debug-queue hold for interactive vLLM testing.
#
# Override PBS log location and PRISM dir via env if needed:
#   PRISM_DIR=/path/to/checkout qsub tools/hold_vllm_test.sh
#
# After it starts, the assigned node is written to
# $PRISM_DIR/logs/hold_vllm_nodefile.txt for the runner shells to pick up.

PROJ="${PRISM_DIR:-$PBS_O_WORKDIR}"
mkdir -p "$PROJ/logs"
exec >> "$PROJ/logs/hold_vllm_test.log" 2>&1

echo "Hold job started: $(date)"
echo "PRISM dir: $PROJ"
echo "Nodes:"
cat "$PBS_NODEFILE" | sort -u
sort -u "$PBS_NODEFILE" > "$PROJ/logs/hold_vllm_nodefile.txt"
echo "Node file written to $PROJ/logs/hold_vllm_nodefile.txt"
echo "Sleeping..."
sleep 3500
echo "Hold job finished: $(date)"

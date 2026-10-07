#!/bin/bash -l
#PBS -N hold-nodes-lustre
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -A AuroraGPT
#PBS -k doe
#PBS -j oe

# 2N debug-scaling hold for vLLM follow-on work (TS encoder + parity).
# Lustre-only (no DAOS) so it schedules when DAOS is down.
PROJ="${PRISM_DIR:-$PBS_O_WORKDIR}"
mkdir -p "$PROJ/logs"
exec >> "$PROJ/logs/hold_nodes_lustre.log" 2>&1

echo "Hold job started: $(date)"
echo "Nodes:"
sort -u "$PBS_NODEFILE"
sort -u "$PBS_NODEFILE" > "$PROJ/logs/hold_nodes_lustre_nodefile.txt"
echo "Sleeping..."
sleep 3500
echo "Hold job finished: $(date)"

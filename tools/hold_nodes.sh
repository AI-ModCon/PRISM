#!/bin/bash -l
#PBS -N hold-nodes
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -q debug-scaling
#PBS -A AuroraGPT
#PBS -k doe
#PBS -j oe

PROJ="${PRISM_DIR:-$PBS_O_WORKDIR}"
mkdir -p "$PROJ/logs"
exec >> "$PROJ/logs/hold_nodes.log" 2>&1

echo "Hold job started: $(date)"
echo "Nodes:"
cat $PBS_NODEFILE | sort -u
# Write nodefile to a known location for the launcher
sort -u $PBS_NODEFILE > "$PROJ/logs/hold_nodefile.txt"
echo "Node file written to logs/hold_nodefile.txt"
echo "Sleeping to hold nodes..."
sleep 3500
echo "Hold job finished: $(date)"

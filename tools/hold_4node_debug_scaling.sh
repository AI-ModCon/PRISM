#!/bin/bash -l
#PBS -N hold-4n-stage-a
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -A AuroraGPT
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/BaseMM_PRISM/logs/hold_4node_debug_scaling.log

# 4-node, 1-hr Lustre-only hold for Stage A IsoFLOP round 1a.
#
# Workflow (Terminal 1 + Terminal 2 from CLAUDE.md):
#   Terminal 1 (UAN):
#     $ qsub tools/hold_4node_debug_scaling.sh
#     $ qstat -f <jobid> | grep exec_host    # discover head node
#   Terminal 2 (UAN):
#     $ python tools/isoflop_launch.py \
#         --plan /flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/stage_a_round1_olmo3_1b.yaml \
#         --csv  /flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/experiments.csv \
#         --launcher tools/launch_aurora_web.py \
#         --filter budget_flops=3.000000e+17 \
#         --sweep-id STAGE-A-ROUND1A-$(date +%Y%m%d)
#     (Internally each cell launches launch_aurora_web.py --nodes 4 which reads
#      $PBS_NODEFILE inside this hold.)
#
# IMPORTANT: filesystems=home:flare (no daos_user_fs) per
# [[aurora-daos-filesystem]] — Stage A is Lustre/webdataset, DAOS not needed.
# Adding daos_user_fs would queue indefinitely when DAOS is down.

echo "Hold-4N Stage A job started: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Nodes:"
cat $PBS_NODEFILE | sort -u

# Persist the node list so other terminals can SSH in.
sort -u $PBS_NODEFILE > /lus/flare/projects/ModCon/ngetty/BaseMM_PRISM/logs/hold_4node_nodefile.txt
echo "Node file written to logs/hold_4node_nodefile.txt"
echo "Head node: $(head -n 1 $PBS_NODEFILE)"

echo ""
echo "Sleeping to hold nodes for 1 hr (debug-scaling walltime)..."
echo "From UAN, run:"
echo "  python tools/isoflop_launch.py --plan .../stage_a_round1_olmo3_1b.yaml \\"
echo "    --launcher tools/launch_aurora_web.py --filter budget_flops=3.000000e+17"
echo ""

sleep 3500
echo "Hold-4N Stage A job finished: $(date)"

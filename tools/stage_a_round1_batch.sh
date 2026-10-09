#!/bin/bash -l
#PBS -N stage-a-round1
#PBS -l select=4
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -A AuroraGPT
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/BaseMM_PRISM/logs/stage_a_round1.log

# Stage A IsoFLOP round 1 — capacity batch job for ALL 18 cells.
#
# Prerequisites:
#   - PR #100 (pre-A) on main: seed plumbing + --nodes-override + web cal handle
#   - experiments/stage_a_round1_olmo3_1b.yaml committed with all 18 cells
#     (or symlinked into scaling-study/ on /flare for working state)
#
# Cell estimates from the actual calibration (cal_fps≈3.07e13 → runtime_fps≈5.9e15
# at 4 nodes × 12 ranks):
#   - 5 cells @ budget=3e17  → max_steps≈51   (~1.4 min each)
#   - 5 cells @ budget=1e18  → max_steps≈169  (~4.8 min each)
#   - 5 cells @ budget=3e18  → max_steps≈506  (~14 min each)
#   - 3 BASE@3e17 seed reps (seeds 1-3) for variance-floor (~1.4 min each)
# Total compute: ~120 min. Plus launcher overhead (venv extract, model
# staging, mpiexec startup) at ~3-5 min/cell → ~210 min wall ≈ 3.5 hrs.
# Comfortably fits in 12-hr capacity allocation.
#
# Idempotent: re-running picks up where it left off via CSV status.
#
# Queue note: `capacity` is the right queue for small-node jobs on Aurora.
# Do NOT use `prod` — it requires a 256-node minimum. `capacity` accepts
# arbitrary node counts (we want 4). `debug-scaling` has a 1-hr cap — use
# hold_4node_debug_scaling.sh for interactive 1-hr validation; this
# script for the full round 1 run.

set -euo pipefail

# PRISM_DIR (orchestrator code root) defaults to $PBS_O_WORKDIR (where qsub
# was run from). When submitted from a worktree, this picks up the worktree's
# version of tools/isoflop_launch.py — which has --launcher-arg, while the
# bare clone may not yet (PR #101 still in review). Without this fallback,
# qsub'ing from a worktree pointed PRISM_DIR at the bare clone, the plan
# YAML didn't exist there, and job 8510735 failed in 1 second on 2026-05-28.
PRISM_DIR="${PRISM_DIR:-${PBS_O_WORKDIR:-/lus/flare/projects/ModCon/ngetty/BaseMM_PRISM}}"

# LAUNCHER_PRISM_DIR (where the venv tarball + stage_shards.py live)
# defaults to the bare clone on /flare — that's the only checkout that has
# `deepspeed_env.tar.gz` (the packed venv extracted on each compute node).
# The worktree has only git-tracked files. We forward this to each cell's
# launcher via `--prism-dir` so the launcher's venv staging works.
LAUNCHER_PRISM_DIR="${LAUNCHER_PRISM_DIR:-/flare/ModCon/ngetty/BaseMM_PRISM}"

SCALING_DIR="${SCALING_DIR:-/flare/ModCon/ngetty/BaseMM_PRISM/scaling-study}"
# Plan YAML — committed in the repo under experiments/.
PLAN_YAML="${PLAN_YAML:-$PRISM_DIR/experiments/stage_a_round1_olmo3_1b.yaml}"
CSV_PATH="${CSV_PATH:-$SCALING_DIR/experiments.csv}"
SWEEP_ID="${SWEEP_ID:-STAGE-A-ROUND1-$(date +%Y%m%d)}"
# WebDataset shards location — Pixmo cap, same as Smoke 4 baseline.
WEBDATASET_DIR="${WEBDATASET_DIR:-/flare/ModCon/ngetty/data/zone_a/pixmo_cap_webdataset}"
# Captured at script start so post-run collect filters out any pre-existing
# perf.jsonl files in $PRISM_DIR/outputs/ (e.g. Smoke 4 leftovers in the
# bare clone's outputs/ dir that would otherwise overwrite the cells we
# actually ran in this job). Format: ISO-8601, accepted by --since.
JOB_START_ISO="$(date +%Y-%m-%dT%H:%M:%S)"

echo "================================================================"
echo "Stage A round 1 — capacity batch"
echo "  Job ID:    ${PBS_JOBID:-<no-pbs>}"
echo "  Started:   $(date)"
echo "  PRISM dir: $PRISM_DIR"
echo "  Plan:      $PLAN_YAML"
echo "  CSV:       $CSV_PATH"
echo "  Sweep:     $SWEEP_ID"
echo "  Nodes:"
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "$PBS_NODEFILE" ]; then
    sort -u "$PBS_NODEFILE" | sed 's/^/    /'
else
    echo "    (no PBS_NODEFILE — running outside PBS allocation)"
fi
echo "================================================================"

cd "$PRISM_DIR"

# Activate the project's packed venv. setup_deepspeed_env.sh produces
# .venv-deepspeed; the launcher will extract deepspeed_env.tar.gz to /tmp
# on each compute node, but the orchestrator (isoflop_launch.py) just needs
# yaml + python. Prefer the explicit venv to avoid system py 3.6.
if [ -f "$PRISM_DIR/.venv-deepspeed/bin/activate" ]; then
    source "$PRISM_DIR/.venv-deepspeed/bin/activate"
elif [ -f "/flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu/bin/activate" ]; then
    source /flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu/bin/activate
else
    echo "WARNING: no venv found; using whatever python is on PATH"
fi
which python

# Sanity: plan + cal dir must exist before we burn an hour of capacity time.
if [ ! -f "$PLAN_YAML" ]; then
    echo "FATAL: plan YAML not found: $PLAN_YAML"
    echo "Generate with: python scripts/build_stage_a_plan.py --output $PLAN_YAML"
    exit 1
fi

# Pre-launch recovery: a previous job may have died mid-training, leaving
# cells stuck in status=running OR (worse) status=done with empty loss_main
# (collect saw startup_param_count only). Reset both to planned so
# isoflop_launch picks them back up via idempotency. Without this, the
# resubmit either skips them as "done" or treats them as in-progress on
# another node.
echo ""
echo "--- Pre-launch CSV recovery ($(date)) ---"
python -c "
import csv, sys
path = '$CSV_PATH'
rows = []
n_reset = 0
with open(path) as f:
    rd = csv.DictReader(f)
    cols = rd.fieldnames
    for r in rd:
        if not r['run_id'].startswith('ISO-text_image-OLMO3-1B-'):
            rows.append(r); continue
        # Reset: stuck-running OR done-without-loss (partial training).
        if r['status'] == 'running' or (r['status'] == 'done' and not r['loss_main']):
            r['status'] = 'planned'
            r['launched_at'] = ''
            r['completed_at'] = ''
            for k in ('n_total_params','n_active_params','n_trainable_params',
                     'flops_per_step','cumulative_flops','samples_per_sec',
                     'tokens_per_sec','seq_p50','seq_p95','seq_p99',
                     'padding_ratio','loss_main','loss_caption','loss_source',
                     'd_compute_tokens','d_label_tokens','d_modality_tokens'):
                if k in r: r[k] = ''
            n_reset += 1
        rows.append(r)
with open(path,'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)
print(f'  reset {n_reset} cells (stuck running OR done with empty loss)')
"

# Wipe stale outputs dirs for any cell now back at 'planned'. Otherwise
# isoflop_collect's --force will find old (partial) perf.jsonl files and
# re-mark them 'done' with whatever data those files contain — which
# might be just startup_param_count again.
echo "--- Wiping stale outputs for planned cells ---"
python -c "
import csv, shutil, os
path = '$CSV_PATH'
out_base = '$PRISM_DIR/outputs'
wiped = 0
with open(path) as f:
    for r in csv.DictReader(f):
        if not r['run_id'].startswith('ISO-text_image-OLMO3-1B-'): continue
        if r['status'] != 'planned': continue
        d = os.path.join(out_base, r['run_id'])
        if os.path.isdir(d):
            shutil.rmtree(d)
            wiped += 1
print(f'  wiped {wiped} stale output dirs')
"

# Single isoflop_launch.py invocation runs all 18 cells in plan order.
# launch order = manifest order = (3e17, 1e18, 3e18) × (BASE,W2X,W4X,D2X,D4X)
# × seed=0, followed by BASE@3e17 × seeds 1,2,3. Cheapest cells first means
# a wall-time hit costs the expensive ones, not the variance-floor ones.
# isoflop_launch blocks on each cell's subprocess and writes status=running
# → status=done (via collector) into CSV, so a crash/timeout leaves an
# auditable trail and re-running picks up where it left off.
# Packed venv tarball — lives in the bare clone only. The launcher computes
# `os.path.abspath(args.packed_env)` relative to its cwd, which is the
# worktree if --packed-env is left as the relative default "deepspeed_env.tar.gz".
# Worktree has no tarball → mpiexec ranks see `tar: Cannot open` and all 48
# ranks crash at the `import webdataset` check (the packed env was never
# extracted to /tmp). Pass the absolute path explicitly.
PACKED_ENV="${PACKED_ENV:-$LAUNCHER_PRISM_DIR/deepspeed_env.tar.gz}"
if [ ! -f "$PACKED_ENV" ]; then
    echo "FATAL: packed venv tarball not found: $PACKED_ENV"
    echo "Build with: bash tools/setup_deepspeed_env.sh && tar -czf deepspeed_env.tar.gz .venv-deepspeed"
    exit 1
fi

echo ""
echo "--- Launching all 18 cells ($(date)) ---"
echo "  WebDataset dir:      $WEBDATASET_DIR"
echo "  Launcher PRISM dir:  $LAUNCHER_PRISM_DIR (cwd for launcher subprocess)"
echo "  Packed env tarball:  $PACKED_ENV"
# DL_NUM_WORKERS=0: belt-and-suspenders for the WDS empty-worker-shard
# bug. PR #102 fixed the underlying shard partition (split_by_local_rank
# replaces split_by_node — 154 shards/node / 12 LOCAL ranks ≈ 13 shards/
# rank), and PR for n_workers cleanup lets `+training.data_num_workers=0`
# work from Hydra. But we keep the env override at 0 here because each
# rank's 13 shards / 4 workers ≈ 3 shards/worker is still tight enough
# that any future shard-count drop could re-trigger
# "No samples found in dataset; perhaps you have fewer shards than workers."
# Cost: ~10% slower than 4 workers — irrelevant for Stage A's loss
# convergence fit. (The original failures were jobs 8510834 + 8510922.)
export DL_NUM_WORKERS=0
echo "  DL_NUM_WORKERS:      $DL_NUM_WORKERS (main-process loading, safety net for WDS shard/worker split)"
python tools/isoflop_launch.py \
    --plan "$PLAN_YAML" \
    --csv "$CSV_PATH" \
    --launcher tools/launch_aurora_web.py \
    --sweep-id "$SWEEP_ID" \
    --launcher-arg=--webdataset-dir \
    --launcher-arg="$WEBDATASET_DIR" \
    --launcher-arg=--prism-dir \
    --launcher-arg="$LAUNCHER_PRISM_DIR" \
    --launcher-arg=--packed-env \
    --launcher-arg="$PACKED_ENV" \
    || echo "WARNING: isoflop_launch returned non-zero; some cells may have failed"

echo ""
echo "================================================================"
echo "All cohorts complete (or wall-time hit). Running collect + fit."
echo "================================================================"

# Collect populates n_*_params, loss_main, cumulative_flops, loss_source
# from perf.jsonl into experiments.csv. Each cell's perf.jsonl is under
# <outputs>/<run_id>/<date>/<time>/checkpoints/perf.jsonl. The launcher
# writes Hydra outputs relative to the orchestrator's cwd ($PRISM_DIR, the
# worktree), so collect points at the worktree's outputs dir.
# `--force` re-collects rows already marked `done` — important when an
# earlier run wrote startup_param_count only (training crashed before
# loss data) and the collector flipped status=done with no loss_main.
# With --force we always pull the latest perf.jsonl values; idempotent
# when data is fully present.
python tools/isoflop_collect.py \
    --csv "$CSV_PATH" \
    --outputs "$PRISM_DIR/outputs" \
    --since "$JOB_START_ISO" \
    --sweep-id "$SWEEP_ID" \
    --force \
    || echo "WARNING: isoflop_collect returned non-zero"

# Fit produces per-budget parabolas + cross-budget α/β power-law fit.
RESULTS_DIR="$SCALING_DIR/results"
mkdir -p "$RESULTS_DIR"
python tools/isoflop_fit.py \
    --csv "$CSV_PATH" \
    --family text_image \
    --output-json "$RESULTS_DIR/stage_a_round1.json" \
    --output-md "$RESULTS_DIR/stage_a_round1.md" \
    || echo "WARNING: isoflop_fit returned non-zero"

# Plot emits a 2x2 PNG per (backbone, family) into RESULTS_DIR/figs.
# Tolerant of `--family` rows that don't yet have valid parabolas (Stage A
# round 1 had α≈0 with all valid vertices; round 2's wider N spread will
# exercise the trust-gate visually).
python tools/isoflop_plot.py \
    --csv "$CSV_PATH" \
    --family text_image \
    --outdir "$RESULTS_DIR/figs" \
    || echo "WARNING: isoflop_plot returned non-zero"

echo ""
echo "Stage A round 1 finished: $(date)"
echo "Results JSON: $RESULTS_DIR/stage_a_round1.json"
echo "Results MD:   $RESULTS_DIR/stage_a_round1.md"
echo "Figures:      $RESULTS_DIR/figs/"

#!/usr/bin/env bash
# 1-node CalvinVLADataset smoke (~20 steps) via PRISM-AURORA-ZONE-A-VLA-CALVIN-SMOKE.
#
# This is the regression baseline established in PR 0.5 — every PR that
# touches `src/train.py`, `trainer_zone_a.py`, `trainer_native.py`, or
# `distributed.py` must keep this passing.
#
# Progress logging cadence: VLA goes through ZoneAVLATrainer (Accelerate),
# which logs "Step N: loss=X" every 10 steps from tqdm. So 20 steps produces
# ~2 "Step N: loss=" lines. The native DDP/FSDP path used elsewhere emits
# "[THROUGHPUT] N samples/sec" instead — we grep both patterns to keep one
# pass criterion across all trainers.
#
# Pass criteria (automated):
#   - PBS job completes (state = C) within 30 min
#   - >= 1 of 2 expected progress lines present (guards against mid-run crash)
# Pass criteria (manual):
#   - loss decreases over the window (no published throughput baseline yet)
#
# Usage:
#   tools/parity/smoke_vla.sh [<run-id>] [<min-steps>]
#
#   <run-id>    arbitrary label; defaults to VLA-SMOKE-<timestamp>
#   <min-steps> minimum throughput lines required to pass; defaults to 1
set -euo pipefail

run_id="${1:-VLA-SMOKE-$(date +%Y%m%d-%H%M%S)}"
min_steps="${2:-1}"
max_steps=20
design="PRISM-AURORA-ZONE-A-VLA-CALVIN-SMOKE"

repo_root="$(git rev-parse --show-toplevel)"
logs_dir="${repo_root}/logs/${design}"

submit_out=$(python tools/launch_aurora_daos.py \
    --id "$run_id" \
    --design "$design" \
    --nodes 1 --batch --queue debug \
    --dist-strategy ddp \
    --no-pil4dfs \
    --max-steps "$max_steps" \
    wandb.mode=offline 2>&1)
echo "$submit_out"

job_id=$(echo "$submit_out" | grep 'Job submitted:' | grep -oE '[0-9]+' | head -1)
if [[ -z "$job_id" ]]; then
    echo "ERROR: could not parse job ID from launcher output" >&2
    exit 1
fi
echo "Polling job ${job_id} (timeout 30 min) ..."

deadline=$((SECONDS + 1800))
while [[ $SECONDS -lt $deadline ]]; do
    state=$(qstat -f "$job_id" 2>/dev/null | awk '/job_state/{print $3}' || echo "?")
    [[ "$state" == "C" ]] && break
    echo "  [$(date +%H:%M:%S)] state=${state} — next check in 30s"
    sleep 30
done

if [[ $SECONDS -ge $deadline ]]; then
    echo "FAIL: timeout — job ${job_id} did not complete within 30 min" >&2
    exit 1
fi

log=$(ls "${logs_dir}/${job_id}"*.OU 2>/dev/null | head -1 \
    || ls "${logs_dir}/"*.OU 2>/dev/null | sort | tail -1 || true)
if [[ -z "$log" ]]; then
    echo "FAIL: no *.OU log found in ${logs_dir}/" >&2
    exit 1
fi

step_count=$(grep -cE 'samples/sec|samp/s|Step [0-9]+: loss=' "$log" 2>/dev/null || echo 0)
if [[ "$step_count" -lt "$min_steps" ]]; then
    echo "FAIL: ${step_count} progress lines logged (need ≥${min_steps}) — ${log}" >&2
    exit 1
fi

echo "PASS: ${step_count} progress lines logged (max_steps=${max_steps}) — ${log}"
echo "Review loss trajectory: grep 'loss' ${log} | tail -20"

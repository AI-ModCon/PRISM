#!/usr/bin/env bash
# 1-node VLA smoke (~20 steps) via the per-modality WebDataset loader.
#
# Companion to smoke_vla.sh (map-style baseline). Drives the
# `training.calvin_loader=webdataset` branch in src/train.py which routes
# CALVIN through ModalityAwareWebDatasetWrapper(modalities=["vla"]). VLA
# stays on Accelerate (--dist-strategy ddp; native DDP/FSDP is guarded
# against at the ZoneAVLATrainer init).
#
# Prereqs:
#   - applications/vla/shard_calvin_vla.py has been run with --out under the path
#     declared in src/conf/data/lustre_datasets.yaml `calvin` group, OR
#     CALVIN_WEBDATASET_ROOT is exported and points at a directory with
#     manifest.json + shards/.
#
# Progress logging cadence: identical to smoke_vla.sh — ZoneAVLATrainer
# logs "Step N: loss=X" every log_every_n_steps; we grep both patterns
# (`samples/sec`, `samp/s`, `Step N: loss=`) to stay tolerant.
#
# Pass criteria (automated):
#   - PBS job completes (state = C) within 45 min
#   - >= 1 progress line in the log (guards against silent rank kill)
# Pass criteria (manual):
#   - loss decreases over the 20-step window
#   - throughput within 10% of map-style smoke_vla.sh baseline
#   - per-dim action_MSE printed in the log
#
# Usage:
#   tools/parity/smoke_vla_web.sh [<run-id>] [<min-steps>]
#
#   <run-id>    arbitrary label; defaults to VLA-WEB-SMOKE-<timestamp>
#   <min-steps> minimum progress lines required to pass; defaults to 1
set -euo pipefail

run_id="${1:-VLA-WEB-SMOKE-$(date +%Y%m%d-%H%M%S)}"
min_steps="${2:-1}"
max_steps=20
design="PRISM-AURORA-ZONE-A-VLA-CALVIN-SMOKE"

repo_root="$(git rev-parse --show-toplevel)"
logs_dir="${repo_root}/logs/${design}"

# Forward CALVIN_WEBDATASET_ROOT if the user exported one, so launchers
# pick it up as a Hydra override. Default falls back to the lustre group's
# `path`, which assumes shards live at /flare/ModCon/sww/vla_training/calvin_webdataset/.
calvin_overrides=(
    "training.calvin_loader=webdataset"
    "training.calvin_webdataset_storage=lustre"
)
if [[ -n "${CALVIN_WEBDATASET_ROOT:-}" ]]; then
    calvin_overrides+=("training.calvin_webdataset_root=${CALVIN_WEBDATASET_ROOT}")
fi

submit_out=$(python tools/launch_aurora_web.py \
    --id "$run_id" \
    --design "$design" \
    --nodes 1 --batch --queue debug \
    --shared-hf-home "${SHARED_HF_HOME:-/flare/ModCon/ngetty/huggingface/hub}" \
    --dist-strategy ddp \
    --max-steps "$max_steps" \
    "${calvin_overrides[@]}" \
    wandb.mode=offline 2>&1)
echo "$submit_out"

job_id=$(echo "$submit_out" | grep 'Job submitted:' | grep -oE '[0-9]+' | head -1)
if [[ -z "$job_id" ]]; then
    echo "ERROR: could not parse job ID from launcher output" >&2
    exit 1
fi
echo "Polling job ${job_id} (timeout 45 min) ..."

deadline=$((SECONDS + 2700))
while [[ $SECONDS -lt $deadline ]]; do
    state=$(qstat -f "$job_id" 2>/dev/null | awk '/job_state/{print $3}' || echo "?")
    [[ "$state" == "C" ]] && break
    echo "  [$(date +%H:%M:%S)] state=${state} — next check in 30s"
    sleep 30
done

if [[ $SECONDS -ge $deadline ]]; then
    echo "FAIL: timeout — job ${job_id} did not complete within 45 min" >&2
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

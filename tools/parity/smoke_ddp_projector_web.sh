#!/usr/bin/env bash
# 1-node DDP projector-only smoke (~50 steps) via Lustre/WebDataset path.
#
# Lustre-backed counterpart to smoke_ddp_projector.sh — used when DAOS is
# unavailable (e.g. daos_user_fs resource off). Uses tools/launch_aurora_web.py
# with --webdataset-dir pointed at pixmo-cap on /flare.
#
# Throughput logging cadence: trainer emits "[THROUGHPUT] N samples/sec"
# every 10 steps, plus a final "Throughput: N samp/s" summary. So 50 steps
# produces ~5 [THROUGHPUT] lines + 1 summary = 6 total throughput lines.
# We count both patterns; pass at >=4 of the expected ~5 cadence lines.
#
# Pass criteria (automated):
#   - PBS job completes (state = C) within 45 min (longer than DAOS — adds
#     shard-staging time from Lustre → /tmp at job start)
#   - >= 4 of 5 expected throughput lines present (guards against mid-run crash)
# Pass criteria (manual):
#   - throughput within 10% of the PRISM-IMAGE-ONLY-2N DDP baseline
#     (Lustre will be slower than DAOS due to staging — expect noisier number)
#
# Usage:
#   tools/parity/smoke_ddp_projector_web.sh [<run-id>] [<min-steps>]
#
#   <run-id>    arbitrary label; defaults to DDP-WEB-SMOKE-<timestamp>
#   <min-steps> minimum throughput lines required to pass; defaults to 4
set -euo pipefail

run_id="${1:-DDP-WEB-SMOKE-$(date +%Y%m%d-%H%M%S)}"
min_steps="${2:-4}"
max_steps=50
design="PRISM-IMAGE-ONLY-2N"
webdataset_dir="${WEBDATASET_DIR:-/flare/ModCon/ngetty/data/zone_a/pixmo_cap_webdataset}"

repo_root="$(git rev-parse --show-toplevel)"
logs_dir="${repo_root}/logs/${design}"

submit_out=$(python tools/launch_aurora_web.py \
    --id "$run_id" \
    --design "$design" \
    --nodes 1 --batch --queue debug \
    --webdataset-dir "$webdataset_dir" \
    --shared-hf-home "${SHARED_HF_HOME:-/flare/ModCon/ngetty/huggingface/hub}" \
    --dist-strategy ddp \
    --max-seq-length 1024 \
    --use-bucketing \
    --max-steps "$max_steps" 2>&1)
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

step_count=$(grep -cE 'samples/sec|samp/s' "$log" 2>/dev/null || echo 0)
if [[ "$step_count" -lt "$min_steps" ]]; then
    echo "FAIL: ${step_count} throughput lines logged (need ≥${min_steps}) — ${log}" >&2
    exit 1
fi

echo "PASS: ${step_count} throughput lines logged (max_steps=${max_steps}) — ${log}"
echo "Review throughput: grep -E 'samples/sec|samp/s' ${log} | awk '{print \$(NF-2), \$(NF-1)}'"

#!/usr/bin/env bash
# 2-node FSDP E2E smoke (~30 steps) via Lustre/WebDataset path.
#
# Lustre-backed counterpart to smoke_fsdp_e2e.sh — used when DAOS is
# unavailable (e.g. daos_user_fs resource off). Uses tools/launch_aurora_web.py
# with --webdataset-dir pointed at pixmo-cap on /flare.
#
# Throughput logging cadence: trainer emits "[THROUGHPUT] N samples/sec"
# every 10 steps, plus a final "Throughput: N samp/s" summary. So 30 steps
# produces ~3 [THROUGHPUT] lines + 1 summary = 4 total throughput lines.
# We count both patterns; pass at >=2 of the expected ~3 cadence lines.
#
# Pass criteria (automated):
#   - PBS job completes (state = C) within 60 min (longer than DAOS — adds
#     shard-staging time from Lustre → /tmp at job start)
#   - >= 2 of 3 expected throughput lines present (guards against mid-run crash)
# Pass criteria (manual):
#   - throughput within 10% of the PRISM-OLMO3-E2E-PROD FSDP baseline
#     (Lustre will be slower than DAOS due to staging — expect noisier number)
#
# Usage:
#   tools/parity/smoke_fsdp_e2e_web.sh [<run-id>] [<min-steps>]
#
#   <run-id>    arbitrary label; defaults to FSDP-WEB-SMOKE-<timestamp>
#   <min-steps> minimum throughput lines required to pass; defaults to 2
set -euo pipefail

run_id="${1:-FSDP-WEB-SMOKE-$(date +%Y%m%d-%H%M%S)}"
min_steps="${2:-2}"
max_steps=30
design="PRISM-OLMO3-E2E-PROD"
webdataset_dir="${WEBDATASET_DIR:-/flare/ModCon/ngetty/data/zone_a/pixmo_cap_webdataset}"

repo_root="$(git rev-parse --show-toplevel)"
logs_dir="${repo_root}/logs/${design}"

submit_out=$(python tools/launch_aurora_web.py \
    --id "$run_id" \
    --design "$design" \
    --nodes 2 --batch --queue debug-scaling \
    --webdataset-dir "$webdataset_dir" \
    --shared-hf-home "${SHARED_HF_HOME:-/flare/ModCon/ngetty/huggingface/hub}" \
    --dist-strategy fsdp --fsdp-sharding full_shard \
    --max-seq-length 1024 \
    --use-bucketing \
    --fsdp-production-mode \
    --max-steps "$max_steps" \
    training.batch_size=4 2>&1)
echo "$submit_out"

job_id=$(echo "$submit_out" | grep 'Job submitted:' | grep -oE '[0-9]+' | head -1)
if [[ -z "$job_id" ]]; then
    echo "ERROR: could not parse job ID from launcher output" >&2
    exit 1
fi
echo "Polling job ${job_id} (timeout 60 min) ..."

deadline=$((SECONDS + 3600))
while [[ $SECONDS -lt $deadline ]]; do
    state=$(qstat -f "$job_id" 2>/dev/null | awk '/job_state/{print $3}' || echo "?")
    [[ "$state" == "C" ]] && break
    echo "  [$(date +%H:%M:%S)] state=${state} — next check in 30s"
    sleep 30
done

if [[ $SECONDS -ge $deadline ]]; then
    echo "FAIL: timeout — job ${job_id} did not complete within 60 min" >&2
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

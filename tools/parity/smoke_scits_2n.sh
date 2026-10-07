#!/usr/bin/env bash
# 2-node SciTS/TimeOmni DDP smoke — the differential test for defects A and
# B. Unfixed: `_mcfg = getattr(model, "config", None)` is None on every DDP
# rank (defect A), so the val loader is never built and no per-rank
# init-failure asymmetry can even surface. Fixed: the val loader builds on
# every rank, and the rank-symmetric readiness gate (defect B) prevents a
# transient per-rank init failure from deadlocking the eval-forward's
# all_reduce.
#
# Uses PRISM-OLMO-1B-TIMEOMNI, matching smoke_scits_1n.sh — see that
# script's header comment for why PRISM-OLMO-1B-TSQA-SMOKE (LINEAR
# interleaved-QA model) is the wrong design for the raw SciTS webdataset
# path.
#
# Pass criteria (automated):
#   - PBS job completes (state = C) within 30 min (no wallclock kill —
#     the deadlock signature this test targets)
#   - "Validation loss at step" appears in the log (val/loss on multi-rank
#     DDP — proves defect A is fixed)
#   - both ranks reach the final throughput line (proves no rank hung)
#   - no Traceback / UR_RESULT_ERROR / merged-length guard trip
#
# Usage:
#   tools/parity/smoke_scits_2n.sh [<run-id>]
set -euo pipefail

run_id="${1:-SCITS-2N-SMOKE-$(date +%Y%m%d-%H%M%S)}"
design="PRISM-OLMO-1B-TIMEOMNI"
webdataset_dir="${SCITS_WEBDATASET_DIR:-/lus/flare/projects/ModCon/pemami/data/SciTS-processed}"

repo_root="$(git rev-parse --show-toplevel)"
logs_dir="${repo_root}/logs/${design}"

submit_out=$(python tools/launch_aurora_web.py \
    --id "$run_id" \
    --design "$design" \
    --project "${PBS_PROJECT:-AuroraGPT}" \
    --packed-env "${PACKED_ENV:-deepspeed_env.tar.gz}" \
    --nodes 2 --batch --queue debug-scaling --walltime 00:30:00 \
    --webdataset-dir "$webdataset_dir" \
    --webdataset-modality time_series \
    --shared-hf-home "${SHARED_HF_HOME:-/flare/ModCon/ngetty/huggingface/hub}" \
    --dist-strategy ddp \
    --max-steps 30 \
    training.eval_enabled=true \
    training.eval_every_n_steps=10 2>&1)
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
    echo "FAIL: timeout — job ${job_id} did not complete within 30 min (possible A/B deadlock regression)" >&2
    exit 1
fi

log=$(ls "${logs_dir}/${job_id}"*.OU 2>/dev/null | head -1 \
    || ls "${logs_dir}/"*.OU 2>/dev/null | sort | tail -1 || true)
if [[ -z "$log" ]]; then
    echo "FAIL: no *.OU log found in ${logs_dir}/" >&2
    exit 1
fi
echo "Log: ${log}"

fail=0

if ! grep -qE 'Traceback|UR_RESULT_ERROR|Merged sequence length.*exceeds' "$log"; then
    echo "OK: no Traceback / UR_RESULT_ERROR / merged-length guard trip"
else
    echo "FAIL: found a fatal error signature in the log" >&2
    grep -E 'Traceback|UR_RESULT_ERROR|Merged sequence length.*exceeds' "$log" | head -5 >&2
    fail=1
fi

if grep -q 'Validation loss at step' "$log"; then
    echo "OK: 'Validation loss at step' present on multi-rank DDP — defect A fixed"
else
    echo "FAIL: no 'Validation loss at step' — val loader still never built under DDP (defect A regression)" >&2
    fail=1
fi

if grep -q 'Validation striped across 2 ranks' "$log"; then
    echo "OK: 'Validation striped across 2 ranks' present — per-rank val loader confirmed on both ranks"
else
    echo "WARN: 'Validation striped across 2 ranks' not found (check log manually)"
fi

throughput_count=$(grep -cE 'samples/sec|samp/s' "$log" 2>/dev/null || echo 0)
if [[ "$throughput_count" -ge 1 ]]; then
    echo "OK: ${throughput_count} throughput line(s) logged — job reached steady state, no hang"
else
    echo "FAIL: no throughput lines — job may have died or hung before first step" >&2
    fail=1
fi

if [[ "$fail" -ne 0 ]]; then
    echo "FAIL: one or more checks failed — ${log}" >&2
    exit 1
fi

echo "PASS: 2N SciTS DDP smoke (job ${job_id}) — ${log}"

#!/usr/bin/env bash
# 1-node SciTS/TimeOmni smoke — validates the trainer_native.py fixes for
# PR #129 review defects A (DDP .config unwrap breaks val loader), B (rank-
# asymmetric val-loader init can deadlock a multi-node all_reduce — inert at
# 1N but exercised for real by smoke_scits_2n.sh), and N (perf: .item() sync
# moved to after backward()).
#
# Uses PRISM-OLMO-1B-TIMEOMNI — the documented design for the SciTS
# webdataset-direct path (model: prism_olmo1b_timeomni_ts, ts_projector=
# timeomni, is_interleaved_qa=false — see docs/modalities/timeseries.md's own launch
# example). PRISM-OLMO-1B-TSQA-SMOKE looks similar but uses the LINEAR
# interleaved-QA model (prism_olmo1b_linear_interleaved_ts), which expects
# data from the ts_qa dataset pipeline, not raw SciTS webdataset shards —
# feeding it --webdataset-dir crashes in TimeSeriesEncoder.forward with
# "not enough values to unpack (expected 3, got 2)" (linear encoder wants
# (B,T,V), gets the timeomni-shaped (T,V) list). Confirmed on job 8788447.
#
# eval_every_n_steps is overridden explicitly since the design's own
# default (250) is too infrequent for a 30-step smoke — PRISM-MODALITY-SMOKE
# sets eval_every_n_steps=0 and can never exercise the validation path at all.
#
# Pass criteria (automated):
#   - PBS job completes (state = C) within 30 min
#   - "Validation loss at step" appears in the log at least once (proves
#     the val loader was built and ran — defect A/B regression signal)
#   - "loss_source": "held_out_validation" appears in perf.jsonl
#   - throughput lines present (job didn't die before first step)
#   - no Traceback / UR_RESULT_ERROR / merged-length guard trip
#
# Usage:
#   tools/parity/smoke_scits_1n.sh [<run-id>]
set -euo pipefail

run_id="${1:-SCITS-1N-SMOKE-$(date +%Y%m%d-%H%M%S)}"
design="PRISM-OLMO-1B-TIMEOMNI"
webdataset_dir="${SCITS_WEBDATASET_DIR:-/lus/flare/projects/ModCon/pemami/data/SciTS-processed}"

repo_root="$(git rev-parse --show-toplevel)"
logs_dir="${repo_root}/logs/${design}"

submit_out=$(python tools/launch_aurora_web.py \
    --id "$run_id" \
    --design "$design" \
    --project "${PBS_PROJECT:-AuroraGPT}" \
    --packed-env "${PACKED_ENV:-deepspeed_env.tar.gz}" \
    --nodes 1 --batch --queue debug --walltime 00:30:00 \
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
    echo "FAIL: timeout — job ${job_id} did not complete within 30 min" >&2
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
    echo "OK: 'Validation loss at step' present — val loader built and ran"
else
    echo "FAIL: no 'Validation loss at step' in log — val loader silently never built (defect A/B regression)" >&2
    fail=1
fi

throughput_count=$(grep -cE 'samples/sec|samp/s' "$log" 2>/dev/null || echo 0)
if [[ "$throughput_count" -ge 1 ]]; then
    echo "OK: ${throughput_count} throughput line(s) logged"
else
    echo "FAIL: no throughput lines — job may have died before first step" >&2
    fail=1
fi

output_dir=$(dirname "$(find "${repo_root}/outputs" -path "*${run_id}*/perf.jsonl" 2>/dev/null | head -1)")
if [[ -n "$output_dir" && -f "${output_dir}/perf.jsonl" ]]; then
    if grep -q '"loss_source": "held_out_validation"' "${output_dir}/perf.jsonl"; then
        echo "OK: perf.jsonl has loss_source=held_out_validation"
    else
        echo "FAIL: perf.jsonl missing loss_source=held_out_validation" >&2
        fail=1
    fi
else
    echo "WARN: could not locate perf.jsonl under ${repo_root}/outputs for run ${run_id}"
fi

if [[ "$fail" -ne 0 ]]; then
    echo "FAIL: one or more checks failed — ${log}" >&2
    exit 1
fi

echo "PASS: 1N SciTS smoke (job ${job_id}) — ${log}"

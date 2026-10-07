#!/usr/bin/env bash
# IsoFLOP PR-1 smoke skeleton — DO NOT EXECUTE without holding a node.
#
# This script documents the two manual validation steps for PR-1
# (M1 instrumentation + M2 projector knobs + tools/isoflop_calibrate.py).
# It is intentionally NOT auto-runnable: each smoke needs an interactive
# PBS hold or a self-submitted batch job, and the pass criteria are
# inspected by hand against `outputs/<ID>/perf.jsonl`.
#
# Mirrors the convention in tools/parity/*.sh — skeletons + qsub blocks
# captured in source so the next operator (or future Claude) knows
# exactly what to run.
#
# Real numbers will be captured during execution and pasted into the
# PR description; the PLAN.md verification table tracks completion.

set -euo pipefail

cat <<'NOTE'
================================================================
IsoFLOP PR-1 smoke skeleton — manual execution only.

Two smokes:
  1. Instrumented training round-trip (perf.jsonl gains 6 new key
     categories).
  2. Calibration JSON written for one variant.

DO NOT run this script as-is — copy/paste the qsub blocks below
into your shell after holding a node.
================================================================
NOTE

cat <<'SMOKE1'
================================================================
[Smoke 1] Instrumented training (1N debug, ~30 steps)

# From a node you hold (see tools/hold_one_node.sh):
python tools/launch_aurora_daos.py \
  --id ISO-CAL-SMOKE-PR1 \
  --design PRISM-MODALITY-SMOKE-1N \
  --nodes 1 --queue debug --walltime 00:20:00 \
  --max-steps 30 \
  model.projector_hidden_mult=2 model.projector_num_layers=2 \
  model.modalities=[text,image]

Pass criteria (inspect outputs/ISO-CAL-SMOKE-PR1/.../perf.jsonl):
  - exactly one `event=startup_param_count` record with total > 0
  - throughput records contain non-null seq_p50, seq_p95, seq_p99,
    seq_max, padding_ratio
  - projector_hidden_mult=2 echoed in every throughput record
  - run exits 0, no OOM, no projector shape mismatch
================================================================
SMOKE1

cat <<'SMOKE2'
================================================================
[Smoke 2] Calibration write (1N debug, ~5 min on OLMo-3 1B)

# Hold a node first:
qsub tools/hold_one_node.sh

# In the held shell, activate the env then:
python tools/isoflop_calibrate.py \
  --backbone allenai/OLMo-2-1B-1124-hf \
  --projector-variant BASE \
  --family text_image \
  --regime projector_only \
  --steps 30 \
  --output /flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/calibration/OLMO3-1B-BASE-text_image-projector_only.json

Pass criteria:
  - JSON file written
  - flops_per_step > 0
  - n_total_params > 1e9
  - re-running without --force exits 0 with "already calibrated"
    and unchanged mtime
  - re-running with --force overwrites and updates calibrated_at

After Smoke 2: repeat for W2X, W4X, D2X, D4X so the scaling-study
calibration dir holds 5 JSONs ready for PR-2's isoflop_plan.py.
================================================================
SMOKE2

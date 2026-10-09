#!/usr/bin/env bash
# 1-node ZeRO-2 30-step E2E smoke. Used by PR 5.
#
# Skeleton — populated when PR 5 introduces `--deepspeed` and the
# `aurora_deepspeed_zero2.yaml` accelerate config.
set -euo pipefail

run_id="${1:-Z2-SMOKE-$(date +%Y%m%d-%H%M%S)}"

echo "ZeRO-2 smoke skeleton — wire up in PR 5 (run id: $run_id)." >&2
exit 64

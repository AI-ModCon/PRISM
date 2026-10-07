#!/bin/bash
# Runner for tools/vllm_eval_time_series.py on a compute node with the
# frameworks/2025.3.1 python (vLLM 0.15.0+xpu).
#
# Resolves PRISM_DIR via $PRISM_DIR / git toplevel / pwd. Override with
# PRISM_DIR=...
#
# `set -u` (nounset) is intentionally omitted — Aurora's Lmod `module load`
# reads unset shell vars and crashes under nounset.
set -eo pipefail

PROJ="${PRISM_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJ"

module load frameworks/2025.3.1 2>&1 | tail -1
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
exec "$PY" tools/vllm_eval_time_series.py "$@"

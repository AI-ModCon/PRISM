#!/bin/bash
# Wrapper to run vllm_parity.py on a compute node with the frameworks/2025.3.1
# python (which ships vLLM 0.15.0+xpu in-tree).
#
# Resolves PRISM_DIR via $PRISM_DIR / git toplevel / pwd. Override with PRISM_DIR=...
set -e
PROJ="${PRISM_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJ"
module load frameworks/2025.3.1 2>&1 | tail -1
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
exec "$PY" tools/vllm_parity.py "$@"

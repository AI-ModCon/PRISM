#!/bin/bash
# setup_deepspeed_env.sh — Tarball workflow wrapper around build_aurora_env.sh.
#
# Builds the venv at .venv-deepspeed/ in the repo root so the existing
#   tar -czf deepspeed_env.tar.gz .venv-deepspeed
# step keeps working for users without /flare write access (Perlmutter, CI).
#
# For shared-venv workflows on /flare, set VENV_PATH and call
# tools/build_aurora_env.sh directly:
#
#   VENV_PATH=/flare/ModCon/$USER/prism-envs/py3.12 bash tools/build_aurora_env.sh
#
# Usage:
#   bash tools/setup_deepspeed_env.sh            # builds if missing, errors on stale
#   bash tools/setup_deepspeed_env.sh --rebuild  # tears down and rebuilds

set -eo pipefail

PRISM_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export VENV_PATH="$PRISM_DIR/.venv-deepspeed"
# Keep in sync with build_aurora_env.sh — overridable so a future point
# release on the same py3.12 + torch 2.10 ABI can reuse the lockfile.
FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks/2025.3.1}"
export FRAMEWORKS_MODULE

REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

echo "=== PRISM DeepSpeed Environment Setup on Aurora ==="
echo "Project Dir: $PRISM_DIR"
echo "Venv Path:   $VENV_PATH (tarball workflow)"

# If the venv exists and --rebuild was not passed, do a quick staleness check
# (interpreter symlink) before handing off. build_aurora_env.sh would refuse
# anyway, but we want the original "Skipping creation" fast path to keep
# working when nothing has drifted.
if [ -d "$VENV_PATH" ] && [ "$REBUILD" -ne 1 ]; then
    # Lazy-load frameworks just for the readlink comparison.
    module load "$FRAMEWORKS_MODULE"
    venv_python_real=$(readlink -f "$VENV_PATH/bin/python" 2>/dev/null || true)
    system_python_real=$(readlink -f "$(which python)" 2>/dev/null || true)
    if [ -z "$venv_python_real" ] || [ -z "$system_python_real" ]; then
        echo "ERROR: unable to resolve venv or system python interpreter."
        echo "  venv:   $VENV_PATH/bin/python -> ${venv_python_real:-MISSING}"
        echo "  system: $(which python) -> ${system_python_real:-MISSING}"
        echo "Rebuild: bash tools/setup_deepspeed_env.sh --rebuild"
        exit 1
    fi
    if [ "$venv_python_real" != "$system_python_real" ]; then
        echo "ERROR: $VENV_PATH was built against a different frameworks python."
        echo "  venv python:   $venv_python_real"
        echo "  module python: $system_python_real"
        echo "Rebuild: bash tools/setup_deepspeed_env.sh --rebuild"
        exit 1
    fi
    echo "Existing venv matches loaded frameworks; nothing to do."
    echo "  source $VENV_PATH/bin/activate"
    exit 0
fi

# Hand off to the shared builder. It loads frameworks, creates the venv,
# installs from the lockfile + nodeps + editable walrus, runs the verify
# block, and writes PRISM_BUILD_INFO.
#
# NOTE: do not use `${REBUILD:+--rebuild}` here — REBUILD=0 is non-empty,
# so :+ would always expand. Use an array conditioned on the integer value.
extra_args=()
[ "$REBUILD" -eq 1 ] && extra_args+=(--rebuild)
exec bash "$PRISM_DIR/tools/build_aurora_env.sh" "${extra_args[@]}"

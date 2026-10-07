#!/bin/bash
# Install the PRISM entry-point metadata so vLLM auto-discovers
# `src.vllm_plugin:register` at engine boot.
#
# Aurora runs PRISM out of a git clone (PYTHONPATH+module imports). The
# `[project.entry-points."vllm.general_plugins"]` declaration in
# pyproject.toml only takes effect once the package's dist-info is on the
# import path. `pip install -e . --no-deps --user` writes that metadata
# without pulling any new packages.
#
# After running this once, `tools/vllm_serve.py` and `tools/vllm_parity.py`
# no longer need their imperative `src.vllm_plugin.register()` call (they
# keep it as a safety net). The serve-smoke runner exercises this exact
# path with PYTHONPATH unset.
#
# Usage:
#   module load frameworks/2025.3.1
#   bash tools/install_prism_entry_point.sh
#
# Re-running after changing the entry-point name/target in pyproject.toml
# just rewrites the dist-info — no manual uninstall needed. To remove the
# registration entirely: `pip uninstall prism-mm` (then the imperative
# `register()` fallback in tools/vllm_*.py takes over again).
#
# Note the two different names: the distribution is `prism-mm` (the PyPI name,
# what pip installs and uninstalls), while `prism` is the entry-point name the
# check below asserts on, and also the console-script name. They are not
# interchangeable — `pip uninstall prism` is a silent no-op.
#
# `set -u` intentionally omitted — Aurora Lmod hits unset vars.
set -eo pipefail

PROJ="${PRISM_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJ"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[install] active venv: $VIRTUAL_ENV — installing into it"
    PIP_TARGET=()
else
    echo "[install] no venv detected — installing --user"
    PIP_TARGET=(--user)
fi

# Resolve Python in this order: explicit PY/PYTHON env var → whatever the
# active module/venv put on PATH → bail. No hardcoded Aurora frameworks
# path: every frameworks bump (we already crossed 2025.2.0 → 2025.3.1)
# would silently rot it.
PY="${PY:-${PYTHON:-$(command -v python3 || command -v python || true)}}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
    echo "[install] ERROR: no python found on PATH; \`module load frameworks/...\` first or set PY=" >&2
    exit 1
fi
echo "[install] using: $PY"

# --user installs land in a per-Python site dir (~/.local/lib/pythonX.Y/),
# so the dist-info is only visible to interpreters with matching X.Y. Warn
# if the user is running this without a venv on a Python whose site-dir
# differs from what they'll likely use to launch vLLM later.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    PY_VER="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    echo "[install] --user install targets ~/.local/lib/python${PY_VER}/site-packages"
    echo "[install] re-run with the same Python you'll launch vLLM under, or use a venv"
fi

"$PY" -m pip install -e . --no-deps "${PIP_TARGET[@]}"

echo "[install] verifying entry point …"
"$PY" - <<'PY'
import sys
from importlib.metadata import entry_points
eps = entry_points(group="vllm.general_plugins")
names = sorted(ep.name for ep in eps)
print(f"vllm.general_plugins entry points: {names}")
if "prism" not in names:
    print("ERROR: 'prism' entry point not registered", file=sys.stderr)
    sys.exit(1)
print("[install] prism entry point registered")
PY

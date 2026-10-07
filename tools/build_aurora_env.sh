#!/bin/bash
# build_aurora_env.sh — Build a PRISM venv at $VENV_PATH using uv + the
# checked-in lockfile. Supports both shared-venv (on /flare) and tarball
# (.venv-deepspeed) workflows.
#
# Usage:
#   VENV_PATH=/flare/ModCon/<user>/prism-envs/py3.12 bash tools/build_aurora_env.sh
#   VENV_PATH=/flare/ModCon/<user>/prism-envs/py3.12 bash tools/build_aurora_env.sh --rebuild
#   VENV_PATH=/flare/ModCon/<user>/prism-envs/py3.12-interns2 bash tools/build_aurora_env.sh --intern-s2
#
# Two mutually exclusive variants:
#
#   default      transformers stays at the system 4.57.6. vLLM 0.15.0+xpu is
#                inside its declared transformers<5,>=4.56.0 range, OLMo-3's
#                >=4.57.0 floor is met, and the `intern_s2` / `intern_s2_397b`
#                time-series encoders are UNAVAILABLE.
#
#   --intern-s2  overlays transformers==5.2.0 (plus the huggingface-hub/hf_xet
#                versions it is coupled to) so those encoders import. This
#                breaks vLLM's declared range; the build prints a warning and
#                continues, because vLLM is a system package, not ours.
#
# The intersection of the two constraints is empty, which is why this is a
# flag and not a default. Build a second venv on a separate VENV_PATH rather
# than rebuilding one in place if you need both.
#
# Requires:
#   - uv (install: curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - module load frameworks/2025.3.1 (loaded by this script)
#
# Refuses to overwrite an existing $VENV_PATH without --rebuild because
# `uv pip sync` is destructive (removes packages not in lockfile).

# NOTE: no `-u` — module load's Lmod init touches ZSH_EVAL_CONTEXT,
# which is unset in non-zsh shells and would abort under set -u.
set -eo pipefail

# FRAMEWORKS_MODULE can be overridden in the environment so the same lockfile
# can be reused across point releases that share the py3.12 + torch 2.10 ABI
# (e.g. a future 2025.4) without editing this script.
FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks/2025.3.1}"
PRISM_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
LOCKFILE="$PRISM_DIR/requirements/aurora-py3.12.lock.txt"
NODEPSFILE="$PRISM_DIR/requirements/aurora-py3.12.nodeps.txt"
INTERN_S2_FILE="$PRISM_DIR/requirements/aurora-py3.12.intern-s2.nodeps.txt"
WALRUS_DIR="$PRISM_DIR/src/libs/walrus"

REBUILD=0
# PRISM_INTERN_S2 can also be set in the environment, so setup_deepspeed_env.sh
# and job scripts can select the variant without threading a flag through.
INTERN_S2="${PRISM_INTERN_S2:-0}"
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        --intern-s2) INTERN_S2=1 ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

if [ "$INTERN_S2" = "1" ]; then
    VARIANT="intern-s2 (transformers 5.2.0 overlay; vLLM out of its declared range)"
else
    VARIANT="default (system transformers 4.57.6; intern_s2 encoders disabled)"
fi

if [ -z "${VENV_PATH:-}" ]; then
    echo "ERROR: VENV_PATH is not set."
    echo "  Shared venv example: VENV_PATH=/flare/ModCon/\$USER/prism-envs/py3.12 bash tools/build_aurora_env.sh"
    echo "  Tarball workflow:    use tools/setup_deepspeed_env.sh (thin wrapper)"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not on PATH."
    echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== PRISM Aurora venv build ==="
echo "  PRISM dir:  $PRISM_DIR"
echo "  VENV_PATH:  $VENV_PATH"
echo "  Lockfile:   $LOCKFILE"
echo "  Variant:    $VARIANT"
echo "  uv:         $(uv --version)"

# Refuse to clobber an existing venv unless --rebuild was passed.
# `uv pip sync` removes packages not in the lockfile, which would silently
# corrupt another user's environment on a shared path.
if [ -e "$VENV_PATH" ] && [ "$REBUILD" -ne 1 ]; then
    echo "ERROR: $VENV_PATH already exists. Pass --rebuild to recreate it."
    echo "  (uv pip sync is destructive and could clobber other users on a shared path.)"
    exit 1
fi

echo "[1/7] Loading $FRAMEWORKS_MODULE..."
module load "$FRAMEWORKS_MODULE"

SYSTEM_PYTHON="$(which python)"
SYSTEM_PYTHON_REAL="$(readlink -f "$SYSTEM_PYTHON")"
echo "  system python: $SYSTEM_PYTHON_REAL"

# Proxy (compute-node builds; UAN can hit PyPI directly)
if [[ ! "${HOSTNAME}" =~ aurora-uan ]]; then
    export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
    export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
    export http_proxy="$HTTP_PROXY"
    export https_proxy="$HTTPS_PROXY"
fi

echo "[2/7] Creating venv at $VENV_PATH (--system-site-packages)..."
if [ -e "$VENV_PATH" ]; then
    rm -rf "$VENV_PATH"
fi
mkdir -p "$(dirname "$VENV_PATH")"
uv venv --system-site-packages -p "$SYSTEM_PYTHON" "$VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "[3/7] Installing lockfile (uv pip install --no-deps)..."
# We use install --no-deps (not sync) because:
# - The lockfile is the *delta* over system site-packages, so transitive deps
#   like torch / numpy are already satisfied by the frameworks module.
# - sync would try to remove system-installed packages it doesn't recognize.
# - --no-deps prevents re-resolving and pulling cu12 wheels from PyPI.
uv pip install --no-deps -r "$LOCKFILE"

echo "[4/7] Installing nodeps tail (uv pip install --no-deps)..."
uv pip install --no-deps -r "$NODEPSFILE"
if [ "$INTERN_S2" = "1" ]; then
    echo "  + Intern-S2 overlay: $INTERN_S2_FILE"
    uv pip install --no-deps -r "$INTERN_S2_FILE"
else
    echo "  (skipping Intern-S2 overlay; pass --intern-s2 to install it)"
fi

echo "[5/7] Installing walrus (editable, --no-deps)..."
if [ -z "$(ls -A "$WALRUS_DIR" 2>/dev/null)" ]; then
    echo "ERROR: $WALRUS_DIR is empty. Initialize the submodule first:"
    echo "  git clone git@github.com:PolymathicAI/walrus.git $WALRUS_DIR"
    exit 1
fi
uv pip install --no-deps -e "$WALRUS_DIR"

echo "[6/7] Verifying modality dependencies..."
# Same import-check matrix as setup_deepspeed_env.sh — lifted verbatim so
# both paths fail at build time, not job start. Also asserts torch loads
# from system site-packages (not the venv) — catches --system-site-packages
# path-order regressions.
PRISM_INTERN_S2="$INTERN_S2" python - <<'PYEOF' || { echo "ERROR: modality dependency verification failed"; exit 1; }
import importlib, importlib.metadata, os, sys
from packaging.version import Version

# Which variant was built. The floors below differ between the two, so read it
# from the environment rather than hardcoding one and hoping.
intern_s2 = os.environ.get("PRISM_INTERN_S2") == "1"

required = {
    "torch": None,
    "torch_geometric": None,    # graph
    "walrus": None,              # geometry
    "the_well": None,            # geometry
    "hydra": None,               # geometry (hydra-core)
    # Default-variant floor: OLMo-3 needs >=4.57.0, and frameworks/2025.3.1
    # ships 4.57.6, so this is satisfied by system site-packages alone. The
    # --intern-s2 variant raises it below.
    "transformers": "4.57.0",
    "deepspeed": None,
    "webdataset": None,
    "uni2ts": None,              # time series (Moirai) — installed --no-deps
    "timm": None,                # vision tower fallback — system-provided
}

if intern_s2:
    # The --intern-s2 overlay's three coupled pins, from
    # requirements/aurora-py3.12.intern-s2.nodeps.txt. Validated together, not
    # independent minimums.
    required["transformers"] = "5.2.0"   # RopeParameters, imported by the
                                         # vendored Intern-S2 config
    required["huggingface_hub"] = "1.32.0"  # transformers 5.2.0 needs
                                            # is_offline_mode (hf-hub 1.x only)
    required["hf_xet"] = "1.6.0"         # hf-hub 1.32.0's Xet download path
                                         # needs a newer hf_xet than Aurora ships

failures = []
for pkg, min_ver in required.items():
    try:
        mod = importlib.import_module(pkg)
    except ImportError as e:
        failures.append(f"{pkg}: {e}")
        continue
    if min_ver is None:
        continue
    # Not every package exposes __version__ (e.g. hf_xet is a compiled
    # extension module with none) — fall back to importlib.metadata, which
    # reads the installed distribution's metadata instead.
    version_str = getattr(mod, "__version__", None)
    if version_str is None:
        try:
            version_str = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError as e:
            failures.append(f"{pkg}: could not determine installed version ({e})")
            continue
    if Version(version_str) < Version(min_ver):
        failures.append(f"{pkg} {version_str} < required {min_ver}")

# Assert torch came from system site-packages, not the venv. If --system-
# site-packages path ordering ever broke, we'd silently install a PyPI torch
# wheel that lacks XPU support — verify here rather than at job start.
import torch
import os
venv_root = os.environ.get("VIRTUAL_ENV", "")
if venv_root and torch.__file__.startswith(venv_root):
    failures.append(
        f"torch loaded from venv ({torch.__file__}) instead of system "
        "site-packages — would shadow XPU-patched build"
    )

# Version metadata is not the same as a working import: these three land via
# --no-deps, so a half-applied overlay still reports the right versions. Check
# the symbol the vendored Intern-S2 config actually imports.
if intern_s2:
    try:
        from transformers.modeling_rope_utils import RopeParameters  # noqa: F401
    except ImportError as e:
        failures.append(
            f"transformers.modeling_rope_utils.RopeParameters is not importable ({e}) "
            "— the Intern-S2 overlay did not take effect"
        )

if failures:
    print("VERIFY FAILED:", *failures, sep="\n  ", file=sys.stderr)
    sys.exit(1)
print("Modality dependency verification: OK")
print(f"  torch:        {torch.__version__} ({torch.__file__})")
_tf = importlib.metadata.version("transformers")
if intern_s2:
    print(f"  transformers: {_tf} (Intern-S2 overlay; vLLM out of declared range)")
else:
    print(f"  transformers: {_tf} (default; intern_s2 encoders unavailable)")
PYEOF

# pip check catches dep-graph inconsistencies in packages we installed.
# We scope the halt to the *lockfile* only — the nodeps file is deliberately
# excluded because those packages (lightning, pytorch-lightning, uni2ts,
# gluonts) declare torch / pandas pins that are known incompatible with
# Aurora's torch 2.10 + system pandas 3.0; that's why they're installed with
# --no-deps in the first place. Pre-existing system inconsistencies (yq
# missing argcomplete, esm pinning transformers<4.48, torchvision wanting
# torch==2.10.0 vs the patched torch==2.10.0a0+git*) are also ignored.
# Halt only if a package from the lockfile has an unsatisfied dep.
#
# Name normalization: lowercase + underscore->hyphen so PEP-503 normalized
# names from pip check (e.g. "the-well") match lockfile entries written
# either way (e.g. "the_well==1.2.0").
echo "  Running pip check..."
# `python -m pip`, not bare `pip`: the frameworks module's `pip` script has a
# hardcoded shebang pointing at the system python interpreter, which bypasses
# --system-site-packages venv resolution entirely and reports on the SYSTEM
# site-packages instead of this venv's (e.g. under --intern-s2 it would see
# system transformers 4.57.6, not this venv's 5.2.0 overlay). `python -m pip`
# runs inside the activated venv's interpreter and sees its actual package set.
PIP_CHECK_OUT=$(python -m pip check 2>&1 || true)
OUR_PKGS=$(awk -F'==' '/^[a-zA-Z0-9_-]+==/{print $1}' "$LOCKFILE" | tr '[:upper:]_' '[:lower:]-' | sort -u)
PIP_CHECK_BAD=""
while IFS= read -r line; do
    [ -z "$line" ] && continue
    [ "$line" = "No broken requirements found." ] && continue
    # Extract the offending package (first whitespace-separated token).
    pkg_lower=$(echo "$line" | awk '{print $1}' | tr '[:upper:]_' '[:lower:]-')
    if echo "$OUR_PKGS" | grep -qx "$pkg_lower"; then
        PIP_CHECK_BAD+="$line"$'\n'
    fi
done <<< "$PIP_CHECK_OUT"

if [ -n "$PIP_CHECK_BAD" ]; then
    echo "ERROR: pip check found inconsistencies in PRISM-installed packages:"
    echo "$PIP_CHECK_BAD"
    exit 1
fi

# vLLM 0.15.0+xpu (bundled in the frameworks module) declares
# transformers<5,>=4.56.0. The --intern-s2 overlay installs transformers 5.2.0
# over it, which puts vLLM outside its own declared range. Since vllm ships in
# system site-packages (not our lockfile), `pip check` reports this but it is
# excluded from the hard-fail gate above by the lockfile-only scope. Surface it
# as a non-fatal warning rather than letting it pass silently — the user asked
# for this trade by passing the flag, so the build continues.
#
# NOTE: `import vllm` / `vllm.LLM(...)` construction have been confirmed to
# still succeed under transformers 5.2.0 in ad hoc testing — but that is NOT
# the same as verified serving. Model loading, worker startup, and actual
# generation under transformers 5.2.0 have not been tested end-to-end.
# Treat vLLM serving in this venv as unverified until someone runs a real
# serving smoke test.
#
# The grep runs in both variants deliberately. In the default build it should
# find nothing — transformers stays at the system 4.57.6, inside vLLM's range —
# and a hit there means something else pulled transformers 5.x in, which is
# worth seeing rather than suppressing.
VLLM_CONFLICT=$(echo "$PIP_CHECK_OUT" | grep -i "vllm" | grep -i "transformers" || true)
if [ -n "$VLLM_CONFLICT" ]; then
    echo "WARNING: pip check reports a vllm/transformers version conflict:"
    echo "$VLLM_CONFLICT"
    if [ "$INTERN_S2" = "1" ]; then
        echo "  Expected for --intern-s2: transformers 5.2.0 is outside vLLM's"
        echo "  declared transformers<5,>=4.56.0. 'import vllm; vllm.LLM(...)'"
        echo "  construction works under 5.2.0 in ad hoc testing, but real"
        echo "  serving (model load, worker startup, generation) is UNVERIFIED."
        echo "  Do not assume vLLM serving works in this venv without testing"
        echo "  it directly. Build a default-variant venv for vLLM work."
    else
        echo "  UNEXPECTED in a default build: transformers should have stayed"
        echo "  at the system version, inside vLLM's declared range. Something"
        echo "  installed transformers 5.x — check the lockfile and nodeps"
        echo "  manifests before using this venv."
    fi
fi

echo "[7/7] Writing PRISM_BUILD_INFO manifest..."
(cd "$PRISM_DIR" && git diff-index --quiet HEAD 2>/dev/null) && DIRTY=no || DIRTY=yes
LOCKFILE_SHA=$(sha256sum "$LOCKFILE" | awk '{print $1}')
# Lmod populates LMOD_FAMILY_FRAMEWORKS_VERSION when frameworks/X.Y.Z is loaded
# (LOADED_MODULEFILES is empty in non-interactive bash, so don't rely on it).
LOADED_FRAMEWORKS="frameworks/${LMOD_FAMILY_FRAMEWORKS_VERSION:-UNKNOWN}"
SYSTEM_TRANSFORMERS=$(python -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null || echo MISSING)
UV_VERSION=$(uv --version 2>&1 | head -1)

{
    echo "build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_sha: $(cd "$PRISM_DIR" && git rev-parse HEAD 2>/dev/null)"
    echo "git_branch: $(cd "$PRISM_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "git_dirty: $DIRTY"
    echo "host: $(hostname)"
    echo "python: $(python -V 2>&1)"
    echo "python_realpath: $SYSTEM_PYTHON_REAL"
    echo "frameworks_module: $LOADED_FRAMEWORKS"
    echo "lockfile_sha256: $LOCKFILE_SHA"
    echo "system_transformers_version: $SYSTEM_TRANSFORMERS"
    echo "variant: $([ "$INTERN_S2" = "1" ] && echo intern-s2 || echo default)"
    echo "intern_s2_encoders: $([ "$INTERN_S2" = "1" ] && echo available || echo disabled)"
    echo "uv_version: $UV_VERSION"
    for pkg in torch torch_geometric transformers huggingface_hub hf_xet deepspeed the_well hydra-core walrus; do
        v=$(python -c "import importlib.metadata as m; print(m.version('$pkg'))" 2>/dev/null)
        if [ "$pkg" = "walrus" ]; then
            echo "$pkg: ${v:-MISSING} (editable, src/libs/walrus)"
        else
            echo "$pkg: ${v:-MISSING}"
        fi
    done
} > "$VENV_PATH/PRISM_BUILD_INFO"
python -m pip freeze > "$VENV_PATH/PRISM_BUILD_PIP_FREEZE.txt"

echo "--- PRISM_BUILD_INFO ---"
cat "$VENV_PATH/PRISM_BUILD_INFO"

echo "=== Build complete ==="
echo "  Variant: $VARIANT"
if [ "$INTERN_S2" != "1" ]; then
    echo "  ts_projector=intern_s2 / intern_s2_397b will NOT load in this venv."
    echo "  Rebuild on a separate VENV_PATH with --intern-s2 if you need them."
fi
echo "To use this venv:"
echo "  module load $FRAMEWORKS_MODULE"
echo "  source $VENV_PATH/bin/activate"

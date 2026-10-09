#!/bin/bash
# Build the PRISM venv on Polaris (NVIDIA A100, CUDA).
#
# Layout: ALCF `conda/2025-09-28` module supplies torch / cuda / nccl /
# deepspeed / accelerate / transformers. We create a venv on top of it with
# --system-site-packages and add the PRISM-only deps from
# requirements/polaris.txt.
#
# Run from the repo root on a Polaris login or compute node:
#     bash tools/setup_polaris_env.sh                       # build .venv-polaris in cwd
#     PRISM_VENV=/eagle/.../envs/polaris-venv bash tools/setup_polaris_env.sh
#
# Idempotent: re-running upgrades packages in place rather than re-building.

set -euo pipefail

PRISM_VENV="${PRISM_VENV:-$(pwd)/.venv-polaris}"
CONDA_MODULE="${POLARIS_CONDA_MODULE:-conda/2025-09-28}"

echo "[setup] Loading $CONDA_MODULE..."
module use /soft/modulefiles
module load "$CONDA_MODULE"
conda activate base

# Verify the base conda env has the load-bearing imports BEFORE we layer on
# top of it. If torch/cuda are broken, bailing here saves ~10 min of pip work.
python - <<'PY'
import sys
import torch
print(f"[setup] python      {sys.version.split()[0]}")
print(f"[setup] torch       {torch.__version__} (cuda {torch.version.cuda})")
print(f"[setup] torch.cuda  available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
import deepspeed
print(f"[setup] deepspeed   {deepspeed.__version__}")
import transformers
print(f"[setup] transformers {transformers.__version__}")
import accelerate
print(f"[setup] accelerate  {accelerate.__version__}")
PY

# Always use stdlib venv + pip on Polaris. uv would speed up installs but
# its resolver treats a `torch>=X` constraint from a transitive dep as a
# trigger to install a fresh PyPI torch wheel into the venv, shadowing the
# ALCF-built NCCL torch from the conda base. plain pip honors the venv's
# `include-system-site-packages=true` and skips packages already
# importable from /soft, which is what we want.

if [ ! -d "$PRISM_VENV" ]; then
    echo "[setup] Creating venv at $PRISM_VENV (inheriting base site-packages)..."
    python -m venv "$PRISM_VENV" --system-site-packages
else
    echo "[setup] Reusing existing venv at $PRISM_VENV"
fi

# shellcheck disable=SC1091
source "$PRISM_VENV/bin/activate"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_TORCH=$(python -c 'import torch; print(torch.__version__)')
BASE_TORCH_PATH=$(python -c 'import torch, os; print(os.path.realpath(os.path.dirname(torch.__file__)))')
echo "[setup] Base conda torch:   $BASE_TORCH"
echo "[setup] Base torch lives at: $BASE_TORCH_PATH"
echo "[setup] Installing requirements from $REPO_ROOT/requirements/polaris.txt..."
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements/polaris.txt"
# Install the_well + walrus with --no-deps. Both pin `torch>=2.1` and a
# default pip install would resolve a fresh torch wheel into the venv,
# shadowing the ALCF-built NCCL torch from the conda base. the_well +
# walrus themselves are pure Python and run fine against the base torch.
# (Other their-own deps: einops, h5py, numpy, fsspec, pyyaml — all
# already in the conda base.)
#
# Pin the_well to Aurora's version (requirements/aurora-py3.12.lock.txt) so
# upstream API churn doesn't silently land on Polaris first.
pip install --no-deps "the_well==1.2.0"
pip install --no-deps -e "$REPO_ROOT/src/libs/walrus"

# Bump transformers to 4.57.6 + matching tokenizers. The conda base ships
# transformers 4.53.3, which lacks `dtype=` kwarg on AutoModelForCausalLM
# (PRISM's src/model.py passes dtype= unconditionally) and lacks OLMo-3
# support. transformers 5.x needs huggingface_hub > what base ships so we
# pin to 4.57.6 (same as Aurora's deepspeed env).
#
# --target=$VENV_SITE forces install INTO the venv (not ~/.local where
# plain `pip install --user` would land); base site-packages takes
# precedence over ~/.local but is shadowed by the venv's own site-packages
# dir. Resolve purelib dynamically so a conda module update that bumps the
# Python minor version doesn't silently install to a nonexistent directory
# (and let the shadow stop working without complaint).
VENV_SITE=$("$PRISM_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [ ! -d "$VENV_SITE" ]; then
    echo "[setup] FAIL: could not resolve venv site-packages (got '$VENV_SITE')." >&2
    exit 1
fi
pip install --no-deps --target="$VENV_SITE" "transformers==4.57.6"
# tokenizers cap from transformers/dependency_versions_check.py: ">=0.22,<=0.23"
# 0.23.1 fails the upper bound, 0.21.x (conda base) fails the lower.
rm -rf "$VENV_SITE"/tokenizers "$VENV_SITE"/tokenizers-*.dist-info
pip install --no-deps --target="$VENV_SITE" "tokenizers>=0.22,<=0.23"

# Sanity check: refuse to ship a venv whose torch shadows the conda base
# torch. The ALCF-built torch carries the NCCL/Slingshot patches; a generic
# pip wheel breaks multi-node collectives.
VENV_TORCH=$("$PRISM_VENV/bin/python" -c 'import torch; print(torch.__version__)')
VENV_TORCH_PATH=$("$PRISM_VENV/bin/python" -c 'import torch, os; print(os.path.realpath(torch.__file__))')
if [[ "$VENV_TORCH_PATH" == "$PRISM_VENV"/* ]]; then
    echo "[setup] FAIL: venv has its own torch at $VENV_TORCH_PATH"
    echo "[setup]       version $VENV_TORCH != base $BASE_TORCH"
    echo "[setup]       A transitive dep pulled torch into the venv and"
    echo "[setup]       will shadow the ALCF-built NCCL torch. Remove the"
    echo "[setup]       offending package (likely the_well / torch_harmonics)"
    echo "[setup]       and re-run."
    exit 1
fi
echo "[setup] Venv torch points to base: $VENV_TORCH_PATH (version $VENV_TORCH)"

# Sanity check the new venv: every modality import must succeed; otherwise
# downstream training will OOM-style crash mid-step instead of failing here.
echo "[setup] Verifying modality imports..."
python - <<'PY'
import importlib

mods = [
    "torch",
    "deepspeed",
    "transformers",
    "accelerate",
    "hydra",
    "omegaconf",
    "webdataset",
    "torch_geometric",
    "the_well",
    "walrus",
    "safetensors",
    "wandb",
    "mpi4py",
]
ok, bad = [], []
for name in mods:
    try:
        importlib.import_module(name)
        ok.append(name)
    except Exception as exc:
        bad.append(f"{name}: {exc.__class__.__name__}: {exc}")

print("[setup] OK :", ", ".join(ok))
if bad:
    print("[setup] FAIL:")
    for line in bad:
        print(f"  {line}")
    raise SystemExit(1)
PY

# Record build provenance so smokes can be linked back to an exact env.
BUILD_INFO="$PRISM_VENV/PRISM_BUILD_INFO"
{
    echo "build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host:       $(hostname)"
    echo "conda:      $CONDA_MODULE"
    echo "repo:       $REPO_ROOT"
    echo "git_sha:    $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "python:     $(python --version 2>&1)"
    echo "torch:      $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
} > "$BUILD_INFO"

pip freeze > "$PRISM_VENV/PRISM_BUILD_PIP_FREEZE.txt" 2>/dev/null || true

echo "[setup] Done. Activate with:"
echo "    module use /soft/modulefiles && module load $CONDA_MODULE && conda activate base"
echo "    source $PRISM_VENV/bin/activate"
echo "[setup] Build info at $BUILD_INFO"

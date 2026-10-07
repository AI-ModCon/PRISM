#!/bin/bash -l
#PBS -N pr169-shard-stride
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A ModCon
#PBS -k doe
#PBS -j oe

# Verify shard->rank assignment at 12 ranks under real MPI, in both regimes:
#   27 train shards / 12 ranks -> shard-level split
#    3 val   shards / 12 ranks -> sample-level stride (the PR #169 case)
#
# Walltime is sized from measured cost: the login-node equivalent ran in
# well under a minute; 20 min covers module load + two mpiexec launches.

set -euo pipefail

PRISM_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PRISM_DIR"

DATA_ROOT="${VERIFY_DATA_ROOT:-/flare/ModCon/pemami/data/SciTS-processed}"

# Lmod's init script dereferences ZSH_EVAL_CONTEXT unguarded, so `module load`
# under `set -u` aborts with "ZSH_EVAL_CONTEXT: unbound variable". Drop -u for
# the module load, then restore it.
set +u
module load frameworks
set -u
# NOTE: deliberately NOT activating .venv-deepspeed. This check only needs
# webdataset + torch.distributed(gloo), both in the frameworks module, and
# the packed venv still pins transformers 4.57.6.

echo "=== PR #169 shard-stride verification ==="
echo "host      : $(hostname)"
echo "prism dir : $PRISM_DIR"
echo "data root : $DATA_ROOT"
python -c "import webdataset, torch; print('webdataset', webdataset.__version__, '| torch', torch.__version__)"

export NUM_NODES=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29591
export VERIFY_LIMIT=150

run_case () {
    local label="$1" dir="$2" nranks="$3"
    echo ""
    echo "--- case: $label ($nranks ranks) ---"
    # Each mpiexec gets its own port so back-to-back gloo groups do not collide.
    # `bash -c`, not `bash -lc`: a login shell re-sources the profile and can
    # undo the frameworks module env we just set up. Ranks inherit it instead.
    # PALS_SIZE is never set by mpiexec on Aurora, so supply it explicitly
    # rather than letting the script guess world size.
    MASTER_PORT=$((29591 + nranks)) \
    VERIFY_SHARDS_DIR="$dir" \
    PALS_SIZE="$nranks" \
    mpiexec -n "$nranks" -ppn "$nranks" --cpu-bind depth --depth 8 \
        bash -c "cd '$PRISM_DIR' && exec python tools/shard_stride_mpi_check.py"
}

rc=0
run_case "27 train shards / 12 ranks (shard-split)" "$DATA_ROOT/shards"     12 || rc=1
run_case "3 val shards / 12 ranks (sample-stride)"  "$DATA_ROOT/val_shards" 12 || rc=1

echo ""
if [ "$rc" -eq 0 ]; then
    echo "OVERALL: PASS"
else
    echo "OVERALL: FAIL"
fi
exit "$rc"

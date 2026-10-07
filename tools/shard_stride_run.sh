#!/bin/bash
# Run the shard->rank verification on a held compute node, over SSH from a UAN.
#
#   bash tools/shard_stride_run.sh <node> [nranks]
#
# Re-runnable: edit tools/shard_stride_mpi_check.py and invoke again against
# the same held node. Each invocation is independent, so a crash costs one
# re-run rather than one queue cycle.
#
# No `set -u`: the remote side loads the frameworks module, and Lmod's init
# dereferences ZSH_EVAL_CONTEXT unguarded (this killed job 8834494).
set -eo pipefail

NODE="${1:-}"
NRANKS="${2:-12}"
if [ -z "$NODE" ]; then
    echo "usage: $0 <node> [nranks]" >&2
    echo "  derive <node> from: qstat -f <jobid> | grep exec_host" >&2
    exit 2
fi

PRISM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${VERIFY_DATA_ROOT:-/flare/ModCon/pemami/data/SciTS-processed}"

echo "=== PR169 shard-stride check ==="
echo "node   : $NODE"
echo "ranks  : $NRANKS"
echo "prism  : $PRISM_DIR"
echo "data   : $DATA_ROOT"
echo ""

# Heredoc is quoted ('REMOTE') so the local shell does not expand anything;
# the values we need are passed through the environment instead.
ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE" \
    PRISM_DIR="$PRISM_DIR" DATA_ROOT="$DATA_ROOT" NRANKS="$NRANKS" \
    bash -s <<'REMOTE'
set -eo pipefail
cd "$PRISM_DIR"

# Lmod aborts under set -u; we never enable it here.
module load frameworks
# Not activating .venv-deepspeed: this check needs only webdataset + gloo,
# and the packed venv still pins transformers 4.57.6.

echo "host       : $(hostname)"
python -c "import webdataset, torch; print('webdataset', webdataset.__version__, '| torch', torch.__version__)"

export NUM_NODES=1
export MASTER_ADDR=127.0.0.1
export VERIFY_LIMIT=150

run_case () {
    label="$1"; dir="$2"; n="$3"
    echo ""
    echo "--- case: $label ($n ranks) ---"
    if [ ! -d "$dir" ]; then
        echo "MISSING shard dir: $dir"
        return 1
    fi
    # Distinct port per case so back-to-back gloo groups do not collide.
    # bash -c (not -lc): a login shell re-sources the profile and would undo
    # the module env. PALS_SIZE is never set by mpiexec on Aurora.
    # `< /dev/null` is load-bearing: this whole script arrives over `bash -s`
    # (i.e. on stdin), and mpiexec reads stdin too. Without the redirect the
    # first mpiexec swallows the remainder of the script and every later case
    # is silently skipped -- the run still exits 0, so it reads as a pass.
    MASTER_PORT=$((29600 + n)) \
    VERIFY_SHARDS_DIR="$dir" \
    PALS_SIZE="$n" \
    mpiexec -n "$n" -ppn "$n" --cpu-bind depth --depth 8 \
        bash -c "cd '$PRISM_DIR' && exec python tools/shard_stride_mpi_check.py" \
        < /dev/null
}

rc=0
run_case "train shards (expect shard-split)" "$DATA_ROOT/shards"     "$NRANKS" || rc=1
run_case "val shards (expect sample-stride)" "$DATA_ROOT/val_shards" "$NRANKS" || rc=1

echo ""
if [ "$rc" -eq 0 ]; then echo "OVERALL: PASS"; else echo "OVERALL: FAIL"; fi
exit "$rc"
REMOTE

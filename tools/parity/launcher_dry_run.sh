#!/usr/bin/env bash
# Launcher dry-run diff harness.
#
# Usage:
#   tools/parity/launcher_dry_run.sh <baseline-ref> <candidate-ref> [-- <extra launcher args>]
#
# Runs `python tools/launch_aurora_daos.py --dry-run --id PARITY ...` against
# both refs, captures the generated mpiexec script, and diffs them. A clean
# diff (or only expected env-var additions) is the pass criterion for every
# launcher-touching PR (1, 2, 5, 6, 8).
#
# Safe to run on a UAN — `--dry-run` does not submit or execute anything.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <baseline-ref> <candidate-ref> [-- <extra launcher args>]" >&2
    exit 2
fi

baseline="$1"
candidate="$2"
shift 2

extra_args=()
if [[ "${1:-}" == "--" ]]; then
    shift
    extra_args=("$@")
fi

# Defaults match the smallest reproducible launch shape.
default_args=(
    --id PARITY
    --dry-run
    --nodes 1
    --design PRISM-IMAGE-ONLY-2N
    --no-pil4dfs
)

repo_root="$(git rev-parse --show-toplevel)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

run_one() {
    local ref="$1"
    local out="$2"
    local worktree="$work/$ref"
    git -C "$repo_root" worktree add --detach "$worktree" "$ref" >/dev/null
    (
        cd "$worktree"
        python tools/launch_aurora_daos.py "${default_args[@]}" "${extra_args[@]}" \
            > "$out" 2>&1 || true
    )
    # --force is safe here: --dry-run cannot modify tracked files, so the
    # worktree is always clean and no work is lost.
    git -C "$repo_root" worktree remove --force "$worktree" >/dev/null
}

run_one "$baseline" "$work/baseline.txt"
run_one "$candidate" "$work/candidate.txt"

echo "=== diff: $baseline -> $candidate ==="
diff -u "$work/baseline.txt" "$work/candidate.txt" || true

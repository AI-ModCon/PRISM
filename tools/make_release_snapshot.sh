#!/bin/bash
# Build the AI-ModCon/PRISM release snapshot from this repository.
#
# PRISM is a published snapshot of BaseMM_PRISM, decided in #232: content is
# the current tree, with no development history. Changes flow one way --
# BaseMM_PRISM -> PRISM, by force-push. Anything edited only in PRISM is lost
# the next time a snapshot is taken.
#
# Two such edits exist, and this script reapplies both. They are deliberately
# NOT upstreamed: BaseMM_PRISM's badges are correct for BaseMM_PRISM, and
# Dependabot should stay active here.
#
#   1. README badges. The CI and Codecov badges point at AI-ModCon/BaseMM_PRISM,
#      which stays private. On a public PRISM the Codecov badge advertises
#      coverage for a repository nobody can see and the CI badge 404s for any
#      external reader. Replaced with static license/python badges that resolve
#      inside the published repo.
#
#   2. Dependabot. Disabled in the snapshot via `open-pull-requests-limit: 0`.
#      A Dependabot PR opened against PRISM cannot be merged usefully -- the
#      next snapshot overwrites it -- and one (PRISM#1) was already opened and
#      auto-closed that way. The config file is kept rather than deleted,
#      because tests/platform/test_dependabot_scope.py reads it and that suite
#      runs in PRISM's CI; deleting the file turns 8 tests into errors.
#
# Usage:
#   bash tools/make_release_snapshot.sh                     # build only
#   bash tools/make_release_snapshot.sh --push              # build and push
#   bash tools/make_release_snapshot.sh --remove-protection # unlock PRISM main
#   bash tools/make_release_snapshot.sh --restore-protection
#
# Without --push it builds the snapshot and stops, so the result can be
# inspected. The two protection flags act on PRISM only and do not build
# anything; run them either side of a push:
#
#   bash tools/make_release_snapshot.sh --remove-protection
#   bash tools/make_release_snapshot.sh --push
#   bash tools/make_release_snapshot.sh --restore-protection
#
# PRISM's main requires status checks and blocks force-pushes, so a snapshot
# cannot be pushed while protection is on. --remove-protection saves the live
# configuration to $PRISM_PROTECTION_FILE (default
# ~/backups/prism/PRISM-branch-protection.json) before deleting it, and
# --restore-protection puts that exact configuration back. Saving on the way
# out is what makes the restore faithful: it cannot drift from whatever
# BaseMM_PRISM happens to have today, and it does not depend on a file
# somebody remembered to create by hand.
#
# The GET and PUT schemas for branch protection differ, so a saved GET
# response is NOT a valid PUT body -- GitHub rejects it with
# `No subschema in "anyOf" matched`. --remove-protection writes the PUT form.
#
# Between the two calls PRISM's main is unprotected. Keep the window short,
# and if a push fails, still run --restore-protection.

# No `set -u`: Lmod's init reads unset shell variables and dies under nounset.
set -eo pipefail

REMOTE="${PRISM_REMOTE:-git@github.com:AI-ModCon/PRISM.git}"
WALRUS_PATH="src/libs/walrus"
# owner/repo for the gh API calls, derived from the remote so the two cannot
# disagree.
SLUG="$(printf '%s' "$REMOTE" | sed -E 's#\.git$##; s#^.*[:/]([^/]+/[^/]+)$#\1#')"
PROT_FILE="${PRISM_PROTECTION_FILE:-$HOME/backups/prism/PRISM-branch-protection.json}"

PUSH=0
ACTION=""
case "${1:-}" in
    --push)                PUSH=1 ;;
    --remove-protection)   ACTION=remove ;;
    --restore-protection)  ACTION=restore ;;
    "")                    ;;
    *) echo "unknown argument: $1" >&2
       sed -n '/^# Usage:/,/^# and if a push fails/p' "$0" >&2
       exit 2 ;;
esac

# The system python3 on an Aurora login node is 3.6 -- no tomllib, and it
# cannot parse `from __future__ import annotations` in this repo's tooling.
# Find a 3.11+ interpreter rather than assuming `python3` is one.
find_python() {
    local c
    for c in "${PRISM_PYTHON:-}" python3.13 python3.12 python3.11 python3 \
             /opt/aurora/default/frameworks/aurora_frameworks-*/bin/python3; do
        [ -n "$c" ] || continue
        command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
        if "$c" -c 'import sys,tomllib; sys.exit(0 if sys.version_info>=(3,11) else 1)' \
             >/dev/null 2>&1; then
            echo "$c"; return 0
        fi
    done
    return 1
}
PY="$(find_python)" || {
    echo "ERROR: no Python 3.11+ with tomllib found. Set PRISM_PYTHON, or" >&2
    echo "       module load frameworks first." >&2
    exit 1
}

# ---- branch protection -----------------------------------------------------
# These act on PRISM and build nothing. They exit when done.

require_gh() {
    command -v gh >/dev/null 2>&1 || {
        echo "ERROR: gh is not on PATH; the protection flags need it." >&2; exit 1; }
    # A stale GITHUB_TOKEN in the environment shadows gh's own credentials on
    # this cluster, so every call here clears it.
}

if [ "$ACTION" = "remove" ]; then
    require_gh
    echo "Saving current protection for $SLUG main -> $PROT_FILE"
    mkdir -p "$(dirname "$PROT_FILE")"
    if ! env -u GITHUB_TOKEN gh api "repos/$SLUG/branches/main/protection" \
            > "$PROT_FILE.get" 2>/dev/null; then
        echo "  branch is already unprotected -- nothing to save, nothing to do."
        rm -f "$PROT_FILE.get"
        exit 0
    fi
    # The GET and PUT schemas differ. Convert, rather than saving the GET and
    # discovering at restore time that GitHub rejects it with
    # `No subschema in "anyOf" matched`.
    "$PY" - "$PROT_FILE.get" "$PROT_FILE" <<'PYX'
import json, sys
g = json.load(open(sys.argv[1]))
rpr = g.get("required_pull_request_reviews")
put = {
    "required_status_checks": {
        "strict":   g["required_status_checks"]["strict"],
        "contexts": g["required_status_checks"]["contexts"],
    } if g.get("required_status_checks") else None,
    "enforce_admins": g["enforce_admins"]["enabled"],
    "required_pull_request_reviews": {
        "dismiss_stale_reviews":           rpr["dismiss_stale_reviews"],
        "require_code_owner_reviews":      rpr["require_code_owner_reviews"],
        "required_approving_review_count": rpr["required_approving_review_count"],
    } if rpr else None,
    # Restrictions are a user/team/app allowlist; this repo has none. Carrying
    # one over would need re-expanding logins, so refuse rather than silently
    # drop it.
    "restrictions": None,
    "required_linear_history":          g["required_linear_history"]["enabled"],
    "allow_force_pushes":               g["allow_force_pushes"]["enabled"],
    "allow_deletions":                  g["allow_deletions"]["enabled"],
    "required_conversation_resolution": g["required_conversation_resolution"]["enabled"],
}
if g.get("restrictions"):
    sys.exit("ERROR: branch has push restrictions; this script would drop them. "
             "Save and restore protection by hand.")
json.dump(put, open(sys.argv[2], "w"), indent=2)
n = len(put["required_status_checks"]["contexts"]) if put["required_status_checks"] else 0
print(f"  saved: {n} required contexts, enforce_admins={put['enforce_admins']}")
PYX
    rm -f "$PROT_FILE.get"
    env -u GITHUB_TOKEN gh api -X DELETE "repos/$SLUG/branches/main/protection" --silent
    echo "Protection REMOVED from $SLUG main. It is unprotected until you run:"
    echo "  bash tools/make_release_snapshot.sh --restore-protection"
    exit 0
fi

if [ "$ACTION" = "restore" ]; then
    require_gh
    [ -f "$PROT_FILE" ] || {
        echo "ERROR: no saved protection at $PROT_FILE" >&2
        echo "       Run --remove-protection first, or point PRISM_PROTECTION_FILE" >&2
        echo "       at a saved PUT payload." >&2
        exit 1; }
    env -u GITHUB_TOKEN gh api -X PUT "repos/$SLUG/branches/main/protection" \
        --input "$PROT_FILE" --silent
    # Read it back rather than trusting the exit code.
    env -u GITHUB_TOKEN gh api "repos/$SLUG/branches/main/protection" --jq \
        '"Protection RESTORED: \(.required_status_checks.contexts|length) required contexts, enforce_admins=\(.enforce_admins.enabled), force_pushes=\(.allow_force_pushes.enabled)"'
    exit 0
fi

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

SRC_SHA="$(git rev-parse --short HEAD)"
if ! git diff --quiet HEAD -- || ! git diff --cached --quiet; then
    echo "ERROR: working tree is dirty. The snapshot is built from HEAD, so" >&2
    echo "       uncommitted work would be silently omitted. Commit or stash." >&2
    exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
OUT="$WORK/snapshot"
mkdir -p "$OUT"

echo "[1/6] Exporting tree at $SRC_SHA"
# git archive, not cp: it emits exactly the tracked tree. A plain copy would
# pick up untracked local state, and `git add -A` in the new repo would then
# re-apply .gitignore -- which silently drops ~100 files that are tracked here
# despite matching ignore patterns (tools/_*.sh, scaling-study/, jobs/).
git archive --format=tar HEAD | tar -x -C "$OUT"

echo "[2/6] Reapplying the README badge replacement"
"$PY" - "$OUT" <<'PY'
import pathlib, sys, re
readme = pathlib.Path(sys.argv[1]) / "README.md"
t = readme.read_text()
pat = re.compile(
    r"\[!\[CI\]\(https://github\.com/AI-ModCon/BaseMM_PRISM[^\n]*\n"
    r"\[!\[codecov\]\(https://codecov\.io/gh/AI-ModCon/BaseMM_PRISM[^\n]*\n"
)
new = ("[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)\n"
       "[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)\n")
t2, n = pat.subn(new, t, count=1)
if n != 1:
    # Loud, not silent: if the badge block was reworded upstream this script
    # must be updated, or the snapshot ships badges pointing at a private repo.
    raise SystemExit("ERROR: BaseMM badge block not found in README.md -- "
                     "it was probably reworded. Update this script.")
readme.write_text(t2)
print("      badges replaced")
PY

echo "[3/6] Disabling Dependabot in the snapshot"
"$PY" - "$OUT" <<'PY'
import pathlib, sys
cfg = pathlib.Path(sys.argv[1]) / ".github" / "dependabot.yml"
if not cfg.exists():
    raise SystemExit("ERROR: .github/dependabot.yml missing -- update this script.")
t = cfg.read_text()
n = t.count("open-pull-requests-limit:")
if n == 0:
    raise SystemExit("ERROR: no open-pull-requests-limit found -- update this script.")
import re
t = re.sub(r"open-pull-requests-limit: *\d+", "open-pull-requests-limit: 0", t)
banner = """# ---------------------------------------------------------------------------
# SNAPSHOT REPOSITORY: Dependabot is disabled here via
# `open-pull-requests-limit: 0`, applied by tools/make_release_snapshot.sh.
# Edit that script, not this file -- this copy is regenerated each snapshot.
#
# A Dependabot PR opened against the snapshot cannot be merged usefully:
# changes flow BaseMM_PRISM -> PRISM by force-push, so the next snapshot
# overwrites it. The file is kept rather than deleted because
# tests/platform/test_dependabot_scope.py reads it and runs in PRISM's CI.
# ---------------------------------------------------------------------------
version: 2"""
if "version: 2" not in t:
    raise SystemExit("ERROR: `version: 2` not found in dependabot.yml.")
t = t.replace("version: 2", banner, 1)
cfg.write_text(t)
print(f"      {n} PR limit(s) set to 0, banner added")
PY

echo "[4/6] Building the root commit"
# .mailmap is tracked in this repository, so git archive already exported it.
# It used to live only in the snapshot, which is why an earlier version of
# this script copied it in; that copy is now redundant and the file must
# compare equal like any other tracked path.
cd "$OUT"
git init -q -b main
git config user.name  "$(git -C "$REPO" config user.name)"
git config user.email "$(git -C "$REPO" config user.email)"
# --force: see the note in step 1. Without it, tracked-but-ignored files vanish.
git add --force -A

# The submodule is a gitlink; git archive does not emit it, so restore the
# pointer explicitly or `git clone --recursive` silently populates nothing.
WALRUS_SHA="$(git -C "$REPO" ls-tree HEAD "$WALRUS_PATH" | awk '{print $3}')"
if [ -z "$WALRUS_SHA" ]; then
    echo "ERROR: no gitlink for $WALRUS_PATH in the source tree." >&2
    exit 1
fi
git update-index --add --cacheinfo "160000,$WALRUS_SHA,$WALRUS_PATH"

AUTHORS="$("$PY" - "$REPO/pyproject.toml" <<'PY'
import sys, tomllib
d = tomllib.load(open(sys.argv[1], "rb"))
for a in d["project"]["authors"]:
    print(f"Co-Authored-By: {a['name']} <{a['email']}>")
PY
)"

git commit -q -F - <<EOF
Initial Release Commit

Snapshot of the PRISM framework, taken from AI-ModCon/BaseMM_PRISM at $SRC_SHA.
No development history: the full history lives in BaseMM_PRISM.

Built by tools/make_release_snapshot.sh, which also replaces the CI/Codecov
badges (they point at the private BaseMM_PRISM) and disables Dependabot in
this repository.

$WALRUS_PATH is a submodule (MIT, Polymathic AI) and is present as a gitlink,
not vendored content. Clone with --recursive.

$AUTHORS
EOF

echo "[5/6] Verifying against the source tree"
git ls-tree -r HEAD | awk '{print $3, $4}' | sort > "$WORK/snap.txt"
git -C "$REPO" ls-tree -r HEAD | awk '{print $3, $4}' | sort > "$WORK/src.txt"
# README and dependabot.yml are expected to differ -- that is the point.
EXPECTED="README.md|.github/dependabot.yml"
if ! diff <(grep -vE " ($EXPECTED)$" "$WORK/src.txt") \
          <(grep -vE " ($EXPECTED)$" "$WORK/snap.txt") > "$WORK/diff.txt"; then
    echo "ERROR: snapshot differs from the source tree beyond the two expected files:" >&2
    head -20 "$WORK/diff.txt" >&2
    exit 1
fi
COMMITS="$(git rev-list --count HEAD)"
COAUTHORS="$(git log -1 --format=%B | grep -c '^Co-Authored-By:')"
echo "      1 root commit ($COMMITS total), $COAUTHORS co-authors, gitlink ${WALRUS_SHA:0:7}"
echo "      every other path byte-identical to $SRC_SHA"

echo "[6/6] Push"
if [ "$PUSH" -eq 1 ]; then
    git push --force "$REMOTE" main:main
    echo "      pushed to $REMOTE"
    echo
    echo "NOW RESTORE BRANCH PROTECTION -- see the header of this script."
else
    # The work dir is removed on exit, so offer a copy that outlives it.
    KEEP="${PRISM_SNAPSHOT_OUT:-/tmp/prism-release-snapshot}"
    rm -rf "$KEEP" && cp -a "$OUT" "$KEEP"
    echo "      not pushed (no --push). Snapshot kept at: $KEEP"
    echo "      inspect, then: git -C $KEEP push --force $REMOTE main:main"
fi

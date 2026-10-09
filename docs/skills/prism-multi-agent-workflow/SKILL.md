---
name: prism-multi-agent-workflow
description: >
  Work safely in the PRISM repo, which multiple coding agents edit in parallel.
  Use before committing, branching, pushing, deleting a branch/worktree, merging
  a stacked PR, or claiming "I'm on branch X." Triggers: "commit", "create a
  branch", "worktree", "delete branch", "clean up branches", "merge the PR",
  "stacked PR", "which branch am I on", "another agent edited this".
metadata:
  version: "1.0"
  project: prism
---

# PRISM — Multi-Agent Workflow

**Default assumption: another agent is editing this clone right now.** This repo
is multi-agent by design (parallel agents on different PRs), so the bare working
tree is a shared resource — another agent's `git checkout` / `git stash` / edit
can silently change your state between turns. For the generic patterns, see the
installed **`multi-agent-git-hygiene`** skill; this is the PRISM-specific drill.

## Mandatory: work in an isolated worktree

- Call `EnterWorktree` at the start of any session that edits files or git state
  → creates `.claude/worktrees/<name>` on a fresh branch off `origin/main`.
- Spawning editing subagents? Pass `isolation: "worktree"` to the `Agent` tool.
- The bare clone (`/lus/flare/projects/ModCon/ngetty/BaseMM_PRISM`) is
  **read-only** — never commit or branch-switch there.

## Mandatory: verify branch before you claim it

Before saying "I created branch X" / "I'm on branch Y" / "I'll commit to Z":

```bash
git rev-parse --abbrev-ref HEAD          # the branch in your context may be stale
```

If you're on a branch you didn't create, **STOP** — `git reflog | head -10`;
another agent almost certainly moved you. Do not commit. Back up your edits to
`/tmp/` first, then sort out where they belong.

A `git status` that shows your earlier edits as "no modifications" is a **drift
signal**, not a no-op. Grep the file for a distinctive token from your edit
before trusting it.

## GitHub PRs are the source of truth for in-flight work

- Before a non-trivial task: `gh pr list --state open` to see what's already in
  flight. Don't pick up something another agent has open.
- **One agent = one branch = one PR.** Don't pile unrelated changes onto a branch
  another agent owns.

## Two destructive traps

1. **Check open PRs before pruning anything.** Always `gh pr list --state open`
   first; never delete a branch/worktree backing an open PR even if the local
   name looks stale. (PR #87's worktree was nuked this way — 2026-05-25.)
2. **Stacked PRs need base-retarget before squash-merge.** Squash-merging a
   stacked PR whose base is a *deleted* branch silently orphans the content.
   Retarget the child PR's base to `main` first. (PRs #80/#81 lost — 2026-05-25.)

## Launching jobs from a worktree

PBS scripts generated from a worktree need a `${PBS_O_WORKDIR:-...}` fallback and
a correct `LAUNCHER_PRISM_DIR` for the venv tarball — a hardcoded bare-clone path
fails in ~1s when a plan YAML doesn't exist there. See prism-launching-jobs.

## See also

- CLAUDE.md "Concurrency" section.
- [prism-launching-jobs](../prism-launching-jobs/SKILL.md).
- Generic installed skills: `multi-agent-git-hygiene`, `shell-quoting-traps`.

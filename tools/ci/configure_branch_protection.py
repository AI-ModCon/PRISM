#!/usr/bin/env python3
"""Configure GitHub branch protection required checks for this repository.

Required-check contexts are the **bare job names** from the workflow files.
GitHub only prefixes a context with ``Workflow / `` when the job comes from a
reusable workflow invoked through ``workflow_call``; neither of this repo's
workflows is called that way, so a prefixed context would never be satisfied
and every merge to the protected branch would deadlock.

A preflight guard enforces the ordering that makes this safe: every required
context must already have reported a ``success`` conclusion on the branch's head
commit. Run the workflows, get them green, then protect the branch.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request

# Job names from .github/workflows/pr_quality.yml (workflow display name "CI").
# This is the whole list: pr_quality.yml is the only workflow that runs on pull
# requests, so nothing else can report a context to require here.
#
# A matrixed job reports one context per leg, named "<job> (<value>)", and does
# NOT report its bare name. `types` and `unit` run a 3.10/3.12 matrix, so they
# appear here once per leg. Requiring the bare name instead would require a
# context nothing ever produces -- which, with enforce_admins on, blocks every
# pull request permanently; see preflight() below, which refuses to write such
# a configuration.
MATRIX_PYTHON_VERSIONS = ["3.10", "3.12"]

QUALITY_CHECKS = [
    "lint",
    "doc_links",
    *[f"types ({v})" for v in MATRIX_PYTHON_VERSIONS],
    *[f"unit ({v})" for v in MATRIX_PYTHON_VERSIONS],
    "multimodal_contract",
    "launcher_contract",
    "coverage_gate",
]

DEFAULT_REQUIRED_CHECKS = QUALITY_CHECKS


def _parse_repo(repo_arg: str | None) -> tuple[str, str]:
    repo_value = repo_arg or os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo_value:
        raise ValueError(
            "Repository is required as OWNER/REPO (pass --repo or set GITHUB_REPOSITORY)."
        )
    owner, name = repo_value.split("/", 1)
    if not owner or not name:
        raise ValueError("Invalid repository format; expected OWNER/REPO.")
    return owner, name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply branch protection required checks")
    parser.add_argument("--repo", default=None, help="Repository in OWNER/REPO format")
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token with repository admin rights",
    )
    parser.add_argument(
        "--required-check",
        dest="required_checks",
        action="append",
        default=[],
        help="Override/extend required checks. Can be repeated.",
    )
    parser.add_argument("--review-count", type=int, default=1)
    parser.add_argument(
        "--no-enforce-admins",
        dest="enforce_admins",
        action="store_false",
        help=(
            "Let repository admins bypass protection. Leave this off in normal use; "
            "it exists as an escape hatch if protection is ever misconfigured."
        ),
    )
    parser.add_argument(
        "--no-require-code-owner-reviews",
        dest="require_code_owner_reviews",
        action="store_false",
        help=(
            "Do not require review from a CODEOWNERS owner. On by default because "
            ".github/CODEOWNERS exists; without a CODEOWNERS file the clause "
            "enforces nothing, so the two are kept in step by "
            "tests/platform/test_branch_protection_contract.py."
        ),
    )
    parser.set_defaults(require_code_owner_reviews=True)
    parser.add_argument(
        "--required-linear-history",
        action="store_true",
        help=(
            "Reject merge commits on the protected branch. Off by default because "
            "this repository merges pull requests with merge commits."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help=(
            "Skip the check that every required context reports success on the "
            "branch head. Only use this when the workflows are known green by "
            "other means."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _request(
    method: str,
    url: str,
    token: str,
    payload: dict | None = None,
) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url=url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, resp.read().decode("utf-8")


# The only conclusion the preflight accepts. ``skipped`` and ``neutral`` are
# deliberately excluded: a job skipped by a top-level ``if:`` leaves a required
# context unsatisfied on GitHub's side, and neither outcome is evidence that the
# job ran and passed. Refusing is the cheap error here -- the script writes
# nothing and --skip-preflight overrides it.
PASSING_CONCLUSION = "success"


def reported_check_conclusions(owner: str, repo: str, branch: str, token: str) -> dict[str, str]:
    """Map each check-run name on the branch's head commit to its outcome.

    The value is the run's ``conclusion`` once it has completed, or its
    ``status`` (``queued``, ``in_progress``) while it is still running, so a
    check that has not finished is never mistaken for one that passed. When a
    name reports more than once -- a re-run, or two apps publishing the same
    name -- the non-success outcome wins, so a stale green never masks a red.
    """
    ref = urllib.parse.quote(branch, safe="")
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{ref}/check-runs?per_page=100"
    _, body = _request("GET", url, token)
    outcomes: dict[str, str] = {}
    for run in json.loads(body).get("check_runs", []):
        if run.get("status") == "completed":
            outcome = run.get("conclusion") or "unknown"
        else:
            outcome = run.get("status") or "unknown"
        previous = outcomes.get(run["name"])
        if previous is None or previous == PASSING_CONCLUSION:
            outcomes[run["name"]] = outcome
    return outcomes


def preflight(owner: str, repo: str, branch: str, token: str, checks: list[str]) -> None:
    """Fail before writing unless every required context is green on the branch head.

    Two distinct mistakes wedge a protected branch, and with ``enforce_admins``
    on there is no way out through the UI:

    * Requiring a context nothing produces -- a workflow left
      ``disabled_manually``, or a context name that does not match the job name.
      It never reports, so every pull request waits on it forever.
    * Requiring a context that reports red. Presence alone is not evidence of a
      working gate: a job that fails on the branch head will fail on pull
      request heads too, and a required context that fails is a merge that
      cannot happen.

    Checking only that a name appeared would pass in the second case, which is
    the more likely one -- the names come from the workflow file, so they show
    up as soon as CI runs at all, green or not.
    """
    outcomes = reported_check_conclusions(owner, repo, branch, token)
    never_reported = [name for name in checks if name not in outcomes]
    not_passing = [
        (name, outcomes[name])
        for name in checks
        if name in outcomes and outcomes[name] != PASSING_CONCLUSION
    ]
    if not never_reported and not not_passing:
        return

    lines = [
        "Refusing to apply branch protection: "
        f"{owner}/{repo}@{branch} does not satisfy every required check."
    ]
    if never_reported:
        lines.append("Never reported on this commit:")
        lines += [f"  - {name}" for name in never_reported]
    if not_passing:
        lines.append("Reported but not passing:")
        lines += [f"  - {name}: {outcome}" for name, outcome in not_passing]
    passing = sorted(name for name, outcome in outcomes.items() if outcome == PASSING_CONCLUSION)
    lines.append("Passing: " + (", ".join(passing) if passing else "(none)"))
    lines.append(
        "Enable the workflows if a context never reported, fix the jobs that "
        "report red, push, and wait for a green run before re-running this "
        "script. Pass --skip-preflight to protect the branch anyway."
    )
    raise SystemExit("\n".join(lines))


def main() -> int:
    args = build_parser().parse_args()
    owner, repo = _parse_repo(args.repo)
    checks = args.required_checks or DEFAULT_REQUIRED_CHECKS

    payload = {
        "required_status_checks": {
            "strict": True,
            "contexts": checks,
        },
        "enforce_admins": args.enforce_admins,
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": args.require_code_owner_reviews,
            "required_approving_review_count": args.review_count,
        },
        "restrictions": None,
        # This repository merges pull requests with merge commits, so requiring a
        # linear history would reject its own workflow. Opt in with the flag.
        "required_linear_history": args.required_linear_history,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "allow_fork_syncing": True,
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/branches/{args.branch}/protection"
    if args.dry_run:
        print(json.dumps({"url": url, "payload": payload}, indent=2))
        return 0

    if not args.token:
        raise SystemExit("GITHUB_TOKEN is required (or pass --token).")

    if not args.skip_preflight:
        try:
            preflight(owner, repo, args.branch, args.token, checks)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(
                f"Could not read check results for {owner}/{repo}@{args.branch} "
                f"({exc.code}).\nResponse: {details}\n"
                "Pass --skip-preflight to apply protection without this guard."
            ) from exc

    try:
        status, body = _request("PUT", url, args.token, payload)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"Branch protection update failed ({exc.code}).\nURL: {url}\nResponse: {details}"
        ) from exc

    print(f"Branch protection updated ({status}) for {owner}/{repo}:{args.branch}")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pin the three settings that keep the Codecov upload from blocking merges.

`coverage_gate` is a required status context on `main` with `enforce_admins`
enabled, so anything that reddens it blocks every PR with no admin bypass. The
upload step inside it is a third-party composite action, and at the pinned SHA
its containment is narrower than it looks:

* `fail_ci_if_error` reaches only the final `dist/codecov.sh` run step
  (action.yml:334). It does not cover the rest of the composite.
* No step in that composite sets `continue-on-error`. Two of them can hard-fail
  before the upload is attempted: the OIDC mint (`core.getIDToken()`, which
  throws when no id-token is issued) and a dependency check that `exit 1`s when
  gpg is absent.
* `use_oidc: true` asks for a token the job can only mint with
  `id-token: write`. Dropping that permission while keeping `use_oidc` is the
  precise combination that makes the mint step throw on every run.

So the three settings are load-bearing together and individually inert-looking,
which is how one gets quietly removed. Coverage itself is enforced by
`--cov-fail-under` in the step above; this upload feeds the badge and the
report, and must never be what blocks a merge.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.unit]

_WORKFLOW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".github",
    "workflows",
    "pr_quality.yml",
)


@pytest.fixture(scope="module")
def upload_step() -> dict:
    yaml = pytest.importorskip("yaml")
    with open(_WORKFLOW, encoding="utf-8") as fh:
        workflow = yaml.safe_load(fh)
    job = workflow["jobs"]["coverage_gate"]
    steps = [
        step
        for step in job["steps"]
        if isinstance(step.get("uses"), str) and "codecov-action" in step["uses"]
    ]
    assert len(steps) == 1, f"expected one codecov-action step, found {len(steps)}"
    return {"job": job, "step": steps[0]}


def test_upload_cannot_fail_the_job(upload_step: dict) -> None:
    step = upload_step["step"]
    assert step.get("continue-on-error") is True, (
        "the Codecov step needs `continue-on-error: true`. `fail_ci_if_error: "
        "false` is not enough: at the pinned SHA it reaches only the final run "
        "step (action.yml:334), while the OIDC mint (action.yml:238) and the "
        "gpg dependency check (action.yml:195) can hard-fail before it. "
        "coverage_gate is a required context under enforce_admins, so either "
        "failure blocks every PR."
    )
    assert step.get("with", {}).get("fail_ci_if_error") is False, (
        "keep `fail_ci_if_error: false` as the inner half of the same guard"
    )


def test_oidc_has_the_permission_it_needs(upload_step: dict) -> None:
    job, step = upload_step["job"], upload_step["step"]
    uses_oidc = step.get("with", {}).get("use_oidc") is True
    permissions = job.get("permissions", {})
    if uses_oidc:
        assert permissions.get("id-token") == "write", (
            "`use_oidc: true` mints a JWT via core.getIDToken(), which throws "
            "unless the job grants `id-token: write`. Found job permissions: "
            f"{permissions!r}. Keep both or remove both."
        )
    else:
        assert permissions.get("id-token") != "write", (
            "`id-token: write` is granted but nothing uses it -- drop the "
            "permission rather than leaving the job over-privileged."
        )


def test_action_is_pinned_to_a_sha(upload_step: dict) -> None:
    """A moving tag would let the containment analysed above change under us."""
    ref = upload_step["step"]["uses"].split("@", 1)[1]
    assert len(ref) == 40 and all(c in "0123456789abcdef" for c in ref), (
        f"codecov-action must be pinned to a full commit SHA, found {ref!r}"
    )

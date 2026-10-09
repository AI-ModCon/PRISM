import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
SCRIPT_PATH = REPO_ROOT / "tools" / "ci" / "configure_branch_protection.py"


def _load_script():
    """Import the script as a module. ``tools/`` has no ``__init__.py``."""
    spec = importlib.util.spec_from_file_location("_branch_protection", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preflight_against(monkeypatch, check_runs: list[dict]) -> None:
    """Run ``preflight`` against a canned check-runs response.

    Stubs the HTTP layer rather than the name/conclusion reader, so the parsing
    of GitHub's payload is under test too -- that parsing is where the presence
    vs. outcome confusion lived.
    """
    module = _load_script()
    monkeypatch.setattr(
        module,
        "_request",
        lambda method, url, token, payload=None: (200, json.dumps({"check_runs": check_runs})),
    )
    module.preflight("owner", "repo", "main", "t0ken", list(module.DEFAULT_REQUIRED_CHECKS))


def _context_for(job: str) -> str:
    """One context name that ``job`` actually reports.

    A matrixed job reports ``<job> (<value>)`` per leg and never its bare name,
    so a test cannot name it with a Python identifier. Resolve it against the
    required list instead of hard-coding a leg, so these tests keep testing
    preflight rather than the matrix's current values.
    """
    module = _load_script()
    for name in module.DEFAULT_REQUIRED_CHECKS:
        if re.sub(r" \(.*\)$", "", name) == job:
            return name
    raise AssertionError(f"no required context reports job {job!r}")


def _all_green(overrides: dict | None = None) -> list[dict]:
    """A completed, successful run for every required context, with overrides.

    Overrides are keyed by the context name exactly as CI reports it -- hence a
    mapping and not keyword arguments, since ``unit (3.10)`` is not an
    identifier. An override that matches nothing is an error: it would leave the
    run green and quietly turn an assertion about a red check into a tautology.
    """
    module = _load_script()
    runs = [
        {"name": name, "status": "completed", "conclusion": "success"}
        for name in module.DEFAULT_REQUIRED_CHECKS
    ]
    for name, patch in (overrides or {}).items():
        matched = [run for run in runs if run["name"] == name]
        assert matched, f"override names no required context: {name!r}"
        for run in matched:
            run.update(patch)
    return runs


def _dry_run_payload() -> dict:
    cmd = [
        sys.executable,
        "tools/ci/configure_branch_protection.py",
        "--repo",
        "owner/repo",
        "--branch",
        "main",
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    return json.loads(result.stdout)


def test_branch_protection_dry_run_contract():
    payload = _dry_run_payload()
    assert payload["url"].endswith("/repos/owner/repo/branches/main/protection")
    contexts = payload["payload"]["required_status_checks"]["contexts"]
    assert "lint" in contexts
    assert "coverage_gate" in contexts


def test_required_contexts_are_bare_job_names():
    """GitHub reports these jobs under bare names, not ``Workflow / job``.

    The prefixed form only appears for reusable workflows invoked through
    ``workflow_call``; neither PRISM workflow is. A prefixed context could
    never be satisfied, so every merge to the protected branch would deadlock.
    """
    contexts = _dry_run_payload()["payload"]["required_status_checks"]["contexts"]
    prefixed = [c for c in contexts if "/" in c]
    assert not prefixed, f"required contexts must be bare job names, got {prefixed}"


def _reportable_context_names() -> set[str]:
    """Every context name the workflows can actually report.

    A plain job reports its bare name. A matrixed job reports one context per
    combination, named ``<job> (<v1>, <v2>, ...)`` in the matrix's own key
    order, and never reports its bare name -- so requiring the bare name of a
    matrixed job requires something that will never arrive.
    """
    names: set[str] = set()
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        for job_name, job in (yaml.safe_load(path.read_text()).get("jobs", {})).items():
            matrix = {}
            if isinstance(job, dict):
                matrix = (job.get("strategy") or {}).get("matrix") or {}
            # `include`/`exclude` are matrix controls, not axes.
            axes = [v for k, v in matrix.items() if k not in ("include", "exclude")]
            if not axes:
                names.add(job_name)
                continue
            combos = [[]]
            for axis in axes:
                combos = [c + [str(v)] for c in combos for v in axis]
            names |= {f"{job_name} ({', '.join(c)})" for c in combos}
    return names


def test_every_required_context_is_a_real_job():
    """Each required context must name something a workflow actually reports.

    This also catches the reverse mistake: adding a matrix to an already-required
    job without updating the contexts leaves the bare name required and never
    reported, which -- with ``enforce_admins`` on -- blocks every pull request
    permanently.
    """
    defined = _reportable_context_names()

    contexts = _dry_run_payload()["payload"]["required_status_checks"]["contexts"]
    unknown = sorted(set(contexts) - defined)
    assert not unknown, f"required contexts with no matching job: {unknown}"


def test_conditional_jobs_are_not_required():
    """A job gated by a top-level ``if:`` reports as skipped, which never satisfies
    a required check. Such jobs must stay out of the required list."""
    conditional = set()
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        for name, job in (yaml.safe_load(path.read_text()).get("jobs", {})).items():
            if isinstance(job, dict) and "if" in job:
                conditional.add(name)

    contexts = set(_dry_run_payload()["payload"]["required_status_checks"]["contexts"])
    # Map "job (3.10)" back to "job" so matrixed jobs are checked too.
    bare = {re.sub(r" \(.*\)$", "", c) for c in contexts}
    overlap = sorted(bare & conditional)
    assert not overlap, f"conditional jobs must not be required contexts: {overlap}"


def test_linear_history_is_off_by_default():
    """The repository merges pull requests with merge commits, so requiring a
    linear history would reject its own merge workflow."""
    assert _dry_run_payload()["payload"]["required_linear_history"] is False


def test_code_owner_reviews_require_a_codeowners_file():
    """``require_code_owner_reviews`` enforces nothing without CODEOWNERS, and
    creates a false assurance of review coverage. Keep the two in step."""
    has_codeowners = any((REPO_ROOT / p / "CODEOWNERS").exists() for p in (".", ".github", "docs"))
    required = _dry_run_payload()["payload"]["required_pull_request_reviews"][
        "require_code_owner_reviews"
    ]
    assert required == has_codeowners, (
        "require_code_owner_reviews is "
        f"{required} but a CODEOWNERS file "
        f"{'exists' if has_codeowners else 'does not exist'}"
    )


def test_documented_required_checks_match_the_script():
    """docs/development/ci_multiplatform.md lists the required checks for humans;
    a drifted list there is how the prefixed names survived in the first place."""
    doc = (REPO_ROOT / "docs" / "development" / "ci_multiplatform.md").read_text()
    section = doc.split("## Required PR checks", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"^- `([^`]+)`", section, flags=re.MULTILINE))
    contexts = set(_dry_run_payload()["payload"]["required_status_checks"]["contexts"])
    assert documented == contexts, (
        f"documented but not required: {sorted(documented - contexts)}; "
        f"required but not documented: {sorted(contexts - documented)}"
    )


def test_required_contexts_come_from_pull_request_workflows():
    """A workflow that does not run on ``pull_request`` never reports a context on
    a PR, so requiring one of its jobs would deadlock the branch."""
    pr_jobs = set()
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        # ``on`` is the YAML 1.1 boolean True once parsed, hence the fallback.
        triggers = workflow.get("on", workflow.get(True, {})) or {}
        if isinstance(triggers, str):
            triggers = {triggers: None}
        if isinstance(triggers, list):
            triggers = dict.fromkeys(triggers)
        if "pull_request" in triggers:
            pr_jobs |= set(workflow.get("jobs", {}))

    contexts = set(_dry_run_payload()["payload"]["required_status_checks"]["contexts"])
    # Map "job (3.10)" back to "job": a matrix leg is reachable exactly when
    # its job is.
    bare = {re.sub(r" \(.*\)$", "", c) for c in contexts}
    unreachable = sorted(bare - pr_jobs)
    assert not unreachable, f"required contexts no pull_request workflow can report: {unreachable}"


def test_preflight_passes_when_every_required_check_is_green(monkeypatch):
    """The ordering the guard exists to enforce: green first, then protect."""
    _preflight_against(monkeypatch, _all_green())


def test_preflight_rejects_a_failing_required_check(monkeypatch):
    """A red required context is a merge that cannot happen.

    This is the case an earlier version of the guard let through: it collected
    check-run *names* and tested membership, so a failing job satisfied it. The
    names come straight out of the workflow file, so they appear the moment CI
    runs at all -- presence proves the workflow is enabled, nothing more.
    """
    with pytest.raises(SystemExit) as excinfo:
        _preflight_against(
            monkeypatch, _all_green({_context_for("types"): {"conclusion": "failure"}})
        )
    message = str(excinfo.value)
    assert "types" in message
    assert "failure" in message


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", "action_required"])
def test_preflight_rejects_every_unsuccessful_conclusion(monkeypatch, conclusion):
    with pytest.raises(SystemExit):
        _preflight_against(
            monkeypatch, _all_green({_context_for("unit"): {"conclusion": conclusion}})
        )


@pytest.mark.parametrize("conclusion", ["skipped", "neutral"])
def test_preflight_rejects_skipped_and_neutral(monkeypatch, conclusion):
    """Neither outcome means the job ran and passed, and GitHub does not count a
    skipped run as satisfying a required context."""
    with pytest.raises(SystemExit):
        _preflight_against(
            monkeypatch, _all_green({_context_for("lint"): {"conclusion": conclusion}})
        )


def test_preflight_rejects_a_check_still_running(monkeypatch):
    """An unfinished run has ``conclusion: null``; that must not read as green."""
    with pytest.raises(SystemExit) as excinfo:
        _preflight_against(
            monkeypatch,
            _all_green(
                {_context_for("coverage_gate"): {"status": "in_progress", "conclusion": None}}
            ),
        )
    assert "in_progress" in str(excinfo.value)


def test_preflight_still_rejects_a_context_that_never_reported(monkeypatch):
    """The original guarantee survives, and reports separately from a red check."""
    runs = [run for run in _all_green() if run["name"] != "doc_links"]
    with pytest.raises(SystemExit) as excinfo:
        _preflight_against(monkeypatch, runs)
    message = str(excinfo.value)
    assert "Never reported" in message
    assert "doc_links" in message


@pytest.mark.parametrize("green_first", [True, False])
def test_preflight_lets_a_red_rerun_outweigh_a_green_one(monkeypatch, green_first):
    """One name can report twice -- a re-run, or two apps publishing the name.
    The unsuccessful outcome has to win in either arrival order, or a stale
    green masks a red."""
    unit = _context_for("unit")
    green = {"name": unit, "status": "completed", "conclusion": "success"}
    red = {"name": unit, "status": "completed", "conclusion": "failure"}
    runs = [run for run in _all_green() if run["name"] != unit]
    runs += [green, red] if green_first else [red, green]
    with pytest.raises(SystemExit):
        _preflight_against(monkeypatch, runs)


def test_preflight_ignores_checks_that_are_not_required(monkeypatch):
    """A red check outside the required set must not block protection; requiring
    it is the separate decision the contexts list makes."""
    runs = _all_green()
    runs.append({"name": "Dependabot", "status": "completed", "conclusion": "failure"})
    _preflight_against(monkeypatch, runs)

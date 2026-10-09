"""Keep the Makefile's recipes identical to the ones CI runs.

pytest does not error on an unknown marker inside a `not` clause -- it
deselects nothing and exits 0. So a typo in the Makefile's UNIT_MARKERS or
COV_MARKERS makes `make test` pass while running a different, larger set than
CI does, and the difference only surfaces when CI rejects the PR. The same
holds, more quietly, for the coverage floor and the ruff/mypy scopes: a local
run that demands less coverage or checks fewer paths than CI is green for the
wrong reason.

Every check below is bound to the SPECIFIC workflow job the Makefile target
mirrors -- `jobs.unit` for UNIT_MARKERS, `jobs.coverage_gate` for COV_MARKERS
and the floor, and so on. Matching a value against "some -m selector, anywhere
in the workflow" would happily accept the two marker sets swapped.

Nothing here hard-codes a scope or a floor: both sides are read at run time and
compared to each other, so widening the ruff scope or raising the coverage gate
needs no edit in this file -- it only has to be done on both sides.
"""

from __future__ import annotations

import os
import re
import shlex

import pytest
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAKEFILE = os.path.join(_ROOT, "Makefile")
_WORKFLOW = os.path.join(_ROOT, ".github", "workflows", "pr_quality.yml")

# Flags that swallow the following token, so the scope reader below does not
# mistake a flag's value for a path. `mypy --python-version <X> src` is the
# only separated form in play today; everything else is spelled `--flag=value`.
# `-k` is listed because it takes a separated value AND changes which tests
# run, so it is compared explicitly rather than read as a path.
_VALUE_FLAGS = {"--python-version", "-m", "-k", "-p", "-n", "--cov-fail-under"}

# Only the Makefile is required module-wide. The workflow is required by the
# readers that actually open it, so a Makefile-only checkout still exercises
# the checks that read no workflow -- test_markers_are_distinct is one.
pytestmark = pytest.mark.skipif(
    not os.path.isfile(_MAKEFILE),
    reason="Makefile absent (not a full checkout)",
)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _make_var(name: str) -> str:
    """The value of a `NAME := value` assignment in the Makefile."""
    match = re.search(rf"(?m)^{name}\s*:?=\s*(.+)$", _read(_MAKEFILE))
    assert match is not None, f"{name} not found in {_MAKEFILE}"
    return match.group(1).strip()


def _make_recipe(target: str) -> str:
    """The recipe lines of `target`, folded into one command string.

    Recipe lines are tab-indented and a trailing backslash continues the same
    shell command onto the next, exactly as in the workflow's `run:` blocks.
    """
    match = re.search(rf"(?m)^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n|\n)*)", _read(_MAKEFILE))
    assert match is not None, f"no `{target}` target in {_MAKEFILE}"
    body = match.group(1)
    assert body.strip(), f"`{target}` in {_MAKEFILE} has an empty recipe"
    return re.sub(r"\\\n\s*", " ", body).replace("\t", " ")


def _job_command(job: str, tool: str) -> str:
    """The `run:` block of `job` that invokes `tool`, folded into one string.

    GitHub `${{ ... }}` expressions collapse to a single token so a matrix
    value cannot be read as a path.
    """
    if not os.path.isfile(_WORKFLOW):
        pytest.skip("pr_quality.yml absent (not a full checkout)")
    jobs = yaml.safe_load(_read(_WORKFLOW)).get("jobs")
    assert isinstance(jobs, dict), f"{_WORKFLOW} declares no jobs mapping"
    assert job in jobs, f"{_WORKFLOW} has no `{job}` job (jobs: {sorted(jobs)})"
    matches = [
        step["run"]
        for step in jobs[job].get("steps", [])
        if isinstance(step.get("run"), str) and re.search(rf"(?m)^\s*{tool}\b", step["run"])
    ]
    assert len(matches) == 1, (
        f"expected exactly one `{tool}` run step in job `{job}`, found {len(matches)}"
    )
    return re.sub(r"\$\{\{.*?\}\}", "EXPR", re.sub(r"\\\n\s*", " ", matches[0]))


def _positional_after(command: str, tool: str) -> list[str]:
    """The non-flag arguments `tool` is given in `command` -- i.e. its scope."""
    tokens = shlex.split(command)
    assert tool in tokens, f"{tool!r} not invoked in: {command.strip()!r}"
    scope: list[str] = []
    skip = False
    for token in tokens[tokens.index(tool) + 1 :]:
        if skip:
            skip = False
        elif token.startswith("-"):
            skip = token in _VALUE_FLAGS
        else:
            scope.append(token)
    return scope


def _marker_selector(command: str) -> str:
    """The single `-m "..."` selector in a pytest invocation."""
    found = re.findall(r'-m\s+"([^"]+)"', command)
    assert len(found) == 1, f"expected one -m selector, found {found} in: {command.strip()!r}"
    return found[0]


def _keyword_selector(command: str) -> list[str]:
    """Any `-k` expressions in a pytest invocation.

    `-k` is a value flag, so the scope reader skips it along with its argument.
    Skipping is right for paths and wrong for parity: `-k` narrows the run just
    as surely as a path does, so it is compared here instead of ignored.
    """
    tokens = shlex.split(command)
    return [tokens[i + 1] for i, tok in enumerate(tokens[:-1]) if tok == "-k"]


def _cov_floor(command: str) -> str:
    """The `--cov-fail-under` value in a pytest invocation."""
    found = re.findall(r"--cov-fail-under[= ](\d+)", command)
    assert len(found) == 1, f"expected one --cov-fail-under, found {found} in: {command.strip()!r}"
    return found[0]


@pytest.mark.parametrize(
    ("var", "job", "target"),
    [
        ("UNIT_MARKERS", "unit", "test"),
        ("COV_MARKERS", "coverage_gate", "test-cov"),
    ],
)
def test_marker_variable_matches_the_job_it_mirrors(var: str, job: str, target: str) -> None:
    """Each variable equals ITS job's selector, not merely some job's selector."""
    recipe = _make_recipe(target)
    assert f"$({var})" in recipe, (
        f"the `{target}` recipe no longer passes $({var}), so checking that "
        f"variable against CI's `{job}` job would prove nothing:\n  {recipe.strip()!r}"
    )
    value = _make_var(var)
    expected = _marker_selector(_job_command(job, "pytest"))
    assert value == expected, (
        f"Makefile {var} is:\n  {value!r}\nbut CI's `{job}` job selects:\n  {expected!r}\n"
        "An unknown marker in a `not` clause deselects nothing and exits 0, so "
        f"`make {target}` would run the wrong set and still look green."
    )


@pytest.mark.parametrize(
    ("job", "target"),
    [
        ("unit", "test"),
        ("coverage_gate", "test-cov"),
        ("multimodal_contract", "multimodal"),
        ("launcher_contract", "launcher"),
    ],
)
def test_pytest_target_paths_match_ci(job: str, target: str) -> None:
    """Each make target collects from the same paths as the job it mirrors.

    Comparing only the `-m` selector leaves this wide open: retargeting `make
    test` from `tests` to `tests/platform`, or pointing `make multimodal` at a
    different file than its job names, both keep the selector identical and
    exit 0. The paths are the other half of what decides which tests run.
    """
    recipe, command = _make_recipe(target), _job_command(job, "pytest")
    actual = _positional_after(recipe, "pytest")
    expected = _positional_after(command, "pytest")
    assert actual == expected, (
        f"`make {target}` collects from {actual} but CI's `{job}` job collects "
        f"from {expected}. Tests only one side runs are passing by accident."
    )
    actual_k, expected_k = _keyword_selector(recipe), _keyword_selector(command)
    assert actual_k == expected_k, (
        f"`make {target}` passes -k {actual_k} but CI's `{job}` job passes "
        f"-k {expected_k}. A local -k narrows the run without narrowing CI's."
    )


def test_coverage_measures_the_same_package_as_ci() -> None:
    """`make test-cov` measures what the gate measures.

    `--cov-fail-under` is meaningless without this: a local run pointed at a
    different package (or with `--cov=` dropped, which makes pytest-cov measure
    everything imported) clears the same 34% floor while saying nothing about
    the package CI gates on.
    """
    pattern = r"--cov[= ]([^\s-][^\s]*)"
    actual = re.findall(pattern, _make_recipe("test-cov"))
    expected = re.findall(pattern, _job_command("coverage_gate", "pytest"))
    assert expected, "CI's coverage_gate job no longer passes --cov=<package>"
    assert actual == expected, (
        f"`make test-cov` measures {actual} but CI's coverage_gate job measures "
        f"{expected}. The --cov-fail-under floor then compares two different "
        "numbers, and the local one is green for the wrong reason."
    )


def test_markers_are_distinct() -> None:
    """The unit job and the coverage gate deliberately select different sets."""
    assert _make_var("UNIT_MARKERS") != _make_var("COV_MARKERS"), (
        "UNIT_MARKERS and COV_MARKERS are identical; one of them was almost "
        "certainly copied over the other. CI's unit job additionally excludes "
        "the multimodal and launcher markers, which have their own jobs."
    )


def test_coverage_floor_matches_ci() -> None:
    """`make test-cov` fails at the same coverage as the gate that blocks a PR."""
    actual = _cov_floor(_make_recipe("test-cov"))
    expected = _cov_floor(_job_command("coverage_gate", "pytest"))
    assert actual == expected, (
        f"`make test-cov` uses --cov-fail-under={actual} but CI's coverage_gate "
        f"job uses {expected}. A lower local floor passes work CI then rejects; "
        "a higher one blocks work CI would accept."
    )


@pytest.mark.parametrize(
    ("tool", "job", "target"),
    [
        ("ruff", "lint", "lint"),
        ("mypy", "types", "type-check"),
    ],
)
def test_checker_scope_matches_ci(tool: str, job: str, target: str) -> None:
    """The paths handed to ruff/mypy locally are the paths CI checks.

    Read off both sides rather than compared to a literal, so widening the
    scope stays a two-file edit and never a three-file one.
    """
    actual = _positional_after(_make_recipe(target), tool)
    expected = _positional_after(_job_command(job, tool), tool)
    assert actual == expected, (
        f"`make {target}` runs {tool} over {actual} but CI's `{job}` job uses "
        f"{expected}. A path only one side checks is clean by accident."
    )


def test_every_pytest_job_has_a_make_target() -> None:
    """No CI pytest job runs work that `make` cannot reproduce locally.

    The four pairs above are checked by name; this catches the case where a
    new pytest job is added to the workflow and no one adds the matching
    target, which would leave that job unreproducible before a push.
    """
    if not os.path.isfile(_WORKFLOW):
        pytest.skip("pr_quality.yml absent (not a full checkout)")
    jobs = yaml.safe_load(_read(_WORKFLOW))["jobs"]
    with_pytest = {
        job
        for job, spec in jobs.items()
        if any(
            isinstance(step.get("run"), str) and re.search(r"(?m)^\s*pytest\b", step["run"])
            for step in spec.get("steps", [])
        )
    }
    covered = {"unit", "coverage_gate", "multimodal_contract", "launcher_contract"}
    assert with_pytest == covered, (
        f"CI runs pytest in {sorted(with_pytest)} but this file binds make "
        f"targets to {sorted(covered)}. Add a target for "
        f"{sorted(with_pytest - covered)} and a pair to the parametrize list "
        "above (or drop the stale entry) so the job stays reproducible locally."
    )

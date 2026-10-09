"""Keep Dependabot away from the HPC requirements files.

Those manifests pin a stack built against a site-provided torch. A resolver
raising a version in one of them does not produce a routine upgrade PR; it
produces a cluster install that aborts at import with
``undefined symbol: __kmpc_fork_call``. The config narrows Dependabot to
``requirements/ci.txt`` and the GitHub Actions workflows, and these tests fail
if that narrowing stops holding.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".github" / "dependabot.yml"

# The CI-only tools Dependabot is allowed to propose. Everything else in
# ci.txt is deliberately unpinned so CI floats it.
CI_TOOLS = {"ruff", "mypy", "pytest", "pytest-cov"}

# The manifests Dependabot is meant to read. Everything else under
# requirements/ is excluded; see test_every_hpc_manifest_is_excluded.
IN_SCOPE = {"ci.txt"}

# Packages whose version is dictated by the site-built torch ABI, never by a
# resolver. Spelled out rather than wildcarded: `torch-*` would also match
# torch-geometric, which is pure Python.
ABI_PINNED = {
    "torch",
    "torchvision",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "pyg-lib",
    "deepspeed",
}


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def pip_update(config: dict) -> dict:
    updates = [u for u in config["updates"] if u["package-ecosystem"] == "pip"]
    assert len(updates) == 1, "expected exactly one pip update block"
    return updates[0]


def test_config_is_version_2(config: dict):
    assert config["version"] == 2


def test_every_hpc_manifest_is_excluded(pip_update: dict):
    """A new file under requirements/ must be excluded or deliberately in scope.

    This is the test the config's header comment points at: without it, adding
    a manifest silently widens Dependabot's reach.
    """
    directory = pip_update["directory"].lstrip("/")
    excluded = set(pip_update["exclude-paths"])

    present = {p.name for p in (REPO_ROOT / directory).iterdir() if p.is_file()}
    uncovered = sorted(present - excluded - IN_SCOPE)
    assert not uncovered, (
        f"manifests under {directory}/ that are neither excluded nor "
        f"deliberately in scope: {uncovered}"
    )


def test_exclude_paths_are_relative_to_the_directory(pip_update: dict):
    """Dependabot resolves ``exclude-paths`` against ``directory``, not the repo root.

    Written as ``requirements/base.txt`` under ``directory: /requirements`` the
    pattern matches nothing, and the file it was meant to protect gets swept in.
    """
    directory = pip_update["directory"].lstrip("/")
    for pattern in pip_update["exclude-paths"]:
        assert not pattern.startswith(f"{directory}/"), (
            f"{pattern!r} is repo-root-relative; patterns resolve against "
            f"{pip_update['directory']!r}"
        )
        assert (REPO_ROOT / directory / pattern).exists(), (
            f"{pattern!r} matches no file under {directory}/"
        )


def test_allow_list_covers_only_ci_tools(pip_update: dict):
    allowed = {entry["dependency-name"] for entry in pip_update["allow"]}
    assert allowed == CI_TOOLS


def test_abi_pinned_packages_are_ignored(pip_update: dict):
    ignored = {entry["dependency-name"] for entry in pip_update["ignore"]}
    missing = sorted(ABI_PINNED - ignored)
    assert not missing, f"ABI-sensitive packages not in the ignore list: {missing}"


def test_torch_wildcard_is_not_used(pip_update: dict):
    """``torch-*`` would also match torch-geometric, which needs no freezing."""
    ignored = {entry["dependency-name"] for entry in pip_update["ignore"]}
    assert "torch-*" not in ignored


def test_no_labels_are_configured(config: dict):
    """Dependabot ignores labels that do not exist in the repository, and naming
    any label replaces the defaults it would otherwise create. The repo defines
    none of the usual dependency labels, so listing them would leave the PRs
    unlabelled entirely."""
    for update in config["updates"]:
        assert "labels" not in update, (
            "remove `labels:` unless the named labels exist in the repository"
        )


# Transcribed from dependabot-core's
# python/lib/dependabot/python/file_fetcher.rb `requirements_file?`, which is
# the gate every candidate .txt passes before the pip fetcher will look at it.
# A line is acceptable when it is blank, a comment, a directive, or a
# requirement the parser recognises. Notably absent from that list: PEP 508
# direct references (`name @ git+https://...`), because `@` is not part of a
# name, is not a comparison operator, and what follows is not a version.
_DIRECTIVE_PREFIXES = ("#", "-r ", "-c ", "-e ", "--")

_NAME = r"[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?"
_EXTRAS = rf"\[(?:{_NAME}(?:\s*,\s*{_NAME})*)\]"
_COMPARISON = r"(?:===|==|!=|>=|<=|<|>|~=)"
_VERSION = r"[0-9a-zA-Z*.+!-]+"
_CONSTRAINT = rf"{_COMPARISON}\s*{_VERSION}"
_REQUIREMENT = re.compile(
    rf"^{_NAME}(?:\s*{_EXTRAS})?\s*(?:{_CONSTRAINT}(?:\s*,\s*{_CONSTRAINT})*)?"
    rf"(?:\s*;.*)?$"
)


def _line_is_fetchable(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith(_DIRECTIVE_PREFIXES):
        return True
    return bool(_REQUIREMENT.match(stripped.split("#", 1)[0].strip()))


def test_in_scope_manifest_is_parseable_by_the_pip_fetcher(pip_update: dict):
    """One unparseable line silently un-schedules Dependabot for the whole repo.

    The fetcher keeps a candidate .txt only if every one of its lines is
    acceptable. ``requirements/ci.txt`` is the only candidate here -- every
    sibling is excluded -- so rejecting it leaves the fetcher with nothing, and
    it does not degrade gracefully: it aborts with "No files found in
    /requirements" and CLOSES the open update PRs, reporting that the
    dependencies are no longer present. #204 and #205 were closed that way by a
    ``torchtune @ git+https://...`` line, which is a valid pip requirement and
    an invalid one to Dependabot.

    The failure is invisible from the repository -- nothing goes red, the PRs
    just stop arriving -- which is why it is asserted here rather than left to
    be noticed.
    """
    directory = pip_update["directory"].lstrip("/")
    for name in sorted(IN_SCOPE):
        manifest = REPO_ROOT / directory / name
        offenders = [
            (number, line)
            for number, line in enumerate(manifest.read_text().splitlines(), start=1)
            if not _line_is_fetchable(line)
        ]
        assert not offenders, (
            f"{directory}/{name} has lines Dependabot's pip fetcher cannot "
            "parse, which makes it skip the file and close its open PRs:\n"
            + "\n".join(f"  line {n}: {line}" for n, line in offenders)
            + "\n\nA git dependency belongs in an HPC manifest (see "
            "requirements/base.txt, which documents the clone-and-install "
            "step), not here."
        )

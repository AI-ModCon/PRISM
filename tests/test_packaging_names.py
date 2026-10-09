"""Keep the packaging names in tests and pyproject.toml from drifting apart.

`test_vllm_entry_point.py` looks the installed distribution up by name and
*skips* when it is absent. That is the right behaviour on a clean checkout,
but it means a rename in pyproject.toml turns those tests into permanent
silent skips rather than failures — the exact regression the entry-point
test exists to catch, hidden by the mechanism meant to report it.

These tests need no install and no TOML parser (the 3.10 CI leg has no
`tomllib`), so they run on every leg and fail loudly on drift.
"""

from __future__ import annotations

import os
import re

from tests.test_vllm_entry_point import DIST_NAME, ENTRY_POINT_NAME

_PYPROJECT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml"
)


def _pyproject_text() -> str:
    with open(_PYPROJECT, encoding="utf-8") as fh:
        return fh.read()


def test_dist_name_matches_pyproject_project_name():
    """`prism-mm` — the PyPI distribution name, `[project].name`."""
    text = _pyproject_text()
    match = re.search(r'(?m)^name\s*=\s*"([^"]+)"', text)
    assert match is not None, f"no [project].name found in {_PYPROJECT}"
    assert match.group(1) == DIST_NAME, (
        f"pyproject [project].name is {match.group(1)!r} but "
        f"tests/test_vllm_entry_point.py looks up {DIST_NAME!r}. Those tests "
        "skip on PackageNotFoundError, so a mismatch disables them silently "
        "instead of failing. Update DIST_NAME."
    )


def test_entry_point_name_matches_pyproject():
    """`prism` — the vLLM plugin entry-point name, deliberately unrenamed."""
    text = _pyproject_text()
    block = text.split('[project.entry-points."vllm.general_plugins"]', 1)
    assert len(block) == 2, (
        'no [project.entry-points."vllm.general_plugins"] table in pyproject.toml'
    )
    # Read to the next table header, so a later section cannot satisfy this.
    body = re.split(r"(?m)^\[", block[1], maxsplit=1)[0]
    names = re.findall(r"(?m)^([A-Za-z0-9_.-]+)\s*=", body)
    assert ENTRY_POINT_NAME in names, (
        f"{ENTRY_POINT_NAME!r} missing from the vllm.general_plugins table "
        f"(found {names}); test_vllm_entry_point.py asserts on that name."
    )


def test_console_script_name_is_unchanged():
    """The `prism` command is user-facing; renaming it would break every doc."""
    text = _pyproject_text()
    block = text.split("[project.scripts]", 1)
    assert len(block) == 2, "no [project.scripts] table in pyproject.toml"
    body = re.split(r"(?m)^\[", block[1], maxsplit=1)[0]
    assert re.search(r'(?m)^prism\s*=\s*"src\.cli:app"', body), (
        "the `prism` console script is gone or repointed; README.md, "
        "docs/getting-started.md and docs/training/cli.md all document it."
    )

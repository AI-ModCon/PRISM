"""Keep every tracked Python file visible to the lint gate.

Ruff honours `.gitignore` even for files git *tracks*, and it does so while
walking a directory. So a `.gitignore` pattern can hide a tracked, shipped file
from `ruff check ... scripts` while `git status` stays silent about it: the job
goes green having never read the file.

That is not hypothetical. `scripts/download_model.py` is tracked, and the bare
pattern `download_model.py` -- bare patterns match at any depth -- hid it from
the gate along with two real findings. `tools/download_docci.py` sat behind
`tools/download_*.py` the same way. Both are un-hidden with `!` negations.

The fix is negations rather than `respect-gitignore = false`, because that
setting would also drag every developer's untracked scratch file into `make
lint`. This test therefore compares the *tracked* set against the visible set
and says nothing about untracked files.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAKEFILE = os.path.join(_ROOT, "Makefile")


def _lint_scope() -> list[str]:
    """The paths `make lint` hands to ruff.

    Read from the Makefile rather than hard-coded here, so widening the gate in
    one place widens this check too. test_makefile_ci_parity.py is what pins the
    Makefile to the workflow.
    """
    with open(_MAKEFILE, encoding="utf-8") as fh:
        match = re.search(r"(?m)^\t.*-m ruff check ([^\n]+)$", fh.read())
    assert match is not None, "no `ruff check` recipe line in the Makefile"
    return match.group(1).split()


def _stdout(args: list[str]) -> str:
    return subprocess.run(
        args, cwd=_ROOT, capture_output=True, text=True, check=False
    ).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")
def test_every_tracked_python_file_is_visible_to_ruff() -> None:
    scope = [p for p in _lint_scope() if os.path.exists(os.path.join(_ROOT, p))]
    assert scope, "no lint-scope directories exist"

    tracked = {
        p for p in _stdout(["git", "ls-files", *scope]).split() if p.endswith(".py")
    }
    assert tracked, f"no tracked .py files found under {scope!r}"

    visible = {
        os.path.relpath(p, _ROOT) if os.path.isabs(p) else p
        for p in _stdout(["ruff", "check", "--show-files", *scope]).split()
        if p.endswith(".py")
    }

    hidden = sorted(tracked - visible)
    assert not hidden, (
        f"tracked by git but invisible to `ruff check {' '.join(scope)}`, so the "
        "lint gate never reads them:\n"
        + "\n".join(f"  {p}" for p in hidden)
        + "\n\nRuff respects .gitignore even for tracked files. Find the pattern "
        "with `git check-ignore -v --no-index <path>`, then add a `!` negation "
        "for the tracked file -- .gitignore already does this for "
        "scripts/download_model.py and tools/download_docci.py."
    )

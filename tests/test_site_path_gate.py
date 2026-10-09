"""Tests for tools/ci/check_site_paths.py — the hardcoded-path ratchet.

The gate's value depends entirely on its regex being neither too greedy nor too
narrow. Too greedy and it flags the sanitized "<placeholder>" templates the
repository is supposed to use, so contributors learn to ignore it; too narrow
and a real "/lus/flare/projects/<real-project>/<real-user>" reaches a public
repository. Both halves are pinned below with literal fragments taken from the
tree.

The end-to-end check also runs the real gate against the real repository, which
is the only way to catch an allowlist that has drifted out of sync with the
files it describes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATE = os.path.join(_ROOT, "tools", "ci", "check_site_paths.py")


def _load_gate():
    """Import the gate by path — tools/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("check_site_paths", _GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


# Fixtures are ASSEMBLED, never written literally. The gate scans tracked
# source text, so a literal "/lus/flare/projects/<something>/<someone>" in this
# file would be a genuine finding against this file — and exempting the test
# from its own gate would be worse than the inconvenience of joining strings.
_SEP = "/"


def _path(*segments: str) -> str:
    return _SEP + _SEP.join(segments)


@pytest.mark.parametrize(
    "text",
    [
        _path("lus", "flare", "projects", "ModCon", "someone", "data"),
        "DATA=" + _path("flare", "ModCon", "someone", "data", "zone_a"),
        '"' + _path("eagle", "projects", "argonne_tpc", "someone", "output") + '"',
        _path("home", "someone", "projects", "modcon", "BaseMM_PRISM"),
        _path("raid", "someone", "PRISM", "data"),
        _path("global", "homes", "s", "someone", "prism"),
    ],
)
def test_real_site_paths_are_flagged(text):
    assert gate.occurrences(text), f"missed a real site path: {text}"


@pytest.mark.parametrize(
    "text",
    [
        # Sanitized templates — the documented replacement, must stay silent.
        "/lus/flare/projects/<project>/<user>/data/zone_a",
        "/flare/<project>/<your_username>/huggingface/hub",
        # Shell and Hydra expansion.
        "/flare/ModCon/$USER/prism-envs/py3.12",
        "${PRISM_DATA_ROOT}/SciTS-processed",
        "${oc.env:PRISM_TOKENIZERS}/prism-olmo-1b-interleaved",
        # Elided paths in docs and test fixtures.
        "/flare/.../pixmo_cap_webdataset",
        # A bare project root is not one user's private directory.
        _SEP + "flare" + _SEP + "ModCon",
        # Unrelated absolute paths outside the HPC filesystem roots.
        "/usr/local/bin/python",
        "/tmp/smoke_scits/shards",
    ],
)
def test_templates_and_non_site_paths_are_not_flagged(text):
    assert not gate.occurrences(text), f"false positive on: {text}"


def test_line_numbers_are_one_based():
    path = _path("flare", "ModCon", "someone", "data")
    found = gate.occurrences(f"first\nsecond {path}\n")
    assert found == [(2, path)]


def test_src_tree_is_free_of_hardcoded_site_paths():
    """src/ is the shipped library; unlike scripts/, it carries no legacy tail.

    Every site root it needs goes through src/site_paths.py or a sanitized
    env-var default, so its count is zero and must stay zero.
    """
    counts, _ = gate.scan()
    offenders = {path: n for path, n in counts.items() if path.startswith("src/")}
    assert not offenders, f"src/ gained hardcoded site paths: {offenders}"


def test_living_documentation_is_scanned():
    """The near-miss this exists for: a setup paragraph naming a real path.

    docs/ prose is where "here is how our team configures this" gets written,
    and it publishes exactly what src/ was cleaned of. Only frozen artifacts
    are exempt.
    """
    assert "docs/" in gate.SCANNED_PREFIXES
    scanned = gate.tracked_files()
    assert "docs/platforms/site_paths.md" in scanned
    assert "docs/getting-started.md" in scanned


@pytest.mark.parametrize(
    "path",
    [
        "docs/assets/image_generation/anything.json",
        "docs/reports/2026-09-21-some-run.md",
        "docs/data/public_image_sources/20260921/delivery-receipt.json",
    ],
)
def test_frozen_artifacts_stay_exempt(path):
    """Rewriting a provenance snapshot or a dated run report destroys it."""
    assert path.startswith(gate.EXEMPT_PREFIXES)
    assert path not in gate.tracked_files()


def test_results_pages_are_scanned_despite_being_dated():
    """``docs/results/`` is read as instructions, so it is not exempt.

    It was exempt alongside ``docs/reports/`` until its paths were cleaned
    up. The two differ in how they are used: ``reports/`` pages are
    date-named records of one run, while ``results/`` pages are undated and
    the index advertises them as reusable ("how to run it", "launch
    commands"), so a reader copies commands out of them.
    """
    assert not "docs/results/".startswith(gate.EXEMPT_PREFIXES)
    scanned = gate.tracked_files()
    assert any(p.startswith("docs/results/") for p in scanned)


def test_the_site_paths_guide_names_no_real_path():
    """The page telling people to stop hardcoding paths must not hardcode one.

    It is in scope now, so the ratchet would catch a regression -- but it
    carries the shared-allocation recipe, which is the single most likely
    place for a real value to be pasted in.
    """
    with open(os.path.join(_ROOT, "docs", "platforms", "site_paths.md"), encoding="utf-8") as fh:
        found = gate.occurrences(fh.read())
    assert not found, f"site_paths.md names a real site path: {found}"


def test_allowlist_matches_the_tree_exactly():
    """The committed allowlist must describe the tree as it is right now.

    A stale entry means the ratchet is measuring against a file state that no
    longer exists, and would let a real regression through on the next edit.
    """
    counts, _ = gate.scan()
    allowed = gate.load_allowlist()
    assert counts == allowed, (
        "allowlist is out of date; run "
        "`python tools/ci/check_site_paths.py --write` and commit the result"
    )


def test_allowlist_is_sorted_and_positive():
    with open(gate.ALLOWLIST, encoding="utf-8") as handle:
        files = json.load(handle)["files"]
    assert list(files) == sorted(files)
    assert all(count > 0 for count in files.values())


def test_gate_exits_zero_on_the_current_tree():
    result = subprocess.run(
        [sys.executable, _GATE], cwd=_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

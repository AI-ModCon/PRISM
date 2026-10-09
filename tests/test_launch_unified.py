"""Tests for tools/launch_aurora_unified.py — the dispatcher.

These are integration tests (subprocess-driven) because the dispatcher uses
`os.execvp` to replace the process with the chosen launcher. We verify:

- ``--storage X --dry-run`` invokes the right launcher (each launcher prints
  a distinct ""Generated Run Script: jobs/run_aurora_{X}_..."" line)
- bare ``--help`` shows our help
- ``--storage X --help`` forwards ``--help`` to the launcher
- unknown ``--storage`` is rejected by argparse
- missing ``--storage`` is rejected by argparse

We avoid actually launching jobs by relying on each launcher's ``--dry-run``
flag, which generates a script in ``jobs/`` and exits without running it.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = REPO_ROOT / "tools" / "launch_aurora_unified.py"


def _extract_script_path(stdout: str) -> Path:
    """Pull the `Generated Run Script: <path>` line out of launcher stdout."""
    match = re.search(r"Generated Run Script:\s*(\S+)", stdout)
    assert match, f"no 'Generated Run Script:' line in stdout:\n{stdout}"
    path = Path(match.group(1))
    if not path.is_absolute():
        path = REPO_ROOT / path
    assert path.exists(), f"launcher claimed to write {path} but it doesn't exist"
    return path


def _run(*args: str, expect_exit: int | None = None, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, str(DISPATCHER), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    if expect_exit is not None:
        assert result.returncode == expect_exit, (
            f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def test_bare_help_shows_dispatcher_help():
    result = _run("--help", expect_exit=0)
    assert "Unified PRISM launcher" in result.stdout
    assert "--storage" in result.stdout
    # Storage choices are listed
    assert "daos" in result.stdout
    assert "lustre" in result.stdout
    assert "webdataset-staged" in result.stdout


def test_missing_storage_is_rejected():
    result = _run(expect_exit=2)
    assert "--storage" in result.stderr or "--storage" in result.stdout


def test_unknown_storage_is_rejected():
    result = _run("--storage", "nfs", expect_exit=2)
    assert "invalid choice" in result.stderr.lower() or "nfs" in result.stderr


def test_storage_daos_forwards_help_to_launcher():
    """`--storage daos --help` should print launch_aurora_daos.py's help."""
    result = _run("--storage", "daos", "--help", expect_exit=0)
    # argparse's prog defaults to the script name — should always be present.
    assert "launch_aurora_daos.py" in result.stdout


def test_storage_lustre_forwards_help_to_launcher():
    result = _run("--storage", "lustre", "--help", expect_exit=0)
    assert "launch_aurora.py" in result.stdout


def test_storage_webdataset_staged_forwards_help_to_launcher():
    result = _run("--storage", "webdataset-staged", "--help", expect_exit=0)
    assert "launch_aurora_web.py" in result.stdout


def test_storage_daos_dry_run_invokes_daos_launcher():
    """End-to-end: --storage daos --dry-run produces a daos launcher script."""
    result = _run(
        "--storage", "daos",
        "--id", "UNIFIED-TEST-DAOS",
        "--design", "PRISM-OLMO3-E2E-PROD",
        "--nodes", "1",
        "--dry-run",
        expect_exit=0,
    )
    # The DAOS launcher prints the script path
    assert "run_aurora_daos_" in result.stdout
    assert "UNIFIED-TEST-DAOS" in result.stdout


def test_storage_webdataset_staged_dry_run_invokes_web_launcher(tmp_path):
    # Both directories must be fixtured, not borrowed from the host. The
    # launcher gates on --webdataset-dir containing manifest.json + shards/,
    # and then again on --shared-hf-home existing; both gates sit upstream of
    # the --dry-run branch, so neither is skipped. Pointing these at real
    # cluster paths is what made this test pass only on Aurora (see #217).
    wds = tmp_path / "pixmo_cap_webdataset"
    (wds / "shards").mkdir(parents=True)
    (wds / "shards" / "000000.tar").touch()
    (wds / "manifest.json").write_text(
        json.dumps({"num_shards": 1, "total_written": 10})
    )
    shared_hf_home = tmp_path / "hf_hub"
    shared_hf_home.mkdir()

    result = _run(
        "--storage", "webdataset-staged",
        "--id", "UNIFIED-TEST-WEB",
        "--design", "PRISM-IMAGE-ONLY-1N",
        "--nodes", "1",
        "--webdataset-dir", str(wds),
        "--shared-hf-home", str(shared_hf_home),
        "--dry-run",
        expect_exit=0,
    )
    assert "run_aurora_web_" in result.stdout
    assert "UNIFIED-TEST-WEB" in result.stdout


def test_storage_lustre_dry_run_invokes_lustre_launcher():
    result = _run(
        "--storage", "lustre",
        "--id", "PRISM-IMAGE-ONLY-1N",  # generic launcher requires --id as exp ID lookup
        "--nodes", "1",
        "--dry-run",
        expect_exit=0,
    )
    # The lustre launcher writes scripts to jobs/run_aurora_<id>_... — distinct
    # from daos (run_aurora_daos_) and web (run_aurora_web_). Positively
    # verify the script was generated, and that it's not one of the others.
    script_path = _extract_script_path(result.stdout)
    name = script_path.name
    assert name.startswith("run_aurora_"), name
    assert not name.startswith("run_aurora_daos_"), name
    assert not name.startswith("run_aurora_web_"), name
    assert "PRISM-IMAGE-ONLY-1N" in name


def test_remaining_flags_forwarded_verbatim():
    """Every flag after --storage should land in the chosen launcher's argv.

    Verified by reading the generated script and grepping for the value that
    only appears if `--nodes 2` was honored (daos launcher writes
    `export NUM_NODES={args.nodes}` into the script).
    """
    result = _run(
        "--storage", "daos",
        "--id", "FORWARDED-FLAGS",
        "--design", "PRISM-OLMO3-E2E-PROD",
        "--nodes", "2",
        "--dry-run",
        expect_exit=0,
    )
    script_path = _extract_script_path(result.stdout)
    script_text = script_path.read_text()
    assert "export NUM_NODES=2" in script_text, (
        "expected `export NUM_NODES=2` in generated script — --nodes 2 was "
        f"not forwarded to the daos launcher. Script:\n{script_text[:2000]}"
    )
    # PBS select line should also reflect the node count when --batch is set;
    # here we're in interactive mode so check the env line only.
    assert "FORWARDED-FLAGS" in script_path.name

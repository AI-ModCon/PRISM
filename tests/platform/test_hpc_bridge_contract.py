import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.launcher]


def _run_cmd(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    return result


@pytest.mark.parametrize(
    "platform,expected_backend",
    [
        ("aurora", "xccl"),
        ("perlmutter", "nccl"),
    ],
)
def test_hpc_bridge_contract(tmp_path: Path, platform: str, expected_backend: str):
    repo_root = Path(__file__).resolve().parents[2]

    dispatch_path = tmp_path / f"dispatch_{platform}.json"
    result_path = tmp_path / f"result_{platform}.json"
    artifacts_dir = tmp_path / f"artifacts_{platform}"

    _run_cmd(
        [
            sys.executable,
            "tools/ci/dispatch_hpc_job.py",
            "--platform",
            platform,
            "--ci-mode",
            "--dry-run",
            "--output",
            str(dispatch_path),
        ],
        cwd=repo_root,
    )
    assert dispatch_path.exists()

    _run_cmd(
        [
            sys.executable,
            "tools/ci/poll_hpc_job.py",
            "--dispatch-file",
            str(dispatch_path),
            "--output",
            str(result_path),
        ],
        cwd=repo_root,
    )
    assert result_path.exists()

    _run_cmd(
        [
            sys.executable,
            "tools/ci/validate_hpc_result.py",
            "--result-file",
            str(result_path),
        ],
        cwd=repo_root,
    )

    _run_cmd(
        [
            sys.executable,
            "tools/ci/fetch_hpc_artifacts.py",
            "--result-file",
            str(result_path),
            "--artifacts-dir",
            str(artifacts_dir),
        ],
        cwd=repo_root,
    )

    payload = json.loads(result_path.read_text())
    assert payload["platform"] == platform
    assert payload["backend"] == expected_backend
    assert payload["status"] == "success"
    assert (artifacts_dir / "summary.log").exists()

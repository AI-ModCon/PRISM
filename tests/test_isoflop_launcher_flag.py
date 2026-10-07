"""Tests for `--target-flops` / `--calibration-json` on launch_aurora_daos.py.

We can't easily run the full launcher in a test (it expects a PBS env and
submits jobs), but we can exercise argparse + the validation block by
invoking the script with `--target-flops` plus inputs that trip each
guardrail. argparse exits 2 on `parser.error`, so the test pattern is
`expect_exit=2` for rejection paths and `expect_exit=0` for happy paths
that get past validation (the script then continues to the runtime
checks which need PBS — out of scope here).

For the happy-path computation, we check the prefix print line
"[isoflop] --target-flops=… / flops_per_step=… -> max_steps=N" rather
than running the whole script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "tools" / "launch_aurora_daos.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Set USER so the launcher doesn't bail at the early `$USER` guard.
    env.setdefault("USER", "ngetty")
    return subprocess.run(
        [sys.executable, str(LAUNCHER), "--id", "TEST-RUN", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=60,
    )


def test_target_flops_with_max_steps_rejected():
    result = _run("--target-flops", "1e18", "--max-steps", "100")
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def test_target_flops_requires_calibration_json():
    result = _run("--target-flops", "1e18")
    assert result.returncode == 2
    assert "requires --calibration-json" in result.stderr


def test_calibration_json_standalone_allowed(tmp_path: Path):
    """Standalone `--calibration-json` is now valid — it's the "env-var only"
    path that `tools/isoflop_launch.py` uses post-PR feedback (the plan owns
    `max_steps`; the launcher just propagates the cal handle so the trainer's
    `_FlopCounter` can attribute cumulative FLOPs).

    Old behavior (pre-feedback PR): `--calibration-json` alone errored with
    "requires --target-flops". We assert the new behavior accepts it.
    """
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"flops_per_step": 2.0e15}))
    result = _run("--calibration-json", str(cal))
    # Argparse must NOT reject with the old "requires --target-flops"
    # message. The launcher may still fail downstream (PBS env checks
    # outside argparse), but the calibration-json gate itself is open.
    assert "requires --target-flops" not in result.stderr, result.stderr


def test_calibration_json_missing_file(tmp_path: Path):
    missing = tmp_path / "no.json"
    result = _run("--target-flops", "1e18", "--calibration-json", str(missing))
    assert result.returncode == 2
    assert "not found" in result.stderr


def test_calibration_json_bad_value_rejected(tmp_path: Path):
    cal = tmp_path / "bad.json"
    cal.write_text(json.dumps({"flops_per_step": -1}))
    result = _run("--target-flops", "1e18", "--calibration-json", str(cal))
    assert result.returncode == 2
    assert "must be > 0" in result.stderr


def test_calibration_json_non_numeric_rejected(tmp_path: Path):
    cal = tmp_path / "str.json"
    cal.write_text(json.dumps({"flops_per_step": "not a number"}))
    result = _run("--target-flops", "1e18", "--calibration-json", str(cal))
    assert result.returncode == 2
    assert "not numeric" in result.stderr


def test_target_flops_computes_max_steps(tmp_path: Path):
    """1e18 / 2e15 = 500 steps. Happy path runs past validation."""
    cal = tmp_path / "good.json"
    cal.write_text(json.dumps({"flops_per_step": 2.0e15, "samples_per_sec": 10.0}))
    result = _run("--target-flops", "1e18", "--calibration-json", str(cal))
    # The launcher continues into runtime checks past validation, so we
    # don't expect exit 0 — we just need to see the isoflop print line.
    combined = result.stdout + result.stderr
    assert "[isoflop]" in combined
    assert "max_steps=500" in combined


def test_target_flops_negative_rejected():
    # Use `=` form so argparse doesn't interpret `-1e18` as a separate flag.
    result = _run("--target-flops=-1e18")
    assert result.returncode == 2
    assert "must be positive" in result.stderr


@pytest.mark.parametrize("flag", ["--target-flops", "--calibration-json"])
def test_flag_listed_in_help(flag: str):
    """Sanity: argparse exposes both flags in --help output."""
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
    )
    assert flag in result.stdout


@pytest.mark.parametrize("flag", ["--finite-webdataset", "--resume-from-checkpoint"])
def test_scaling_resume_flags_listed_in_help(flag: str):
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
    )
    assert flag in result.stdout


def test_resume_from_checkpoint_conflicts_with_weights_only():
    result = _run(
        "--resume-from-checkpoint", "/tmp/full",
        "--resume-weights-only", "/tmp/weights",
    )
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def test_finite_webdataset_and_resume_emit_script_controls():
    result = subprocess.run(
        [
            sys.executable, str(LAUNCHER),
            "--id", "TEST-FINITE-RESUME",
            "--design", "PRISM-IMAGE-ONLY-1N",
            "--nodes", "1",
            "--finite-webdataset",
            "--resume-from-checkpoint", "/tmp/prism_ckpt/step_500",
            "--dry-run",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr

    import re
    matches = re.findall(
        r"jobs/run_aurora_daos_TEST-FINITE-RESUME[^\s]*\.sh", result.stdout
    )
    assert matches, f"no qsub script path in stdout: {result.stdout[-500:]}"
    script_path = REPO_ROOT / matches[-1]
    script_body = script_path.read_text()
    assert "export WEBDATASET_RESAMPLED=0" in script_body
    assert "training.resume_from_checkpoint=/tmp/prism_ckpt/step_500" in script_body
    script_path.unlink(missing_ok=True)


def test_target_flops_dedupes_max_steps_override(tmp_path: Path):
    """`--target-flops` injection should evict any existing `training.max_steps=`.

    The PRISM-IMAGE-ONLY-1N design ships `training.max_steps=5000`. With
    `--target-flops`, the launcher computes a different max_steps and
    injects it; without dedupe, BOTH end up on the cmd line. Hydra
    last-wins still works, but the cmd is messy. This test asserts the
    generated qsub script contains exactly one `training.max_steps=`.
    """
    cal = tmp_path / "good.json"
    cal.write_text(json.dumps({"flops_per_step": 2.0e15, "samples_per_sec": 10.0}))
    result = subprocess.run(
        [
            sys.executable, str(LAUNCHER),
            "--id", "TEST-DEDUPE-1B",
            "--design", "PRISM-IMAGE-ONLY-1N",
            "--nodes", "1",
            "--target-flops", "1e18",
            "--calibration-json", str(cal),
            "--dry-run",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    # The launcher's INFO line should announce the replacement.
    assert "launcher override 'training.max_steps=500'" in result.stdout
    # The generated qsub script should contain exactly one max_steps=.
    import re
    matches = re.findall(r"jobs/run_aurora_daos_TEST-DEDUPE-1B[^\s]*\.sh", result.stdout)
    assert matches, f"no qsub script path in stdout: {result.stdout[-500:]}"
    script_path = REPO_ROOT / matches[-1]
    script_body = script_path.read_text()
    max_step_lines = re.findall(r"training\.max_steps=\d+", script_body)
    assert len(max_step_lines) == 1, (
        f"expected exactly one training.max_steps= override, got {max_step_lines}"
    )
    assert max_step_lines[0] == "training.max_steps=500"
    # Cleanup
    script_path.unlink(missing_ok=True)


def test_max_steps_cli_arg_also_dedupes(tmp_path: Path):
    """The dedupe helper applies to plain `--max-steps` too, not just `--target-flops`."""
    result = subprocess.run(
        [
            sys.executable, str(LAUNCHER),
            "--id", "TEST-DEDUPE-PLAIN",
            "--design", "PRISM-IMAGE-ONLY-1N",
            "--nodes", "1",
            "--max-steps", "42",
            "--dry-run",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    import re
    matches = re.findall(r"jobs/run_aurora_daos_TEST-DEDUPE-PLAIN[^\s]*\.sh", result.stdout)
    assert matches
    script_path = REPO_ROOT / matches[-1]
    script_body = script_path.read_text()
    max_step_lines = re.findall(r"training\.max_steps=\d+", script_body)
    assert len(max_step_lines) == 1
    assert max_step_lines[0] == "training.max_steps=42"
    script_path.unlink(missing_ok=True)


def test_hf_token_not_materialized_in_daos_dry_run(monkeypatch):
    """Generated DAOS job scripts must reference HF_TOKEN by name.

    The submit environment can forward the variable, but the script itself
    must not contain the literal secret.
    """
    import re

    secret = "hf_TEST_SECRET_DO_NOT_WRITE"
    monkeypatch.setenv("HF_TOKEN", secret)

    result = subprocess.run(
        [
            sys.executable, str(LAUNCHER),
            "--id", "TEST-HF-TOKEN-REFERENCE",
            "--design", "PRISM-IMAGE-ONLY-1N",
            "--nodes", "1",
            "--max-steps", "1",
            "--dry-run",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert secret not in combined

    matches = re.findall(r"jobs/run_aurora_daos_TEST-HF-TOKEN-REFERENCE[^\s]*\.sh", result.stdout)
    assert matches, f"no qsub script path in stdout: {result.stdout[-500:]}"
    script_path = REPO_ROOT / matches[-1]
    script_body = script_path.read_text()
    assert secret not in script_body
    assert 'export HF_TOKEN="$HF_TOKEN"' in script_body
    script_path.unlink(missing_ok=True)


def test_use_shared_venv_accepts_exported_venv_path(monkeypatch):
    import re

    venv_path = "/flare/ModCon/sww/prism-envs/qwen3-siglip-py3.12"
    monkeypatch.setenv("VENV_PATH", venv_path)

    result = subprocess.run(
        [
            sys.executable, str(LAUNCHER),
            "--id", "TEST-SHARED-VENV-ENV",
            "--design", "PRISM-IMAGE-ONLY-1N",
            "--nodes", "1",
            "--max-steps", "1",
            "--use-shared-venv",
            "--dry-run",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr

    matches = re.findall(r"jobs/run_aurora_daos_TEST-SHARED-VENV-ENV[^\s]*\.sh", result.stdout)
    assert matches, f"no qsub script path in stdout: {result.stdout[-500:]}"
    script_path = REPO_ROOT / matches[-1]
    script_body = script_path.read_text()
    assert f'source "{venv_path}/bin/activate"' in script_body
    assert 'export HF_HOME="/tmp/huggingface"' in script_body
    assert 'export HF_HUB_CACHE="/tmp/huggingface/hub"' in script_body
    script_path.unlink(missing_ok=True)


def test_runtime_flops_per_step_exports_env(tmp_path: Path):
    """`--runtime-flops-per-step` must surface as `export RUNTIME_FLOPS_PER_STEP=`
    in the generated qsub script so the trainer's _FlopCounter picks it up.

    Closes the cal/runtime FLOP undercount described in PR #98 review.
    """
    import re
    result = subprocess.run(
        [
            sys.executable, str(LAUNCHER),
            "--id", "TEST-RFPS-EXPORT",
            "--design", "PRISM-IMAGE-ONLY-1N",
            "--nodes", "1",
            "--max-steps", "10",
            "--runtime-flops-per-step", "9.6e16",
            "--dry-run",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    matches = re.findall(r"jobs/run_aurora_daos_TEST-RFPS-EXPORT[^\s]*\.sh", result.stdout)
    assert matches, f"no qsub script path in stdout: {result.stdout[-500:]}"
    script_path = REPO_ROOT / matches[-1]
    script_body = script_path.read_text()
    assert 'export RUNTIME_FLOPS_PER_STEP="9.600000e+16"' in script_body, (
        "qsub script must export RUNTIME_FLOPS_PER_STEP for the trainer's "
        "_FlopCounter to pick it up"
    )
    script_path.unlink(missing_ok=True)


def test_runtime_flops_per_step_negative_rejected():
    """Validation: --runtime-flops-per-step must be positive."""
    result = _run("--runtime-flops-per-step=-1e15")
    assert result.returncode == 2
    assert "must be positive" in result.stderr


def test_target_flops_emits_rescale_warning(tmp_path: Path):
    """When --target-flops is used directly (not via isoflop_launch.py),
    print a loud warning that no runtime rescale is applied. The warning
    is what stops a user from silently launching a job at 1/rescale_factor
    of their intended budget.
    """
    cal = tmp_path / "good.json"
    cal.write_text(json.dumps({"flops_per_step": 2.0e15, "samples_per_sec": 10.0}))
    result = _run("--target-flops", "1e18", "--calibration-json", str(cal))
    combined = result.stdout + result.stderr
    assert "WARNING: --target-flops uses RAW cal_fps" in combined

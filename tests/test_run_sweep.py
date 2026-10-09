"""Tests for tools/run_sweep.py."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP = REPO_ROOT / "tools" / "run_sweep.py"


def _run(*args: str, expect_exit: int = 0) -> str:
    result = subprocess.run(
        [sys.executable, str(SWEEP), *args],
        capture_output=True,
        text=True,
    )
    if expect_exit is not None:
        assert result.returncode == expect_exit, (
            f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result.stdout


def test_print_emits_one_cmd_per_combination():
    out = _run(
        "--preset", "text_only,text_image",
        "--designs", "PRISM-IMAGE-ONLY-2N,PRISM-OLMO3-E2E-PROD",
        "--print",
    )
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    assert len(lines) == 4  # 2 presets × 2 designs


def test_emitted_cmd_has_modalities_override():
    out = _run("--preset", "text_image", "--designs", "PRISM-IMAGE-ONLY-2N", "--print")
    assert "model.modalities=[text,image]" in out


def test_sweep_id_propagates_to_exp_field():
    out = _run(
        "--preset", "text_only",
        "--designs", "PRISM-IMAGE-ONLY-2N",
        "--sweep-id", "test-sweep-001",
        "--print",
    )
    assert "exp.sweep_id=test-sweep-001" in out


def test_preset_label_propagates():
    out = _run("--preset", "all6", "--designs", "PRISM-IMAGE-ONLY-2N", "--print")
    assert "exp.preset=all6" in out


def test_dry_run_flag_propagates_to_launcher():
    out = _run("--preset", "text_image", "--designs", "PRISM-IMAGE-ONLY-2N", "--print", "--dry-run")
    assert "--dry-run" in out


def test_unknown_preset_exits_nonzero():
    _run("--preset", "no_such_preset", "--designs", "X", "--print", expect_exit=2)


def test_storage_selects_launcher():
    out = _run(
        "--preset", "text_image",
        "--designs", "PRISM-IMAGE-ONLY-2N",
        "--storage", "lustre",
        "--print",
    )
    assert "launch_aurora.py" in out
    assert "launch_aurora_daos.py" not in out


def test_extra_args_passed_through():
    out = _run(
        "--preset", "text_image",
        "--designs", "PRISM-IMAGE-ONLY-2N",
        "--print",
        "--",
        "--nodes", "2",
        "--no-pil4dfs",
    )
    assert "--nodes" in out
    assert "--no-pil4dfs" in out

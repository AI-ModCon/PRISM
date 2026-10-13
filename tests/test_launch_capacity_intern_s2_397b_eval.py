import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "applications" / "timeseries" / "launch_capacity_intern_s2_397b_eval.sh"


def _post_training_hook():
    launcher = REPO_ROOT / "applications" / "timeseries" / "launch_capacity_intern_s2_397b.sh"
    content = launcher.read_text()
    return content.split('cat >> "$RUN_SCRIPT" <<EOF\n', 1)[1].split("\nEOF", 1)[0]


def test_post_training_evaluation_initializes_worker_modules():
    hook = _post_training_hook()
    assert "mpiexec -n 1 -ppn 1 bash -lc '" in hook
    assert "\n".join([
        "module purge || exit",
        "module use /opt/aurora/26.26.0/modulefiles || exit",
        "module load oneapi/release/2025.3.1 || exit",
        "module load mpich/prd/5.0.0.aurora_test.51a9474 || exit",
        "module load frameworks/2025.3.1 || exit",
    ]) in hook
    assert "module load frameworks/2025.3.1 || exit" in hook
    assert hook.index("module load frameworks/2025.3.1") < hook.index("bin/activate")


@pytest.mark.parametrize("failed_module_command", [
    "purge", "use /opt/aurora/26.26.0/modulefiles",
    "load oneapi/release/2025.3.1",
    "load mpich/prd/5.0.0.aurora_test.51a9474",
    "load frameworks/2025.3.1",
])
def test_post_training_worker_stops_when_framework_load_fails(tmp_path, failed_module_command):
    checkpoint = tmp_path / "checkpoints" / "step_50" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    hook = _post_training_hook().replace("\\$", "$").replace(
        "$TRAIN_OUTPUT_DIR", str(tmp_path)
    )
    setup = """
module() {
    if [[ "$*" == "$FAILED_MODULE_COMMAND" ]]; then return 37; fi
    return 0
}
python() { echo UNEXPECTED_EVALUATION; }
mpiexec() {
    [[ "$1 $2 $3 $4 $5 $6" == '-n 1 -ppn 1 bash -lc' ]] || return 99
    /bin/bash -c "$7"
}
export -f module python mpiexec
true
"""
    result = subprocess.run(
        ["bash", "-c", setup + hook],
        env={**os.environ, "BASH_ENV": "/dev/null", "FAILED_MODULE_COMMAND": failed_module_command},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 37, result.stderr
    assert "UNEXPECTED_EVALUATION" not in result.stdout


def test_dry_run_generates_scits_evaluation_job(tmp_path):
    train_output_dir = tmp_path / "train-output"
    checkpoint = train_output_dir / "checkpoints" / "step_12" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            str(train_output_dir),
            "--queue",
            "debug",
            "--nodes",
            "2",
            "--walltime",
            "00:15:00",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PRISM_VAL_SHARDS_DIR": str(tmp_path / "forecast-val")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_script = Path(result.stdout.strip().splitlines()[-1].removeprefix("Evaluation run script (not submitted): "))
    try:
        content = run_script.read_text()
    finally:
        run_script.unlink(missing_ok=True)

    assert "#PBS -l select=2" in content
    assert "#PBS -l walltime=00:15:00" in content
    assert "#PBS -q debug" in content
    assert "mpiexec -n 1 -ppn 1 bash -lc '" in content
    assert "module use /soft/modulefiles || exit" in content
    assert "module load frameworks/2025.3.1 || exit" in content
    assert content.index("module load hdf5") < content.index("set -u")
    assert f'export PRISM_VAL_SHARDS_DIR="{tmp_path / "forecast-val"}"' in content
    assert str(checkpoint) in content
    assert "--mode verify_timeseries_scits" in content
    assert "--validation" in content
    assert "--limit 0" in content
    assert "--backbone Qwen/Qwen3-0.6B" in content
    assert "export HF_HOME=/flare/ModCon/pemami" in content
    assert "export HF_HUB_CACHE=/flare/ModCon/pemami/hub" in content
    assert "export HF_HUB_OFFLINE=1" in content
    assert "export TRANSFORMERS_OFFLINE=1" in content


def test_rejects_missing_training_output_directory(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER), str(tmp_path / "missing"), "--dry-run"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Training output directory does not exist" in result.stderr
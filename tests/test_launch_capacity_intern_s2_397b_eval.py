import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "tools" / "launch_capacity_intern_s2_397b_eval.sh"


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
    assert content.index("module load hdf5") < content.index("set -u")
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
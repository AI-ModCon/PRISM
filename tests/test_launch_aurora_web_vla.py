"""Aurora Web launcher tests.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.launcher, pytest.mark.aurora]
pytest.importorskip("yaml")


def _write_design_file(path: Path, task_name: str):
    path.write_text(
        "\n".join(
            [
                "experiments:",
                "  - id: TEST-DESIGN",
                '    name: "test design"',
                "    resources:",
                "      ngpus: 1",
                '      walltime: "00:10:00"',
                "    common_overrides:",
                f'      training.task: "{task_name}"',
                '      model.backbone_id: "allenai/OLMo-1B-0724-hf"',
            ]
        )
    )


def _write_webdataset_root(path: Path):
    (path / "shards").mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"num_shards": 1, "total_written": 1}))


def _write_lustre_dataset_config(path: Path, root: Path):
    path.write_text(
        "\n".join(
            [
                "dastr:",
                f"  mount_base: {root}",
                "groups:",
                "  pixmo:",
                "    datasets:",
                "      pixmo_cap:",
                "        path: pixmo_cap_webdataset",
                "        samples: 10",
                "        shards: 1",
                "        weight: 1.0",
            ]
        )
    )


def _run_launcher(tmp_path: Path, task_name: str, extra_args=(), extra_env=None):
    design_file = tmp_path / "design.yaml"
    webdataset_root = tmp_path / "webdataset"
    shared_hf_home = tmp_path / "hf_hub"
    shared_hf_home.mkdir()
    _write_design_file(design_file, task_name)
    _write_webdataset_root(webdataset_root)

    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "tools" / "launch_aurora_web.py"
    cmd = [
        sys.executable,
        str(launcher),
        "--file",
        str(design_file),
        "--design",
        "TEST-DESIGN",
        "--id",
        "CODEX_TEST_LAUNCH",
        "--webdataset-dir",
        str(webdataset_root),
        "--shared-hf-home",
        str(shared_hf_home),
        "--dry-run",
        *extra_args,
    ]
    env = None
    if extra_env:
        import os

        env = os.environ.copy()
        env.update(extra_env)

    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    return result


def _run_launcher_ok(tmp_path: Path, task_name: str, extra_args=()):
    result = _run_launcher(tmp_path, task_name, extra_args=extra_args)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"

    generated_line = None
    for line in result.stdout.splitlines():
        if line.startswith("Generated Run Script: "):
            generated_line = line
            break
    assert generated_line is not None, f"Missing generated script line in:\n{result.stdout}"

    rel_script_path = generated_line.split(": ", 1)[1].strip()
    script_text = (tmp_path / rel_script_path).read_text()
    return result.stdout, script_text


def test_vla_launch_forces_accelerate(tmp_path):
    stdout, script_text = _run_launcher_ok(tmp_path, "vla_calvin")
    assert "forcing Accelerate mode" in stdout
    assert "# Using Accelerate" in script_text
    assert "export USE_NATIVE_DDP=1" not in script_text
    assert "${{" not in script_text


def test_tarball_mode_stages_standalone_intern_s2_artifacts(tmp_path):
    shared_hf_root = tmp_path / "huggingface"
    shared_hf_hub = shared_hf_root / "hub"
    shared_hf_hub.mkdir(parents=True)

    _, script_text = _run_launcher_ok(
        tmp_path,
        "vlm",
        extra_args=(
            "--hf-home",
            str(shared_hf_root),
            "--shared-hf-home",
            str(shared_hf_hub),
        ),
    )

    assert f'export SHARED_HF_ROOT="{shared_hf_root}"' in script_text
    assert 'export LOCAL_HF_ROOT="/tmp/huggingface"' in script_text
    assert '"intern-s2-preview-timeseries"' in script_text
    assert '"intern-s2-preview-397b-timeseries"' in script_text
    assert 'ln -sfn "$SHARED_HF_ROOT/$ARTIFACT" "$LOCAL_HF_ROOT/$ARTIFACT"' in script_text


def test_intern_s2_design_emits_transformers_preflight(tmp_path):
    design_file = tmp_path / "design.yaml"
    webdataset_root = tmp_path / "webdataset"
    shared_hf_home = tmp_path / "hf_hub"
    shared_hf_home.mkdir()
    _write_webdataset_root(webdataset_root)
    design_file.write_text(
        "\n".join(
            [
                "experiments:",
                "  - id: TEST-INTERN-S2-DESIGN",
                '    name: "test intern s2 design"',
                "    resources:",
                "      ngpus: 1",
                '      walltime: "00:10:00"',
                "    common_overrides:",
                '      training.task: "vlm"',
                "      model: prism_qwen3_0_6b_intern_s2_397b_ts",
            ]
        )
    )

    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "tools" / "launch_aurora_web.py"
    result = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--file", str(design_file),
            "--design", "TEST-INTERN-S2-DESIGN",
            "--id", "CODEX_TEST_INTERN_S2",
            "--webdataset-dir", str(webdataset_root),
            "--webdataset-modality", "time_series",
            "--shared-hf-home", str(shared_hf_home),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    generated_line = next(
        line for line in result.stdout.splitlines()
        if line.startswith("Generated Run Script: ")
    )
    script_text = (tmp_path / generated_line.split(": ", 1)[1].strip()).read_text()

    assert "Intern-S2 transformers preflight" in script_text
    assert "transformers.configuration_utils" in script_text
    assert "Use --use-shared-venv" in script_text
    syntax_check = subprocess.run(
        ["bash", "-n", str(tmp_path / generated_line.split(": ", 1)[1].strip())],
        check=False,
        text=True,
        capture_output=True,
    )
    assert syntax_check.returncode == 0, syntax_check.stderr


def test_non_vla_launch_defaults_to_native_ddp(tmp_path):
    stdout, script_text = _run_launcher_ok(tmp_path, "vlm")
    assert "forcing Accelerate mode" not in stdout
    assert "export USE_NATIVE_DDP=1" in script_text
    assert "${{" not in script_text


def test_max_steps_emitted_as_hydra_override(tmp_path):
    _, script_text = _run_launcher_ok(tmp_path, "vlm", extra_args=("--max-steps", "50"))
    assert "training.max_steps=50" in script_text


def test_max_steps_rejects_zero(tmp_path):
    result = _run_launcher(tmp_path, "vlm", extra_args=("--max-steps", "0"))
    assert result.returncode != 0
    assert "--max-steps must be positive" in result.stderr


def test_calibration_json_exports_env_var(tmp_path):
    """`--calibration-json <path>` must export CALIBRATION_JSON in the qsub script.
    The trainer's _FlopCounter reads it to attribute cumulative FLOPs in
    perf.jsonl. Mirrors launch_aurora_daos.py's env-var-only path.
    Stage A IsoFLOP uses launch_aurora_web.py (Lustre + webdataset).
    """
    cal_json = tmp_path / "cal.json"
    cal_json.write_text(json.dumps({
        "flops_per_step": 2.0e15, "samples_per_sec": 10.0,
        "mean_seq_len": 2048, "batch_size": 8, "seq_len": 2048,
    }))
    _, script_text = _run_launcher_ok(
        tmp_path, "vlm",
        extra_args=("--calibration-json", str(cal_json)),
    )
    assert f'export CALIBRATION_JSON="{cal_json}"' in script_text, script_text


def test_runtime_flops_per_step_exports_env_var(tmp_path):
    """`--runtime-flops-per-step <fps>` must export RUNTIME_FLOPS_PER_STEP so
    cumulative_flops in perf.jsonl matches the plan's `budget_flops` instead
    of undercounting by `rescale_factor`. See PR feedback on PR #98.
    """
    _, script_text = _run_launcher_ok(
        tmp_path, "vlm",
        extra_args=("--runtime-flops-per-step", "9.6e16"),
    )
    assert 'export RUNTIME_FLOPS_PER_STEP="9.600000e+16"' in script_text, script_text


def test_calibration_json_missing_is_fatal(tmp_path):
    """Pointing at a missing file must error at submit time, not surface
    silently when the trainer tries to read CALIBRATION_JSON on the compute node.
    """
    result = _run_launcher(
        tmp_path, "vlm",
        extra_args=("--calibration-json", str(tmp_path / "nope.json")),
    )
    assert result.returncode != 0
    assert "--calibration-json not found" in result.stderr


def test_runtime_flops_per_step_rejects_nonpositive(tmp_path):
    result = _run_launcher(
        tmp_path, "vlm",
        extra_args=("--runtime-flops-per-step", "0"),
    )
    assert result.returncode != 0
    assert "--runtime-flops-per-step must be positive" in result.stderr


def test_resume_from_checkpoint_emits_hydra_override(tmp_path):
    _, script_text = _run_launcher_ok(
        tmp_path,
        "vlm",
        extra_args=("--resume-from-checkpoint", "/tmp/prism_ckpt/step_500"),
    )
    assert "training.resume_from_checkpoint=/tmp/prism_ckpt/step_500" in script_text


def test_resume_from_checkpoint_conflicts_with_weights_only(tmp_path):
    result = _run_launcher(
        tmp_path,
        "vlm",
        extra_args=(
            "--resume-from-checkpoint", "/tmp/full",
            "--resume-weights-only", "/tmp/weights",
        ),
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_missing_shared_hf_home_is_fatal(tmp_path):
    """Regression guard: previously printed a warning and proceeded, which
    silently wasted the queue allocation when model staging failed mid-job."""
    design_file = tmp_path / "design.yaml"
    webdataset_root = tmp_path / "webdataset"
    _write_design_file(design_file, "vlm")
    _write_webdataset_root(webdataset_root)

    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "tools" / "launch_aurora_web.py"
    cmd = [
        sys.executable,
        str(launcher),
        "--file",
        str(design_file),
        "--design",
        "TEST-DESIGN",
        "--id",
        "CODEX_TEST_LAUNCH",
        "--webdataset-dir",
        str(webdataset_root),
        "--shared-hf-home",
        str(tmp_path / "does_not_exist"),
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=tmp_path, check=False, text=True, capture_output=True)
    assert result.returncode != 0
    assert "does not exist" in result.stderr


def test_lustre_multi_dataset_mode_emits_no_daos_filesystem(tmp_path):
    design_file = tmp_path / "design.yaml"
    shared_hf_home = tmp_path / "hf_hub"
    dataset_root = tmp_path / "lustre_data"
    dataset_config = tmp_path / "lustre_datasets.yaml"
    shared_hf_home.mkdir()
    (dataset_root / "pixmo_cap_webdataset").mkdir(parents=True)
    _write_design_file(design_file, "vlm")
    _write_lustre_dataset_config(dataset_config, dataset_root)

    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "tools" / "launch_aurora_web.py"
    result = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--file", str(design_file),
            "--design", "TEST-DESIGN",
            "--id", "CODEX_TEST_LUSTRE_MULTI",
            "--dataset-groups", "pixmo",
            "--dataset-config", str(dataset_config),
            "--shared-hf-home", str(shared_hf_home),
            "--finite-webdataset",
            "--batch",
            "--dry-run",
        ],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr

    generated_line = next(
        line for line in result.stdout.splitlines()
        if line.startswith("Generated Run Script: ")
    )
    script_text = (tmp_path / generated_line.split(": ", 1)[1].strip()).read_text()
    assert "#PBS -l filesystems=home:flare" in script_text
    assert "daos_user_fs" not in script_text
    assert "export USE_MULTI_DATASET=1" in script_text
    assert f'export DATASET_ROOT="{dataset_root}"' in script_text
    assert 'export DATASET_GROUPS="pixmo"' in script_text
    assert f'export DATASET_CONFIG="{dataset_config}"' in script_text
    assert "export WEBDATASET_RESAMPLED=0" in script_text


def test_web_launcher_reads_hf_token_from_env_file_without_literal(tmp_path):
    # The generated script must never carry the token's literal value, because
    # jobs/*.sh lands on a group-readable filesystem.
    #
    # It also must not use the old `export HF_TOKEN="$HF_TOKEN"` form: qsub runs
    # without -V and the script carries no `#PBS -V`, so PBS Pro never
    # propagates the submitting shell's environment and "$HF_TOKEN" expanded to
    # the empty string on the node. The launcher reads the key out of .env at
    # runtime instead (tools/launch_aurora_web.py:545-567).
    result = _run_launcher(
        tmp_path,
        "vlm",
        extra_env={"HF_TOKEN": "hf_should_not_be_written"},
    )
    assert result.returncode == 0, result.stderr
    generated_line = next(
        line for line in result.stdout.splitlines()
        if line.startswith("Generated Run Script: ")
    )
    script_text = (tmp_path / generated_line.split(": ", 1)[1].strip()).read_text()
    assert "hf_should_not_be_written" not in script_text
    assert 'export HF_TOKEN="$HF_TOKEN"' not in script_text
    # The .env-reading block, which HF_TOKEN is one of the two keys of.
    assert "PRISM_ENV_FILE=" in script_text
    assert "for _k in HF_TOKEN WANDB_API_KEY; do" in script_text
    assert 'export "${_k}=$_v"' in script_text


def test_web_shared_venv_uses_environment_venv_path(tmp_path):
    result = _run_launcher(
        tmp_path,
        "vlm",
        extra_args=("--use-shared-venv",),
        extra_env={"VENV_PATH": "/flare/ModCon/sww/prism-envs/qwen3-siglip-py3.12"},
    )
    assert result.returncode == 0, result.stderr
    generated_line = next(
        line for line in result.stdout.splitlines()
        if line.startswith("Generated Run Script: ")
    )
    script_text = (tmp_path / generated_line.split(": ", 1)[1].strip()).read_text()
    assert 'source "/flare/ModCon/sww/prism-envs/qwen3-siglip-py3.12/bin/activate"' in script_text

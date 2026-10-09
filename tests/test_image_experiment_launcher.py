import argparse
import json
import subprocess
from pathlib import Path

import pytest
from tools.launch_aurora_image_experiment import render_job


def arguments(tmp_path, entrypoint="overfit_prism_image_connector.py"):
    command = tmp_path / "command.json"
    command.write_text(
        json.dumps(
            [
                entrypoint,
                "--output-dir",
                "/tmp/run with spaces",
                "--device",
                "xpu",
                "--manifest",
                "/tmp/prompt$(bad).jsonl",
            ]
        )
    )
    return argparse.Namespace(
        project="ModCon",
        queue="debug",
        name="overfit",
        minutes=40,
        repo=Path("/tmp/repo with spaces"),
        venv=Path("/tmp/venv"),
        upstream=Path("/tmp/upstream"),
        job_dir=tmp_path,
        command_file=command,
    )


def test_explicit_single_tile_command_and_shell_quoting(tmp_path):
    pbs, worker, argv, output = render_job(arguments(tmp_path))
    assert "walltime=00:40:00" in pbs
    assert "mpiexec -n 1 --ppn 1" in pbs
    assert "ZE_AFFINITY_MASK=0" in worker
    assert "'/tmp/prompt$(bad).jsonl'" in worker
    assert str(output) == "/tmp/run with spaces"
    for name, body in (("job.pbs", pbs), ("worker.sh", worker)):
        file = tmp_path / name
        file.write_text(body)
        subprocess.run(["bash", "-n", str(file)], check=True)


@pytest.mark.parametrize(
    "field,value",
    [("minutes", 61), ("minutes", 4), ("queue", "prod"), ("name", "bad\n#PBS -l select=8")],
)
def test_rejects_unbounded_resources(tmp_path, field, value):
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        render_job(args)


def test_rejects_arbitrary_entrypoint(tmp_path):
    with pytest.raises(ValueError, match="entry points"):
        render_job(arguments(tmp_path, "train.py"))


def test_dense_diffusion_pilot_allocates_cpu_optimizer_threads(tmp_path):
    pbs, worker, _, _ = render_job(arguments(tmp_path, "train_prism_image_diffusion.py"))
    assert "mpiexec -n 1 --ppn 1" in pbs
    assert "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32" in worker
    assert "--cpu-bind depth --depth 32" in pbs
    assert "ZE_AFFINITY_MASK=0" in worker


def test_long_dense_training_requires_capacity_and_keeps_single_tile(tmp_path):
    args = arguments(tmp_path, "train_prism_image_diffusion.py")
    args.queue = "capacity"
    args.minutes = 720
    pbs, worker, _, _ = render_job(args)
    assert "walltime=12:00:00" in pbs
    assert "#PBS -q capacity" in pbs
    assert "mpiexec -n 1 --ppn 1" in pbs
    assert "ZE_AFFINITY_MASK=0" in worker
    args.queue = "debug"
    with pytest.raises(ValueError, match="capacity"):
        render_job(args)
    args.queue = "capacity"
    args.minutes = 1441
    with pytest.raises(ValueError):
        render_job(args)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "diagnose_prism_image_conditioning.py",
        "diagnose_prism_joint_components.py",
        "align_prism_image_conditioning.py",
    ],
)
def test_conditioning_audit_uses_bounded_diagnostic_allocation(tmp_path, entrypoint):
    args = arguments(tmp_path, entrypoint)
    pbs, worker, _, _ = render_job(args)
    assert entrypoint in worker
    assert "mpiexec -n 1 --ppn 1" in pbs
    args.queue = "capacity"
    args.minutes = 61
    with pytest.raises(ValueError, match="dense diffusion"):
        render_job(args)


def with_pre_command(
    args,
    tmp_path,
    *,
    entrypoint="diagnose_image_decoder_repeatability.py",
    output="/tmp/native rerun",
):
    path = tmp_path / "pre-command.json"
    path.write_text(
        json.dumps(
            [
                entrypoint,
                "--output-dir",
                output,
                "--device",
                "xpu",
                "--prior-run",
                "/tmp/previous run",
            ]
        )
    )
    args.pre_command_file = path
    return args


def test_pre_command_uses_same_single_tile_allocation_and_precedes_training(tmp_path):
    args = with_pre_command(arguments(tmp_path), tmp_path)
    pbs, worker, command, output = render_job(args)
    assert command[0] == "overfit_prism_image_connector.py"
    assert str(output) == "/tmp/run with spaces"
    assert worker.index("diagnose_image_decoder_repeatability.py") < worker.index(
        "overfit_prism_image_connector.py"
    )
    assert "'/tmp/previous run'" in worker
    assert "set -eo pipefail" in worker
    assert pbs.count("mpiexec") == 1
    path = tmp_path / "worker.sh"
    path.write_text(worker)
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize(
    "entrypoint", ["overfit_prism_image_connector.py", "validate_prism_image_parent.py", "train.py"]
)
def test_pre_command_restricts_entrypoint(tmp_path, entrypoint):
    args = with_pre_command(arguments(tmp_path), tmp_path, entrypoint=entrypoint)
    with pytest.raises(ValueError):
        render_job(args)


def test_pre_command_rejects_reused_output_directory(tmp_path):
    args = with_pre_command(arguments(tmp_path), tmp_path, output="/tmp/run with spaces")
    with pytest.raises(ValueError, match="distinct output"):
        render_job(args)


@pytest.mark.parametrize(
    "command",
    [
        ["diagnose_image_decoder_repeatability.py", "--output-dir", "/tmp/new", "--device"],
        [
            "diagnose_image_decoder_repeatability.py",
            "--output-dir",
            "/tmp/new",
            "--device",
            "xpu",
            "--device",
            "xpu",
        ],
    ],
)
def test_pre_command_rejects_missing_or_ambiguous_option_values(tmp_path, command):
    args = with_pre_command(arguments(tmp_path), tmp_path)
    args.pre_command_file.write_text(json.dumps(command))
    with pytest.raises(ValueError):
        render_job(args)


def _submit_fixture(tmp_path):
    args = arguments(tmp_path)
    for name in ("repo", "venv", "upstream"):
        path = tmp_path / name
        path.mkdir()
        setattr(args, name, path)
    (args.repo / "tools").mkdir()
    for entrypoint in (
        "overfit_prism_image_connector.py",
        "diagnose_image_decoder_repeatability.py",
    ):
        (args.repo / "tools" / entrypoint).write_text("# bounded fixture\n")
    args.job_dir = tmp_path / "job"
    prior = tmp_path / "prior"
    prior.mkdir()
    output = tmp_path / "new-main"
    pre_output = tmp_path / "new-pre"
    args.command_file.write_text(
        json.dumps(
            ["overfit_prism_image_connector.py", "--output-dir", str(output), "--device", "xpu"]
        )
    )
    with_pre_command(args, tmp_path, output=str(pre_output))
    args.pre_command_file.write_text(
        json.dumps(
            [
                "diagnose_image_decoder_repeatability.py",
                "--output-dir",
                str(pre_output),
                "--device",
                "xpu",
                "--prior-run",
                str(prior),
            ]
        )
    )
    cli = [
        item
        for key in ("repo", "venv", "upstream", "job_dir", "command_file", "pre_command_file")
        for item in ("--" + key.replace("_", "-"), str(getattr(args, key)))
    ]
    return args, cli, output, pre_output


def test_submit_records_both_command_and_entrypoint_hashes(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from tools.launch_aurora_image_experiment import main

    args, cli, output, pre_output = _submit_fixture(tmp_path)
    original = subprocess.run
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "qsub":
            return SimpleNamespace(stdout="12345.server\n")
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    assert main(cli + ["--submit"]) == 0
    launch = json.loads((args.job_dir / "launch.json").read_text())
    submitted = json.loads((args.job_dir / "submission.json").read_text())
    assert submitted.pop("job_id") == "12345.server"
    assert submitted == launch
    assert launch["pre_command"] == json.loads(args.pre_command_file.read_text())
    assert (
        launch["pre_command_file_sha256"]
        == hashlib.sha256(args.pre_command_file.read_bytes()).hexdigest()
    )
    assert (
        launch["pre_entrypoint_sha256"]
        == hashlib.sha256(
            (args.repo / "tools/diagnose_image_decoder_repeatability.py").read_bytes()
        ).hexdigest()
    )
    assert launch["output_dir"] == str(output)
    assert launch["pre_output_dir"] == str(pre_output)
    assert sum(call[0] == "qsub" for call in calls) == 1


@pytest.mark.parametrize("existing", ["main", "pre"])
def test_submit_checks_both_outputs_before_creating_job(tmp_path, existing):
    from tools.launch_aurora_image_experiment import main

    args, cli, output, pre_output = _submit_fixture(tmp_path)
    (output if existing == "main" else pre_output).mkdir()
    with pytest.raises(ValueError, match="Both output directories must be new"):
        main(cli + ["--submit"])
    assert not args.job_dir.exists()


@pytest.mark.parametrize("report_location", ["pre-command", "unrelated", "without-pre-command"])
def test_submit_allows_only_exact_report_created_by_pre_command(
    tmp_path, monkeypatch, report_location
):
    from types import SimpleNamespace

    from tools.launch_aurora_image_experiment import main

    args, cli, _, pre_output = _submit_fixture(tmp_path)
    report = pre_output / "manifest.json"
    if report_location == "unrelated":
        report = tmp_path / "uncreated-report.json"
    if report_location == "without-pre-command":
        index = cli.index("--pre-command-file")
        del cli[index : index + 2]
    command = json.loads(args.command_file.read_text())
    command.extend(["--repeatability-report", str(report)])
    args.command_file.write_text(json.dumps(command))
    original = subprocess.run
    submissions = []

    def run(command, **kwargs):
        if command[0] == "qsub":
            submissions.append(command)
            return SimpleNamespace(stdout="12345.server\n")
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    if report_location == "pre-command":
        assert main(cli + ["--submit"]) == 0
        assert len(submissions) == 1
        worker = (args.job_dir / "worker.sh").read_text()
        assert worker.index("require_repeatability") < worker.rindex("--repeatability-report")
    else:
        with pytest.raises(FileNotFoundError):
            main(cli + ["--submit"])
        assert not args.job_dir.exists()
        assert not submissions


def test_pre_command_execution_failure_prevents_training(tmp_path):
    args, _, _, _ = _submit_fixture(tmp_path)
    (args.venv / "bin").mkdir()
    (args.venv / "bin/activate").write_text("# fixture activation\n")
    marker = tmp_path / "main-ran"
    python = args.venv / "bin/python"
    python.write_text(
        '#!/bin/bash\ncase "$2" in\n*diagnose_image_decoder_repeatability.py) exit 17;;\n*overfit_prism_image_connector.py) touch "'
        + str(marker)
        + '";;\nesac\n'
    )
    python.chmod(0o755)
    _, worker, _, _ = render_job(args)
    # Supply the cluster's module command as a harmless shell function.
    worker = worker.replace("set -eo pipefail", "set -eo pipefail\nmodule() { return 0; }")
    path = tmp_path / "worker.sh"
    path.write_text(worker)
    result = subprocess.run(["bash", str(path)], capture_output=True)
    assert result.returncode == 17
    assert not marker.exists()


@pytest.mark.parametrize(
    "failed", ["native_repeat", "adapter_vs_native", "native_after_adapter", "prior_process"]
)
def test_numerical_difference_blocks_training(tmp_path, failed):
    from tools.launch_aurora_image_experiment import require_repeatability

    path = tmp_path / "manifest.json"
    checks = {
        key: {"passed": key != failed}
        for key in ("native_repeat", "adapter_vs_native", "native_after_adapter", "prior_process")
    }
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "evidence_kind": "real_checkpoint_diagnostic",
                "fixture": False,
                "comparisons": checks,
            }
        )
    )
    with pytest.raises(ValueError, match=failed):
        require_repeatability(path, require_prior=True)


def test_numerical_gate_ignores_explicit_cross_policy_reference_difference(tmp_path):
    from tools.launch_aurora_image_experiment import require_repeatability

    path = tmp_path / "manifest.json"
    checks = {
        key: {"passed": True}
        for key in ("native_repeat", "adapter_vs_native", "native_after_adapter", "prior_process")
    }
    checks["original_reference"] = {"passed": False}
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "evidence_kind": "real_checkpoint_diagnostic",
                "fixture": False,
                "comparisons": checks,
            }
        )
    )
    require_repeatability(path, require_prior=True)

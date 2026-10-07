import subprocess
from types import SimpleNamespace

import pytest
from tools.launch_aurora_image_smoke import main, render_job


def args(tmp_path):
    result = {
        name: tmp_path / (name + " with spaces")
        for name in ("repo", "venv", "checkpoint", "upstream", "cases", "output_dir", "job_dir")
    }
    return SimpleNamespace(
        **result, project="ModCon", queue="debug", name="image-smoke", minutes=10, steps=2
    )


def test_smoke_pbs_is_bounded_and_syntax_valid(tmp_path):
    pbs, worker = render_job(args(tmp_path))
    assert "select=1" in pbs and "walltime=00:10:00" in pbs
    assert "mpiexec -n 1 --ppn 1" in pbs
    assert "export ZE_FLAT_DEVICE_HIERARCHY=FLAT\nexport ZE_AFFINITY_MASK=0\n" in worker
    assert "export ONEAPI_DEVICE_SELECTOR=level_zero:gpu" in worker
    assert "Smoke requires exactly one visible XPU tile" in worker
    assert worker.index("xpu.device_count()") < worker.index("tools/validate_image_decoder.py")
    assert worker.index("module load frameworks/2025.3.1") < worker.index("/bin/activate")
    assert "--smoke" in worker and "--steps 2" in worker
    for name, body in (("job.pbs", pbs), ("worker.sh", worker)):
        path = tmp_path / name
        path.write_text(body)
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_excess_budget_and_header_injection_rejected(tmp_path):
    config = args(tmp_path)
    config.minutes = 60
    with pytest.raises(ValueError, match="budget"):
        render_job(config)
    config.minutes = 10
    config.project = "ModCon\n#PBS -l select=100"
    with pytest.raises(ValueError, match="PBS"):
        render_job(config)


def test_optional_parity_uses_same_reference_and_budget(tmp_path):
    config = args(tmp_path)
    config.with_parity = True
    pbs, worker = render_job(config)
    assert "--parity --smoke" in worker
    assert "--parity-mode full_pipeline --reference-dir" in worker
    assert "walltime=00:10:00" in pbs


def test_submit_does_not_allocate_when_preflight_fails(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ("repo", "venv", "upstream", "checkpoint", "cases")}
    for path in paths.values():
        path.mkdir()
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(subprocess, "run", fail)
    arguments = [
        "--submit",
        "--output-dir",
        str(tmp_path / "out"),
        "--job-dir",
        str(tmp_path / "job"),
    ]
    for name, path in paths.items():
        arguments += ["--" + name, str(path)]
    with pytest.raises(subprocess.CalledProcessError):
        main(arguments)
    assert len(calls) == 1 and "--preflight" in calls[0]


def test_repeatability_runs_two_fresh_processes_within_one_budget(tmp_path):
    config = args(tmp_path)
    config.repeatability_reference = tmp_path / "completed reference"
    pbs, worker = render_job(config)
    assert worker.count("tools/diagnose_image_decoder_repeatability.py") == 2
    assert "--prior-run" in worker and "--reference-dir" in worker
    assert "-fresh" in worker and "walltime=00:10:00" in pbs
    for name, body in (("diagnostic.pbs", pbs), ("diagnostic-worker.sh", worker)):
        path = tmp_path / name
        path.write_text(body)
        subprocess.run(["bash", "-n", str(path)], check=True)

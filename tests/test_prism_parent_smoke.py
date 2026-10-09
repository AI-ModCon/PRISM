import json
import subprocess
from types import SimpleNamespace

import pytest
from tools.launch_aurora_prism_image import render_job
from tools.smoke_prism_image_training import validate_budget


@pytest.mark.parametrize("nested", [False, True])
def test_metadata_preflight_reads_flat_and_nested_generator_paths(tmp_path, monkeypatch, nested):
    from tools import launch_aurora_prism_image as launcher

    paths = {}
    for name in ("repo", "venv", "upstream", "tokenizer", "source_processor", "generator"):
        paths[name] = tmp_path / name
        paths[name].mkdir()
    (paths["generator"] / "prism_checkpoint_provenance.json").write_text("{}")
    image = {"model_id": str(paths["generator"])}
    config = {
        "llm_backbone_id": str(paths["repo"]),
        "image_encoder_id": str(paths["repo"]),
        "decoder_configs": {
            "image": {"generator": {"type": "omnigen2", **image}} if nested else image
        },
    }
    paths["model_config"] = tmp_path / "model.json"
    paths["model_config"].write_text(json.dumps(config))
    for name in ("checkpoint", "cases"):
        paths[name] = tmp_path / name
        paths[name].write_text("fixture")
    paths["output_dir"] = tmp_path / "outputs"
    paths["job_dir"] = tmp_path / "job"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        assert command[0] in {"bash", "qsub"}
        return SimpleNamespace(stdout="fixture-job\n")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    argv = ["--mode", "connector-smoke", "--submit"]
    for name, path in paths.items():
        if name != "generator":
            argv.extend(["--" + name.replace("_", "-"), str(path)])
    assert launcher.main(argv) == 0
    assert [command[0] for command in commands] == ["bash", "bash", "qsub"]
    receipt = json.loads((paths["job_dir"] / "submission.json").read_text())
    assert receipt["job_id"] == "fixture-job"


@pytest.mark.parametrize("mode", ["parent", "connector-smoke"])
def test_parent_smoke_launcher_is_bounded_and_quoted(tmp_path, mode):
    paths = {
        name: tmp_path / (name + " with ' spaces")
        for name in (
            "repo",
            "venv",
            "upstream",
            "model_config",
            "checkpoint",
            "tokenizer",
            "source_processor",
            "cases",
            "output_dir",
            "job_dir",
        )
    }
    args = SimpleNamespace(
        **paths,
        project="ModCon",
        queue="debug",
        name="parent-smoke",
        minutes=10,
        steps=2,
        mode=mode,
    )
    pbs, worker = render_job(args)
    assert "select=1" in pbs and "walltime=00:10:00" in pbs
    assert "mpiexec -n 1 --ppn 1" in pbs
    assert "ZE_AFFINITY_MASK=0" in worker
    assert "--device xpu --dtype bfloat16" in worker
    assert "train_image_decoder.py" not in worker
    assert ("--cases" in worker) == (mode == "parent")
    for name, body in (("job.pbs", pbs), ("worker.sh", worker)):
        path = tmp_path / name
        path.write_text(body)
        subprocess.run(["bash", "-n", str(path)], check=True)
    args.minutes = 60
    with pytest.raises(ValueError, match="budget"):
        render_job(args)


@pytest.mark.parametrize(
    "values", [(5, 1, 256, 256, 2), (1, 5, 256, 256, 2), (1, 1, 512, 256, 2), (1, 1, 256, 256, 50)]
)
def test_optimization_diagnostic_cannot_expand_to_full_training(values):
    with pytest.raises(ValueError):
        validate_budget(*values)


def test_optimization_diagnostic_artifact_not_accepted_as_trained_connector(tmp_path):
    import torch
    from src.decoders.loading import load_image_connector
    from torch import nn

    checkpoint = tmp_path / "smoke.pt"
    torch.save(
        {"schema_version": 1, "evidence_kind": "real_checkpoint_optimization_smoke"}, checkpoint
    )
    with pytest.raises(ValueError, match="accepted training artifact"):
        load_image_connector(
            nn.Linear(1, 1),
            checkpoint,
            parent_checkpoint_sha256="a" * 64,
            reference_checkpoint_sha256="b" * 64,
        )

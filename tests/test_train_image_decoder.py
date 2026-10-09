"""CPU fixtures test the runner's control flow, never actual OmniGen2 readiness."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tools.train_image_decoder import (
    P1_ACCEPTANCE_CHECKS,
    frozen_state_hashes,
    main,
    run_connector_training,
    sha256_file,
    validate_training_evidence,
)
from torch import nn

pytestmark = pytest.mark.unit


class FixtureImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.connector = nn.Linear(3, 3)
        self.generator = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 1))
        self.generator.register_buffer("nonpersistent", torch.tensor(0.0), persistent=False)


class FixtureModel(nn.Module):
    def __init__(self, *, behavior="normal"):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.decoders = nn.ModuleDict({"image": FixtureImageDecoder()})
        self.behavior = behavior
        self.calls = []

    def forward_outputs(
        self, inputs, targets, requested_outputs, output_specs, native_context, decoder_kwargs
    ):
        self.calls.append(
            {
                "requested_outputs": requested_outputs,
                "generator_training": self.decoders["image"].generator.training,
                "connector_training": self.decoders["image"].connector.training,
                "native_context": native_context,
            }
        )
        decoder = self.decoders["image"]
        hidden = self.backbone(inputs["text"])
        prediction = decoder.generator(decoder.connector(hidden))
        loss = (prediction - targets["image"]).square().mean()
        if self.behavior == "detached":
            loss = loss.detach()
        elif self.behavior == "nan":
            loss = loss * float("nan")
        elif self.behavior == "mutate_buffer":
            decoder.generator.nonpersistent.add_(1)
        elif self.behavior == "zero_grad":
            loss = loss * 0
        return SimpleNamespace(predictions={"image": prediction}, losses={"image": loss}, loss=loss)


def _model(**kwargs):
    torch.manual_seed(7)
    return FixtureModel(**kwargs)


def _batches():
    return [
        {
            "inputs": {"text": torch.tensor([[1.0, 0.5, 0.3], [0.1, 1.2, -0.4]])},
            "targets": {"image": torch.tensor([[2.0], [-1.0]])},
            "native_context": {"image": {"source_ids": [["a"], ["b"]]}},
            "output_specs": {"image": {"height": 8, "width": 8}},
            "metadata": [{"id": "one"}, {"id": "two"}],
        }
    ]


def _run(model, output_dir, **kwargs):
    return run_connector_training(
        model,
        _batches(),
        output_dir=output_dir,
        max_steps=kwargs.pop("max_steps", 3),
        fixture=True,
        **kwargs,
    )


def _evidence(tmp_path):
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"local parent checkpoint fixture")
    config = tmp_path / "config.json"
    config.write_text("{}")
    reference = tmp_path / "reference.json"
    identity = {
        "checkpoint_revision": "b" * 40,
        "upstream_revision": "c" * 40,
        "case_manifest_sha256": "d" * 64,
        "reference_checkpoint_sha256": "a" * 64,
    }
    report = {
        "schema_version": 1,
        "stage": "P0",
        "mode": "reference",
        "status": "passed",
        "real_checkpoint": True,
        "numerical_parity": False,
        "identity": identity,
        "suite": {"complete_prespecified_size": True, "case_count": 30, "output_count": 90},
        "records": [
            {"case_id": f"case-{case}", "seed": seed, "status": "passed"}
            for case in range(30)
            for seed in range(3)
        ],
    }
    reference.write_text(json.dumps(report))
    gate = tmp_path / "gate.json"
    report.update(
        stage="P1",
        mode="full_pipeline",
        numerical_parity=True,
        validation_scope="reference_adapter_numerical_parity",
        p1_acceptance={"status": "unproven"},
    )
    report["identity"] = {**identity, "reference_manifest_sha256": sha256_file(reference)}
    gate.write_text(json.dumps(report))
    alignment = tmp_path / "alignment.json"
    alignment.write_text(
        json.dumps(
            {
                "evidence_kind": "measured_input_alignment",
                "status": "passed",
                "metrics": {"retrieval_accuracy": 0.7},
                "checkpoint_sha256": sha256_file(checkpoint),
                "model_config_sha256": sha256_file(config),
                "tokenizer_sha256": "1" * 64,
                "source_processor_sha256": "2" * 64,
            }
        )
    )
    # Synthetic report-shaped fixtures test validation only, never real acceptance.
    checks = {}
    for name in P1_ACCEPTANCE_CHECKS:
        artifact = tmp_path / f"{name}.txt"
        artifact.write_text("synthetic evidence bytes for validator unit tests")
        checks[name] = {
            "status": "passed",
            "evidence_kind": "real_checkpoint",
            "metrics": {"max_absolute_error": 0.0, "cases_measured": 1},
            "artifacts": [{"path": artifact.name, "sha256": sha256_file(artifact)}],
        }
    (tmp_path / "acceptance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "P1_acceptance",
                "status": "passed",
                "evidence_kind": "real_checkpoint",
                "checks": checks,
                "parity_manifest_sha256": sha256_file(gate),
                "parent_checkpoint_sha256": sha256_file(checkpoint),
                "model_config_sha256": sha256_file(config),
                "reference_checkpoint_sha256": identity["reference_checkpoint_sha256"],
            }
        )
    )
    return gate, alignment, checkpoint, config, reference


def test_connector_only_gradient_and_frozen_buffer_integrity(tmp_path):
    model = _model()
    frozen = frozen_state_hashes(model, ["decoders.image.connector"])
    initial = model.decoders["image"].connector.weight.detach().clone()
    result = _run(model, tmp_path / "run")
    assert result["status"] == "completed"
    assert result["completed_steps"] == 3
    assert result["evidence_kind"] == "fixture_only"
    assert result["p2_gate"] == "not_evaluated"
    assert result["frozen_state_unchanged"]
    assert frozen_state_hashes(model, ["decoders.image.connector"]) == frozen
    assert "decoders.image.generator.nonpersistent" in frozen
    assert not torch.equal(initial, model.decoders["image"].connector.weight)
    assert all(call["connector_training"] for call in model.calls)
    assert all(not call["generator_training"] for call in model.calls)
    assert all(call["requested_outputs"] == ["image"] for call in model.calls)
    assert model.backbone.weight.grad is None
    assert model.decoders["image"].generator[1].weight.grad is None


def test_artifact_records_optimizer_rng_and_only_connector_state(tmp_path):
    output = tmp_path / "run"
    result = _run(_model(), output)
    saved = torch.load(output / "connector.pt", weights_only=True)
    assert saved["step"] == 3
    assert saved["optimizer_state_dict"]["state"]
    assert "torch_cpu" in saved["rng_state"]
    assert all(key.startswith("decoders.image.connector.") for key in saved["connector_state_dict"])
    assert saved["evidence_kind"] == "fixture_only"
    assert result["connector_checkpoint_sha256"] == sha256_file(output / "connector.pt")
    log = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert [entry["step"] for entry in log] == [1, 2, 3]
    assert all(entry["losses"]["image"] > 0 for entry in log)
    restored = _model()
    restored.load_state_dict(saved["connector_state_dict"], strict=False)
    for key, value in saved["connector_state_dict"].items():
        assert torch.equal(value, restored.state_dict()[key])


def test_same_seed_reproduces_connector_updates(tmp_path):
    first, second = _model(), _model()
    _run(first, tmp_path / "first", seed=31)
    _run(second, tmp_path / "second", seed=31)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    assert (tmp_path / "first" / "steps.jsonl").read_text() == (
        tmp_path / "second" / "steps.jsonl"
    ).read_text()


@pytest.mark.parametrize(
    ("behavior", "error", "match"),
    [
        ("detached", RuntimeError, "detached"),
        ("nan", FloatingPointError, "Nonfinite training loss"),
        ("zero_grad", RuntimeError, "gradients are zero"),
        ("mutate_buffer", RuntimeError, "Frozen parameters or buffers changed"),
    ],
)
def test_invalid_training_fails_and_records_failure(tmp_path, behavior, error, match):
    output = tmp_path / behavior
    with pytest.raises(error, match=match):
        _run(_model(behavior=behavior), output)
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "failed"
    assert report["p2_gate"] == "not_evaluated"
    assert not (output / "connector.pt").exists()
    if behavior == "mutate_buffer":
        assert not report["frozen_state_unchanged"]


def test_unused_connector_parameter_is_rejected(tmp_path):
    model = _model()
    model.decoders["image"].connector.register_parameter("unused", nn.Parameter(torch.ones(2)))
    with pytest.raises(RuntimeError, match="received no gradient"):
        _run(model, tmp_path / "run")


@pytest.mark.parametrize(
    "modules",
    [
        ["backbone"],
        ["projectors.image"],
        ["decoders.image"],
        ["decoders.image.connector_extra"],
        ["decoders.image.connector.1"],
        ["decoders.image.connector", "backbone"],
    ],
)
def test_only_complete_output_connector_module_may_train(tmp_path, modules):
    with pytest.raises(ValueError, match="Only the complete image output connector"):
        _run(_model(), tmp_path / "run", connector_modules=modules)
    assert not (tmp_path / "run").exists()


def test_shared_connector_parameter_cannot_unfreeze_backbone(tmp_path):
    model = _model()
    model.backbone.weight = model.decoders["image"].connector.weight
    with pytest.raises(ValueError, match="shared with a frozen module"):
        _run(model, tmp_path / "run")


def test_existing_run_and_invalid_steps_are_rejected(tmp_path):
    output = tmp_path / "run"
    _run(_model(), output, max_steps=1)
    original = (output / "run.json").read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _run(_model(), output)
    assert (output / "run.json").read_bytes() == original
    with pytest.raises(ValueError, match="positive integer"):
        _run(_model(), tmp_path / "zero", max_steps=0)


def test_real_api_cannot_bypass_gates_with_generic_flag(tmp_path):
    with pytest.raises(ValueError, match="validated P0/P1"):
        run_connector_training(
            _model(),
            _batches(),
            output_dir=tmp_path / "run",
            max_steps=1,
            provenance={"evidence_kind": "real_checkpoint"},
        )
    assert not (tmp_path / "run").exists()


def test_evidence_requires_matching_local_checkpoint_and_config(tmp_path):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    result = validate_training_evidence(
        gate,
        alignment,
        reference_report=reference,
        checkpoint=checkpoint,
        model_config=config,
        acceptance_report=tmp_path / "acceptance.json",
    )
    assert result["parent_checkpoint_sha256"] == sha256_file(checkpoint)
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        validate_training_evidence(
            gate,
            alignment,
            reference_report=reference,
            checkpoint=checkpoint,
            model_config=config,
            acceptance_report=tmp_path / "acceptance.json",
        )


@pytest.mark.parametrize("kind", [False, "mock", "passed"])
def test_fixture_evidence_never_authorizes_real_training(tmp_path, kind):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    body = json.loads(gate.read_text())
    body["real_checkpoint"] = kind
    gate.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="real_checkpoint evidence"):
        validate_training_evidence(
            gate,
            alignment,
            reference_report=reference,
            checkpoint=checkpoint,
            model_config=config,
            acceptance_report=tmp_path / "acceptance.json",
        )


def test_cli_checks_evidence_before_importing_loader(tmp_path):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    gate.write_text(json.dumps({"status": "passed"}))
    with pytest.raises(ValueError, match="real_checkpoint evidence"):
        main(
            [
                "--model-config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(tmp_path / "absent.jsonl"),
                "--tokenizer",
                str(tmp_path / "absent-tokenizer"),
                "--source-processor",
                str(tmp_path / "absent-processor"),
                "--gate-report",
                str(gate),
                "--alignment-report",
                str(alignment),
                "--acceptance-report",
                str(tmp_path / "acceptance.json"),
                "--reference-report",
                str(reference),
                "--output-dir",
                str(tmp_path / "run"),
                "--steps",
                "1",
                "--bundle-factory",
                "nonexistent_module:nonexistent_factory",
            ]
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("mode", "replay", "full_pipeline"),
        ("numerical_parity", False, "numerical parity"),
        ("status", "blocked", "must have passed"),
    ],
)
def test_replay_or_blocked_report_cannot_authorize_training(tmp_path, field, value, match):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    body = json.loads(gate.read_text())
    body[field] = value
    gate.write_text(json.dumps(body))
    with pytest.raises(ValueError, match=match):
        validate_training_evidence(
            gate,
            alignment,
            reference_report=reference,
            checkpoint=checkpoint,
            model_config=config,
            acceptance_report=tmp_path / "acceptance.json",
        )


def test_partial_suite_and_wrong_reference_hash_rejected(tmp_path):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    body = json.loads(gate.read_text())
    body["records"].pop()
    gate.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="complete 30-case"):
        validate_training_evidence(
            gate,
            alignment,
            reference_report=reference,
            checkpoint=checkpoint,
            model_config=config,
            acceptance_report=tmp_path / "acceptance.json",
        )
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    body = json.loads(gate.read_text())
    body["identity"]["reference_manifest_sha256"] = "e" * 64
    gate.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="exact passed P0"):
        validate_training_evidence(
            gate,
            alignment,
            reference_report=reference,
            checkpoint=checkpoint,
            model_config=config,
            acceptance_report=tmp_path / "acceptance.json",
        )


def test_gate_alias_must_match_full_manifest(tmp_path):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(gate.read_bytes())
    body = json.loads(gate.read_text())
    body.pop("records")
    body["manifest_sha256"] = sha256_file(manifest)
    gate.write_text(json.dumps(body))
    validate_training_evidence(
        gate,
        alignment,
        reference_report=reference,
        checkpoint=checkpoint,
        model_config=config,
        acceptance_report=tmp_path / "acceptance.json",
    )
    body["status"] = "blocked"
    gate.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="alias disagrees"):
        validate_training_evidence(
            gate,
            alignment,
            reference_report=reference,
            checkpoint=checkpoint,
            model_config=config,
            acceptance_report=tmp_path / "acceptance.json",
        )


def test_actual_loaded_reference_identity_checked_before_training(tmp_path):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    evidence = validate_training_evidence(
        gate,
        alignment,
        reference_report=reference,
        checkpoint=checkpoint,
        model_config=config,
        acceptance_report=tmp_path / "acceptance.json",
    )
    model = _model()
    model.decoders["image"].backend = SimpleNamespace(
        checkpoint_manifest=lambda: {"manifest_sha256": "f" * 64},
        ensure_loaded=lambda: None,
        to=lambda **kwargs: None,
    )
    with pytest.raises(ValueError, match="Loaded reference checkpoint"):
        run_connector_training(
            model, _batches(), output_dir=tmp_path / "run", max_steps=1, provenance=evidence
        )
    assert not model.calls
    assert not (tmp_path / "run").exists()


def test_default_loader_import_and_cli_help_are_offline(capsys):
    from src.decoders.loading import load_image_training_bundle

    assert callable(load_image_training_bundle)
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--reference-report" in capsys.readouterr().out


@pytest.mark.parametrize("field", ["tokenizer_sha256", "source_processor_sha256"])
def test_alignment_must_bind_each_preprocessing_asset(tmp_path, field):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    report = json.loads(alignment.read_text())
    report.pop(field)
    alignment.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=f"bind preprocessing identity: {field}"):
        validate_training_evidence(
            gate,
            alignment,
            reference_report=reference,
            checkpoint=checkpoint,
            model_config=config,
            acceptance_report=tmp_path / "acceptance.json",
        )


@pytest.mark.parametrize("field", ["tokenizer_sha256", "source_processor_sha256"])
def test_cli_rejects_changed_loaded_preprocessing_before_training(tmp_path, monkeypatch, field):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    evidence = validate_training_evidence(
        gate,
        alignment,
        reference_report=reference,
        checkpoint=checkpoint,
        model_config=config,
        acceptance_report=tmp_path / "acceptance.json",
    )
    factory_module = ModuleType("fixture_training_bundle")
    fixture_model = _model()
    loaded_provenance = dict(evidence)
    # Equal dimensions/vocabulary sizes cannot replace equality of asset contents.
    loaded_provenance[field] = "f" * 64
    factory_module.load = lambda **kwargs: {
        "model": fixture_model,
        "provenance": loaded_provenance,
    }
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    tokenizer_dir = tmp_path / "tokenizer"
    processor_dir = tmp_path / "processor"
    tokenizer_dir.mkdir()
    processor_dir.mkdir()
    manifest = tmp_path / "data.jsonl"
    manifest.write_text("")
    with pytest.raises(ValueError, match=f"Loaded preprocessing.*{field}"):
        main(
            [
                "--model-config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--tokenizer",
                str(tokenizer_dir),
                "--source-processor",
                str(processor_dir),
                "--gate-report",
                str(gate),
                "--alignment-report",
                str(alignment),
                "--acceptance-report",
                str(tmp_path / "acceptance.json"),
                "--reference-report",
                str(reference),
                "--output-dir",
                str(tmp_path / "run"),
                "--steps",
                "1",
                "--bundle-factory",
                "fixture_training_bundle:load",
            ]
        )
    assert not fixture_model.calls
    assert not (tmp_path / "run").exists()


def _validate_fixture_evidence(tmp_path, paths):
    gate, alignment, checkpoint, config, reference = paths
    return validate_training_evidence(
        gate,
        alignment,
        reference_report=reference,
        checkpoint=checkpoint,
        model_config=config,
        acceptance_report=tmp_path / "acceptance.json",
    )


def test_adapter_parity_alone_does_not_authorize_training(tmp_path):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    with pytest.raises(ValueError, match="separate real-checkpoint P1 acceptance"):
        validate_training_evidence(
            gate, alignment, reference_report=reference, checkpoint=checkpoint, model_config=config
        )


@pytest.mark.parametrize("metric", [True, float("nan"), "passed"])
def test_alignment_requires_finite_numeric_measurements(tmp_path, metric):
    paths = _evidence(tmp_path)
    alignment = paths[1]
    report = json.loads(alignment.read_text())
    report["metrics"] = {"retrieval_accuracy": metric}
    alignment.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="measured input-alignment report"):
        _validate_fixture_evidence(tmp_path, paths)


@pytest.mark.parametrize(
    "field",
    [
        "parity_manifest_sha256",
        "parent_checkpoint_sha256",
        "model_config_sha256",
        "reference_checkpoint_sha256",
    ],
)
def test_acceptance_report_binds_exact_inputs(tmp_path, field):
    paths = _evidence(tmp_path)
    path = tmp_path / "acceptance.json"
    report = json.loads(path.read_text())
    report[field] = "f" * 64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=f"acceptance identity mismatch: {field}"):
        _validate_fixture_evidence(tmp_path, paths)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("fixture_report", "real-checkpoint report"),
        ("smoke_report", "real-checkpoint report"),
        ("missing_check", "checkpoint_reload"),
        ("boolean_check", "checkpoint_reload"),
        ("fixture_check", "checkpoint_reload"),
        ("boolean_metric", "finite numeric metrics"),
        ("nan_metric", "finite numeric metrics"),
        ("empty_metrics", "finite numeric metrics"),
        ("missing_artifacts", "hashed artifacts"),
    ],
)
def test_acceptance_needs_measured_nonfixture_named_checks(tmp_path, mutation, match):
    paths = _evidence(tmp_path)
    path = tmp_path / "acceptance.json"
    report = json.loads(path.read_text())
    check = report["checks"]["checkpoint_reload"]
    if mutation == "fixture_report":
        report["evidence_kind"] = "fixture_only"
    elif mutation == "smoke_report":
        report["smoke"] = True
    elif mutation == "missing_check":
        report["checks"].pop("checkpoint_reload")
    elif mutation == "boolean_check":
        report["checks"]["checkpoint_reload"] = True
    elif mutation == "fixture_check":
        check["evidence_kind"] = "fixture_only"
    elif mutation == "boolean_metric":
        check["metrics"] = {"passed": True}
    elif mutation == "nan_metric":
        check["metrics"] = {"error": float("nan")}
    elif mutation == "empty_metrics":
        check["metrics"] = {}
    else:
        check["artifacts"] = []
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        _validate_fixture_evidence(tmp_path, paths)


@pytest.mark.parametrize("missing", [False, True])
def test_acceptance_artifacts_must_exist_and_match_hash(tmp_path, missing):
    paths = _evidence(tmp_path)
    artifact = tmp_path / "checkpoint_reload.txt"
    if missing:
        artifact.unlink()
    else:
        artifact.write_text("changed bytes")
    with pytest.raises(
        ValueError, match="artifact missing" if missing else "artifact SHA256 mismatch"
    ):
        _validate_fixture_evidence(tmp_path, paths)


def test_fixture_artifact_cannot_be_relabelled_real_in_acceptance_report(tmp_path):
    paths = _evidence(tmp_path)
    artifact = tmp_path / "fixture.json"
    artifact.write_text(json.dumps({"evidence_kind": "fixture_only", "status": "passed"}))
    path = tmp_path / "acceptance.json"
    report = json.loads(path.read_text())
    report["checks"]["checkpoint_reload"]["artifacts"] = [
        {"path": artifact.name, "sha256": sha256_file(artifact)}
    ]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="artifact is fixture/smoke"):
        _validate_fixture_evidence(tmp_path, paths)


def test_real_programmatic_training_revalidates_acceptance_before_backend_load(tmp_path):
    paths = _evidence(tmp_path)
    evidence = _validate_fixture_evidence(tmp_path, paths)
    assert evidence["p1_acceptance"] == "passed"
    assert evidence["acceptance_artifact_sha256"]
    (tmp_path / "checkpoint_reload.txt").write_text("tampered after preflight")
    model = _model()  # Has no backend; validation must fail before backend access.
    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        run_connector_training(
            model, _batches(), output_dir=tmp_path / "run", max_steps=1, provenance=evidence
        )
    assert not model.calls


def test_cli_rejects_acceptance_before_loading_any_bundle(tmp_path):
    gate, alignment, checkpoint, config, reference = _evidence(tmp_path)
    (tmp_path / "checkpoint_reload.txt").unlink()
    options = {
        "--model-config": config,
        "--checkpoint": checkpoint,
        "--manifest": tmp_path / "absent.jsonl",
        "--tokenizer": tmp_path / "absent-tokenizer",
        "--source-processor": tmp_path / "absent-processor",
        "--gate-report": gate,
        "--alignment-report": alignment,
        "--reference-report": reference,
        "--acceptance-report": tmp_path / "acceptance.json",
        "--output-dir": tmp_path / "run",
        "--steps": 1,
        "--bundle-factory": "nonexistent_module:nonexistent_factory",
    }
    with pytest.raises(ValueError, match="artifact missing"):
        main([part for key, value in options.items() for part in (key, str(value))])

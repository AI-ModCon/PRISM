"""Offline pilot control-flow checks, not real checkpoint capability evidence."""

import io
import json
import tarfile
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from src.decoders.loading import file_sha256
from tools.prepare_docci_webdataset import convert_docci
from tools.train_prism_image_connector import (
    PRECHECK_SOURCES,
    ROOT,
    ShuffledEpochOrder,
    main,
    validate_repeatability_report,
)
from torch import nn


class TinyTokenizer:
    def __call__(self, prompts, **kwargs):
        assert kwargs == {"padding": True, "truncation": False, "return_tensors": "pt"}
        ids = torch.tensor([[sum(map(ord, prompt)) % 7 + 1, 2, 3] for prompt in prompts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TinyBackend(nn.Linear):
    def __init__(self):
        super().__init__(3, 3, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.eye(3))
        self.register_buffer("frozen_counter", torch.tensor(0), persistent=False)
        self.last_trace = {}

    def ensure_loaded(self):
        return self

    def checkpoint_manifest(self):
        return {"manifest_sha256": "b" * 64, "fixture": True}

    def provenance(self):
        return {"kernel_policy": {"policy": "fixture"}, "fixture": True}

    def sample(self, options):
        latent = options.get("latents")
        if latent is None:
            latent = torch.randn(1, 3, 2, 2, generator=options["generator"])
        self.last_trace = {"latents.initial": latent.detach().clone()}
        return [Image.new("RGB", (32, 32), "blue")]

    def generate_reference(self, context, **options):
        assert context["reference_images"] == [[]]
        assert context["prompt"]
        return self.sample(options)


class TinyParent(nn.Module):
    def __init__(self, *, poison=False, mutate=False):
        super().__init__()
        self.backbone = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(3))
        image = nn.Module()
        image.connector = nn.Linear(3, 3)
        image.backend = TinyBackend()
        self.decoders = nn.ModuleDict({"image": image})
        self.poison, self.mutate = poison, mutate
        self.optimization_targets = []
        self.predict_calls = 0

    def forward_outputs(self, inputs, *, targets, requested_outputs, native_context, output_specs):
        assert set(inputs) == {"text", "text_attention_mask"}
        assert set(targets) == {"image"} and requested_outputs == ["image"]
        assert native_context["image"]["reference_images"] == [[]] * len(inputs["text"])
        assert not self.backbone.training and not self.backbone.weight.requires_grad
        image = self.decoders["image"]
        assert not image.backend.weight.requires_grad
        target = targets["image"].mean((-2, -1))
        if torch.is_grad_enabled():
            self.optimization_targets.extend(target.detach().tolist())
            if self.mutate:
                image.backend.frozen_counter += 1
        prediction = image.backend(image.connector(self.backbone(inputs["text"].float())))
        loss = (prediction - target + torch.randn_like(prediction) * 0.01).square().mean()
        if self.poison and torch.is_grad_enabled():
            loss *= float("nan")
        return SimpleNamespace(losses={"image": loss})

    def predict(self, *, inputs, requested_outputs, native_context, decoder_kwargs):
        assert set(inputs) == {"text", "text_attention_mask"}
        assert native_context["image"]["reference_images"] == [[]]
        self.predict_calls += 1
        return SimpleNamespace(
            predictions={"image": self.decoders["image"].backend.sample(decoder_kwargs["image"])}
        )


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    import src.decoders.loading as loading

    rows = []
    archive = tmp_path / "images.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for number, split in enumerate(["train"] * 4 + ["qual_dev"] * 3 + ["test", "qual_test"]):
            identifier = f"{split}_{number:05d}"
            rows.append(
                {
                    "example_id": identifier,
                    "image_file": identifier + ".jpg",
                    "split": split,
                    "description": f"A photograph of square number {number}.",
                }
            )
            buffer = io.BytesIO()
            Image.new("RGB", (12, 8), (15 + 20 * number, 40, 170)).save(buffer, format="JPEG")
            content = buffer.getvalue()
            header = tarfile.TarInfo("images/" + identifier + ".jpg")
            header.size = len(content)
            handle.addfile(header, io.BytesIO(content))
    descriptions = tmp_path / "descriptions.jsonl"
    descriptions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    convert_docci(descriptions, archive, tmp_path / "data", expected_counts=None, shard_records=2)
    precheck = {
        "status": "completed",
        "mode": "repeatability_diagnostic",
        "evidence_kind": "real_checkpoint_diagnostic",
        "fixture": False,
        "comparisons": {
            key: {"passed": True}
            for key in ("native_repeat", "adapter_vs_native", "native_after_adapter")
        },
        "identity": {
            "dtype": "float32",
            "device_type": "cpu",
            "device_name": "cpu",
            "torch_version": str(torch.__version__),
            "numerical_policy": {"deterministic_algorithms": True, "attention_backend": "math"},
            "kernel_policy": {"policy": "fixture"},
            "checkpoint_manifest_sha256": "b" * 64,
        },
        "source_hashes": {name: file_sha256(ROOT / name) for name in PRECHECK_SOURCES},
    }
    precheck_path = tmp_path / "repeatability.json"
    precheck_path.write_text(json.dumps(precheck))
    paths = {}
    for name in ("model-config", "checkpoint", "tokenizer", "source-processor"):
        paths[name] = tmp_path / name
        paths[name].write_text("fixture")
    paths.update(
        {
            "train-index": tmp_path / "data/train.jsonl",
            "validation-index": tmp_path / "data/validation.jsonl",
            "repeatability-report": precheck_path,
        }
    )
    models, flags = [], {"poison": False, "mutate": False}

    def load(*args):
        model = TinyParent(**flags)
        models.append(model)
        return {
            "model": model,
            "tokenizer": TinyTokenizer(),
            "source_transform": None,
            "provenance": {
                "parent_checkpoint_sha256": "a" * 64,
                "fixture": True,
                "restoration": {
                    "strict_parent": True,
                    "loaded_key_count": 2,
                    "missing_parent_keys": [],
                    "unexpected_keys": [],
                },
            },
        }

    monkeypatch.setattr(loading, "load_image_training_bundle", load)
    args = [value for key, path in paths.items() for value in ("--" + key, str(path))]
    args += [
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--height",
        "32",
        "--width",
        "32",
        "--expected-parent-tensors",
        "2",
        "--deterministic",
        "--gradient-accumulation",
        "2",
        "--probe-count",
        "2",
        "--checkpoint-every",
        "2",
        "--eval-every",
        "2",
        "--sample-every",
        "2",
        "--sampling-steps",
        "2",
        "--learning-rate",
        "0.01",
    ]
    return SimpleNamespace(
        root=tmp_path,
        args=args,
        models=models,
        flags=flags,
        precheck=precheck,
        precheck_path=precheck_path,
    )


def run(experiment, name, steps, *extra):
    output = experiment.root / name
    result = main(experiment.args + ["--output-dir", str(output), "--steps", str(steps), *extra])
    return result, output, json.loads((output / "report.json").read_text())


def test_training_uses_shuffled_train_only_and_fixed_full_validation(experiment):
    result, output, report = run(experiment, "pilot", 4, "--native-baseline")
    assert result == 0 and report["status"] == "completed"
    assert report["evidence_kind"] == "fixture_only" and report["qualification"] == "unqualified"
    assert report["p2_gate"] == "not_evaluated" and not report["quality_benchmark"]
    assert report["frozen_hashes_before"] == report["frozen_hashes_after"]
    assert "decoders.image.backend.frozen_counter" in report["frozen_hashes_after"]
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    ids = [
        identifier
        for step in steps
        for batch in step["microbatches"]
        for identifier in batch["ids"]
    ]
    assert len(ids) == 8 and all(identifier.startswith("train_") for identifier in ids)
    assert len(set(ids[:4])) == len(set(ids[4:])) == 4
    assert all(step["gradient_norm_before_clip"] > 0 for step in steps)
    assert len(experiment.models[-1].optimization_targets) == 8
    assert all(target[0] < -0.3 for target in experiment.models[-1].optimization_targets)
    assert report["evaluations"][0]["splits"]["validation"]["count"] == 2
    train_probes = report["evaluations"][0]["splits"]["train"]["examples"]
    assert [row["id"] for row in train_probes] == ids[:2]
    assert not any(row["optimized_before_evaluation"] for row in train_probes)
    assert all(
        row["optimized_before_evaluation"]
        for row in report["evaluations"][-1]["splits"]["train"]["examples"]
    )
    assert report["evaluations"][-1]["splits"]["validation"]["count"] == 3
    assert report["evaluations"][-1]["full_validation"]
    assert len((output / "evaluations.jsonl").read_text().splitlines()) == 3
    for identifier in set(row["id"] for row in report["samples"]):
        samples = [row for row in report["samples"] if row["id"] == identifier]
        assert len({row["initial_latent_sha256"] for row in samples}) == 1
        assert all(row["sampling_target_free"] for row in samples)
    assert len([row for row in report["samples"] if row["stage"] == "native"]) == 2
    assert all(
        row["optimized_before_sampling"]
        for row in report["samples"]
        if row["stage"] == "trained" and row["split"] == "train"
    )
    assert report["sampler_state"]["examples_seen"] == 8


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_equal(a, b)
    else:
        assert left == right


def test_resume_matches_uninterrupted_optimizer_rng_and_data_order(experiment):
    _, full_output, full = run(experiment, "full", 4)
    _, first_output, first = run(experiment, "first", 2)
    _, resume_output, resumed = run(
        experiment, "resume", 4, "--resume", first["checkpoints"][-1]["path"]
    )
    full_state = torch.load(full["checkpoints"][-1]["path"], weights_only=True)
    resumed_state = torch.load(resumed["checkpoints"][-1]["path"], weights_only=True)
    for key in ("connector_state_dict", "optimizer_state_dict", "rng_state", "sampler_state"):
        _assert_equal(full_state[key], resumed_state[key])
    all_steps = [
        json.loads(line)
        for path in (first_output, resume_output)
        for line in (path / "steps.jsonl").read_text().splitlines()
    ]
    full_steps = [
        json.loads(line) for line in (full_output / "steps.jsonl").read_text().splitlines()
    ]
    assert [row["microbatches"] for row in all_steps] == [row["microbatches"] for row in full_steps]
    assert [row["loss"] for row in all_steps] == [row["loss"] for row in full_steps]
    assert resumed["evaluations"][-1] == full["evaluations"][-1]


def test_smoke_final_validation_limit_is_explicit(experiment):
    _, _, report = run(experiment, "smoke", 1, "--final-validation-count", "1")
    final = report["evaluations"][-1]
    assert final["final"] and not final["full_validation"]
    assert final["splits"]["validation"]["count"] == 1


def test_nonfinite_gradient_path_aborts_and_frozen_mutation_fails(experiment):
    experiment.flags["poison"] = True
    with pytest.raises(RuntimeError, match="finite scalar"):
        run(experiment, "bad_loss", 1)
    failed = json.loads((experiment.root / "bad_loss/report.json").read_text())
    assert failed["status"] == "failed" and failed["completed_steps"] == 0
    experiment.flags.update(poison=False, mutate=True)
    result, _, report = run(experiment, "mutated", 1)
    assert result == 2 and report["status"] == "failed" and not report["frozen_state_unchanged"]


@pytest.mark.parametrize("field", ["fixture", "comparison", "checkpoint", "sources", "dtype"])
def test_repeatability_report_rejects_missing_or_mismatched_evidence(experiment, field):
    report = experiment.precheck
    if field == "fixture":
        report["fixture"] = True
    elif field == "comparison":
        report["comparisons"]["native_repeat"]["passed"] = False
    elif field == "checkpoint":
        report["identity"]["checkpoint_manifest_sha256"] = "c" * 64
    elif field == "sources":
        report["source_hashes"].pop(PRECHECK_SOURCES[0])
    else:
        report["identity"]["dtype"] = "bfloat16"
    experiment.precheck_path.write_text(json.dumps(report))
    args = SimpleNamespace(
        dtype="float32", device="cpu", deterministic=True, attention_backend="math"
    )
    with pytest.raises(ValueError, match="[Rr]epeatability|real-checkpoint"):
        validate_repeatability_report(
            experiment.precheck_path,
            args,
            checkpoint_manifest={"manifest_sha256": "b" * 64},
            backend={"kernel_policy": {"policy": "fixture"}},
        )


def test_sampler_checkpoint_preserves_next_unread_example_and_rejects_bad_cursor():
    order = ShuffledEpochOrder(5, 19)
    first = order.take(7)
    assert set(first[:5]) == set(range(5))
    saved = order.state_dict()
    replay = ShuffledEpochOrder(5, 19)
    replay.load_state_dict(saved)
    assert replay.take(12) == order.take(12)
    saved["cursor"] = 6
    with pytest.raises(ValueError, match="cursor"):
        replay.load_state_dict(saved)

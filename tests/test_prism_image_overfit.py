"""Offline control-flow tests; fixture models cannot establish image capability."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from src.decoders.loading import load_image_connector
from tools.overfit_prism_image_connector import EVIDENCE_KIND, main, validate_budget
from torch import nn


class FixtureTokenizer:
    def __call__(self, prompts, **kwargs):
        assert kwargs == {"padding": True, "truncation": False, "return_tensors": "pt"}
        ids = torch.tensor([[len(prompt) % 4 + 1, 2, 3] for prompt in prompts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FixtureBackend(nn.Linear):
    def __init__(self):
        super().__init__(3, 3, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.eye(3))
        self.register_buffer("frozen_counter", torch.tensor(0), persistent=False)
        self.last_trace = {}
        self.samples = []

    def ensure_loaded(self):
        return self

    def checkpoint_manifest(self):
        return {"manifest_sha256": "b" * 64, "fixture": True}

    def provenance(self):
        return {"fixture": True}

    def generate_reference(self, context, **options):
        assert context["prompt"].startswith("A photograph of ")
        assert context["reference_images"] == [[]]
        assert options["num_inference_steps"] == 50
        assert options["text_guidance_scale"] == 5.0
        assert options["image_guidance_scale"] == 2.0
        assert options["negative_prompt"] == ""
        assert options["trace"]
        assert "latents" not in options
        self.last_trace = {
            "latents.initial": torch.randn(1, 3, 2, 2, generator=options["generator"])
        }
        self.samples.append(("native", self.last_trace["latents.initial"].clone()))
        return [Image.new("RGB", (32, 32), "blue")]


class FixtureParent(nn.Module):
    def __init__(self, poison=False):
        super().__init__()
        self.backbone = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(3))
        image = nn.Module()
        image.connector = nn.Linear(3, 3)
        image.backend = FixtureBackend()
        self.decoders = nn.ModuleDict({"image": image})
        self.supervised_calls = 0
        self.predict_calls = 0
        self.poison = poison

    def forward_outputs(self, inputs, *, targets, requested_outputs, native_context, output_specs):
        assert requested_outputs == ["image"]
        assert set(inputs) == {"text", "text_attention_mask"}
        assert set(targets) == {"image"}
        assert not self.backbone.training
        assert not self.backbone.weight.requires_grad
        assert not self.decoders["image"].backend.weight.requires_grad
        assert output_specs == {"image": {"height": 32, "width": 32}}
        self.supervised_calls += 1
        image = self.decoders["image"]
        features = self.backbone(inputs["text"].float())
        prediction = image.backend(image.connector(features))
        noise = torch.randn_like(prediction) * 0.01
        loss = (prediction - targets["image"].mean((-2, -1)) + noise).square().mean()
        if self.poison and torch.is_grad_enabled():
            loss = loss * float("nan")
        return SimpleNamespace(losses={"image": loss})

    def predict(self, *, inputs, requested_outputs, native_context, decoder_kwargs):
        # No targets argument; the sample input keys cannot contain target supervision.
        assert set(inputs) == {"text", "text_attention_mask"}
        assert requested_outputs == ["image"]
        assert native_context["image"]["reference_images"] == [[]]
        options = decoder_kwargs["image"]
        assert options["num_inference_steps"] == 50
        assert options["text_guidance_scale"] == 5.0
        assert options["image_guidance_scale"] == 2.0
        assert options["negative_prompt"] == ""
        self.predict_calls += 1
        backend = self.decoders["image"].backend
        backend.last_trace = {"latents.initial": options["latents"].detach().clone()}
        backend.samples.append(("prism", options["latents"].detach().clone()))
        return SimpleNamespace(predictions={"image": [Image.new("RGB", (32, 32), "green")]})


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    import src.decoders.loading as loading

    assets = {}
    for name in ("model-config", "checkpoint", "tokenizer", "source-processor"):
        path = tmp_path / name
        path.write_text("fixture")
        assets[name] = path
    for split, colors in (("train", ("red", "green")), ("validation", ("blue", "yellow"))):
        rows = []
        for index, color in enumerate(colors):
            path = tmp_path / f"{split}-{index}.png"
            Image.new("RGB", (32, 32), color).save(path)
            rows.append(
                {
                    "id": f"{split}-{index}",
                    "task": "t2i",
                    "prompt": f"A photograph of a {color} square.",
                    "source_images": [],
                    "target_image": path.name,
                    "split": split,
                    "group_ids": [f"{split}-{index}"],
                }
            )
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        assets["manifest" if split == "train" else "validation-manifest"] = path
    models = []
    poison = [False]

    def load(*args):
        model = FixtureParent(poison=poison[0])
        models.append(model)
        return {
            "model": model,
            "tokenizer": FixtureTokenizer(),
            "source_transform": lambda image: image,
            "provenance": {"parent_checkpoint_sha256": "a" * 64, "fixture": True},
        }

    monkeypatch.setattr(loading, "load_image_training_bundle", load)
    args = [value for name, path in assets.items() for value in ("--" + name, str(path))]
    args += [
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--height",
        "32",
        "--width",
        "32",
        "--learning-rate",
        "0.01",
        "--sample-every",
        "2",
    ]
    return SimpleNamespace(root=tmp_path, args=args, models=models, assets=assets, poison=poison)


def _run(experiment, name, steps, *extra):
    output = experiment.root / name
    result = main(experiment.args + ["--output-dir", str(output), "--steps", str(steps), *extra])
    assert result == 0
    return output, json.loads((output / "report.json").read_text())


def test_overfit_has_fixed_multiexample_probes_native_noise_replay_and_unqualified_artifacts(
    experiment,
):
    output, report = _run(
        experiment, "overfit", 4, "--deterministic", "--attention-backend", "math"
    )
    assert report["status"] == "completed"
    assert report["evidence_kind"] == EVIDENCE_KIND
    assert report["qualification"] == "unqualified"
    assert report["p2_gate"] == "not_evaluated"
    assert report["parent_heldout_visual_quality_established"] is False
    assert report["source_image_conditioning_exercised"] is False
    assert report["completed_steps"] == 4
    assert report["frozen_hashes_before"] == report["frozen_hashes_after"]
    assert "decoders.image.backend.frozen_counter" in report["frozen_hashes_before"]
    assert len(report["evaluations"]) == 3
    for evaluation in report["evaluations"]:
        assert set(evaluation["splits"]) == {"train", "validation"}
        assert all(len(values["examples"]) == 2 for values in evaluation["splits"].values())
    assert len(report["samples"]) == 16  # four each: native, untrained, trained at 2/4
    for identifier in ("train-0", "train-1", "validation-0", "validation-1"):
        entries = [row for row in report["samples"] if row["id"] == identifier]
        assert len({row["initial_latent_sha256"] for row in entries}) == 1
        assert all(row["sampling_target_free"] for row in entries)
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert [step["id"] for step in steps] == ["train-0", "train-1"] * 2
    assert all(any(norm > 0 for norm in step["gradient_norms"].values()) for step in steps)
    for checkpoint in report["checkpoints"]:
        saved = torch.load(checkpoint["path"], weights_only=True)
        assert saved["qualification"] == "unqualified"
        assert "rng_state" in saved and "optimizer_state_dict" in saved
        assert len(saved["initial_latents"]) == 4
        with pytest.raises(ValueError, match="accepted training artifact"):
            load_image_connector(
                experiment.models[-1],
                checkpoint["path"],
                parent_checkpoint_sha256="a" * 64,
                reference_checkpoint_sha256="b" * 64,
            )


def test_resume_produces_same_optimizer_and_connector_as_uninterrupted_run(experiment):
    full_path, full = _run(experiment, "full", 4)
    _, pilot = _run(experiment, "pilot", 2)
    resumed_path, resumed = _run(
        experiment, "resumed", 4, "--resume", pilot["checkpoints"][-1]["path"]
    )
    full_saved = torch.load(full["checkpoints"][-1]["path"], weights_only=True)
    resumed_saved = torch.load(resumed["checkpoints"][-1]["path"], weights_only=True)
    assert resumed["completed_steps"] == 4
    assert len(resumed["origin_baseline_samples"]) == 8
    assert [row["stage"] for row in resumed["samples"]] == ["trained"] * 4
    for name, expected in full_saved["connector_state_dict"].items():
        assert torch.equal(expected, resumed_saved["connector_state_dict"][name])
    for index, expected in full_saved["optimizer_state_dict"]["state"].items():
        for name, tensor in expected.items():
            assert torch.equal(tensor, resumed_saved["optimizer_state_dict"]["state"][index][name])
    full_steps = (full_path / "steps.jsonl").read_text().splitlines()
    resumed_steps = (resumed_path / "steps.jsonl").read_text().splitlines()
    assert [json.loads(row)["loss"] for row in full_steps[2:]] == [
        json.loads(row)["loss"] for row in resumed_steps
    ]


def test_resume_rejects_changed_data_or_numerical_policy(experiment):
    _, pilot = _run(experiment, "pilot", 1)
    with pytest.raises(ValueError, match="protocol/data/parent/runtime"):
        _run(
            experiment,
            "changed",
            2,
            "--resume",
            pilot["checkpoints"][-1]["path"],
            "--deterministic",
        )


def test_cross_split_duplicate_pixels_rejected_before_checkpoint_load(experiment):
    from shutil import copyfile

    copyfile(experiment.root / "train-0.png", experiment.root / "validation-0.png")
    with pytest.raises(ValueError, match="split leakage"):
        _run(experiment, "leaked", 1)
    assert not experiment.models
    assert not (experiment.root / "leaked").exists()


def test_nonfinite_training_loss_fails_without_qualifying_checkpoint(experiment):
    experiment.poison[0] = True
    with pytest.raises(RuntimeError, match="finite differentiable"):
        _run(experiment, "nonfinite", 1)
    report = json.loads((experiment.root / "nonfinite" / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["completed_steps"] == 0
    assert report["frozen_state_unchanged"]
    assert report["qualification"] == "unqualified"


@pytest.mark.parametrize(
    "steps,train_count,validation_count,height,width,sampling",
    [
        (1001, 16, 8, 256, 256, 50),
        (500, 17, 8, 256, 256, 50),
        (500, 16, 9, 256, 256, 50),
        (500, 16, 8, 512, 256, 50),
        (500, 16, 8, 256, 256, 51),
    ],
)
def test_budget_rejects_unbounded_runs(
    steps, train_count, validation_count, height, width, sampling
):
    with pytest.raises(ValueError):
        validate_budget(steps, train_count, validation_count, height, width, sampling)


def test_sampling_does_not_read_training_or_validation_target_pixels(experiment, monkeypatch):
    from src.data.image_generation import ImageGenerationDataset

    original = ImageGenerationDataset._read_rgb
    output = experiment.root / "target-free"
    accesses = []

    def read(path):
        report_path = output / "report.json"
        phase = json.loads(report_path.read_text())["phase"] if report_path.exists() else "setup"
        assert not phase.startswith("sampling_"), "T2I inference must not open target image pixels"
        accesses.append(phase)
        return original(path)

    monkeypatch.setattr(ImageGenerationDataset, "_read_rgb", staticmethod(read))
    _run(experiment, "target-free", 1)
    assert "fixed_probes" in accesses and "training" in accesses


def test_changed_sampling_noise_is_rejected(experiment, monkeypatch):
    original = FixtureParent.predict

    def corrupt(self, **kwargs):
        result = original(self, **kwargs)
        self.decoders["image"].backend.last_trace["latents.initial"].add_(1)
        return result

    monkeypatch.setattr(FixtureParent, "predict", corrupt)
    with pytest.raises(RuntimeError, match="initial latents differ"):
        _run(experiment, "corrupt-noise", 1)
    report = json.loads((experiment.root / "corrupt-noise" / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["completed_steps"] == 0


def test_caption_ablation_changes_only_caption_preserves_target_and_rng(experiment, monkeypatch):
    original = FixtureParent.forward_outputs
    probes = []

    def capture(self, inputs, **kwargs):
        if not torch.is_grad_enabled():
            probes.append(
                {
                    "text": inputs["text"].clone(),
                    "target": kwargs["targets"]["image"].clone(),
                    "rng": torch.get_rng_state().clone(),
                }
            )
        return original(self, inputs, **kwargs)

    monkeypatch.setattr(FixtureParent, "forward_outputs", capture)
    _, report = _run(experiment, "ablation", 1, "--conditioning-ablation")
    assert len(probes) == 16  # before/after x 2 splits x 2 examples x matched/shuffled
    for correct, shuffled in zip(probes[::2], probes[1::2], strict=True):
        assert torch.equal(correct["target"], shuffled["target"])
        assert torch.equal(correct["rng"], shuffled["rng"])
        assert not torch.equal(correct["text"], shuffled["text"])
    for evaluation in report["evaluations"]:
        for split, values in evaluation["splits"].items():
            assert values["conditioning_ablation"]["applicable_count"] == 2
            gaps = []
            for row in values["examples"]:
                assert row["conditioning_ablation_applicable"]
                assert row["shuffled_prompt_id"] != row["id"]
                assert row["shuffled_prompt_id"].startswith(split + "-")
                assert row["shuffled_minus_correct"] == pytest.approx(
                    row["shuffled_loss"] - row["loss"]
                )
                gaps.append(row["shuffled_minus_correct"])
            assert values["conditioning_ablation"]["mean_shuffled_minus_correct"] == pytest.approx(
                sum(gaps) / 2
            )
    assert report["frozen_state_unchanged"]
    assert report["completed_steps"] == 1


def test_caption_ablation_skips_identical_captions(experiment):
    for name in ("manifest", "validation-manifest"):
        path = experiment.assets[name]
        rows = [json.loads(row) for row in path.read_text().splitlines()]
        for row in rows:
            row["prompt"] = "A photograph of a square."
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _, report = _run(experiment, "identical-captions", 1, "--conditioning-ablation")
    for evaluation in report["evaluations"]:
        for values in evaluation["splits"].values():
            assert values["conditioning_ablation"]["applicable_count"] == 0
            assert values["conditioning_ablation"]["mean_shuffled_loss"] is None
            for row in values["examples"]:
                assert not row["conditioning_ablation_applicable"]
                assert row["conditioning_ablation_skip_reason"] == "no_distinct_t2i_caption"
                assert "shuffled_loss" not in row

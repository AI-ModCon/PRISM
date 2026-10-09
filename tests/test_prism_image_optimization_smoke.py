"""Offline fixtures verify smoke control flow, never pretrained-model readiness."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from src.decoders.loading import load_image_connector
from tools.smoke_prism_image_training import main
from torch import nn


class FixtureBackend(nn.Linear):
    def __init__(self):
        super().__init__(3, 3, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.eye(3))
        self.register_buffer("frozen_counter", torch.tensor(0), persistent=False)
        self.loaded = False

    def ensure_loaded(self):
        self.loaded = True

    def checkpoint_manifest(self):
        return {"manifest_sha256": "b" * 64, "fixture": True}

    def provenance(self):
        return {"fixture": True}


class FixtureParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(3))
        image = nn.Module()
        image.connector = nn.Linear(3, 3)
        with torch.no_grad():
            image.connector.weight.fill_(0.1)
            image.connector.bias.fill_(0.1)
        image.backend = FixtureBackend()
        self.decoders = nn.ModuleDict({"image": image})
        self.supervised_calls = 0
        self.predict_calls = 0

    def forward_outputs(self, inputs, *, targets, requested_outputs, native_context, output_specs):
        assert requested_outputs == ["image"]
        assert set(inputs) == {"features"}
        assert native_context == {"image": {"reference_images": [[]]}}
        assert output_specs == {"image": {"height": 32, "width": 32}}
        assert not self.backbone.training
        self.supervised_calls += 1
        image = self.decoders["image"]
        features = self.backbone(inputs["features"])
        prediction = image.backend(image.connector(features))
        return SimpleNamespace(losses={"image": (prediction - targets["image"]).square().mean()})

    def predict(self, *, inputs, requested_outputs, native_context, decoder_kwargs):
        # Signature deliberately has no targets: evaluation must be target-free.
        assert set(inputs) == {"features"}
        assert requested_outputs == ["image"]
        assert native_context == {"image": {"reference_images": [[]]}}
        assert decoder_kwargs["image"]["num_inference_steps"] == 1
        self.predict_calls += 1
        return SimpleNamespace(predictions={"image": [Image.new("RGB", (32, 32), "blue")]})


def test_fixture_optimization_smoke_updates_only_connector_and_keeps_artifact_unqualified(
    tmp_path, monkeypatch
):
    import src.data.image_generation as data
    import src.decoders.loading as loading

    model = FixtureParent()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    assets = {}
    for name in ("model-config", "checkpoint", "tokenizer", "source-processor", "manifest"):
        path = tmp_path / name
        path.write_text("fixture")
        assets[name] = path

    monkeypatch.setattr(
        loading,
        "load_image_training_bundle",
        lambda *args: {
            "model": model,
            "tokenizer": object(),
            "source_transform": lambda image: image,
            "provenance": {"parent_checkpoint_sha256": "a" * 64, "fixture": True},
        },
    )
    monkeypatch.setattr(
        data,
        "load_image_generation_manifest",
        lambda path: [SimpleNamespace(split="train", id="fixture-1")],
    )

    class DatasetFixture:
        data_fingerprint = "fixture-data"

        def __init__(self, *args, **kwargs):
            assert kwargs["target_size"] == (32, 32)
            assert kwargs["split"] == "train"

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "inputs": {"features": torch.tensor([[1.0, 2.0, 3.0]])},
                "targets": {"image": torch.full((1, 3), 1.0)},
                "native_context": {"image": {"reference_images": [[]]}},
                "output_specs": {"image": {"height": 32, "width": 32}},
            }

    monkeypatch.setattr(data, "ImageGenerationDataset", DatasetFixture)
    monkeypatch.setattr(data, "ImageGenerationCollator", lambda tokenizer: lambda rows: rows[0])
    output = tmp_path / "smoke"
    args = [argument for name, path in assets.items() for argument in ("--" + name, str(path))]
    assert (
        main(
            args
            + [
                "--output-dir",
                str(output),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--steps",
                "2",
                "--sampling-steps",
                "1",
                "--height",
                "32",
                "--width",
                "32",
                "--learning-rate",
                "0.01",
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["qualification"] == "unqualified"
    assert report["p2_gate"] == "not_evaluated"
    assert report["frozen_state_unchanged"] is True
    assert report["frozen_hashes_before"] == report["frozen_hashes_after"]
    assert "decoders.image.backend.frozen_counter" in report["frozen_hashes_before"]
    assert len(report["steps"]) == 2
    assert all(
        any(norm > 0 for norm in step["gradient_norms"].values()) for step in report["steps"]
    )
    assert report["fixed_probe_loss_after"] < report["fixed_probe_loss_before"]
    assert model.supervised_calls == 4  # before + two optimizer steps + after
    assert model.predict_calls == 1
    assert model.decoders["image"].backend.loaded
    assert report["generated_image"]["sampling_target_free"]
    assert (output / report["generated_image"]["path"]).is_file()
    changed = {
        key for key, value in model.state_dict().items() if not torch.equal(value, original[key])
    }
    assert changed and all(key.startswith("decoders.image.connector.") for key in changed)
    artifact = output / "connector-smoke.pt"
    saved = torch.load(artifact, weights_only=True)
    assert saved["evidence_kind"] == "real_checkpoint_optimization_smoke"
    assert saved["qualification"] == "unqualified"
    assert set(saved["connector_state_dict"]) == changed
    with pytest.raises(ValueError, match="accepted training artifact"):
        load_image_connector(
            model,
            artifact,
            parent_checkpoint_sha256="a" * 64,
            reference_checkpoint_sha256="b" * 64,
        )

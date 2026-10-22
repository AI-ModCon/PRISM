from types import SimpleNamespace

import pytest
import src.training.trainer_zone_a as trainer_module
import torch
import torch.nn as nn
from src.config import TrainingConfig
from src.training.trainer_zone_a import (
    ZoneATrainer,
    _configure_encoder_for_training,
    _labels_for_batch,
    _refresh_encoder_frozen_cache,
)


class _ProtectedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.codec = nn.Linear(2, 2).float()
        self.shared = nn.Linear(2, 2).float()
        self.dtype_casts = []

    def to(self, *args, **kwargs):
        if "dtype" in kwargs:
            self.dtype_casts.append(kwargs["dtype"])
        return super().to(*args, **kwargs)

    def set_shared_encoder_trainable(self, trainable):
        self.codec.float()
        self.codec.eval()
        for parameter in self.codec.parameters():
            parameter.requires_grad_(False)
        for parameter in self.shared.parameters():
            parameter.requires_grad_(trainable)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(modalities=["image"], is_interleaved_qa=False)
        self.backbone = None
        self.encoders = nn.ModuleDict({"image": _ProtectedEncoder()})
        self.projectors = nn.ModuleDict({"image": nn.Linear(2, 2)})
        self._encoder_frozen_cache = {"image": False}
        self.refresh_calls = 0

    def refresh_encoder_frozen_cache(self):
        self.refresh_calls += 1
        self._encoder_frozen_cache = {
            name: not any(parameter.requires_grad for parameter in encoder.parameters())
            for name, encoder in self.encoders.items()
        }
        return self._encoder_frozen_cache


class _FakeAccelerator:
    def __init__(self, *args, **kwargs):
        self.state = SimpleNamespace(deepspeed_plugin=None)
        self.is_main_process = False
        self.device = torch.device("cpu")
        # The scheduler expresses warmup/decay in optimizer steps by scaling
        # with the world size; single-process stub means 1.
        self.num_processes = 1

    def print(self, *args, **kwargs):
        return None

    def prepare(self, *items):
        return items if len(items) > 1 else items[0]


class _Dataset:
    active_modalities = ["image"]


def test_capability_aware_encoder_configuration_preserves_codec_fp32():
    encoder = _ProtectedEncoder()

    _configure_encoder_for_training(
        encoder,
        trainable=True,
        target_dtype=torch.bfloat16,
    )

    assert encoder.dtype_casts == []
    assert encoder.codec.weight.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in encoder.codec.parameters())
    assert all(parameter.requires_grad for parameter in encoder.shared.parameters())

    _configure_encoder_for_training(
        encoder,
        trainable=False,
        target_dtype=torch.bfloat16,
    )
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


@pytest.mark.parametrize("freeze_vit", [True, False])
def test_zone_a_optimizer_respects_shared_encoder_boundary(monkeypatch, freeze_vit):
    monkeypatch.setattr(trainer_module, "Accelerator", _FakeAccelerator)
    model = _Model()
    config = TrainingConfig(
        max_steps=1,
        warmup_steps=0,
        wandb_project=None,
        device="cpu",
        freeze_vit=freeze_vit,
    )
    loader = SimpleNamespace(dataset=_Dataset())

    trainer = ZoneATrainer(model, config, loader)

    assert model.refresh_calls == 1
    assert model.encoders["image"].codec.weight.dtype == torch.float32
    assert all(
        not parameter.requires_grad for parameter in model.encoders["image"].codec.parameters()
    )

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    expected = {"projectors.image.weight", "projectors.image.bias"}
    if not freeze_vit:
        expected |= {"encoders.image.shared.weight", "encoders.image.shared.bias"}
    assert trainable_names == expected

    vit_groups = [group for group in trainer.optimizer.param_groups if group["name"] == "vit"]
    if freeze_vit:
        assert vit_groups == []
        assert model._encoder_frozen_cache == {"image": True}
    else:
        assert len(vit_groups) == 1
        optimizer_ids = {id(parameter) for parameter in vit_groups[0]["params"]}
        shared_ids = {id(parameter) for parameter in model.encoders["image"].shared.parameters()}
        codec_ids = {id(parameter) for parameter in model.encoders["image"].codec.parameters()}
        assert optimizer_ids == shared_ids
        assert optimizer_ids.isdisjoint(codec_ids)
        assert model._encoder_frozen_cache == {"image": False}


def test_encoder_cache_fallback_and_assistant_only_labels():
    encoder = _ProtectedEncoder()
    model = SimpleNamespace(
        encoders={"image": encoder},
        _encoder_frozen_cache={"image": True},
    )
    _configure_encoder_for_training(encoder, trainable=True, target_dtype=torch.float32)
    _refresh_encoder_frozen_cache(model)
    assert model._encoder_frozen_cache == {"image": False}

    text = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[-100, -100, 3]])
    assert _labels_for_batch({"text": text, "labels": labels}) is labels
    assert _labels_for_batch({"text": text}) is text

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from src.config import ModelConfig
from src.encoders.aion_image import AION_MODEL_REVISION
from src.encoders.aion_spectrum import (
    AION_RAW_SPECTRUM_LENGTH,
    AION_SPECTRUM_CODEC_REVISION,
    AION_SPECTRUM_TOKENS,
    AION_SPECTRUM_WIDTH,
)
from src.modalities import Modality
from src.model import UnifiedTransformer

pytestmark = pytest.mark.unit


class _FakeAIONSpectrumEncoder(nn.Module):
    output_dim = AION_SPECTRUM_WIDTH

    def __init__(self, *, token_count: int = AION_SPECTRUM_TOKENS) -> None:
        super().__init__()
        self.shared_weight = nn.Parameter(torch.ones(1))
        self.codec_weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.token_count = token_count
        self.trainability_calls: list[bool] = []

    def tokens_per_instance(self) -> int:
        return self.token_count

    def set_shared_encoder_trainable(self, trainable: bool):
        self.trainability_calls.append(bool(trainable))
        self.shared_weight.requires_grad_(bool(trainable))
        self.codec_weight.requires_grad_(False)
        return self

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.shared_weight * torch.ones(
            inputs["flux"].shape[0],
            self.token_count,
            self.output_dim,
            device=inputs["flux"].device,
        )


def _spectrum_config(*, freeze_encoders: bool = True, **kwargs) -> ModelConfig:
    values = {
        "d_model": 32,
        "vocab_size": 64,
        "num_layers": 1,
        "num_heads": 4,
        "num_experts": 2,
        "modalities": ["spectrum"],
        "d_spectrum": AION_SPECTRUM_WIDTH,
        "spectrum_encoder_type": "aion",
        "freeze_encoders": freeze_encoders,
        "aion_model_name_or_path": "local/aion-base",
        "aion_cache_dir": "local/cache",
    }
    values.update(kwargs)
    return ModelConfig(**values)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"d_spectrum": AION_SPECTRUM_WIDTH - 1}, "d_spectrum=768"),
        (
            {"aion_num_spectrum_tokens": AION_SPECTRUM_TOKENS - 1},
            "exactly 273",
        ),
        (
            {"aion_raw_spectrum_length": AION_RAW_SPECTRUM_LENGTH - 1},
            "7781 samples",
        ),
        ({"spectrum_encoder_type": "generic"}, "must be 'aion'"),
    ],
)
def test_spectrum_config_rejects_incompatible_feature_contract(override, message):
    with pytest.raises(ValueError, match=message):
        _spectrum_config(**override)


def test_model_selects_aion_spectrum_with_its_own_768_adapter():
    encoder = _FakeAIONSpectrumEncoder()

    with patch("src.model.AIONSpectrumEncoder", return_value=encoder) as factory:
        model = UnifiedTransformer(_spectrum_config())

    factory.assert_called_once_with(
        model_name_or_path="local/aion-base",
        model_revision=AION_MODEL_REVISION,
        codec_revision=AION_SPECTRUM_CODEC_REVISION,
        cache_dir="local/cache",
        local_files_only=True,
        raw_spectrum_length=AION_RAW_SPECTRUM_LENGTH,
    )
    assert model.encoders["spectrum"] is encoder
    assert model.projectors["spectrum"].fc1.in_features == AION_SPECTRUM_WIDTH
    assert encoder.trainability_calls == [False]
    assert model._encoder_frozen_cache == {"spectrum": True}


def test_spectrum_encoder_hook_never_unfreezes_the_codec():
    encoder = _FakeAIONSpectrumEncoder()

    with patch("src.model.AIONSpectrumEncoder", return_value=encoder):
        model = UnifiedTransformer(_spectrum_config(freeze_encoders=False))

    assert encoder.trainability_calls == [True]
    assert encoder.shared_weight.requires_grad
    assert not encoder.codec_weight.requires_grad
    assert model._encoder_frozen_cache == {"spectrum": False}

    assert model.set_encoder_trainable("spectrum", False) == 0
    assert not encoder.shared_weight.requires_grad
    assert not encoder.codec_weight.requires_grad
    assert model._encoder_frozen_cache == {"spectrum": True}

    assert model.set_encoder_trainable("spectrum", True) == encoder.shared_weight.numel()
    assert encoder.shared_weight.requires_grad
    assert not encoder.codec_weight.requires_grad
    assert model._encoder_frozen_cache == {"spectrum": False}


def test_model_rejects_aion_spectrum_encoder_with_unexpected_token_count():
    encoder = _FakeAIONSpectrumEncoder(token_count=AION_SPECTRUM_TOKENS - 1)
    with (
        patch("src.model.AIONSpectrumEncoder", return_value=encoder),
        pytest.raises(RuntimeError, match="spectrum token count"),
    ):
        UnifiedTransformer(_spectrum_config())


def test_config_accepts_joint_aion_with_explicit_projector_overrides():
    config = ModelConfig(
        modalities=["image", "spectrum"],
        image_encoder_type="aion",
        spectrum_encoder_type="aion",
        d_img=768,
        image_projector_norm_mode="none",
        image_projector_modality_embed_pos="after_norm",
        spectrum_projector_norm_mode="text_scaled_layernorm",
        spectrum_projector_target_norm=1.04,
        spectrum_projector_modality_embed_pos="before_norm",
    )

    assert config.modalities == [Modality.IMAGE, Modality.SPECTRUM]
    assert config.image_projector_norm_mode == "none"
    assert config.image_projector_modality_embed_pos == "after_norm"
    assert config.spectrum_projector_norm_mode == "text_scaled_layernorm"
    assert config.spectrum_projector_target_norm == pytest.approx(1.04)
    assert config.spectrum_projector_modality_embed_pos == "before_norm"

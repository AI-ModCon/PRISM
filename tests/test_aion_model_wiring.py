from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import yaml
from src.config import ModelConfig
from src.encoders.aion_image import (
    AION_IMAGE_CODEC_REVISION,
    AION_IMAGE_TOKENS,
    AION_IMAGE_WIDTH,
    AION_LEGACY_SURVEY_BANDS,
    AION_MODEL_REVISION,
    AION_REPO_ID,
)
from src.model import UnifiedTransformer

pytestmark = pytest.mark.unit


class _FakeAIONImageEncoder(nn.Module):
    output_dim = AION_IMAGE_WIDTH

    def __init__(self, *, token_count: int = AION_IMAGE_TOKENS) -> None:
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.shared_weight * torch.ones(
            inputs.shape[0],
            self.token_count,
            self.output_dim,
            device=inputs.device,
        )


def _aion_config(*, freeze_encoders: bool = True, **kwargs) -> ModelConfig:
    values = {
        "d_model": 32,
        "vocab_size": 64,
        "num_layers": 1,
        "num_heads": 4,
        "num_experts": 2,
        "modalities": ["image"],
        "d_img": AION_IMAGE_WIDTH,
        "image_encoder_type": "aion",
        "freeze_encoders": freeze_encoders,
        "aion_model_name_or_path": "local/aion-base",
        "aion_cache_dir": "local/cache",
    }
    values.update(kwargs)
    return ModelConfig(**values)


def test_siglip_remains_the_default_image_encoder_type():
    assert ModelConfig().image_encoder_type == "siglip"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"d_img": AION_IMAGE_WIDTH - 1}, "d_img=768"),
        ({"aion_num_image_tokens": AION_IMAGE_TOKENS - 1}, "exactly 576"),
        (
            {"aion_image_bands": ("DES-R", "DES-G", "DES-I", "DES-Z")},
            "Legacy Survey channels",
        ),
        ({"aion_raw_image_size": (96, 96)}, "160x160"),
    ],
)
def test_aion_config_rejects_incompatible_feature_contract(override, message):
    with pytest.raises(ValueError, match=message):
        _aion_config(**override)


def test_model_selects_aion_with_pinned_offline_controls_and_768_adapter():
    encoder = _FakeAIONImageEncoder()

    with patch("src.model.AIONImageEncoder", return_value=encoder) as factory:
        model = UnifiedTransformer(_aion_config())

    factory.assert_called_once_with(
        model_name_or_path="local/aion-base",
        model_revision=AION_MODEL_REVISION,
        codec_revision=AION_IMAGE_CODEC_REVISION,
        cache_dir="local/cache",
        local_files_only=True,
        bands=AION_LEGACY_SURVEY_BANDS,
        raw_image_size=(160, 160),
    )
    assert model.encoders["image"] is encoder
    assert model.projectors["image"].fc1.in_features == AION_IMAGE_WIDTH
    assert encoder.trainability_calls == [False]
    assert model._encoder_frozen_cache == {"image": True}


def test_model_trainability_hook_never_unfreezes_the_aion_codec():
    encoder = _FakeAIONImageEncoder()

    with patch("src.model.AIONImageEncoder", return_value=encoder):
        model = UnifiedTransformer(_aion_config(freeze_encoders=False))

    assert encoder.trainability_calls == [True]
    assert encoder.shared_weight.requires_grad
    assert not encoder.codec_weight.requires_grad
    assert model._encoder_frozen_cache == {"image": False}

    assert model.set_encoder_trainable("image", False) == 0
    assert not encoder.shared_weight.requires_grad
    assert not encoder.codec_weight.requires_grad
    assert model._encoder_frozen_cache == {"image": True}

    assert model.set_encoder_trainable("image", True) == encoder.shared_weight.numel()
    assert encoder.shared_weight.requires_grad
    assert not encoder.codec_weight.requires_grad
    assert model._encoder_frozen_cache == {"image": False}


def test_refresh_encoder_frozen_cache_tracks_external_hook_changes():
    encoder = _FakeAIONImageEncoder()
    with patch("src.model.AIONImageEncoder", return_value=encoder):
        model = UnifiedTransformer(_aion_config())

    encoder.set_shared_encoder_trainable(True)
    assert model._encoder_frozen_cache == {"image": True}
    assert model.refresh_encoder_frozen_cache() == {"image": False}


def test_model_rejects_aion_encoder_with_unexpected_token_count():
    encoder = _FakeAIONImageEncoder(token_count=AION_IMAGE_TOKENS - 1)
    with (
        patch("src.model.AIONImageEncoder", return_value=encoder),
        pytest.raises(RuntimeError, match="token count"),
    ):
        UnifiedTransformer(_aion_config())


def test_astroprism_hydra_preset_is_raw_aion_qwen3_4b():
    config_path = Path("src/conf/model/prism_astroprism_qwen3_4b.yaml")
    cfg = yaml.safe_load(config_path.read_text())

    asset_root = "${oc.env:ASTROPRISM_ROOT,_local/astroprism}"
    assert cfg["backbone_id"] == f"{asset_root}/hf/models/qwen3-4b"
    assert cfg["tokenizer_id"] == f"{asset_root}/hf/models/qwen3-4b"
    assert cfg["image_encoder_type"] == "aion"
    assert cfg["d_img"] == AION_IMAGE_WIDTH
    assert cfg["aion_model_name_or_path"] == AION_REPO_ID
    # null means the ambient HF cache. huggingface_hub's cache_dir is the hub
    # directory itself, unlike HF_HOME which has hub/ appended, so a path here
    # resolved one level too high and never found the weights. The released
    # checkpoint's training_config.yaml records null for the same reason.
    assert cfg["aion_cache_dir"] is None
    assert cfg["aion_model_revision"] == AION_MODEL_REVISION
    assert cfg["aion_codec_revision"] == AION_IMAGE_CODEC_REVISION
    assert cfg["aion_local_files_only"] is True
    assert tuple(cfg["aion_image_bands"]) == AION_LEGACY_SURVEY_BANDS
    assert cfg["aion_num_image_tokens"] == AION_IMAGE_TOKENS
    assert cfg["aion_raw_image_size"] == [160, 160]
    assert cfg["image_processor_id"] is None
    assert cfg["projector_norm_mode"] == "none"
    assert cfg["projector_modality_embed_pos"] == "after_norm"

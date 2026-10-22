from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import src.encoders.aion_multimodal as aion_multimodal_module
import torch
import torch.nn as nn
from src.config import ModelConfig
from src.encoders.aion_image import (
    AION_IMAGE_CODEC_REVISION,
    AION_IMAGE_TOKENS,
    AION_IMAGE_WIDTH,
    AION_MODEL_REVISION,
)
from src.encoders.aion_multimodal import AIONMultimodalEncoder
from src.encoders.aion_spectrum import (
    AION_RAW_SPECTRUM_LENGTH,
    AION_SPECTRUM_CODEC_REVISION,
    AION_SPECTRUM_TOKENS,
)
from src.model import UnifiedTransformer

pytestmark = pytest.mark.unit


@dataclass
class _FakeLegacySurveyImage:
    flux: torch.Tensor
    bands: list[str]

    token_key = "tok_image"


@dataclass
class _FakeDESISpectrum:
    flux: torch.Tensor
    ivar: torch.Tensor
    mask: torch.Tensor
    wavelength: torch.Tensor

    token_key = "tok_spectrum_desi"


class _FakeImageCodec(nn.Module):
    from_pretrained_calls: list[tuple[tuple, dict]] = []

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=torch.float64))
        self.last_image: _FakeLegacySurveyImage | None = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def encode(self, image: _FakeLegacySurveyImage) -> torch.Tensor:
        self.last_image = image
        image.flux.zero_()
        return torch.zeros(
            image.flux.shape[0],
            AION_IMAGE_TOKENS,
            dtype=torch.long,
            device=image.flux.device,
        )


class _FakeSpectrumCodec(nn.Module):
    from_pretrained_calls: list[tuple[tuple, dict]] = []

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=torch.float64))
        self.last_spectrum: _FakeDESISpectrum | None = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def encode(self, spectrum: _FakeDESISpectrum) -> torch.Tensor:
        self.last_spectrum = spectrum
        spectrum.flux.zero_()
        return torch.zeros(
            spectrum.flux.shape[0],
            AION_SPECTRUM_TOKENS,
            dtype=torch.long,
            device=spectrum.flux.device,
        )


class _FakeAION(nn.Module):
    from_pretrained_calls: list[tuple[tuple, dict]] = []

    def __init__(self) -> None:
        super().__init__()
        self.encoder_embeddings = nn.ModuleDict(
            {
                "tok_image": nn.Embedding(8, 1),
                "tok_spectrum_desi": nn.Embedding(8, 1),
                "tok_unused": nn.Embedding(8, 1),
            }
        )
        self.encoder = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        self.encoder_norm = nn.LayerNorm(1)
        self.decoder_proj_context = nn.Linear(1, 1, bias=False)
        self.unused_decoder = nn.Linear(1, 1)
        self.encode_calls: list[tuple[dict[str, torch.Tensor], int]] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def encode(self, inputs: dict[str, torch.Tensor], *, num_encoder_tokens: int):
        self.encode_calls.append((inputs, num_encoder_tokens))
        batch_size = next(iter(inputs.values())).shape[0]
        scale = self.encoder[0].weight.reshape(1, 1, 1)
        return scale * torch.ones(
            batch_size,
            num_encoder_tokens,
            AION_IMAGE_WIDTH,
            dtype=scale.dtype,
            device=scale.device,
        )


@pytest.fixture(autouse=True)
def _mock_joint_aion_types(monkeypatch):
    _FakeAION.from_pretrained_calls.clear()
    _FakeImageCodec.from_pretrained_calls.clear()
    _FakeSpectrumCodec.from_pretrained_calls.clear()
    monkeypatch.setattr(
        aion_multimodal_module,
        "_load_aion_types",
        lambda: (
            _FakeAION,
            _FakeImageCodec,
            _FakeSpectrumCodec,
            _FakeLegacySurveyImage,
            _FakeDESISpectrum,
        ),
    )


def _make_spectrum_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    wavelength = torch.linspace(3600.0, 9800.0, AION_RAW_SPECTRUM_LENGTH)
    return {
        "flux": torch.randn(batch_size, AION_RAW_SPECTRUM_LENGTH),
        "ivar": torch.rand(batch_size, AION_RAW_SPECTRUM_LENGTH),
        "mask": torch.zeros(batch_size, AION_RAW_SPECTRUM_LENGTH, dtype=torch.bool),
        "wavelength": wavelength.repeat(batch_size, 1),
    }


def _make_encoder(**kwargs) -> AIONMultimodalEncoder:
    return AIONMultimodalEncoder(model_name_or_path="local/aion-base", **kwargs)


def _joint_config(**kwargs) -> ModelConfig:
    values = {
        "d_model": 32,
        "vocab_size": 64,
        "num_layers": 1,
        "num_heads": 4,
        "num_experts": 2,
        "modalities": ["image", "spectrum"],
        "d_img": AION_IMAGE_WIDTH,
        "d_spectrum": AION_IMAGE_WIDTH,
        "image_encoder_type": "aion",
        "spectrum_encoder_type": "aion",
        "freeze_encoders": True,
        "aion_model_name_or_path": "local/aion-base",
        "aion_cache_dir": "local/cache",
        "image_projector_norm_mode": "none",
        "image_projector_modality_embed_pos": "after_norm",
        "spectrum_projector_norm_mode": "text_scaled_layernorm",
        "spectrum_projector_target_norm": 1.04,
        "spectrum_projector_modality_embed_pos": "before_norm",
    }
    values.update(kwargs)
    return ModelConfig(**values)


def test_joint_encoder_loads_one_transformer_and_two_pinned_codecs(tmp_path: Path):
    encoder = _make_encoder(cache_dir=tmp_path)

    assert _FakeAION.from_pretrained_calls == [
        (
            ("local/aion-base",),
            {
                "revision": AION_MODEL_REVISION,
                "cache_dir": str(tmp_path),
                "local_files_only": True,
            },
        )
    ]
    assert _FakeImageCodec.from_pretrained_calls[0][1]["revision"] == (
        AION_IMAGE_CODEC_REVISION
    )
    assert _FakeImageCodec.from_pretrained_calls[0][1]["modality"] is (
        _FakeLegacySurveyImage
    )
    assert _FakeSpectrumCodec.from_pretrained_calls[0][1]["revision"] == (
        AION_SPECTRUM_CODEC_REVISION
    )
    assert _FakeSpectrumCodec.from_pretrained_calls[0][1]["modality"] is (
        _FakeDESISpectrum
    )
    assert encoder.tokens_per_instance("image") == AION_IMAGE_TOKENS
    assert encoder.tokens_per_instance("spectrum") == AION_SPECTRUM_TOKENS


def test_joint_encoder_routes_both_modalities_without_mutating_raw_inputs():
    encoder = _make_encoder()
    image = torch.randn(2, 4, 160, 160)
    image_original = image.clone()
    spectrum = _make_spectrum_inputs()
    spectrum_original = {key: value.clone() for key, value in spectrum.items()}

    image_features = encoder.encode_modality("image", image)
    spectrum_features = encoder.encode_modality("spectrum", spectrum)

    assert image_features.shape == (2, AION_IMAGE_TOKENS, AION_IMAGE_WIDTH)
    assert spectrum_features.shape == (2, AION_SPECTRUM_TOKENS, AION_IMAGE_WIDTH)
    assert torch.equal(image, image_original)
    assert all(torch.equal(spectrum[key], spectrum_original[key]) for key in spectrum)
    assert [set(call[0]) for call in encoder.aion_model.encode_calls] == [
        {"tok_image"},
        {"tok_spectrum_desi"},
    ]
    assert [call[1] for call in encoder.aion_model.encode_calls] == [
        AION_IMAGE_TOKENS,
        AION_SPECTRUM_TOKENS,
    ]


def test_joint_encoder_registers_shared_transformer_once_and_protects_codecs():
    encoder = _make_encoder().to(dtype=torch.bfloat16)
    encoder.set_shared_encoder_trainable(True).train()

    state_keys = set(encoder.state_dict())
    assert "aion_model.encoder.0.weight" in state_keys
    assert sum(key.endswith("aion_model.encoder.0.weight") for key in state_keys) == 1
    assert not any("image_encoder.aion_model" in key for key in state_keys)
    assert not any("spectrum_encoder.aion_model" in key for key in state_keys)

    for codec in (encoder.image_codec, encoder.spectrum_codec):
        parameters = list(codec.parameters())
        assert parameters
        assert all(parameter.dtype == torch.float32 for parameter in parameters)
        assert all(not parameter.requires_grad for parameter in parameters)
        assert codec.training is False

    assert all(
        any(parameter.requires_grad for parameter in module.parameters())
        for module in encoder._shared_encoder_modules()
    )
    assert all(module.training for module in encoder._shared_encoder_modules())
    assert all(
        not parameter.requires_grad
        for parameter in encoder.aion_model.encoder_embeddings["tok_unused"].parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in encoder.aion_model.unused_decoder.parameters()
    )

    encoder.encode_modality("image", torch.randn(1, 4, 160, 160)).sum().backward()
    assert encoder.aion_model.encoder[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in encoder.image_codec.parameters())
    assert all(parameter.grad is None for parameter in encoder.spectrum_codec.parameters())

    encoder.set_shared_encoder_trainable(False).train()
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert encoder.image_codec.training is False
    assert encoder.spectrum_codec.training is False


def test_joint_model_owns_one_aion_and_keeps_adapter_contracts_separate():
    model = UnifiedTransformer(_joint_config())

    assert list(model.encoders) == ["aion"]
    assert model._encoder_key_by_modality == {
        "image": "aion",
        "spectrum": "aion",
    }
    assert model._encoder_frozen_cache == {"image": True, "spectrum": True}
    assert len(_FakeAION.from_pretrained_calls) == 1

    image_projector = model.projectors["image"]
    spectrum_projector = model.projectors["spectrum"]
    assert image_projector.norm_mode == "none"
    assert image_projector.modality_embed_pos == "after_norm"
    assert image_projector.final_norm is None
    assert spectrum_projector.norm_mode == "text_scaled_layernorm"
    assert spectrum_projector.target_norm == pytest.approx(1.04)
    assert spectrum_projector.modality_embed_pos == "before_norm"
    assert spectrum_projector.final_norm is not None

    [(image_name, image_features)] = model._process_multimodal_embeddings(
        {"image": torch.randn(1, 4, 160, 160)}
    )
    [(spectrum_name, spectrum_features)] = model._process_multimodal_embeddings(
        {"spectrum": _make_spectrum_inputs(batch_size=1)}
    )
    assert str(image_name) == "image"
    assert image_features.shape == (1, AION_IMAGE_TOKENS, 32)
    assert str(spectrum_name) == "spectrum"
    assert spectrum_features.shape == (1, AION_SPECTRUM_TOKENS, 32)

    state_keys = set(model.state_dict())
    assert sum(key.endswith("aion_model.encoder.0.weight") for key in state_keys) == 1
    assert "encoders.aion.aion_model.encoder.0.weight" in state_keys


def test_joint_model_trainability_aliases_toggle_the_one_owner():
    model = UnifiedTransformer(_joint_config())
    encoder = model.encoders["aion"]

    trainable_count = model.set_encoder_trainable("image", True)
    assert trainable_count > 0
    assert encoder.shared_encoder_trainable is True
    assert model._encoder_frozen_cache == {"image": False, "spectrum": False}
    assert all(not parameter.requires_grad for parameter in encoder.image_codec.parameters())
    assert all(
        not parameter.requires_grad for parameter in encoder.spectrum_codec.parameters()
    )

    assert model.set_encoder_trainable("spectrum", False) == 0
    assert encoder.shared_encoder_trainable is False
    assert model._encoder_frozen_cache == {"image": True, "spectrum": True}


def test_per_modality_projector_fields_fall_back_to_global_defaults():
    config = _joint_config(
        projector_norm_mode="layernorm",
        projector_target_norm=0.5,
        projector_modality_embed_pos="none",
        image_projector_norm_mode=None,
        image_projector_modality_embed_pos=None,
        spectrum_projector_norm_mode=None,
        spectrum_projector_target_norm=None,
        spectrum_projector_modality_embed_pos=None,
    )
    model = UnifiedTransformer(config)

    for modality in ("image", "spectrum"):
        projector = model.projectors[modality]
        assert projector.norm_mode == "layernorm"
        assert projector.target_norm == pytest.approx(0.5)
        assert projector.modality_embed_pos == "none"

@pytest.mark.parametrize(
    ("modality", "start_idx", "end_idx", "token_count"),
    [
        ("image", 60, 61, AION_IMAGE_TOKENS),
        ("spectrum", 62, 63, AION_SPECTRUM_TOKENS),
    ],
)
def test_joint_merge_routes_token_count_through_shared_owner(
    monkeypatch, modality: str, start_idx: int, end_idx: int, token_count: int
):
    model = UnifiedTransformer(_joint_config())
    model.config.modality_start_end_token_indices = {
        modality: (start_idx, end_idx)
    }
    owner = model.encoders["aion"]
    calls: list[str] = []
    original_tokens_per_instance = owner.tokens_per_instance

    def record_tokens_per_instance(routed_modality: str) -> int:
        calls.append(routed_modality)
        return original_tokens_per_instance(routed_modality)

    monkeypatch.setattr(owner, "tokens_per_instance", record_tokens_per_instance)

    input_ids = torch.tensor([[5, start_idx, end_idx, 6]], dtype=torch.long)
    input_embeds = torch.arange(4 * 32, dtype=torch.float32).reshape(1, 4, 32)
    modality_features = torch.full((1, token_count, 32), 7.0)

    final_embedding, attention_mask, position_ids, labels = (
        model._merge_text_input_ids_with_modality_embeds(
            input_ids=input_ids,
            input_embeds=input_embeds,
            embeddings_list=[
                ("text", input_embeds),
                (modality, modality_features),
            ],
            metadata=["4 0"],
        )
    )

    assert calls == [modality, modality]
    assert final_embedding.shape == (1, token_count + 2, 32)
    assert torch.equal(final_embedding[:, 1 : token_count + 1], modality_features)
    assert torch.equal(final_embedding[:, 0], input_embeds[:, 0])
    assert torch.equal(final_embedding[:, -1], input_embeds[:, -1])
    assert attention_mask.tolist() == [[1] * (token_count + 2)]
    assert position_ids.tolist() == [list(range(token_count + 2))]
    assert torch.all(labels == -100)


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import src.encoders.aion_image as aion_image_module
import torch
import torch.nn as nn
from src.encoders.aion_image import (
    AION_IMAGE_CODEC_REVISION,
    AION_IMAGE_TOKENS,
    AION_IMAGE_WIDTH,
    AION_LEGACY_SURVEY_BANDS,
    AION_MODEL_REVISION,
    AIONImageEncoder,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeLegacySurveyImage:
    flux: torch.Tensor
    bands: list[str]

    token_key = "tok_image"


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
        batch_size = image.flux.shape[0]
        # Deliberately mimic AION's in-place preprocessing to prove the wrapper
        # cloned the caller's tensor.
        image.flux.zero_()
        return torch.zeros(
            batch_size, AION_IMAGE_TOKENS, dtype=torch.long, device=image.flux.device
        )


class _FakeAION(nn.Module):
    from_pretrained_calls: list[tuple[tuple, dict]] = []

    def __init__(self) -> None:
        super().__init__()
        self.encoder_embeddings = nn.ModuleDict({"tok_image": nn.Embedding(8, 1)})
        self.encoder = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        self.encoder_norm = nn.LayerNorm(1)
        self.decoder_proj_context = nn.Linear(1, 1, bias=False)
        self.unused_decoder = nn.Linear(1, 1)
        self.encode_calls: list[tuple[dict[str, torch.Tensor], int]] = []
        self.output_shape: tuple[int, int] = (AION_IMAGE_TOKENS, AION_IMAGE_WIDTH)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def encode(self, inputs: dict[str, torch.Tensor], *, num_encoder_tokens: int):
        self.encode_calls.append((inputs, num_encoder_tokens))
        batch_size = next(iter(inputs.values())).shape[0]
        token_count, width = self.output_shape
        # Keep the output connected to the shared encoder for the unfreeze test.
        scale = self.encoder[0].weight.reshape(1, 1, 1)
        return scale * torch.ones(
            batch_size,
            token_count,
            width,
            dtype=scale.dtype,
            device=scale.device,
        )


@pytest.fixture(autouse=True)
def _mock_aion_types(monkeypatch):
    _FakeAION.from_pretrained_calls.clear()
    _FakeImageCodec.from_pretrained_calls.clear()
    monkeypatch.setattr(
        aion_image_module,
        "_load_aion_types",
        lambda: (_FakeAION, _FakeImageCodec, _FakeLegacySurveyImage),
    )


def _make_encoder(**kwargs) -> AIONImageEncoder:
    return AIONImageEncoder(model_name_or_path="local/aion-base", **kwargs)


def test_loads_model_and_codec_with_explicit_pinned_offline_controls(tmp_path: Path):
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
    assert _FakeImageCodec.from_pretrained_calls == [
        (
            ("local/aion-base",),
            {
                "modality": _FakeLegacySurveyImage,
                "revision": AION_IMAGE_CODEC_REVISION,
                "cache_dir": str(tmp_path),
                "local_files_only": True,
            },
        )
    ]
    assert encoder.tokens_per_instance() == AION_IMAGE_TOKENS
    assert encoder.output_dim == AION_IMAGE_WIDTH


def test_raw_image_is_cloned_and_encoded_as_exactly_576_aion_tokens():
    encoder = _make_encoder()
    raw = torch.randn(2, 4, 160, 160)
    original = raw.clone()

    features = encoder(raw)

    assert features.shape == (2, AION_IMAGE_TOKENS, AION_IMAGE_WIDTH)
    assert torch.equal(raw, original)
    assert encoder.image_codec.last_image is not None
    assert encoder.image_codec.last_image.bands == list(AION_LEGACY_SURVEY_BANDS)
    assert torch.count_nonzero(encoder.image_codec.last_image.flux) == 0

    [(encoded_inputs, token_budget)] = encoder.aion_model.encode_calls
    assert set(encoded_inputs) == {"tok_image"}
    assert encoded_inputs["tok_image"].shape == (2, AION_IMAGE_TOKENS)
    assert token_budget == AION_IMAGE_TOKENS


def test_codec_stays_float32_frozen_and_eval_when_encoder_is_unfrozen():
    encoder = _make_encoder().to(dtype=torch.bfloat16)
    encoder.set_shared_encoder_trainable(True)
    encoder.train()

    codec_parameters = list(encoder.image_codec.parameters())
    assert codec_parameters
    assert all(parameter.dtype == torch.float32 for parameter in codec_parameters)
    assert all(not parameter.requires_grad for parameter in codec_parameters)
    assert encoder.image_codec.training is False

    shared_modules = encoder._shared_encoder_modules()
    assert all(
        any(parameter.requires_grad for parameter in module.parameters())
        for module in shared_modules
    )
    assert all(module.training for module in shared_modules)
    assert all(
        not parameter.requires_grad for parameter in encoder.aion_model.unused_decoder.parameters()
    )

    features = encoder(torch.randn(1, 4, 160, 160))
    features.sum().backward()
    assert encoder.aion_model.encoder[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in codec_parameters)

    encoder.set_shared_encoder_trainable(False)
    encoder.train()
    assert all(not parameter.requires_grad for parameter in encoder.aion_model.parameters())
    assert all(not module.training for module in shared_modules)
    assert encoder.image_codec.training is False


@pytest.mark.parametrize(
    "bad_input,exception,message",
    [
        (torch.zeros(4, 160, 160), ValueError, "shape"),
        (torch.zeros(1, 3, 160, 160), ValueError, "shape"),
        (torch.zeros(1, 4, 96, 96), ValueError, "shape"),
        (torch.zeros(1, 4, 160, 160, dtype=torch.long), TypeError, "floating point"),
        (
            torch.full((1, 4, 160, 160), float("nan")),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_rejects_invalid_raw_image_inputs(bad_input, exception, message):
    encoder = _make_encoder()
    with pytest.raises(exception, match=message):
        encoder(bad_input)


def test_rejects_unverified_band_order():
    with pytest.raises(ValueError, match="g/r/i/z"):
        _make_encoder(bands=("DES-R", "DES-G", "DES-I", "DES-Z"))


def test_rejects_unexpected_codec_token_shape():
    encoder = _make_encoder()

    def wrong_shape(_image):
        return torch.zeros(1, AION_IMAGE_TOKENS - 1, dtype=torch.long)

    encoder.image_codec.encode = wrong_shape
    with pytest.raises(RuntimeError, match="ImageCodec must emit token ids"):
        encoder(torch.randn(1, 4, 160, 160))


def test_rejects_unexpected_shared_encoder_feature_shape():
    encoder = _make_encoder()
    encoder.aion_model.output_shape = (AION_IMAGE_TOKENS, AION_IMAGE_WIDTH - 1)

    with pytest.raises(RuntimeError, match="shared encoder must emit features"):
        encoder(torch.randn(1, 4, 160, 160))

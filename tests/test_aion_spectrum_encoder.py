from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import src.encoders.aion_spectrum as aion_spectrum_module
import torch
import torch.nn as nn
from src.encoders.aion_image import AION_MODEL_REVISION
from src.encoders.aion_spectrum import (
    AION_RAW_SPECTRUM_LENGTH,
    AION_SPECTRUM_CODEC_REVISION,
    AION_SPECTRUM_TOKENS,
    AION_SPECTRUM_WIDTH,
    AIONSpectrumEncoder,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeDESISpectrum:
    flux: torch.Tensor
    ivar: torch.Tensor
    mask: torch.Tensor
    wavelength: torch.Tensor

    token_key = "tok_spectrum_desi"


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
        batch_size = spectrum.flux.shape[0]
        # Mimic in-place preprocessing to prove the wrapper isolates callers.
        spectrum.flux.zero_()
        spectrum.ivar.zero_()
        spectrum.mask.fill_(True)
        spectrum.wavelength.zero_()
        return torch.zeros(
            batch_size,
            AION_SPECTRUM_TOKENS,
            dtype=torch.long,
            device=spectrum.flux.device,
        )


class _FakeAION(nn.Module):
    from_pretrained_calls: list[tuple[tuple, dict]] = []

    def __init__(self) -> None:
        super().__init__()
        self.encoder_embeddings = nn.ModuleDict({"tok_spectrum_desi": nn.Embedding(8, 1)})
        self.encoder = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        self.encoder_norm = nn.LayerNorm(1)
        self.decoder_proj_context = nn.Linear(1, 1, bias=False)
        self.unused_decoder = nn.Linear(1, 1)
        self.encode_calls: list[tuple[dict[str, torch.Tensor], int]] = []
        self.output_shape = (AION_SPECTRUM_TOKENS, AION_SPECTRUM_WIDTH)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def encode(self, inputs: dict[str, torch.Tensor], *, num_encoder_tokens: int):
        self.encode_calls.append((inputs, num_encoder_tokens))
        batch_size = next(iter(inputs.values())).shape[0]
        token_count, width = self.output_shape
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
    _FakeSpectrumCodec.from_pretrained_calls.clear()
    monkeypatch.setattr(
        aion_spectrum_module,
        "_load_aion_types",
        lambda: (_FakeAION, _FakeSpectrumCodec, _FakeDESISpectrum),
    )


def _make_encoder(**kwargs) -> AIONSpectrumEncoder:
    return AIONSpectrumEncoder(model_name_or_path="local/aion-base", **kwargs)


def _make_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    wavelength = torch.linspace(3600.0, 9800.0, AION_RAW_SPECTRUM_LENGTH)
    return {
        "flux": torch.randn(batch_size, AION_RAW_SPECTRUM_LENGTH),
        "ivar": torch.rand(batch_size, AION_RAW_SPECTRUM_LENGTH),
        "mask": torch.zeros(batch_size, AION_RAW_SPECTRUM_LENGTH, dtype=torch.bool),
        "wavelength": wavelength.repeat(batch_size, 1),
    }


def test_loads_model_and_spectrum_codec_with_pinned_offline_controls(tmp_path: Path):
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
    assert _FakeSpectrumCodec.from_pretrained_calls == [
        (
            ("local/aion-base",),
            {
                "modality": _FakeDESISpectrum,
                "revision": AION_SPECTRUM_CODEC_REVISION,
                "cache_dir": str(tmp_path),
                "local_files_only": True,
            },
        )
    ]
    assert encoder.tokens_per_instance() == AION_SPECTRUM_TOKENS
    assert encoder.output_dim == AION_SPECTRUM_WIDTH


def test_raw_spectrum_is_cloned_and_encoded_as_exactly_273_aion_tokens():
    encoder = _make_encoder()
    inputs = _make_inputs()
    originals = {name: tensor.clone() for name, tensor in inputs.items()}

    features = encoder(inputs)

    assert features.shape == (2, AION_SPECTRUM_TOKENS, AION_SPECTRUM_WIDTH)
    for name in inputs:
        assert torch.equal(inputs[name], originals[name])

    spectrum = encoder.spectrum_codec.last_spectrum
    assert spectrum is not None
    assert spectrum.flux.dtype == torch.float32
    assert spectrum.ivar.dtype == torch.float32
    assert spectrum.wavelength.dtype == torch.float32
    assert spectrum.mask.dtype == torch.bool
    assert torch.count_nonzero(spectrum.flux) == 0

    [(encoded_inputs, token_budget)] = encoder.aion_model.encode_calls
    assert set(encoded_inputs) == {"tok_spectrum_desi"}
    assert encoded_inputs["tok_spectrum_desi"].shape == (2, AION_SPECTRUM_TOKENS)
    assert token_budget == AION_SPECTRUM_TOKENS


def test_codec_stays_float32_frozen_and_eval_when_shared_encoder_is_unfrozen():
    encoder = _make_encoder().to(dtype=torch.bfloat16)
    encoder.set_shared_encoder_trainable(True)
    encoder.train()

    codec_parameters = list(encoder.spectrum_codec.parameters())
    assert codec_parameters
    assert all(parameter.dtype == torch.float32 for parameter in codec_parameters)
    assert all(not parameter.requires_grad for parameter in codec_parameters)
    assert encoder.spectrum_codec.training is False

    shared_modules = encoder._shared_encoder_modules()
    assert all(
        any(parameter.requires_grad for parameter in module.parameters())
        for module in shared_modules
    )
    assert all(module.training for module in shared_modules)
    assert all(
        not parameter.requires_grad for parameter in encoder.aion_model.unused_decoder.parameters()
    )

    encoder(_make_inputs(batch_size=1)).sum().backward()
    assert encoder.aion_model.encoder[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in codec_parameters)

    encoder.set_shared_encoder_trainable(False)
    encoder.train()
    assert all(not parameter.requires_grad for parameter in encoder.aion_model.parameters())
    assert all(not module.training for module in shared_modules)
    assert encoder.spectrum_codec.training is False


def test_rejects_missing_or_malformed_spectrum_fields():
    encoder = _make_encoder()
    inputs = _make_inputs(batch_size=1)

    missing = dict(inputs)
    missing.pop("ivar")
    with pytest.raises(ValueError, match="missing fields"):
        encoder(missing)

    wrong_shape = dict(inputs)
    wrong_shape["flux"] = torch.zeros(1, AION_RAW_SPECTRUM_LENGTH - 1)
    with pytest.raises(ValueError, match="must have shape"):
        encoder(wrong_shape)

    wrong_mask = dict(inputs)
    wrong_mask["mask"] = inputs["mask"].float()
    with pytest.raises(TypeError, match="mask.*boolean"):
        encoder(wrong_mask)

    negative_ivar = dict(inputs)
    negative_ivar["ivar"] = -torch.ones_like(inputs["ivar"])
    with pytest.raises(ValueError, match="non-negative"):
        encoder(negative_ivar)

    nonmonotonic = dict(inputs)
    nonmonotonic["wavelength"] = inputs["wavelength"].flip(-1)
    with pytest.raises(ValueError, match="strictly increasing"):
        encoder(nonmonotonic)

    nonfinite = dict(inputs)
    nonfinite["flux"] = inputs["flux"].clone()
    nonfinite["flux"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        encoder(nonfinite)


def test_rejects_unexpected_codec_token_shape():
    encoder = _make_encoder()
    encoder.spectrum_codec.encode = lambda _spectrum: torch.zeros(
        1, AION_SPECTRUM_TOKENS - 1, dtype=torch.long
    )

    with pytest.raises(RuntimeError, match="SpectrumCodec must emit token ids"):
        encoder(_make_inputs(batch_size=1))


def test_rejects_unexpected_shared_encoder_feature_shape():
    encoder = _make_encoder()
    encoder.aion_model.output_shape = (
        AION_SPECTRUM_TOKENS,
        AION_SPECTRUM_WIDTH - 1,
    )

    with pytest.raises(RuntimeError, match="shared encoder must emit features"):
        encoder(_make_inputs(batch_size=1))


def test_codec_values_survive_a_blanket_reduced_precision_cast():
    """Dtype alone is not enough: rounded weights can change token ids.

    ZoneATrainer casts the whole encoder ModuleDict to the backbone dtype
    (src/training/trainer_zone_a.py), which reaches this encoder through
    nn.Module._apply. Restoring float32 afterwards would recover the dtype but
    not the values, and the codec quantizes hard.
    """
    encoder = _make_encoder()
    before = [p.detach().clone() for p in encoder.spectrum_codec.parameters()]
    assert before

    import torch.nn as nn

    bundle = nn.ModuleDict({"aion": encoder})
    bundle.to(dtype=torch.bfloat16)

    after = list(encoder.spectrum_codec.parameters())
    assert all(p.dtype == torch.float32 for p in after)
    for original, current in zip(before, after, strict=True):
        assert torch.equal(original, current)

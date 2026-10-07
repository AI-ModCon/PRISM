"""Regression tests for heterogeneous-length Intern-S2 padding masks."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from src.config import ModelConfig
from src.encoders.time_series import TimeSeriesEncoder
from src.model import UnifiedTransformer

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]


def _encoder_for_pad_test(encoder_type: str = "intern_s2"):
    encoder = TimeSeriesEncoder.__new__(TimeSeriesEncoder)
    nn.Module.__init__(encoder)
    encoder.encoder_type = encoder_type
    encoder.max_ts_length = 8
    encoder.num_vars = 1
    encoder.intern_s2_sampling_rate = 1.0
    encoder._last_pad_mask = None
    encoder.tokens_per_instance = MethodType(lambda self: 4, encoder)
    encoder.model = nn.Linear(1, 1)
    return encoder


@pytest.mark.parametrize("encoder_type", ["intern_s2", "intern_s2_397b"])
def test_intern_s2_pad_mask_is_padded_with_embeddings(encoder_type):
    encoder = _encoder_for_pad_test(encoder_type)
    embeddings = torch.arange(6 * 2, dtype=torch.float32).reshape(2, 3, 2)
    pad_mask = torch.tensor(
        [[False, False, True], [False, False, False]], dtype=torch.bool
    )
    padded, padded_mask = encoder._pad_intern_s2_tokens(embeddings, pad_mask)

    assert padded.shape == (2, 4, 2)
    assert torch.equal(
        padded_mask,
        torch.tensor(
            [[False, False, True, True], [False, False, False, True]],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(padded[0, 3], torch.zeros(2))


class _InternStub(nn.Module):
    def __init__(self, return_three=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.return_three = return_three

    def forward(self, **kwargs):
        embeddings = torch.ones(2, 3, 2)
        pad_mask = torch.tensor(
            [[False, False, True], [False, False, False]], dtype=torch.bool
        )
        if self.return_three:
            return embeddings, pad_mask, None
        return embeddings, pad_mask


@pytest.mark.parametrize(
    ("encoder_type", "return_three"),
    [("intern_s2", False), ("intern_s2_397b", True)],
)
def test_intern_s2_forward_exposes_last_pad_mask(encoder_type, return_three):
    encoder = _encoder_for_pad_test(encoder_type)
    encoder.model = _InternStub(return_three=return_three)
    result = encoder._forward_intern_s2_any(
        [torch.ones(2, 1), torch.ones(3, 1)]
    )

    assert result.shape == (2, 4, 2)
    assert torch.equal(
        encoder._last_pad_mask,
        torch.tensor(
            [[False, False, True, True], [False, False, False, True]],
            dtype=torch.bool,
        ),
    )


class _BackboneStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 4)
        self.config = SimpleNamespace(_attn_implementation="sdpa", hidden_size=4)
        self.dtype = torch.float32
        self.last_attention_mask = None

    def get_input_embeddings(self):
        return self.embed

    def parameters(self, recurse=True):
        return self.embed.parameters(recurse=recurse)

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        self.last_attention_mask = attention_mask.detach().clone()
        output = MagicMock()
        output.logits = torch.zeros(
            inputs_embeds.shape[0], inputs_embeds.shape[1], self.embed.num_embeddings
        )
        return output


def test_prefix_attention_mask_excludes_padded_intern_s2_tokens():
    backbone = _BackboneStub()
    config = ModelConfig(
        d_model=4,
        llm_backbone_id=None,
        modalities=["time_series", "text"],
        mask_padded_modality_tokens=True,
    )
    model = UnifiedTransformer.__new__(UnifiedTransformer)
    nn.Module.__init__(model)
    model.config = config
    model.backbone = backbone
    model.backbone_dim = 4
    model.is_vla = False
    model.text_decoder = MagicMock(return_value=(None, torch.tensor(0.0)))
    model.decoders = nn.ModuleDict()
    model.encoders = nn.ModuleDict()
    model.projectors = nn.ModuleDict()
    model._encoder_frozen_cache = {}
    model._modality_pad_masks = {
        "time_series": torch.tensor(
            [[False, False, True, True], [False, False, False, True]], dtype=torch.bool
        )
    }
    model._process_multimodal_embeddings = MagicMock(
        return_value=[
            ("time_series", torch.ones(2, 4, 4)),
            ("text", torch.ones(2, 2, 4)),
        ]
    )
    model._resolve_pad_id = lambda: None
    model.training = False

    model({"time_series": torch.ones(2, 1, 1), "text": torch.ones(2, 2, dtype=torch.long)})

    assert torch.equal(
        backbone.last_attention_mask,
        torch.tensor([[1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 1, 1]], dtype=torch.long),
    )
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


def test_intern_s2_long_embeddings_are_pooled_across_the_full_sequence():
    encoder = _encoder_for_pad_test("intern_s2_397b")
    encoder.max_ts_length = 4
    encoder.tokens_per_instance = lambda: 8
    embeddings = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    pad_mask = torch.zeros(1, 8, dtype=torch.bool)

    pooled, pooled_mask = encoder._pad_intern_s2_tokens(embeddings, pad_mask)

    assert pooled.shape == (1, 4, 1)
    assert torch.allclose(pooled.flatten(), torch.tensor([0.5, 2.5, 4.5, 6.5]))
    assert not pooled_mask.any()


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


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_prefix_attention_mask_excludes_padded_intern_s2_tokens(attn_implementation):
    backbone = _BackboneStub()
    backbone.config._attn_implementation = attn_implementation
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

    expected_valid = torch.tensor(
        [[1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 1, 1]], dtype=torch.bool
    )
    if attn_implementation == "eager":
        expected_allowed = torch.ones(6, 6, dtype=torch.bool).tril()[None, None]
        expected_allowed = expected_allowed & expected_valid[:, None, None, :]
        assert torch.equal(backbone.last_attention_mask == 0, expected_allowed)
    else:
        assert torch.equal(backbone.last_attention_mask.bool(), expected_valid)


@pytest.mark.integration
@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
@pytest.mark.parametrize("text_length", [1, 3])
def test_padded_prefix_values_cannot_change_training_loss_or_gradients(
    attn_implementation, text_length
):
    from src.decoders import LMHeadDecoder, TimeSeriesDecoder
    from transformers import Qwen3Config, Qwen3ForCausalLM

    backbone = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        attn_implementation=attn_implementation,
    )).eval()
    model = UnifiedTransformer.__new__(UnifiedTransformer)
    nn.Module.__init__(model)
    model.config = ModelConfig(
        d_model=4, modalities=["time_series", "text"],
        mask_padded_modality_tokens=True,
    )
    model.backbone = backbone
    model.backbone_dim = 4
    model.is_vla = False
    model.text_decoder = LMHeadDecoder()
    model.decoders = nn.ModuleDict({
        "time_series": TimeSeriesDecoder(4, horizon=2, num_vars=1, pool="last")
    })
    model.encoders = nn.ModuleDict()
    model.projectors = nn.ModuleDict()
    model._encoder_frozen_cache = {}
    pad_mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    model._modality_pad_masks = {"time_series": pad_mask}
    model._resolve_pad_id = lambda: None
    model.training = False
    text = torch.tensor([[3, 4, 5], [6, 7, 8]])[:, :text_length]
    inputs = {
        "text": text,
        "time_series_target": torch.randn(2, 2, 1),
        "time_series_target_mask": torch.ones(2, 2, 1, dtype=torch.bool),
    }
    prefix = torch.randn(2, 4, 4)
    losses = []
    gradients = []
    logits = []
    for values in (prefix, prefix.masked_fill(pad_mask.unsqueeze(-1), 100.0)):
        values = values.detach().requires_grad_()
        model.zero_grad(set_to_none=True)
        model._process_multimodal_embeddings = MagicMock(return_value=[
            ("time_series", values), ("text", backbone.get_input_embeddings()(text))
        ])
        prediction, loss = model(inputs, labels=text)
        loss.backward()
        losses.append(loss.detach())
        logits.append(prediction[:, -text.shape[1]:].detach())
        gradients.append(values.grad.detach())

    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(logits[0], logits[1])
    torch.testing.assert_close(gradients[0], gradients[1])
    assert torch.count_nonzero(gradients[0][pad_mask]) == 0
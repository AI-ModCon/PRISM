"""Offline checks for the time-series encoder token-width contract."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from src.encoders.time_series import TimeSeriesEncoder
from torch import nn

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]


class FakeInternS2(nn.Module):
    def __init__(self, is_397b=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(out_hidden_size=8)
        self.encoder_embed = SimpleNamespace(transformer_encoder=None)
        self.is_397b = is_397b

    def forward(self, time_series_signals, ts_lens, **kwargs):
        token_count = max(1, int(ts_lens.max()) // 8)
        embeddings = torch.zeros(time_series_signals.shape[0], token_count, 8)
        pad_mask = torch.zeros(embeddings.shape[:2], dtype=torch.bool)
        if self.is_397b:
            return embeddings, pad_mask, torch.zeros_like(embeddings)
        return embeddings, pad_mask


class FakeMoirai(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.d_model = 8
        self.patch_size = 16
        self.in_proj = nn.Linear(32, 8)

    def scaler(self, inputs, observed_mask, sample_id, variate_id):
        return torch.zeros_like(inputs), torch.ones_like(inputs)

    def encoder(self, inputs, attn_mask, **kwargs):
        return inputs


@pytest.mark.parametrize("num_vars", [1, 3])
@pytest.mark.parametrize(
    "encoder_type", ["linear", "moirai", "intern_s2", "intern_s2_397b", "timeomni"]
)
def test_tokens_per_instance_matches_forward_width(monkeypatch, encoder_type, num_vars):
    if encoder_type == "moirai":
        monkeypatch.setattr("src.encoders.time_series._load_moirai_model", lambda *args, **kwargs: FakeMoirai())
        torch_util = types.ModuleType("uni2ts.common.torch_util")
        torch_util.packed_causal_attention_mask = lambda *args: None
        monkeypatch.setitem(sys.modules, "uni2ts.common.torch_util", torch_util)
    elif encoder_type.startswith("intern_s2"):
        monkeypatch.setattr(
            "src.encoders.time_series._load_intern_s2_model",
            lambda *args, **kwargs: FakeInternS2(encoder_type == "intern_s2_397b"),
        )

    encoder = TimeSeriesEncoder(
        encoder_type=encoder_type,
        num_vars=num_vars,
        d_ts=8,
        max_ts_length=64,
        load_pretrained=False,
        timeomni_patch_len=16,
        timeomni_max_patches=16,
        timeomni_d_model=8,
        is_interleaved=True,
    )
    with torch.no_grad():
        output = encoder(torch.zeros(2, 64, num_vars))

    assert encoder.tokens_per_instance() == output.shape[1]
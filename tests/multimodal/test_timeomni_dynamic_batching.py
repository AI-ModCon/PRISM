from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoders.time_series import TimeSeriesEncoder

pytestmark = [pytest.mark.unit, pytest.mark.multimodal, pytest.mark.timeseries]


def test_timeomni_encoder_accepts_variable_length_list_inputs():
    enc = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=2,
        d_ts=32,
        max_ts_length=1024,
        timeomni_patch_len=[16, 32, 64],
        timeomni_stride=[16, 32, 64],
        timeomni_d_model=48,
    )
    series = [
        torch.randn(96, 2),
        torch.randn(320, 2),
    ]

    out = enc(series)

    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[-1] == 48


def test_timeomni_encoder_selects_patch_per_sample_length(monkeypatch):
    enc = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=1,
        d_ts=32,
        max_ts_length=1024,
        timeomni_patch_len=[16, 32, 64],
        timeomni_stride=[16, 32, 64],
        timeomni_d_model=32,
    )
    called = []
    original = enc._select_patch_embedding

    def _wrapped(length: int) -> int:
        called.append(length)
        return original(length)

    monkeypatch.setattr(enc, "_select_patch_embedding", _wrapped)

    _ = enc([torch.randn(40, 1), torch.randn(280, 1)])
    assert called == [40, 280]


def test_timeomni_interleaved_outputs_fixed_token_budget():
    enc = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=1,
        d_ts=32,
        max_ts_length=960,
        is_interleaved=True,
        timeomni_patch_len=[16, 32, 64],
        timeomni_stride=[16, 32, 64],
        timeomni_d_model=32,
    )
    out = enc([torch.randn(40, 1), torch.randn(280, 1)])
    assert out.shape[1] == enc.tokens_per_instance()


def test_timeomni_interleaved_budget_independent_of_configured_num_vars():
    enc = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=7,
        d_ts=32,
        max_ts_length=960,
        is_interleaved=True,
        timeomni_patch_len=[16, 32, 64],
        timeomni_stride=[16, 32, 64],
        timeomni_d_model=32,
        timeomni_max_patches=12,
    )
    # Different variate counts per sample are supported by flattening each
    # sample to a univariate stream before patching.
    out = enc([torch.randn(80, 2), torch.randn(80, 5)])
    assert enc.tokens_per_instance() == 12
    assert out.shape == (2, 12, 32)


@pytest.mark.parametrize("num_vars, patch_len", [(1, 16), (2, 32), (5, 64)])
def test_timeomni_serializes_multivariate_values_into_one_stream(
    monkeypatch, num_vars: int, patch_len: int
):
    """Multivariate TimeOmni deliberately patches one time-major value stream.

    The channel dimension passed to the patcher is one; its sequence axis is
    the exact time-major flattening of the normalized (T, V) input.
    """
    enc = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=num_vars,
        d_ts=16,
        max_ts_length=512,
        is_interleaved=True,
        timeomni_patch_len=[patch_len],
        timeomni_stride=[patch_len],
        timeomni_d_model=16,
        timeomni_max_patches=32,
    )
    captured_shapes = []
    original_forward = enc.timeomni_patch_embeddings[str(patch_len)].forward

    def capture_input(serialized):
        captured_shapes.append(tuple(serialized.shape))
        return original_forward(serialized)

    monkeypatch.setattr(
        enc.timeomni_patch_embeddings[str(patch_len)], "forward", capture_input
    )
    out = enc(torch.arange(12 * num_vars, dtype=torch.float32).reshape(12, num_vars))

    assert captured_shapes == [(1, 1, 12 * num_vars)]
    assert out.shape == (1, 32, 16)


def test_timeomni_patch_selection_respects_max_patch_budget():
    enc = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=1,
        d_ts=32,
        max_ts_length=576,
        timeomni_patch_len=[16, 32, 64],
        timeomni_stride=[16, 32, 64],
        timeomni_d_model=32,
        timeomni_max_patches=10,
    )
    selected = enc._select_patch_embedding(576)
    assert selected == 64
    num_patches = enc._num_timeomni_patches(576, 64, 64)
    assert num_patches <= 10


def test_timeomni_patch_selection_raises_when_no_candidate_meets_budget():
    enc = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=1,
        d_ts=32,
        max_ts_length=576,
        timeomni_patch_len=[16, 32, 64],
        timeomni_stride=[16, 32, 64],
        timeomni_d_model=32,
        timeomni_max_patches=2,
    )
    with pytest.raises(RuntimeError, match="max patch budget"):
        enc._select_patch_embedding(576)


def test_timeomni_patch_selection_rejects_infeasible_patch_len():
    """Patch configs where patch_len > T + stride must be excluded (would crash unfold).

    patch_len=100, stride=8: for T=5, T+stride=13 < patch_len=100, so
    _num_timeomni_patches returns a negative value. _select_patch_embedding
    must reject this and raise rather than silently passing it to unfold.
    """
    enc = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=1,
        d_ts=32,
        max_ts_length=1024,
        timeomni_patch_len=[100],
        timeomni_stride=[8],
        timeomni_d_model=32,
        timeomni_max_patches=100,
    )
    # T=5: T+stride=13 < patch_len=100 → _num_timeomni_patches returns negative value
    # _select_patch_embedding must raise, not silently accept this candidate
    with pytest.raises(RuntimeError, match="max patch budget"):
        enc._select_patch_embedding(5)

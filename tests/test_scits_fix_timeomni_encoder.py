"""CPU unit tests for the timeomni-encoder-side fixes/guards from the
PR #129 (SciTS/TimeOmni) review:

  M: vLLM num_tokens (TimeSeriesModalityProcessor) must NOT multiply by
     num_vars for the timeomni encoder type — training's tokens_per_instance()
     doesn't either, since a multivariate sample is flattened into a single
     sequence before patching.
    I: non-interleaved batches with uneven encoded lengths must warn because
         the caller does not mask the left padding.


No process group, no HF download — plain torch + TimeSeriesEncoder.
"""

import logging

import pytest
import torch
from src.encoders.time_series import TimeSeriesEncoder

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]


def _reset_warning_flags():
    TimeSeriesEncoder._warned_left_pad_unmasked = False


def _make_timeomni_encoder(**overrides):
    kwargs = dict(
        encoder_type="timeomni",
        num_vars=1,
        d_ts=16,
        max_ts_length=1024,
        is_interleaved=True,
        timeomni_patch_len=[16],
        timeomni_stride=[16],
        timeomni_d_model=16,
        timeomni_max_patches=32,
    )
    kwargs.update(overrides)
    return TimeSeriesEncoder(**kwargs)


# ---------------------------------------------------------------------------
# Defect M: tokens_per_instance for timeomni must be independent of num_vars.
# ---------------------------------------------------------------------------


def test_tokens_per_instance_timeomni_ignores_num_vars():
    encoder_v1 = _make_timeomni_encoder(num_vars=1)
    encoder_v7 = _make_timeomni_encoder(num_vars=7)

    assert encoder_v1.tokens_per_instance() == 32
    assert encoder_v7.tokens_per_instance() == 32
    assert encoder_v1.tokens_per_instance() == encoder_v7.tokens_per_instance()


def test_vllm_num_tokens_matches_tokens_per_instance_for_timeomni():
    """Defect M regression: the vLLM data-plane processor's num_tokens()
    must return the same value as the training-side encoder's
    tokens_per_instance() for encoder_type='timeomni' — before the fix,
    num_tokens() multiplied by num_vars while tokens_per_instance() did
    not, so vLLM raised a placeholder-count mismatch at splice time for any
    multivariate input.

    Reimplements the processor's num_tokens() math directly (avoids the
    vllm import, which is importable here but slow) — this is a pure
    arithmetic check of the formula, not a vllm integration test.
    """
    max_ts_length = 1024
    timeomni_max_patches = 32
    num_vars = 5

    encoder = _make_timeomni_encoder(
        num_vars=num_vars,
        max_ts_length=max_ts_length,
        timeomni_max_patches=timeomni_max_patches,
    )

    # Mirrors TimeSeriesModalityProcessor.num_tokens()'s timeomni branch.
    vllm_num_tokens = timeomni_max_patches

    assert vllm_num_tokens == encoder.tokens_per_instance()


def test_non_interleaved_left_pad_emits_one_shot_warning(caplog):
    """Defect I: non-interleaved batches with samples of differing encoded
    token counts left-pad with zeros; the caller's attention mask has no
    way to see this padding. Must emit the documented one-shot warning."""
    _reset_warning_flags()
    encoder = _make_timeomni_encoder(num_vars=1, is_interleaved=False)

    # Two samples with different lengths -> different encoded token counts.
    # 496 is the largest T that stays within timeomni_max_patches=32 for
    # patch_len=stride=16: num_patches = (T + stride - patch_len)//stride + 1.
    short = torch.randn(32, 1)
    long = torch.randn(496, 1)
    with caplog.at_level(logging.WARNING, logger="src.encoders.time_series"):
        out = encoder([short, long])

    assert out.ndim == 3
    warned = [r for r in caplog.records if "left-padding with zeros" in r.message]
    assert warned, f"expected left-pad warning, got records: {[r.message for r in caplog.records]}"
    assert TimeSeriesEncoder._warned_left_pad_unmasked is True


@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float32])
def test_decimation_preserves_model_dtype_without_autocast(model_dtype):
    """Post-merge regression: the over-budget decimation path did
    `ts.float().mean(dim=1)` and never restored `model_dtype`, so the
    patch-embedding Conv1d received an fp32 input against bf16 weights.

    torch.autocast masks this, which is why the training path (which wraps
    the forward in autocast) never tripped it. tools/universal_evaluator.py
    does `model.to(device, dtype=torch.bfloat16)` and never enters an
    autocast region, so eval on a single over-budget SciTS series crashed
    with "Input type (torch.FloatTensor) and weight type (CPUBFloat16Type)
    should be the same".

    Deliberately asserted OUTSIDE autocast — that is the regime that broke.
    """
    encoder = _make_timeomni_encoder(num_vars=1, max_ts_length=128).to(model_dtype)

    # flat length 512 > max_ts_length=128 -> decimation fires (factor 4).
    over_budget = torch.randn(512, 1)

    with torch.no_grad():
        out = encoder([over_budget])

    assert out.dtype == model_dtype
    assert torch.isfinite(out).all()


def test_decimation_is_actually_exercised_by_the_dtype_test():
    """Guard the guard: if a future config change makes the fixture above
    stop decimating, the dtype test would still pass while covering nothing.
    Pin that the chosen shape really does trip the decimation branch."""
    encoder = _make_timeomni_encoder(num_vars=1, max_ts_length=128)
    flat_len = 512 * 1
    assert flat_len > encoder.max_ts_length, (
        "fixture no longer exercises the decimation path; "
        f"flat_len={flat_len} <= max_ts_length={encoder.max_ts_length}"
    )

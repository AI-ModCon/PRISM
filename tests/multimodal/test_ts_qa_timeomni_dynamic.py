from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.data.multimodal import StreamingMultimodalDataset
from src.encoders.time_series import TimeSeriesEncoder

pytestmark = [pytest.mark.unit, pytest.mark.multimodal, pytest.mark.timeseries]


class _MockTokenizer:
    eos_token = "<|endoftext|>"

    def decode(self, token_ids):
        if isinstance(token_ids, list) and len(token_ids) == 1:
            if token_ids[0] == 50280:
                return "<ts>"
            if token_ids[0] == 50281:
                return "<ts/>"
        return "<unk>"


class _DummyDatasetContext:
    def __init__(
        self,
        max_ts_length: int,
        is_interleaved_qa: bool = False,
    ):
        self.model_config = ModelConfig(
            is_timeseries=True,
            ts_projector="timeomni",
            max_ts_length=max_ts_length,
            normalize_ts_in_encoder=True,
            is_interleaved_qa=is_interleaved_qa,
            modality_start_end_token_indices={"time_series": (50280, 50281)},
        )
        self.tokenizer = _MockTokenizer()
        self.max_seq_length = None


def test_timeomni_ts_qa_keeps_2d_shape_and_values():
    dummy = _DummyDatasetContext(max_ts_length=8)
    item = {
        "instruction": "Describe this time series.",
        "output": "ok",
        # (T, V) = (5, 2)
        "timeseries": [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ],
    }

    tensor, _, _ = StreamingMultimodalDataset._process_ts_qa(dummy, item)

    assert tensor.shape == (5, 2)
    expected = torch.tensor(item["timeseries"], dtype=torch.float)
    assert torch.equal(tensor, expected)


def test_timeomni_ts_qa_inserts_single_interleaved_span_when_missing():
    dummy = _DummyDatasetContext(max_ts_length=8, is_interleaved_qa=True)
    item = {
        "instruction": "Describe this time series without placeholders.",
        "output": "ok",
        "timeseries": [[1.0], [2.0], [3.0]],
    }
    _, prompt_target, meta = StreamingMultimodalDataset._process_ts_qa(dummy, item)
    assert meta[0].count("<ts><ts/>") == 1
    assert "<ts><ts/>" in prompt_target


def test_timeomni_ts_qa_raises_when_multiple_interleaved_spans_present():
    dummy = _DummyDatasetContext(max_ts_length=8, is_interleaved_qa=True)
    item = {
        "instruction": "A <ts><ts/> and B <ts><ts/>",
        "output": "ok",
        "timeseries": [[1.0], [2.0], [3.0]],
    }
    with pytest.raises(RuntimeError, match="exactly one"):
        StreamingMultimodalDataset._process_ts_qa(dummy, item)


def test_timeomni_ts_qa_transposes_legacy_vars_first_2d_series():
    dummy = _DummyDatasetContext(max_ts_length=8)
    # Legacy shape (V, T) = (2, 5) should become (T, V) = (5, 2)
    item = {
        "instruction": "Describe this time series.",
        "output": "ok",
        "timeseries": [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ],
    }
    tensor, _, _ = StreamingMultimodalDataset._process_ts_qa(dummy, item)
    assert tensor.shape == (5, 2)
    assert tensor[0].tolist() == [1.0, 10.0]


def test_timeomni_ts_qa_1d_series_uses_dynamic_path():
    """1D univariate input under timeomni must NOT be truncated/padded/normalized."""
    # budget = 8, series length = 6 — should NOT be padded to 8
    dummy = _DummyDatasetContext(max_ts_length=8)
    raw_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    item = {
        "instruction": "Describe this time series.",
        "output": "ok",
        "timeseries": raw_vals,
    }

    tensor, _, _ = StreamingMultimodalDataset._process_ts_qa(dummy, item)

    # Must preserve length 6, not pad to 8
    assert tensor.shape == (6, 1), f"Expected (6, 1), got {tensor.shape}"
    # Values must be unmodified (no normalization)
    assert tensor[:, 0].tolist() == raw_vals


def test_timeomni_encoder_raises_on_zero_variate_tensor():
    """Encoder must not crash with an ambiguous reshape when n_vars=0.

    Passing a (T, 0) tensor produces x of shape (1, 0, T) in
    _forward_timeomni, which makes _TimeOmniPatchEmbedding return
    enc_out with 0 elements.  The subsequent
      enc_out.view(1, n_vars, num_patches, -1)
    then fails with:
      RuntimeError: cannot reshape tensor of 0 elements into shape
      [1, 0, 1, -1] because the unspecified dimension size -1 can be
      any value and is ambiguous
    """
    enc = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=1,
        d_ts=32,
        max_ts_length=1024,
        timeomni_patch_len=[16],
        timeomni_stride=[16],
        timeomni_d_model=32,
    )
    # A tensor with T=5 time steps but V=0 variates triggers n_vars=0 inside
    # _TimeOmniPatchEmbedding, causing the ambiguous view reshape.
    zero_variate = torch.zeros(5, 0)
    with pytest.raises(RuntimeError):
        enc(zero_variate)

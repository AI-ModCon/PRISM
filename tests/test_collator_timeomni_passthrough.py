from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import MultimodalCollator

pytestmark = [pytest.mark.unit, pytest.mark.multimodal, pytest.mark.timeseries]


class _FakeTokenizer:
    pad_token_id = 0


def test_multimodal_collator_time_series_passthrough_mode():
    collator = MultimodalCollator(
        _FakeTokenizer(),
        passthrough_time_series=True,
        max_seq_length=16,
    )
    batch = [
        {
            "text": torch.arange(1, 6, dtype=torch.long),
            "time_series": torch.randn(5, 2),
        },
        {
            "text": torch.arange(1, 8, dtype=torch.long),
            "time_series": torch.randn(9, 2),
        },
    ]

    out = collator(batch)

    assert isinstance(out["time_series"], list)
    assert out["time_series"][0].shape == (5, 2)
    assert out["time_series"][1].shape == (9, 2)
    assert out["text"].shape == (2, 7)


def test_multimodal_collator_default_keeps_padding_behavior_for_time_series():
    collator = MultimodalCollator(_FakeTokenizer(), max_seq_length=16)
    batch = [
        {
            "text": torch.arange(1, 6, dtype=torch.long),
            "time_series": torch.randn(5, 2),
        },
        {
            "text": torch.arange(1, 8, dtype=torch.long),
            "time_series": torch.randn(9, 2),
        },
    ]

    out = collator(batch)

    assert isinstance(out["time_series"], torch.Tensor)
    assert out["time_series"].shape == (2, 9, 2)

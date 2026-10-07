"""Regression tests for MultimodalCollator sequence-length capping (issue #120).

Issue #120: the plain MultimodalCollator had no `max_seq_length` parameter and
performed no truncation, so on the non-bucketed path oversized text sequences
were padded to the batch max with no cap. BucketedCollator already truncated
(src/data/collate.py); MultimodalCollator did not, and the launcher dropped the
cap when constructing it (src/train.py, src/training/trainer_native.py).

The fix gives MultimodalCollator the SAME truncate-before-pad behavior as
BucketedCollator, gated on an explicit `max_seq_length`. Callers on the
interleaved ts_qa path pass `max_seq_length=None` because tail-truncating an
interleaved sample would desync the <ts>/<ts/> token balance and the
prompt/target metadata the model relies on (that path is capped at the variate
level in _process_ts_qa instead — see tests/multimodal/test_ts_qa_variate_cap.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import MultimodalCollator

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]


class _FakeTokenizer:
    pad_token_id = 0


def _text_batch(lengths):
    """A batch of plain text-only samples with the given 1D token lengths."""
    return [
        {"text": torch.arange(1, n + 1, dtype=torch.long)} for n in lengths
    ]


def _interleaved_text_batch():
    ts_start, ts_end, pad_id = 50280, 50281, 0
    return [
        {
            "text": torch.tensor(
                [101, ts_start, ts_end, 7, ts_start, ts_end, 8, pad_id],
                dtype=torch.long,
            )
        },
        {
            "text": torch.tensor(
                [101, 9, ts_start, ts_end, 9, 10, 7, 8],
                dtype=torch.long,
            )
        },
    ]

def test_multimodal_collator_truncates_text_over_cap():
    """A sample longer than max_seq_length must be truncated before padding."""
    lens = [1024, 2048, 4096]
    for length in lens:
        collator = MultimodalCollator(_FakeTokenizer(), max_seq_length=length)
        batch = _text_batch([100, 5878])  # one short, one way over the cap

        out = collator(batch)

        assert out["text"].shape[1] == length, (
            f"Text should be capped at {length}, got width {out['text'].shape[1]}"
        )
        # Short sample's real tokens preserved (then padded to {length})
        assert out["text"][0, :100].tolist() == list(range(1, 101))
        # Long sample truncated to the first {length} tokens
        assert out["text"][1, :length].tolist() == list(range(1, length + 1))

def test_multimodal_collator_no_cap_by_default():
    """Default (no max_seq_length) preserves the previous behavior: pad to the
    batch max with no truncation. This is what the interleaved path relies on."""
    collator = MultimodalCollator(_FakeTokenizer())  # no cap
    batch = _text_batch([100, 5878])

    out = collator(batch)

    assert out["text"].shape[1] == 5878, (
        f"Without a cap, width should be the batch max 5878, got {out['text'].shape[1]}"
    )


def test_multimodal_collator_interleaved_tokens_preserved_without_cap():
    collator = MultimodalCollator(_FakeTokenizer(), max_seq_length=None)
    batch = _interleaved_text_batch()

    out = collator(batch)

    assert out["text"].shape == (2, 8)
    assert out["text"][0, 1].item() == 50280
    assert out["text"][0, 2].item() == 50281
    assert out["text"][1, 2].item() == 50280
    assert out["text"][1, 3].item() == 50281


def test_multimodal_collator_cap_above_lengths_is_noop():
    """A cap larger than every sequence changes nothing but the pad width."""
    collator = MultimodalCollator(_FakeTokenizer(), max_seq_length=4096)
    batch = _text_batch([100, 300])

    out = collator(batch)

    assert out["text"].shape[1] == 300  # pad to batch max, not to the cap
    assert out["text"][1, :300].tolist() == list(range(1, 301))


def test_multimodal_collator_does_not_mutate_input_samples():
    """Truncation must not corrupt the caller's sample dicts in place across
    epochs (the tensors may be reused). Verify original lengths are intact."""
    collator = MultimodalCollator(_FakeTokenizer(), max_seq_length=50)
    batch = _text_batch([200])
    original_len = batch[0]["text"].shape[0]

    collator(batch)

    assert batch[0]["text"].shape[0] == original_len, (
        "Collator must not truncate the caller's sample tensor in place"
    )

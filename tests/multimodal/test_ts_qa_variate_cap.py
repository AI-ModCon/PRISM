"""Root-cause tests for issue #120: interleaved ts_qa merged-length blowup.

The reported symptom ("merged sequences reached ~5878 tokens despite
--max-seq-length 2048") is NOT a collate-time problem. The collator only sees
the *raw* input_ids (a 20-variate sample is ~315 raw tokens). The blowup happens
*inside* the model at src/model.py:_merge_text_input_ids_with_modality_embeds,
where each <ts> span expands to `max_ts_length * num_vars` embedding tokens. A
ts_qa sample with V variates expands to ~V * max_ts_length merged tokens.

The real lever is therefore the variate count, capped in _process_ts_qa so that
V * max_ts_length stays within the configured max_seq_length budget. Because the
2D branch emits exactly one <ts><ts/> pair per variate row, truncating the
tensor rows also truncates the prompt's <ts> pairs — keeping the merge
invariant (#<ts> pairs == #variate rows == encoder spans) intact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.data.multimodal import StreamingMultimodalDataset

pytestmark = [pytest.mark.unit, pytest.mark.multimodal, pytest.mark.timeseries]


class _MockTokenizer:
    """Minimal tokenizer mock supporting .decode() for the 2D interleaved path."""

    eos_token = "<|endoftext|>"

    def decode(self, token_ids):
        if isinstance(token_ids, list) and len(token_ids) == 1:
            if token_ids[0] == 50280:
                return "<ts>"
            if token_ids[0] == 50281:
                return "<ts/>"
        return "<unk>"


class _DummyDatasetContext:
    def __init__(self, max_ts_length: int, max_seq_length: int | None = None):
        self.model_config = ModelConfig(
            is_timeseries=True,
            is_interleaved_qa=True,
            max_ts_length=max_ts_length,
            modality_start_end_token_indices={"time_series": (50280, 50281)},
        )
        self.tokenizer = _MockTokenizer()
        # The cap the launcher passes through (MAX_SEQ_LENGTH). None => no cap.
        self.max_seq_length = max_seq_length


def _make_multivariate_item(num_vars: int, steps: int = 256):
    """A ts_qa item shaped like ChatTS: (num_vars, steps) series with a prompt
    containing one <ts><ts/> placeholder per variate."""
    placeholders = "".join(
        f"Metric {i}: <ts><ts/>;\n" for i in range(num_vars)
    )
    prompt = f"There are {num_vars} metrics:\n{placeholders}Describe them."
    series = [[float(i + j) for j in range(steps)] for i in range(num_vars)]
    return {"input": prompt, "timeseries": series, "output": "A description."}


def _count_ts_pairs(text: str) -> int:
    # The 2D prompt-mutation rewrites each placeholder as
    # "<ts><ts/> Mean: .., Std: .. " — it INSERTS stats after the pair but never
    # splits the "<ts><ts/>" substring itself, so counting that literal is a
    # robust proxy for the number of emitted <ts> spans (== surviving variates).
    return text.count("<ts><ts/>")


def test_variate_cap_reserves_text_headroom():
    """#120 REOPENED: budgeting only the TS contribution (n_vars * max_ts_length)
    let merged length = TS + text exceed max_seq_length, OOMing the XPU tile
    (hardware-confirmed: OLMo-1B fwd+bwd OOMs at seq >= ~4608 on a 64GB tile).

    The cap must budget the TOTAL merged length: kept variates must leave
    headroom for text, so kept * max_ts_length is STRICTLY below max_seq_length,
    not equal to it."""
    dummy = _DummyDatasetContext(max_ts_length=256, max_seq_length=2048)
    item = _make_multivariate_item(num_vars=20, steps=256)

    tensor, prompt_target, meta = StreamingMultimodalDataset._process_ts_qa(dummy, item)

    kept = tensor.shape[0] // 256
    ts_only_budget = 2048 // 256  # 8 — the OLD (buggy) cap
    # New cap must reserve text headroom → strictly fewer than the TS-only cap.
    assert kept < ts_only_budget, (
        f"Cap must reserve text headroom: kept={kept} should be < {ts_only_budget}"
    )
    # TS tokens alone must leave room under the budget for text.
    assert tensor.shape[0] < 2048, (
        f"TS tokens {tensor.shape[0]} must be strictly under max_seq_length=2048"
    )
    # Invariant preserved: <ts> pairs == surviving variate rows.
    assert _count_ts_pairs(meta[0]) == kept


def test_variate_cap_reopened_4096_case():
    """The exact reopened scenario: max_seq_length=4096, max_ts_length=256, 16
    variates (Patrick's align_256 data). The OLD TS-only cap = 4096//256 = 16 →
    kept ALL 16 → merged ~4400-6100 → OOM. The fix must drop below 16."""
    dummy = _DummyDatasetContext(max_ts_length=256, max_seq_length=4096)
    item = _make_multivariate_item(num_vars=16, steps=256)

    tensor, _, meta = StreamingMultimodalDataset._process_ts_qa(dummy, item)

    kept = tensor.shape[0] // 256
    assert kept < 16, (
        f"Reopened bug: at max_seq_length=4096 the TS-only cap kept all 16 "
        f"variates → OOM. Fix must keep < 16, got {kept}"
    )
    assert tensor.shape[0] < 4096
    assert _count_ts_pairs(meta[0]) == kept


def test_variate_cap_ts_pairs_match_tensor_rows():
    """The merge invariant: #<ts> pairs in the prompt == #variate rows in the
    flattened tensor / max_ts_length. A mismatch crashes model.py:609."""
    dummy = _DummyDatasetContext(max_ts_length=128, max_seq_length=512)
    item = _make_multivariate_item(num_vars=10, steps=128)

    tensor, _, meta = StreamingMultimodalDataset._process_ts_qa(dummy, item)

    kept_rows = tensor.shape[0] // 128
    assert _count_ts_pairs(meta[0]) == kept_rows
    # Total budget with text headroom: kept TS tokens strictly under max_seq_length.
    assert kept_rows * 128 < 512


def test_no_cap_keeps_all_variates():
    """With no max_seq_length cap (None), behavior is unchanged: all variates
    survive (this preserves existing runs / the chatts.json fixture contract)."""
    dummy = _DummyDatasetContext(max_ts_length=256, max_seq_length=None)
    item = _make_multivariate_item(num_vars=16, steps=256)

    tensor, _, meta = StreamingMultimodalDataset._process_ts_qa(dummy, item)

    assert tensor.shape == (16 * 256, 1)
    assert _count_ts_pairs(meta[0]) == 16


def test_cap_noop_when_under_budget():
    """A sample already within budget is untouched by the cap."""
    dummy = _DummyDatasetContext(max_ts_length=256, max_seq_length=2048)
    item = _make_multivariate_item(num_vars=3, steps=256)  # 3×256=768 < 2048

    tensor, _, meta = StreamingMultimodalDataset._process_ts_qa(dummy, item)

    assert tensor.shape == (3 * 256, 1)
    assert _count_ts_pairs(meta[0]) == 3

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.data.multimodal import StreamingMultimodalDataset

pytestmark = [pytest.mark.integration, pytest.mark.multimodal, pytest.mark.timeseries]

TS_QA_SFT_DIR = Path("/flare/ModCon/ngetty/data/zone_a/ts_qa/sft")
TS_QA_SFT_TRAIN_JSONL = TS_QA_SFT_DIR / "train.jsonl"


class _MockTokenizer:
    """Minimal tokenizer mock that supports .decode() for the 2D TS path."""
    eos_token = "<|endoftext|>"
    
    def decode(self, token_ids):
        # Return the special token strings that the 2D branch expects
        if isinstance(token_ids, list) and len(token_ids) == 1:
            if token_ids[0] == 50280:
                return "<ts>"
            elif token_ids[0] == 50281:
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
        # Merged-length variate-cap budget (issue #120). None disables the cap,
        # matching the real StreamingMultimodalDataset default; these fixtures
        # assert the uncapped multivariate shape (e.g. 16×512 → (8192, 1)).
        self.max_seq_length = max_seq_length


def test_process_ts_qa_chatts_repo_timeseries():
    dummy = _DummyDatasetContext(max_ts_length=512)

    # Load data from chatts.json
    chatts_path = Path(__file__).parent / "data" / "chatts.json"
    with open(chatts_path) as f:
        item = json.load(f)
    item["id"] = "chatts-1"

    tensor, prompt_target, meta = StreamingMultimodalDataset._process_ts_qa(dummy, item)
    print(tensor.shape)
    assert tensor.shape == (8192, 1), f"Expected (8192, 1), got {tensor.shape}"

    # The 2D branch injects mean/std stats into the prompt, so original input
    # won't be an exact substring. Check that target (output) is preserved
    # and that key content from the original input is still present.
    assert item["output"] in prompt_target, "Target text should be in prompt_target"
    # The non-TS-token parts of the original prompt should still be present
    assert (
        "Sports Analytics" in prompt_target
    ), "Key prompt content should survive mean/std injection"

    # meta is [modified_prompt, target]
    assert isinstance(meta, list) and len(meta) == 2, "Meta should be [prompt, target]"
    output_with_eos = item["output"] + dummy.tokenizer.eos_token
    assert meta[1] == output_with_eos, "Target in metadata should match original output"


def test_process_ts_instruction_synthesizes_qa_pair():
    """ts_instruction's real JSONL schema has description/characteristics/series
    (no q/a fields). The handler must synthesize a (prompt, target) pair and
    route through the _process_ts_qa path."""
    dummy = _DummyDatasetContext(max_ts_length=512)
    # _process_ts_instruction delegates to self._process_ts_qa — bind it on the
    # dummy so the unbound-method call pattern works.
    dummy._process_ts_qa = lambda synthetic: StreamingMultimodalDataset._process_ts_qa(
        dummy, synthetic
    )

    item = {
        "description": (
            "A scientist measures freezer temperature over two weeks; an outage "
            "raises the temperature for 3 days before recovery."
        ),
        "description_short": "Freezer temperature over two weeks with a 3-day outage.",
        "description_tiny": "Freezer Temperature Time Series",
        "characteristics": (
            "1) Continuous increment during outage, 2) Slow decrease after restoration, "
            "3) Stable values before/after, 4) Slow ascending/descending trends."
        ),
        "series": [20.0 + (i * 0.05) for i in range(300)],
        "metadata": {"units": "Celsius", "frequency": "Every 6 hours"},
    }

    tensor, prompt_target, meta = StreamingMultimodalDataset._process_ts_instruction(
        dummy, item
    )

    # Same tuple shape as _process_ts_qa: (tensor, prompt+target, [prompt, target])
    assert tensor.shape == (512, 1), f"Expected (512, 1) tensor, got {tensor.shape}"
    assert isinstance(meta, list) and len(meta) == 2

    # Target should be one of the description fields (short preferred)
    assert "Freezer temperature" in prompt_target
    # Characteristics should appear in the synthesized prompt
    assert "Continuous increment" in prompt_target

    # Target metadata gets eos appended (per _process_ts_qa contract)
    assert meta[1].endswith(dummy.tokenizer.eos_token)


def test_process_ts_instruction_rejects_empty_series():
    dummy = _DummyDatasetContext(max_ts_length=512)
    item = {
        "description": "x",
        "description_short": "short",
        "characteristics": "c",
        "series": [],
    }
    with pytest.raises(RuntimeError, match="series"):
        StreamingMultimodalDataset._process_ts_instruction(dummy, item)


def test_process_ts_qa_one_real_sft_example():
    """Load one real ts_qa JSONL record from local sft path and process it."""
    if not TS_QA_SFT_TRAIN_JSONL.exists():
        pytest.skip(f"Local ts_qa JSONL not found: {TS_QA_SFT_TRAIN_JSONL}")

    first_item = None
    with open(TS_QA_SFT_TRAIN_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first_item = json.loads(line)
            break

    if first_item is None:
        pytest.skip(f"No JSON records found in: {TS_QA_SFT_TRAIN_JSONL}")

    dummy = _DummyDatasetContext(max_ts_length=512)
    tensor, prompt_target, meta = StreamingMultimodalDataset._process_ts_qa(dummy, first_item)

    assert tensor.ndim == 2 and tensor.shape[1] == 1, (
        f"Expected 2D tensor with shape (N, 1), got {tuple(tensor.shape)}"
    )
    assert tensor.shape[0] > 0, "Processed tensor must have at least one timestep"
    assert isinstance(prompt_target, str) and prompt_target.strip(), "prompt_target must be non-empty"
    assert isinstance(meta, list) and len(meta) == 2, "Meta should be [prompt, target]"
    assert meta[1].endswith(dummy.tokenizer.eos_token), "Target in metadata should end with eos"


if __name__ == "__main__":
    test_process_ts_qa_chatts_repo_timeseries()
    test_process_ts_qa_one_real_sft_example()
    print("All tests passed!")

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.encoders.base import ModalityEncoder
from src.model import UnifiedTransformer

pytestmark = [pytest.mark.unit, pytest.mark.network, pytest.mark.multimodal, pytest.mark.timeseries]


class _DummyEncoder(ModalityEncoder):
    def __init__(self, tokens_per_instance: int):
        super().__init__(output_dim=3)
        self._tokens_per_instance = tokens_per_instance

    def tokens_per_instance(self) -> int:
        return self._tokens_per_instance

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Just return the input for testing
        return inputs


def _build_model_with_ts_config(d_model=3, tokens_per_instance=4):
    config = ModelConfig(
        d_model=d_model,
        num_layers=1,
        num_heads=1,
        num_experts=1,
        vocab_size=128,
        d_text=d_model,
        modalities=["text"],
    )
    config.modality_start_end_token_indices = {"time_series": (32000, 32001)}
    model = UnifiedTransformer(config)
    model.config.modalities = ["text", "time_series"]
    model.encoders["time_series"] = _DummyEncoder(tokens_per_instance)
    return model


def _build_model_with_ts_img_config(d_model=3, ts_tokens_per_instance=2, img_tokens_per_instance=3):
    config = ModelConfig(
        d_model=d_model,
        num_layers=1,
        num_heads=1,
        num_experts=1,
        vocab_size=128,
        d_text=d_model,
        modalities=["text"],
    )
    config.modality_start_end_token_indices = {
        "time_series": (32000, 32001),
        "image": (33000, 33001),
    }
    model = UnifiedTransformer(config)
    model.config.modalities = ["text", "time_series", "image"]
    model.encoders["time_series"] = _DummyEncoder(ts_tokens_per_instance)
    model.encoders["image"] = _DummyEncoder(img_tokens_per_instance)
    return model


def test_merge_interleave_two_modalities():
    model = _build_model_with_ts_config(d_model=3)

    ts_start, ts_end = model.config.modality_start_end_token_indices["time_series"]
    pad_id = 0

    # <bos> text stuff <ts> </ts> text stuff <ts> </ts> text stuff <pad>
    # <bos> text stuff <ts> </ts> text stuff
    input_ids = torch.tensor(
        [
            [101, ts_start, ts_end, 7, ts_start, ts_end, 8, pad_id],
            [101, 9, ts_start, ts_end, 9, 10, 7, 8],
        ],
        dtype=torch.long,
    )
    inputs_embeds = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3) + 10

    time_series_features = torch.arange(1000, 1000 + 2 * 8 * 3, dtype=torch.float32).reshape(
        2, 8, 3
    )
    time_series_features[1, 4:] = 0  # Simulate padding for the second instance
    metadata = ["8 0", "8 0"]

    final_embedding, final_attention_mask, position_ids, final_labels = (
        model._merge_text_input_ids_with_modality_embeds(
            input_ids=input_ids,
            input_embeds=inputs_embeds,
            embeddings_list=[("text", inputs_embeds), ("time_series", time_series_features)],
            metadata=metadata,
            pad_id=pad_id,
        )
    )

    assert final_embedding.shape == (2, 11, 3)
    assert final_attention_mask.shape == (2, 11)
    assert position_ids.shape == (2, 11)
    assert final_labels.shape == (2, 11)

    assert final_attention_mask.sum(dim=-1).tolist() == [11, 10]
    assert position_ids[0, -1].item() == 10
    assert position_ids[1, -1].item() == 0
    assert position_ids[1, 9].item() == 9

    ignore_count = (final_labels == -100).sum().item()
    assert ignore_count == (2 * 11)  # 2 x max_len


def test_merge_interleave_three_modalities():
    model = _build_model_with_ts_img_config(
        d_model=3, ts_tokens_per_instance=2, img_tokens_per_instance=3
    )

    ts_start, ts_end = model.config.modality_start_end_token_indices["time_series"]
    img_start, img_end = model.config.modality_start_end_token_indices["image"]
    pad_id = 0

    input_ids = torch.tensor(
        [
            [101, ts_start, ts_end, 5, img_start, img_end, 7, 8, pad_id],
            [101, img_start, img_end, 9, ts_start, ts_end, ts_start, ts_end, 11],
        ],
        dtype=torch.long,
    )
    inputs_embeds = torch.arange(2 * 9 * 3, dtype=torch.float32).reshape(2, 9, 3) + 20
    metadata = ["8 1", "8 1"]

    time_series_features = torch.arange(1000, 1000 + 2 * 4 * 3, dtype=torch.float32).reshape(
        2, 4, 3
    )
    image_features = torch.arange(2000, 2000 + 2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)

    final_embedding, final_attention_mask, position_ids, final_labels = (
        model._merge_text_input_ids_with_modality_embeds(
            input_ids=input_ids,
            input_embeds=inputs_embeds,
            embeddings_list=[
                ("text", inputs_embeds),
                ("time_series", time_series_features),
                ("image", image_features),
            ],
            metadata=metadata,
            pad_id=pad_id,
        )
    )

    assert final_embedding.shape == (2, 10, 3)
    assert final_attention_mask.shape == (2, 10)
    assert position_ids.shape == (2, 10)
    assert final_labels.shape == (2, 10)

    assert final_attention_mask.sum(dim=-1).tolist() == [9, 10]
    assert position_ids[0, -1].item() == 0
    assert position_ids[1, -1].item() == 9

    # count number of ts features in final_embedding
    ts_feature_count = ((final_embedding >= 1000) & (final_embedding < 2000)).sum().item()
    assert ts_feature_count == 3 * 2 * 3  # 3 instances, 2 ts tokens each, 3 features per token
    # count number of image features in final_embedding
    img_feature_count = ((final_embedding >= 2000) & (final_embedding < 3000)).sum().item()
    assert img_feature_count == 2 * 3 * 3  # 2 instances,	3 img tokens each, 3 features per token

    ignore_count = (final_labels == -100).sum().item()
    assert ignore_count == (2 * 9)  # 2 x max_len-1


def test_merge_labels_prompt_target_alignment():
    """Verify full_labels places the correct answer token IDs at the correct
    positions after modality expansion, with no off-by-one errors.

    Batch element 0: one TS span in the prompt → sequence grows.
    Batch element 1: pure text (no modality tokens) with trailing padding.
    """
    model = _build_model_with_ts_config(d_model=3, tokens_per_instance=4)
    ts_start, ts_end = model.config.modality_start_end_token_indices["time_series"]
    pad_id = 0

    # Element 0 layout (no padding):
    #   Prompt (5): [101, <ts>, </ts>, 5, 6]
    #   Target (3): [10, 11, 12]
    #   input_ids len = 8
    #
    #   After merge (2 tokens → 4 ts tokens, growth = +2):
    #     [101, ts0, ts1, ts2, ts3, 5, 6, 10, 11, 12]
    #      0    1    2    3    4    5  6  7   8   9
    #   new_prompt_len = 5 + 2 = 7
    #   Expected labels: [-100]*7 + [10, 11, 12]
    #
    # Element 1 layout (3 pad tokens):
    #   Prompt (3): [101, 7, 8]
    #   Target (2): [20, 21]
    #   Pad:        [0, 0, 0]
    #
    #   After merge (no expansion, padding removed → 5 active, padded to max_len=10):
    #     [101, 7, 8, 20, 21, -, -, -, -, -]
    #      0    1  2  3   4
    #   new_prompt_len = 3
    #   Expected labels: [-100]*3 + [20, 21] + [-100]*5

    input_ids = torch.tensor(
        [
            [101, ts_start, ts_end, 5, 6, 10, 11, 12],
            [101, 7, 8, 20, 21, pad_id, pad_id, pad_id],
        ],
        dtype=torch.long,
    )
    inputs_embeds = torch.randn(2, 8, 3)
    ts_features = torch.randn(2, 4, 3)
    metadata = ["5 3", "3 2"]

    _, _, _, full_labels = model._merge_text_input_ids_with_modality_embeds(
        input_ids=input_ids,
        input_embeds=inputs_embeds,
        embeddings_list=[("text", inputs_embeds), ("time_series", ts_features)],
        metadata=metadata,
        pad_id=pad_id,
    )

    assert full_labels.shape[1] == 10, f"Expected max_len=10, got {full_labels.shape[1]}"

    # --- Element 0 ---
    # All prompt positions must be ignored
    assert (full_labels[0, :7] == -100).all(), (
        f"Elem 0 prompt: expected all -100, got {full_labels[0, :7].tolist()}"
    )
    # Boundary: last prompt vs first target
    assert full_labels[0, 6].item() == -100, "Off-by-one: last prompt position is not -100"
    assert full_labels[0, 7].item() == 10, "Off-by-one: first target position wrong"
    # Full target
    assert full_labels[0, 7:10].tolist() == [10, 11, 12]
    assert (full_labels[0] != -100).sum().item() == 3

    # --- Element 1 ---
    assert (full_labels[1, :3] == -100).all(), (
        f"Elem 1 prompt: expected all -100, got {full_labels[1, :3].tolist()}"
    )
    assert full_labels[1, 2].item() == -100, "Off-by-one: last prompt position is not -100"
    assert full_labels[1, 3].item() == 20, "Off-by-one: first target position wrong"
    assert full_labels[1, 3:5].tolist() == [20, 21]
    # Padding region must be -100
    assert (full_labels[1, 5:] == -100).all(), (
        f"Elem 1 padding: expected all -100, got {full_labels[1, 5:].tolist()}"
    )
    assert (full_labels[1] != -100).sum().item() == 2


def test_merge_labels_two_spans_target_alignment():
    """Two TS insertions in the prompt — total growth accumulates.
    Verifies that the label offset accounts for *all* expansions, not just one.
    """
    model = _build_model_with_ts_config(d_model=3, tokens_per_instance=3)
    ts_start, ts_end = model.config.modality_start_end_token_indices["time_series"]

    # Prompt (7): [101, <ts>, </ts>, 5, <ts>, </ts>, 6]
    # Target (2): [30, 31]
    # input_ids len = 9
    #
    # Each span: 2 input tokens → 3 ts tokens → +1 growth per span → total growth = +2
    # old_len = 9, total_len = 1+3+0+1+3+0+1+1+1 = 11
    # new_prompt_len = 7 + 2 = 9
    #
    # Merged: [101, ts0, ts1, ts2, 5, ts3, ts4, ts5, 6, 30, 31]
    #          0    1    2    3    4  5    6    7    8  9   10

    input_ids = torch.tensor(
        [[101, ts_start, ts_end, 5, ts_start, ts_end, 6, 30, 31]],
        dtype=torch.long,
    )
    inputs_embeds = torch.randn(1, 9, 3)
    ts_features = torch.randn(1, 6, 3)  # 2 spans × 3 tokens
    metadata = ["7 2"]

    _, _, _, full_labels = model._merge_text_input_ids_with_modality_embeds(
        input_ids=input_ids,
        input_embeds=inputs_embeds,
        embeddings_list=[("text", inputs_embeds), ("time_series", ts_features)],
        metadata=metadata,
        pad_id=None,
    )

    assert full_labels.shape == (1, 11)
    assert (full_labels[0, :9] == -100).all(), full_labels[0].tolist()
    assert full_labels[0, 8].item() == -100, "Off-by-one: last prompt position is not -100"
    assert full_labels[0, 9].item() == 30, "Off-by-one: first target position wrong"
    assert full_labels[0, 9:11].tolist() == [30, 31]
    assert (full_labels[0] != -100).sum().item() == 2


def test_merge_labels_modality_at_prompt_boundary():
    """TS span is the last thing in the prompt, immediately before the target.
    This is the tightest boundary — any off-by-one makes labels land on the
    expanded modality tokens or miss the first answer token.
    """
    model = _build_model_with_ts_config(d_model=3, tokens_per_instance=4)
    ts_start, ts_end = model.config.modality_start_end_token_indices["time_series"]

    # Prompt (4): [101, 5, <ts>, </ts>]
    # Target (2): [40, 41]
    # input_ids len = 6
    #
    # TS: 2 → 4 tokens → growth = +2
    # old_len = 6, total_len = 1+1+4+0+1+1 = 8
    # new_prompt_len = 4 + 2 = 6
    #
    # Merged: [101, 5, ts0, ts1, ts2, ts3, 40, 41]
    #          0    1  2    3    4    5    6   7

    input_ids = torch.tensor(
        [[101, 5, ts_start, ts_end, 40, 41]],
        dtype=torch.long,
    )
    inputs_embeds = torch.randn(1, 6, 3)
    ts_features = torch.randn(1, 4, 3)
    metadata = ["4 2"]

    _, _, _, full_labels = model._merge_text_input_ids_with_modality_embeds(
        input_ids=input_ids,
        input_embeds=inputs_embeds,
        embeddings_list=[("text", inputs_embeds), ("time_series", ts_features)],
        metadata=metadata,
        pad_id=None,
    )

    assert full_labels.shape == (1, 8)
    assert (full_labels[0, :6] == -100).all(), full_labels[0].tolist()
    assert full_labels[0, 5].item() == -100, "Off-by-one: last ts token position is not -100"
    assert full_labels[0, 6].item() == 40, "Off-by-one: first target position wrong"
    assert full_labels[0, 6:8].tolist() == [40, 41]
    assert (full_labels[0] != -100).sum().item() == 2


def test_merge_labels_single_token_target():
    """Edge case: target is exactly one token. Off-by-one would cause either
    zero valid labels or the label landing on the wrong position.
    """
    model = _build_model_with_ts_config(d_model=3, tokens_per_instance=4)
    ts_start, ts_end = model.config.modality_start_end_token_indices["time_series"]

    # Prompt (5): [101, <ts>, </ts>, 5, 6]
    # Target (1): [50]
    # input_ids len = 6
    #
    # growth = +2 → total_len = 8, new_prompt_len = 7
    #
    # Merged: [101, ts0, ts1, ts2, ts3, 5, 6, 50]
    #          0    1    2    3    4    5  6  7

    input_ids = torch.tensor(
        [[101, ts_start, ts_end, 5, 6, 50]],
        dtype=torch.long,
    )
    inputs_embeds = torch.randn(1, 6, 3)
    ts_features = torch.randn(1, 4, 3)
    metadata = ["5 1"]

    _, _, _, full_labels = model._merge_text_input_ids_with_modality_embeds(
        input_ids=input_ids,
        input_embeds=inputs_embeds,
        embeddings_list=[("text", inputs_embeds), ("time_series", ts_features)],
        metadata=metadata,
        pad_id=None,
    )

    assert full_labels.shape == (1, 8)
    assert (full_labels[0, :7] == -100).all(), full_labels[0].tolist()
    assert full_labels[0, 6].item() == -100, "Off-by-one: last prompt position is not -100"
    assert full_labels[0, 7].item() == 50, "Off-by-one: single target token wrong"
    assert (full_labels[0] != -100).sum().item() == 1


def _over_length_merge_inputs():
    """Shared fixture for guard tests: a single-sample interleaved merge whose
    merged length is a known 10 tokens.

    Prompt (5): [101, <ts>, </ts>, 5, 6]  Target (3): [10, 11, 12]
    TS span: 2 input tokens -> 4 ts tokens (growth +2) -> total_len = 10.
    """
    model = _build_model_with_ts_config(d_model=3, tokens_per_instance=4)
    ts_start, ts_end = model.config.modality_start_end_token_indices["time_series"]
    input_ids = torch.tensor(
        [[101, ts_start, ts_end, 5, 6, 10, 11, 12]],
        dtype=torch.long,
    )
    inputs_embeds = torch.randn(1, 8, 3)
    ts_features = torch.randn(1, 4, 3)
    metadata = ["5 3"]
    kwargs = dict(
        input_ids=input_ids,
        input_embeds=inputs_embeds,
        embeddings_list=[("text", inputs_embeds), ("time_series", ts_features)],
        metadata=metadata,
        pad_id=None,
    )
    return model, kwargs


def test_merged_length_guard_raises():
    """Guard (issue #123): merged length over max_merged_seq_length raises a
    clear ValueError naming the merged length and the limit."""
    model, kwargs = _over_length_merge_inputs()
    model.config.max_merged_seq_length = 8  # merged length is 10 > 8
    model.config.merged_seq_length_guard = "error"

    with pytest.raises(ValueError) as excinfo:
        model._merge_text_input_ids_with_modality_embeds(**kwargs)

    msg = str(excinfo.value)
    assert "10" in msg, f"merged length not reported: {msg}"
    assert "8" in msg, f"limit not reported: {msg}"


def test_merged_length_guard_warn_mode(caplog):
    """In warn mode the guard logs but does not raise, and the merge completes."""
    import logging

    model, kwargs = _over_length_merge_inputs()
    model.config.max_merged_seq_length = 8
    model.config.merged_seq_length_guard = "warn"

    with caplog.at_level(logging.WARNING):
        final_embedding, _, _, _ = model._merge_text_input_ids_with_modality_embeds(
            **kwargs
        )

    assert final_embedding.shape == (1, 10, 3)
    assert any("max_merged_seq_length" in r.message for r in caplog.records), (
        "expected a warning mentioning max_merged_seq_length"
    )


def test_merged_length_guard_disabled_by_default():
    """With max_merged_seq_length unset (None, the default), an over-length
    merge succeeds unchanged — the guard is inert unless explicitly enabled."""
    model, kwargs = _over_length_merge_inputs()
    assert model.config.max_merged_seq_length is None

    final_embedding, _, _, _ = model._merge_text_input_ids_with_modality_embeds(
        **kwargs
    )
    assert final_embedding.shape == (1, 10, 3)


if __name__ == "__main__":
    test_merge_interleave_two_modalities()
    test_merge_interleave_three_modalities()
    test_merge_labels_prompt_target_alignment()
    test_merge_labels_two_spans_target_alignment()
    test_merge_labels_modality_at_prompt_boundary()
    test_merge_labels_single_token_target()
    test_merged_length_guard_raises()
    test_merged_length_guard_warn_mode()
    test_merged_length_guard_disabled_by_default()
    print("All tests passed!")

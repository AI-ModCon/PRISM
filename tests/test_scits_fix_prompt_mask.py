"""CPU unit tests for defect J: the prefix-path prompt-mask clamp in
UnifiedTransformer.forward() must bound `_prompt_len` against the sample's
real (unpadded) content length, not the batch's padded width.

Uses the same MagicMock-backbone pattern as tests/test_resize_token_embeddings.py
to avoid downloading a real HF checkpoint.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from src.config import ModelConfig
from src.model import UnifiedTransformer

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]

PAD_ID = 0


class _StubBackbone(nn.Module):
    """Minimal HF-like backbone: an embedding + config.hidden_size."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self._embed = nn.Embedding(vocab_size, hidden_size)
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.dtype = torch.float32

    def get_input_embeddings(self):
        return self._embed

    def resize_token_embeddings(self, new_size: int):
        self._embed = nn.Embedding(new_size, self._embed.embedding_dim)
        return self._embed

    def parameters(self, recurse: bool = True):
        return self._embed.parameters(recurse=recurse)

    def forward(self, inputs_embeds, attention_mask=None, labels=None,
                return_dict=True, use_cache=False, output_hidden_states=False):
        B, T, _ = inputs_embeds.shape
        logits = torch.zeros(B, T, self._embed.num_embeddings)
        out = MagicMock()
        out.logits = logits
        return out


def _make_tokenizer(vocab_size: int):
    tok = MagicMock()
    tok.__len__ = lambda self=None: vocab_size
    tok.pad_token_id = PAD_ID
    return tok


def _config():
    return ModelConfig(
        d_model=8,
        llm_backbone_id="Qwen/Qwen3-0.6B",
        modalities=["text"],
        is_interleaved_qa=False,
    )


def _build_model(vocab_size: int = 64, hidden_size: int = 8):
    backbone = _StubBackbone(vocab_size=vocab_size, hidden_size=hidden_size)
    tokenizer = _make_tokenizer(vocab_size)
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=backbone),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        model = UnifiedTransformer(_config())
    model.eval()
    return model


def _resolve_pad_id_mock(self):
    return PAD_ID


def test_prompt_mask_clamp_uses_real_length_not_padded_width(offline_hf):
    """Regression for defect J: a short sample (its answer occupies only the
    first few real tokens, right-padded to the batch's max width with
    PAD_ID) must still retain >=1 supervised token after prompt masking.

    Before the fix, `_pl` was clamped against `assigned.shape[1] - 1` (the
    padded batch width), so a row shorter than that width had every real
    token masked to -100 as well -> NaN loss on that row.
    """
    model = _build_model()
    with patch.object(UnifiedTransformer, "_resolve_pad_id", _resolve_pad_id_mock):
        # Batch of 2: row 0 is a short real sample (3 real tokens, prompt_len=2,
        # so 1 real answer token), right-padded with PAD_ID to width 8 to match
        # row 1's length.
        text_ids = torch.tensor(
            [
                [5, 6, 7, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
                [5, 6, 7, 8, 9, 10, 11, 12],
            ],
            dtype=torch.long,
        )
        labels = text_ids.clone()
        prompt_lens = torch.tensor([2, 2], dtype=torch.long)

        inputs = {"text": text_ids, "_prompt_len": prompt_lens}
        logits, loss = model(inputs, labels=labels)

    assert loss is not None
    assert torch.isfinite(loss), "loss is NaN/inf — short row's answer token was fully masked"


def test_prompt_mask_clamp_all_pad_row_is_skipped_not_negative_length(offline_hf):
    """A row that is entirely PAD (real_len == 0) must be skipped, not
    crash on a negative-length slice."""
    model = _build_model()
    with patch.object(UnifiedTransformer, "_resolve_pad_id", _resolve_pad_id_mock):
        text_ids = torch.tensor(
            [
                [PAD_ID] * 6,
                [5, 6, 7, 8, 9, 10],
            ],
            dtype=torch.long,
        )
        labels = text_ids.clone()
        prompt_lens = torch.tensor([3, 3], dtype=torch.long)

        inputs = {"text": text_ids, "_prompt_len": prompt_lens}
        # Must not raise.
        logits, loss = model(inputs, labels=labels)

    assert torch.isfinite(loss)

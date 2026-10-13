"""Forecast samples must be supervised only by the forecast decoder.

SciTS forecast captions are pure prompts (no ``Answer:`` delimiter), so
``_prompt_len`` is 0 and, without row masking, next-token CE ran over the whole
prompt alongside the pinball loss. Uses the stub-backbone pattern from
tests/test_scits_fix_prompt_mask.py to avoid loading a real checkpoint.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from src.config import ModelConfig
from src.model import UnifiedTransformer

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]

PAD_ID = 0
AUX_LOSS = 0.7


class _StubBackbone(nn.Module):
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
        out = MagicMock()
        out.logits = inputs_embeds.sum(-1, keepdim=True) * 0.0 + torch.randn(
            B, T, self._embed.num_embeddings
        )
        out.hidden_states = (inputs_embeds,) if output_hidden_states else None
        return out


class _StubForecastDecoder(nn.Module):
    def forward(self, hidden, targets=None, **kwargs):
        return hidden, torch.tensor(AUX_LOSS)


def _build_model(with_forecast_decoder: bool = True):
    backbone = _StubBackbone(vocab_size=64, hidden_size=8)
    tokenizer = MagicMock()
    tokenizer.__len__ = lambda self=None: 64
    tokenizer.pad_token_id = PAD_ID
    config = ModelConfig(
        d_model=8, llm_backbone_id="Qwen/Qwen3-0.6B", modalities=["text"], is_interleaved_qa=False
    )
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=backbone),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        model = UnifiedTransformer(config)
    if with_forecast_decoder:
        model.decoders["time_series"] = _StubForecastDecoder()
    return model.eval()


def _run(model, target_rows):
    """Forward a 2-row batch; ``target_rows[i]`` marks row i as a forecast sample."""
    text = torch.tensor([[5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16]], dtype=torch.long)
    target_mask = torch.zeros(2, 4, 1, dtype=torch.bool)
    for row, has_target in enumerate(target_rows):
        target_mask[row] = has_target
    inputs = {
        "text": text,
        "_prompt_len": torch.zeros(2, dtype=torch.long),
        "time_series_target": torch.zeros(2, 4, 1),
        "time_series_target_mask": target_mask,
    }
    captured = {}
    original = model.text_decoder.forward

    def spy(logits, targets=None, **kwargs):
        captured["labels"] = targets.clone()
        return original(logits, targets=targets, **kwargs)

    with (
        patch.object(UnifiedTransformer, "_resolve_pad_id", lambda self: PAD_ID),
        patch.object(model.text_decoder, "forward", side_effect=spy),
    ):
        _, loss = model(inputs, labels=text.clone())
    return captured.get("labels"), loss


def test_forecast_row_text_labels_fully_masked_and_text_row_kept(offline_hf):
    labels, loss = _run(_build_model(), target_rows=[True, False])
    assert (labels[0] == -100).all()
    assert (labels[1] != -100).any()
    assert torch.isfinite(loss)


def test_all_forecast_batch_loss_is_forecast_loss_only(offline_hf):
    labels, loss = _run(_build_model(), target_rows=[True, True])
    assert labels is None, "text CE must be skipped when every row is a forecast sample"
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(AUX_LOSS)


def test_no_forecast_decoder_leaves_text_labels_untouched(offline_hf):
    labels, _ = _run(_build_model(with_forecast_decoder=False), target_rows=[True, True])
    assert (labels != -100).any(dim=1).all()


def test_mixed_batch_reports_text_and_forecast_components(offline_hf):
    model = _build_model()
    labels, loss = _run(model, target_rows=[True, False])
    components = model.last_loss_components
    text_loss, text_count = components["text"]
    ts_loss, ts_count = components["time_series"]
    assert int(text_count) == int((labels[:, 1:] != -100).sum()) > 0
    assert int(ts_count) == 1
    assert float(ts_loss) == pytest.approx(AUX_LOSS)
    assert float(text_loss) + float(ts_loss) == pytest.approx(float(loss), rel=1e-5)
    assert not text_loss.requires_grad


def test_all_forecast_batch_reports_no_text_component(offline_hf):
    model = _build_model()
    _run(model, target_rows=[True, True])
    assert "text" not in model.last_loss_components
    assert int(model.last_loss_components["time_series"][1]) == 2


def test_components_reset_between_forwards(offline_hf):
    model = _build_model()
    _run(model, target_rows=[True, False])
    _run(model, target_rows=[True, True])
    assert "text" not in model.last_loss_components

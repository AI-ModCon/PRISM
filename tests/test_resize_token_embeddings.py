"""When the configured tokenizer is larger than the backbone's embedding matrix
(e.g. PRISM's custom tokenizer that adds <ts>/<image>/... modality tokens),
UnifiedTransformer must call resize_token_embeddings so input ids beyond the
original vocab are valid.
"""
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from src.config import ModelConfig
from src.model import UnifiedTransformer

pytestmark = pytest.mark.unit


class _StubBackbone(nn.Module):
    """Minimal HF-like backbone: an embedding + config.hidden_size."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self._embed = nn.Embedding(vocab_size, hidden_size)
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.resize_calls: list[int] = []

    def get_input_embeddings(self):
        return self._embed

    def resize_token_embeddings(self, new_size: int):
        self.resize_calls.append(new_size)
        self._embed = nn.Embedding(new_size, self._embed.embedding_dim)
        return self._embed

    def parameters(self, recurse: bool = True):
        return self._embed.parameters(recurse=recurse)


def _make_tokenizer(length: int):
    tok = MagicMock()
    tok.__len__ = lambda self=tok: length
    return tok


def _config():
    return ModelConfig(
        d_model=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        llm_backbone_id="Qwen/Qwen3-0.6B",
        modalities=["text", "image"],
    )


def test_resize_invoked_when_tokenizer_larger_than_embedding(offline_hf):
    backbone = _StubBackbone(vocab_size=100, hidden_size=32)
    tokenizer = _make_tokenizer(112)

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=backbone),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        model = UnifiedTransformer(_config())

    assert backbone.resize_calls == [112], backbone.resize_calls
    assert model.backbone.get_input_embeddings().weight.shape[0] == 112


def test_resize_skipped_when_sizes_match(offline_hf):
    backbone = _StubBackbone(vocab_size=128, hidden_size=32)
    tokenizer = _make_tokenizer(128)

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=backbone),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        UnifiedTransformer(_config())

    assert backbone.resize_calls == []


def test_resize_never_shrinks_below_native_embedding(offline_hf):
    """Regression for issue #117: a tokenizer SHORTER than the backbone's
    native embedding must NOT shrink the table. OLMo-1B ships a padded 50304-row
    embedding while the custom interleaved tokenizer is len 50292; the old
    `len(embed) != len(tokenizer)` check shrank it to 50292 (and the distributed
    path shrank it all the way to the base tokenizer's 50280), dropping the rows
    that <ts>=50280/<ts/>=50281 index into. On XPU that OOB embedding gather
    corrupts device memory and later faults (`drm_neo.cpp`, write page-fault).
    The resize must be grow-only: never below the current embedding size.
    """
    backbone = _StubBackbone(vocab_size=50304, hidden_size=32)
    tokenizer = _make_tokenizer(50292)  # custom tokenizer, shorter than native

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=backbone),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        model = UnifiedTransformer(_config())

    # No shrink: table stays at least as large as the native embedding so every
    # valid token id (including the modality tokens) remains in bounds.
    assert backbone.resize_calls == [], backbone.resize_calls
    assert model.backbone.get_input_embeddings().weight.shape[0] == 50304


def test_resize_deterministic_across_ranks(offline_hf):
    """Without sync_module_states FSDP would diverge; the resize path seeds
    RNG so every rank produces identical new rows."""
    results = []
    for _ in range(2):
        torch.manual_seed(999)  # different per-rank RNG before resize
        backbone = _StubBackbone(vocab_size=100, hidden_size=8)
        tokenizer = _make_tokenizer(110)
        with (
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=backbone),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        ):
            UnifiedTransformer(_config())
        results.append(backbone.get_input_embeddings().weight.detach().clone())

    torch.testing.assert_close(results[0], results[1])

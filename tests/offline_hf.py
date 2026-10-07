"""Offline stand-ins for the HuggingFace downloads `UnifiedTransformer` reaches.

Building a `UnifiedTransformer` against a real backbone id touches up to six
distinct download sites, and which ones fire depends on the modality list:

    src/hf_cache.py:32            snapshot_download        (backbone tokenizer)
    src/model.py:117/141/183      AutoModelForCausalLM     (backbone weights)
    src/encoders/image.py:26      AutoModel                (SigLIP vision tower)
    src/encoders/text.py:28-29    AutoTokenizer/AutoModel  (external text encoder)
    src/encoders/time_series.py   Moirai2Module/hf_hub_download
    src/encoders/geometry.py:206  hf_hub_download          (walrus.pt)

Stubbing one is not enough, which is why the tests that build a real model were
first marked `network` rather than patched in place. This module holds the
stubs; the `offline_hf` and `offline_backbone` fixtures in `conftest.py` install
them.

The trap worth recording: `load_cached_tokenizer` calls `snapshot_download`
*before* it ever reaches `AutoTokenizer.from_pretrained`, so the obvious
`patch("transformers.AutoTokenizer.from_pretrained")` never intercepts it. And
because `src/model.py:20` does `from .hf_cache import load_cached_tokenizer`,
the name to rebind is `src.model.load_cached_tokenizer`, not the one in
`src.hf_cache`. The encoders have the same shape — they do
`from transformers import AutoModel` at module scope.

The stubs carry real shapes (`config.hidden_size`, real `nn.Parameter`s) because
the encoders read those to size their projections. They are deliberately tiny:
the point is exercising assembly logic, not weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

# Larger than any PRISM modality-token id, so a test that falls through to this
# tokenizer sees a grow-only resize rather than an accidental shrink.
STUB_VOCAB_SIZE = 1024
STUB_HIDDEN_SIZE = 64


class StubTokenizer:
    """Minimal tokenizer covering what `UnifiedTransformer.__init__` calls.

    `src/model.py` uses `len()`, `get_vocab()`, `add_special_tokens()` and
    `convert_tokens_to_ids()` on the backbone tokenizer; nothing else.
    """

    def __init__(self, vocab_size: int = STUB_VOCAB_SIZE):
        self._vocab = {f"tok{i}": i for i in range(vocab_size)}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __len__(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def add_special_tokens(self, payload: Any) -> int:
        tokens = payload
        if isinstance(payload, dict):
            tokens = payload.get("additional_special_tokens", [])
        added = 0
        for tok in tokens:
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, 0)


class StubEncoderModel(nn.Module):
    """Stands in for a SigLIP vision tower or an external text encoder.

    Carries a real `config.hidden_size` so `ImageEncoder`/`TextEncoder` size
    their projections exactly as they would against the real checkpoint.
    """

    def __init__(self, hidden_size: int = STUB_HIDDEN_SIZE, seq_len: int = 4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        # A real parameter, so `.parameters()` is non-empty and `.to(device)`
        # plus freeze/unfreeze behave as they do with a real encoder.
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs: Any = None, *args: Any, **kwargs: Any) -> Any:
        batch = getattr(inputs, "shape", (1,))[0]
        hidden = torch.zeros(batch, self.seq_len, self.hidden_size)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=hidden[:, 0])


class AutoStub:
    """Stands in for the `AutoModel` / `AutoTokenizer` class object itself.

    The encoders hold a reference to the class and call `.from_pretrained` on
    it, so the replacement has to be an object exposing that method.
    """

    def __init__(self, tokenizer: bool = False):
        self._tokenizer = tokenizer

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        return StubTokenizer() if self._tokenizer else StubEncoderModel()


def build_tiny_causal_lm(transformers: Any) -> Any:
    """A minimal real `Qwen3ForCausalLM`, for tests that run a forward pass.

    Same approach `tests/test_time_series_forecast_config.py` already uses: a
    genuine model small enough to construct in a second, so no weights are
    fetched but the arithmetic is real. Seeded, so repeated construction inside
    one test session is deterministic.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(937)
        config = transformers.Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=64,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        config._attn_implementation = "eager"
        return transformers.Qwen3ForCausalLM(config)

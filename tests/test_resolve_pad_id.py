"""Behavioral coverage for UnifiedTransformer._resolve_pad_id.

The helper has no model-state dependencies — it walks two attributes
(`backbone_tokenizer`, then `tokenizer`) and returns the first
non-None `pad_token_id`. We call it as an unbound method against a
stub `self` to keep the test free of the heavy model fixtures.
"""

from types import SimpleNamespace

import pytest
from src.model import UnifiedTransformer

pytestmark = [pytest.mark.unit]


def _resolve(backbone_tok=None, tok=None):
    stub = SimpleNamespace()
    if backbone_tok is not None:
        stub.backbone_tokenizer = backbone_tok
    if tok is not None:
        stub.tokenizer = tok
    return UnifiedTransformer._resolve_pad_id(stub)


def test_path_a_backbone_tokenizer_wins():
    """When backbone_tokenizer has a pad_token_id, prefer it over self.tokenizer."""
    assert _resolve(
        backbone_tok=SimpleNamespace(pad_token_id=1),
        tok=SimpleNamespace(pad_token_id=2),
    ) == 1


def test_path_b_falls_back_to_self_tokenizer():
    """Path B has no backbone_tokenizer; fall back to self.tokenizer."""
    assert _resolve(tok=SimpleNamespace(pad_token_id=7)) == 7


def test_backbone_tokenizer_with_none_pad_falls_through():
    """If backbone_tokenizer.pad_token_id is None, skip and try self.tokenizer."""
    assert _resolve(
        backbone_tok=SimpleNamespace(pad_token_id=None),
        tok=SimpleNamespace(pad_token_id=5),
    ) == 5


def test_no_tokenizer_at_all_returns_none():
    """Neither attribute set → None (caller skips the masked_fill)."""
    assert _resolve() is None

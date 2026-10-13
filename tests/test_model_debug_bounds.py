"""PRISM_DEBUG_BOUNDS guards against out-of-range token ids reaching XPU kernels."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.model import UnifiedTransformer, _check_token_bounds

pytestmark = pytest.mark.unit


def _fake_model():
    embedding = nn.Embedding(10, 4)
    return SimpleNamespace(backbone=SimpleNamespace(get_input_embeddings=lambda: embedding))


def test_out_of_range_text_ids_fail_loudly_when_enabled(monkeypatch):
    monkeypatch.setenv("PRISM_DEBUG_BOUNDS", "1")
    with pytest.raises(ValueError, match="input_ids"):
        UnifiedTransformer.get_input_embeddings(_fake_model(), {"text": torch.tensor([[1, 10]])})


def test_in_range_text_ids_pass_and_check_is_off_by_default(monkeypatch):
    monkeypatch.setenv("PRISM_DEBUG_BOUNDS", "1")
    out = UnifiedTransformer.get_input_embeddings(_fake_model(), {"text": torch.tensor([[0, 9]])})
    assert out.shape == (1, 2, 4)
    monkeypatch.delenv("PRISM_DEBUG_BOUNDS")
    _check_token_bounds("labels", torch.tensor([[50]]), 10)


@pytest.mark.parametrize(
    ("labels", "ok"),
    [([[-100, 0, 9]], True), ([[-100, 10]], False), ([[-5, 1]], False)],
)
def test_label_bounds_ignore_only_ignore_index(monkeypatch, labels, ok):
    monkeypatch.setenv("PRISM_DEBUG_BOUNDS", "1")
    labels = torch.tensor(labels)
    if ok:
        _check_token_bounds("labels", labels, 10, ignore_index=-100)
    else:
        with pytest.raises(ValueError, match="labels"):
            _check_token_bounds("labels", labels, 10, ignore_index=-100)

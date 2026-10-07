"""Phase 0 tests for the output-decoder abstraction.

Locks the invariants of the output-decoder abstraction:
- the DECODERS registry is well-formed,
- LMHeadDecoder reproduces the old inline next-token CE bit-for-bit,
- RegressionDecoder reproduces the old VLA action-head math bit-for-bit,
- the is_vla -> output_decoders back-compat mapping (config-level).

These are pure-unit tests: no backbone, no network. The full no-regression
guarantee is covered by tests/test_generation.py and
tests/test_vla_regressions.py still passing after the refactor.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from src.decoders import (
    DECODERS,
    LMHeadDecoder,
    OutputDecoder,
    RegressionDecoder,
)

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_entries_are_output_decoders():
    assert DECODERS, "registry must not be empty"
    for name, cls in DECODERS.items():
        assert isinstance(name, str)
        assert issubclass(cls, OutputDecoder), f"{name} -> {cls} is not an OutputDecoder"


def test_registry_known_names():
    # text + the VLA action head must be registered (Phase 0 scope).
    assert DECODERS["text"] is LMHeadDecoder
    assert DECODERS["action"] is RegressionDecoder


def test_decoder_descriptive_attrs():
    assert LMHeadDecoder.output_kind == "text"
    assert LMHeadDecoder.loss_kind == "cross_entropy"
    assert RegressionDecoder.output_kind == "tensor"
    assert RegressionDecoder.loss_kind == "mse"


# --------------------------------------------------------------------------
# LMHeadDecoder — bit-identical to the pre-refactor inline CE
# --------------------------------------------------------------------------


def _reference_text_loss(logits, labels):
    """The exact computation that used to live inline in model.forward."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def test_lm_head_decoder_bit_identical_loss():
    torch.manual_seed(0)
    logits = torch.randn(3, 6, 11)
    labels = torch.randint(0, 11, (3, 6))
    labels[0, 0] = -100  # exercise the ignore_index path
    labels[2, 3] = -100

    out_logits, loss = LMHeadDecoder()(logits, targets=labels)
    ref = _reference_text_loss(logits, labels)

    assert torch.equal(out_logits, logits), "logits must pass through unchanged"
    assert torch.equal(loss, ref), (loss.item(), ref.item())


def test_lm_head_decoder_inference_returns_none_loss():
    logits = torch.randn(2, 4, 5)
    out_logits, loss = LMHeadDecoder()(logits, targets=None)
    assert torch.equal(out_logits, logits)
    assert loss is None


def test_lm_head_decoder_is_parameterless():
    # On the HF-backbone path the LM head lives in the backbone, so this
    # decoder must own no parameters (keeps checkpoint keys unchanged).
    assert list(LMHeadDecoder().parameters()) == []


# --------------------------------------------------------------------------
# RegressionDecoder — bit-identical to the pre-refactor VLA action head
# --------------------------------------------------------------------------


def test_regression_decoder_head_shape_and_structure():
    dec = RegressionDecoder(8, 7)
    feats = torch.randn(4, 8)
    pred = dec.predict(feats)
    assert pred.shape == (4, 7)
    # MLP must be Linear -> ReLU -> Linear (3 modules), matching the old head.
    assert len(dec.head) == 3


def test_regression_decoder_loss_terms_match_reference():
    torch.manual_seed(1)
    dec = RegressionDecoder(8, 7)
    pred = dec.predict(torch.randn(5, 8))
    target = torch.randn(5, 7)

    loss, per_dim = dec.loss_terms(pred, target)
    ref_per_dim = F.mse_loss(pred, target, reduction="none").mean(dim=0)
    ref_loss = ref_per_dim.mean()

    assert torch.equal(per_dim, ref_per_dim)
    assert torch.equal(loss, ref_loss)
    assert per_dim.shape == (7,)


def test_regression_decoder_forward_matches_predict_plus_loss():
    torch.manual_seed(2)
    dec = RegressionDecoder(6, 3)
    feats = torch.randn(4, 6)
    target = torch.randn(4, 3)

    pred_a = dec.predict(feats)
    loss_a, _ = dec.loss_terms(pred_a, target)
    pred_b, loss_b = dec(feats, targets=target)

    assert torch.equal(pred_a, pred_b)
    assert torch.equal(loss_a, loss_b)


def test_regression_decoder_shape_mismatch_raises():
    dec = RegressionDecoder(6, 3)
    feats = torch.randn(4, 6)
    bad_target = torch.randn(4, 5)  # wrong output dim
    with pytest.raises(RuntimeError, match="shape"):
        dec(feats, targets=bad_target)


def test_regression_decoder_inference_returns_none_loss():
    dec = RegressionDecoder(6, 3)
    pred, loss = dec(torch.randn(2, 6), targets=None)
    assert pred.shape == (2, 3)
    assert loss is None

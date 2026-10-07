"""Checkpoint back-compat for the OutputDecoder refactor.

The Phase 0 refactor moved the VLA action head from a bare nn.Sequential to a
RegressionDecoder wrapper, renaming its params:

    action_head.{0,2}.{weight,bias}  ->  action_head.head.{0,2}.{weight,bias}

Both native weights-only loading and Accelerate's direct model-state loading
must restore pre-refactor VLA checkpoints without changing their parameters.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from src.decoders import remap_legacy_decoder_keys
from src.decoders.regression import RegressionDecoder
from torch import nn

pytestmark = pytest.mark.unit


def test_remaps_legacy_action_head_keys():
    old = {
        "action_head.0.weight": torch.zeros(3, 3),
        "action_head.0.bias": torch.zeros(3),
        "action_head.2.weight": torch.zeros(7, 3),
        "action_head.2.bias": torch.zeros(7),
    }
    new = remap_legacy_decoder_keys(old)
    assert set(new) == {
        "action_head.head.0.weight",
        "action_head.head.0.bias",
        "action_head.head.2.weight",
        "action_head.head.2.bias",
    }


def test_preserves_tensor_values():
    w = torch.randn(7, 3)
    new = remap_legacy_decoder_keys({"action_head.2.weight": w})
    assert torch.equal(new["action_head.head.2.weight"], w)


def test_idempotent_on_new_layout():
    already = {
        "action_head.head.0.weight": torch.zeros(3, 3),
        "action_head.head.2.bias": torch.zeros(7),
    }
    new = remap_legacy_decoder_keys(already)
    assert set(new) == set(already)


def test_leaves_unrelated_keys_untouched():
    sd = {
        "backbone.layers.0.weight": torch.zeros(2, 2),
        "projectors.image.fc1.weight": torch.zeros(2, 2),
        "pose_embed.0.weight": torch.zeros(2, 2),
    }
    new = remap_legacy_decoder_keys(sd)
    assert set(new) == set(sd)


def test_does_not_mutate_input():
    old = {"action_head.0.weight": torch.zeros(3, 3)}
    snapshot = set(old)
    remap_legacy_decoder_keys(old)
    assert set(old) == snapshot  # original dict keys unchanged


class _TinyVLA(nn.Module):
    def __init__(self, legacy):
        super().__init__()
        self.action_head = (
            nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 7))
            if legacy else RegressionDecoder(4, 7)
        )

    def forward(self, features):
        if isinstance(self.action_head, RegressionDecoder):
            return self.action_head.predict(features)
        return self.action_head(features)


@pytest.mark.parametrize("legacy", [True, False])
def test_recursive_strict_load_accepts_legacy_and_current_layout(legacy):
    source = _TinyVLA(legacy)
    destination = _TinyVLA(False)
    state = deepcopy(source.state_dict())
    keys_before = list(state)
    destination.load_state_dict(state, strict=True)
    assert list(state) == keys_before
    features = torch.randn(2, 4)
    torch.testing.assert_close(destination(features), source(features), rtol=0, atol=0)
    assert set(destination.state_dict()) == {
        "action_head.head.0.weight", "action_head.head.0.bias",
        "action_head.head.2.weight", "action_head.head.2.bias",
    }


@pytest.mark.parametrize("strict", [True, False])
def test_conflicting_legacy_alias_cannot_override_current_weights(strict):
    model = _TinyVLA(False)
    state = deepcopy(model.state_dict())
    state["action_head.0.weight"] = state["action_head.head.0.weight"] + 1
    with pytest.raises(RuntimeError, match="Ambiguous regression checkpoint"):
        model.load_state_dict(state, strict=strict)


def test_legacy_strict_load_still_rejects_missing_and_unexpected_keys():
    model = _TinyVLA(False)
    state = deepcopy(_TinyVLA(True).state_dict())
    state.pop("action_head.2.bias")
    state["action_head.unknown"] = torch.zeros(1)
    with pytest.raises(RuntimeError) as error:
        model.load_state_dict(state, strict=True)
    assert 'Missing key(s) in state_dict: "action_head.head.2.bias"' in str(error.value)
    assert 'Unexpected key(s) in state_dict: "action_head.unknown"' in str(error.value)


def test_accelerate_legacy_full_state_restores_optimizer_and_next_update(tmp_path):
    Accelerator = pytest.importorskip("accelerate").Accelerator
    accelerator = Accelerator(cpu=True)
    legacy = _TinyVLA(True)
    optimizer = torch.optim.AdamW(legacy.parameters(), lr=1e-3)
    legacy, optimizer = accelerator.prepare(legacy, optimizer)
    features = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    target = torch.arange(21, dtype=torch.float32).reshape(3, 7) / 20

    def update(model, optim):
        optim.zero_grad()
        accelerator.backward((model(features) - target).square().mean())
        optim.step()

    try:
        update(legacy, optimizer)
        saved_prediction = legacy(features).detach().clone()
        saved_optimizer = deepcopy(optimizer.state_dict())
        accelerator.save_state(str(tmp_path / "checkpoint"))
        update(legacy, optimizer)
        expected_prediction = legacy(features).detach().clone()
        expected_optimizer = deepcopy(optimizer.state_dict())
        accelerator.free_memory()

        restored = _TinyVLA(False)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored, restored_optimizer = accelerator.prepare(restored, restored_optimizer)
        # This is the actual ZoneAVLATrainer resume API, including Adam state.
        accelerator.load_state(str(tmp_path / "checkpoint"))
        torch.testing.assert_close(restored(features), saved_prediction, rtol=0, atol=0)
        torch.testing.assert_close(restored_optimizer.state_dict(), saved_optimizer, rtol=0, atol=0)
        update(restored, restored_optimizer)
        torch.testing.assert_close(restored(features), expected_prediction, rtol=0, atol=0)
        torch.testing.assert_close(restored_optimizer.state_dict(), expected_optimizer, rtol=0, atol=0)
    finally:
        accelerator.free_memory()

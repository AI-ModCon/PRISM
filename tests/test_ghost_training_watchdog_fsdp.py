"""Regression test for the ghost-training watchdog under FSDP wrapping.

The watchdog in ZoneATrainer.check_parameter_status uses
    n.startswith("projectors.text")
to identify text-projector params and decide whether the model has any
NON-text modalities. Under FSDP, named_parameters() returns names like
    _fsdp_wrapped_module.projectors.text.weight
and the bare prefix check returns False — incorrectly flagging the text
projector as a non-text modality, which can fire a false-positive
"Ghost Training detected. Aborting." on FSDP text_only runs.

The fix iterates `self._unwrap_model().named_parameters()` instead.
This file verifies:
  1. The watchdog source still iterates the unwrapped model (not self.model).
  2. The predicate evaluates correctly when names DO have the FSDP prefix.
  3. The predicate evaluates correctly on bare names (DDP / unwrapped).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAINER = ROOT / "src" / "training" / "trainer_zone_a.py"


def _non_text_modality_present(named_params_iter):
    """Mirror of the predicate in src/training/trainer_zone_a.py."""
    return any(
        "projectors." in n and not n.startswith("projectors.text")
        for n, _ in named_params_iter
    )


# --- Predicate behavioral tests (mirror the trainer expression) ---

def test_predicate_on_bare_names_text_only():
    """text_only model under DDP: only projectors.text params present.
    non_text_modality_present must be False (so watchdog skips abort)."""
    names = [
        ("projectors.text.weight", None),
        ("projectors.text.bias", None),
        ("backbone.weight", None),
    ]
    assert _non_text_modality_present(names) is False


def test_predicate_on_bare_names_with_image():
    """text+image model under DDP: image projector counts as non-text."""
    names = [
        ("projectors.text.weight", None),
        ("projectors.image.weight", None),
        ("backbone.weight", None),
    ]
    assert _non_text_modality_present(names) is True


def test_predicate_on_fsdp_prefixed_names_text_only_fails_without_unwrap():
    """THIS IS THE BUG: with FSDP wrapping, the bare prefix check sees
    the text params as non-text, returning True for a text-only model.

    The fix wraps via _unwrap_model() to strip the FSDP prefix first.
    This test pins the bug behavior on raw FSDP-prefixed names so a
    regression that removes the _unwrap_model() call gets caught by the
    source-inspection test below."""
    names = [
        ("_fsdp_wrapped_module.projectors.text.weight", None),
        ("_fsdp_wrapped_module.projectors.text.bias", None),
        ("_fsdp_wrapped_module.backbone.weight", None),
    ]
    # Without unwrap, the bare predicate misclassifies — this is the bug.
    assert _non_text_modality_present(names) is True


def test_predicate_after_unwrap_strips_fsdp_prefix():
    """After _unwrap_model() runs, names should NOT carry the FSDP prefix
    (because accelerator.unwrap_model returns the inner module). The
    predicate then evaluates correctly."""
    # Simulate what _unwrap_model returns: bare names (no FSDP prefix).
    unwrapped_names = [
        ("projectors.text.weight", None),
        ("projectors.text.bias", None),
        ("backbone.weight", None),
    ]
    assert _non_text_modality_present(unwrapped_names) is False


# --- Source-inspection guard: pin the _unwrap_model() call site ---

def test_watchdog_iterates_unwrapped_model():
    """The watchdog must iterate the unwrapped model's named_parameters,
    not self.model's — otherwise FSDP wrapping breaks the prefix check.
    This is the source guard that catches a refactor dropping the unwrap.
    """
    src = TRAINER.read_text()
    # The watchdog block must contain both the comment AND the unwrap call,
    # AND the iteration must use _unwrapped (not self.model) for the
    # non_text_modality_present check.
    assert "_unwrapped = self._unwrap_model()" in src, (
        "Watchdog must call self._unwrap_model() before iterating named_parameters; "
        "without this, FSDP name prefixes break the projectors.text check."
    )
    # Pin the iteration source: must be _unwrapped, not self.model.
    assert "for n, _ in _unwrapped.named_parameters()" in src, (
        "non_text_modality_present must iterate _unwrapped.named_parameters() "
        "(not self.model.named_parameters()) so FSDP-prefixed names don't "
        "trigger a false-positive ghost-training abort."
    )


def test_watchdog_self_model_named_parameters_not_used_in_check():
    """Negative guard: the ghost-training check must NOT use
    `self.model.named_parameters()` for the non_text_modality_present
    predicate. If a refactor reintroduces it, this test fails.
    """
    src = TRAINER.read_text()
    # Find the watchdog block (between the comment header and the abort msg).
    start = src.find("# Ghost-training watchdog:")
    end = src.find("CRITICAL: Projectors have requires_grad=False!")
    assert start != -1 and end != -1, "Could not locate ghost-training watchdog block"
    block = src[start:end]
    assert "self.model.named_parameters()" not in block, (
        "Watchdog block must not iterate self.model.named_parameters() — "
        "use self._unwrap_model() to strip FSDP/DDP prefixes."
    )

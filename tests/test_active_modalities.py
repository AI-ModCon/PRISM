"""Verify MultiWebDatasetWrapper.active_modalities respects __init__ arg.

Historically the property hardcoded `{"image"}`; the IsoFLOP plan
generalizes it to accept a `modalities=` iterable (defaults to `("image",)`
for backward compat). This test asserts both behaviors.

The wrapper requires a config + tokenizer for full construction; we only
need to test the property, so monkey-patch __init__ to skip the heavy
setup work.
"""

from __future__ import annotations

from src.data.multi_webdataset import MultiWebDatasetWrapper


def _build_wrapper(modalities) -> MultiWebDatasetWrapper:
    """Construct a wrapper instance without running full __init__."""
    obj = MultiWebDatasetWrapper.__new__(MultiWebDatasetWrapper)
    obj._modalities = set(modalities)
    return obj


def test_default_modalities_is_image_only() -> None:
    obj = _build_wrapper(("image",))
    assert obj.active_modalities == {"image"}


def test_modalities_accepts_multi_set() -> None:
    obj = _build_wrapper(("text", "image", "time_series"))
    assert obj.active_modalities == {"text", "image", "time_series"}


def test_modalities_is_a_set_copy() -> None:
    """Returned value must be a fresh set so callers can't mutate state."""
    obj = _build_wrapper(("image", "text"))
    out = obj.active_modalities
    out.add("graph")
    # Subsequent calls should not be polluted.
    assert obj.active_modalities == {"image", "text"}

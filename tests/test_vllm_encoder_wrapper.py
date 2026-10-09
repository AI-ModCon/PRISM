"""Login-node tests for VLLM-2: _PrismEncoderWrapper + ModuleDict refactor.

No engine boot. We unit-test the wrapper's forward composition with a dummy
nn.Linear inner module, and verify that the model class refactor preserves
the PR #41 alias surface (`vision_tower`, `multi_modal_projector`).
"""

from __future__ import annotations

import pytest
import torch

vllm = pytest.importorskip("vllm")


def test_encoder_wrapper_composes_inner_then_proj():
    """forward(x) == proj(forward_fn(inner, x))."""
    from src.vllm_plugin.prism_for_conditional_generation import (
        _PrismEncoderWrapper,
    )

    inner = torch.nn.Linear(4, 6)
    proj = torch.nn.Linear(6, 8)
    # forward_fn just calls the module — image's "model(pixel_values=x)"
    # variant is exercised in the smoke runner.
    wrapper = _PrismEncoderWrapper(
        inner, proj, forward_fn=lambda m, x: m(x)
    )

    x = torch.randn(2, 4)
    out = wrapper(x)
    assert out.shape == (2, 8)
    torch.testing.assert_close(out, proj(inner(x)))


def test_encoder_wrapper_proj_identity_passes_through():
    from src.vllm_plugin.prism_for_conditional_generation import (
        _PrismEncoderWrapper,
    )

    inner = torch.nn.Linear(4, 6)
    wrapper = _PrismEncoderWrapper(
        inner, torch.nn.Identity(), forward_fn=lambda m, x: m(x)
    )
    x = torch.randn(3, 4)
    torch.testing.assert_close(wrapper(x), inner(x))


def test_legacy_alias_name():
    """`_PrismVisionTower` must remain importable as an alias of
    `_PrismEncoderWrapper` for PR #41 callers."""
    from src.vllm_plugin.prism_for_conditional_generation import (
        _PrismEncoderWrapper,
        _PrismVisionTower,
    )

    assert _PrismVisionTower is _PrismEncoderWrapper


def test_canonicalize_weight_key_renames_legacy_image_keys():
    """PR #41 checkpoint keys use `vision_tower.*` and
    `multi_modal_projector.*`; VLLM-2 canonical names are
    `encoders.image.*` / `multi_modal_projectors.image.*`.
    AutoWeightsLoader sees both via the alias, so we must translate the
    legacy names so the loader doesn't double-load."""
    from types import SimpleNamespace

    from src.vllm_plugin.prism_for_conditional_generation import (
        PrismForConditionalGeneration,
    )

    # _canonicalize_weight_key only reads self._LEGACY_KEY_RENAMES; bind a
    # bare namespace carrying that class attribute so we can exercise the
    # method without booting the full model.
    fake_self = SimpleNamespace(
        _LEGACY_KEY_RENAMES=PrismForConditionalGeneration._LEGACY_KEY_RENAMES
    )
    canon = PrismForConditionalGeneration._canonicalize_weight_key

    # Legacy image-tower keys get rewritten under encoders.image.*.
    assert (
        canon(fake_self, "vision_tower.model.layer0.weight")
        == "encoders.image.model.layer0.weight"
    )
    # Legacy projector keys get rewritten under multi_modal_projectors.image.*.
    assert (
        canon(fake_self, "multi_modal_projector.fc.weight")
        == "multi_modal_projectors.image.fc.weight"
    )
    # Already-canonical keys pass through unchanged.
    assert (
        canon(fake_self, "encoders.image.model.layer0.weight")
        == "encoders.image.model.layer0.weight"
    )
    assert (
        canon(fake_self, "language_model.model.embed_tokens.weight")
        == "language_model.model.embed_tokens.weight"
    )
    # Non-matching prefixes pass through unchanged.
    assert canon(fake_self, "some.other.key") == "some.other.key"


class _StubProc:
    """Minimal stand-in for ModalityProcessor — only `mm_kwarg_key` is read
    by embed_multimodal."""

    def __init__(self, mm_kwarg_key: str) -> None:
        self.mm_kwarg_key = mm_kwarg_key


def _make_stub_model(
    modality_to_kwarg: dict[str, str],
    *,
    inner_in: int = 4,
    inner_out: int = 6,
    proj_out: int = 8,
) -> torch.nn.Module:
    """Build a bare nn.Module with the attribute surface embed_multimodal
    reads. Avoids booting PrismForConditionalGeneration.__init__ (which
    needs a full vllm_config + HF download)."""
    from src.vllm_plugin.prism_for_conditional_generation import (
        PrismForConditionalGeneration,
        _PrismEncoderWrapper,
    )

    class _Stub(torch.nn.Module):
        embed_multimodal = PrismForConditionalGeneration.embed_multimodal
        _stack_per_modality = PrismForConditionalGeneration._stack_per_modality

        def __init__(self) -> None:
            super().__init__()
            self.encoders = torch.nn.ModuleDict(
                {
                    m: _PrismEncoderWrapper(
                        torch.nn.Linear(inner_in, inner_out),
                        torch.nn.Identity(),
                        forward_fn=lambda mod, x: mod(x),
                    )
                    for m in modality_to_kwarg
                }
            )
            self.multi_modal_projectors = torch.nn.ModuleDict(
                {m: torch.nn.Linear(inner_out, proj_out) for m in modality_to_kwarg}
            )
            self._modality_processors = {
                m: _StubProc(k) for m, k in modality_to_kwarg.items()
            }

    return _Stub()


def test_embed_multimodal_general_path_single_modality_returns_tensor():
    """One non-image modality present → bare tensor return (matches the
    single-image fast path contract)."""
    model = _make_stub_model({"time_series": "time_series_values"})
    raw = torch.randn(2, 4)
    out = model.embed_multimodal(time_series_values=raw)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8)


def test_embed_multimodal_general_path_multi_modality_returns_list():
    """Two modalities present → list[Tensor] return; vLLM's splicer
    accepts list[Tensor] (interfaces.py MultiModalEmbeddings alias)."""
    model = _make_stub_model(
        {"image": "pixel_values", "time_series": "time_series_values"}
    )
    img_raw = torch.randn(2, 4)
    ts_raw = torch.randn(3, 4)
    out = model.embed_multimodal(
        pixel_values=img_raw, time_series_values=ts_raw
    )
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(t, torch.Tensor) for t in out)
    # Order matches insertion order of _modality_processors.
    assert out[0].shape == (2, 8)
    assert out[1].shape == (3, 8)


def test_embed_multimodal_general_path_missing_input_skipped():
    """A registered modality with no tensor in kwargs is skipped (no
    KeyError, no empty-tensor stand-in)."""
    model = _make_stub_model(
        {"image": "pixel_values", "time_series": "time_series_values"}
    )
    # Only image provided; time_series silently absent.
    out = model.embed_multimodal(pixel_values=torch.randn(2, 4))
    # Single tensor return (only image populated).
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8)


def test_embed_multimodal_general_path_empty_returns_empty_list():
    """No modality inputs at all → empty list (vLLM treats this as a
    text-only request)."""
    model = _make_stub_model(
        {"image": "pixel_values", "time_series": "time_series_values"}
    )
    out = model.embed_multimodal()
    assert out == []


def test_image_processor_build_encoder_returns_inner_hidden_forward():
    """Image processor's build_encoder() returns the (inner, hidden, fn)
    triple the model class expects. Uses AutoConfig only (no weight
    download), so this is safe for a login-node test."""
    from src.vllm_plugin.processors import ImageModalityProcessor

    proc = ImageModalityProcessor(
        placeholder_token_id=50300,
        prism_subconfig={
            "encoder_model": "google/siglip2-base-patch16-224",
            "d_img": 768,
        },
    )
    inner, hidden, fwd = proc.build_encoder()
    # Inner is a torch.nn.Module; SigLIP2-base hidden_size is 768.
    assert isinstance(inner, torch.nn.Module)
    assert hidden == 768
    # The forward closure delegates to model(pixel_values=...) — exercising
    # it would require a real GPU forward; we just confirm it's callable.
    assert callable(fwd)

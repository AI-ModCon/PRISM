"""Login-node smoke tests for the vLLM plugin.

These run with `module load frameworks` on the UAN — no XPU required. They
verify (a) registration succeeds and (b) the model class is wired to vLLM's
multimodal registry. Anything that needs actual GPU kernels (model
construction, generate) belongs in a compute-node test.
"""

import pytest


def test_register_is_idempotent():
    pytest.importorskip("vllm")
    import src.vllm_plugin

    src.vllm_plugin.register()
    src.vllm_plugin.register()  # no-op the second time


def test_model_arch_registered():
    pytest.importorskip("vllm")
    import src.vllm_plugin

    src.vllm_plugin.register()

    from vllm import ModelRegistry

    assert "PrismForConditionalGeneration" in ModelRegistry.get_supported_archs()


def test_supports_multimodal_protocol():
    pytest.importorskip("vllm")
    import src.vllm_plugin

    src.vllm_plugin.register()

    from src.vllm_plugin.prism_for_conditional_generation import (
        PrismForConditionalGeneration,
    )
    from vllm.model_executor.models.interfaces import supports_multimodal

    assert supports_multimodal(PrismForConditionalGeneration)


def test_processor_classes_importable():
    """Catches import-time errors in processor.py without booting vLLM."""
    pytest.importorskip("vllm")
    from src.vllm_plugin.processor import (  # noqa: F401
        PrismDummyInputsBuilder,
        PrismMultiModalProcessor,
        PrismProcessingInfo,
        _build_image_transform,
    )

    transform = _build_image_transform()
    # Trivial: the transform should produce a (3, 224, 224) tensor in [-1, 1].
    from PIL import Image

    img = Image.new("RGB", (50, 50), color=(127, 127, 127))
    out = transform(img)
    assert out.shape == (3, 224, 224)
    assert -1.01 <= out.min().item() and out.max().item() <= 1.01

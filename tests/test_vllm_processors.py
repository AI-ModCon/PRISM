"""Login-node tests for the per-modality processor abstraction (VLLM-1).

No vLLM engine boot — these cover the contract of ModalityProcessor,
ImageModalityProcessor, and the registry/dispatch without going near XPU.
"""

import pytest

vllm = pytest.importorskip("vllm")  # ensures the multimodal types exist


def test_registry_contains_image():
    from src.vllm_plugin.processors import MODALITY_PROCESSORS

    assert "image" in MODALITY_PROCESSORS


def test_get_active_modalities_defaults_to_image():
    from src.vllm_plugin.processors.registry import get_active_modalities

    # PR #41 layout (no active_modalities key).
    assert get_active_modalities({"image_token_id": 50300}) == ["image"]
    # Explicit list overrides.
    assert get_active_modalities(
        {"active_modalities": ["image", "time_series"]}
    ) == ["image", "time_series"]


def test_build_modality_processors_legacy_flat_keys():
    """PR #41 exports carry image_token / image_token_id at the top level
    (no per-modality `image` block). The image factory must still produce a
    valid processor."""
    from src.vllm_plugin.processors import build_modality_processors

    procs = build_modality_processors(
        {"image_token": "<image>", "image_token_id": 50300}
    )
    assert set(procs.keys()) == {"image"}
    assert procs["image"].placeholder_token == "<image>"
    assert procs["image"].placeholder_token_id == 50300
    assert procs["image"].mm_kwarg_key == "pixel_values"


def test_build_modality_processors_per_modality_block():
    """VLLM-1.5 layout: per-modality block carries placeholder + size."""
    from src.vllm_plugin.processors import build_modality_processors

    procs = build_modality_processors(
        {
            "active_modalities": ["image"],
            "image": {
                "placeholder_token": "<image>",
                "placeholder_token_id": 50300,
                "image_size": 224,
                "num_image_tokens": 196,
            },
        }
    )
    proc = procs["image"]
    assert proc.placeholder_token_id == 50300
    assert proc.num_tokens(item=None) == 196


def test_image_processor_encodes_pil_image():
    from PIL import Image
    from src.vllm_plugin.processors import ImageModalityProcessor

    proc = ImageModalityProcessor(placeholder_token_id=50300)
    out = proc.encode(Image.new("RGB", (10, 10), (255, 0, 0)))
    assert tuple(out.shape) == (1, 3, 224, 224)
    assert -1.01 <= out.min().item() and out.max().item() <= 1.01


def test_image_processor_encodes_list_of_pil_images():
    from PIL import Image
    from src.vllm_plugin.processors import ImageModalityProcessor

    proc = ImageModalityProcessor(placeholder_token_id=50300)
    out = proc.encode(
        [
            Image.new("RGB", (10, 10), (255, 0, 0)),
            Image.new("RGB", (20, 30), (0, 255, 0)),
            Image.new("RGB", (15, 15), (0, 0, 255)),
        ]
    )
    assert tuple(out.shape) == (3, 3, 224, 224)


def test_image_processor_accepts_image_or_images_key():
    from PIL import Image
    from src.vllm_plugin.processors import ImageModalityProcessor

    proc = ImageModalityProcessor(placeholder_token_id=50300)
    img = Image.new("RGB", (10, 10), (255, 255, 255))

    # Canonical key
    assert proc.normalize_mm_data_key({"image": img}) is img
    # Plural alias (demo path)
    assert proc.normalize_mm_data_key({"images": [img]}) == [img]
    # Missing
    assert proc.normalize_mm_data_key({}) is None


def test_image_processor_field_config_is_batched():
    from src.vllm_plugin.processors import ImageModalityProcessor
    from vllm.multimodal.inputs import MultiModalFieldConfig

    proc = ImageModalityProcessor(placeholder_token_id=50300)
    fc = proc.field_config()
    assert isinstance(fc, MultiModalFieldConfig)


def test_register_modality_processor_decorator():
    """Confirms new modalities can register themselves without editing registry.py."""
    from src.vllm_plugin.processors import MODALITY_PROCESSORS
    from src.vllm_plugin.processors.registry import register_modality_processor

    @register_modality_processor("__test_modality__")
    def _factory(_cfg):
        from src.vllm_plugin.processors import ImageModalityProcessor

        return ImageModalityProcessor(placeholder_token_id=99999)

    try:
        assert "__test_modality__" in MODALITY_PROCESSORS
    finally:
        MODALITY_PROCESSORS.pop("__test_modality__", None)


def test_unknown_modality_raises():
    from src.vllm_plugin.processors import build_modality_processors

    with pytest.raises(KeyError, match="__no_such_modality__"):
        build_modality_processors(
            {"active_modalities": ["__no_such_modality__"]}
        )

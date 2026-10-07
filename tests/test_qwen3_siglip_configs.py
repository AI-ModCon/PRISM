from pathlib import Path

import pytest
import yaml
from src.training.distributed import DECODER_LAYER_MAP

pytestmark = pytest.mark.unit


QWEN3_CONFIGS = {
    "prism_qwen3_0_6b_image_only.yaml": ("Qwen/Qwen3-0.6B", 1024),
    "prism_qwen3_1_7b_image_only.yaml": ("Qwen/Qwen3-1.7B", 2048),
    "prism_qwen3_4b_image_only.yaml": ("Qwen/Qwen3-4B", 2560),
    "prism_qwen3_8b_image_only.yaml": ("Qwen/Qwen3-8B", 4096),
    "prism_qwen3_14b_image_only.yaml": ("Qwen/Qwen3-14B", 5120),
    "prism_qwen3_32b_image_only.yaml": ("Qwen/Qwen3-32B", 5120),
}


@pytest.mark.parametrize("filename,expected", QWEN3_CONFIGS.items())
def test_qwen3_image_only_configs_are_dense_image_text_configs(filename, expected):
    expected_backbone, expected_d_text = expected
    config_path = Path("src/conf/model") / filename
    cfg = yaml.safe_load(config_path.read_text())

    assert cfg["backbone_id"] == expected_backbone
    assert cfg["tokenizer_id"] == expected_backbone
    assert cfg["d_text"] == expected_d_text
    assert cfg["d_img"] == 768
    assert cfg["image_encoder_id"] == "google/siglip2-base-patch16-224"
    assert cfg["image_processor_id"] == "google/siglip2-base-patch16-224"
    assert cfg["image_processor_strict"] is True
    assert cfg["modalities"] == ["text", "image"]


def test_qwen3_decoder_layer_has_native_fsdp_mapping():
    assert DECODER_LAYER_MAP["qwen3"] == (
        "transformers.models.qwen3.modeling_qwen3",
        "Qwen3DecoderLayer",
    )

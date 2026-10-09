from unittest.mock import patch

import pytest
from src.config import ModelConfig
from src.model import UnifiedTransformer

pytestmark = pytest.mark.unit


def test_configured_hf_backbone_load_failure_is_fatal():
    config = ModelConfig(
        d_model=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        llm_backbone_id="Qwen/Qwen3-0.6B",
        modalities=["text", "image"],
    )

    with (
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=OSError("missing model cache"),
        ),
        pytest.raises(RuntimeError, match="Could not load HF backbone"),
    ):
        UnifiedTransformer(config)

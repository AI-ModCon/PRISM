import pytest
import torch

pytest.importorskip("einops")

from src.config import ModelConfig
from src.model import UnifiedTransformer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.slow,
]

OLMO_BACKBONES = [
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-7B-0724-hf",
]


@pytest.mark.parametrize("backbone_id", OLMO_BACKBONES)
def test_olmo_multimodal_generate_smoke(backbone_id):
    cfg = ModelConfig(
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        modalities=["text"],
    )
    model = UnifiedTransformer(cfg)

    tokenizer = model.backbone_tokenizer
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        outputs = model.generate({"text": input_ids}, max_new_tokens=10)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert isinstance(text, str)
    assert len(text) > 0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytest.importorskip("einops")

from src.config import ModelConfig
from src.model import UnifiedTransformer

pytestmark = [pytest.mark.unit]

OLMO_BACKBONES = [
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-7B-0724-hf",
]


class _DummyHFBackbone(nn.Module):
    def __init__(self, hidden_size=32, vocab_size=128):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, labels=None, return_dict=True, **kwargs):
        logits = self.head(inputs_embeds)
        loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=[inputs_embeds])

    def generate(self, inputs_embeds=None, max_new_tokens=5, **kwargs):
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        return torch.zeros((batch_size, seq_len + max_new_tokens), dtype=torch.long)


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


@pytest.mark.parametrize("backbone_id", OLMO_BACKBONES)
def test_olmo_generation_contract(offline_hf, backbone_id):
    cfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        d_text=32,
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        modalities=["text"],
    )

    with (
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=_DummyHFBackbone(hidden_size=32, vocab_size=128),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_DummyTokenizer(),
        ),
    ):
        model = UnifiedTransformer(cfg)

    input_ids = torch.randint(0, 32, (2, 6), dtype=torch.long)
    out = model.generate({"text": input_ids}, max_new_tokens=4)

    assert out.shape == (2, 10)
    assert out.dtype == torch.long

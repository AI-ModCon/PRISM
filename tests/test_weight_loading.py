import os
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
        return SimpleNamespace(logits=logits, loss=torch.tensor(0.0), hidden_states=[inputs_embeds])


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class _DummyImageEncoder(nn.Module):
    def __init__(self, output_dim=32, seq_len=4):
        super().__init__()
        self.proj = nn.Linear(output_dim, output_dim)
        self.output_dim = output_dim
        self.seq_len = seq_len

    def forward(self, x):
        return torch.randn(x.shape[0], self.seq_len, self.output_dim)


@pytest.mark.parametrize("backbone_id", OLMO_BACKBONES)
@pytest.mark.parametrize("distributed,local_rank", [(False, 0), (True, 0), (True, 1)])
def test_olmo_backbone_initialization(offline_hf, backbone_id, distributed, local_rank):
    cfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        d_text=32,
        d_img=32,
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        modalities=["text", "image"],
    )

    def load_backbone(_model_id, *, torch_dtype, **kwargs):
        # Transformers 4.51.3 (the pinned image stack) consumes torch_dtype;
        # dtype would instead reach the model constructor and raise TypeError.
        assert "dtype" not in kwargs
        assert torch_dtype in (torch.float16, torch.bfloat16)
        return _DummyHFBackbone(hidden_size=32, vocab_size=128)

    with (
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=load_backbone,
        ),
        patch("torch.distributed.is_initialized", return_value=distributed),
        patch("torch.distributed.barrier"),
        patch.dict(os.environ, {"LOCAL_RANK": str(local_rank), "PALS_LOCAL_RANKID": str(local_rank)}),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_DummyTokenizer(),
        ),
        patch(
            "src.model.ImageEncoder",
            side_effect=lambda **kwargs: _DummyImageEncoder(output_dim=32, seq_len=4),
        ),
    ):
        model = UnifiedTransformer(cfg)

    assert model.backbone is not None
    assert model.config.llm_backbone_id == backbone_id
    assert "image" in model.encoders
    assert "image" in model.projectors
    assert all(not p.requires_grad for p in model.backbone.parameters())

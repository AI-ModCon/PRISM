import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

pytest.importorskip("einops")
from src.config import ModelConfig, TrainingConfig
from src.model import UnifiedTransformer
from src.training.trainer_zone_a import ZoneATrainer

pytestmark = [pytest.mark.integration]

OLMO_BACKBONES = [
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-7B-0724-hf",
]


# --- Mock Data ---
class MockDataset(torch.utils.data.Dataset):
    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return dummy data for all zones
        return {
            "text": torch.randint(0, 100, (16,)),
            "image": torch.randn(3, 224, 224),
            "prompt_text": torch.randint(0, 100, (16,)),
            "chosen_text": torch.randint(0, 100, (16,)),
            "rejected_text": torch.randint(0, 100, (16,)),
        }


def mock_collate(batch):
    # Simple collate
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


@pytest.fixture
def mock_config():
    return TrainingConfig(
        batch_size=2,
        max_steps=2,
        wandb_project=None,  # Disable wandb
        vocab_size=100,
        device="cpu",
    )


@pytest.fixture
def mock_model_config():
    return ModelConfig(
        vocab_size=100,
        d_model=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        modalities=["text", "image"],
    )


@pytest.fixture
def mock_loader():
    dataset = MockDataset()
    return torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=mock_collate)


# --- Tests ---


def test_zone_a_trainer(mock_config, mock_model_config, mock_loader):
    """Test Zone A Trainer (Freezing Logic + Training Step)"""

    # Instantiate real model (loads weights, might be slow but robust)
    # To speed up, we could mock the encoder classes, but let's just use the real init
    # and then swap them out.
    model = UnifiedTransformer(mock_model_config)

    # Swap heavy encoders with simple Linear layers for testing
    model.encoders["text"] = nn.Linear(10, 32)
    model.encoders["image"] = nn.Linear(10, 32)

    # Mock forward to return (logits, aux_loss) with grad
    # We mock the whole forward to skip the complex internal logic during trainer test
    model.forward = MagicMock(
        return_value=(
            torch.randn(2, 16, 100, requires_grad=True),
            torch.tensor(0.1, requires_grad=True),
        )
    )

    trainer = ZoneATrainer(model, mock_config, mock_loader)

    # Check Freezing
    # Encoders should be frozen
    assert model.encoders["text"].weight.requires_grad is False
    # Backbone (MoE) should be frozen
    assert model.blocks[0].attn.c_attn.weight.requires_grad is False
    # Projectors should be UNFROZEN
    # ModalityProjector uses 'net' (Sequential)
    assert model.projectors["text"].net[0].weight.requires_grad is True

    # Run Train Loop
    trainer.train()
    assert model.forward.called


class _DummyHFBackbone(nn.Module):
    def __init__(self, hidden_size=32, vocab_size=100):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        inputs_embeds=None,
        labels=None,
        return_dict=True,
        output_hidden_states=False,
        **kwargs,
    ):
        logits = self.head(inputs_embeds)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, logits.shape[-1]), shift_labels.view(-1))

        hidden_states = [inputs_embeds] if output_hidden_states else None
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=hidden_states)

    def generate(self, inputs_embeds=None, max_new_tokens=4, **kwargs):
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        return torch.zeros((batch_size, seq_len + max_new_tokens), dtype=torch.long)


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, token_ids, **kwargs):
        return "decoded"


class _DummyModalityEncoder(nn.Module):
    def __init__(self, output_dim=32, seq_len=4):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.proj = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        if isinstance(x, dict):
            tensor = x.get("x", torch.randn(1, 1, self.output_dim))
            batch_size = tensor.shape[0]
        else:
            batch_size = x.shape[0]
        return torch.randn(batch_size, self.seq_len, self.output_dim)

    def tokens_per_instance(self):
        return self.seq_len


class _ZoneAModalityDataset(torch.utils.data.Dataset):
    def __init__(self, modality):
        self.modality = modality
        self.active_modalities = [modality]

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        sample = {"text": torch.randint(0, 100, (16,), dtype=torch.long)}
        if self.modality == "image":
            sample["image"] = torch.randn(3, 16, 16)
        elif self.modality == "time_series":
            sample["time_series"] = torch.randn(16, 1)
        elif self.modality == "graph":
            sample["graph"] = {"x": torch.randn(8, 8)}
        elif self.modality == "geometry":
            sample["geometry"] = torch.randn(16, 6)
        elif self.modality == "table":
            sample["table"] = torch.randint(0, 100, (16,), dtype=torch.long)
        return sample


@pytest.mark.parametrize("backbone_id", OLMO_BACKBONES)
@pytest.mark.parametrize(
    "modality,patch_target",
    [
        ("image", "src.model.ImageEncoder"),
        ("time_series", "src.model.TimeSeriesEncoder"),
        ("graph", "src.model.GraphEncoder"),
        ("geometry", "src.model.GeometryEncoder"),
        ("table", "src.model.TableEncoder"),
    ],
)
def test_zone_a_encoder_projector_scope_for_olmo_backbones(backbone_id, modality, patch_target):
    train_config = TrainingConfig(
        batch_size=2,
        max_steps=1,
        wandb_project=None,
        vocab_size=100,
        device="cpu",
    )
    model_config = ModelConfig(
        vocab_size=100,
        d_model=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        d_text=32,
        d_img=32,
        d_table=32,
        d_ts=32,
        d_geo=32,
        d_graph=32,
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        modalities=["text", modality],
    )

    loader = torch.utils.data.DataLoader(_ZoneAModalityDataset(modality), batch_size=2)

    def encoder_factory(**kwargs):
        return _DummyModalityEncoder(output_dim=32, seq_len=4)

    with (
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=_DummyHFBackbone(hidden_size=32, vocab_size=100),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_DummyTokenizer(),
        ),
        patch(
            patch_target,
            side_effect=encoder_factory,
        ),
        patch(
            "src.training.trainer_zone_a.EvaluatorRegistry.get",
            return_value=None,
        ),
    ):
        model = UnifiedTransformer(model_config)
        trainer = ZoneATrainer(model, train_config, loader)

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    assert unwrapped.config.llm_backbone_id == backbone_id
    assert modality in unwrapped.encoders
    assert modality in unwrapped.projectors
    assert all(not p.requires_grad for p in unwrapped.backbone.parameters())
    assert all(not p.requires_grad for p in unwrapped.encoders[modality].parameters())
    assert any(p.requires_grad for p in unwrapped.projectors[modality].parameters())



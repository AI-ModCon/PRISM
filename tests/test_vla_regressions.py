import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("einops")

from src.config import ModelConfig
from src.data.calvin_vla import CalvinVLADataset
from src.model import UnifiedTransformer
from src.training.trainer_zone_a_vla import ZoneAVLATrainer

pytestmark = [pytest.mark.integration, pytest.mark.slow]

OLMO_BACKBONES = [
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-7B-0724-hf",
]


class TinyVLADataset(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return {
            "text": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "text_attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
            "image_head": torch.randn(3, 8, 8),
            "image_wrist": torch.randn(3, 8, 8),
            "pose": torch.randn(15),
            "action": torch.randn(7),
        }


class TinyVLAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4, bias=False)
        self.encoders = nn.ModuleDict({"image": nn.Linear(4, 4)})
        self.projectors = nn.ModuleDict({"image": nn.Linear(4, 4)})
        self.pose_embed = nn.Sequential(nn.Linear(15, 4), nn.ReLU(), nn.Linear(4, 4))
        self.action_head = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 7))
        self.pose_modality_embedding = nn.Parameter(torch.zeros(1, 1, 4))
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        features = self.pose_embed(batch["pose"])
        pred_action = self.action_head(features)
        per_dim_mse = ((pred_action - batch["action"]) ** 2).mean(dim=0)
        return pred_action, per_dim_mse.mean(), per_dim_mse


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

    def forward(self, inputs_embeds=None, output_hidden_states=False, **kwargs):
        logits = self.head(inputs_embeds)
        hidden = [inputs_embeds] if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden)


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


def _make_vla_config(output_dir: Path, max_steps=3, grad_accum=2, resume_from_checkpoint=None):
    return SimpleNamespace(
        gradient_accumulation_steps=grad_accum,
        wandb_project=None,
        wandb_entity=None,
        wandb_mode="disabled",
        wandb_run_name="pytest-vla",
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        max_steps=max_steps,
        log_every_n_steps=1000,
        save_every_n_steps=1000,
        output_dir=str(output_dir),
        resume_from_checkpoint=resume_from_checkpoint,
    )


def test_vla_trainer_accumulation_and_trainable_set(tmp_path):
    model = TinyVLAModel()
    loader = DataLoader(TinyVLADataset(), batch_size=2)
    cfg = _make_vla_config(tmp_path / "vla_train", max_steps=3, grad_accum=2)
    trainer = ZoneAVLATrainer(model, cfg, loader)

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    assert unwrapped.pose_modality_embedding.requires_grad
    assert all(not p.requires_grad for p in unwrapped.backbone.parameters())

    trainer.train()

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    assert unwrapped.forward_calls == cfg.max_steps * cfg.gradient_accumulation_steps


def test_vla_trainer_resume_step_restore(tmp_path):
    output_dir = tmp_path / "resume_ckpt"
    loader = DataLoader(TinyVLADataset(), batch_size=2)

    first_cfg = _make_vla_config(output_dir, max_steps=2, grad_accum=1)
    first_trainer = ZoneAVLATrainer(TinyVLAModel(), first_cfg, loader)
    first_trainer.config.save_every_n_steps = 1
    first_trainer.train()

    resume_path = output_dir / "step_1"
    second_cfg = _make_vla_config(
        output_dir, max_steps=3, grad_accum=1, resume_from_checkpoint=str(resume_path)
    )
    second_trainer = ZoneAVLATrainer(TinyVLAModel(), second_cfg, loader)
    assert second_trainer.resume_step == 1
    second_trainer.train()


@pytest.mark.parametrize("backbone_id", OLMO_BACKBONES)
def test_vla_zone_a_olmo_trainable_components(backbone_id, tmp_path):
    model_cfg = ModelConfig(
        vocab_size=100,
        d_model=32,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        d_text=32,
        d_img=32,
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        is_vla=True,
        modalities=["text", "image"],
    )
    train_cfg = _make_vla_config(
        tmp_path / f"vla_{backbone_id.split('/')[-1]}",
        max_steps=1,
        grad_accum=1,
    )

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
            "src.model.ImageEncoder",
            side_effect=lambda **kwargs: _DummyImageEncoder(output_dim=32, seq_len=4),
        ),
    ):
        model = UnifiedTransformer(model_cfg)

    loader = DataLoader(TinyVLADataset(), batch_size=2)
    trainer = ZoneAVLATrainer(model, train_cfg, loader)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)

    assert all(not p.requires_grad for p in unwrapped.backbone.parameters())
    assert any(p.requires_grad for p in unwrapped.projectors["image"].parameters())
    assert any(p.requires_grad for p in unwrapped.pose_embed.parameters())
    assert any(p.requires_grad for p in unwrapped.action_head.parameters())
    assert unwrapped.pose_modality_embedding.requires_grad


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        raise RuntimeError("Tokenizer should not be used during dataset initialization")


def _write_calvin_metadata(root: Path):
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"splits": {"train": "0:2"}}))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick block"}) + "\n")
    episodes = [
        {"episode_index": 0, "length": 2, "tasks": ["pick block"]},
        {"episode_index": 1, "length": 2, "tasks": ["pick block"]},
    ]
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes) + "\n")


def test_calvin_dataset_skip_reporting_and_strict_mode(tmp_path):
    root = tmp_path / "calvin"
    _write_calvin_metadata(root)
    chunk = root / "data" / "chunk-000"
    chunk.mkdir(parents=True)
    (chunk / "episode_000000.parquet").write_text("placeholder")

    dataset = CalvinVLADataset(
        root_dir=root,
        tokenizer=DummyTokenizer(),
        split="train",
        strict_integrity=False,
        max_skipped_fraction=1.0,
    )
    assert dataset.dataset_stats["kept_episodes"] == 1
    assert dataset.dataset_stats["skipped_missing_parquet"] == 1

    with pytest.raises(RuntimeError, match="strict_integrity=True"):
        CalvinVLADataset(
            root_dir=root,
            tokenizer=DummyTokenizer(),
            split="train",
            strict_integrity=True,
            max_skipped_fraction=1.0,
        )

    with pytest.raises(RuntimeError, match="max_skipped_fraction"):
        CalvinVLADataset(
            root_dir=root,
            tokenizer=DummyTokenizer(),
            split="train",
            strict_integrity=False,
            max_skipped_fraction=0.4,
        )


def test_analyze_vla_dataset_signature():
    import train as train_module

    sig = inspect.signature(train_module.analyze_vla_dataset)
    assert list(sig.parameters.keys()) == ["dataset", "train_config"]

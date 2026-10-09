from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from src.training.distributed import (
    _align_checkpoint_vocab_for_resume,
    save_native_ddp_checkpoint,
)
from src.training.trainer_native import _load_optimizer_state_for_resume


class _LenTokenizer:
    def __init__(self, length: int):
        self._length = length

    def __len__(self):
        return self._length


class _FakeBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab_size, dim)
        self.lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab_size)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def resize_token_embeddings(self, vocab_size: int):
        old_embeddings = self.model.embed_tokens
        old_lm_head = self.lm_head
        dim = old_embeddings.weight.shape[1]
        new_embeddings = torch.nn.Embedding(vocab_size, dim)
        new_lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        rows = min(old_embeddings.weight.shape[0], vocab_size)
        with torch.no_grad():
            new_embeddings.weight[:rows].copy_(old_embeddings.weight[:rows])
            new_lm_head.weight[:rows].copy_(old_lm_head.weight[:rows])
        self.model.embed_tokens = new_embeddings
        self.lm_head = new_lm_head
        self.config.vocab_size = vocab_size
        return new_embeddings


class _FakePrismModel(torch.nn.Module):
    def __init__(self, vocab_size: int, dim: int, tokenizer_len: int | None = None):
        super().__init__()
        self.backbone = _FakeBackbone(vocab_size, dim)
        self.config = SimpleNamespace(vocab_size=vocab_size)
        if tokenizer_len is not None:
            self.backbone_tokenizer = _LenTokenizer(tokenizer_len)


def test_native_checkpoint_saves_optimizer_scheduler_and_dataloader_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    x = torch.ones(1, 2)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    config = SimpleNamespace(
        output_dir=str(tmp_path),
        gradient_accumulation_steps=2,
        finite_epoch_steps=7,
    )
    save_native_ddp_checkpoint(model, optimizer, scheduler, step=3, config=config, rank=0)

    checkpoint_dir = tmp_path / "step_3"
    state = torch.load(checkpoint_dir / "training_state.pt", weights_only=False)
    assert "optimizer" in state
    assert "scheduler" in state
    assert state["dataloader"] == {
        "completed_steps": 3,
        "completed_microbatches": 6,
        "finite_epoch_steps": 7,
        "resume_strategy": "deterministic_replay_skip",
    }

    metadata = json.loads((checkpoint_dir / "training_state.json").read_text())
    assert metadata["step"] == 3
    assert metadata["dataloader"]["completed_microbatches"] == 6
    assert metadata["dist_strategy"]


def test_resume_resizes_backbone_when_checkpoint_vocab_is_larger():
    model = _FakePrismModel(vocab_size=8, dim=4, tokenizer_len=8)
    checkpoint_state = {
        "backbone.model.embed_tokens.weight": torch.arange(
            40, dtype=torch.float32
        ).reshape(10, 4),
        "backbone.lm_head.weight": torch.ones(10, 4),
    }

    adapted = _align_checkpoint_vocab_for_resume(model, checkpoint_state)

    assert adapted is checkpoint_state
    assert model.backbone.get_input_embeddings().weight.shape == (10, 4)
    assert model.backbone.lm_head.weight.shape == (10, 4)
    model.load_state_dict(adapted, strict=False)


def test_resume_pads_checkpoint_vocab_when_tokenizer_is_larger():
    model = _FakePrismModel(vocab_size=12, dim=4, tokenizer_len=12)
    checkpoint_embeddings = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    checkpoint_lm_head = torch.ones(10, 4)
    checkpoint_state = {
        "backbone.model.embed_tokens.weight": checkpoint_embeddings,
        "backbone.lm_head.weight": checkpoint_lm_head,
    }

    adapted = _align_checkpoint_vocab_for_resume(model, checkpoint_state)

    assert model.backbone.get_input_embeddings().weight.shape == (12, 4)
    assert adapted["backbone.model.embed_tokens.weight"].shape == (12, 4)
    assert adapted["backbone.lm_head.weight"].shape == (12, 4)
    torch.testing.assert_close(
        adapted["backbone.model.embed_tokens.weight"][:10],
        checkpoint_embeddings,
    )
    torch.testing.assert_close(
        adapted["backbone.lm_head.weight"][:10],
        checkpoint_lm_head,
    )
    model.load_state_dict(adapted, strict=False)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_ddp_optimizer_state_converts_for_fsdp_resume():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    init_file = tempfile.NamedTemporaryFile(delete=False)
    init_file.close()
    initialized_here = False
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                "gloo",
                init_method=f"file://{init_file.name}",
                rank=0,
                world_size=1,
            )
            initialized_here = True

        source = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        loss = source(torch.randn(3, 4)).sum()
        loss.backward()
        source_optimizer.step()

        target = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )
        target.load_state_dict(source.state_dict())
        wrapped_target = FSDP(
            target,
            use_orig_params=True,
            device_id=torch.device("cpu"),
        )
        target_optimizer = torch.optim.AdamW(wrapped_target.parameters(), lr=1e-3)
        config = SimpleNamespace(
            freeze_llm=True,
            freeze_vit=True,
            lr_connector=None,
            learning_rate=1e-3,
            weight_decay=0.0,
        )

        _load_optimizer_state_for_resume(
            target_optimizer,
            source_optimizer.state_dict(),
            model=wrapped_target,
            unwrapped_model=target,
            config=config,
            is_fsdp_model=True,
            current_dist_strategy="fsdp",
            source_dist_strategy=None,
            is_main=False,
        )

        resumed_loss = wrapped_target(torch.randn(3, 4)).sum()
        resumed_loss.backward()
        target_optimizer.step()
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
        try:
            os.unlink(init_file.name)
        except FileNotFoundError:
            pass

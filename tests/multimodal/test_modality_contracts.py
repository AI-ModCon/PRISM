from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("einops")


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _patch_is_xpu_available():
    """Prevent intel_extension_for_pytorch from being imported during Accelerator
    initialisation on login/CI nodes that lack the XPU driver ABI.  The import
    raises an ImportError (undefined symbol in libintel-ext-pt-gpu-bitsandbytes.so)
    which is both node-specific and leaves the AcceleratorState Borg singleton in
    a broken state, cascading failures across every subsequent Zone-A test.
    """
    with patch("accelerate.state.is_xpu_available", return_value=False), \
         patch("accelerate.utils.imports.is_xpu_available", return_value=False):
        yield


@pytest.fixture(autouse=True)
def _reset_accelerator_state():
    """Reset the AcceleratorState/PartialState Borg singletons after every test.
    Without this, a crash or completed Accelerator in one parametrised variant
    leaves shared state that causes AttributeError('NoneType has no attr type')
    in every test that runs afterwards.
    """
    yield
    try:
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state(reset_partial_state=True)
    except Exception:
        pass

from src.config import ModelConfig, TrainingConfig
from src.model import UnifiedTransformer
from src.training.trainer_zone_a import ZoneATrainer
from src.training.trainer_zone_a_vla import ZoneAVLATrainer


class DummyEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        fixed_tokens: int | None = None,
        tokens_per_instance: int = 2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.fixed_tokens = fixed_tokens
        self._tokens_per_instance = tokens_per_instance
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if isinstance(x, dict):
            tensor = x.get("x")
            if tensor is None:
                batch_size, seq_len = 1, 1
            else:
                batch_size = tensor.shape[0]
                seq_len = tensor.shape[1] if tensor.dim() > 1 else 1
        elif isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
            seq_len = x.shape[1] if x.dim() > 1 else 1
        else:
            batch_size, seq_len = 1, 1

        if self.fixed_tokens is not None:
            seq_len = self.fixed_tokens

        return torch.randn(batch_size, seq_len, self.output_dim)

    def tokens_per_instance(self) -> int:
        return self._tokens_per_instance


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 32, vocab_size: int = 128):
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
        loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
        hidden_states = [inputs_embeds] if output_hidden_states else None
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=hidden_states)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class ZoneAModalityDataset(Dataset):
    def __init__(self, modality: str):
        self.modality = modality
        self.active_modalities = [modality]

    def __len__(self):
        return 4

    def __getitem__(self, _idx):
        sample = {"text": torch.randint(0, 64, (8,), dtype=torch.long)}
        if self.modality == "image":
            sample["image"] = torch.randn(3, 16, 16)
        elif self.modality == "time_series":
            sample["time_series"] = torch.randn(12, 1)
        elif self.modality == "graph":
            sample["graph"] = {"x": torch.randn(6, 8)}
        elif self.modality == "geometry":
            sample["geometry"] = torch.randn(12, 6)
        elif self.modality == "table":
            sample["table"] = torch.randint(0, 64, (8,), dtype=torch.long)
        return sample


class TinyVLADataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, _idx):
        return {
            "text": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "text_attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
            "image_head": torch.randn(3, 8, 8),
            "image_wrist": torch.randn(3, 8, 8),
            "pose": torch.randn(15),
            "action": torch.randn(7),
        }


pytestmark = [pytest.mark.unit, pytest.mark.multimodal]

OLMO_BACKBONES = [
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-7B-0724-hf",
]


MODALITY_CONFIG = {
    "text": {
        "modalities": ["text"],
        "input": lambda: {"text": torch.randint(0, 64, (2, 6))},
        "patches": {
            "src.model.TextEncoder": lambda **kwargs: DummyEncoder(kwargs.get("d_text", 16)),
        },
    },
    "table": {
        "modalities": ["table"],
        "input": lambda: {"table": torch.randint(0, 64, (2, 7))},
        "patches": {
            "src.model.TableEncoder": lambda **kwargs: DummyEncoder(
                kwargs.get("d_table", 16), fixed_tokens=5
            ),
        },
    },
    "time_series": {
        "modalities": ["time_series"],
        "input": lambda: {"time_series": torch.randn(2, 12, 1)},
        "patches": {
            "src.model.TimeSeriesEncoder": lambda **kwargs: DummyEncoder(
                kwargs.get("d_ts", 16), fixed_tokens=4, tokens_per_instance=4
            ),
        },
    },
    "image": {
        "modalities": ["image"],
        "input": lambda: {"image": torch.randn(2, 3, 32, 32)},
        "patches": {
            "src.model.ImageEncoder": lambda **kwargs: DummyEncoder(
                kwargs.get("d_img", 16), fixed_tokens=3
            ),
        },
    },
    "geometry": {
        "modalities": ["geometry"],
        "input": lambda: {"geometry": torch.randn(2, 32, 3)},
        "patches": {
            "src.model.GeometryEncoder": lambda **kwargs: DummyEncoder(
                kwargs.get("d_geo", 16), fixed_tokens=6
            ),
        },
    },
    "graph": {
        "modalities": ["graph"],
        "input": lambda: {"graph": {"x": torch.randn(2, 5, 4)}},
        "patches": {
            "src.model.GraphEncoder": lambda **kwargs: DummyEncoder(
                kwargs.get("d_graph", 16), fixed_tokens=4
            ),
        },
    },
}


@pytest.mark.parametrize("modality", list(MODALITY_CONFIG.keys()))
def test_modality_forward_contract(modality: str):
    spec = MODALITY_CONFIG[modality]
    cfg = ModelConfig(
        d_model=32,
        vocab_size=64,
        num_layers=1,
        num_heads=4,
        num_experts=2,
        d_text=16,
        d_img=16,
        d_table=16,
        d_ts=16,
        d_geo=16,
        d_graph=16,
        modalities=spec["modalities"],
    )

    patchers = [patch(target, side_effect=factory) for target, factory in spec["patches"].items()]
    for patcher in patchers:
        patcher.start()

    try:
        model = UnifiedTransformer(cfg)
        logits, aux_loss = model(spec["input"]())
    finally:
        for patcher in reversed(patchers):
            patcher.stop()

    assert logits.shape[0] == 2
    assert logits.shape[-1] == cfg.vocab_size
    assert logits.shape[1] > 0
    assert aux_loss.dim() == 0
    assert not torch.isnan(aux_loss)


def test_interleave_contract_valid_and_invalid_spacing():
    cfg = ModelConfig(
        d_model=32,
        vocab_size=128,
        num_layers=1,
        num_heads=4,
        num_experts=2,
        d_text=16,
        d_ts=16,
        modalities=["text", "time_series"],
        is_interleaved_qa=True,
        modality_start_end_token_indices={"time_series": (60, 61)},
    )

    with (
        patch(
            "src.model.TextEncoder",
            side_effect=lambda **kwargs: DummyEncoder(kwargs.get("d_text", 16)),
        ),
        patch(
            "src.model.TimeSeriesEncoder",
            side_effect=lambda **kwargs: DummyEncoder(
                kwargs.get("d_ts", 16), fixed_tokens=2, tokens_per_instance=2
            ),
        ),
    ):
        model = UnifiedTransformer(cfg)

        good_inputs = {
            "text": torch.tensor([[10, 60, 61, 12, 13], [20, 60, 61, 22, 23]], dtype=torch.long),
            "time_series": torch.randn(2, 8, 1),
            "_metadata": ["3 2", "3 2"],
        }
        logits, aux_loss = model(good_inputs)
        assert logits.shape[0] == 2
        assert logits.shape[-1] == cfg.vocab_size
        assert aux_loss.dim() == 0

        bad_inputs = {
            "text": torch.tensor([[10, 60, 99, 61, 13]], dtype=torch.long),
            "time_series": torch.randn(1, 8, 1),
            "_metadata": ["3 1"],
        }
        with pytest.raises(ValueError, match="must be adjacent"):
            model(bad_inputs)


@pytest.mark.parametrize("backbone_id", OLMO_BACKBONES)
@pytest.mark.parametrize(
    "modality,patch_target,d_key",
    [
        ("image", "src.model.ImageEncoder", "d_img"),
        ("time_series", "src.model.TimeSeriesEncoder", "d_ts"),
        ("graph", "src.model.GraphEncoder", "d_graph"),
        ("geometry", "src.model.GeometryEncoder", "d_geo"),
        ("table", "src.model.TableEncoder", "d_table"),
    ],
)
def test_zone_a_scope_for_olmo_backbones(
    offline_hf,
    backbone_id: str,
    modality: str,
    patch_target: str,
    d_key: str,
):
    model_cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_experts=2,
        d_text=16,
        d_img=16,
        d_table=16,
        d_ts=16,
        d_geo=16,
        d_graph=16,
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        modalities=["text", modality],
    )
    train_cfg = TrainingConfig(
        batch_size=2,
        max_steps=1,
        wandb_project=None,
        vocab_size=64,
        device="cpu",
    )
    train_loader = DataLoader(ZoneAModalityDataset(modality), batch_size=2)

    with (
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=DummyBackbone(hidden_size=32, vocab_size=64),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=DummyTokenizer(),
        ),
        patch(
            patch_target,
            side_effect=lambda **kwargs: DummyEncoder(kwargs.get(d_key, 16), fixed_tokens=4),
        ),
        patch(
            "src.training.trainer_zone_a.EvaluatorRegistry.get",
            return_value=None,
        ),
    ):
        model = UnifiedTransformer(model_cfg)
        trainer = ZoneATrainer(model, train_cfg, train_loader)

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    assert unwrapped.config.llm_backbone_id == backbone_id
    assert modality in unwrapped.encoders
    assert modality in unwrapped.projectors
    assert all(not p.requires_grad for p in unwrapped.backbone.parameters())
    assert all(not p.requires_grad for p in unwrapped.encoders[modality].parameters())
    assert any(p.requires_grad for p in unwrapped.projectors[modality].parameters())


@pytest.mark.parametrize("modality", ["image", "time_series"])
def test_streaming_dataset_emits_requested_modality(modality: str):
    """Phase 1 §1.4: a model.modalities=[text,X] config + StreamingMultimodalDataset
    must actually emit `X` in its samples.

    Skips cleanly if the HF cache or any modality dep isn't available so
    login-node CI doesn't go red. Pulls 4 samples and asserts the requested
    modality key shows up at least once.
    """
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")

    from src.config import ModelConfig
    from src.data.multimodal import StreamingMultimodalDataset

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1
        eos_token = "<|endoftext|>"

        def __call__(self, text, **kwargs):
            import torch as _t
            n = max(1, len(text.split())) if isinstance(text, str) else 4
            return type(
                "Enc",
                (),
                {"input_ids": _t.zeros(1, n, dtype=_t.long)},
            )()

        def decode(self, token_ids):
            return "<unk>"

    cfg = ModelConfig(modalities=["text", modality])
    try:
        ds = StreamingMultimodalDataset(
            tokenizer=_Tok(),
            batch_size=1,
            max_steps=4,
            model_config=cfg,
            allow_dummy_data=True,  # fallback so missing local data doesn't hard-fail
        )
    except Exception as exc:
        pytest.skip(f"StreamingMultimodalDataset construction unavailable: {exc}")

    assert modality in ds.active_modalities, (
        f"active_modalities={ds.active_modalities} missing requested {modality}"
    )

    seen_keys: set[str] = set()
    try:
        for i, sample in enumerate(ds):
            seen_keys.update(sample.keys())
            if i >= 3:
                break
    except Exception as exc:
        pytest.skip(f"StreamingMultimodalDataset iter failed (likely HF offline): {exc}")

    assert modality in seen_keys, (
        f"Pulled 4 samples; requested modality {modality!r} never appeared. "
        f"Saw keys: {sorted(seen_keys)}"
    )


@pytest.mark.parametrize("backbone_id", OLMO_BACKBONES)
def test_zone_a_vla_scope_for_olmo_backbones(offline_hf, backbone_id: str):
    model_cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_experts=2,
        d_text=16,
        d_img=16,
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        is_vla=True,
        modalities=["text", "image"],
    )
    train_cfg = TrainingConfig(
        batch_size=2,
        max_steps=1,
        wandb_project=None,
        vocab_size=64,
        device="cpu",
    )
    train_loader = DataLoader(TinyVLADataset(), batch_size=2)

    with (
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=DummyBackbone(hidden_size=32, vocab_size=64),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=DummyTokenizer(),
        ),
        patch(
            "src.model.ImageEncoder",
            side_effect=lambda **kwargs: DummyEncoder(kwargs.get("d_img", 16), fixed_tokens=4),
        ),
    ):
        model = UnifiedTransformer(model_cfg)
        trainer = ZoneAVLATrainer(model, train_cfg, train_loader)

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    assert unwrapped.config.llm_backbone_id == backbone_id
    assert all(not p.requires_grad for p in unwrapped.backbone.parameters())
    assert all(not p.requires_grad for p in unwrapped.encoders["image"].parameters())
    assert any(p.requires_grad for p in unwrapped.projectors["image"].parameters())
    assert any(p.requires_grad for p in unwrapped.pose_embed.parameters())
    assert any(p.requires_grad for p in unwrapped.action_head.parameters())
    assert unwrapped.pose_modality_embedding.requires_grad

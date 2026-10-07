"""Offline contract tests; these fixtures do not validate pretrained checkpoints."""

from types import SimpleNamespace

import pytest
import torch
from src.config import ModelConfig
from src.decoders import (
    DecoderCondition,
    GeometryDecoder,
    GraphDecoder,
    LMHeadDecoder,
    TimeSeriesDecoder,
)
from src.decoders.base import OutputDecoder
from src.decoders.types import masked_pool
from src.model import UnifiedTransformer
from torch import nn


class FixtureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.head = nn.Linear(4, 32)
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        self.calls.append((inputs_embeds.detach().clone(), attention_mask.clone()))
        hidden = inputs_embeds.cumsum(dim=1)
        return SimpleNamespace(hidden_states=[hidden], logits=self.head(hidden))

    def generate(self, **kwargs):
        return torch.tensor([[7, 8]])


class FixtureImageEncoder(nn.Module):
    def forward(self, pixels):
        return pixels.mean((1, 2, 3))[:, None, None].expand(-1, 2, 4)

    def tokens_per_instance(self):
        return 2


class CaptureDecoder(OutputDecoder):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 1)
        self.conditions = []

    def forward_condition(self, condition, targets=None, **kwargs):
        self.conditions.append(condition)
        return self.forward(
            condition.hidden_states, targets, attention_mask=condition.attention_mask
        )

    def forward(self, hidden_states, targets=None, **kwargs):
        prediction = self.head(masked_pool(hidden_states, kwargs["attention_mask"], "mean"))
        return prediction, None if targets is None else (prediction - targets).square().mean()


def make_model(interleave=False, names=("text", "image")):
    model = UnifiedTransformer.__new__(UnifiedTransformer)
    nn.Module.__init__(model)
    model.config = ModelConfig(
        modalities=["text", "image"],
        output_decoders=list(names),
        llm_backbone_id="fixture",
        is_interleaved_qa=interleave,
        modality_start_end_token_indices={"image": [30, 31]},
        max_merged_seq_length=32,
    )
    model.backbone = FixtureBackbone()
    model.is_vla = False
    model.encoders = nn.ModuleDict({"image": FixtureImageEncoder()})
    model.projectors = nn.ModuleDict({"image": nn.Identity()})
    model.decoders = nn.ModuleDict({"text": LMHeadDecoder(), "image": CaptureDecoder()})
    model.text_decoder = model.decoders["text"]
    model.tokenizer = SimpleNamespace(pad_token_id=0)
    return model


def test_prompt_only_targets_cannot_leak_into_condition():
    model = make_model()
    inputs = {"text": torch.tensor([[2, 3, 0]]), "image_target": torch.ones(1, 3, 2, 2)}
    a = model.forward_outputs(
        inputs, targets={"image": torch.ones(1, 1)}, requested_outputs=["image"]
    )
    b = model.forward_outputs(
        inputs, targets={"image": torch.zeros(1, 1)}, requested_outputs=["image"]
    )
    assert torch.equal(a.predictions["image"], b.predictions["image"])
    assert set(a.losses) == {"image"}
    assert model.backbone.calls[-1][0].shape[1] == 2


def test_multi_reference_order_and_masks():
    model = make_model()
    refs = torch.stack([torch.ones(2, 3, 2, 2), torch.ones(2, 3, 2, 2) * 5])
    refs[0, 1] *= 2
    refs[1, 1] = 999  # padding cannot enter the condition
    inputs = {
        "text": torch.tensor([[2, 3], [4, 0]]),
        "image": refs,
        "image_mask": torch.tensor([[1, 1], [1, 0]]),
    }
    model.forward_outputs(inputs, requested_outputs=["image"])
    x, mask = model.backbone.calls[-1]
    assert mask.sum(1).tolist() == [6, 3]
    assert x[0, :4, 0].tolist() == [1, 1, 2, 2]
    assert x[1, :2, 0].tolist() == [5, 5]
    condition = model.decoders["image"].conditions[-1]
    assert condition.modality_spans["image"] == [[(0, 2), (2, 4)], [(0, 2)]]


def test_prompt_interleaving_without_qa_metadata():
    model = make_model(interleave=True)
    refs = torch.ones(1, 2, 3, 2, 2)
    refs[:, 1] *= 9
    inputs = {"text": torch.tensor([[2, 30, 31, 3, 30, 31, 4, 0]]), "image": refs}
    result = model.forward_outputs(inputs, requested_outputs=["image"])
    assert result.loss is None
    x, mask = model.backbone.calls[-1]
    assert mask.sum() == 7
    assert x[0, 1:3, 0].tolist() == [1, 1]
    assert x[0, 4:6, 0].tolist() == [9, 9]


def test_text_only_interleave_allows_absent_image():
    model = make_model(interleave=True)
    assert (
        model.forward_outputs({"text": torch.tensor([[2, 3]])}, requested_outputs=["image"]).loss
        is None
    )


def test_missing_or_unconsumed_references_fail_before_backbone():
    model = make_model(interleave=True)
    with pytest.raises(ValueError, match="Missing reference"):
        model.forward_outputs({"text": torch.tensor([[2, 30, 31]])}, requested_outputs=["image"])
    with pytest.raises(ValueError, match="Unconsumed"):
        model.forward_outputs(
            {"text": torch.tensor([[2, 3]]), "image": torch.ones(1, 1, 3, 2, 2)},
            requested_outputs=["image"],
        )
    assert not model.backbone.calls


def test_loss_weights_and_gradient_flow_through_frozen_backbone():
    model = make_model()
    model.backbone.requires_grad_(False)
    model.config.decoder_loss_weights = {"image": 2.5}
    result = model(
        {"text": torch.tensor([[2, 3]])},
        targets={"image": torch.ones(1, 1)},
        requested_outputs=["image"],
    )
    torch.testing.assert_close(result.loss, 2.5 * result.losses["image"])
    result.loss.backward()
    assert model.decoders["image"].head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_text_target_alignment_and_combined_outputs():
    model = make_model(interleave=True)
    inputs = {"text": torch.tensor([[2, 30, 31, 3, 4]]), "image": torch.ones(1, 1, 3, 2, 2)}
    result = model.forward_outputs(
        inputs,
        targets={"text": torch.tensor([[-100, -100, -100, 3, 4]]), "image": torch.ones(1, 1)},
        requested_outputs=["text", "image"],
    )
    assert set(result.losses) == {"text", "image"}
    torch.testing.assert_close(result.loss, sum(result.losses.values()))
    with pytest.raises(ValueError, match="no next-token"):
        model.forward_outputs(
            inputs, targets={"text": torch.full((1, 5), -100)}, requested_outputs=["text"]
        )


def test_explicit_text_mask_preserves_valid_eos_equal_to_pad():
    model = make_model()
    model.forward_outputs(
        {"text": torch.tensor([[2, 0, 0]]), "text_attention_mask": torch.tensor([[1, 1, 0]])},
        requested_outputs=["image"],
    )
    assert model.backbone.calls[-1][1].sum() == 2


def test_length_guard_before_backbone_and_unknown_decoder():
    model = make_model()
    model.config.max_merged_seq_length = 2
    with pytest.raises(ValueError, match="Merged sequence"):
        model.forward_outputs({"text": torch.tensor([[1, 2, 3]])}, requested_outputs=["image"])
    with pytest.raises(ValueError, match="not configured"):
        model.forward_outputs({"text": torch.tensor([[1]])}, requested_outputs=["geometry"])
    assert not model.backbone.calls


@pytest.mark.parametrize("pool", ["mean", "last"])
@pytest.mark.parametrize("kind", ["time_series", "geometry"])
def test_scientific_pooling_ignores_padding(kind, pool):
    decoder = (
        TimeSeriesDecoder(4, 2, pool=pool)
        if kind == "time_series"
        else GeometryDecoder(4, 2, 1, pool=pool)
    )
    short = torch.randn(1, 2, 4)
    padded = torch.cat([short, torch.full((1, 3, 4), 1000.0)], 1)
    a = decoder.generate_condition(DecoderCondition(short, torch.ones(1, 2)))
    b = decoder.generate_condition(DecoderCondition(padded, torch.tensor([[1, 1, 0, 0, 0]])))
    torch.testing.assert_close(a, b)


def test_graph_requires_node_correspondence():
    decoder = GraphDecoder(4, 2)
    hidden = torch.randn(1, 4, 4)
    with pytest.raises(ValueError, match="explicit node_indices"):
        decoder.generate_condition(DecoderCondition(hidden, torch.ones(1, 4)))
    condition = DecoderCondition(
        hidden,
        torch.tensor([[1, 1, 1, 0]]),
        native_context={"node_indices": torch.tensor([[2, 0]])},
    )
    torch.testing.assert_close(
        decoder.generate_condition(condition), decoder.generate(hidden[:, [2, 0]])
    )
    condition.native_context["node_indices"] = torch.tensor([[3]])
    with pytest.raises(ValueError, match="padding"):
        decoder.generate_condition(condition)


def test_invalid_loss_weights():
    with pytest.raises(ValueError, match="nonnegative"):
        ModelConfig(decoder_loss_weights={"text": float("nan")})


def test_predict_routes_native_generation():
    model = make_model()
    result = model.predict({"text": torch.tensor([[2, 3]])}, requested_outputs=["text", "image"])
    assert result.predictions["text"].tolist() == [[7, 8]]
    assert result.predictions["image"].shape == (1, 1)
    assert result.loss is None


def test_native_trainer_calls_public_forward_with_named_targets():
    from src.training.trainer_native import forward_training_batch

    model = make_model()
    seen = []
    handle = model.register_forward_hook(lambda *args: seen.append(True))
    logits, loss, losses = forward_training_batch(
        model,
        {
            "inputs": {"text": torch.tensor([[2, 3]])},
            "targets": {"image": torch.ones(1, 1)},
            "requested_outputs": ["image"],
        },
    )
    handle.remove()
    assert seen == [True] and logits is None and loss.requires_grad
    assert set(losses) == {"image"}


def test_padding_never_enters_batch_statistic_projector():
    from src.modules import ModalityProjector

    model = make_model()
    model.projectors["image"] = ModalityProjector(4, 4, norm_mode="match_text_stats")
    refs = torch.randn(2, 2, 3, 2, 2)
    inputs = {
        "text": torch.tensor([[2], [3]]),
        "image": refs,
        "image_mask": torch.tensor([[1, 1], [1, 0]]),
    }
    a = model.forward_outputs(inputs, requested_outputs=["image"]).predictions["image"]
    refs[1, 1] = 100000
    b = model.forward_outputs(inputs, requested_outputs=["image"]).predictions["image"]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_ambiguous_pad_and_eos_require_mask():
    model = make_model()
    model.tokenizer.eos_token_id = 0
    with pytest.raises(ValueError, match="PAD and EOS"):
        model.forward_outputs({"text": torch.tensor([[2, 0]])}, requested_outputs=["image"])

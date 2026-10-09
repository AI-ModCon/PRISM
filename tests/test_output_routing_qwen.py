"""Offline HF API regression tests with a tiny, randomly initialized Qwen3.

No pretrained weights, tokenizer, network access, or image generator are used.
These tests establish the Transformers call/mask contract, not checkpoint or
image-generation acceptance.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3ForCausalLM"):
    pytest.skip("Qwen3 requires a recent Transformers release", allow_module_level=True)

from src.config import ModelConfig
from src.decoders import GeometryDecoder, LMHeadDecoder, TimeSeriesDecoder
from src.model import UnifiedTransformer

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def random_qwen_prism():
    """Construct locally from config; never call any from_pretrained loader."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(314)
        qwen_config = transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=None,
        )
        qwen_config._attn_implementation = "eager"
        model = UnifiedTransformer.__new__(UnifiedTransformer)
        nn.Module.__init__(model)
        model.config = ModelConfig(
            modalities=["text"],
            output_decoders=["text", "time_series", "geometry"],
            llm_backbone_id="random-qwen3-contract-fixture",
            max_merged_seq_length=64,
        )
        model.backbone = transformers.Qwen3ForCausalLM(qwen_config)
        model.is_vla = False
        model.encoders = nn.ModuleDict()
        model.projectors = nn.ModuleDict()
        model.decoders = nn.ModuleDict(
            {
                "text": LMHeadDecoder(),
                "time_series": TimeSeriesDecoder(32, horizon=2, num_vars=1, pool="mean"),
                "geometry": GeometryDecoder(32, num_points=3, num_channels=2, pool="last"),
            }
        )
        model.text_decoder = model.decoders["text"]
        model.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
    return model.eval()


def padded_inputs():
    return {
        "text": torch.tensor([[0, 0, 4, 5], [6, 7, 8, 9]]),
        "text_attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
    }


def single_row(inputs, row):
    ids = inputs["text"][row][inputs["text_attention_mask"][row].bool()].unsqueeze(0)
    return {"text": ids, "text_attention_mask": torch.ones_like(ids)}


def test_output_condition_uses_actual_qwen_final_hidden_states(random_qwen_prism):
    model = random_qwen_prism
    with torch.no_grad():
        condition, logits, positions, embeddings = model._output_condition(padded_inputs())
        direct = model.backbone(
            input_ids=torch.tensor([[4, 5]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            position_ids=torch.tensor([[0, 1]]),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    assert condition.hidden_states.shape == (2, 4, 32)
    assert logits.shape == (2, 4, 64)
    assert embeddings.shape == (2, 4, 32)
    assert condition.attention_mask.tolist() == [[True, True, False, False], [True] * 4]
    assert positions.tolist() == [[-1, -1, 0, 1], [0, 1, 2, 3]]
    assert torch.isfinite(condition.hidden_states).all()
    torch.testing.assert_close(condition.hidden_states[:1, :2], direct.hidden_states[-1])
    torch.testing.assert_close(logits[:1, :2], direct.logits)


def test_left_padded_qwen_generation_matches_each_unpadded_row(random_qwen_prism):
    model = random_qwen_prism
    inputs = padded_inputs()
    masks = []

    def capture_mask(module, args, kwargs):
        if kwargs.get("inputs_embeds") is not None:
            masks.append(kwargs["attention_mask"].detach().clone())

    handle = model.backbone.register_forward_pre_hook(capture_mask, with_kwargs=True)
    options = {"text": {"max_new_tokens": 4, "do_sample": False, "use_cache": True}}
    try:
        batch = model.predict(inputs, requested_outputs=["text"], decoder_kwargs=options)
    finally:
        handle.remove()
    assert batch.predictions["text"].shape == (2, 4)
    # The compiler right-pads, but HF generation must receive left-padded prefill.
    assert any(torch.equal(mask, inputs["text_attention_mask"]) for mask in masks)
    for row in range(2):
        single = model.predict(
            single_row(inputs, row), requested_outputs=["text"], decoder_kwargs=options
        )
        torch.testing.assert_close(batch.predictions["text"][row], single.predictions["text"][0])


def test_scientific_predictions_ignore_padding_with_actual_qwen_attention(random_qwen_prism):
    model = random_qwen_prism
    inputs = padded_inputs()
    names = ["time_series", "geometry"]
    batch = model.predict(inputs, requested_outputs=names)
    assert batch.loss is None
    assert batch.predictions["time_series"].shape == (2, 2, 1)
    assert batch.predictions["geometry"].shape == (2, 3, 2)
    for row in range(2):
        single = model.predict(single_row(inputs, row), requested_outputs=names)
        for name in names:
            torch.testing.assert_close(
                batch.predictions[name][row], single.predictions[name][0], atol=1e-6, rtol=1e-5
            )
    changed = {**inputs, "text": inputs["text"].clone()}
    changed["text"][0, :2] = torch.tensor([52, 53])
    perturbed = model.predict(changed, requested_outputs=names)
    for name in names:
        torch.testing.assert_close(
            batch.predictions[name], perturbed.predictions[name], atol=0, rtol=0
        )

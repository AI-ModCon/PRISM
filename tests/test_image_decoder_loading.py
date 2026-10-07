import pytest
import torch
from src.decoders.loading import load_image_connector, preprocessing_sha256, restore_prism_parent
from torch import nn


def test_preprocessing_identity_tracks_contents_not_location(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for root in (a, b):
        (root / "tokenizer.json").write_text('{"vocab":{"a":0,"b":1}}')
        (root / "model.safetensors").write_bytes(b"ignored model weights")
    assert preprocessing_sha256(a) == preprocessing_sha256(b)
    (b / "tokenizer.json").write_text('{"vocab":{"a":1,"b":0}}')
    assert preprocessing_sha256(a) != preprocessing_sha256(b)


def test_connector_reload_and_parent_mismatch(tmp_path):
    model = nn.Module()
    image = nn.Module()
    image.connector = nn.Linear(3, 4)
    model.decoders = nn.ModuleDict({"image": image})
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = tmp_path / "connector.pt"
    torch.save(
        {
            "schema_version": 1,
            "evidence_kind": "real_checkpoint_training",
            "provenance": {
                "parent_checkpoint_sha256": "a" * 64,
                "reference_checkpoint_sha256": "b" * 64,
            },
            "connector_state_dict": state,
            "step": 2,
        },
        checkpoint,
    )
    with torch.no_grad():
        image.connector.weight.zero_()
    load_image_connector(
        model, checkpoint, parent_checkpoint_sha256="a" * 64, reference_checkpoint_sha256="b" * 64
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key])
    with pytest.raises(ValueError, match="parent identity"):
        load_image_connector(
            model,
            checkpoint,
            parent_checkpoint_sha256="c" * 64,
            reference_checkpoint_sha256="b" * 64,
        )


def test_fixture_checkpoint_cannot_masquerade_as_trained_connector(tmp_path):
    path = tmp_path / "fixture.pt"
    torch.save({"schema_version": 1, "evidence_kind": "fixture_only"}, path)
    with pytest.raises(ValueError, match="accepted training"):
        load_image_connector(
            nn.Module(),
            path,
            parent_checkpoint_sha256="a" * 64,
            reference_checkpoint_sha256="b" * 64,
        )


class FixtureTokenizer:
    """Tiny local vocabulary; no pretrained-model claim."""

    all_special_ids = [0, 1]

    def __len__(self):
        return 5

    def get_vocab(self):
        return {str(index): index for index in range(5)}


class FixtureBackbone(nn.Module):
    def __init__(self, rows=8, *, tied=False):
        super().__init__()
        # Deliberately not Qwen's model.embed_tokens name.
        self.tokens = nn.Embedding(rows, 3, dtype=torch.float16)
        self.output = nn.Linear(3, rows, bias=False, dtype=torch.float16)
        self.tied = tied
        if tied:
            self.output.weight = self.tokens.weight

    def get_input_embeddings(self):
        return self.tokens

    def resize_token_embeddings(self, rows):
        self.tokens = nn.Embedding(rows, 3, dtype=torch.float16)
        self.output = nn.Linear(3, rows, bias=False, dtype=torch.float16)
        if self.tied:
            self.output.weight = self.tokens.weight


def _parent_fixture(*, tied=False):
    model = nn.Module()
    model.backbone = FixtureBackbone(tied=tied)
    model.encoders = nn.ModuleDict({"image": nn.Linear(2, 2)})
    model.projectors = nn.ModuleDict({"image": nn.Linear(2, 3)})
    image = nn.Module()
    image.connector = nn.Linear(3, 4)
    model.decoders = nn.ModuleDict({"image": image})
    state = {
        name: value.detach().clone().to(torch.bfloat16)
        for name, value in model.state_dict().items()
        if not name.startswith("decoders.image.connector.")
    }
    for name in ("backbone.tokens.weight", "backbone.output.weight"):
        state[name] = state[name][:6].clone()
    return model, state


def test_parent_restores_trained_lm_encoder_projector_and_unpadded_vocab():
    model, state = _parent_fixture()
    # BF16 represents this value, but a temporary FP16 backbone overflows it.
    state["backbone.tokens.weight"][0, 0] = 100_352
    fresh_connector = {name: value.clone() for name, value in model.decoders["image"].state_dict().items()}
    rng = torch.get_rng_state()
    report = restore_prism_parent(model, state, FixtureTokenizer())
    assert torch.equal(rng, torch.get_rng_state())
    restored = model.state_dict()
    for name, value in state.items():
        assert restored[name].dtype == value.dtype
        assert torch.equal(restored[name], value)
    for name, value in model.decoders["image"].state_dict().items():
        assert torch.equal(value, fresh_connector[name])
    assert report["backbone_vocab_resize"] == {
        "embedding_key": "backbone.tokens.weight", "old_rows": 8, "checkpoint_rows": 6
    }
    assert report["strict_parent"]
    assert report["missing_parent_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["loaded_keys_by_component"] == {"backbone": 2, "encoders": 2, "projectors": 2}
    assert report["checkpoint_tensor_dtypes"] == {"torch.bfloat16": 6}
    assert report["loaded_key_count"] == 6


def test_parent_normalizes_only_known_wrappers_and_rejects_collisions():
    model, state = _parent_fixture()
    wrapped = {"module._fsdp_wrapped_module." + key: value for key, value in state.items()}
    wrapped["module._fsdp_wrapped_module.backbone._orig_mod.tokens.weight"] = wrapped.pop(
        "module._fsdp_wrapped_module.backbone.tokens.weight"
    )
    report = restore_prism_parent(model, wrapped, FixtureTokenizer())
    assert len(report["normalized_keys"]) == len(state)
    wrapped["backbone.tokens.weight"] = state["backbone.tokens.weight"]
    with pytest.raises(ValueError, match="collide"):
        restore_prism_parent(model, wrapped, FixtureTokenizer())


@pytest.mark.parametrize("component", ["backbone", "encoders", "projectors"])
def test_parent_requires_every_trained_component(component):
    model, state = _parent_fixture()
    state = {name: value for name, value in state.items() if not name.startswith(component + ".")}
    with pytest.raises(ValueError, match="Parent checkpoint incompatible"):
        restore_prism_parent(model, state, FixtureTokenizer())


def test_parent_rejects_tokenizer_ids_outside_checkpoint_vocabulary():
    model, state = _parent_fixture()
    tokenizer = FixtureTokenizer()
    tokenizer.get_vocab = lambda: {"sparse_high_id": 8}
    with pytest.raises(ValueError, match="cannot represent tokenizer IDs"):
        restore_prism_parent(model, state, tokenizer)


def test_parent_rejects_wrong_shapes_and_extra_semantic_keys():
    model, state = _parent_fixture()
    state["projectors.image.weight"] = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="tensor shapes"):
        restore_prism_parent(model, state, FixtureTokenizer())
    model, state = _parent_fixture()
    state["other_model.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="unexpected"):
        restore_prism_parent(model, state, FixtureTokenizer())


def test_parent_restores_tied_weights_and_rejects_inconsistent_alias_values():
    model, state = _parent_fixture(tied=True)
    report = restore_prism_parent(model, state, FixtureTokenizer())
    assert model.backbone.tokens.weight is model.backbone.output.weight
    assert report["restored_alias_groups"] == [["backbone.tokens.weight", "backbone.output.weight"]]
    model, state = _parent_fixture(tied=True)
    state["backbone.output.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="inconsistent tied tensors"):
        restore_prism_parent(model, state, FixtureTokenizer())


def test_random_hf_non_qwen_parent_preserves_vocab_tying_and_checkpoint_precision():
    """Exercise actual HF resize/load APIs with random offline GPT2, not trained weights."""
    from transformers import GPT2Config, GPT2LMHeadModel

    model = nn.Module()
    model.backbone = GPT2LMHeadModel(
        GPT2Config(vocab_size=16, n_embd=8, n_layer=1, n_head=2, n_positions=16)
    ).eval()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name in ("backbone.transformer.wte.weight", "backbone.lm_head.weight"):
        state[name] = state[name][:12].clone()
    model.half()
    model.requires_grad_(False)
    report = restore_prism_parent(model, state, FixtureTokenizer())
    assert report["backbone_vocab_resize"]["checkpoint_rows"] == 12
    assert model.backbone.config.vocab_size == 12
    assert model.backbone.lm_head.weight is model.backbone.transformer.wte.weight
    assert not any(parameter.requires_grad for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert value.dtype == state[name].dtype
        assert torch.equal(value, state[name])
    with torch.no_grad():
        logits = model.backbone(input_ids=torch.tensor([[1, 2]])).logits
    assert logits.shape == (1, 2, 12)
    assert torch.isfinite(logits).all()

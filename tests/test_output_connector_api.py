"""CPU contracts for the OmniGen2 output-connector route.

These tests establish API/legacy compatibility, masking, and autograd, not
pretrained checkpoint acceptance, accelerator parity, or image quality.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from src.config import ModelConfig
from src.connectors import (
    BackboneFeatures,
    ConditioningBridge,
    DecoderContext,
    Readout,
    ReadoutResult,
    build_bridge,
    build_readout,
    pack_right_padded,
)
from src.decoders.image import ImageDecoder
from src.decoders.omnigen2_backend import DEFAULT_REVISION, OmniGen2Backend
from torch import nn

from test_image_decoder import FixtureBackend, condition

pytestmark = pytest.mark.unit


def explicit_route():
    return {
        "readout": {"type": "select", "layers": "final", "positions": "all_valid"},
        "bridge": {"type": "layernorm_linear", "output_dim": 5},
        "generator": {"type": "omnigen2", "conditioning_dim": 5},
    }


def old_connect(connector, hidden, mask):
    """Independent copy of the pre-refactor connector's public behavior."""
    valid = mask.bool()
    sanitized = hidden.masked_fill(~valid.unsqueeze(-1), 0)
    connected = connector(sanitized.to(connector[1].weight))
    order = torch.argsort(valid.to(torch.int64), dim=1, descending=True, stable=True)
    length = int(valid.sum(1).max())
    connected = connected.gather(1, order.unsqueeze(-1).expand_as(connected))[:, :length]
    packed_mask = torch.arange(length)[None, :] < valid.sum(1)[:, None]
    return connected.masked_fill(~packed_mask.unsqueeze(-1), 0), packed_mask


def test_explicit_omnigen2_route_has_exact_legacy_initialization_and_state_keys():
    torch.manual_seed(31)
    legacy = ImageDecoder(8, backend=FixtureBackend())
    torch.manual_seed(31)
    explicit = ImageDecoder(8, backend=FixtureBackend(), **explicit_route())
    assert set(legacy.state_dict()) == set(explicit.state_dict())
    assert set(explicit.connector.state_dict()) == {"0.weight", "0.bias", "1.weight", "1.bias"}
    assert list(explicit.readout.parameters()) == []
    for name, tensor in legacy.state_dict().items():
        torch.testing.assert_close(tensor, explicit.state_dict()[name], rtol=0, atol=0)
    assert isinstance(explicit.readout, Readout)
    assert isinstance(explicit.connector, ConditioningBridge)
    assert isinstance(explicit.connector, nn.Sequential)
    assert isinstance(explicit.connector[0], nn.LayerNorm)
    assert isinstance(explicit.connector[1], nn.Linear)


def test_current_connector_initialization_matches_original_sequential_module():
    torch.manual_seed(31)
    FixtureBackend()  # Reproduce the injected backend's random-number consumption.
    original = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 5))
    torch.manual_seed(31)
    current = ImageDecoder(8, backend=FixtureBackend(), **explicit_route())
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, current.connector.state_dict()[name], rtol=0, atol=0)


def test_explicit_route_matches_legacy_output_and_gradients_with_masked_nans():
    torch.manual_seed(19)
    decoder = ImageDecoder(8, backend=FixtureBackend(), **explicit_route())
    oracle = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 5))
    oracle.load_state_dict(decoder.connector.state_dict(), strict=True)
    hidden = torch.randn(2, 5, 8)
    mask = torch.tensor([[0, 1, 0, 1, 1], [1, 0, 0, 0, 1]])
    hidden.masked_fill_(~mask.bool().unsqueeze(-1), float("nan"))
    actual_hidden = hidden.clone().requires_grad_()
    expected_hidden = hidden.clone().requires_grad_()
    actual, actual_mask = decoder.connect(condition(actual_hidden, mask))
    expected, expected_mask = old_connect(oracle, expected_hidden, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual_hidden.grad, expected_hidden.grad, rtol=0, atol=0)
    assert actual_hidden.grad[~mask.bool()].eq(0).all()
    for (_, new), (_, old) in zip(
        decoder.connector.named_parameters(), oracle.named_parameters(), strict=True
    ):
        assert torch.isfinite(new.grad).all()
        torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0)


@pytest.mark.parametrize("explicit_source", [False, True])
def test_checkpoint_load_is_strict_in_both_directions(explicit_source):
    source = ImageDecoder(
        8, backend=FixtureBackend(), **(explicit_route() if explicit_source else {})
    )
    destination = ImageDecoder(
        8, backend=FixtureBackend(), **({} if explicit_source else explicit_route())
    )
    incompatible = destination.load_state_dict(deepcopy(source.state_dict()), strict=True)
    assert incompatible.missing_keys == incompatible.unexpected_keys == []
    cond = condition(mask=torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]]))
    torch.testing.assert_close(
        source.generate_condition(cond), destination.generate_condition(cond), rtol=0, atol=0
    )


def test_readout_bridge_and_layout_preserve_original_coordinate_metadata():
    hidden = torch.arange(80, dtype=torch.float32).reshape(2, 5, 8)
    mask = torch.tensor([[0, 1, 0, 1, 1], [1, 0, 0, 0, 1]])
    spans = {"image": [[(1, 2), (3, 4)], [(0, 1)]]}
    provenance = {"parent_checkpoint": "fixture", "source": {"layer": "final"}}
    features = BackboneFeatures(hidden, mask, modality_spans=spans, provenance=provenance)
    readout = build_readout(explicit_route()["readout"], d_model=8)
    bridge = build_bridge(explicit_route()["bridge"], input_dim=8, output_dim=5)
    selected = readout(features)
    assert isinstance(selected, ReadoutResult)
    connected = bridge.connect(selected)
    assert isinstance(connected, DecoderContext)
    packed = pack_right_padded(connected)
    assert packed.tokens.shape == (2, 3, 5)
    assert packed.attention_mask.tolist() == [[True, True, True], [True, True, False]]
    assert packed.source_positions.tolist() == [[1, 3, 4], [0, 4, -1]]
    assert packed.source_modality_spans == spans
    assert packed.provenance["parent_checkpoint"] == "fixture"
    assert packed.provenance["source"] == {"layer": "final"}
    assert packed.tokens[1, 2].eq(0).all()
    torch.testing.assert_close(packed.tokens[0], bridge(hidden[0, [1, 3, 4]]))
    torch.testing.assert_close(packed.tokens[1, :2], bridge(hidden[1, [0, 4]]))
    assert spans == {"image": [[(1, 2), (3, 4)], [(0, 1)]]}
    assert provenance == {"parent_checkpoint": "fixture", "source": {"layer": "final"}}


def test_prepare_condition_exposes_normalized_contract_without_targets_or_native_payload():
    decoder = ImageDecoder(8, backend=FixtureBackend(), **explicit_route())
    cond = condition(
        native_context={"input_images": ["source-image"]},
        output_spec={"num_inference_steps": 2},
        provenance={"parent_checkpoint": "fixture"},
    )
    context = decoder.prepare_condition(cond)
    assert isinstance(context, DecoderContext)
    connected, mask = decoder.connect(cond)
    torch.testing.assert_close(context.tokens, connected, rtol=0, atol=0)
    torch.testing.assert_close(context.attention_mask, mask, rtol=0, atol=0)
    assert context.provenance["parent_checkpoint"] == "fixture"
    assert not hasattr(context, "targets")
    assert not hasattr(context, "native_context")
    expected = {
        "schema_version": 1,
        "readout": {"type": "select", "layers": "final", "positions": "all_valid"},
        "bridge": {"type": "layernorm_linear", "input_dim": 8, "output_dim": 5},
        "layout": {"type": "right_padded", "preserves_valid_order": True},
        "generator": {"type": "omnigen2", "conditioning_dim": 5},
    }
    assert decoder.conditioning_contract() == expected
    assert ImageDecoder(8, backend=FixtureBackend()).conditioning_contract() == expected


def test_nested_generator_preserves_lazy_pinned_backend(monkeypatch):
    monkeypatch.setattr(
        OmniGen2Backend, "_load", lambda *a, **k: pytest.fail("attempted checkpoint loading")
    )
    generator = {
        "type": "omnigen2",
        "model_id": "/fixture/omnigen2",
        "revision": DEFAULT_REVISION,
        "local_files_only": True,
        "conditioning_dim": 5,
    }
    decoder = ImageDecoder(8, generator=generator)
    assert decoder.backend._pipeline is None
    assert decoder.backend.model_id == "/fixture/omnigen2"
    assert decoder.backend.revision == DEFAULT_REVISION
    assert decoder.backend.local_files_only is True
    assert decoder.connector[1].out_features == 5
    assert generator["conditioning_dim"] == 5


def test_checked_in_connector_config_preserves_parent_and_instantiates_lazy_omnigen2(monkeypatch):
    monkeypatch.setattr(
        OmniGen2Backend, "_load", lambda *a, **k: pytest.fail("attempted checkpoint loading")
    )
    directory = Path(__file__).resolve().parents[1] / "src/conf/image_generation"
    legacy_data = json.loads((directory / "qwen3_1_7b_prism_harness_omnigen2.json").read_text())
    explicit_data = json.loads(
        (directory / "qwen3_1_7b_prism_harness_omnigen2_connectors.json").read_text()
    )
    assert {k: v for k, v in explicit_data.items() if k != "decoder_configs"} == {
        k: v for k, v in legacy_data.items() if k != "decoder_configs"
    }
    generator = explicit_data["decoder_configs"]["image"]["generator"]
    assert generator == {"type": "omnigen2", **legacy_data["decoder_configs"]["image"]}
    configured = ModelConfig(**explicit_data)
    decoder = ImageDecoder(d_model=configured.d_model, **configured.decoder_configs["image"])
    legacy = ImageDecoder(d_model=legacy_data["d_model"], **legacy_data["decoder_configs"]["image"])
    assert decoder.conditioning_contract() == legacy.conditioning_contract()
    assert decoder.conditioning_dim == configured.d_model == 2048
    assert decoder.backend._pipeline is None
    assert decoder.backend.revision == DEFAULT_REVISION
    assert decoder.backend.local_files_only is True
    assert decoder.backend.model_id == legacy.backend.model_id
    assert set(decoder.state_dict()) == set(legacy.state_dict())


@pytest.mark.parametrize("nested_generator", [False, True])
def test_training_bundle_records_configured_generator_identity_offline(
    tmp_path, monkeypatch, nested_generator
):
    from src.decoders import loading

    paths = {}
    for name in ("backbone", "image_encoder", "tokenizer", "processor", "omnigen2"):
        paths[name] = tmp_path / name
        paths[name].mkdir()
        (paths[name] / "config.json").write_text(json.dumps({"fixture": name}))
    reference = {
        "model_id": str(paths["omnigen2"]),
        "revision": "1" * 40,
        "local_files_only": True,
        "conditioning_dim": 5,
    }
    image_config = (
        {**explicit_route(), "generator": {"type": "omnigen2", **reference}}
        if nested_generator
        else reference
    )
    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(
            {
                "llm_backbone_id": str(paths["backbone"]),
                "image_encoder_id": str(paths["image_encoder"]),
                "d_model": 8,
                "modalities": ["text", "image"],
                "output_decoders": ["text", "image"],
                "decoder_configs": {"image": image_config},
            }
        )
    )
    tokenizer = SimpleNamespace(pad_token_id=0)
    processor = SimpleNamespace()
    frontend_calls = []

    def load_frontend(kind, path, *, local_files_only):
        frontend_calls.append((kind, path, local_files_only))
        return tokenizer if kind == "tokenizer" else processor

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda path, **kwargs: load_frontend("tokenizer", path, **kwargs)
            ),
            AutoImageProcessor=SimpleNamespace(
                from_pretrained=lambda path, **kwargs: load_frontend("processor", path, **kwargs)
            ),
        ),
    )
    resolved_paths = []

    def local_snapshot(identifier):
        resolved_paths.append(identifier)
        assert Path(identifier).is_dir()
        return str(Path(identifier).resolve())

    monkeypatch.setattr(loading, "_local_snapshot", local_snapshot)

    class ParentFixture(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.backbone = nn.Linear(8, 8, bias=False)
            self.decoders = nn.ModuleDict(
                {"image": ImageDecoder(config.d_model, **config.decoder_configs["image"])}
            )

    monkeypatch.setitem(sys.modules, "src.model", SimpleNamespace(UnifiedTransformer=ParentFixture))
    monkeypatch.setattr(
        OmniGen2Backend, "_load", lambda *a, **k: pytest.fail("attempted checkpoint loading")
    )
    checkpoint_path = tmp_path / "parent.pt"
    saved_weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torch.save({"backbone.weight": saved_weight}, checkpoint_path)
    bundle = loading.load_image_training_bundle(
        config_path, checkpoint_path, paths["tokenizer"], paths["processor"]
    )
    provenance = bundle["provenance"]
    assert provenance["reference_model_id"] == str(paths["omnigen2"])
    assert provenance["reference_revision"] == "1" * 40
    assert provenance["parent_checkpoint_sha256"] == loading.file_sha256(checkpoint_path)
    assert provenance["model_config_sha256"] == loading.file_sha256(config_path)
    assert provenance["restoration"]["strict_parent"] is True
    assert provenance["restoration"]["loaded_key_count"] == 1
    assert provenance["new_connector_keys"] == [
        "decoders.image.connector.0.bias",
        "decoders.image.connector.0.weight",
        "decoders.image.connector.1.bias",
        "decoders.image.connector.1.weight",
    ]
    torch.testing.assert_close(bundle["model"].backbone.weight, saved_weight, rtol=0, atol=0)
    backend = bundle["model"].decoders["image"].backend
    assert backend._pipeline is None
    assert backend.model_id == provenance["reference_model_id"]
    assert backend.revision == provenance["reference_revision"]
    assert frontend_calls == [
        ("tokenizer", str(paths["tokenizer"]), True),
        ("processor", str(paths["processor"]), True),
    ]
    assert resolved_paths == [
        str(paths["backbone"]),
        str(paths["image_encoder"]),
        str(paths["tokenizer"]),
        str(paths["processor"]),
    ]


@pytest.mark.parametrize(
    "config",
    [
        {"type": "query_transformer"},
        {"type": "perceiver"},
        {"type": "select", "layers": "middle"},
        {"type": "select", "positions": "text_only"},
        {"type": "select", "num_queries": 128},
        {"type": "select", "unknown_option": True},
    ],
)
def test_unimplemented_readouts_fail_explicitly(config):
    with pytest.raises(ValueError):
        build_readout(config, d_model=8)


@pytest.mark.parametrize(
    "config",
    [
        {"type": "mlp"},
        {"type": "layernorm_linear", "output_dim": 6},
        {"type": "layernorm_linear", "output_dim": 0},
        {"type": "layernorm_linear", "input_dim": 8},
        {"type": "layernorm_linear", "unknown_option": True},
    ],
)
def test_invalid_bridge_configuration_fails_explicitly(config):
    with pytest.raises(ValueError):
        build_bridge(config, input_dim=8, output_dim=5)


@pytest.mark.parametrize("config", ["select", [], 5])
def test_malformed_component_configuration_is_not_silently_ignored(config):
    with pytest.raises(TypeError):
        build_readout(config, d_model=8)
    with pytest.raises(TypeError):
        build_bridge(config, input_dim=8, output_dim=5)
    with pytest.raises(TypeError):
        ImageDecoder(8, generator=config)


@pytest.mark.parametrize(
    "flat",
    [
        {"model_id": "OmniGen2/OmniGen2"},
        {"revision": DEFAULT_REVISION},
        {"local_files_only": True},
        {"conditioning_dim": 5},
    ],
)
def test_nested_and_explicit_flat_generator_options_cannot_mix(flat):
    with pytest.raises(ValueError):
        ImageDecoder(8, backend=FixtureBackend(), generator={"type": "omnigen2"}, **flat)


@pytest.mark.parametrize(
    "config",
    [{"type": "another_generator"}, {"type": "omnigen2", "unknown_option": True}],
)
def test_unknown_generator_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        ImageDecoder(8, backend=FixtureBackend(), generator=config)


def test_injected_backend_still_enforces_conditioning_width():
    with pytest.raises(ValueError):
        ImageDecoder(
            8,
            backend=FixtureBackend(),
            generator={"type": "omnigen2", "conditioning_dim": 6},
        )


@pytest.mark.parametrize(
    "hidden,mask",
    [
        (torch.zeros(2, 8), torch.ones(2)),
        (torch.zeros(2, 3, 8), torch.ones(2, 2)),
        (torch.zeros(2, 3, 8), torch.tensor([[1, 0, 0], [0, 0, 0]])),
        (torch.zeros(2, 3, 8), torch.tensor([[1, 2, 0], [1, 0, 0]])),
    ],
)
def test_backbone_contract_rejects_invalid_shapes_or_masks(hidden, mask):
    with pytest.raises(ValueError):
        BackboneFeatures(hidden, mask)

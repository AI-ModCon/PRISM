"""Offline contracts for pooled conditioning and its parameter-free bridge."""

import pytest
import torch
from src.connectors import (
    BackboneFeatures,
    ConditioningBridge,
    DecoderContext,
    FinalStateReadout,
    IdentityBridge,
    LayerNormLinearBridge,
    PooledStateReadout,
    Readout,
    ReadoutResult,
    build_bridge,
    build_readout,
)
from torch import nn

pytestmark = pytest.mark.unit


def features_with_padding():
    hidden = torch.tensor(
        [
            [[99.0, 99.0], [1.0, 2.0], [99.0, 99.0], [3.0, 4.0], [99.0, 99.0]],
            [[5.0, 6.0], [99.0, 99.0], [7.0, 8.0], [99.0, 99.0], [9.0, 10.0]],
        ]
    )
    mask = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
    hidden.masked_fill_(~mask.bool().unsqueeze(-1), float("nan"))
    hidden.requires_grad_()
    return BackboneFeatures(
        hidden,
        mask,
        modality_spans={"time_series": [[(1, 2)], [(0, 1)]]},
        provenance={"checkpoint": "fixture"},
    )


@pytest.mark.parametrize("pool", ["last", "mean"])
def test_pooling_preserves_valid_evidence_and_gradients_with_interior_padding(pool):
    features = features_with_padding()
    readout = build_readout({"type": "pool", "layers": "final", "pool": pool}, d_model=2)
    assert isinstance(readout, Readout)
    assert isinstance(readout, PooledStateReadout)
    selected = readout(features)
    assert selected.tokens.shape == (2, 1, 2)
    assert selected.attention_mask.tolist() == [[True], [True]]
    if pool == "last":
        torch.testing.assert_close(selected.tokens, torch.tensor([[[3.0, 4.0]], [[9.0, 10.0]]]))
        assert selected.source_positions.tolist() == [[3], [4]]
    else:
        torch.testing.assert_close(selected.tokens, torch.tensor([[[2.0, 3.0]], [[7.0, 8.0]]]))
        assert selected.source_positions is None
    assert selected.source_modality_spans == features.modality_spans
    assert selected.provenance == features.provenance
    assert selected.source_modality_spans is not features.modality_spans
    assert selected.provenance is not features.provenance

    selected.tokens.sum().backward()
    expected_grad = torch.zeros_like(features.hidden_states)
    if pool == "last":
        expected_grad[0, 3] = 1
        expected_grad[1, 4] = 1
    else:
        expected_grad[0, [1, 3]] = 0.5
        expected_grad[1, [0, 2, 4]] = 1 / 3
    torch.testing.assert_close(features.hidden_states.grad, expected_grad)
    assert torch.isfinite(features.hidden_states.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pool", ["last", "mean"])
def test_single_token_qwen_width_is_preserved(pool, dtype):
    hidden = torch.randn(2, 1, 1024).to(dtype).requires_grad_()
    features = BackboneFeatures(hidden, torch.ones(2, 1, dtype=torch.bool))
    context = IdentityBridge(1024, 1024).connect(PooledStateReadout(1024, pool)(features))
    assert isinstance(context, DecoderContext)
    assert context.tokens.dtype == dtype
    torch.testing.assert_close(context.tokens, hidden, rtol=0, atol=0)
    context.tokens.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)


def test_bfloat16_mean_retains_integer_count_at_long_sequence_lengths():
    # 257 is not representable in BF16. The previous masked_pool divides the
    # reduced BF16 sum by an integer count, rather than a rounded BF16 count.
    hidden = torch.ones(1, 260, 2, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(1, 260, dtype=torch.bool)
    mask[:, -3:] = False
    actual = PooledStateReadout(2, "mean")(BackboneFeatures(hidden, mask)).tokens[:, 0]
    expected = hidden.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0, 0].item() == 0.99609375
    actual.sum().backward()
    actual_grad = hidden.grad.clone()
    hidden.grad = None
    expected.sum().backward()
    torch.testing.assert_close(actual_grad, hidden.grad, rtol=0, atol=0)


def test_identity_bridge_keeps_layout_and_sanitizes_only_padding():
    features = features_with_padding()
    readout = FinalStateReadout(2)(features)
    bridge = build_bridge({"type": "identity", "output_dim": 2}, input_dim=2, output_dim=2)
    assert isinstance(bridge, ConditioningBridge)
    assert isinstance(bridge, nn.Identity)
    context = bridge.connect(readout)
    assert context.tokens.dtype == features.hidden_states.dtype
    assert torch.isfinite(context.tokens).all()
    torch.testing.assert_close(
        context.tokens[features.attention_mask.bool()],
        features.hidden_states[features.attention_mask.bool()],
    )
    assert context.tokens[~features.attention_mask.bool()].eq(0).all()
    torch.testing.assert_close(context.attention_mask, readout.attention_mask)
    torch.testing.assert_close(context.source_positions, readout.source_positions)
    assert context.source_modality_spans == readout.source_modality_spans
    assert context.source_modality_spans is not readout.source_modality_spans
    assert context.provenance == readout.provenance
    assert context.provenance is not readout.provenance
    context.tokens.sum().backward()
    expected = features.attention_mask.bool().unsqueeze(-1).expand_as(features.hidden_states)
    torch.testing.assert_close(features.hidden_states.grad, expected.float())


def test_parameter_free_primitives_preserve_rng_and_generator_state_keys():
    torch.manual_seed(77)
    rng_before = torch.random.get_rng_state().clone()
    readout = build_readout({"type": "pool"}, d_model=1024)
    bridge = build_bridge({"type": "identity"}, input_dim=1024, output_dim=1024)
    assert readout.pool == "last"
    assert readout.state_dict() == {}
    assert bridge.state_dict() == {}
    assert list(readout.parameters()) == list(bridge.parameters()) == []
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    actual = nn.Linear(1024, 96 * 3)
    torch.random.set_rng_state(rng_before)
    expected = nn.Linear(1024, 96 * 3)
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)


def test_omitted_factory_configs_retain_existing_parameterized_route():
    readout = build_readout(None, d_model=2)
    assert isinstance(readout, FinalStateReadout)
    torch.manual_seed(53)
    bridge = build_bridge(None, input_dim=2, output_dim=3)
    assert isinstance(bridge, LayerNormLinearBridge)
    torch.manual_seed(53)
    expected = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 3))
    assert set(bridge.state_dict()) == {"0.weight", "0.bias", "1.weight", "1.bias"}
    for name, value in bridge.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "config",
    [
        {"type": "select", "pool": "last"},
        {"type": "pool", "positions": "all_valid"},
        {"type": "pool", "pool": "first"},
        {"type": "pool", "pool": None},
        {"type": "pool", "layers": "middle"},
        {"type": "pool", "num_queries": 5},
    ],
)
def test_type_specific_readout_fields_fail_explicitly(config):
    with pytest.raises(ValueError):
        build_readout(config, d_model=2)


@pytest.mark.parametrize(
    "config,input_dim,output_dim",
    [
        ({"type": "identity"}, 2, 3),
        ({"type": "identity", "output_dim": 3}, 2, 2),
        ({"type": "identity", "output_dim": True}, 2, 2),
        ({"type": "identity", "input_dim": 2}, 2, 2),
        ({"type": "identity"}, 0, 2),
        ({"type": "identity"}, 2, 0),
    ],
)
def test_identity_bridge_dimension_and_unknown_field_guards(config, input_dim, output_dim):
    with pytest.raises(ValueError):
        build_bridge(config, input_dim=input_dim, output_dim=output_dim)


def test_width_mismatch_is_rejected_before_pooling_or_connecting():
    features = features_with_padding()
    with pytest.raises(ValueError, match="d_model"):
        PooledStateReadout(3)(features)
    with pytest.raises(ValueError, match="input width"):
        IdentityBridge(3, 3).connect(ReadoutResult(features.hidden_states, features.attention_mask))


def test_empty_evidence_is_not_admitted_for_pooling():
    with pytest.raises(ValueError, match="at least one valid token"):
        BackboneFeatures(torch.randn(2, 4, 2), torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]]))

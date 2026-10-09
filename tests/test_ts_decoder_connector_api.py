"""Offline contracts for the simple time-series output connector.

An independent copy of the former pooling/linear/pinball computation checks
numerical and checkpoint compatibility. These tests do not validate forecasting
accuracy, pretrained checkpoints, or the external Intern-S2 input encoder.
"""

from copy import deepcopy

import pytest
import torch
from src.connectors import ConditioningBridge, DecoderContext, Readout
from src.decoders.image import ImageDecoder
from src.decoders.time_series import TimeSeriesDecoder
from src.decoders.types import DecoderCondition
from torch import nn

from test_image_decoder import FixtureBackend

pytestmark = pytest.mark.unit


def explicit_route(pool="last", horizon=3, num_vars=2):
    return {
        "readout": {"type": "pool", "layers": "final", "pool": pool},
        "bridge": {"type": "identity"},
        "generator": {
            "type": "linear_quantile",
            "horizon": horizon,
            "num_vars": num_vars,
            "quantiles": [0.1, 0.5, 0.9],
        },
    }


def old_pool(hidden, mask, pool):
    """Former pooling formula, independent of new readout/bridge code."""
    if hidden.ndim == 2:
        return hidden
    if mask is None:
        return hidden[:, -1] if pool == "last" else hidden.mean(1)
    valid = mask.bool()
    if pool == "last":
        indices = torch.arange(valid.shape[1]).expand_as(valid)
        last = indices.masked_fill(~valid, -1).max(1).values
        return hidden[torch.arange(hidden.shape[0]), last]
    return hidden.masked_fill(~valid[..., None], 0).sum(1) / valid.sum(1, keepdim=True)


def old_prediction_and_loss(head, hidden, mask, pool, target):
    prediction = head(old_pool(hidden, mask, pool).to(head.weight.dtype)).reshape(2, 3, 2, 3)
    levels = torch.tensor([0.1, 0.5, 0.9], dtype=prediction.dtype)
    error = target[..., None] - prediction
    return prediction, torch.maximum(levels * error, (levels - 1) * error).mean()


def condition(hidden, mask):
    return DecoderCondition(
        hidden,
        mask,
        modality_spans={"time_series": [[(1, 2)], [(0, 1)]]},
        provenance={"parent_checkpoint": "offline-fixture", "source": {"layer": "final"}},
        native_context={"context_series": "not-a-connector-token"},
        output_spec={"horizon": 3},
    )


@pytest.mark.parametrize("pool", ["last", "mean"])
def test_nested_path_preserves_legacy_initialization_rng_and_checkpoint_keys(pool):
    torch.manual_seed(61)
    original_head = nn.Linear(8, 3 * 2 * 3)
    original_rng = torch.random.get_rng_state()
    torch.manual_seed(61)
    legacy = TimeSeriesDecoder(8, horizon=3, num_vars=2, pool=pool)
    legacy_rng = torch.random.get_rng_state()
    torch.manual_seed(61)
    explicit = TimeSeriesDecoder(8, **explicit_route(pool))
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    assert torch.equal(legacy_rng, original_rng)
    assert set(explicit.state_dict()) == set(legacy.state_dict()) == {"head.weight", "head.bias"}
    assert isinstance(explicit.readout, Readout)
    assert isinstance(explicit.connector, ConditioningBridge)
    assert list(explicit.readout.parameters()) == list(explicit.connector.parameters()) == []
    for name, tensor in original_head.state_dict().items():
        torch.testing.assert_close(explicit.state_dict()[f"head.{name}"], tensor, rtol=0, atol=0)
        torch.testing.assert_close(legacy.state_dict()[f"head.{name}"], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("pool", ["last", "mean"])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_explicit_forward_loss_and_gradients_match_old_formula_with_padded_nans(pool, source_dtype):
    decoder = TimeSeriesDecoder(8, **explicit_route(pool))
    oracle = nn.Linear(8, 18)
    oracle.load_state_dict(decoder.head.state_dict(), strict=True)
    hidden = torch.randn(2, 5, 8, dtype=source_dtype)
    mask = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
    hidden.masked_fill_(~mask.bool()[..., None], float("nan"))
    actual_hidden = hidden.clone().requires_grad_()
    oracle_hidden = hidden.clone().requires_grad_()
    target = torch.randn(2, 3, 2)
    actual, actual_loss = decoder.forward_condition(condition(actual_hidden, mask), targets=target)
    expected, expected_loss = old_prediction_and_loss(oracle, oracle_hidden, mask, pool, target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual_hidden.grad, oracle_hidden.grad, rtol=0, atol=0)
    assert torch.isfinite(actual_hidden.grad).all()
    assert actual_hidden.grad[~mask.bool()].eq(0).all()
    for name, parameter in decoder.head.named_parameters():
        expected_parameter = dict(oracle.named_parameters())[name]
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("pool", ["last", "mean"])
def test_prepared_context_is_single_token_and_keeps_source_provenance(pool):
    decoder = TimeSeriesDecoder(8, **explicit_route(pool))
    hidden = torch.arange(80, dtype=torch.float32).reshape(2, 5, 8)
    mask = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
    cond = condition(hidden, mask)
    context = decoder.prepare_condition(cond)
    assert isinstance(context, DecoderContext)
    assert context.tokens.shape == (2, 1, 8)
    assert context.attention_mask.tolist() == [[True], [True]]
    torch.testing.assert_close(context.tokens[:, 0], old_pool(hidden, mask, pool), rtol=0, atol=0)
    if pool == "last":
        assert context.source_positions.tolist() == [[3], [4]]
    else:
        assert context.source_positions is None  # Averaging has no single source position.
    assert context.source_modality_spans == cond.modality_spans
    assert context.provenance["parent_checkpoint"] == "offline-fixture"
    assert context.provenance["source"] == {"layer": "final"}
    assert context.provenance["conditioning_contract"] == decoder.conditioning_contract()
    assert not hasattr(context, "native_context")
    assert not hasattr(context, "targets")
    assert not hasattr(context, "output_spec")
    assert "conditioning_contract" not in cond.provenance


@pytest.mark.parametrize("pool", ["last", "mean"])
def test_normalized_contract_describes_legacy_and_explicit_path(pool):
    decoder = TimeSeriesDecoder(8, **explicit_route(pool))
    expected = {
        "schema_version": 1,
        "readout": {"type": "pool", "layers": "final", "pool": pool},
        "bridge": {"type": "identity", "input_dim": 8, "output_dim": 8},
        "layout": {"type": "single_token"},
        "generator": {
            "type": "linear_quantile", "conditioning_dim": 8,
            "horizon": 3, "num_vars": 2, "quantiles": [0.1, 0.5, 0.9],
        },
    }
    assert decoder.conditioning_contract() == expected
    assert TimeSeriesDecoder(8, 3, num_vars=2, pool=pool).conditioning_contract() == expected


@pytest.mark.parametrize("pool", ["last", "mean"])
@pytest.mark.parametrize("explicit_source", [False, True])
def test_strict_checkpoint_restore_in_both_directions(pool, explicit_source):
    options = explicit_route(pool)
    flat = {"horizon": 3, "num_vars": 2, "pool": pool}
    source = TimeSeriesDecoder(8, **(options if explicit_source else flat))
    destination = TimeSeriesDecoder(8, **(flat if explicit_source else options))
    restored = destination.load_state_dict(deepcopy(source.state_dict()), strict=True)
    assert restored.missing_keys == restored.unexpected_keys == []
    cond = condition(torch.randn(2, 5, 8), torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]))
    torch.testing.assert_close(source.generate_condition(cond), destination.generate_condition(cond), rtol=0, atol=0)


@pytest.mark.parametrize("pool", ["last", "mean"])
def test_tensor_and_condition_routes_keep_median_and_mask_semantics(pool):
    decoder = TimeSeriesDecoder(8, **explicit_route(pool))
    hidden = torch.randn(2, 5, 8)
    mask = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
    target = torch.randn(2, 3, 2)
    cond = condition(hidden, mask)
    predicted, loss = decoder.forward_condition(cond, targets=target)
    legacy_predicted, legacy_loss = decoder(hidden, targets=target, attention_mask=mask)
    torch.testing.assert_close(predicted, legacy_predicted, rtol=0, atol=0)
    torch.testing.assert_close(loss, legacy_loss, rtol=0, atol=0)
    torch.testing.assert_close(decoder.generate_condition(cond), predicted[..., 1], rtol=0, atol=0)
    torch.testing.assert_close(decoder.generate(hidden, attention_mask=mask), predicted[..., 1], rtol=0, atol=0)
    assert decoder.forward_condition(cond)[1] is None
    changed_target, changed_loss = decoder.forward_condition(cond, targets=target + 50)
    torch.testing.assert_close(changed_target, predicted, rtol=0, atol=0)
    assert not torch.isclose(changed_loss, loss)
    pooled = old_pool(hidden, mask, pool)
    torch.testing.assert_close(decoder(pooled)[0], predicted, rtol=0, atol=0)
    expected_unmasked = decoder(old_pool(hidden, None, pool))[0]
    torch.testing.assert_close(decoder(hidden)[0], expected_unmasked, rtol=0, atol=0)


@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16])
def test_unmasked_half_precision_mean_keeps_legacy_reduction_and_finite_values(source_dtype):
    """A raw sum before dividing can overflow FP16 despite a finite mean."""
    decoder = TimeSeriesDecoder(8, **explicit_route("mean"))
    oracle = nn.Linear(8, 18)
    oracle.load_state_dict(decoder.head.state_dict(), strict=True)
    hidden = torch.full((2, 7, 8), 40_000.0, dtype=source_dtype)
    actual_hidden = hidden.clone().requires_grad_()
    oracle_hidden = hidden.clone().requires_grad_()
    targets = torch.zeros(2, 3, 2)
    actual, actual_loss = decoder(actual_hidden, targets=targets)
    expected, expected_loss = old_prediction_and_loss(oracle, oracle_hidden, None, "mean", targets)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual_hidden.grad, oracle_hidden.grad, rtol=0, atol=0)
    for name, parameter in decoder.head.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(oracle.named_parameters())[name].grad, rtol=0, atol=0)


@pytest.mark.parametrize("options", [
    {"generator": {"type": "diffusion", "horizon": 3}},
    {"generator": {"type": "linear_quantile", "horizon": 3, "unused": True}},
    {"generator": {"type": "linear_quantile", "horizon": 3}, "horizon": 3},
    {"generator": {"type": "linear_quantile", "horizon": 3}, "num_vars": 1},
    {"generator": {"type": "linear_quantile", "horizon": 3}, "quantiles": [0.5]},
    {"generator": {"type": "linear_quantile", "horizon": 3, "conditioning_dim": 9}},
    {"horizon": 3, "readout": {"type": "pool", "pool": "last"}, "pool": "last"},
    {"horizon": 3, "readout": {"type": "select"}},
    {"horizon": 3, "readout": {"type": "query"}},
    {"horizon": 3, "readout": {"type": "pool", "layers": "middle"}},
    {"horizon": 3, "readout": {"type": "pool", "pool": "max"}},
    {"horizon": 3, "readout": {"type": "pool", "num_queries": 1}},
    {"horizon": 3, "bridge": {"type": "layernorm_linear", "output_dim": 8}},
    {"horizon": 3, "bridge": {"type": "identity", "output_dim": 9}},
    {"horizon": 3, "bridge": {"type": "identity", "unused": True}},
])
def test_invalid_nested_contracts_fail_before_use(options):
    with pytest.raises((TypeError, ValueError)):
        TimeSeriesDecoder(8, **options)


@pytest.mark.parametrize("field", ["readout", "bridge", "generator"])
def test_nested_sections_require_mappings(field):
    options = {"horizon": 3, field: "not-a-mapping"}
    if field == "generator":
        del options["horizon"]
    with pytest.raises(TypeError):
        TimeSeriesDecoder(8, **options)


@pytest.mark.parametrize("options", [
    {"readout": {"type": "pool", "layers": "final", "pool": "last"}},
    {"bridge": {"type": "identity"}},
])
def test_omnigen2_rejects_new_pool_or_identity_routes(options):
    with pytest.raises(ValueError):
        ImageDecoder(8, backend=FixtureBackend(), **options)

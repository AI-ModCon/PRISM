from __future__ import annotations

import pytest
import torch
from src.decoders import DECODERS, TimeSeriesDecoder
from src.decoders.base import OutputDecoder

pytestmark = pytest.mark.unit


def test_history_statistics_ignore_missing_and_use_population_std():
    from src.encoders.time_series import TimeSeriesEncoder

    history = torch.tensor([[1.0, 7.0, float("nan")], [3.0, 9.0, float("nan")], [999.0, 11.0, float("nan")]], dtype=torch.bfloat16, requires_grad=True)
    observed = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
    mean, scale = TimeSeriesEncoder.forecast_statistics(history, observed)
    torch.testing.assert_close(mean, torch.tensor([2.0, 7.0, 0.0]))
    torch.testing.assert_close(scale, torch.tensor([1.0, 1.0, 1.0]))
    assert mean.dtype == scale.dtype == torch.float32
    assert not mean.requires_grad and not scale.requires_grad
    mean, scale = TimeSeriesEncoder.forecast_statistics(torch.full((4, 2), 5.0))
    torch.testing.assert_close(mean, torch.tensor([5.0, 5.0]))
    torch.testing.assert_close(scale, torch.ones(2))


def test_history_statistics_stay_finite_for_extreme_values_and_nonfinite_gaps():
    from src.encoders.time_series import TimeSeriesEncoder

    history = torch.tensor(
        [[3.0e38, float("nan"), float("inf")],
         [3.0e38, float("inf"), float("-inf")],
         [3.0e38, float("nan"), 1.0]],
        dtype=torch.float32,
    )

    mean, scale = TimeSeriesEncoder.forecast_statistics(history)

    assert torch.isfinite(mean).all()
    assert torch.isfinite(scale).all()
    torch.testing.assert_close(mean, torch.tensor([3.0e38, 0.0, 1.0]))
    torch.testing.assert_close(scale, torch.ones(3))


def test_history_statistics_use_fallbacks_for_empty_history():
    from src.encoders.time_series import TimeSeriesEncoder

    mean, scale = TimeSeriesEncoder.forecast_statistics(torch.empty(0, 2))

    torch.testing.assert_close(mean, torch.zeros(2))
    torch.testing.assert_close(scale, torch.ones(2))


def test_forecast_head_scores_standardized_targets_and_generates_original_units():
    from src.decoders.time_series import TimeSeriesDecoder

    decoder = TimeSeriesDecoder(4, horizon=2, num_vars=1)
    with torch.no_grad():
        decoder.head.weight.zero_()
        decoder.head.bias.zero_()
    hidden = torch.zeros(1, 4)
    mean = torch.tensor([[10.0]])
    scale = torch.tensor([[2.0]])
    targets = torch.tensor([[[12.0], [14.0]]])
    prediction, loss = decoder(hidden, targets=targets, history_mean=mean, history_scale=scale)
    torch.testing.assert_close(prediction, torch.full_like(prediction, 10.0))
    torch.testing.assert_close(loss, torch.tensor(0.75))
    generated = decoder.generate(hidden, history_mean=mean, history_scale=scale)
    torch.testing.assert_close(generated, torch.full((1, 2, 1), 10.0))
    masked_targets = torch.tensor([[[12.0], [999.0]]])
    _, masked_loss = decoder(hidden, targets=masked_targets, target_mask=torch.tensor([[[True], [False]]]), history_mean=mean, history_scale=scale)
    torch.testing.assert_close(masked_loss, torch.tensor(0.5))


def test_masked_extreme_forecast_target_cannot_overflow_standardization():
    decoder = TimeSeriesDecoder(4, horizon=2, num_vars=1)
    with torch.no_grad():
        decoder.head.weight.zero_()
        decoder.head.bias.fill_(-1.0e38)
    hidden = torch.zeros(1, 4)
    targets = torch.tensor([[[-2.0e38], [2.0e38]]])
    target_mask = torch.tensor([[[True], [False]]])

    _, loss = decoder(
        hidden,
        targets=targets,
        target_mask=target_mask,
        history_mean=torch.tensor([[-2.0e38]]),
        history_scale=torch.tensor([[0.1]]),
    )

    assert torch.isfinite(loss)


def _forecast_wrapper(multi_ds, history_cap=None):
    from types import SimpleNamespace

    from src.data.multi_webdataset import ModalityAwareWebDatasetWrapper

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    wrapper = ModalityAwareWebDatasetWrapper.__new__(ModalityAwareWebDatasetWrapper)
    wrapper._primary_modality = "time_series"
    wrapper.tokenizer = Tokenizer()
    wrapper.max_length = 16
    wrapper.multi_ds = multi_ds
    if history_cap is not None:
        wrapper._ts_history_cap = history_cap
    return wrapper


def test_forecast_loader_stats_use_capped_most_recent_history():
    # Values 1000/999 and the unobserved 5.0 fall outside the 3-step cap.
    history = torch.tensor(
        [[1000.0, 1.0], [999.0, 1.0], [5.0, 0.0], [2.0, 1.0], [4.0, 1.0]]
    )
    target = torch.tensor([4.0, 6.0])
    wrapper = _forecast_wrapper(
        [{"time_series": history, "time_series_target": target, "text": "f", "metadata": {}}],
        history_cap=3,
    )

    sample = next(iter(wrapper))

    torch.testing.assert_close(sample["time_series"], history[-3:])
    torch.testing.assert_close(sample["time_series_history_mean"], torch.tensor([3.0]))
    torch.testing.assert_close(sample["time_series_history_scale"], torch.tensor([1.0]))


def test_forecast_loader_uses_task_and_history_layout_metadata():
    history = torch.tensor(
        [[1.0, 10.0, 1.0, 1.0], [3.0, 20.0, 1.0, 0.0], [999.0, 999.0, 0.0, 0.0]]
    )
    wrapper = _forecast_wrapper(
        [{
            "time_series": history,
            "time_series_target": torch.tensor([7.0, 8.0]),
            "text": "forecast",
            "metadata": {
                "task_type": "forecasting",
                "history_value_channels": 2,
                "history_has_observation_mask": True,
            },
        }]
    )

    sample = next(iter(wrapper))

    assert sample["numerical_task"] == "forecasting"
    torch.testing.assert_close(sample["time_series_history_mean"], torch.tensor([2.0, 10.0]))
    torch.testing.assert_close(sample["time_series_history_scale"], torch.tensor([1.0, 1.0]))
    decoder = TimeSeriesDecoder(4, horizon=2, num_vars=10)
    prediction, loss = decoder(
        torch.zeros(1, 4),
        targets=sample["time_series_target"].reshape(1, 2, 1),
    )
    assert prediction.shape == (1, 2, 10, 3)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("length", [2, 3])
def test_forecast_loader_leaves_history_at_or_below_cap_untouched(length):
    history = torch.arange(length, dtype=torch.float32)
    wrapper = _forecast_wrapper(
        [{"time_series": history, "text": "f", "metadata": {}}], history_cap=3
    )

    sample = next(iter(wrapper))

    torch.testing.assert_close(sample["time_series"], history)


@pytest.mark.parametrize(
    ("projector", "expected"), [("intern_s2_397b", 300_000), ("timeomni", None), (None, None)]
)
def test_forecast_loader_history_cap_matches_intern_s2_397b_encoder(
    monkeypatch, projector, expected
):
    from types import SimpleNamespace

    import src.data.multi_webdataset as mwd
    from src.encoders.time_series import TimeSeriesEncoder

    class FakeMultiWebDataset:
        def __init__(self, **kwargs):
            pass

        def get_stats(self):
            return {"num_datasets": 0, "total_samples": 0}

    monkeypatch.setattr(mwd, "MultiWebDataset", FakeMultiWebDataset)
    monkeypatch.setattr(mwd, "build_image_transform", lambda model_config: None)
    model_config = None if projector is None else SimpleNamespace(ts_projector=projector)

    wrapper = mwd.ModalityAwareWebDatasetWrapper(
        tokenizer=None, modalities=["time_series"], config={}, model_config=model_config
    )

    assert wrapper._ts_history_cap == expected
    if expected is not None:
        assert expected == TimeSeriesEncoder.INTERN_S2_397B_MAX_HISTORY


def test_forecast_loader_history_cap_follows_model_config_override(monkeypatch):
    from types import SimpleNamespace

    import src.data.multi_webdataset as mwd

    class FakeMultiWebDataset:
        def __init__(self, **kwargs):
            pass

        def get_stats(self):
            return {"num_datasets": 0, "total_samples": 0}

    monkeypatch.setattr(mwd, "MultiWebDataset", FakeMultiWebDataset)
    monkeypatch.setattr(mwd, "build_image_transform", lambda model_config: None)
    model_config = SimpleNamespace(ts_projector="intern_s2_397b", ts_max_history=512)

    wrapper = mwd.ModalityAwareWebDatasetWrapper(
        tokenizer=None, modalities=["time_series"], config={}, model_config=model_config
    )

    assert wrapper._ts_history_cap == 512


def test_ts_max_history_defaults_to_encoder_cap():
    from src.config import ModelConfig
    from src.encoders.time_series import TimeSeriesEncoder

    assert ModelConfig().ts_max_history == TimeSeriesEncoder.INTERN_S2_397B_MAX_HISTORY


def test_forecast_loader_collator_and_model_pass_detached_statistics():
    from types import SimpleNamespace

    from src.data.collate import MultimodalCollator
    from src.data.multi_webdataset import ModalityAwareWebDatasetWrapper
    from src.model import UnifiedTransformer

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    history = torch.tensor([[1.0, 1.0], [3.0, 1.0], [999.0, 0.0]])
    wrapper = ModalityAwareWebDatasetWrapper.__new__(ModalityAwareWebDatasetWrapper)
    wrapper._primary_modality = "time_series"
    wrapper.tokenizer = Tokenizer()
    wrapper.max_length = 16
    wrapper.multi_ds = [{"time_series": history, "time_series_target": torch.tensor([4.0, 6.0]), "text": "forecast", "metadata": {}}]
    sample = next(iter(wrapper))
    torch.testing.assert_close(sample["time_series"], history)
    torch.testing.assert_close(sample["time_series_history_mean"], torch.tensor([2.0]))
    torch.testing.assert_close(sample["time_series_history_scale"], torch.tensor([1.0]))
    batch = MultimodalCollator(Tokenizer(), passthrough_time_series=True)([sample])
    model = SimpleNamespace(decoders={"time_series": TimeSeriesDecoder(4, horizon=2, num_vars=2)})
    stats = UnifiedTransformer._forecast_statistics_kwargs(model, batch)
    torch.testing.assert_close(stats["history_mean"], torch.tensor([[2.0, 0.0]]))
    torch.testing.assert_close(stats["history_scale"], torch.tensor([[1.0, 1.0]]))
    decoder = model.decoders["time_series"]
    _, loss = decoder(torch.zeros(1, 4), targets=batch["time_series_target"], target_mask=batch["time_series_target_mask"], **stats)
    assert torch.isfinite(loss)
    loss.backward()
    assert decoder.head.weight.grad is not None
    wrapper.multi_ds = [{"time_series": history, "text": "forecast", "metadata": {"history_value_channels": 1}}]
    inference_sample = next(iter(wrapper))
    assert "time_series_history_mean" not in inference_sample
    assert "time_series_history_scale" not in inference_sample
    from src.decoders.types import DecoderCondition

    condition = DecoderCondition(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool))
    with torch.no_grad():
        decoder.head.weight.zero_()
        decoder.head.bias.zero_()
    typed_forecast = decoder.generate_condition(condition, **stats)
    torch.testing.assert_close(typed_forecast, torch.tensor([[[2.0, 0.0], [2.0, 0.0]]]))

"""Phase 2 tests: TimeSeriesDecoder.

Locks the time-series decoder contract:
- forward: hidden states -> quantile forecast of the right shape/dtype,
- pinball loss is finite, asymmetric across quantiles, and decreases on a
  1-step overfit (the "correctness, not just runs" check),
- generate() returns a point-forecast tensor (median quantile), not text,
- registry exposes the decoder.

Pure unit tests — no backbone, no network.
"""

def test_registered():
    assert DECODERS["time_series"] is TimeSeriesDecoder
    assert issubclass(TimeSeriesDecoder, OutputDecoder)
    assert TimeSeriesDecoder.output_kind == "tensor"
    assert TimeSeriesDecoder.loss_kind == "pinball"


def test_forward_shapes_from_sequence_hidden():
    dec = TimeSeriesDecoder(d_model=16, horizon=8, num_vars=2, quantiles=(0.1, 0.5, 0.9))
    hidden = torch.randn(4, 5, 16)  # (B, T, d_model)
    pred, loss = dec(hidden, targets=None)
    assert pred.shape == (4, 8, 2, 3)  # (B, H, V, Q)
    assert loss is None


def test_forward_accepts_pooled_hidden():
    dec = TimeSeriesDecoder(d_model=16, horizon=8, num_vars=2)
    hidden = torch.randn(4, 16)  # already pooled (B, d_model)
    pred, _ = dec(hidden, targets=None)
    assert pred.shape == (4, 8, 2, len(dec.quantiles))


def test_loss_finite_and_shape_checked():
    dec = TimeSeriesDecoder(d_model=16, horizon=8, num_vars=2)
    hidden = torch.randn(4, 5, 16)
    target = torch.randn(4, 8, 2)  # (B, H, V)
    pred, loss = dec(hidden, targets=target)
    assert loss is not None and torch.isfinite(loss)
    # wrong target shape -> clear error
    with pytest.raises(RuntimeError, match="target shape"):
        dec(hidden, targets=torch.randn(4, 8, 3))


def test_masked_loss_pads_short_targets_and_ignores_invalid_cells():
    dec = TimeSeriesDecoder(d_model=4, horizon=5, num_vars=3, quantiles=(0.1, 0.5, 0.9))
    predictions = torch.zeros(2, 5, 3, 3, requires_grad=True)
    target = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    target_mask = torch.tensor([[[True], [False]], [[True], [True]]])

    loss = dec.pinball_loss(predictions, target, target_mask=target_mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.count_nonzero(predictions.grad[:, 2:, :, :]) == 0
    assert torch.count_nonzero(predictions.grad[:, :2, 1:, :]) == 0
    assert torch.count_nonzero(predictions.grad[0, 1, 0, :]) == 0
    assert torch.count_nonzero(predictions.grad[1, :2, 0, :]) > 0


def test_all_masked_target_has_zero_loss():
    dec = TimeSeriesDecoder(d_model=4, horizon=2, num_vars=1)
    predictions = torch.randn(1, 2, 1, 3, requires_grad=True)
    target = torch.zeros(1, 1, 1)
    target_mask = torch.zeros_like(target, dtype=torch.bool)

    loss = dec.pinball_loss(predictions, target, target_mask=target_mask)

    assert loss.item() == 0.0


def test_aux_decoder_loss_ignores_masked_target_values():
    from types import SimpleNamespace

    from src.model import UnifiedTransformer

    decoder = TimeSeriesDecoder(d_model=8, horizon=4, num_vars=2)
    model = SimpleNamespace(decoders={"time_series": decoder})
    hidden = torch.randn(1, 3, 8)
    target = torch.zeros(1, 2, 1)
    target_mask = torch.tensor([[[True], [False]]])
    changed_target = target.clone()
    changed_target[:, 1, :] = 10_000

    baseline = UnifiedTransformer._aux_decoder_loss(
        model,
        hidden,
        {"time_series": target},
        target_masks={"time_series": target_mask},
    )
    changed = UnifiedTransformer._aux_decoder_loss(
        model,
        hidden,
        {"time_series": changed_target},
        target_masks={"time_series": target_mask},
    )

    torch.testing.assert_close(baseline, changed)


def test_pinball_loss_is_asymmetric():
    # Pinball loss must penalize under- and over-prediction differently per
    # quantile. With a single high quantile (0.9), under-prediction (target >
    # pred) should cost ~9x more than the same-magnitude over-prediction.
    dec = TimeSeriesDecoder(d_model=4, horizon=1, num_vars=1, quantiles=(0.9,))
    pred_q = torch.zeros(1, 1, 1, 1)
    under = dec.pinball_loss(pred_q, torch.ones(1, 1, 1))   # target above pred
    over = dec.pinball_loss(pred_q, -torch.ones(1, 1, 1))   # target below pred
    assert under > over
    assert torch.allclose(under / over, torch.tensor(9.0), atol=1e-4)


def test_generate_returns_point_forecast_tensor():
    dec = TimeSeriesDecoder(d_model=16, horizon=8, num_vars=2, quantiles=(0.1, 0.5, 0.9))
    hidden = torch.randn(4, 5, 16)
    out = dec.generate(hidden)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 8, 2)  # (B, H, V), quantile dim collapsed to median


def test_overfit_one_batch_reduces_loss():
    torch.manual_seed(0)
    dec = TimeSeriesDecoder(d_model=16, horizon=4, num_vars=1)
    hidden = torch.randn(8, 16)
    target = torch.randn(8, 4, 1)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-2)

    _, first = dec(hidden, targets=target)
    for _ in range(50):
        opt.zero_grad()
        _, loss = dec(hidden, targets=target)
        loss.backward()
        opt.step()
    _, last = dec(hidden, targets=target)
    assert last < first * 0.5, (first.item(), last.item())


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        TimeSeriesDecoder(d_model=8, horizon=0)
    with pytest.raises(ValueError):
        TimeSeriesDecoder(d_model=8, horizon=4, num_vars=0)
    with pytest.raises(ValueError):
        TimeSeriesDecoder(d_model=8, horizon=4, quantiles=(1.5,))
    with pytest.raises(ValueError):
        TimeSeriesDecoder(d_model=8, horizon=4, pool="bogus")


# --------------------------------------------------------------------------
# Target plumbing — the generic collator stacks "<name>_target" with no
# special-casing (Phase 2 needs no collator change).
# --------------------------------------------------------------------------


def test_collator_stacks_time_series_target():
    from src.data.collate import MultimodalCollator

    class _Tok:
        pad_token_id = 0

    col = MultimodalCollator(tokenizer=_Tok())
    batch = [
        {"text": torch.tensor([1, 2, 3]), "time_series_target": torch.randn(6, 1)},
        {"text": torch.tensor([1, 2, 3]), "time_series_target": torch.randn(6, 1)},
    ]
    out = col(batch)
    assert tuple(out["time_series_target"].shape) == (2, 6, 1)


def test_collator_pads_variable_targets_and_keeps_targetless_samples_masked():
    from src.data.collate import MultimodalCollator

    class _Tok:
        pad_token_id = 0

    collator = MultimodalCollator(tokenizer=_Tok(), passthrough_time_series=True)
    batch = [
        {
            "text": torch.tensor([1, 2]),
            "time_series": torch.ones(4, 2),
            "time_series_target": torch.tensor([[5.0]]),
            "time_series_target_mask": torch.tensor([[True]]),
        },
        {"text": torch.tensor([1]), "time_series": torch.ones(3, 2)},
    ]

    out = collator(batch)

    assert isinstance(out["time_series"], list)
    assert tuple(out["time_series_target"].shape) == (2, 1, 1)
    assert tuple(out["time_series_target_mask"].shape) == (2, 1, 1)
    assert out["time_series_target_mask"].tolist() == [[[True]], [[False]]]


# --------------------------------------------------------------------------
# Forward integration on a real (small) backbone. Marked slow/integration:
# loads OLMo-1B and exercises the aux-decoder path in UnifiedTransformer.
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_forward_aux_decoder_adds_loss_only_when_target_present():
    from src.config import ModelConfig
    from src.model import UnifiedTransformer

    cfg = ModelConfig(
        modalities=["text"],
        llm_backbone_id="allenai/OLMo-1B-0724-hf",
        output_decoders=["text", "time_series"],
        ts_variates=1,
        decoder_configs={"time_series": {"horizon": 6}},
        attn_implementation="eager",
    )
    model = UnifiedTransformer(cfg).eval()

    B, T = 2, 5
    inputs = {"text": torch.randint(0, 100, (B, T))}
    labels = torch.randint(0, 100, (B, T))

    # Text-only: aux path is a no-op (no time_series_target supplied).
    _, text_loss = model(inputs, labels=labels)
    assert torch.isfinite(text_loss)

    # With a target: the forecast loss is added to the total.
    inputs_ts = dict(inputs)
    inputs_ts["time_series_target"] = torch.randn(B, 6, 1)
    _, total_loss = model(inputs_ts, labels=labels)
    assert torch.isfinite(total_loss)
    assert float(total_loss) != float(text_loss)

    with torch.no_grad():
        model.decoders["time_series"].head.weight.zero_()
        model.decoders["time_series"].head.bias.zero_()
    inputs_ts["time_series_history_mean"] = torch.full((B, 1), 10.0)
    inputs_ts["time_series_history_scale"] = torch.full((B, 1), 2.0)
    inputs_ts["time_series_target"] = torch.full((B, 6, 1), 12.0)
    _, normalized_loss = model(inputs_ts, labels=labels)
    torch.testing.assert_close(normalized_loss, text_loss.float() + 0.5)
    result = model.forward_outputs(
        inputs_ts, targets={"time_series": inputs_ts["time_series_target"]},
        requested_outputs=["time_series"],
    )
    torch.testing.assert_close(result.loss, torch.tensor(0.5))
    torch.testing.assert_close(result.predictions["time_series"], torch.full((B, 6, 1, 3), 10.0))
    inference_inputs = {key: value for key, value in inputs_ts.items() if key != "time_series_target"}
    forecast = model.predict(inference_inputs, requested_outputs=["time_series"])
    torch.testing.assert_close(forecast.predictions["time_series"], torch.full((B, 6, 1), 10.0))

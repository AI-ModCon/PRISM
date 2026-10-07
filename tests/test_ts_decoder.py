"""Phase 2 tests: TimeSeriesDecoder.

Locks the time-series decoder contract:
- forward: hidden states -> quantile forecast of the right shape/dtype,
- pinball loss is finite, asymmetric across quantiles, and decreases on a
  1-step overfit (the "correctness, not just runs" check),
- generate() returns a point-forecast tensor (median quantile), not text,
- registry exposes the decoder.

Pure unit tests — no backbone, no network.
"""

from __future__ import annotations

import pytest
import torch
from src.decoders import DECODERS, TimeSeriesDecoder
from src.decoders.base import OutputDecoder

pytestmark = pytest.mark.unit


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

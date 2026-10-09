"""Phase 3 tests: GeometryDecoder.

Locks the geometry decoder contract:
- forward: hidden states -> field (B, N, C) of the right shape/dtype,
- MSE loss finite + shape-checked,
- encode->decode round-trip reconstructs a field above a trivial baseline
  (overfit-reduces-loss),
- generate() returns a field tensor,
- registry + native_decoder delegation hook.

Pure unit tests — no backbone, no Walrus (which is an optional dep).
"""

from __future__ import annotations

import pytest
import torch
from src.decoders import DECODERS, GeometryDecoder
from src.decoders.base import OutputDecoder

pytestmark = pytest.mark.unit


def test_registered():
    assert DECODERS["geometry"] is GeometryDecoder
    assert issubclass(GeometryDecoder, OutputDecoder)
    assert GeometryDecoder.output_kind == "tensor"
    assert GeometryDecoder.loss_kind == "mse"


def test_forward_shapes_from_sequence_hidden():
    dec = GeometryDecoder(d_model=16, num_points=10, num_channels=3)
    hidden = torch.randn(4, 5, 16)  # (B, T, d_model)
    pred, loss = dec(hidden, targets=None)
    assert pred.shape == (4, 10, 3)
    assert loss is None


def test_forward_accepts_pooled_hidden():
    dec = GeometryDecoder(d_model=16, num_points=10, num_channels=3)
    pred, _ = dec(torch.randn(4, 16), targets=None)
    assert pred.shape == (4, 10, 3)


def test_loss_finite_and_shape_checked():
    dec = GeometryDecoder(d_model=16, num_points=10, num_channels=3)
    hidden = torch.randn(4, 5, 16)
    target = torch.randn(4, 10, 3)
    _, loss = dec(hidden, targets=target)
    assert loss is not None and torch.isfinite(loss)
    with pytest.raises(RuntimeError, match="geometry target shape"):
        dec(hidden, targets=torch.randn(4, 10, 5))


def test_generate_returns_field_tensor():
    dec = GeometryDecoder(d_model=16, num_points=8, num_channels=2)
    out = dec.generate(torch.randn(3, 5, 16))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 8, 2)


def test_overfit_one_batch_reduces_loss():
    torch.manual_seed(0)
    dec = GeometryDecoder(d_model=16, num_points=6, num_channels=2)
    hidden = torch.randn(8, 16)
    target = torch.randn(8, 6, 2)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-2)
    _, first = dec(hidden, targets=target)
    for _ in range(50):
        opt.zero_grad()
        _, loss = dec(hidden, targets=target)
        loss.backward()
        opt.step()
    _, last = dec(hidden, targets=target)
    assert last < first * 0.5, (first.item(), last.item())


def test_native_decoder_hook_is_used():
    # When a native_decoder is provided, predict() delegates to it (the Walrus
    # bridge extension point).
    class _Native(torch.nn.Module):
        def forward(self, h):
            return torch.full((h.shape[0], 4, 2), 7.0)

    dec = GeometryDecoder(d_model=16, num_points=4, num_channels=2, native_decoder=_Native())
    out = dec.predict(torch.randn(3, 5, 16))
    assert out.shape == (3, 4, 2)
    assert torch.all(out == 7.0)


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        GeometryDecoder(d_model=8, num_points=0, num_channels=2)
    with pytest.raises(ValueError):
        GeometryDecoder(d_model=8, num_points=4, num_channels=0)
    with pytest.raises(ValueError):
        GeometryDecoder(d_model=8, num_points=4, num_channels=2, pool="bogus")


def test_model_requires_geometry_decoder_config(offline_hf):
    # The model can't guess num_points/num_channels — it must be configured.
    from src.config import ModelConfig
    from src.model import UnifiedTransformer

    cfg = ModelConfig(
        modalities=["text"], llm_backbone_id=None, output_decoders=["text", "geometry"]
    )
    with pytest.raises(ValueError, match="num_points"):
        UnifiedTransformer(cfg)


@pytest.mark.slow
@pytest.mark.integration
def test_forward_geometry_aux_loss_only_with_target(offline_hf, offline_backbone):
    from src.config import ModelConfig
    from src.model import UnifiedTransformer

    cfg = ModelConfig(
        modalities=["text"],
        llm_backbone_id="allenai/OLMo-1B-0724-hf",
        output_decoders=["text", "geometry"],
        attn_implementation="eager",
        decoder_configs={"geometry": {"num_points": 5, "num_channels": 2}},
    )
    model = UnifiedTransformer(cfg).eval()
    B, T = 2, 4
    inputs = {"text": torch.randint(0, 100, (B, T))}
    labels = torch.randint(0, 100, (B, T))
    _, base = model(inputs, labels=labels)
    inputs_g = dict(inputs)
    inputs_g["geometry_target"] = torch.randn(B, 5, 2)
    _, with_geo = model(inputs_g, labels=labels)
    assert torch.isfinite(with_geo)
    assert float(with_geo) != float(base)

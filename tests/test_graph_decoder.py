"""Phase 3 tests: GraphDecoder.

Locks the graph decoder contract:
- per-node regression + classification forward/loss,
- masked-feature reconstruction recovers held-out node features above a
  trivial baseline (overfit-reduces-loss),
- the §2 scope boundary: node decoding only, never de-novo structure
  generation (the decoder emits no edge set),
- registry + native_decoder delegation hook.

Pure unit tests — no backbone, no torch_geometric (optional dep).
"""

from __future__ import annotations

import pytest
import torch
from src.decoders import DECODERS, GraphDecoder
from src.decoders.base import OutputDecoder

pytestmark = pytest.mark.unit


def test_registered():
    assert DECODERS["graph"] is GraphDecoder
    assert issubclass(GraphDecoder, OutputDecoder)


def test_regression_forward_and_loss():
    dec = GraphDecoder(d_model=16, out_dim=4, task="regression")
    assert dec.loss_kind == "mse"
    hidden = torch.randn(2, 7, 16)  # (B, N, d_model)
    target = torch.randn(2, 7, 4)
    pred, loss = dec(hidden, targets=target)
    assert pred.shape == (2, 7, 4)
    assert loss is not None and torch.isfinite(loss)
    with pytest.raises(RuntimeError, match="graph regression target shape"):
        dec(hidden, targets=torch.randn(2, 7, 5))


def test_classification_forward_and_loss():
    dec = GraphDecoder(d_model=16, out_dim=5, task="classification")
    assert dec.loss_kind == "cross_entropy"
    hidden = torch.randn(2, 7, 16)
    labels = torch.randint(0, 5, (2, 7))
    pred, loss = dec(hidden, targets=labels)
    assert pred.shape == (2, 7, 5)
    assert loss is not None and torch.isfinite(loss)
    with pytest.raises(RuntimeError, match="graph classification target shape"):
        dec(hidden, targets=torch.randint(0, 5, (2, 9)))


def test_accepts_single_graph_2d_input():
    dec = GraphDecoder(d_model=16, out_dim=4)
    pred, _ = dec(torch.randn(7, 16), targets=None)  # (N, d_model)
    assert pred.shape == (1, 7, 4)


def test_emits_no_edges_scope_guard():
    # The §2 boundary: this decoder labels nodes, it does not generate graph
    # structure. Output is strictly (B, N, out_dim) — no edge tensor anywhere.
    dec = GraphDecoder(d_model=8, out_dim=3)
    pred, _ = dec(torch.randn(2, 4, 8), targets=None)
    assert pred.dim() == 3  # (B, N, out_dim), never an (2, E) edge index
    assert not hasattr(dec, "generate_structure")


def test_overfit_node_features_reduces_loss():
    torch.manual_seed(0)
    dec = GraphDecoder(d_model=16, out_dim=4, task="regression")
    hidden = torch.randn(2, 6, 16)
    target = torch.randn(2, 6, 4)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-2)
    _, first = dec(hidden, targets=target)
    for _ in range(60):
        opt.zero_grad()
        _, loss = dec(hidden, targets=target)
        loss.backward()
        opt.step()
    _, last = dec(hidden, targets=target)
    assert last < first * 0.5, (first.item(), last.item())


def test_classification_ignore_index():
    dec = GraphDecoder(d_model=8, out_dim=3, task="classification")
    hidden = torch.randn(1, 4, 8)
    labels = torch.tensor([[0, 1, 2, -100]])  # last node ignored
    _, loss = dec(hidden, targets=labels)
    assert torch.isfinite(loss)


def test_native_decoder_hook_is_used():
    class _Native(torch.nn.Module):
        def forward(self, h):
            return torch.full((h.shape[0], h.shape[1], 3), 5.0)

    dec = GraphDecoder(d_model=16, out_dim=3, native_decoder=_Native())
    out = dec.predict(torch.randn(2, 4, 16))
    assert out.shape == (2, 4, 3)
    assert torch.all(out == 5.0)


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        GraphDecoder(d_model=8, out_dim=0)
    with pytest.raises(ValueError):
        GraphDecoder(d_model=8, out_dim=3, task="bogus")

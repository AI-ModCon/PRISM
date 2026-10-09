"""Regression test for the (2, E) edge-index special case in BucketedCollator.

PR #91 added an `elif items[0].dim() == 2 and items[0].shape[0] == 2` branch
to BucketedCollator.__call__ to match MultimodalCollator's behavior on graph
edge indices. Without it, text_graph cells crashed at collate time with:
    "size of tensor a (E1) must match the size of tensor b (E2) at non-singleton
     dimension 1"
because pad_sequence aligns the longest axis (E) across items of shape (2, E).

The fix returns the list of tensors unchanged so the downstream encoder's
per-graph batching can handle it. This test pins that behavior so the
text_graph crash cannot silently recur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import BucketedCollator, MultimodalCollator


class _FakeTokenizer:
    pad_token_id = 0


def _make_graph_batch(edge_shapes):
    """Build a batch of dicts with a 'text' tensor (so the bucketer is happy)
    and an 'edge_index' tensor of shape (2, E) for each item."""
    batch = []
    for _i, e in enumerate(edge_shapes):
        batch.append(
            {
                "text": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "edge_index": torch.arange(2 * e, dtype=torch.long).reshape(2, e),
            }
        )
    return batch


def test_bucketed_collator_passes_variable_edge_indices_through_as_list():
    """Edge indices with mismatched E counts must be returned as a list of
    tensors — pad_sequence on dim 0 would try to align E and crash."""
    collator = BucketedCollator(_FakeTokenizer(), log_efficiency=False)
    batch = _make_graph_batch(edge_shapes=[3, 7, 5])  # variable E

    out = collator(batch)

    assert "edge_index" in out
    assert isinstance(out["edge_index"], list), (
        f"Variable-E edge_index must come back as a list, got {type(out['edge_index'])}"
    )
    assert len(out["edge_index"]) == 3
    for t in out["edge_index"]:
        assert t.dim() == 2 and t.shape[0] == 2


def test_bucketed_collator_matches_multimodal_collator_for_edge_indices():
    """Both collators must produce equivalent output on the (2, E) branch —
    the comment in BucketedCollator explicitly cites parity with
    MultimodalCollator. A divergence is a quiet regression."""
    bucketed = BucketedCollator(_FakeTokenizer(), log_efficiency=False, sort_within_batch=False)
    legacy = MultimodalCollator(_FakeTokenizer())

    batch = _make_graph_batch(edge_shapes=[4, 9])
    # Need to copy because BucketedCollator may mutate via truncation/sorting
    import copy

    out_b = bucketed(copy.deepcopy(batch))
    out_l = legacy(copy.deepcopy(batch))

    assert isinstance(out_b["edge_index"], list)
    assert isinstance(out_l["edge_index"], list)
    assert len(out_b["edge_index"]) == len(out_l["edge_index"])
    for a, b in zip(out_b["edge_index"], out_l["edge_index"], strict=True):
        assert torch.equal(a, b)


def test_bucketed_collator_uniform_edge_indices_still_stack():
    """If every edge_index has the same shape (uniform E), it is NOT variable
    and the standard torch.stack path runs. The (2, E) branch only fires under
    variable shapes."""
    collator = BucketedCollator(_FakeTokenizer(), log_efficiency=False, sort_within_batch=False)
    batch = _make_graph_batch(edge_shapes=[5, 5, 5])

    out = collator(batch)
    # All same shape → torch.stack path → tensor of shape (B, 2, E)
    assert isinstance(out["edge_index"], torch.Tensor), (
        f"Uniform edge_index should stack into a tensor, got {type(out['edge_index'])}"
    )
    assert out["edge_index"].shape == (3, 2, 5)


def test_bucketed_collator_2d_first_dim_is_two_but_uniform_stacks():
    """When the first dim of a 2D tensor IS 2 but all items are uniform shape,
    the is_variable=False path stacks them — the edge-index branch never fires.
    Pin this so a future shape-only check (without an is_variable guard) is
    caught."""
    collator = BucketedCollator(_FakeTokenizer(), log_efficiency=False, sort_within_batch=False)
    batch = [
        {
            "text": torch.tensor([1, 2, 3], dtype=torch.long),
            "edge_index": torch.arange(10, dtype=torch.long).reshape(2, 5),
        },
        {
            "text": torch.tensor([4, 5, 6], dtype=torch.long),
            "edge_index": torch.arange(10, dtype=torch.long).reshape(2, 5),
        },
    ]
    out = collator(batch)
    # Uniform shape -> stack -> tensor, not list
    assert isinstance(out["edge_index"], torch.Tensor)
    assert out["edge_index"].shape == (2, 2, 5)

"""Unit tests for the per-modality WebDataset wrapper (PR #73).

Covers the decoder dispatch + input-validation paths that don't require
DAOS or actual shards on disk. Real shard round-trip is exercised by the
Phase 2 sweep harness on a compute node; this module just locks the
class-level invariants so a refactor can't silently break them.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

pytest.importorskip("webdataset")  # _IterableDataset import would fail otherwise

from src.data.multi_webdataset import (
    ModalityAwareWebDatasetWrapper as Wrapper,
)

# ---------------------------------------------------------------------------
# Decoder round-trips — these are static-ish and don't need a wrapper instance.
# We grab them off a stub instance created with __new__ to avoid triggering
# __init__ (which would want a tokenizer + DAOS config).
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_wrapper():
    """Wrapper instance bypassing __init__ — only the decoders are exercised."""
    w = Wrapper.__new__(Wrapper)
    # _decode_image consults self._image_transform; the other decoders don't
    # touch any instance state.
    w._image_transform = None
    return w


def test_decode_time_series_from_numpy(stub_wrapper):
    arr = np.linspace(0.0, 1.0, 128, dtype="float32")
    out = stub_wrapper._decode_time_series(arr)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (128,)
    assert torch.allclose(out, torch.from_numpy(arr))


def test_decode_time_series_from_npy_bytes(stub_wrapper):
    arr = np.arange(64, dtype="float32")
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    out = stub_wrapper._decode_time_series(buf.getvalue())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (64,)
    assert torch.equal(out, torch.from_numpy(arr))


def test_decode_time_series_from_tensor_returns_float(stub_wrapper):
    t = torch.arange(16, dtype=torch.int64)
    out = stub_wrapper._decode_time_series(t)
    assert out.dtype == torch.float32


def test_decode_time_series_unknown_type_raises(stub_wrapper):
    with pytest.raises(TypeError):
        stub_wrapper._decode_time_series("not a tensor")


def test_decode_graph_passthrough_dict(stub_wrapper):
    g = {
        "x": torch.zeros(3, 1),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "num_nodes": torch.tensor(3, dtype=torch.long),
    }
    out = stub_wrapper._decode_graph(g)
    assert out is g  # passthrough


def test_decode_graph_from_torch_save_bytes(stub_wrapper):
    g = {
        "x": torch.tensor([[6.0], [1.0]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "num_nodes": torch.tensor(2, dtype=torch.long),
    }
    buf = io.BytesIO()
    torch.save(g, buf)
    out = stub_wrapper._decode_graph(buf.getvalue())
    assert set(out.keys()) == {"x", "edge_index", "num_nodes"}
    assert torch.equal(out["x"], g["x"])


def test_decode_graph_unknown_type_raises(stub_wrapper):
    with pytest.raises(TypeError):
        stub_wrapper._decode_graph(42)


# ---------------------------------------------------------------------------
# __init__ validation — these don't need DAOS because we expect the call to
# raise before MultiWebDataset is instantiated.
# ---------------------------------------------------------------------------


def test_init_rejects_empty_modalities():
    with pytest.raises(ValueError, match="modalities cannot be empty"):
        Wrapper(tokenizer=None, modalities=[])


def test_init_rejects_unsupported_modality():
    with pytest.raises(ValueError, match="unsupported modalities"):
        Wrapper(tokenizer=None, modalities=["table"])


def test_init_rejects_multi_modality_list():
    # Mixed modalities require parallel pipelines; not supported.
    # The error message names "(possibly composite)" so we match the stable
    # "exactly one" substring instead.
    with pytest.raises(ValueError, match="exactly one"):
        Wrapper(tokenizer=None, modalities=["image", "time_series"])


# ---------------------------------------------------------------------------
# MultiWebDataset modality dispatch — verify the per-modality tuple spec table
# stays in sync with what shard_modality.py writes.
# ---------------------------------------------------------------------------


def test_multi_webdataset_pipeline_table_covers_shard_modality_outputs():
    """Per-modality tuple_spec must include every key its sharder writes,
    or `to_tuple` will silently drop samples.
    """
    from src.data.multi_webdataset import MultiWebDataset

    table = MultiWebDataset._MODALITY_PIPELINE
    assert set(table.keys()) == {"image", "time_series", "graph", "vla"}
    # tuple spec for time_series must include the ext written by shard_modality.py
    assert "ts.npy" in table["time_series"][0]
    assert "graph.pt" in table["graph"][0]
    vla_spec = table["vla"][0]
    for key in ("head.jpg", "wrist.jpg", "pose.npy", "action.npy", "instruction.txt"):
        assert key in vla_spec, f"vla pipeline missing {key}"


def test_multi_webdataset_rejects_unknown_modality():
    from src.data.multi_webdataset import MultiWebDataset

    with pytest.raises(ValueError, match="unknown modality"):
        MultiWebDataset(config={"groups": {}}, modality="table")

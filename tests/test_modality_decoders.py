"""Unit tests for the per-modality WebDataset wrapper (PR #73).

Covers the decoder dispatch + input-validation paths that don't require
DAOS or actual shards on disk. Real shard round-trip is exercised by the
Phase 2 sweep harness on a compute node; this module just locks the
class-level invariants so a refactor can't silently break them.
"""

from __future__ import annotations

import io
import tarfile

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
    assert "gt_ts.npy" in table["time_series"][0]
    assert "gt_ts_mask.npy" in table["time_series"][0]
    assert "graph.pt" in table["graph"][0]
    vla_spec = table["vla"][0]
    for key in ("head.jpg", "wrist.jpg", "pose.npy", "action.npy", "instruction.txt"):
        assert key in vla_spec, f"vla pipeline missing {key}"


def test_time_series_target_and_task_id_reach_wrapper_optional_target():
    from types import SimpleNamespace

    from src.data.multi_webdataset import MultiWebDataset

    def npy_bytes(array):
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()

    input_series = np.array([[1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    target = np.array([[3.0], [4.0]], dtype=np.float32)
    target_mask = np.array([[True], [False]])
    metadata = b'{"id":"s1","task_id":"ENG02"}'
    process = MultiWebDataset._process_time_series_sample
    forecast_sample = process(
        None,
        (
            npy_bytes(input_series),
            b"Forecast the series.",
            metadata,
            npy_bytes(target),
            npy_bytes(target_mask),
        ),
    )
    targetless_sample = process(
        None,
        (npy_bytes(input_series), b"A nonforecast sample.", metadata, None, None),
    )

    class _Tokenizer:
        pad_token_id = 0

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    wrapper = Wrapper.__new__(Wrapper)
    wrapper._primary_modality = "time_series"
    wrapper._image_transform = None
    wrapper.multi_ds = [forecast_sample, targetless_sample]
    wrapper.tokenizer = _Tokenizer()
    wrapper.max_length = 16

    forecast_output, targetless_output = list(wrapper)
    assert torch.equal(forecast_output["time_series_target"], torch.from_numpy(target))
    assert torch.equal(
        forecast_output["time_series_target_mask"], torch.from_numpy(target_mask)
    )
    assert forecast_output["task_id"] == "ENG02"
    assert "time_series_target" not in targetless_output


def test_synthetic_forecast_tar_reaches_masked_decoder_loss(tmp_path):
    import webdataset as wds
    from types import SimpleNamespace

    from src.data.collate import MultimodalCollator
    from src.decoders import TimeSeriesDecoder
    from src.data.multi_webdataset import MultiWebDataset

    def payload(array):
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()

    shard_path = tmp_path / "forecast.tar"
    members = {
        "sample.ts.npy": payload(np.array([[1, 1], [2, 0]], dtype=np.float32)),
        "sample.gt_ts.npy": payload(np.array([[3.0], [4.0]], dtype=np.float32)),
        "sample.gt_ts_mask.npy": payload(np.array([[True], [False]])),
        "sample.text": b"Forecast the future.",
        "sample.meta.json": b'{"id":"sample","task_id":"MEG03"}',
        "targetless.ts.npy": payload(np.array([[8.0, 1.0]], dtype=np.float32)),
        "targetless.text": b"A nonforecast sample.",
        "targetless.meta.json": b'{"id":"targetless","task_id":"MEU01"}',
    }
    with tarfile.open(shard_path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    tuple_spec = MultiWebDataset._MODALITY_PIPELINE["time_series"][0]
    samples = list(
        wds.WebDataset(str(shard_path), shardshuffle=False)
        .to_tuple(*tuple_spec, missing_is_error=False, none_is_error=False)
        .map(lambda sample: MultiWebDataset._process_time_series_sample(None, sample))
    )

    class _Tokenizer:
        pad_token_id = 0

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    wrapper = Wrapper.__new__(Wrapper)
    wrapper._primary_modality = "time_series"
    wrapper._image_transform = None
    wrapper.multi_ds = samples
    wrapper.tokenizer = _Tokenizer()
    wrapper.max_length = 16
    decoded = list(wrapper)

    batch = MultimodalCollator(_Tokenizer(), passthrough_time_series=True)(decoded)
    decoder = TimeSeriesDecoder(d_model=8, horizon=5, num_vars=3)
    prediction, loss = decoder(
        torch.randn(2, 3, 8),
        targets=batch["time_series_target"],
        target_mask=batch["time_series_target_mask"],
    )
    loss.backward()

    assert len(decoded) == 2
    assert len(batch["time_series"]) == 2
    assert batch["time_series"][0].shape == (2, 2)
    assert "time_series_target" not in decoded[1]
    assert batch["time_series_target_mask"].sum().item() == 1
    assert prediction.shape == (2, 5, 3, 3)
    assert torch.isfinite(loss)
    assert decoder.head.weight.grad is not None


def test_multi_webdataset_rejects_unknown_modality():
    from src.data.multi_webdataset import MultiWebDataset

    with pytest.raises(ValueError, match="unknown modality"):
        MultiWebDataset(config={"groups": {}}, modality="table")

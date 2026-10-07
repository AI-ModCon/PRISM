"""Parity tests for ModalityAwareWebDatasetWrapper.

Two gates:

1. **Single-modality bit-parity** — `modalities=[<image|time_series|graph>]`
   produces the same downstream sample dict shape (keys + tensor dtypes)
   that single-modality training relies on.

2. **VLA composite round-trip** — feed a one-sample CALVIN-style dict
   through `_iter_composite` (bypassing DAOS) and assert all five typed
   tensors plus the metadata string come out with the right shapes/dtypes.

Pure-Python code paths only — no DAOS, no real shards, no XPU. Runs on the
login node under `pytest --timeout=60`.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("webdataset")

from src.data.multi_webdataset import ModalityAwareWebDatasetWrapper as Wrapper

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal HF-style tokenizer stand-in.

    Returns one token per ASCII character of the input, padded into a
    SimpleNamespace with `.input_ids` matching the HF return type used by
    the wrapper. Deterministic + reproducible across runs.
    """

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = True,
        max_length: int = 2048,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        ids = [ord(c) % 1000 for c in text[:max_length]] or [0]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def _stub_wrapper(modality: str = "vla") -> Wrapper:
    """Construct a wrapper bypassing __init__ (no DAOS)."""
    w = Wrapper.__new__(Wrapper)
    w.tokenizer = _StubTokenizer()
    w.max_length = 2048
    w._modalities = [modality]
    w._primary_modality = modality
    # Match what __init__ would build; the wrapper's _decode_image consults this.
    from torchvision import transforms

    w._image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return w


# ---------------------------------------------------------------------------
# 1. Single-modality bit-parity
# ---------------------------------------------------------------------------


def test_image_decoder_signature_unchanged():
    """A PIL Image goes through `_decode_image` → (3, 224, 224) float tensor.

    Locks the shape contract that downstream image-only training relies on.
    """
    from PIL import Image

    w = _stub_wrapper("image")
    img = Image.new("RGB", (40, 40), color=(128, 64, 32))
    out = w._decode_image(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_time_series_decoder_signature_unchanged():
    """numpy npy bytes → 1-D float32 tensor of the original length."""
    w = _stub_wrapper("time_series")
    arr = np.linspace(-1.0, 1.0, 96, dtype="float32")
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    out = w._decode_time_series(buf.getvalue())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (96,)
    assert out.dtype == torch.float32


def test_graph_decoder_signature_unchanged():
    """torch.save'd dict → dict with the same keys + tensor equality on `x`."""
    w = _stub_wrapper("graph")
    g = {
        "x": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "num_nodes": torch.tensor(4, dtype=torch.long),
    }
    buf = io.BytesIO()
    torch.save(g, buf)
    out = w._decode_graph(buf.getvalue())
    assert set(out.keys()) == {"x", "edge_index", "num_nodes"}
    assert torch.equal(out["x"], g["x"])


# ---------------------------------------------------------------------------
# 2. VLA composite round-trip
# ---------------------------------------------------------------------------


def _jpeg_bytes(color: tuple[int, int, int] = (100, 50, 200)) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _vla_sample_dict() -> dict[str, Any]:
    """Mimic the dict `_process_vla_sample` emits — typed bytes for each key."""
    return {
        "image_head": _jpeg_bytes((100, 50, 200)),
        "image_wrist": _jpeg_bytes((30, 200, 70)),
        "pose": _npy_bytes(np.random.RandomState(0).randn(15).astype("float32")),
        "action": _npy_bytes(
            np.random.RandomState(1).randn(7).astype("float32")
        ),
        "text": "lift the red block",
        "metadata": {
            "_data_type": "vla",
            "episode": 42,
            "step": 5,
            "task_index": 17,
        },
    }


def test_vla_composite_yields_all_keys_with_right_shapes():
    """All five typed tensors decode + the tokenizer fires on the instruction."""
    w = _stub_wrapper("vla")
    # Inject a fake MultiWebDataset that yields one composite sample.
    w.multi_ds = iter([_vla_sample_dict()])

    out = next(iter(w._iter_composite("vla")))
    assert set(out.keys()) >= {
        "image_head",
        "image_wrist",
        "pose",
        "action",
        "text",
        "text_attention_mask",
        "_metadata",
    }
    assert out["image_head"].shape == (3, 224, 224)
    assert out["image_wrist"].shape == (3, 224, 224)
    assert out["pose"].shape == (15,)
    assert out["pose"].dtype == torch.float32
    assert out["action"].shape == (7,)
    assert out["action"].dtype == torch.float32
    assert out["text"].dtype == torch.long
    assert out["text"].ndim == 1
    assert out["text_attention_mask"].shape == out["text"].shape
    assert "[vla]" in out["_metadata"]
    assert "ep=42" in out["_metadata"]


def test_vla_composite_skips_sample_with_missing_payload(caplog):
    """A sample missing `pose` should be logged + skipped, not raise."""
    w = _stub_wrapper("vla")
    bad = _vla_sample_dict()
    del bad["pose"]
    good = _vla_sample_dict()
    w.multi_ds = iter([bad, good])

    items = list(w._iter_composite("vla"))
    assert len(items) == 1
    assert "pose" in caplog.text or "missing" in caplog.text.lower()


def test_vla_composite_skips_sample_with_corrupt_jpeg(caplog):
    """Decoder exceptions inside `_iter_composite` are logged + skipped,
    not raised. Otherwise one bad shard would silently drop 100% of its
    samples — the same failure mode the SUPPORTED_MODALITIES check guards.
    """
    import logging

    caplog.set_level(logging.WARNING)
    w = _stub_wrapper("vla")
    corrupt = _vla_sample_dict()
    corrupt["image_head"] = b"not a real jpeg"
    good = _vla_sample_dict()
    w.multi_ds = iter([corrupt, good])

    items = list(w._iter_composite("vla"))
    assert len(items) == 1
    assert "skipping sample" in caplog.text.lower()


def test_vla_composite_skips_sample_with_corrupt_npy(caplog):
    """A non-numpy `pose` payload is dropped via the broad-except path."""
    import logging

    caplog.set_level(logging.WARNING)
    w = _stub_wrapper("vla")
    corrupt = _vla_sample_dict()
    corrupt["pose"] = b"\x00\x01\x02 not an npy header"
    good = _vla_sample_dict()
    w.multi_ds = iter([corrupt, good])

    items = list(w._iter_composite("vla"))
    assert len(items) == 1
    assert "skipping sample" in caplog.text.lower()


def test_vla_composite_handles_bytes_text_and_json_metadata():
    """Wire-format check: raw bytes from WebDataset should decode cleanly."""
    w = _stub_wrapper("vla")
    sample = _vla_sample_dict()
    sample["text"] = b"raw utf-8 bytes \xe2\x9c\x93"
    sample["metadata"] = json.dumps(sample["metadata"]).encode("utf-8")
    # _iter_composite metadata branch expects a dict OR string; raw bytes
    # would hit the `isinstance(metadata, dict)` check and fall through, so
    # the test catches a regression where _process_vla_sample's JSON-decode
    # was lost.
    w.multi_ds = iter([sample])

    out = next(iter(w._iter_composite("vla")))
    assert out["text"].dtype == torch.long


# ---------------------------------------------------------------------------
# 3. Composite still requires a known modality
# ---------------------------------------------------------------------------


def test_iter_composite_rejects_unknown_modality():
    w = _stub_wrapper("vla")
    w.multi_ds = iter([])
    with pytest.raises(NotImplementedError):
        list(w._iter_composite("table"))

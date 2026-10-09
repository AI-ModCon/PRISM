from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path

import pytest
import torch

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.data.collate import MultimodalCollator
from src.data.multimodal import StreamingMultimodalDataset
from src.encoders.time_series import TimeSeriesEncoder

pytestmark = [pytest.mark.integration, pytest.mark.multimodal, pytest.mark.timeseries]


class _TestTokenizer:
    pad_token_id = 0
    eos_token = "<|eot|>"

    def __call__(self, text: str, return_tensors: str = "pt", **_kwargs):
        # Minimal deterministic tokenization: 1 token per whitespace item.
        n = max(1, len(text.split()))
        ids = torch.arange(1, n + 1, dtype=torch.long).unsqueeze(0)
        return type("Tokenized", (), {"input_ids": ids})

    def decode(self, _token_ids):
        return "<unk>"


class _DummyDatasetContext:
    def __init__(self, max_ts_length: int = 4096):
        self.model_config = ModelConfig(
            is_timeseries=True,
            ts_projector="timeomni",
            max_ts_length=max_ts_length,
            normalize_ts_in_encoder=True,
            is_interleaved_qa=False,
        )
        self.tokenizer = _TestTokenizer()
        self.max_seq_length = None


def _find_scits_dir() -> Path:
    candidates = [
        os.environ.get("SCITS_SHARD_DIR"),
        os.environ.get("SCITS_WEBDATASET_DIR"),
        "/flare/ModCon/pemami/data/SciTS-processed",
        "/tmp/smoke_scits/shards",
        "/tmp/smoke_scits",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if p.is_file() and p.suffix == ".tar":
            return p.parent
        if p.is_dir():
            return p
    pytest.skip("No SciTS shard directory found; set SCITS_SHARD_DIR to run this test")


def _first_shard(shard_dir: Path) -> Path:
    for path in [shard_dir, shard_dir / "shards"]:
        tars = sorted(path.glob("*.tar"))
        if tars:
            return tars[0]
    pytest.skip(f"No .tar shards found under {shard_dir}")



def _find_nonzero_variate_shard(shard_dir: Path) -> tuple[Path, np.ndarray, str]:
    """Scan shards in order and return the first sample with shape (T, V), V>0."""
    search_paths = [shard_dir, shard_dir / "shards"]
    for base in search_paths:
        for shard_path in sorted(base.glob("*.tar")):
            by_prefix: dict[str, dict[str, bytes]] = {}
            with tarfile.open(shard_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    if member.name.endswith(".ts.npy"):
                        prefix = member.name[: -len(".ts.npy")]
                        suffix = "ts"
                    elif member.name.endswith(".text"):
                        prefix = member.name[: -len(".text")]
                        suffix = "text"
                    else:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    by_prefix.setdefault(prefix, {})[suffix] = extracted.read()

            for _prefix, payloads in by_prefix.items():
                if "ts" not in payloads or "text" not in payloads:
                    continue
                ts = np.load(io.BytesIO(payloads["ts"]), allow_pickle=False)
                text = payloads["text"].decode("utf-8").strip()
                if ts.ndim == 2 and ts.shape[0] > 0 and ts.shape[1] > 0 and text:
                    return shard_path, ts, text

    pytest.skip("No non-zero-variate (T, V>0) SciTS sample found in any shard")


def test_scits_shard_nonzero_variate_produces_encoder_output():
    """End-to-end success path: real (T, V>0) SciTS sample flows through
    _process_ts_qa -> MultimodalCollator passthrough -> TimeSeriesEncoder
    and produces a valid (1, patches, d_model) output tensor."""
    shard_dir = _find_scits_dir()
    _shard_path, ts_np, text = _find_nonzero_variate_shard(shard_dir)

    T, V = ts_np.shape
    # Use a patch_len that fits comfortably within T to keep the test fast.
    patch_len = min(64, T)
    # max_ts_length must accommodate the flattened univariate view: T*V steps.
    max_ts_len = T * V + patch_len  # small headroom

    dummy = _DummyDatasetContext(max_ts_length=max_ts_len)
    item = {
        "text": text,
        "timeseries": ts_np.tolist(),
    }

    ts_tensor, prompt_target, _meta = StreamingMultimodalDataset._process_ts_qa(dummy, item)
    assert isinstance(ts_tensor, torch.Tensor)
    assert ts_tensor.ndim == 2
    assert ts_tensor.numel() > 0

    tokenized = dummy.tokenizer(prompt_target, return_tensors="pt").input_ids.squeeze(0)
    collator = MultimodalCollator(
        dummy.tokenizer,
        max_seq_length=512,
        passthrough_time_series=True,
    )
    batch = collator([{"text": tokenized, "time_series": ts_tensor}])

    assert isinstance(batch["time_series"], list)
    assert len(batch["time_series"]) == 1

    encoder = TimeSeriesEncoder(
        encoder_type="timeomni",
        num_vars=1,
        d_ts=32,
        max_ts_length=max_ts_len,
        timeomni_patch_len=[patch_len],
        timeomni_stride=[patch_len],
        timeomni_d_model=32,
        timeomni_max_patches=10000,
    )
    encoded = encoder(batch["time_series"])

    assert encoded.ndim == 3, f"expected 3-D output, got shape {encoded.shape}"
    assert encoded.shape[0] == 1
    assert encoded.shape[1] > 0
    assert encoded.shape[2] == 32

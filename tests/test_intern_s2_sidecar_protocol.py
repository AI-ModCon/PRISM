"""Unit tests for tools/intern_s2_sidecar_protocol.py's binary wire framing.

No sockets, no torch dependency required beyond what encode/decode_tensor
themselves need (torch is a project-wide dependency already, same as
tests/test_vllm_ts_route.py's dtype round-trip tests, which this mirrors).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from tools.intern_s2_sidecar_protocol import (
    MAGIC,
    MAX_NDIM,
    MAX_PAYLOAD_BYTES,
    MsgType,
    ProtocolError,
    decode_tensor,
    encode_tensor,
    pack_frame,
    read_frame,
)


def _reader(data: bytes):
    buf = io.BytesIO(data)
    return buf.read


# ---------------------------------------------------------------------------
# pack_frame / read_frame round-trip


def test_ping_pong_round_trip():
    data = pack_frame(MsgType.PING)
    frame = read_frame(_reader(data))
    assert frame.msg_type == MsgType.PING
    assert frame.dtype is None
    assert frame.shape is None
    assert frame.payload == b""


def test_error_frame_round_trip():
    data = pack_frame(MsgType.ERROR, payload=b"boom")
    frame = read_frame(_reader(data))
    assert frame.msg_type == MsgType.ERROR
    assert frame.payload == b"boom"


def test_tensor_frame_round_trip_float32():
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    data = pack_frame(MsgType.TENSOR, dtype="float32", shape=(2, 4), payload=arr.tobytes())
    frame = read_frame(_reader(data))
    assert frame.msg_type == MsgType.TENSOR
    assert frame.dtype == "float32"
    assert frame.shape == (2, 4)
    assert frame.payload == arr.tobytes()


def test_tensor_frame_round_trip_float16():
    arr = (np.arange(6, dtype=np.float32) * 0.5).astype(np.float16).reshape(3, 2)
    data = pack_frame(MsgType.TENSOR, dtype="float16", shape=(3, 2), payload=arr.tobytes())
    frame = read_frame(_reader(data))
    assert frame.dtype == "float16"
    assert frame.shape == (3, 2)


def test_tensor_frame_round_trip_bfloat16():
    t = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    raw = t.contiguous().view(torch.uint16).numpy().tobytes()
    data = pack_frame(MsgType.TENSOR, dtype="bfloat16", shape=(1, 2), payload=raw)
    frame = read_frame(_reader(data))
    assert frame.dtype == "bfloat16"
    assert frame.payload == raw


def test_tensor_frame_round_trip_3d_shape():
    arr = np.zeros((2, 3, 4), dtype=np.float32)
    data = pack_frame(MsgType.TENSOR, dtype="float32", shape=(2, 3, 4), payload=arr.tobytes())
    frame = read_frame(_reader(data))
    assert frame.shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# pack_frame validation


def test_pack_frame_rejects_unknown_msg_type():
    with pytest.raises(ProtocolError):
        pack_frame(99)


def test_pack_frame_rejects_dtype_without_shape():
    with pytest.raises(ProtocolError):
        pack_frame(MsgType.TENSOR, dtype="float32", shape=None)


def test_pack_frame_rejects_shape_without_dtype():
    with pytest.raises(ProtocolError):
        pack_frame(MsgType.TENSOR, dtype=None, shape=(1, 2))


def test_pack_frame_rejects_unsupported_dtype():
    with pytest.raises(ProtocolError):
        pack_frame(MsgType.TENSOR, dtype="int8", shape=(1,), payload=b"\x00")


def test_pack_frame_rejects_too_many_dims():
    shape = tuple([1] * (MAX_NDIM + 1))
    with pytest.raises(ProtocolError):
        pack_frame(MsgType.TENSOR, dtype="float32", shape=shape, payload=b"")


def test_pack_frame_rejects_oversized_payload():
    with pytest.raises(ProtocolError):
        pack_frame(MsgType.ERROR, payload=b"\x00" * (MAX_PAYLOAD_BYTES + 1))


# ---------------------------------------------------------------------------
# read_frame malformed-input rejection


def test_read_frame_rejects_bad_magic():
    data = b"XXXX" + pack_frame(MsgType.PING)[len(MAGIC) :]
    with pytest.raises(ProtocolError, match="Bad magic"):
        read_frame(_reader(data))


def test_read_frame_rejects_truncated_header():
    with pytest.raises(ProtocolError, match="Connection closed"):
        read_frame(_reader(b"IS2"))


def test_read_frame_rejects_truncated_payload():
    full = pack_frame(MsgType.TENSOR, dtype="float32", shape=(2,), payload=b"\x00" * 8)
    truncated = full[:-4]
    with pytest.raises(ProtocolError, match="Connection closed"):
        read_frame(_reader(truncated))


def test_read_frame_rejects_unknown_msg_type():
    import struct

    header = struct.pack(">4sBBBB", MAGIC, 1, 250, 0, 0)
    length = struct.pack(">Q", 0)
    with pytest.raises(ProtocolError, match="Unknown msg_type"):
        read_frame(_reader(header + length))


def test_read_frame_rejects_unsupported_version():
    import struct

    header = struct.pack(">4sBBBB", MAGIC, 99, MsgType.PING, 0, 0)
    length = struct.pack(">Q", 0)
    with pytest.raises(ProtocolError, match="Unsupported protocol version"):
        read_frame(_reader(header + length))


def test_read_frame_rejects_oversized_length_field():
    import struct

    header = struct.pack(">4sBBBB", MAGIC, 1, MsgType.ERROR, 0, 0)
    length = struct.pack(">Q", MAX_PAYLOAD_BYTES + 1)
    with pytest.raises(ProtocolError, match="exceeds MAX_PAYLOAD_BYTES"):
        read_frame(_reader(header + length))


def test_read_frame_rejects_ndim_over_max():
    import struct

    header = struct.pack(">4sBBBB", MAGIC, 1, MsgType.TENSOR, 1, MAX_NDIM + 1)
    with pytest.raises(ProtocolError, match="exceeds MAX_NDIM"):
        read_frame(_reader(header))


# ---------------------------------------------------------------------------
# encode_tensor / decode_tensor


def test_encode_decode_tensor_round_trip_float32():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dtype, shape, raw = encode_tensor(t)
    out = decode_tensor(dtype, shape, raw)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, t)


def test_encode_decode_tensor_round_trip_float16():
    t = (torch.arange(8, dtype=torch.float32) * 0.25).to(torch.float16).reshape(2, 4)
    dtype, shape, raw = encode_tensor(t)
    out = decode_tensor(dtype, shape, raw)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, t)


def test_encode_decode_tensor_round_trip_bfloat16():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    dtype, shape, raw = encode_tensor(t)
    out = decode_tensor(dtype, shape, raw)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, t)


def test_encode_tensor_rejects_unsupported_dtype():
    t = torch.zeros(2, 2, dtype=torch.int64)
    with pytest.raises(ProtocolError, match="Unsupported tensor dtype"):
        encode_tensor(t)


def test_decode_tensor_rejects_length_mismatch():
    with pytest.raises(ProtocolError, match="expects"):
        decode_tensor("float32", (2, 2), b"\x00" * 8)  # needs 16 bytes


def test_decode_tensor_rejects_unsupported_dtype():
    with pytest.raises(ProtocolError, match="Unsupported dtype"):
        decode_tensor("int8", (2,), b"\x00\x00")

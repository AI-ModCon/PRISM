"""Wire protocol for the Intern-S2 sidecar's Unix domain socket.

vLLM (transformers<5) and Intern-S2's vendored config (transformers>=5.2.0,
for `RopeParameters`) cannot share a process. The sidecar runs the encoder
in its own venv; this module defines the small binary framing both sides
use to exchange tensors over a local UDS.

Not HTTP/JSON/base64: the caller is our own client in the same process tree
on the same node, so the event-loop and base64/JSON framing tax of a real
HTTP server buys nothing here. The dtype table matches
`src.vllm_plugin.openai_schema._SUPPORTED_DTYPES` so the same three dtypes
(float32/float16/bfloat16) are supported identically to the existing
`/v1/prism/ts` route.

Frame layout (all integers big-endian / network byte order):

    magic     4 bytes   b"IS2P"
    version   1 byte    protocol version (currently 1)
    msg_type  1 byte    MsgType value
    dtype     1 byte    DTYPE_CODES value, or 0 when the frame carries no tensor
    ndim      1 byte    number of shape dims, 0 when the frame carries no tensor
    shape     ndim * 4 bytes   uint32 per dimension
    length    8 bytes   uint64 payload length in bytes
    payload   `length` bytes  raw tensor bytes, a UTF-8 error message, or empty
"""

from __future__ import annotations

import socket
import struct

MAGIC = b"IS2P"
VERSION = 1

# Cap ndim so a corrupt/malicious header can't make us allocate an
# arbitrarily large shape tuple before we've even read the payload length.
MAX_NDIM = 8
# Refuse to allocate more than this many payload bytes for one frame. Well
# above any real Intern-S2 batch (a 397B-model batch of raw time series is
# still orders of magnitude smaller than this), and small enough that a
# corrupt length field fails fast instead of trying to recv gigabytes.
MAX_PAYLOAD_BYTES = 1 << 30  # 1 GiB


class ProtocolError(ValueError):
    """Malformed frame: bad magic, unknown msg_type/dtype, or a length that
    doesn't match what was actually read."""


class MsgType:
    PING = 0
    PONG = 1
    TENSOR = 2
    ERROR = 3


_VALID_MSG_TYPES = {MsgType.PING, MsgType.PONG, MsgType.TENSOR, MsgType.ERROR}

# Mirrors src.vllm_plugin.openai_schema._SUPPORTED_DTYPES's key set. Codes
# are protocol-stable (never renumber existing entries).
DTYPE_CODES: dict[str, int] = {"float32": 1, "float16": 2, "bfloat16": 3}
CODE_TO_DTYPE: dict[int, str] = {v: k for k, v in DTYPE_CODES.items()}
NO_DTYPE = 0

_HEADER_FIXED = struct.Struct(">4sBBBB")  # magic, version, msg_type, dtype, ndim
_LENGTH = struct.Struct(">Q")
_DIM = struct.Struct(">I")


class Frame:
    """A decoded protocol frame: msg_type plus an optional tensor descriptor."""

    __slots__ = ("msg_type", "dtype", "shape", "payload")

    def __init__(
        self,
        msg_type: int,
        dtype: str | None,
        shape: tuple[int, ...] | None,
        payload: bytes,
    ) -> None:
        self.msg_type = msg_type
        self.dtype = dtype
        self.shape = shape
        self.payload = payload


def pack_frame(
    msg_type: int,
    *,
    dtype: str | None = None,
    shape: tuple[int, ...] | None = None,
    payload: bytes = b"",
) -> bytes:
    """Encode one frame. `dtype`/`shape` are required together for TENSOR
    frames and must be omitted (None) for every other msg_type."""
    if msg_type not in _VALID_MSG_TYPES:
        raise ProtocolError(f"Unknown msg_type {msg_type!r}")
    if (dtype is None) != (shape is None):
        raise ProtocolError("dtype and shape must both be set or both be None")
    if dtype is not None and shape is not None:
        if dtype not in DTYPE_CODES:
            raise ProtocolError(
                f"Unsupported dtype {dtype!r}; expected one of {sorted(DTYPE_CODES)}"
            )
        if len(shape) > MAX_NDIM:
            raise ProtocolError(f"ndim={len(shape)} exceeds MAX_NDIM={MAX_NDIM}")
        dtype_code = DTYPE_CODES[dtype]
        ndim = len(shape)
        shape_bytes = b"".join(_DIM.pack(d) for d in shape)
    else:
        dtype_code = NO_DTYPE
        ndim = 0
        shape_bytes = b""

    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"payload of {len(payload)} bytes exceeds MAX_PAYLOAD_BYTES={MAX_PAYLOAD_BYTES}"
        )

    header = _HEADER_FIXED.pack(MAGIC, VERSION, msg_type, dtype_code, ndim)
    return header + shape_bytes + _LENGTH.pack(len(payload)) + payload


def _recv_exact(recv_fn, n: int) -> bytes:
    """Read exactly `n` bytes via `recv_fn(nbytes) -> bytes`, or raise
    ProtocolError on short read (peer closed mid-frame)."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = recv_fn(remaining)
        if not chunk:
            raise ProtocolError(f"Connection closed after {n - remaining} of {n} expected bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(recv_fn) -> Frame:
    """Decode one frame from a byte source. `recv_fn(nbytes) -> bytes` should
    behave like `socket.socket.recv` (may return fewer than requested bytes,
    empty bytes on EOF)."""
    fixed = _recv_exact(recv_fn, _HEADER_FIXED.size)
    magic, version, msg_type, dtype_code, ndim = _HEADER_FIXED.unpack(fixed)
    if magic != MAGIC:
        raise ProtocolError(f"Bad magic {magic!r}; expected {MAGIC!r}")
    if version != VERSION:
        raise ProtocolError(f"Unsupported protocol version {version}; expected {VERSION}")
    if msg_type not in _VALID_MSG_TYPES:
        raise ProtocolError(f"Unknown msg_type {msg_type!r}")
    if ndim > MAX_NDIM:
        raise ProtocolError(f"ndim={ndim} exceeds MAX_NDIM={MAX_NDIM}")

    if dtype_code == NO_DTYPE:
        if ndim != 0:
            raise ProtocolError("dtype=NO_DTYPE but ndim != 0")
        dtype = None
        shape = None
    else:
        if dtype_code not in CODE_TO_DTYPE:
            raise ProtocolError(f"Unknown dtype code {dtype_code!r}")
        dtype = CODE_TO_DTYPE[dtype_code]
        shape_bytes = _recv_exact(recv_fn, ndim * _DIM.size)
        shape = tuple(_DIM.unpack_from(shape_bytes, i * _DIM.size)[0] for i in range(ndim))

    (length,) = _LENGTH.unpack(_recv_exact(recv_fn, _LENGTH.size))
    if length > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"payload length {length} exceeds MAX_PAYLOAD_BYTES={MAX_PAYLOAD_BYTES}"
        )
    payload = _recv_exact(recv_fn, length) if length else b""
    return Frame(msg_type, dtype, shape, payload)


def send_frame(sock: socket.socket, msg_type: int, **kwargs) -> None:
    sock.sendall(pack_frame(msg_type, **kwargs))


def recv_frame(sock: socket.socket) -> Frame:
    return read_frame(sock.recv)


# ---------------------------------------------------------------------------
# Tensor <-> bytes. Deliberately independent of numpy/torch import order so
# `pack_frame`/`read_frame` above stay usable with no ML dependency at all
# (protocol-only unit tests import nothing but this module).


def encode_tensor(tensor) -> tuple[str, tuple[int, ...], bytes]:
    """torch.Tensor -> (dtype_name, shape, raw_bytes). Mirrors
    `openai_schema._decode_tensor`'s bfloat16 handling in reverse: bfloat16
    has no numpy dtype, so we view it as uint16 before taking raw bytes."""
    import torch

    dtype_name = _torch_dtype_to_name(tensor.dtype)
    t = tensor.contiguous()
    if t.dtype == torch.bfloat16:
        raw = t.view(torch.uint16).cpu().numpy().tobytes()
    else:
        raw = t.cpu().numpy().tobytes()
    return dtype_name, tuple(t.shape), raw


def decode_tensor(dtype: str, shape: tuple[int, ...], raw: bytes):
    """(dtype_name, shape, raw_bytes) -> torch.Tensor. Inverse of encode_tensor."""
    import numpy as np
    import torch

    if dtype not in DTYPE_CODES:
        raise ProtocolError(f"Unsupported dtype {dtype!r}; expected one of {sorted(DTYPE_CODES)}")

    n_elem = 1
    for d in shape:
        n_elem *= d
    elem_size = 2 if dtype in ("float16", "bfloat16") else 4
    expected = n_elem * elem_size
    if len(raw) != expected:
        raise ProtocolError(
            f"payload is {len(raw)} bytes, but shape {shape} + dtype {dtype} "
            f"expects {expected} bytes ({n_elem} elements * {elem_size} bytes each)"
        )

    if dtype == "bfloat16":
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
        return torch.from_numpy(arr).view(torch.bfloat16)
    np_dtype = {"float32": np.float32, "float16": np.float16}[dtype]
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
    return torch.from_numpy(arr)


def _torch_dtype_to_name(dtype) -> str:
    import torch

    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    if dtype not in mapping:
        raise ProtocolError(
            f"Unsupported tensor dtype {dtype!r}; expected one of {sorted(DTYPE_CODES)}"
        )
    return mapping[dtype]


__all__ = [
    "CODE_TO_DTYPE",
    "DTYPE_CODES",
    "MAGIC",
    "MAX_NDIM",
    "MAX_PAYLOAD_BYTES",
    "NO_DTYPE",
    "VERSION",
    "Frame",
    "MsgType",
    "ProtocolError",
    "decode_tensor",
    "encode_tensor",
    "pack_frame",
    "read_frame",
    "recv_frame",
    "send_frame",
]

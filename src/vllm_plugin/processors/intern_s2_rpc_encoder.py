"""Client-side stand-in for Intern-S2 when it runs in the sidecar process.

`InternS2RPCEncoder` duck-types the `(inner, hidden, forward_fn)` contract
`TimeSeriesModalityProcessor.build_encoder()` returns for every other
encoder type: it's an `nn.Module` with zero real parameters whose forward
pass round-trips a `(B, T, V)` tensor through the sidecar's Unix domain
socket and returns the `(B, Patches, Hidden)` result. This keeps
`PrismForConditionalGeneration.embed_multimodal()` (which just calls
`enc(raw)`) unaware that the actual computation happened out-of-process.

See `tools/intern_s2_sidecar_protocol.py` for the wire format and
`tools/intern_s2_sidecar.py` for the server this talks to.
"""

from __future__ import annotations

import socket
import threading

import torch
from tools.intern_s2_sidecar_protocol import (
    MsgType,
    ProtocolError,
    decode_tensor,
    encode_tensor,
    recv_frame,
    send_frame,
)
from torch import nn


class SidecarConnectionError(RuntimeError):
    """The sidecar is unreachable, closed the connection, or returned a
    protocol-level error. Raised uncaught out of forward() by design — a
    mid-request sidecar crash should fail that request loudly, the same way
    an OOM would, rather than silently degrade or auto-retry (see the
    process-isolation plan's "fail fast" decision)."""


class InternS2RPCEncoder(nn.Module):
    """Zero-parameter nn.Module that forwards `(B, T, V)` tensors to the
    Intern-S2 sidecar over a UDS and returns its `(B, Patches, Hidden)`
    reply.

    The connection is lazy: the sidecar may not be up yet when vLLM
    constructs `PrismForConditionalGeneration` (model class construction
    happens before the launcher's readiness gate passes in some call
    orders), so we connect on first `forward()` call, not in `__init__`.
    Reconnects on any failure — cheap for a UDS, and means a sidecar
    restart doesn't wedge a still-running vLLM process even though v1 has
    no auto-restart of the sidecar itself.
    """

    def __init__(self, hidden_dim: int, socket_path: str, timeout_s: float = 30.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        try:
            sock.connect(self.socket_path)
        except OSError as exc:
            sock.close()
            raise SidecarConnectionError(
                f"Could not connect to Intern-S2 sidecar at {self.socket_path!r}: {exc}"
            ) from exc
        return sock

    def _get_connection(self) -> socket.socket:
        if self._sock is None:
            self._sock = self._connect()
        return self._sock

    def _drop_connection(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One blocking call per embed_multimodal invocation, matching the
        # existing in-process call site's synchronous, non-batched-across-
        # requests shape. Serialized by `_lock` since a single persistent
        # connection can't interleave two in-flight requests.
        with self._lock:
            try:
                dtype_name, shape, raw = encode_tensor(x)
                sock = self._get_connection()
                send_frame(sock, MsgType.TENSOR, dtype=dtype_name, shape=shape, payload=raw)
                frame = recv_frame(sock)
            except (OSError, ProtocolError) as exc:
                self._drop_connection()
                raise SidecarConnectionError(f"Intern-S2 sidecar request failed: {exc}") from exc

        if frame.msg_type == MsgType.ERROR:
            raise SidecarConnectionError(
                f"Intern-S2 sidecar reported an error: {frame.payload.decode('utf-8', errors='replace')}"
            )
        if frame.msg_type != MsgType.TENSOR:
            raise SidecarConnectionError(
                f"Intern-S2 sidecar sent unexpected msg_type {frame.msg_type}"
            )
        if frame.dtype is None or frame.shape is None:
            raise SidecarConnectionError(
                "Intern-S2 sidecar sent a TENSOR frame without dtype/shape"
            )
        return decode_tensor(frame.dtype, frame.shape, frame.payload)


__all__ = ["InternS2RPCEncoder", "SidecarConnectionError"]

"""Standalone process that runs the Intern-S2 time-series encoder out of
vLLM's process, so vLLM (transformers<5) and Intern-S2's vendored config
(transformers>=5.2.0, for `RopeParameters`) never share a Python process.

Serves one Unix domain socket. Each connection is served sequentially: read
a frame, run it through the encoder if it's a tensor, write the reply,
repeat until the client disconnects. This matches the client's usage
pattern (one blocking, synchronous call per `embed_multimodal` invocation
from `src.vllm_plugin.processors.intern_s2_rpc_encoder.InternS2RPCEncoder`)
— no concurrency inside a connection, so no locking around the encoder call.

A per-request encode failure sends back an ERROR frame and keeps serving
(a single malformed/oversized request shouldn't take down a warm model).
An unrecoverable failure (e.g. the peer breaks the connection mid-write)
drops that connection and returns to accept() — the client sees a broken
pipe / closed connection and fails its in-flight request loudly, by design
(see the process-isolation plan's "fail fast" decision — no auto-restart,
no silent degrade in v1).

Usage:
    python tools/intern_s2_sidecar.py \\
        --encoder-type intern_s2 --model-name $HF_HOME/intern-s2-preview-timeseries \\
        --num-vars 1 --d-ts 2048 --max-ts-length 512 \\
        --socket-path /tmp/prism-intern-s2-$PBS_JOBID-$(hostname).sock

    # Stub mode (no real weights, no transformers>=5 dependency) — for local
    # testing of the server/protocol without Intern-S2's actual weights:
    python tools/intern_s2_sidecar.py --stub --stub-hidden-dim 64 \\
        --socket-path /tmp/test.sock
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.intern_s2_sidecar_protocol import (  # noqa: E402
    MsgType,
    ProtocolError,
    decode_tensor,
    encode_tensor,
    recv_frame,
    send_frame,
)

logger = logging.getLogger(__name__)

# How long the accept()/recv() loop blocks before re-checking the stop flag.
# Short enough that SIGTERM shutdown (see `serve()`) is responsive, long
# enough not to busy-loop.
_POLL_INTERVAL_S = 0.5


def default_socket_path() -> str:
    job_id = os.environ.get("PBS_JOBID", str(os.getpid()))
    return f"/tmp/prism-intern-s2-{job_id}-{socket.gethostname()}.sock"


class SidecarServer:
    """Accepts UDS connections and serves TENSOR/PING requests against a
    single encoder module. `forward_fn(encoder, tensor) -> tensor` matches
    the `(inner, hidden, forward_fn)` contract
    `TimeSeriesModalityProcessor.build_encoder()` already returns, so the
    same closure training/vLLM would have used in-process is reused here
    unchanged."""

    def __init__(self, encoder: Any, forward_fn: Any, socket_path: str) -> None:
        self.encoder = encoder
        self.forward_fn = forward_fn
        self.socket_path = socket_path
        self._server_sock: socket.socket | None = None
        self._stop = threading.Event()

    def bind(self) -> None:
        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        sock.listen(1)
        sock.settimeout(_POLL_INTERVAL_S)
        self._server_sock = sock
        logger.info("Intern-S2 sidecar listening on %s", self.socket_path)

    def serve_forever(self) -> None:
        if self._server_sock is None:
            raise RuntimeError("bind() must be called before serve_forever()")
        while not self._stop.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            try:
                self._serve_connection(conn)
            finally:
                conn.close()

    def _serve_connection(self, conn: socket.socket) -> None:
        conn.settimeout(_POLL_INTERVAL_S)
        while not self._stop.is_set():
            try:
                frame = recv_frame(conn)
            except TimeoutError:
                continue
            except ProtocolError:
                logger.info("Dropping connection: malformed frame or peer closed")
                return
            if frame.msg_type == MsgType.PING:
                send_frame(conn, MsgType.PONG)
            elif frame.msg_type == MsgType.TENSOR:
                self._handle_tensor(conn, frame)
            else:
                send_frame(
                    conn,
                    MsgType.ERROR,
                    payload=f"unexpected msg_type {frame.msg_type}".encode(),
                )

    def _handle_tensor(self, conn: socket.socket, frame) -> None:
        import torch

        try:
            tensor = decode_tensor(frame.dtype, frame.shape, frame.payload)
            with torch.no_grad():
                out = self.forward_fn(self.encoder, tensor)
            dtype_name, shape, raw = encode_tensor(out)
            send_frame(conn, MsgType.TENSOR, dtype=dtype_name, shape=shape, payload=raw)
        except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
            logger.exception("Intern-S2 sidecar failed to encode a request")
            send_frame(conn, MsgType.ERROR, payload=str(exc).encode("utf-8"))

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._server_sock is not None:
            self._server_sock.close()
            self._server_sock = None
        Path(self.socket_path).unlink(missing_ok=True)


def _build_real_encoder(args: argparse.Namespace) -> tuple[Any, Any]:
    # Lazy import: this is the whole point of the sidecar — TimeSeriesEncoder
    # pulls in Intern-S2's vendored transformers>=5.2.0-only config, which
    # must never be imported in vLLM's process.
    from src.encoders.time_series import TimeSeriesEncoder

    encoder = TimeSeriesEncoder(
        encoder_type=args.encoder_type,
        num_vars=args.num_vars,
        d_ts=args.d_ts,
        model_name=args.model_name,
        max_ts_length=args.max_ts_length,
        is_interleaved=False,
        intern_s2_sampling_rate=args.intern_s2_sampling_rate,
    )
    encoder.eval()

    def _forward(model: Any, x: Any) -> Any:
        return model(x)

    return encoder, _forward


class _StubEncoder:
    """Deterministic placeholder encoder for tests / manual protocol
    exercises that don't want a real Intern-S2 checkpoint or a
    transformers>=5.2.0 venv. Maps (B, T, V) -> (B, stub_num_tokens, hidden)
    by averaging over T and V and broadcasting — shape-correct, not
    semantically meaningful."""

    def __init__(self, hidden_dim: int, num_tokens: int) -> None:
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

    def eval(self) -> None:
        pass

    def __call__(self, x):
        import torch

        b = x.shape[0]
        pooled = x.mean(dim=(1, 2))  # (B,)
        out = pooled.view(b, 1, 1).expand(b, self.num_tokens, self.hidden_dim)
        return out.to(dtype=torch.float32).contiguous()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--socket-path", default=None, help="Defaults to PRISM_INTERN_S2_SOCKET or a computed path."
    )
    p.add_argument("--encoder-type", default="intern_s2", choices=["intern_s2", "intern_s2_397b"])
    p.add_argument(
        "--model-name", default=None, help="HF_HOME-local path to the extracted Intern-S2 weights."
    )
    p.add_argument("--num-vars", type=int, default=1)
    p.add_argument("--d-ts", type=int, default=None)
    p.add_argument("--max-ts-length", type=int, default=512)
    p.add_argument("--intern-s2-sampling-rate", type=float, default=1.0)
    p.add_argument(
        "--stub",
        action="store_true",
        help="Serve a deterministic placeholder encoder instead of loading real "
        "Intern-S2 weights. For local protocol/server testing only.",
    )
    p.add_argument("--stub-hidden-dim", type=int, default=64)
    p.add_argument("--stub-num-tokens", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.stub:
        encoder = _StubEncoder(args.stub_hidden_dim, args.stub_num_tokens)
        forward_fn = lambda model, x: model(x)  # noqa: E731
    else:
        if not args.model_name or args.d_ts is None:
            raise SystemExit("--model-name and --d-ts are required unless --stub is set")
        encoder, forward_fn = _build_real_encoder(args)

    socket_path = (
        args.socket_path or os.environ.get("PRISM_INTERN_S2_SOCKET") or default_socket_path()
    )
    server = SidecarServer(encoder, forward_fn, socket_path)
    server.bind()

    def _handle_sigterm(signum: int, frame: Any) -> None:
        logger.info("Received signal %d, shutting down", signum)
        server.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        server.serve_forever()
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

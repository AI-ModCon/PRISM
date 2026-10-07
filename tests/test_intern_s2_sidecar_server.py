"""Unit tests for tools/intern_s2_sidecar.py's SidecarServer against a stub
encoder over a real Unix domain socket on a temp path.

No real Intern-S2 weights, no transformers>=5.2.0 dependency — the point of
these tests is to prove the server/protocol plumbing (accept, serve, encode,
reply, clean shutdown) is correct in isolation from the actual model.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import torch
from tools.intern_s2_sidecar import SidecarServer, _StubEncoder
from tools.intern_s2_sidecar_protocol import (
    MsgType,
    encode_tensor,
    pack_frame,
    recv_frame,
    send_frame,
)


@pytest.fixture
def running_server(tmp_path):
    socket_path = str(tmp_path / "test.sock")
    encoder = _StubEncoder(hidden_dim=8, num_tokens=3)
    server = SidecarServer(encoder, lambda model, x: model(x), socket_path)
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, socket_path
    finally:
        server.stop()
        thread.join(timeout=5)
        server.close()


def _connect(socket_path: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5)
    # bind() may briefly precede the accept loop starting; retry a few
    # times rather than sleeping a fixed amount.
    deadline = time.monotonic() + 5
    while True:
        try:
            sock.connect(socket_path)
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def test_ping_pong(running_server):
    _server, socket_path = running_server
    sock = _connect(socket_path)
    try:
        send_frame(sock, MsgType.PING)
        frame = recv_frame(sock)
        assert frame.msg_type == MsgType.PONG
    finally:
        sock.close()


def test_tensor_round_trip_matches_stub_shape(running_server):
    _server, socket_path = running_server
    sock = _connect(socket_path)
    try:
        x = torch.zeros(2, 16, 1)
        dtype, shape, raw = encode_tensor(x)
        send_frame(sock, MsgType.TENSOR, dtype=dtype, shape=shape, payload=raw)
        frame = recv_frame(sock)
        assert frame.msg_type == MsgType.TENSOR
        assert frame.shape == (2, 3, 8)  # (B, stub_num_tokens, hidden_dim)
    finally:
        sock.close()


def test_multiple_sequential_requests_on_one_connection(running_server):
    _server, socket_path = running_server
    sock = _connect(socket_path)
    try:
        for _ in range(3):
            x = torch.ones(1, 4, 1)
            dtype, shape, raw = encode_tensor(x)
            send_frame(sock, MsgType.TENSOR, dtype=dtype, shape=shape, payload=raw)
            frame = recv_frame(sock)
            assert frame.msg_type == MsgType.TENSOR
    finally:
        sock.close()


def test_malformed_frame_drops_connection_without_crashing_server(running_server):
    server, socket_path = running_server
    sock = _connect(socket_path)
    try:
        sock.sendall(b"not a valid frame at all")
    finally:
        sock.close()

    # Server must still be up and servicing new connections afterward.
    sock2 = _connect(socket_path)
    try:
        send_frame(sock2, MsgType.PING)
        frame = recv_frame(sock2)
        assert frame.msg_type == MsgType.PONG
    finally:
        sock2.close()


def test_encoder_exception_returns_error_frame_not_crash(running_server, monkeypatch):
    server, socket_path = running_server

    def _raise(model, x):
        raise RuntimeError("synthetic encoder failure")

    server.forward_fn = _raise

    sock = _connect(socket_path)
    try:
        x = torch.zeros(1, 4, 1)
        dtype, shape, raw = encode_tensor(x)
        send_frame(sock, MsgType.TENSOR, dtype=dtype, shape=shape, payload=raw)
        frame = recv_frame(sock)
        assert frame.msg_type == MsgType.ERROR
        assert "synthetic encoder failure" in frame.payload.decode()
    finally:
        sock.close()

    # Server must still be alive for the next connection.
    server.forward_fn = lambda model, x: model(x)
    sock2 = _connect(socket_path)
    try:
        send_frame(sock2, MsgType.PING)
        frame = recv_frame(sock2)
        assert frame.msg_type == MsgType.PONG
    finally:
        sock2.close()


def test_unexpected_msg_type_from_client_returns_error(running_server):
    _server, socket_path = running_server
    sock = _connect(socket_path)
    try:
        sock.sendall(pack_frame(MsgType.PONG))
        frame = recv_frame(sock)
        assert frame.msg_type == MsgType.ERROR
    finally:
        sock.close()


def test_stop_and_close_removes_socket_file(tmp_path):
    socket_path = str(tmp_path / "cleanup.sock")
    encoder = _StubEncoder(hidden_dim=4, num_tokens=1)
    server = SidecarServer(encoder, lambda model, x: model(x), socket_path)
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    from pathlib import Path

    assert Path(socket_path).exists()
    server.stop()
    thread.join(timeout=5)
    server.close()
    assert not Path(socket_path).exists()

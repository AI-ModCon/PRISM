"""Unit tests for InternS2RPCEncoder round-tripping through a real (stub)
sidecar server over a Unix domain socket.

Proves the client half of the duck-typed (inner, hidden, forward_fn)
contract that TimeSeriesModalityProcessor.build_encoder() returns actually
works end to end against tools/intern_s2_sidecar.py's server, without any
real Intern-S2 weights or transformers>=5.2.0 dependency.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

# `src.vllm_plugin.processors.__init__` imports `time_series`, which imports
# vLLM at module scope, so importing the RPC encoder pulls vLLM in transitively
# even though nothing here uses it. vLLM is an Aurora/XPU build and is not
# installable on CI's platform, so without this guard the module raises
# ModuleNotFoundError at collection -- and a collection error makes pytest exit
# 2 having run zero tests, taking the whole `unit` and `coverage_gate` jobs down
# with it rather than just skipping this file. Same idiom as
# tests/test_vllm_ts_encoder.py:17.
pytest.importorskip("vllm")

from src.vllm_plugin.processors.intern_s2_rpc_encoder import (  # noqa: E402
    InternS2RPCEncoder,
    SidecarConnectionError,
)
from tools.intern_s2_sidecar import SidecarServer, _StubEncoder  # noqa: E402


@pytest.fixture
def running_server(tmp_path):
    socket_path = str(tmp_path / "rpc_client_test.sock")
    encoder = _StubEncoder(hidden_dim=8, num_tokens=3)
    server = SidecarServer(encoder, lambda model, x: model(x), socket_path)
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield socket_path
    finally:
        server.stop()
        thread.join(timeout=5)
        server.close()


def test_forward_round_trips_through_stub_server(running_server):
    client = InternS2RPCEncoder(hidden_dim=8, socket_path=running_server, timeout_s=5.0)
    x = torch.zeros(2, 16, 1)
    out = client.forward(x)
    assert out.shape == (2, 3, 8)


def test_forward_reuses_persistent_connection(running_server):
    client = InternS2RPCEncoder(hidden_dim=8, socket_path=running_server, timeout_s=5.0)
    x = torch.ones(1, 4, 1)
    client.forward(x)
    sock_after_first = client._sock
    client.forward(x)
    assert client._sock is sock_after_first


def test_forward_lazy_connects_not_at_init(tmp_path):
    # No server bound at this path — construction must not raise.
    client = InternS2RPCEncoder(
        hidden_dim=8, socket_path=str(tmp_path / "nonexistent.sock"), timeout_s=1.0
    )
    assert client._sock is None
    with pytest.raises(SidecarConnectionError):
        client.forward(torch.zeros(1, 4, 1))


def test_forward_raises_sidecar_connection_error_when_server_unreachable(tmp_path):
    client = InternS2RPCEncoder(
        hidden_dim=8, socket_path=str(tmp_path / "nope.sock"), timeout_s=1.0
    )
    with pytest.raises(SidecarConnectionError):
        client.forward(torch.zeros(1, 4, 1))


def test_forward_drops_connection_and_raises_on_server_shutdown(tmp_path):
    socket_path = str(tmp_path / "shutdown_test.sock")
    encoder = _StubEncoder(hidden_dim=8, num_tokens=3)
    server = SidecarServer(encoder, lambda model, x: model(x), socket_path)
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    client = InternS2RPCEncoder(hidden_dim=8, socket_path=socket_path, timeout_s=5.0)
    client.forward(torch.zeros(1, 4, 1))
    assert client._sock is not None

    server.stop()
    thread.join(timeout=5)
    server.close()
    time.sleep(0.1)

    with pytest.raises(SidecarConnectionError):
        client.forward(torch.zeros(1, 4, 1))
    assert client._sock is None


def test_forward_raises_on_encoder_error_from_server(running_server):
    client = InternS2RPCEncoder(hidden_dim=8, socket_path=running_server, timeout_s=5.0)
    # An unsupported dtype tensor should surface as a SidecarConnectionError
    # raised client-side before ever hitting the wire (encode_tensor itself
    # rejects it), proving errors propagate rather than silently degrading.
    with pytest.raises(SidecarConnectionError):
        client.forward(torch.zeros(1, 4, 1, dtype=torch.int64))

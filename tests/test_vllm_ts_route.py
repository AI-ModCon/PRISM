"""Login-node tests for VLLM-7: /v1/prism/ts schema + tensor decode.

No engine boot. Validates the pydantic surface and the base64-tensor
decode helper so a bad request doesn't crash a PBS-scheduled smoke job.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
import torch
from fastapi import HTTPException

vllm = pytest.importorskip("vllm")


def test_payload_round_trips_float32():
    from src.vllm_plugin.openai_schema import PrismTSPayload, _decode_tensor

    arr = np.arange(32 * 2, dtype=np.float32).reshape(32, 2)
    payload = PrismTSPayload(
        data_b64=base64.b64encode(arr.tobytes()).decode("ascii"),
        shape=[32, 2],
        dtype="float32",
    )
    out = _decode_tensor(payload)
    assert tuple(out.shape) == (32, 2)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.from_numpy(arr))


def test_payload_round_trips_float16():
    from src.vllm_plugin.openai_schema import PrismTSPayload, _decode_tensor

    arr = (np.arange(16, dtype=np.float32) * 0.25).astype(np.float16).reshape(8, 2)
    payload = PrismTSPayload(
        data_b64=base64.b64encode(arr.tobytes()).decode("ascii"),
        shape=[8, 2],
        dtype="float16",
    )
    out = _decode_tensor(payload)
    assert tuple(out.shape) == (8, 2)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, torch.from_numpy(arr))


def test_payload_round_trips_bfloat16():
    from src.vllm_plugin.openai_schema import PrismTSPayload, _decode_tensor

    # bfloat16 round-trip via raw uint16 bit pattern.
    src_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    raw = src_tensor.contiguous().view(torch.uint16).numpy().tobytes()
    payload = PrismTSPayload(
        data_b64=base64.b64encode(raw).decode("ascii"),
        shape=[2, 2],
        dtype="bfloat16",
    )
    out = _decode_tensor(payload)
    assert tuple(out.shape) == (2, 2)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, src_tensor)


def test_payload_rejects_unsupported_dtype():
    from src.vllm_plugin.openai_schema import PrismTSPayload, _decode_tensor

    payload = PrismTSPayload(
        data_b64="AAAA",
        shape=[2, 2],
        dtype="int8",
    )
    with pytest.raises(HTTPException) as exc_info:
        _decode_tensor(payload)
    assert exc_info.value.status_code == 400
    assert "Unsupported dtype" in str(exc_info.value.detail)


def test_payload_rejects_byte_length_mismatch():
    from src.vllm_plugin.openai_schema import PrismTSPayload, _decode_tensor

    # 64 bytes sent, but shape (8, 2) * float16 = 32 bytes expected.
    payload = PrismTSPayload(
        data_b64=base64.b64encode(b"\x00" * 64).decode("ascii"),
        shape=[8, 2],
        dtype="float16",
    )
    with pytest.raises(HTTPException) as exc_info:
        _decode_tensor(payload)
    assert exc_info.value.status_code == 400
    assert "decoded to 64 bytes" in str(exc_info.value.detail)


def test_payload_rejects_negative_shape():
    from src.vllm_plugin.openai_schema import PrismTSPayload, _decode_tensor

    payload = PrismTSPayload(
        data_b64=base64.b64encode(b"\x00" * 8).decode("ascii"),
        shape=[-1, 2],
        dtype="float32",
    )
    with pytest.raises(HTTPException) as exc_info:
        _decode_tensor(payload)
    assert exc_info.value.status_code == 400
    assert "positive" in str(exc_info.value.detail)


def test_payload_rejects_bad_base64():
    from src.vllm_plugin.openai_schema import PrismTSPayload, _decode_tensor

    payload = PrismTSPayload(
        data_b64="not!valid!base64!@#$",
        shape=[2, 2],
        dtype="float32",
    )
    with pytest.raises(HTTPException) as exc_info:
        _decode_tensor(payload)
    assert exc_info.value.status_code == 400
    assert "valid base64" in str(exc_info.value.detail)


def test_request_pydantic_required_fields():
    """PrismTSRequest must reject missing required fields."""
    from pydantic import ValidationError
    from src.vllm_plugin.openai_schema import PrismTSRequest

    with pytest.raises(ValidationError):
        PrismTSRequest(prompt="hello")  # missing model + time_series


def test_request_pydantic_defaults():
    from src.vllm_plugin.openai_schema import PrismTSPayload, PrismTSRequest

    req = PrismTSRequest(
        model="exported/synthetic",
        prompt="<time_series>p",
        time_series=PrismTSPayload(
            data_b64=base64.b64encode(b"\x00" * 16).decode("ascii"),
            shape=[2, 2],
            dtype="float32",
        ),
    )
    assert req.max_tokens == 64
    assert req.temperature == 0.0


def test_attach_prism_routes_adds_post_endpoint():
    """attach_prism_routes wires up /v1/prism/ts as a POST handler."""
    from fastapi import FastAPI
    from src.vllm_plugin.openai_schema import attach_prism_routes, prism_ts_handler

    app = FastAPI()
    attach_prism_routes(app)
    paths = {(r.path, tuple(sorted(r.methods))) for r in app.routes if hasattr(r, "methods")}
    assert ("/v1/prism/ts", ("POST",)) in paths
    # Handler is the shared module-level function.
    [route] = [r for r in app.routes if getattr(r, "path", None) == "/v1/prism/ts"]
    assert route.endpoint is prism_ts_handler


def test_attach_prism_routes_is_idempotent():
    """A second call on the same app must not register a duplicate route."""
    from fastapi import FastAPI
    from src.vllm_plugin.openai_schema import attach_prism_routes

    app = FastAPI()
    attach_prism_routes(app)
    attach_prism_routes(app)
    ts_routes = [r for r in app.routes if getattr(r, "path", None) == "/v1/prism/ts"]
    assert len(ts_routes) == 1


# --------------------------------------------------------------------------- #
# _collect_final: terminal-chunk finish_reason propagation
# --------------------------------------------------------------------------- #


class _FakeCompletionOutput:
    def __init__(self, text: str, finish_reason: str | None):
        self.text = text
        self.finish_reason = finish_reason


class _FakeRequestOutput:
    def __init__(self, text: str, finish_reason: str | None, *, has_outputs: bool = True):
        self.outputs = [_FakeCompletionOutput(text, finish_reason)] if has_outputs else []


async def _make_async_gen(chunks):
    for c in chunks:
        yield c


def _run(coro):
    """Tiny asyncio.run wrapper so each test gets its own event loop."""
    import asyncio

    return asyncio.run(coro)


def test_collect_final_returns_terminal_stop():
    """Stream of incremental chunks (finish_reason=None) ending in a 'stop'
    terminal chunk -> caller sees the terminal text + 'stop'."""
    from src.vllm_plugin.openai_schema import _collect_final

    chunks = [
        _FakeRequestOutput("he", None),
        _FakeRequestOutput("hello", None),
        _FakeRequestOutput("hello world", "stop"),
    ]
    text, reason = _run(_collect_final(_make_async_gen(chunks)))
    assert text == "hello world"
    assert reason == "stop"


def test_collect_final_returns_terminal_length():
    """max_tokens hit -> vLLM emits finish_reason='length' on terminal chunk."""
    from src.vllm_plugin.openai_schema import _collect_final

    chunks = [
        _FakeRequestOutput("the cat sat on", None),
        _FakeRequestOutput("the cat sat on the", "length"),
    ]
    text, reason = _run(_collect_final(_make_async_gen(chunks)))
    assert text == "the cat sat on the"
    assert reason == "length"


def test_collect_final_returns_none_when_no_terminal_chunk():
    """If the stream ends without a terminal output (engine bug / shutdown race)
    _collect_final must return finish_reason=None, NOT fabricate 'stop'.
    The handler is responsible for the user-facing fallback."""
    from src.vllm_plugin.openai_schema import _collect_final

    chunks = [
        _FakeRequestOutput("partial", None),
        _FakeRequestOutput("partial text", None),
    ]
    text, reason = _run(_collect_final(_make_async_gen(chunks)))
    assert text == "partial text"
    assert reason is None


def test_collect_final_handles_empty_stream():
    """Generator that yields nothing -> ('', None)."""
    from src.vllm_plugin.openai_schema import _collect_final

    text, reason = _run(_collect_final(_make_async_gen([])))
    assert text == ""
    assert reason is None


def test_collect_final_skips_chunks_with_no_outputs():
    """Some engine streams yield bookkeeping chunks with outputs=[].
    Those must not clobber the last real finish_reason."""
    from src.vllm_plugin.openai_schema import _collect_final

    chunks = [
        _FakeRequestOutput("done", "stop"),
        _FakeRequestOutput("", None, has_outputs=False),
    ]
    text, reason = _run(_collect_final(_make_async_gen(chunks)))
    assert text == "done"
    assert reason == "stop"


# --------------------------------------------------------------------------- #
# prism_ts_handler: finish_reason None -> 'stop' fallback
# --------------------------------------------------------------------------- #


def _build_handler_request(monkeypatch, *, chunks, max_tokens: int = 8):
    """Construct a Request + body wired to a stub engine_client whose
    `generate` returns the given chunks."""
    from fastapi import FastAPI, Request
    from src.vllm_plugin import openai_schema as mod

    class _StubEngineClient:
        def __init__(self):
            self.generate_called_with = None

        def generate(self, *, prompt, sampling_params, request_id):
            self.generate_called_with = (prompt, sampling_params, request_id)
            return _make_async_gen(chunks)

    app = FastAPI()
    engine = _StubEngineClient()
    app.state.engine_client = engine

    # Avoid importing real vllm.SamplingParams just to get a plain holder.
    class _SP:
        def __init__(self, **kw):
            self.kw = kw

    monkeypatch.setattr("vllm.SamplingParams", _SP, raising=False)

    scope = {
        "type": "http",
        "app": app,
        "headers": [],
        "method": "POST",
        "path": "/v1/prism/ts",
    }
    request = Request(scope)

    body = mod.PrismTSRequest(
        model="exported/synthetic",
        prompt="<time_series>forecast:",
        time_series=mod.PrismTSPayload(
            data_b64=base64.b64encode(np.zeros((2, 2), dtype=np.float32).tobytes()).decode("ascii"),
            shape=[2, 2],
            dtype="float32",
        ),
        max_tokens=max_tokens,
    )
    return request, body, engine


def test_handler_propagates_length_finish_reason(monkeypatch):
    """max_tokens hit -> response finish_reason='length' (not 'stop')."""
    import json

    from src.vllm_plugin.openai_schema import prism_ts_handler

    chunks = [
        _FakeRequestOutput("abcdefgh", "length"),
    ]
    request, body, _engine = _build_handler_request(monkeypatch, chunks=chunks)
    resp = _run(prism_ts_handler(request, body))
    payload = json.loads(resp.body)
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["choices"][0]["text"] == "abcdefgh"


def test_handler_defaults_none_finish_reason_to_stop(monkeypatch, caplog):
    """Stream ends without a terminal finish_reason -> handler logs a warning
    and falls back to 'stop' so strict OpenAI clients accept the response."""
    import json
    import logging

    from src.vllm_plugin.openai_schema import prism_ts_handler

    chunks = [
        _FakeRequestOutput("partial", None),
    ]
    request, body, _engine = _build_handler_request(monkeypatch, chunks=chunks)
    with caplog.at_level(logging.WARNING, logger="src.vllm_plugin.openai_schema"):
        resp = _run(prism_ts_handler(request, body))
    payload = json.loads(resp.body)
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["text"] == "partial"
    assert any("without terminal finish_reason" in r.message for r in caplog.records)

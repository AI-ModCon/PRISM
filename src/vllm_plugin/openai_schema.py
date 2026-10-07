"""Custom OpenAI-style routes for PRISM-specific multimodal inputs.

The OpenAI Chat Completions API has no native field for tensor payloads.
PRISM's `time_series` modality (and future modalities — `geometry`, `dna`)
needs to ship raw multi-dimensional float arrays from the client to the
engine. Rather than overloading `image_url` with custom base64 conventions
(which leaks into lmms-eval-style harnesses that expect images there),
this module adds a new `/v1/prism/ts` endpoint accepting base64-packed
tensors in a JSON envelope.

vLLM's `build_app(args)` returns the FastAPI app; we attach our routes
to it after init_app_state has populated `app.state.engine_client`. See
`tools/vllm_serve.py` for the monkey-patch entry point.

Request schema (POST /v1/prism/ts):

    {
      "model": "<served model id>",                # required (echoed back)
      "prompt": "<time_series>forecast:",          # required
      "time_series": {
        "data_b64": "<base64-encoded raw bytes>",  # required
        "shape": [T, V],                            # required, 2-D (T,V) or 3-D (B,T,V)
        "dtype": "float32"                          # optional, default float32
      },
      "max_tokens": 64,                             # optional, default 64
      "temperature": 0.0                            # optional, default 0.0 (greedy)
    }

Response shape (matches OpenAI Completions):

    {
      "id": "<request id>",
      "model": "<served model id>",
      "object": "prism.ts.completion",
      "choices": [{"index": 0, "text": "<completion>", "finish_reason": "<stop|length|...>"}]
    }
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


_SUPPORTED_DTYPES = {
    "float32": ("f4", 4),
    "float16": ("f2", 2),
    "bfloat16": ("bf16", 2),  # numpy has no native bfloat16; we'll round-trip via torch
}


class PrismTSPayload(BaseModel):
    """The raw tensor payload inside a /v1/prism/ts request."""

    data_b64: str = Field(
        ...,
        description="Base64-encoded raw bytes of the time-series tensor "
        "(C-contiguous, dtype-specified). Length must be prod(shape) * sizeof(dtype).",
    )
    shape: list[int] = Field(
        ...,
        description="Tensor shape. Either (T, V) for a single item or (B, T, V) "
        "for a batch.",
        min_length=2,
        max_length=3,
    )
    dtype: str = Field(
        default="float32",
        description="One of: float32, float16, bfloat16. Default float32.",
    )


class PrismTSRequest(BaseModel):
    """POST /v1/prism/ts body."""

    model: str = Field(..., description="Served model id (echoed back in response).")
    prompt: str = Field(
        ...,
        description="Text prompt with one <time_series> placeholder per ts item.",
    )
    time_series: PrismTSPayload
    max_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


def _decode_tensor(payload: PrismTSPayload):
    """base64 + shape + dtype -> torch.Tensor."""
    import numpy as np
    import torch

    if payload.dtype not in _SUPPORTED_DTYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported dtype {payload.dtype!r}; "
                f"expected one of {sorted(_SUPPORTED_DTYPES)}."
            ),
        )

    try:
        raw = base64.b64decode(payload.data_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"data_b64 is not valid base64: {exc}",
        ) from exc

    n_elem = 1
    for d in payload.shape:
        if d <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"shape entries must be positive; got {payload.shape}",
            )
        n_elem *= d
    _, elem_size = _SUPPORTED_DTYPES[payload.dtype]
    expected_bytes = n_elem * elem_size
    if len(raw) != expected_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"data_b64 decoded to {len(raw)} bytes, but shape {payload.shape} "
                f"+ dtype {payload.dtype} expects {expected_bytes} bytes "
                f"({n_elem} elements * {elem_size} bytes each)."
            ),
        )

    if payload.dtype == "bfloat16":
        # numpy has no bfloat16 view, so allocate a uint16 buffer and
        # reinterpret as torch.bfloat16. Safe because both occupy 2 bytes.
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(payload.shape).copy()
        return torch.from_numpy(arr).view(torch.bfloat16)
    np_dtype, _ = _SUPPORTED_DTYPES[payload.dtype]
    arr = np.frombuffer(raw, dtype=np.dtype(np_dtype)).reshape(payload.shape).copy()
    return torch.from_numpy(arr)


async def _collect_final(generator) -> tuple[str, str | None]:
    """Consume an engine_client.generate AsyncGenerator and return the final
    (text, finish_reason).

    vLLM emits incremental RequestOutput chunks with finish_reason=None until
    the terminal chunk, which carries the real finish_reason
    ({"stop", "length", "abort", ...}). We track the last-seen value, so the
    returned reason is the terminal one when the stream ends normally. Returns
    finish_reason=None only if the stream ends without ever producing a chunk
    with outputs — caller decides how to surface that."""
    final_text = ""
    final_reason: str | None = None
    async for out in generator:
        if out.outputs:
            final_text = out.outputs[0].text
            final_reason = out.outputs[0].finish_reason
    return final_text, final_reason


async def prism_ts_handler(request: Request, body: PrismTSRequest) -> JSONResponse:
    """POST /v1/prism/ts handler. Submits the ts tensor + prompt to the
    engine and returns an OpenAI-style completion JSON."""
    from vllm import SamplingParams

    engine_client = request.app.state.engine_client
    if engine_client is None:
        raise HTTPException(
            status_code=503,
            detail="engine_client is not initialized — server still booting.",
        )

    tensor = _decode_tensor(body.time_series)

    sampling = SamplingParams(
        max_tokens=body.max_tokens, temperature=body.temperature
    )
    request_id = f"prism-ts-{uuid.uuid4().hex}"

    prompt_dict: dict[str, Any] = {
        "prompt": body.prompt,
        "multi_modal_data": {"time_series": tensor},
    }

    # engine_client.generate is an AsyncGenerator that yields incremental
    # RequestOutput objects. We collect the final text and return synchronously
    # — streaming /v1/prism/ts is out of scope for this PR (no
    # OpenAI-server streaming conventions to mirror for a custom modality).
    #
    # If the client disconnects mid-generation, FastAPI raises CancelledError
    # here; abort the engine-side request so we don't keep occupying KV-cache
    # budget. Mirrors the pattern used by upstream OpenAI handlers.
    generator = engine_client.generate(
        prompt=prompt_dict,
        sampling_params=sampling,
        request_id=request_id,
    )
    try:
        final_text, finish_reason = await _collect_final(generator)
    except asyncio.CancelledError:
        try:
            await engine_client.abort(request_id)
        except Exception:  # noqa: BLE001
            logger.exception("prism_ts_handler: abort(%s) failed", request_id)
        raise

    # vLLM emits a terminal CompletionOutput with finish_reason set to one of
    # {"stop", "length", "abort", ...} — "length" when max_tokens is hit, "stop"
    # on EOS/stop string. Reaching None here means the stream ended without a
    # terminal output (engine bug or shutdown race). Fall back to "stop" so
    # strict OpenAI clients with Literal[...] finish_reason schemas don't reject
    # the response.
    if finish_reason is None:
        logger.warning(
            "prism_ts_handler: stream for %s ended without terminal finish_reason; "
            "defaulting to 'stop'",
            request_id,
        )
        finish_reason = "stop"

    return JSONResponse(
        {
            "id": request_id,
            "model": body.model,
            "object": "prism.ts.completion",
            "choices": [
                {
                    "index": 0,
                    "text": final_text,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


_PRISM_TS_PATH = "/v1/prism/ts"


def attach_prism_routes(app: FastAPI) -> None:
    """Add PRISM-custom routes to an existing FastAPI app.

    Call AFTER vllm's `build_app(args)` (which composes the OpenAI-compatible
    routes) so our routes share the same `app.state.engine_client`. The
    monkey-patch hook lives in `tools/vllm_serve.py`.

    Idempotent: a second call on the same app is a no-op, so test code and
    accidental double-patches don't end up with two route entries for
    `/v1/prism/ts`.
    """
    for route in app.routes:
        if getattr(route, "path", None) == _PRISM_TS_PATH:
            return
    # Use add_api_route rather than @app.post so a single function reference
    # is shared between this module-level handler and tests that may import
    # `prism_ts_handler` directly.
    app.add_api_route(
        _PRISM_TS_PATH,
        prism_ts_handler,
        methods=["POST"],
        summary="PRISM time-series completion",
        response_class=JSONResponse,
    )


__all__ = [
    "PrismTSPayload",
    "PrismTSRequest",
    "attach_prism_routes",
    "prism_ts_handler",
]

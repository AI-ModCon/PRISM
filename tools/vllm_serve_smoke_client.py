"""Verify the vLLM server booted and exposes the PRISM model via /v1/models.

Companion to tools/_vllm_serve_smoke_runner.sh.

We deliberately do NOT issue an image chat-completion request: OLMo's
tokenizer ships no chat template (transformers >= 4.44 rejects missing
templates with HTTP 400), and the OpenAI Completions endpoint doesn't
accept multimodal payloads in vLLM 0.15. The smoke gate's purpose is to
prove plugin auto-registration worked in spawn workers — and the proof
is that the server's `/v1/models` returns the PRISM model id (which can
only happen if PrismForConditionalGeneration was in ModelRegistry when
the engine booted).

A full chat-template-aware request belongs in the eval / parity scripts,
not the smoke gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

# Aurora compute nodes set http_proxy=proxy.alcf.anl.gov which can't reach
# localhost. Strip proxy env vars before any urllib import-time discovery so
# `urlopen` talks to 127.0.0.1 directly.
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_v, None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--vllm-model",
        required=False,
        default=None,
        help="Expected model id; asserts /v1/models lists this id.",
    )
    args = p.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(
            f"[client] HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}",
            file=sys.stderr,
        )
        raise SystemExit(f"FAILED: HTTP {e.code}") from e

    models = body.get("data", [])
    if not models:
        raise SystemExit(f"FAILED: /v1/models returned empty data: {body}")

    listed_ids = [str(m.get("id")) for m in models]
    print(f"[client] /v1/models returned: {listed_ids}")

    if args.vllm_model:
        if args.vllm_model not in listed_ids:
            raise SystemExit(
                f"FAILED: expected model id {args.vllm_model!r} not in {listed_ids}"
            )
        print(f"[client] OK: {args.vllm_model!r} is served")
    else:
        print(f"[client] OK: server has {len(models)} model(s) loaded")


if __name__ == "__main__":
    main()

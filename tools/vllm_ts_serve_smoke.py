"""POST a base64 time-series tensor to /v1/prism/ts and assert a completion.

Companion to tools/_vllm_serve_smoke_runner.sh — used after the server is
verified up via /v1/models to exercise the custom PRISM route.

Builds a small random ts tensor in-process, base64-encodes its raw bytes,
posts to /v1/prism/ts, asserts the response is well-formed and the
generated text is non-empty.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

# Aurora compute nodes set http_proxy=proxy.alcf.anl.gov which can't reach
# localhost. Strip proxy env before urllib import-time discovery.
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_v, None)


def _resolve_model_id(port: int) -> str:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/models", timeout=15
    ) as r:
        body = json.loads(r.read().decode("utf-8"))
    data = body.get("data", [])
    if not data:
        raise SystemExit("FAILED: /v1/models returned empty data")
    return str(data[0]["id"])


def _build_ts_payload(*, t: int, v: int, dtype: str, seed: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    if dtype == "float32":
        arr = rng.standard_normal(size=(t, v)).astype(np.float32)
    elif dtype == "float16":
        arr = rng.standard_normal(size=(t, v)).astype(np.float16)
    else:
        raise SystemExit(f"unsupported test dtype: {dtype}")
    raw = arr.tobytes()
    return {
        "data_b64": base64.b64encode(raw).decode("ascii"),
        "shape": [t, v],
        "dtype": dtype,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=None,
                   help="Served model id; defaults to first from /v1/models.")
    p.add_argument("--prompt", default="<time_series>forecast:",
                   help="Prompt with one <time_series> placeholder.")
    p.add_argument("--t", type=int, default=32, help="ts T axis (timesteps).")
    p.add_argument("--v", type=int, default=1, help="ts V axis (variates).")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model_id = args.model or _resolve_model_id(args.port)
    print(f"[ts-route] model={model_id!r} prompt={args.prompt!r}")

    payload = {
        "model": model_id,
        "prompt": args.prompt,
        "time_series": _build_ts_payload(
            t=args.t, v=args.v, dtype=args.dtype, seed=args.seed
        ),
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/prism/ts",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(
            f"[ts-route] HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}",
            file=sys.stderr,
        )
        raise SystemExit(f"FAILED: HTTP {e.code}") from e

    if resp.get("object") != "prism.ts.completion":
        raise SystemExit(f"FAILED: unexpected response object: {resp}")
    choices = resp.get("choices", [])
    if not choices or not choices[0].get("text"):
        raise SystemExit(f"FAILED: empty completion: {resp}")

    text = choices[0]["text"]
    print(f"[ts-route] >>> {text!r}")
    print("[ts-route] PASSED")


if __name__ == "__main__":
    main()

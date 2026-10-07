"""Throughput sanity check for the PRISM vLLM backend.

Builds N requests by tiling a small set of (image, prompt) pairs, runs them
through one llm.generate() call, and reports wall clock + tokens/sec.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import src.vllm_plugin  # noqa: E402
from PIL import Image  # noqa: E402

src.vllm_plugin.register()

from vllm import LLM, SamplingParams  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()

    images = sorted(
        args.image_dir / f
        for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    )
    if not images:
        raise SystemExit("No images")

    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    pil = [Image.open(p).convert("RGB") for p in images]
    requests = [
        {
            "prompt": "<image> The image shows",
            "multi_modal_data": {"image": pil[i % len(pil)]},
        }
        for i in range(args.n)
    ]

    # Warmup pass (single request) so JIT / encoder is hot.
    llm.generate(requests[:1], sampling)

    t0 = time.time()
    outs = llm.generate(requests, sampling)
    dt = time.time() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    print("\n=== throughput ===")
    print(f"n_requests={args.n}, wall={dt:.2f}s")
    print(f"total_out_tokens={total_out}, tok/s={total_out/dt:.1f}")
    print(f"req/s={args.n/dt:.2f}")


if __name__ == "__main__":
    main()

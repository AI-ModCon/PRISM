"""Image+text eval against PRISM via vLLM.

Uses vLLM's batched PagedAttention engine to run image+text completions
against an exported PRISM checkpoint.

Usage:
    module load frameworks
    python tools/vllm_eval.py \\
        --model exported/prism-olmo1b-image \\
        --image test_images/ \\
        --prompt "The image shows" \\
        --limit 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src.* imports resolve when running from the repo root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import src.vllm_plugin  # noqa: E402
from PIL import Image  # noqa: E402

src.vllm_plugin.register()

from vllm import LLM, SamplingParams  # noqa: E402


def collect_images(path: Path, limit: int) -> list[Path]:
    if path.is_dir():
        files = sorted(
            path / f
            for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        )
        return files[:limit]
    if path.is_file():
        return [path]
    raise FileNotFoundError(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path,
                   help="Exported PRISM dir (output of checkpoint_export.py)")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--prompt", default="The image shows")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    images = collect_images(args.image, args.limit)
    if not images:
        raise SystemExit("No images found")

    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        dtype=args.dtype,
        # PRISM uses a single image per prompt today.
        limit_mm_per_prompt={"image": 1},
        # torch.compile is not viable on Aurora XPU (CLAUDE.md, Feb 2026).
        enforce_eager=True,
    )

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    prompt_text = f"<image> {args.prompt}"

    requests = [
        {
            "prompt": prompt_text,
            "multi_modal_data": {"image": Image.open(p).convert("RGB")},
        }
        for p in images
    ]

    outputs = llm.generate(requests, sampling)

    for img_path, out in zip(images, outputs, strict=False):
        text = out.outputs[0].text
        print(f"=== {img_path.name} ===")
        print(f"  Prediction: {text}\n")


if __name__ == "__main__":
    main()

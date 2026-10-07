#!/usr/bin/env python3
"""Download a materials text backbone into a compute-node-visible HF cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_cache_dir() -> Path:
    configured = os.environ.get("MATERIALS_HF_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured)
    return Path(f"/eagle/projects/ModCon/{os.environ.get('USER', 'unknown')}/huggingface/hub")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a Hugging Face model in the shared cache used by materials jobs."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to stage the model") from exc

    # ALCF login nodes reach Hugging Face through this proxy. Respect any
    # site/user-specific value that is already set.
    os.environ.setdefault("HTTP_PROXY", "http://proxy.alcf.anl.gov:3128")
    os.environ.setdefault("HTTPS_PROXY", "http://proxy.alcf.anl.gov:3128")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    print(f"Staged {args.model} at {snapshot}")
    print(f"HF_HUB_CACHE={args.cache_dir}")


if __name__ == "__main__":
    main()

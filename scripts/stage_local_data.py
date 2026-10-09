#!/usr/bin/env python3
"""Stage non-WebDataset local data (JSONL/CSV/Arrow) to compute-node /tmp.

The launcher's existing stage_shards.py covers WebDataset tar shards; this
script covers the per-modality sweep cells that load raw JSONL/CSV/Arrow
files via the local-dir dispatch fix. Source directories are read from
the cell preset's dataset_overrides (local_path field). Destination is
LOCAL_DATA_STAGE_DIR (default /tmp/local_data).

Usage (called once per node, rank-0):
    python scripts/stage_local_data.py \\
        --sources /flare/ModCon/ngetty/data/zone_a/ts_qa/align_256 \\
                  /flare/ModCon/sandeep/PRISM/data/zone_a/ts_instruction \\
        --local-dir /tmp/local_data

Sources are mirrored as basename subdirs:
    /tmp/local_data/align_256/train.jsonl
    /tmp/local_data/ts_instruction/TS_Dataset.jsonl
    ...

The dispatch fix's local_path override is set per-cell in the preset yaml.
This script just copies the bytes; the consumer redirects to /tmp.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


def stage_one(src: Path, dst: Path) -> tuple[int, float]:
    if not src.exists():
        print(f"WARN: source {src} does not exist; skipping", file=sys.stderr)
        return 0, 0.0
    dst.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_bytes = 0
    t0 = time.time()
    for root, _, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            sp = Path(root) / f
            dp = dst / rel / f
            if dp.exists() and dp.stat().st_size == sp.stat().st_size:
                continue  # already staged
            shutil.copy2(sp, dp)
            n_files += 1
            n_bytes += sp.stat().st_size
    elapsed = time.time() - t0
    return n_files, elapsed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True,
                   help="Source directory paths to stage")
    p.add_argument("--local-dir", default="/tmp/local_data",
                   help="Destination root on compute node")
    args = p.parse_args()

    local_root = Path(args.local_dir)
    total_files = 0
    total_time = 0.0
    for s in args.sources:
        src = Path(s)
        dst = local_root / src.name
        print(f"Staging {src} → {dst}")
        nf, dt = stage_one(src, dst)
        total_files += nf
        total_time += dt
        print(f"  {nf} files in {dt:.1f}s")

    print(f"Total: {total_files} files in {total_time:.1f}s → {local_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit source composition of a globally shuffled WebDataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--global-batch-size", type=int, default=960)
    parser.add_argument("--window-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def manifest_shards(dataset_dir: Path) -> list[Path]:
    with open(dataset_dir / "manifest.json") as f:
        manifest = json.load(f)
    shards = manifest.get("shards", [])
    if isinstance(shards, list):
        names = [s["name"] if isinstance(s, dict) else str(s) for s in shards]
    else:
        names = [f"shard-{i:06d}.tar" for i in range(int(manifest["num_shards"]))]
    return [dataset_dir / "shards" / name for name in names]


def iter_sources(dataset_dir: Path):
    for shard_path in manifest_shards(dataset_dir):
        with tarfile.open(shard_path, "r:*") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                try:
                    metadata = json.loads(fh.read().decode("utf-8"))
                except Exception:
                    metadata = {}
                provenance = metadata.get("_global_shuffle", {})
                yield (
                    provenance.get("source_group", "unknown"),
                    provenance.get("source_dataset", "unknown"),
                )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    window_samples = args.global_batch_size * args.window_steps
    max_samples = args.global_batch_size * args.max_steps if args.max_steps else 0
    out_csv = args.out_csv or str(dataset_dir / "audit_source_windows.csv")

    rows = []
    counter: Counter[str] = Counter()
    total = 0
    window_index = 0

    def flush() -> None:
        nonlocal counter, window_index
        if not counter:
            return
        n = sum(counter.values())
        step_start = window_index * args.window_steps
        step_end = step_start + args.window_steps
        row = {
            "window": window_index,
            "step_start": step_start,
            "step_end": step_end,
            "samples": n,
        }
        for source, count in counter.most_common(args.top_k):
            row[f"{source}_frac"] = count / n
            row[f"{source}_count"] = count
        rows.append(row)
        print(
            f"window={window_index:03d} steps={step_start:05d}-{step_end:05d} "
            f"samples={n} top="
            + ", ".join(f"{s}:{c / n:.3f}" for s, c in counter.most_common(6))
        )
        counter = Counter()
        window_index += 1

    for group, dataset in iter_sources(dataset_dir):
        counter[f"{group}/{dataset}"] += 1
        total += 1
        if total % window_samples == 0:
            flush()
        if max_samples and total >= max_samples:
            break
    flush()

    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["window", "step_start", "step_end", "samples"]
    fieldnames = preferred + [f for f in fieldnames if f not in preferred]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

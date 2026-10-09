#!/usr/bin/env python3
"""Stage dataset-group shards as a node-local multi-dataset mirror."""

import argparse
import glob
import json
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml


def _datasets(config, groups):
    presets = config.get("presets", {})
    if groups in presets:
        preset = presets[groups]
        group_names = preset["groups"]
        weight_overrides = preset.get("weight_overrides", {})
        proportion_overrides = preset.get("proportion_overrides", {})
    else:
        group_names = [g.strip() for g in groups.split(",") if g.strip()]
        weight_overrides = {}
        proportion_overrides = {}

    out = []
    for group_name in group_names:
        for name, info in config["groups"][group_name]["datasets"].items():
            weight = weight_overrides.get(name, info.get("weight", 1.0))
            if info.get("skip") or weight <= 0:
                continue
            item = dict(info)
            item["name"] = name
            item["group"] = group_name
            item["weight"] = weight
            item["proportion"] = proportion_overrides.get(
                name, info.get("proportion", 1.0)
            )
            out.append(item)
    return out


def _manifest_shards(base_dir, fallback_samples, max_train_shards):
    manifest_path = os.path.join(base_dir, "manifest.json")
    shards_dir = os.path.join(base_dir, "shards")
    shard_rows = None

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        shards = manifest.get("shards")
        if isinstance(shards, list):
            if shards and isinstance(shards[0], dict):
                shard_rows = [
                    (s["name"], int(s.get("samples", fallback_samples)))
                    for s in shards
                ]
            else:
                shard_rows = [(str(name), fallback_samples) for name in shards]

            if (
                max_train_shards
                and len(shard_rows) > max_train_shards
                and "num_train_shards" not in manifest
            ):
                shard_rows = shard_rows[:max_train_shards]
        else:
            fallback_samples = int(manifest.get("samples_per_shard", fallback_samples))

    if shard_rows is None:
        shard_rows = [
            (os.path.basename(path), fallback_samples)
            for path in sorted(glob.glob(os.path.join(shards_dir, "*.tar")))
        ]

    out = []
    for name, samples in shard_rows:
        src = os.path.join(shards_dir, name)
        if os.path.exists(src):
            out.append({"name": name, "samples": samples, "src": src})
    return out


def _copy_one(entry, local_dir):
    dataset, shard = entry
    dst_dir = os.path.join(local_dir, dataset["path"], "shards")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, shard["name"])
    if not os.path.exists(dst):
        shutil.copy2(shard["src"], dst)
    return dataset["name"], dataset["path"], shard["name"], shard["samples"]


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "groups", "dataset-root", "local-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--max-shards-per-node", type=int, default=0)
    parser.add_argument("--max-shards-per-dataset-per-node", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    entries = []
    for index, dataset in enumerate(_datasets(config, args.groups)):
        base_dir = os.path.join(args.dataset_root, dataset["path"])
        fallback = max(
            1, int(dataset.get("samples", 0)) // max(1, int(dataset.get("shards", 1)))
        )
        shards = _manifest_shards(base_dir, fallback, int(dataset.get("shards", 0)))
        keep = max(1, int(len(shards) * float(dataset.get("proportion", 1.0))))
        shards = shards[:keep]

        ds_seed = args.seed + index * 1009 + sum(ord(c) for c in dataset["name"])
        random.Random(ds_seed).shuffle(shards)
        assigned = [
            shard
            for shard_index, shard in enumerate(shards)
            if shard_index % args.num_nodes == args.node_rank
        ]
        if args.max_shards_per_dataset_per_node > 0:
            assigned = assigned[: args.max_shards_per_dataset_per_node]

        for shard in assigned:
            entries.append((dataset, shard))

    random.Random(args.seed + args.node_rank).shuffle(entries)
    if args.max_shards_per_node > 0:
        entries = entries[: args.max_shards_per_node]

    os.makedirs(args.local_dir, exist_ok=True)
    max_workers = max(1, min(args.workers, len(entries)))
    staged_by_dataset = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_copy_one, entry, args.local_dir) for entry in entries]
        for future in as_completed(futures):
            dataset_name, dataset_path, shard_name, samples = future.result()
            staged_by_dataset[(dataset_name, dataset_path)].append(
                {"name": shard_name, "samples": samples}
            )

    summary = []
    for (dataset_name, dataset_path), shards in sorted(staged_by_dataset.items()):
        shards.sort(key=lambda item: item["name"])
        dataset_dir = os.path.join(args.local_dir, dataset_path)
        manifest = {
            "dataset": dataset_name,
            "node_rank": args.node_rank,
            "num_nodes": args.num_nodes,
            "num_train_shards": len(shards),
            "num_shards": len(shards),
            "total_written": sum(item["samples"] for item in shards),
            "shards": shards,
        }
        with open(os.path.join(dataset_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        summary.append(
            {
                "dataset": dataset_name,
                "path": dataset_path,
                "shards": len(shards),
                "samples": manifest["total_written"],
            }
        )

    manifest = {
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "local_dir": args.local_dir,
        "datasets": summary,
        "total_shards": sum(item["shards"] for item in summary),
        "total_samples_estimate": sum(item["samples"] for item in summary),
    }
    with open(os.path.join(args.local_dir, "staging_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Staged {manifest['total_shards']} shards across {len(summary)} datasets "
        f"to {args.local_dir} ({max_workers} workers)"
    )


if __name__ == "__main__":
    main()

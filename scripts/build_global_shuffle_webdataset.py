#!/usr/bin/env python3
"""Build a pooled, globally shuffled WebDataset from PRISM source datasets.

The output is a normal WebDataset directory that can be consumed through
PRISM's existing MultiWebDataset loader:

    output_dir/
      manifest.json
      shards/shard-000000.tar
      shards/shard-000001.tar
      ...

The shuffle is deterministic. Each input sample gets a random key from
blake2b(seed, source_dataset, source_shard, source_key). Samples are bucketed
by the high bits of that key, each bucket is sorted in memory, and buckets are
written in key order. This is an external sort over sample payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import struct
import tarfile
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import webdataset as wds
except ImportError as exc:  # pragma: no cover - runtime guard for PBS jobs
    raise SystemExit("webdataset is required in the active Python environment") from exc


LOG = logging.getLogger("global_shuffle_builder")
LEN_STRUCT = struct.Struct("<I")
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif"}


@dataclass(frozen=True)
class SourceShard:
    group: str
    dataset: str
    shard_index: int
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, help="PRISM dataset YAML")
    parser.add_argument("--groups", default="vlm_diverse_unweighted")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tmp-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--bucket-bits", type=int, default=13)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maxcount", type=int, default=5000)
    parser.add_argument("--maxsize", type=float, default=3_000_000_000.0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--max-source-shards", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-bucket", action="store_true")
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--log-every", type=int, default=100_000)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_groups(config: dict[str, Any], groups: str) -> list[str]:
    if groups in config.get("presets", {}):
        return list(config["presets"][groups]["groups"])
    return [g.strip() for g in groups.split(",") if g.strip()]


def source_root(config: dict[str, Any], override: str | None) -> str:
    root = (
        override
        or config.get("dastr", {}).get("mount_base")
        or config.get("daos", {}).get("mount_base")
    )
    if not root:
        raise ValueError("--source-root is required when config has no mount_base")
    return os.path.abspath(os.path.expandvars(root))


def shard_names_from_manifest(
    manifest: dict[str, Any],
    dataset_name: str,
    config_shards: int,
) -> list[str] | None:
    if isinstance(manifest.get("shards"), list):
        names = [s["name"] if isinstance(s, dict) else str(s) for s in manifest["shards"]]
        if "num_train_shards" not in manifest and config_shards and len(names) > config_shards:
            names = names[:config_shards]
        return names

    count = None
    for key in ("num_train_shards", "num_shards", "train_shards", "total_shards"):
        value = manifest.get(key)
        if isinstance(value, int):
            count = value
            break
    if count is None:
        return None

    dataset_stem = str(manifest.get("dataset", dataset_name)).replace("_", "-")
    generic = [f"shard-{i:06d}.tar" for i in range(count)]
    specific = [f"{dataset_stem}-{i:06d}.tar" for i in range(count)]
    return generic, specific


def discover_source_shards(
    config: dict[str, Any],
    groups: str,
    root: str,
    max_source_shards: int = 0,
) -> list[SourceShard]:
    selected_groups = resolve_groups(config, groups)
    shards: list[SourceShard] = []

    for group_name in selected_groups:
        group = config.get("groups", {}).get(group_name)
        if not group:
            raise ValueError(f"Unknown group {group_name!r} in dataset config")
        for dataset_name, info in group.get("datasets", {}).items():
            if info.get("skip", False):
                continue
            if float(info.get("weight", 1.0)) <= 0:
                continue

            base = os.path.join(root, info["path"])
            shard_dir = os.path.join(base, "shards")
            manifest_path = os.path.join(base, "manifest.json")
            config_shards = int(info.get("shards") or 0)

            names: list[str] | tuple[list[str], list[str]] | None = None
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    names = shard_names_from_manifest(json.load(f), dataset_name, config_shards)

            paths: list[str] = []
            if isinstance(names, tuple):
                generic, specific = names
                first_generic = os.path.join(shard_dir, generic[0]) if generic else ""
                chosen = generic if first_generic and os.path.exists(first_generic) else specific
                paths = [os.path.join(shard_dir, n) for n in chosen]
            elif names:
                paths = [os.path.join(shard_dir, n) for n in names]
            else:
                paths = sorted(str(p) for p in Path(shard_dir).glob("*.tar"))

            paths = [p for p in paths if os.path.exists(p)]
            if max_source_shards:
                paths = paths[:max_source_shards]

            LOG.info(
                "source %-24s group=%-14s shards=%d path=%s",
                dataset_name,
                group_name,
                len(paths),
                base,
            )
            for idx, path in enumerate(paths):
                shards.append(SourceShard(group_name, dataset_name, idx, path))

    if not shards:
        raise ValueError("No input shards discovered")
    return shards


def split_member_name(name: str) -> tuple[str, str] | None:
    base = os.path.basename(name)
    if "." not in base:
        return None
    key, ext = base.rsplit(".", 1)
    if not key or not ext:
        return None
    return key, ext.lower()


def iter_tar_samples(shard: SourceShard) -> Iterator[tuple[str, dict[str, bytes]]]:
    current_key: str | None = None
    current: dict[str, bytes] = {}

    with tarfile.open(shard.path, "r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parsed = split_member_name(member.name)
            if parsed is None:
                continue
            key, ext = parsed
            fh = tf.extractfile(member)
            if fh is None:
                continue
            payload = fh.read()

            if current_key is None:
                current_key = key
            elif key != current_key:
                if current:
                    yield current_key, current
                current_key = key
                current = {}

            current[ext] = payload

    if current_key is not None and current:
        yield current_key, current


def random_key(seed: int, shard: SourceShard, sample_key: str) -> int:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(seed).encode("utf-8"))
    h.update(b"\0")
    h.update(shard.group.encode("utf-8"))
    h.update(b"\0")
    h.update(shard.dataset.encode("utf-8"))
    h.update(b"\0")
    h.update(os.path.basename(shard.path).encode("utf-8"))
    h.update(b"\0")
    h.update(sample_key.encode("utf-8", errors="replace"))
    return int.from_bytes(h.digest(), "big")


class LruBucketFiles:
    def __init__(self, root: Path, max_open: int = 256):
        self.root = root
        self.max_open = max_open
        self.handles: OrderedDict[int, Any] = OrderedDict()

    def _path(self, bucket: int) -> Path:
        # Keep directories small enough for Lustre metadata operations.
        subdir = self.root / f"{bucket // 1024:03d}"
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"bucket-{bucket:05d}.bin"

    def write(self, bucket: int, payload: bytes) -> None:
        handle = self.handles.get(bucket)
        if handle is None:
            if len(self.handles) >= self.max_open:
                _, old = self.handles.popitem(last=False)
                old.close()
            handle = open(self._path(bucket), "ab", buffering=4 * 1024 * 1024)
            self.handles[bucket] = handle
        else:
            self.handles.move_to_end(bucket)
        handle.write(LEN_STRUCT.pack(len(payload)))
        handle.write(payload)

    def close(self) -> None:
        while self.handles:
            _, handle = self.handles.popitem(last=False)
            handle.close()


def bucket_worker(
    worker_id: int,
    shards: list[SourceShard],
    tmp_root: str,
    seed: int,
    bucket_bits: int,
    limit_samples: int,
    log_every: int,
) -> dict[str, int]:
    shift = 128 - bucket_bits
    writer = LruBucketFiles(Path(tmp_root) / "buckets" / f"worker-{worker_id:03d}")
    n_samples = 0
    n_shards = 0
    started = time.time()

    try:
        for shard in shards:
            n_shards += 1
            for source_key, members in iter_tar_samples(shard):
                if not any(ext in members for ext in IMAGE_EXTS) or "txt" not in members:
                    continue
                key_int = random_key(seed, shard, source_key)
                bucket = key_int >> shift
                record = (
                    key_int,
                    shard.group,
                    shard.dataset,
                    os.path.basename(shard.path),
                    source_key,
                    members,
                )
                writer.write(bucket, pickle.dumps(record, protocol=5))
                n_samples += 1
                if log_every > 0 and n_samples % log_every == 0:
                    rate = n_samples / max(1.0, time.time() - started)
                    LOG.info("worker %03d bucketed %d samples (%.1f/s)", worker_id, n_samples, rate)
                if limit_samples and n_samples >= limit_samples:
                    return {"worker": worker_id, "samples": n_samples, "shards": n_shards}
    finally:
        writer.close()

    return {"worker": worker_id, "samples": n_samples, "shards": n_shards}


def bucket_phase(
    shards: list[SourceShard],
    tmp_dir: Path,
    seed: int,
    bucket_bits: int,
    workers: int,
    limit_samples: int,
    log_every: int,
) -> int:
    bucket_root = tmp_dir / "buckets"
    if bucket_root.exists():
        shutil.rmtree(bucket_root)
    bucket_root.mkdir(parents=True)

    workers = max(1, min(workers, len(shards)))
    assignments = [[] for _ in range(workers)]
    for idx, shard in enumerate(shards):
        assignments[idx % workers].append(shard)

    if limit_samples:
        per_worker_limit = max(1, (limit_samples + workers - 1) // workers)
    else:
        per_worker_limit = 0

    LOG.info(
        "bucket phase: shards=%d workers=%d bucket_count=%d per_worker_limit=%d",
        len(shards),
        workers,
        1 << bucket_bits,
        per_worker_limit,
    )
    with mp.Pool(processes=workers) as pool:
        jobs = [
            pool.apply_async(
                bucket_worker,
                (worker_id, chunk, str(tmp_dir), seed, bucket_bits, per_worker_limit, log_every),
            )
            for worker_id, chunk in enumerate(assignments)
        ]
        results = [job.get() for job in jobs]

    total = sum(r["samples"] for r in results)
    LOG.info("bucket phase complete: samples=%d details=%s", total, results)
    return total


def iter_bucket_records(bucket_root: Path, bucket: int) -> Iterator[tuple]:
    rel = Path(f"{bucket // 1024:03d}") / f"bucket-{bucket:05d}.bin"
    for worker_dir in sorted(bucket_root.glob("worker-*")):
        path = worker_dir / rel
        if not path.exists():
            continue
        with open(path, "rb", buffering=4 * 1024 * 1024) as f:
            while True:
                header = f.read(LEN_STRUCT.size)
                if not header:
                    break
                if len(header) != LEN_STRUCT.size:
                    raise OSError(f"Truncated record header in {path}")
                (length,) = LEN_STRUCT.unpack(header)
                payload = f.read(length)
                if len(payload) != length:
                    raise OSError(f"Truncated record payload in {path}")
                yield pickle.loads(payload)


def metadata_with_provenance(
    original_json: bytes | None,
    key_int: int,
    group: str,
    dataset: str,
    source_shard: str,
    source_key: str,
) -> bytes:
    metadata: dict[str, Any]
    if original_json:
        try:
            metadata = json.loads(original_json.decode("utf-8"))
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}
    else:
        metadata = {}

    metadata["_global_shuffle"] = {
        "source_group": group,
        "source_dataset": dataset,
        "source_shard": source_shard,
        "source_key": source_key,
        "random_key": f"{key_int:032x}",
    }
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_phase(
    tmp_dir: Path,
    output_dir: Path,
    seed: int,
    bucket_bits: int,
    maxcount: int,
    maxsize: float,
) -> int:
    output_tmp = output_dir.with_name(output_dir.name + ".incomplete")
    if output_tmp.exists():
        shutil.rmtree(output_tmp)
    shard_dir = output_tmp / "shards"
    shard_dir.mkdir(parents=True)

    writer = wds.ShardWriter(
        str(shard_dir / "shard-%06d.tar"),
        maxcount=maxcount,
        maxsize=maxsize,
        verbose=1,
    )
    bucket_root = tmp_dir / "buckets"
    total = 0
    bucket_count = 1 << bucket_bits
    source_counts: dict[str, int] = {}

    try:
        for bucket in range(bucket_count):
            records = list(iter_bucket_records(bucket_root, bucket))
            if not records:
                continue
            records.sort(key=lambda r: (r[0], r[2], r[3], r[4]))
            for key_int, group, dataset, source_shard, source_key, members in records:
                sample = {"__key__": f"{total:012d}"}
                sample.update(members)
                sample["json"] = metadata_with_provenance(
                    members.get("json"),
                    key_int,
                    group,
                    dataset,
                    source_shard,
                    source_key,
                )
                writer.write(sample)
                source_counts[dataset] = source_counts.get(dataset, 0) + 1
                total += 1
            if bucket % 128 == 0:
                LOG.info("write phase bucket=%d/%d total=%d", bucket, bucket_count, total)
    finally:
        writer.close()

    shard_files = sorted(shard_dir.glob("*.tar"))
    manifest = {
        "dataset": output_dir.name,
        "description": "Pooled deterministic global shuffle for PRISM VLM scaling",
        "seed": seed,
        "bucket_bits": bucket_bits,
        "total_samples": total,
        "total_written": total,
        "num_shards": len(shard_files),
        "num_train_shards": len(shard_files),
        "shards": [{"name": p.name} for p in shard_files],
        "source_counts": dict(sorted(source_counts.items())),
    }
    with open(output_tmp / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_tmp.rename(output_dir)
    LOG.info("write phase complete: samples=%d shards=%d output=%s", total, len(shard_files), output_dir)
    return total


def write_dataset_config(output_dir: Path, total_samples: int) -> None:
    config = {
        "dastr": {"mount_base": str(output_dir.parent)},
        "groups": {
            output_dir.name: {
                "description": "Pooled globally shuffled VLM dataset",
                "datasets": {
                    output_dir.name: {
                        "path": output_dir.name,
                        "samples": int(total_samples),
                        "shards": 0,
                        "weight": 1.0,
                        "description": "Single-source global shuffle artifact",
                    }
                },
            }
        },
        "presets": {
            output_dir.name: {
                "description": "Single pooled globally shuffled VLM dataset",
                "groups": [output_dir.name],
                "mix_strategy": "single",
            }
        },
    }
    with open(output_dir / "dataset_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.bucket_bits <= 0 or args.bucket_bits > 20:
        raise ValueError("--bucket-bits must be in [1, 20]")

    output_dir = Path(args.output_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()

    if output_dir.exists() and not args.overwrite and not args.skip_write:
        raise SystemExit(f"Output exists; pass --overwrite to replace: {output_dir}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.source_config)
    root = source_root(cfg, args.source_root)
    shards = discover_source_shards(cfg, args.groups, root, args.max_source_shards)
    LOG.info("discovered %d source shards", len(shards))

    total_bucketed = 0
    if not args.skip_bucket:
        total_bucketed = bucket_phase(
            shards,
            tmp_dir,
            args.seed,
            args.bucket_bits,
            args.workers,
            args.limit_samples,
            args.log_every,
        )

    if not args.skip_write:
        total_written = write_phase(
            tmp_dir,
            output_dir,
            args.seed,
            args.bucket_bits,
            args.maxcount,
            args.maxsize,
        )
        write_dataset_config(output_dir, total_written)
        LOG.info("dataset config: %s", output_dir / "dataset_config.yaml")
    else:
        LOG.info("skip-write set; bucketed samples=%d", total_bucketed)


if __name__ == "__main__":
    main()

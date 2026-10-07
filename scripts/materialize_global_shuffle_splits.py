#!/usr/bin/env python3
"""Materialize train/cooldown views of a PRISM global-shuffle WebDataset."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import tarfile
from collections.abc import Iterator
from pathlib import Path

import yaml

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--train-name",
        default="vlm_diverse_global_shuffle_v1_train25k_gbs768",
    )
    parser.add_argument(
        "--cooldown-name",
        default="vlm_diverse_global_shuffle_v1_cooldown1k_gbs768",
    )
    parser.add_argument("--global-batch-size", type=int, default=768)
    parser.add_argument("--train-steps", type=int, default=25_000)
    parser.add_argument("--cooldown-steps", type=int, default=1_000)
    parser.add_argument("--source-maxcount", type=int, default=5_000)
    parser.add_argument("--cooldown-shards", type=int, default=192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-train",
        action="store_true",
        help="Leave an existing train split in place and only materialize cooldown.",
    )
    parser.add_argument(
        "--skip-cooldown-reshard",
        action="store_true",
        help="Only create the train symlink view and metadata.",
    )
    return parser.parse_args()


def load_manifest(source_dir: Path) -> dict:
    manifest_path = source_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not isinstance(manifest.get("shards"), list):
        raise ValueError(f"{manifest_path} must contain an explicit shards list")
    if "total_samples" not in manifest:
        raise ValueError(f"{manifest_path} must contain total_samples")
    return manifest


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    (path / "shards").mkdir(parents=True)


def symlink_shards(source_dir: Path, out_dir: Path, shard_names: list[str]) -> None:
    shards_dir = out_dir / "shards"
    for name in shard_names:
        src = source_dir / "shards" / name
        dst = shards_dir / name
        os.symlink(src, dst)


def write_manifest(
    out_dir: Path,
    *,
    dataset: str,
    description: str,
    seed: int | None,
    shard_names: list[str],
    total_samples: int,
    source_dataset: str,
    source_shard_start: int,
    source_shard_end_exclusive: int,
) -> None:
    manifest = {
        "dataset": dataset,
        "description": description,
        "seed": seed,
        "num_shards": len(shard_names),
        "num_train_shards": len(shard_names),
        "total_samples": total_samples,
        "total_written": total_samples,
        "source_dataset": source_dataset,
        "source_shard_start": source_shard_start,
        "source_shard_end_exclusive": source_shard_end_exclusive,
        "shards": [{"name": name} for name in shard_names],
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def write_dataset_config(out_dir: Path, dataset: str, samples: int, shards: int) -> None:
    config = {
        "dastr": {"mount_base": str(out_dir.parent)},
        "groups": {
            dataset: {
                "description": f"{dataset} split",
                "datasets": {
                    dataset: {
                        "path": dataset,
                        "samples": samples,
                        "shards": shards,
                        "weight": 1.0,
                        "description": f"{dataset} split",
                    }
                },
            }
        },
        "presets": {
            dataset: {
                "description": f"{dataset} split",
                "groups": [dataset],
                "mix_strategy": "single",
            }
        },
    }
    with open(out_dir / "dataset_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def split_member_name(name: str) -> tuple[str, str] | None:
    base = os.path.basename(name)
    if "." not in base:
        return None
    key, ext = base.rsplit(".", 1)
    if not key or not ext:
        return None
    return key, ext.lower()


def iter_tar_samples(path: Path) -> Iterator[dict[str, bytes]]:
    current_key: str | None = None
    current: dict[str, bytes] = {}

    with tarfile.open(path, "r:*") as tf:
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
                    yield current
                current_key = key
                current = {}

            current[ext] = payload

    if current_key is not None and current:
        yield current


def add_member(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    tf.addfile(info, io.BytesIO(payload))


def write_cooldown_shards(
    *,
    source_dir: Path,
    out_dir: Path,
    source_shard_names: list[str],
    total_samples: int,
    num_output_shards: int,
) -> list[str]:
    if num_output_shards <= 0:
        raise ValueError("--cooldown-shards must be positive")
    if total_samples < num_output_shards:
        raise ValueError(
            f"cooldown has only {total_samples} samples for {num_output_shards} shards"
        )

    base = total_samples // num_output_shards
    extra = total_samples % num_output_shards
    targets = [base + (1 if i < extra else 0) for i in range(num_output_shards)]

    out_names = [f"shard-{i:06d}.tar" for i in range(num_output_shards)]
    shard_index = 0
    sample_in_shard = 0
    sample_index = 0
    tf: tarfile.TarFile | None = None

    def open_current() -> tarfile.TarFile:
        path = out_dir / "shards" / out_names[shard_index]
        return tarfile.open(path, "w")

    try:
        tf = open_current()
        for source_name in source_shard_names:
            for members in iter_tar_samples(source_dir / "shards" / source_name):
                if "txt" not in members or not any(ext in members for ext in IMAGE_EXTS):
                    continue

                if sample_in_shard >= targets[shard_index]:
                    tf.close()
                    shard_index += 1
                    sample_in_shard = 0
                    if shard_index >= num_output_shards:
                        raise RuntimeError("more cooldown samples than expected")
                    tf = open_current()

                key = f"{sample_index:09d}"
                for ext in sorted(members):
                    add_member(tf, f"{key}.{ext}", members[ext])
                sample_index += 1
                sample_in_shard += 1
    finally:
        if tf is not None:
            tf.close()

    if sample_index != total_samples:
        raise RuntimeError(f"wrote {sample_index} samples, expected {total_samples}")
    if shard_index != num_output_shards - 1:
        raise RuntimeError(
            f"wrote through shard index {shard_index}, expected {num_output_shards - 1}"
        )
    return out_names


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else source_dir.parent
    manifest = load_manifest(source_dir)

    source_shard_names = [s["name"] for s in manifest["shards"]]
    total_samples = int(manifest["total_samples"])
    train_samples = args.global_batch_size * args.train_steps
    cooldown_min_samples = args.global_batch_size * args.cooldown_steps

    if train_samples % args.source_maxcount != 0:
        raise ValueError(
            f"train_samples={train_samples} is not divisible by "
            f"source_maxcount={args.source_maxcount}; cannot split cleanly by shard"
        )
    train_shards = train_samples // args.source_maxcount
    if train_shards >= len(source_shard_names):
        raise ValueError("train split would consume all source shards")

    reserve_samples = total_samples - train_samples
    if reserve_samples < cooldown_min_samples:
        raise ValueError(
            f"reserve has {reserve_samples} samples, but cooldown needs "
            f"{cooldown_min_samples}"
        )

    train_dir = output_root / args.train_name
    cooldown_dir = output_root / args.cooldown_name

    train_names = source_shard_names[:train_shards]
    if args.reuse_train and train_dir.exists():
        print(f"train: reusing existing {train_dir}")
    else:
        reset_dir(train_dir, args.overwrite)
        symlink_shards(source_dir, train_dir, train_names)
        write_manifest(
            train_dir,
            dataset=args.train_name,
            description=(
                f"First {train_shards} shards of {manifest.get('dataset', source_dir.name)}; "
                f"{args.train_steps} steps at global batch {args.global_batch_size}"
            ),
            seed=manifest.get("seed"),
            shard_names=train_names,
            total_samples=train_samples,
            source_dataset=manifest.get("dataset", source_dir.name),
            source_shard_start=0,
            source_shard_end_exclusive=train_shards,
        )
        write_dataset_config(train_dir, args.train_name, train_samples, len(train_names))

    if not args.skip_cooldown_reshard:
        reset_dir(cooldown_dir, args.overwrite)
        tail_names = source_shard_names[train_shards:]
        cooldown_names = write_cooldown_shards(
            source_dir=source_dir,
            out_dir=cooldown_dir,
            source_shard_names=tail_names,
            total_samples=reserve_samples,
            num_output_shards=args.cooldown_shards,
        )
        min_per_rank = math.floor(reserve_samples / args.cooldown_shards)
        write_manifest(
            cooldown_dir,
            dataset=args.cooldown_name,
            description=(
                f"Held-out tail of {manifest.get('dataset', source_dir.name)}; "
                f"at least {args.cooldown_steps} steps at global batch "
                f"{args.global_batch_size}"
            ),
            seed=manifest.get("seed"),
            shard_names=cooldown_names,
            total_samples=reserve_samples,
            source_dataset=manifest.get("dataset", source_dir.name),
            source_shard_start=train_shards,
            source_shard_end_exclusive=len(source_shard_names),
        )
        write_dataset_config(
            cooldown_dir,
            args.cooldown_name,
            reserve_samples,
            len(cooldown_names),
        )
        print(
            f"cooldown: {cooldown_dir} shards={len(cooldown_names)} "
            f"samples={reserve_samples} min_samples_per_shard={min_per_rank}"
        )

    print(f"train: {train_dir} shards={len(train_names)} samples={train_samples}")
    print(
        f"reserve samples={reserve_samples}; cooldown requirement={cooldown_min_samples}"
    )


if __name__ == "__main__":
    main()

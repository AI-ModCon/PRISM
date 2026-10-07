#!/usr/bin/env python3
"""
convert_s1mmalign_to_webdataset.py - Convert S1-MMAlign dataset to WebDataset format

S1-MMAlign has two different structures:
- arxiv: Nested tar.gz (images_YEAR.tar.gz containing paper.tar.gz containing images)
- biorxiv/others: Single tar.gz with flat image paths

Usage:
    # Test with small limit first
    python scripts/convert_s1mmalign_to_webdataset.py \
        --source biorxiv \
        --output-dir /flare/ModCon/ngetty/data/zone_a/s1mmalign_biorxiv_webdataset \
        --limit 1000
"""

import argparse
import glob
import gzip
import io
import json
import logging
import os
import random
import subprocess
import tarfile
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = "/flare/ModCon/ngetty/data/zone_a/s1mmalign_raw"


def get_split_tar_files(source_dir):
    """
    Find tar.gz files, handling split files (.partaa, .partab, etc.)
    Returns list of (name, [parts]) tuples.
    """
    all_files = glob.glob(os.path.join(source_dir, "*.tar.gz*"))

    # Group by base name
    file_groups = defaultdict(list)
    for f in all_files:
        if ".part" in f:
            base = f.rsplit(".part", 1)[0]
            file_groups[base].append(f)
        else:
            file_groups[f].append(f)

    result = []
    for base, parts in file_groups.items():
        parts = sorted(parts)
        result.append((os.path.basename(base), parts))

    return result


def create_sample_key(idx):
    return f"{idx:08d}"


def write_shard(samples, shard_path, dataset_name):
    """Write samples to a TAR shard."""
    written = 0
    with tarfile.open(shard_path, "w") as tar:
        for i, sample in enumerate(samples):
            if sample is None:
                continue

            image_bytes, ext, text, metadata = sample
            key = create_sample_key(metadata.get("global_idx", i))

            if ext.lower() in ["jpeg"]:
                ext = "jpg"
            elif ext.lower() not in ["jpg", "png", "webp", "gif"]:
                ext = "png"

            img_info = tarfile.TarInfo(name=f"{key}.{ext}")
            img_info.size = len(image_bytes)
            tar.addfile(img_info, io.BytesIO(image_bytes))

            text_bytes = text.encode("utf-8")
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(text_bytes)
            tar.addfile(txt_info, io.BytesIO(text_bytes))

            meta_bytes = json.dumps(metadata).encode("utf-8")
            meta_info = tarfile.TarInfo(name=f"{key}.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, io.BytesIO(meta_bytes))

            written += 1

    return written


def load_images_from_flat_tar(tar_parts, progress_callback=None):
    """
    Load images from a flat tar.gz (biorxiv style).
    Returns dict: {image_path: image_bytes}
    """
    images = {}

    # Reassemble split files if needed
    if len(tar_parts) > 1:
        logger.info(f"  Reassembling {len(tar_parts)} split parts...")
        # Use cat + tar to stream
        cat_cmd = ["cat"] + tar_parts
        cat_proc = subprocess.Popen(cat_cmd, stdout=subprocess.PIPE)

        with gzip.GzipFile(fileobj=cat_proc.stdout) as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tar:
                for member in tar:
                    if member.isfile() and not member.name.endswith("/"):
                        try:
                            f = tar.extractfile(member)
                            if f:
                                images[member.name] = f.read()
                                if progress_callback:
                                    progress_callback(len(images))
                        except Exception as e:
                            logger.debug(f"Failed to extract {member.name}: {e}")
        cat_proc.wait()
    else:
        # Single file
        with tarfile.open(tar_parts[0], "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting images"):
                if member.isfile():
                    try:
                        f = tar.extractfile(member)
                        if f:
                            images[member.name] = f.read()
                    except Exception as e:
                        logger.debug(f"Failed to extract {member.name}: {e}")

    return images


def load_images_from_nested_tar(tar_parts, needed_paths, progress_callback=None):
    """
    Load images from nested tar.gz (arxiv style).

    Structure: outer.tar.gz > YYMM/paper.tar.gz > images

    Args:
        tar_parts: List of tar.gz parts
        needed_paths: Set of image paths we need (for filtering)

    Returns dict: {image_path: image_bytes}
    """
    images = {}

    # Parse needed paths to know which inner tars to open
    inner_tars_needed = set()
    path_to_inner = {}  # image_path -> (inner_tar_path, image_name)

    for path in needed_paths:
        # Format: images/1805/1704.07661.tar.gz/fig017.png
        parts = path.split("/")
        if len(parts) >= 4 and parts[0] == "images":
            inner_tar = "/".join(parts[1:3])  # "1805/1704.07661.tar.gz"
            image_name = parts[3]  # "fig017.png"
            inner_tars_needed.add(inner_tar)
            path_to_inner[path] = (inner_tar, image_name)

    logger.info(
        f"  Need {len(inner_tars_needed)} inner tar files for {len(needed_paths)} images"
    )

    # Open outer tar
    cat_proc = None
    if len(tar_parts) > 1:
        logger.info(f"  Reassembling {len(tar_parts)} split parts...")
        cat_cmd = ["cat"] + tar_parts
        cat_proc = subprocess.Popen(cat_cmd, stdout=subprocess.PIPE)
        outer_tar = tarfile.open(
            fileobj=gzip.GzipFile(fileobj=cat_proc.stdout), mode="r|"
        )
    else:
        outer_tar = tarfile.open(tar_parts[0], "r:gz")

    inner_cache = {}  # inner_tar_path -> {image_name: bytes}

    try:
        for member in outer_tar:
            if not member.isfile():
                continue

            # Check if this is an inner tar we need
            if member.name in inner_tars_needed or any(
                member.name.endswith(it) for it in inner_tars_needed
            ):
                try:
                    inner_file = outer_tar.extractfile(member)
                    if inner_file:
                        # Decompress inner tar.gz
                        inner_data = gzip.decompress(inner_file.read())
                        with tarfile.open(
                            fileobj=io.BytesIO(inner_data), mode="r:"
                        ) as inner_tar:
                            cache = {}
                            for inner_member in inner_tar:
                                if inner_member.isfile():
                                    f = inner_tar.extractfile(inner_member)
                                    if f:
                                        cache[os.path.basename(inner_member.name)] = (
                                            f.read()
                                        )
                            inner_cache[member.name] = cache
                except Exception as e:
                    logger.debug(f"Failed to extract inner tar {member.name}: {e}")
    finally:
        outer_tar.close()
        if cat_proc is not None:
            cat_proc.wait()

    # Map back to original paths
    for orig_path, (inner_tar, image_name) in path_to_inner.items():
        # Find matching inner tar in cache
        for cached_path, cache in inner_cache.items():
            if cached_path.endswith(inner_tar) or inner_tar in cached_path:
                if image_name in cache:
                    images[orig_path] = cache[image_name]
                    break

    return images


def detect_format(source_dir, sample_path):
    """
    Detect whether source uses flat or nested tar structure.
    """
    if sample_path.startswith("images/") and ".tar.gz/" in sample_path:
        return "nested"  # arxiv style
    else:
        return "flat"  # biorxiv style


def convert_s1mmalign_source(
    source, output_dir, samples_per_shard=1000, limit=None, val_split=0.01
):
    """
    Convert a single s1mmalign source to WebDataset format.
    """
    source_dir = os.path.join(BASE_DIR, source)

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    os.makedirs(output_dir, exist_ok=True)
    shards_dir = os.path.join(output_dir, "shards")
    val_shards_dir = os.path.join(output_dir, "val_shards")
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(val_shards_dir, exist_ok=True)

    # Find JSONL files
    jsonl_dir = os.path.join(source_dir, "jsonl")
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*recaption.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {jsonl_dir}")

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    # Load JSONL metadata
    logger.info("Loading JSONL metadata...")
    all_samples = []
    for jf in tqdm(jsonl_files, desc="Loading JSONL"):
        with open(jf) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    all_samples.append(data)
                except json.JSONDecodeError:
                    continue

    total_samples = len(all_samples)
    logger.info(f"Total samples: {total_samples:,}")

    if limit:
        all_samples = all_samples[:limit]
        logger.info(f"Limited to {limit} samples")

    # Detect format from first sample
    first_path = all_samples[0].get("image_path", "") if all_samples else ""
    format_type = detect_format(source_dir, first_path)
    logger.info(f"Detected format: {format_type}")
    logger.info(f"Sample path: {first_path}")

    # Find tar files
    tar_files = get_split_tar_files(source_dir)
    logger.info(f"Found {len(tar_files)} tar archive(s)")
    for name, parts in tar_files:
        logger.info(f"  {name}: {len(parts)} part(s)")

    if not tar_files:
        raise FileNotFoundError(f"No tar.gz files found in {source_dir}")

    # Get all needed image paths
    needed_paths = set(
        s.get("image_path", "") for s in all_samples if s.get("image_path")
    )
    logger.info(f"Need {len(needed_paths)} unique images")

    # Load images
    logger.info("Loading images from tar archives...")
    all_images = {}

    for tar_name, tar_parts in tar_files:
        logger.info(f"Processing {tar_name}...")

        if format_type == "nested":
            # Filter needed paths for this tar (by year)
            # tar_name like "images_2018.tar.gz" -> year "18"
            if "images_" in tar_name:
                year = tar_name.replace("images_", "").replace(".tar.gz", "")
                year_prefix = year[2:]  # "2018" -> "18"
                tar_needed = {
                    p
                    for p in needed_paths
                    if "/" + year_prefix in p or p.startswith(f"images/{year_prefix}")
                }
            else:
                tar_needed = needed_paths

            if tar_needed:
                images = load_images_from_nested_tar(tar_parts, tar_needed)
                all_images.update(images)
        else:
            # Flat format - load all
            images = load_images_from_flat_tar(tar_parts)
            all_images.update(images)

    logger.info(f"Loaded {len(all_images)} images")

    # Process samples
    logger.info("Processing samples...")
    processed_samples = []
    stats = {"matched": 0, "missing_image": 0, "invalid_image": 0}

    for idx, sample in enumerate(tqdm(all_samples, desc="Processing")):
        image_path = sample.get("image_path", "")

        # Try to find image with various path formats
        image_bytes = all_images.get(image_path)

        # Also try without leading source name
        if image_bytes is None and image_path.startswith(source + "/"):
            alt_path = image_path[len(source) + 1 :]
            image_bytes = all_images.get(alt_path)

        # Try matching by filename only
        if image_bytes is None:
            basename = os.path.basename(image_path)
            for stored_path, stored_bytes in all_images.items():
                if stored_path.endswith(basename):
                    image_bytes = stored_bytes
                    break

        if image_bytes is None:
            stats["missing_image"] += 1
            continue

        # Validate image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            ext = img.format.lower() if img.format else "png"
            img.verify()
        except Exception:
            stats["invalid_image"] += 1
            continue

        # Get recaption text
        text = sample.get("recaption", sample.get("caption", ""))
        if not text:
            continue

        metadata = {
            "id": sample.get("arxiv_id", sample.get("doi", str(idx))),
            "title": sample.get("title", ""),
            "categories": sample.get("categories", ""),
            "image_path": image_path,
            "global_idx": idx,
            "source": source,
        }

        processed_samples.append((image_bytes, ext, text, metadata))
        stats["matched"] += 1

    logger.info("\nProcessing stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v:,}")

    if not processed_samples:
        logger.error("No samples processed!")
        return None

    # Shuffle and split
    random.seed(42)
    random.shuffle(processed_samples)

    val_size = int(len(processed_samples) * val_split)
    train_samples = processed_samples[val_size:]
    val_samples = processed_samples[:val_size]

    logger.info(f"Train: {len(train_samples):,}, Val: {len(val_samples):,}")

    # Write shards
    num_train_shards = (len(train_samples) + samples_per_shard - 1) // samples_per_shard
    dataset_name = f"s1mmalign-{source}"

    manifest = {
        "dataset": dataset_name,
        "source": "s1mmalign_raw",
        "source_subset": source,
        "format_type": format_type,
        "total_samples": len(processed_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "samples_per_shard": samples_per_shard,
        "num_shards": num_train_shards,
        "stats": stats,
        "shards": [],
        "val_shards": [],
    }

    total_written = 0
    logger.info(f"Writing {num_train_shards} training shards...")

    for shard_idx in tqdm(range(num_train_shards), desc="Training shards"):
        start = shard_idx * samples_per_shard
        end = min(start + samples_per_shard, len(train_samples))
        batch = train_samples[start:end]

        shard_name = f"{dataset_name}-{shard_idx:06d}.tar"
        shard_path = os.path.join(shards_dir, shard_name)
        written = write_shard(batch, shard_path, dataset_name)

        manifest["shards"].append(
            {
                "name": shard_name,
                "samples": written,
                "size_bytes": os.path.getsize(shard_path),
            }
        )
        total_written += written

    # Write validation shards
    if val_samples:
        num_val_shards = max(
            1, (len(val_samples) + samples_per_shard - 1) // samples_per_shard
        )
        logger.info(f"Writing {num_val_shards} validation shards...")

        for shard_idx in range(num_val_shards):
            start = shard_idx * samples_per_shard
            end = min(start + samples_per_shard, len(val_samples))
            batch = val_samples[start:end]

            val_shard_idx = num_train_shards + shard_idx
            shard_name = f"{dataset_name}-{val_shard_idx:06d}.tar"
            shard_path = os.path.join(val_shards_dir, shard_name)
            written = write_shard(batch, shard_path, dataset_name)

            manifest["val_shards"].append(
                {
                    "name": shard_name,
                    "samples": written,
                    "size_bytes": os.path.getsize(shard_path),
                }
            )

    manifest["total_written"] = total_written
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n=== Conversion Complete: {dataset_name} ===")
    logger.info(f"Total written: {total_written:,}")
    logger.info(f"Output: {output_dir}")

    return manifest


# Sources ordered by size (smallest first for quick wins)
ALL_SOURCES = [
    "metarxiv",  # 27 MB, 365 samples
    "edrxiv",  # 69 MB, 1,388 samples
    "engrxiv",  # 1.3 GB, 25K samples
    "psyarxiv",  # 1 GB, 17K samples
    "chemrxiv",  # 12 GB, 180K samples
    "medrxiv",  # Medium
    "nature_comunication",  # Medium
    "biorxiv",  # ~120 GB, 1.1M samples
    "arxiv",  # ~2.7 TB, 15M samples (largest, last)
]


def convert_all_sources(output_base_dir, skip_sources=None, **kwargs):
    """
    Convert all S1-MMAlign sources to WebDataset format.
    Sources are processed in order of size (smallest first).
    """
    skip_sources = set(skip_sources or [])
    os.makedirs(output_base_dir, exist_ok=True)

    results = []

    for source in ALL_SOURCES:
        if source in skip_sources:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"SKIPPING: {source} (user requested)")
            logger.info(f"{'=' * 60}")
            results.append((source, 0, "skipped"))
            continue

        source_dir = os.path.join(BASE_DIR, source)
        if not os.path.isdir(source_dir):
            logger.warning(f"Skipping {source}: directory not found")
            results.append((source, 0, "not found"))
            continue

        output_dir = os.path.join(output_base_dir, f"{source}_webdataset")

        # Check if already converted
        manifest_path = os.path.join(output_dir, "manifest.json")
        if os.path.exists(manifest_path):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"SKIPPING: {source} (already converted)")
            logger.info(f"{'=' * 60}")
            with open(manifest_path) as f:
                manifest = json.load(f)
            results.append((source, manifest.get("total_written", 0), "already done"))
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"CONVERTING: {source}")
        logger.info(f"{'=' * 60}")

        try:
            manifest = convert_s1mmalign_source(
                source=source, output_dir=output_dir, **kwargs
            )
            if manifest:
                results.append((source, manifest["total_written"], "success"))
            else:
                results.append((source, 0, "no samples"))
        except Exception as e:
            logger.error(f"Error converting {source}: {e}")
            import traceback

            traceback.print_exc()
            results.append((source, 0, f"error: {str(e)[:50]}"))

    # Print summary
    print("\n" + "=" * 70)
    print("S1-MMALIGN CONVERSION SUMMARY")
    print("=" * 70)
    total = 0
    for name, count, status in results:
        status_str = f"[{status}]"
        print(f"  {name:25s} {count:>10,} samples  {status_str}")
        total += count
    print("-" * 70)
    print(f"  {'TOTAL':25s} {total:>10,} samples")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert S1-MMAlign dataset to WebDataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all sources (smallest first)
  python scripts/convert_s1mmalign_to_webdataset.py --convert-all \\
      --output-base-dir /flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets

  # Skip large sources for now
  python scripts/convert_s1mmalign_to_webdataset.py --convert-all \\
      --output-base-dir /flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets \\
      --skip arxiv biorxiv

  # Convert single source
  python scripts/convert_s1mmalign_to_webdataset.py --source metarxiv \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/s1mmalign_metarxiv_webdataset
        """,
    )

    # Single source mode
    parser.add_argument(
        "--source",
        choices=ALL_SOURCES,
        help="Single source to convert",
    )
    parser.add_argument(
        "--output-dir", help="Output directory for WebDataset (single source mode)"
    )

    # Convert all mode
    parser.add_argument(
        "--convert-all",
        action="store_true",
        help="Convert all sources (smallest first)",
    )
    parser.add_argument(
        "--output-base-dir",
        default="/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets",
        help="Base output directory for all converted datasets",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Sources to skip (e.g., --skip arxiv biorxiv)",
    )

    # Common options
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit samples per source (for testing)"
    )
    parser.add_argument("--val-split", type=float, default=0.01)

    args = parser.parse_args()

    if args.convert_all:
        # Convert all sources
        convert_all_sources(
            output_base_dir=args.output_base_dir,
            skip_sources=args.skip,
            samples_per_shard=args.samples_per_shard,
            limit=args.limit,
            val_split=args.val_split,
        )
    elif args.source and args.output_dir:
        # Single source mode
        convert_s1mmalign_source(
            source=args.source,
            output_dir=args.output_dir,
            samples_per_shard=args.samples_per_shard,
            limit=args.limit,
            val_split=args.val_split,
        )
    else:
        parser.print_help()
        print("\nError: Provide either --convert-all or both --source and --output-dir")


if __name__ == "__main__":
    main()

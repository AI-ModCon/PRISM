#!/usr/bin/env python3
"""
convert_arxiv_streaming.py - Convert S1-MMAlign arxiv to WebDataset (streaming)

This version processes one year at a time to avoid loading all images into memory.
Designed for the 2.6TB arxiv partition with ~13M samples.

The arxiv tar structure is:
  images_YEAR.tar.gz contains: YYMM/arxiv_id.tar.gz/figXXX.png
  (The .tar.gz in the path is a DIRECTORY name, not a nested archive!)

JSONL paths look like: images/0705/0606282.tar.gz/fig002.png
Tar paths look like:   0705/0606282.tar.gz/fig002.png

Usage:
    python applications/text/convert_arxiv_streaming.py \\
        --output-dir /flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/arxiv_webdataset \\
        --samples-per-shard 5000

    # Resume from a specific year
    python applications/text/convert_arxiv_streaming.py \\
        --output-dir /flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/arxiv_webdataset \\
        --start-year 2020
"""

import argparse
import gc
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = "/flare/ModCon/ngetty/data/zone_a/s1mmalign_raw"
SOURCE = "arxiv"


def get_split_tar_files(source_dir: str) -> dict[str, list[str]]:
    """
    Find tar.gz files, handling split files (.partaa, .partab, etc.)
    Returns dict: {year: [parts]}
    """
    all_files = glob.glob(os.path.join(source_dir, "*.tar.gz*"))

    # Group by year
    year_files = defaultdict(list)
    for f in all_files:
        basename = os.path.basename(f)
        # Extract year from "images_2018.tar.gz" or "images_2018.tar.gz.partaa"
        if "images_" in basename:
            year = basename.split("images_")[1].split(".")[0]
            year_files[year].append(f)

    # Sort parts within each year
    for year in year_files:
        year_files[year] = sorted(year_files[year])

    return dict(year_files)


def create_sample_key(idx: int) -> str:
    return f"{idx:08d}"


def write_shard(samples: list[tuple], shard_path: str) -> int:
    """Write samples to a TAR shard."""
    written = 0
    with tarfile.open(shard_path, "w") as tar:
        for image_bytes, ext, text, metadata in samples:
            if image_bytes is None:
                continue

            key = create_sample_key(metadata.get("global_idx", written))

            if ext.lower() in ["jpeg"]:
                ext = "jpg"
            elif ext.lower() not in ["jpg", "png", "webp", "gif"]:
                ext = "png"

            # Image
            img_info = tarfile.TarInfo(name=f"{key}.{ext}")
            img_info.size = len(image_bytes)
            tar.addfile(img_info, io.BytesIO(image_bytes))

            # Text
            text_bytes = text.encode("utf-8")
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(text_bytes)
            tar.addfile(txt_info, io.BytesIO(text_bytes))

            # Metadata
            meta_bytes = json.dumps(metadata).encode("utf-8")
            meta_info = tarfile.TarInfo(name=f"{key}.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, io.BytesIO(meta_bytes))

            written += 1

    return written


def load_jsonl_by_year(jsonl_dir: str) -> dict[str, list[dict]]:
    """
    Load JSONL metadata and group by year.
    Returns: {year: [samples]}
    """
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*recaption.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {jsonl_dir}")

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    samples_by_year = defaultdict(list)
    total = 0

    for jf in tqdm(jsonl_files, desc="Loading JSONL"):
        with open(jf) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Extract year from image_path like "images/0705/..." -> "2007"
                    # The folder YYMM maps to year 20YY (e.g., 0705 -> 2007, 1805 -> 2018)
                    image_path = data.get("image_path", "")
                    if image_path.startswith("images/"):
                        yymm = image_path.split("/")[1]  # "0705", "1805", etc.
                        year_prefix = yymm[:2]  # "07", "18", etc.
                        year = "20" + year_prefix
                        samples_by_year[year].append(data)
                        total += 1
                except json.JSONDecodeError:
                    continue

    logger.info(f"Total samples: {total:,}")
    logger.info(f"Years: {sorted(samples_by_year.keys())}")
    for year in sorted(samples_by_year.keys()):
        logger.info(f"  {year}: {len(samples_by_year[year]):,} samples")

    return dict(samples_by_year)


def extract_images_streaming(
    tar_parts: list[str],
    needed_paths: set[str],
    progress_bar: tqdm | None = None,
) -> dict[str, bytes]:
    """
    Extract images from tar.gz in streaming fashion.

    The tar contains flat files like: 0705/0606282.tar.gz/fig002.png
    (The .tar.gz is a directory name, not a nested archive)

    Args:
        tar_parts: List of tar file parts to cat together
        needed_paths: Set of paths we need (WITHOUT the 'images/' prefix)

    Returns:
        Dict of {tar_path: image_bytes}
    """
    images = {}

    # Open tar (streaming for split files)
    if len(tar_parts) > 1:
        logger.info(f"    Streaming {len(tar_parts)} split parts...")
        cat_cmd = ["cat"] + sorted(tar_parts)
        cat_proc = subprocess.Popen(
            cat_cmd, stdout=subprocess.PIPE, bufsize=64 * 1024 * 1024
        )
        tar_file = tarfile.open(
            fileobj=gzip.GzipFile(fileobj=cat_proc.stdout), mode="r|"
        )
    else:
        cat_proc = None
        tar_file = tarfile.open(tar_parts[0], "r:gz")

    extracted = 0
    try:
        for member in tar_file:
            if not member.isfile():
                continue

            # member.name is like "0705/0606282.tar.gz/fig002.png"
            # Check if this path is needed
            if member.name in needed_paths:
                try:
                    f = tar_file.extractfile(member)
                    if f:
                        images[member.name] = f.read()
                        extracted += 1
                        if progress_bar:
                            progress_bar.update(1)
                            progress_bar.set_postfix(found=extracted)
                except Exception as e:
                    logger.debug(f"Failed to extract {member.name}: {e}")

    finally:
        tar_file.close()
        if cat_proc:
            cat_proc.wait()

    return images


def process_year(
    year: str,
    tar_parts: list[str],
    samples: list[dict],
    shards_dir: str,
    shard_start_idx: int,
    samples_per_shard: int,
    global_idx_start: int,
) -> tuple[int, int, int, dict]:
    """
    Process all samples for a single year.

    Returns:
        (num_shards_written, num_samples_written, next_global_idx, stats)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing year {year}: {len(samples):,} samples")
    logger.info(f"{'=' * 60}")

    # Build set of needed tar paths (remove 'images/' prefix from JSONL paths)
    needed_paths = set()
    path_to_sample_idx = defaultdict(list)  # tar_path -> [sample indices]

    for idx, sample in enumerate(samples):
        image_path = sample.get("image_path", "")
        # Convert JSONL path to tar path
        # JSONL: "images/0705/0606282.tar.gz/fig002.png"
        # Tar:   "0705/0606282.tar.gz/fig002.png"
        if image_path.startswith("images/"):
            tar_path = image_path[7:]  # Remove "images/" prefix
            needed_paths.add(tar_path)
            path_to_sample_idx[tar_path].append(idx)

    logger.info(f"Need {len(needed_paths):,} unique images")

    # Extract images for this year
    logger.info("Extracting images...")
    with tqdm(total=len(needed_paths), desc=f"Year {year}", unit="img") as pbar:
        images = extract_images_streaming(tar_parts, needed_paths, pbar)

    logger.info(f"Extracted {len(images):,} images")

    # Process samples
    processed_samples = []
    stats = {"matched": 0, "missing_image": 0, "invalid_image": 0, "no_text": 0}
    global_idx = global_idx_start

    for _idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
        image_path = sample.get("image_path", "")

        # Convert to tar path
        if image_path.startswith("images/"):
            tar_path = image_path[7:]
        else:
            tar_path = image_path

        image_bytes = images.get(tar_path)

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
            stats["no_text"] += 1
            continue

        metadata = {
            "id": sample.get("arxiv_id", str(global_idx)),
            "title": sample.get("title", ""),
            "categories": sample.get("categories", ""),
            "image_path": image_path,
            "global_idx": global_idx,
            "source": SOURCE,
            "year": year,
        }

        processed_samples.append((image_bytes, ext, text, metadata))
        stats["matched"] += 1
        global_idx += 1

    # Free image memory
    del images
    gc.collect()

    logger.info(f"Processing stats for year {year}:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v:,}")

    if not processed_samples:
        logger.warning(f"No samples processed for year {year}!")
        return 0, 0, global_idx, stats

    # Shuffle samples within year
    random.shuffle(processed_samples)

    # Write shards
    num_shards = (len(processed_samples) + samples_per_shard - 1) // samples_per_shard
    total_written = 0

    logger.info(f"Writing {num_shards} shards...")
    for i in tqdm(range(num_shards), desc="Writing shards"):
        start = i * samples_per_shard
        end = min(start + samples_per_shard, len(processed_samples))
        shard_samples = processed_samples[start:end]

        shard_idx = shard_start_idx + i
        shard_path = os.path.join(shards_dir, f"shard-{shard_idx:06d}.tar")
        written = write_shard(shard_samples, shard_path)
        total_written += written

    # Free sample memory
    del processed_samples
    gc.collect()

    logger.info(
        f"Year {year} complete: {total_written:,} samples in {num_shards} shards"
    )

    return num_shards, total_written, global_idx, stats


def convert_arxiv_streaming(
    output_dir: str,
    samples_per_shard: int = 5000,
    start_year: str | None = None,
    years_to_process: list[str] | None = None,
):
    """
    Convert arxiv dataset to WebDataset format, processing one year at a time.
    """
    source_dir = os.path.join(BASE_DIR, SOURCE)

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Setup output directories
    os.makedirs(output_dir, exist_ok=True)
    shards_dir = os.path.join(output_dir, "shards")
    val_shards_dir = os.path.join(output_dir, "val_shards")
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(val_shards_dir, exist_ok=True)

    # Find tar files by year
    tar_files_by_year = get_split_tar_files(source_dir)
    logger.info(f"Found {len(tar_files_by_year)} years of tar archives")
    for year in sorted(tar_files_by_year.keys()):
        parts = tar_files_by_year[year]
        logger.info(f"  {year}: {len(parts)} part(s)")

    # Load JSONL metadata grouped by year
    jsonl_dir = os.path.join(source_dir, "jsonl")
    samples_by_year = load_jsonl_by_year(jsonl_dir)

    # Determine which years to process
    all_years = sorted(set(tar_files_by_year.keys()) & set(samples_by_year.keys()))

    if years_to_process:
        all_years = [y for y in all_years if y in years_to_process]
    elif start_year:
        all_years = [y for y in all_years if y >= start_year]

    logger.info(f"Will process years: {all_years}")

    # Check for existing progress
    existing_shards = glob.glob(os.path.join(shards_dir, "shard-*.tar"))
    if existing_shards:
        last_shard = max(
            int(os.path.basename(s).split("-")[1].split(".")[0])
            for s in existing_shards
        )
        shard_start_idx = last_shard + 1
        logger.info(
            f"Resuming from shard {shard_start_idx} (found {len(existing_shards)} existing)"
        )
    else:
        shard_start_idx = 0

    # Progress tracking
    progress_file = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
        completed_years = set(progress.get("completed_years", []))
        global_idx = progress.get("global_idx", 0)
        total_written = progress.get("total_written", 0)
        total_shards = progress.get("total_shards", shard_start_idx)
    else:
        completed_years = set()
        global_idx = 0
        total_written = 0
        total_shards = shard_start_idx

    # Process each year
    all_stats = defaultdict(int)
    random.seed(42)

    for year in all_years:
        if year in completed_years:
            logger.info(f"Skipping year {year} (already completed)")
            continue

        if year not in tar_files_by_year:
            logger.warning(f"No tar files for year {year}, skipping")
            continue

        if year not in samples_by_year:
            logger.warning(f"No samples for year {year}, skipping")
            continue

        tar_parts = tar_files_by_year[year]
        samples = samples_by_year[year]

        try:
            num_shards, num_written, global_idx, stats = process_year(
                year=year,
                tar_parts=tar_parts,
                samples=samples,
                shards_dir=shards_dir,
                shard_start_idx=total_shards,
                samples_per_shard=samples_per_shard,
                global_idx_start=global_idx,
            )

            total_shards += num_shards
            total_written += num_written

            for k, v in stats.items():
                all_stats[k] += v

            # Mark year as completed
            completed_years.add(year)

            # Save progress
            with open(progress_file, "w") as f:
                json.dump(
                    {
                        "completed_years": list(completed_years),
                        "global_idx": global_idx,
                        "total_written": total_written,
                        "total_shards": total_shards,
                        "stats": dict(all_stats),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Progress saved. Total: {total_written:,} samples, {total_shards} shards"
            )

        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
            import traceback

            traceback.print_exc()
            # Continue with next year
            continue

    # Write final manifest
    manifest = {
        "dataset": "s1mmalign-arxiv",
        "total_written": total_written,
        "num_shards": total_shards,
        "samples_per_shard": samples_per_shard,
        "years": sorted(list(completed_years)),
        "stats": dict(all_stats),
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("ARXIV CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Total samples: {total_written:,}")
    print(f"Total shards: {total_shards}")
    print(f"Years processed: {sorted(completed_years)}")
    print("\nStats:")
    for k, v in all_stats.items():
        print(f"  {k}: {v:,}")
    print("=" * 70)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Convert S1-MMAlign arxiv to WebDataset (streaming)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all years
  python applications/text/convert_arxiv_streaming.py \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/arxiv_webdataset

  # Resume from year 2020
  python applications/text/convert_arxiv_streaming.py \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/arxiv_webdataset \\
      --start-year 2020

  # Process specific years only
  python applications/text/convert_arxiv_streaming.py \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/arxiv_webdataset \\
      --years 2020 2021 2022
        """,
    )

    parser.add_argument(
        "--output-dir", required=True, help="Output directory for WebDataset"
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=5000,
        help="Samples per shard (default: 5000)",
    )
    parser.add_argument(
        "--start-year", help="Start from this year (skip earlier years)"
    )
    parser.add_argument("--years", nargs="+", help="Process only these specific years")

    args = parser.parse_args()

    convert_arxiv_streaming(
        output_dir=args.output_dir,
        samples_per_shard=args.samples_per_shard,
        start_year=args.start_year,
        years_to_process=args.years,
    )


if __name__ == "__main__":
    main()

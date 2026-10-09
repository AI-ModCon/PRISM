#!/usr/bin/env python3
"""
hydrate_pixmo_pointing.py - Download images for pixmo-points and pixmo-count datasets

These datasets only contain image_url + annotations. This script downloads
the images and creates a hydrated dataset with embedded image bytes.

Now with CHUNKED PROCESSING to avoid memory issues with large datasets.
Each chunk is written to disk immediately, enabling:
- Memory-safe processing of millions of images
- Resume support (skips completed chunks)
- Crash resilience (completed chunks are preserved)

Usage:
    # Hydrate pixmo-points (chunked, with resume support)
    python applications/vision_language/hydrate_pixmo_pointing.py \
        --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points/data \
        --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points-hydrated \
        --workers 64 \
        --chunk-size 50000

    # Hydrate pixmo-count (smaller dataset)
    python applications/vision_language/hydrate_pixmo_pointing.py \
        --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count/data \
        --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count-hydrated \
        --workers 32

Output:
    Creates chunked parquet files with an added 'image' column:
    output_dir/
        chunk_000.parquet
        chunk_001.parquet
        ...
        manifest.json
"""

import argparse
import glob
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global session for connection pooling
_SESSION = None


def init_session(workers=32):
    """Initialize a requests session with connection pooling."""
    global _SESSION
    _SESSION = requests.Session()

    # Reduce retries to speed up - failed URLs shouldn't block progress
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=1,  # Only 1 retry (down from 3)
        backoff_factor=0,  # No backoff delay
        status_forcelist=[500, 502, 503, 504],  # Only retry server errors
    )

    adapter = HTTPAdapter(
        pool_connections=workers, pool_maxsize=workers, max_retries=retry_strategy
    )
    _SESSION.mount("http://", adapter)
    _SESSION.mount("https://", adapter)

    # Disable SSL warnings for speed (images are public)
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_image(url, timeout=5, verify_ssl=False):
    """
    Download image from URL and return bytes.
    Returns (image_bytes, format) or (None, error_msg)

    Uses short timeout (5s) to avoid blocking on slow/dead hosts.
    """
    if not url:
        return None, "No URL"

    try:
        getter = _SESSION.get if _SESSION else requests.get
        response = getter(url, timeout=timeout, verify=verify_ssl)

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"

        # Validate it's a real image
        img_bytes = response.content
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()  # Verify it's a valid image
        except Exception as e:
            return None, f"Invalid image: {e}"

        return img_bytes, "OK"

    except requests.Timeout:
        return None, "Timeout"
    except requests.ConnectionError:
        return None, "Connection error"
    except Exception as e:
        return None, f"Error: {str(e)[:50]}"


def process_row(row_tuple):
    """
    Download image for a single row.

    Args:
        row_tuple: (index, row_dict)

    Returns:
        (index, image_data, status)
        where image_data is {'bytes': bytes, 'path': None} or None
    """
    idx, row = row_tuple
    url = row.get("image_url")

    img_bytes, status = download_image(url)

    if img_bytes is None:
        return idx, None, status
    else:
        return idx, {"bytes": img_bytes, "path": None}, "Downloaded"


def get_completed_chunks(output_dir):
    """Get set of already completed chunk indices."""
    completed = set()
    chunk_files = glob.glob(os.path.join(output_dir, "chunk_*.parquet"))
    for f in chunk_files:
        basename = os.path.basename(f)
        # Extract chunk number from "chunk_000.parquet"
        try:
            chunk_num = int(basename.replace("chunk_", "").replace(".parquet", ""))
            # Verify file is not empty/corrupt
            if os.path.getsize(f) > 0:
                completed.add(chunk_num)
        except ValueError:
            continue
    return completed


def process_chunk(df_chunk, chunk_idx, output_dir, workers):
    """
    Process a single chunk: download images and save to parquet.

    Returns: (num_downloaded, num_failed)
    """
    chunk_size = len(df_chunk)

    # Preallocate for this chunk only
    image_data = [None] * chunk_size
    stats = {"Downloaded": 0, "Failed": 0}

    # Create row tuples for processing
    row_tuples = [(i, row.to_dict()) for i, row in df_chunk.iterrows()]
    # Re-index to 0-based for this chunk
    row_tuples = [(i, row_tuples[i][1]) for i in range(len(row_tuples))]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_row, rt): rt[0] for rt in row_tuples}

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Chunk {chunk_idx:03d}",
            leave=False,
        ):
            local_idx = futures[future]
            try:
                _, img_data, status = future.result()
                image_data[local_idx] = img_data

                if img_data is not None:
                    stats["Downloaded"] += 1
                else:
                    stats["Failed"] += 1
            except Exception:
                stats["Failed"] += 1

    # Add image column to chunk dataframe
    df_chunk = df_chunk.copy()
    df_chunk["image"] = image_data

    # Filter out failed downloads
    df_hydrated = df_chunk[df_chunk["image"].notna()].reset_index(drop=True)

    # Save chunk
    chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:03d}.parquet")
    df_hydrated.to_parquet(chunk_file, index=False)

    return stats["Downloaded"], stats["Failed"], len(df_hydrated)


def hydrate_dataset(input_dir, output_dir, workers=32, chunk_size=50000, limit=None):
    """
    Hydrate a pixmo pointing dataset by downloading images.

    Uses chunked processing to avoid memory issues:
    - Processes data in chunks of `chunk_size` samples
    - Each chunk is written to disk immediately
    - Supports resume (skips completed chunks)
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    init_session(workers)

    # Find parquet files
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files")

    # Load all data (just metadata, not images yet)
    logger.info("Loading parquet files...")
    dfs = []
    for pf in parquet_files:
        # Skip test/validation if we only want train
        if "test" in pf or "validation" in pf:
            logger.info(f"Skipping {os.path.basename(pf)}")
            continue
        df = pd.read_parquet(pf)
        dfs.append(df)
        logger.info(f"  Loaded {os.path.basename(pf)}: {len(df):,} rows")

    if not dfs:
        raise ValueError("No train parquet files loaded")

    df = pd.concat(dfs, ignore_index=True)
    total_rows = len(df)
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Columns: {list(df.columns)}")

    if limit:
        df = df.head(limit)
        total_rows = len(df)
        logger.info(f"Limited to {limit} rows")

    # Calculate chunks
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    logger.info(
        f"Will process in {num_chunks} chunks of up to {chunk_size:,} samples each"
    )

    # Check for completed chunks (resume support)
    completed_chunks = get_completed_chunks(output_dir)
    if completed_chunks:
        logger.info(
            f"Found {len(completed_chunks)} completed chunks: {sorted(completed_chunks)}"
        )
        logger.info("Will skip these chunks (resume mode)")

    # Process each chunk
    total_downloaded = 0
    total_failed = 0
    total_written = 0
    chunk_info = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)

        # Skip if already completed
        if chunk_idx in completed_chunks:
            # Load existing chunk to get stats
            chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:03d}.parquet")
            existing_df = pd.read_parquet(chunk_file)
            chunk_written = len(existing_df)
            total_written += chunk_written
            chunk_info.append(
                {"chunk": chunk_idx, "samples": chunk_written, "status": "resumed"}
            )
            logger.info(
                f"Chunk {chunk_idx:03d}: SKIPPED (already complete, {chunk_written:,} samples)"
            )
            continue

        df_chunk = df.iloc[start_idx:end_idx]

        logger.info(
            f"\nChunk {chunk_idx:03d}/{num_chunks - 1}: rows {start_idx:,}-{end_idx - 1:,} ({len(df_chunk):,} samples)"
        )

        downloaded, failed, written = process_chunk(
            df_chunk, chunk_idx, output_dir, workers
        )

        total_downloaded += downloaded
        total_failed += failed
        total_written += written

        success_rate = downloaded / len(df_chunk) * 100 if len(df_chunk) > 0 else 0

        chunk_info.append(
            {
                "chunk": chunk_idx,
                "samples": written,
                "downloaded": downloaded,
                "failed": failed,
                "success_rate": f"{success_rate:.1f}%",
                "status": "completed",
            }
        )

        logger.info(
            f"  Downloaded: {downloaded:,}, Failed: {failed:,}, "
            f"Written: {written:,} ({success_rate:.1f}% success)"
        )

    # Save manifest
    manifest = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "total_source_rows": total_rows,
        "total_written": total_written,
        "total_downloaded": total_downloaded,
        "total_failed": total_failed,
        "overall_success_rate": f"{total_downloaded / total_rows * 100:.1f}%"
        if total_rows > 0
        else "0%",
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "chunks": chunk_info,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("HYDRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total source rows: {total_rows:,}")
    logger.info(f"Total downloaded: {total_downloaded:,}")
    logger.info(f"Total failed: {total_failed:,}")
    logger.info(f"Total written: {total_written:,}")
    logger.info(
        f"Overall success rate: {total_downloaded / total_rows * 100:.1f}%"
        if total_rows > 0
        else "N/A"
    )
    logger.info(f"Chunks written: {num_chunks}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Manifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Hydrate pixmo-points/pixmo-count by downloading images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hydrate pixmo-points (2.4M images - chunked processing)
  python applications/vision_language/hydrate_pixmo_pointing.py \\
      --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points/data \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points-hydrated \\
      --workers 64 \\
      --chunk-size 50000

  # Resume after crash (automatically skips completed chunks)
  python applications/vision_language/hydrate_pixmo_pointing.py \\
      --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points/data \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points-hydrated \\
      --workers 64

  # Hydrate pixmo-count (37K images - faster)
  python applications/vision_language/hydrate_pixmo_pointing.py \\
      --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count/data \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count-hydrated \\
      --workers 32
        """,
    )

    parser.add_argument(
        "--input-dir", required=True, help="Directory containing parquet files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for hydrated dataset"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of download workers (default: 32)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Samples per chunk (default: 50000). Each chunk is saved immediately to avoid memory issues.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit total samples (for testing)"
    )

    args = parser.parse_args()

    hydrate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

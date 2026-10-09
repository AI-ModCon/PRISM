#!/usr/bin/env python3
"""
convert_parquet_to_webdataset.py - Convert Parquet datasets to WebDataset format

Handles datasets with embedded images in Parquet files (like CoSyn-point, pixmo-points).
Creates TAR shards compatible with the existing WebDataset pipeline.

Usage:
    python scripts/convert_parquet_to_webdataset.py \
        --input-dir /flare/ModCon/ngetty/data/zone_a/CoSyn-point/data \
        --output-dir /flare/ModCon/ngetty/data/zone_a/CoSyn-point_webdataset \
        --dataset-name cosyn-point \
        --images-per-shard 1000 \
        --workers 8

Supported formats:
    - Parquet with embedded image bytes (image column as dict with 'bytes' key)
    - Parquet with image paths (for external images)
    - Various caption/text field names

Output structure:
    output_dir/
        shards/
            {dataset_name}-000000.tar
            {dataset_name}-000001.tar
            ...
        val_shards/
            {dataset_name}-{val_shard_idx}.tar
        manifest.json
"""

import argparse
import glob
import io
import json
import logging
import os
import tarfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_sample_key(idx):
    """Create a unique key for each sample."""
    return f"{idx:08d}"


def extract_image_bytes(image_data, base_path=None):
    """
    Extract image bytes from various formats.

    Handles:
    - Dict with 'bytes' key (HuggingFace format)
    - Dict with 'path' key (external file reference)
    - Raw bytes
    - PIL Image
    - String path
    """
    if image_data is None:
        return None, None

    try:
        # Case 1: Dict with bytes
        if isinstance(image_data, dict):
            if "bytes" in image_data and image_data["bytes"]:
                img_bytes = image_data["bytes"]
                if isinstance(img_bytes, str):
                    # Base64 encoded? Try decoding
                    import base64

                    try:
                        img_bytes = base64.b64decode(img_bytes)
                    except Exception:
                        return None, None
                if not isinstance(img_bytes, bytes):
                    return None, None
                # Detect format
                img = Image.open(io.BytesIO(img_bytes))
                fmt = img.format.lower() if img.format else "jpg"
                return img_bytes, fmt
            elif "path" in image_data and image_data["path"]:
                path = image_data["path"]
                if base_path:
                    path = os.path.join(base_path, path)
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        img_bytes = f.read()
                    ext = os.path.splitext(path)[1].lower().lstrip(".")
                    return img_bytes, ext or "jpg"

        # Case 2: Raw bytes
        elif isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
            fmt = img.format.lower() if img.format else "jpg"
            return image_data, fmt

        # Case 3: PIL Image
        elif isinstance(image_data, Image.Image):
            buf = io.BytesIO()
            fmt = image_data.format or "JPEG"
            image_data.save(buf, format=fmt)
            return buf.getvalue(), fmt.lower()

        # Case 4: String path
        elif isinstance(image_data, str):
            path = image_data
            if base_path:
                path = os.path.join(base_path, path)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    img_bytes = f.read()
                ext = os.path.splitext(path)[1].lower().lstrip(".")
                return img_bytes, ext or "jpg"

    except Exception as e:
        logger.debug(f"Failed to extract image: {e}")
        return None, None

    return None, None


def extract_text(row, text_keys=None):
    """Extract caption/text from row with fallback options."""
    if text_keys is None:
        text_keys = [
            "caption",
            "text",
            "description",
            "label",
            "question",
            "answer",
            "instruction",
            "response",
            "output",
            "input",
        ]

    for key in text_keys:
        if key in row and row[key]:
            val = row[key]
            if isinstance(val, str):
                return val
            elif isinstance(val, list):
                return " ".join(str(v) for v in val)
            else:
                return str(val)

    return ""


def extract_metadata(row, image_key="image", text_key="caption"):
    """Extract useful metadata from row."""
    metadata = {}

    # Standard fields
    for key in ["id", "image_id", "sample_id", "idx"]:
        if key in row:
            metadata["id"] = str(row[key])
            break

    # Points data (for pointing datasets)
    for key in ["points", "point", "coordinates", "bbox", "bboxes"]:
        if key in row and row[key] is not None:
            metadata[key] = row[key]

    # Labels
    for key in ["label", "labels", "category", "class"]:
        if key in row and row[key] is not None:
            metadata[key] = row[key]

    return metadata


def process_row(row, idx, base_path=None, image_key="image"):
    """
    Process a single row and return data for TAR.
    Returns: (key, image_bytes, ext, caption, metadata) or None if failed
    """
    try:
        # Extract image
        image_data = row.get(image_key)
        img_bytes, ext = extract_image_bytes(image_data, base_path)

        if img_bytes is None:
            return None

        # Validate image
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()
        except Exception as e:
            logger.debug(f"Invalid image at idx {idx}: {e}")
            return None

        # Extract text
        caption = extract_text(row)

        # Extract metadata
        metadata = extract_metadata(row)
        metadata["original_idx"] = idx

        return img_bytes, ext, caption, metadata

    except Exception as e:
        logger.debug(f"Error processing row {idx}: {e}")
        return None


def write_shard(samples, shard_path, start_idx, dataset_name):
    """
    Write a list of samples to a TAR shard.

    Each sample in the TAR has:
    - {key}.{ext} - Image file
    - {key}.txt - Caption
    - {key}.json - Metadata
    """
    written = 0
    with tarfile.open(shard_path, "w") as tar:
        for i, sample in enumerate(samples):
            if sample is None:
                continue

            image_bytes, ext, caption, metadata = sample
            key = create_sample_key(start_idx + i)

            # Normalize extension
            if ext in ["jpeg"]:
                ext = "jpg"
            elif ext not in ["jpg", "png", "webp", "gif"]:
                ext = "jpg"

            # Add image
            img_info = tarfile.TarInfo(name=f"{key}.{ext}")
            img_info.size = len(image_bytes)
            tar.addfile(img_info, io.BytesIO(image_bytes))

            # Add caption
            caption_bytes = (caption or "").encode("utf-8")
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(caption_bytes)
            tar.addfile(txt_info, io.BytesIO(caption_bytes))

            # Add metadata
            meta_bytes = json.dumps(metadata).encode("utf-8")
            meta_info = tarfile.TarInfo(name=f"{key}.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, io.BytesIO(meta_bytes))

            written += 1

    return written


def detect_image_key(df):
    """Detect which column contains images."""
    candidates = ["image", "img", "photo", "picture", "frame"]
    for col in candidates:
        if col in df.columns:
            # Check if it looks like image data
            sample = df[col].iloc[0] if len(df) > 0 else None
            if sample is not None:
                if isinstance(sample, (dict, bytes)) or (
                    isinstance(sample, str) and os.path.exists(sample)
                ):
                    return col

    # Check for columns that might contain images
    for col in df.columns:
        sample = df[col].iloc[0] if len(df) > 0 else None
        if isinstance(sample, dict) and ("bytes" in sample or "path" in sample):
            return col

    return "image"  # Default


def load_parquet_with_nested_data(parquet_files):
    """
    Load parquet files that may contain nested data structures.

    Uses iter_batches with batch_size=1 to work around PyArrow's
    limitation with nested data in chunked arrays.

    Returns: list of row dicts, column names
    """
    all_rows = []
    columns = None

    for pf_path in tqdm(parquet_files, desc="Loading parquet files"):
        try:
            pf = pq.ParquetFile(pf_path)

            if columns is None:
                columns = pf.schema_arrow.names

            # Use iter_batches to handle nested data
            num_rows = pf.metadata.num_rows
            with tqdm(
                total=num_rows, desc=f"  {os.path.basename(pf_path)}", leave=False
            ) as pbar:
                for batch in pf.iter_batches(batch_size=100):
                    for i in range(len(batch)):
                        row = {col: batch[col][i].as_py() for col in batch.column_names}
                        all_rows.append(row)
                    pbar.update(len(batch))

        except Exception as e:
            logger.warning(f"Failed to load {pf_path}: {e}")
            continue

    return all_rows, columns


def try_load_parquet_standard(parquet_files):
    """
    Try loading parquet files with standard pandas method.
    Returns (dataframe, success) tuple.
    """
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            # If any file fails, return failure
            return None, False, str(e)

    if dfs:
        return pd.concat(dfs, ignore_index=True), True, None
    return None, False, "No files loaded"


def convert_parquet_to_webdataset(
    input_dir,
    output_dir,
    dataset_name,
    images_per_shard=1000,
    workers=8,
    val_split=0.01,
    base_image_path=None,
):
    """Convert Parquet files to WebDataset shards."""

    # Create output directories
    shards_dir = os.path.join(output_dir, "shards")
    val_shards_dir = os.path.join(output_dir, "val_shards")
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(val_shards_dir, exist_ok=True)

    # Find Parquet files
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files in {input_dir}")

    # Try standard pandas loading first
    logger.info("Loading parquet files...")
    full_df, success, error_msg = try_load_parquet_standard(parquet_files)

    use_row_list = False
    all_rows: list = []  # Will hold rows if using row-by-row loading

    if success and full_df is not None:
        total_samples = len(full_df)
        logger.info(f"Loaded with pandas: {total_samples:,} samples")
        logger.info(f"Columns: {list(full_df.columns)}")
        # Detect image column
        image_key = detect_image_key(full_df)
    else:
        # Fall back to row-by-row loading for nested data
        logger.warning(f"Standard loading failed: {error_msg}")
        logger.info("Falling back to row-by-row loading for nested data...")

        all_rows, columns = load_parquet_with_nested_data(parquet_files)

        if not all_rows:
            raise ValueError("No valid parquet files loaded")

        use_row_list = True
        full_df = None  # Explicitly set to None
        total_samples = len(all_rows)
        logger.info(f"Loaded with iter_batches: {total_samples:,} samples")
        logger.info(f"Columns: {columns}")

        # Detect image column from first row
        first_row = all_rows[0]
        image_key = "image"  # Default
        for col in ["image", "img", "photo", "picture"]:
            if col in first_row and first_row[col] is not None:
                image_key = col
                break

    logger.info(f"Using image column: '{image_key}'")

    # Split train/val
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    # Shuffle indices for random split
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    logger.info(f"Train samples: {train_size:,}, Val samples: {val_size:,}")

    # Calculate shard counts
    num_train_shards = (train_size + images_per_shard - 1) // images_per_shard
    num_val_shards = (
        max(1, (val_size + images_per_shard - 1) // images_per_shard)
        if val_size > 0
        else 0
    )

    logger.info(
        f"Creating {num_train_shards} training shards, {num_val_shards} validation shards"
    )

    # Initialize manifest
    manifest = {
        "dataset": dataset_name,
        "total_samples": total_samples,
        "train_samples": train_size,
        "val_samples": val_size,
        "images_per_shard": images_per_shard,
        "num_train_shards": num_train_shards,
        "num_val_shards": num_val_shards,
        "num_shards": num_train_shards,  # For compatibility
        "shards": [],
        "val_shards": [],
    }

    # Helper function to get row data
    def get_row_dict(idx):
        """Get row as dictionary, handling both DataFrame and list cases."""
        if use_row_list:
            return all_rows[idx]
        else:
            # full_df is guaranteed to be a DataFrame when use_row_list is False
            assert full_df is not None
            return full_df.iloc[idx].to_dict()

    # Process training shards
    total_written = 0
    logger.info("Processing training shards...")

    for shard_idx in tqdm(range(num_train_shards), desc="Training shards"):
        start_idx = shard_idx * images_per_shard
        end_idx = min(start_idx + images_per_shard, train_size)

        batch_indices = train_indices[start_idx:end_idx]

        # Process samples
        samples = []
        for local_idx, original_idx in enumerate(batch_indices):
            global_idx = start_idx + local_idx
            row_dict = get_row_dict(original_idx)
            sample = process_row(row_dict, global_idx, base_image_path, image_key)
            samples.append(sample)

        # Write shard
        shard_name = f"{dataset_name}-{shard_idx:06d}.tar"
        shard_path = os.path.join(shards_dir, shard_name)
        written = write_shard(samples, shard_path, start_idx, dataset_name)

        shard_size = os.path.getsize(shard_path)
        manifest["shards"].append(
            {
                "name": shard_name,
                "samples": written,
                "size_bytes": shard_size,
                "start_idx": start_idx,
            }
        )

        total_written += written

    # Process validation shards
    val_written = 0
    if val_size > 0:
        logger.info("Processing validation shards...")

        for shard_idx in tqdm(range(num_val_shards), desc="Validation shards"):
            start_idx = shard_idx * images_per_shard
            end_idx = min(start_idx + images_per_shard, val_size)

            batch_indices = val_indices[start_idx:end_idx]

            samples = []
            for local_idx, original_idx in enumerate(batch_indices):
                global_idx = num_train_shards * images_per_shard + start_idx + local_idx
                row_dict = get_row_dict(original_idx)
                sample = process_row(row_dict, global_idx, base_image_path, image_key)
                samples.append(sample)

            # Val shard naming continues from train
            val_shard_idx = num_train_shards + shard_idx
            shard_name = f"{dataset_name}-{val_shard_idx:06d}.tar"
            shard_path = os.path.join(val_shards_dir, shard_name)
            written = write_shard(samples, shard_path, start_idx, dataset_name)

            shard_size = os.path.getsize(shard_path)
            manifest["val_shards"].append(
                {
                    "name": shard_name,
                    "samples": written,
                    "size_bytes": shard_size,
                    "start_idx": start_idx,
                }
            )

            val_written += written

    # Update manifest
    manifest["total_written"] = total_written
    manifest["val_written"] = val_written
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Conversion Complete: {dataset_name}")
    print(f"{'=' * 50}")
    print(f"Total samples: {total_samples:,}")
    print(f"Training: {total_written:,} written ({num_train_shards} shards)")
    print(f"Validation: {val_written:,} written ({num_val_shards} shards)")
    print(f"Skipped (corrupt/missing): {total_samples - total_written - val_written:,}")
    print(f"\nOutput: {output_dir}")
    print(f"Manifest: {manifest_path}")

    # Multi-node distribution info
    print("\n=== Multi-Node Distribution ===")
    for n_nodes in [1, 2, 4, 8, 12]:
        shards_per_node = num_train_shards // n_nodes
        remainder = num_train_shards % n_nodes
        print(
            f"  {n_nodes:2d} nodes: {shards_per_node} shards/node"
            + (f" (+{remainder} remainder)" if remainder else "")
        )

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet dataset to WebDataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert CoSyn-point
  python scripts/convert_parquet_to_webdataset.py \\
      --input-dir /flare/ModCon/ngetty/data/zone_a/CoSyn-point/data \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/CoSyn-point_webdataset \\
      --dataset-name cosyn-point

  # Convert pixmo-points
  python scripts/convert_parquet_to_webdataset.py \\
      --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points/data \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points_webdataset \\
      --dataset-name pixmo-points

  # Convert pixmo-count
  python scripts/convert_parquet_to_webdataset.py \\
      --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count/data \\
      --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count_webdataset \\
      --dataset-name pixmo-count
        """,
    )

    parser.add_argument(
        "--input-dir", required=True, help="Directory containing Parquet files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for WebDataset"
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name prefix for shard files (e.g., 'cosyn-point')",
    )
    parser.add_argument(
        "--images-per-shard",
        type=int,
        default=1000,
        help="Images per TAR shard (default: 1000)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel workers (default: 8)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.01,
        help="Validation split ratio (default: 0.01)",
    )
    parser.add_argument(
        "--base-image-path",
        default=None,
        help="Base path for external image references",
    )

    args = parser.parse_args()

    convert_parquet_to_webdataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        images_per_shard=args.images_per_shard,
        workers=args.workers,
        val_split=args.val_split,
        base_image_path=args.base_image_path,
    )


if __name__ == "__main__":
    main()

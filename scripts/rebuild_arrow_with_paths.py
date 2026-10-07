#!/usr/bin/env python3
"""
rebuild_arrow_with_paths.py - Rebuild Arrow dataset with local_path column

This script takes the original Arrow files and existing downloaded images,
and creates a new Arrow dataset with the local_path column correctly populated.

Usage:
    python scripts/rebuild_arrow_with_paths.py \
        --source-arrow-dir /path/to/original/arrow \
        --images-dir /path/to/downloaded/images \
        --output-dir /path/to/output
"""

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm


def get_filename_from_url(url):
    """Extract filename from URL."""
    if not url:
        return None
    return os.path.basename(url)

def rebuild_dataset(source_arrow_dir, images_dir, output_dir):
    """Rebuild Arrow dataset with local_path column."""
    
    # 1. Build index of available images
    print(f"Scanning images directory: {images_dir}")
    available_images = set()
    for f in tqdm(os.listdir(images_dir), desc="Indexing images"):
        available_images.add(f)
    print(f"Found {len(available_images):,} images")
    
    # 2. Load source Arrow files
    # Note: glob.glob() hangs on dfuse/DAOS mounts — use os.listdir() + filter.
    print(f"\nLoading source Arrow files from: {source_arrow_dir}")
    arrow_files = sorted(
        os.path.join(source_arrow_dir, f)
        for f in os.listdir(source_arrow_dir)
        if f.endswith(".arrow")
    )
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {source_arrow_dir}")
    print(f"Found {len(arrow_files)} Arrow shards")
    
    ds = load_dataset("arrow", data_files=arrow_files, split="train")
    print(f"Dataset size: {len(ds):,} samples")
    
    # 3. Map each item to local path
    print("\nMapping URLs to local paths...")
    local_paths = []
    matched = 0
    missing = 0
    
    for item in tqdm(ds, desc="Mapping"):
        url = item.get('image_url', '')
        filename = get_filename_from_url(url)
        
        if filename and filename in available_images:
            local_paths.append(os.path.join(images_dir, filename))
            matched += 1
        else:
            local_paths.append(None)
            missing += 1
    
    print("\nMapping Results:")
    print(f"  Matched: {matched:,} ({matched/len(ds)*100:.1f}%)")
    print(f"  Missing: {missing:,} ({missing/len(ds)*100:.1f}%)")
    
    # 4. Add local_path column
    print("\nAdding local_path column...")
    ds_with_path = ds.add_column("local_path", local_paths)
    
    # 5. Drop problematic 'transcripts' column (has List type that causes loading issues)
    if "transcripts" in ds_with_path.column_names:
        print("Dropping 'transcripts' column to avoid schema issues...")
        ds_with_path = ds_with_path.remove_columns(["transcripts"])
    
    # 6. Filter out items without local path
    print("Filtering items without local paths...")
    ds_final = ds_with_path.filter(lambda x: x['local_path'] is not None)
    print(f"Final dataset: {len(ds_final):,} samples")
    
    # 7. Save
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving to {output_dir}...")
    ds_final.save_to_disk(output_dir)
    print("✅ Done!")
    
    return len(ds_final)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-arrow-dir", required=True, 
                        help="Directory containing original Arrow files (with image_url)")
    parser.add_argument("--images-dir", required=True,
                        help="Directory containing downloaded images")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for new Arrow dataset")
    
    args = parser.parse_args()
    
    rebuild_dataset(args.source_arrow_dir, args.images_dir, args.output_dir)

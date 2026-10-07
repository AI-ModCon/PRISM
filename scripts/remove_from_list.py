#!/usr/bin/env python3
"""
remove_from_list.py - Remove files listed in corrupt_images.txt

Fast script that reads the corrupt list and removes those files directly,
without re-validating everything.

Usage:
    python scripts/remove_from_list.py \
        --list-file /path/to/corrupt_images.txt \
        --images-dir /path/to/images \
        --max-pixels 89478485  # Also remove images larger than this
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm


def check_size(filepath, max_pixels):
    """Check if image exceeds max pixels."""
    try:
        with Image.open(filepath) as img:
            w, h = img.size
            if w * h > max_pixels:
                return filepath, True
        return filepath, False
    except Exception:
        return filepath, False  # Already in corrupt list

def remove_from_list(list_file, images_dir, dry_run=False, max_pixels=None, workers=32):
    """Remove files listed in the corrupt list."""
    
    # Read corrupt list
    files_to_remove = set()
    print(f"Reading corrupt list: {list_file}")
    with open(list_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                filename = parts[0]
                files_to_remove.add(filename)
    
    print(f"Found {len(files_to_remove):,} files in corrupt list")
    
    # Optionally scan for oversized images
    if max_pixels:
        print(f"\nScanning for oversized images (>{max_pixels:,} pixels)...")
        all_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                     if os.path.isfile(os.path.join(images_dir, f))]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(check_size, fp, max_pixels): fp for fp in all_files}
            oversized = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Checking sizes"):
                filepath, is_oversized = future.result()
                if is_oversized:
                    files_to_remove.add(os.path.basename(filepath))
                    oversized += 1
        print(f"Found {oversized:,} additional oversized files")
    
    print(f"\nTotal files to remove: {len(files_to_remove):,}")
    
    if dry_run:
        print("[DRY RUN] Would remove these files")
        return
    
    # Remove files
    removed = 0
    missing = 0
    errors = 0
    
    for filename in tqdm(files_to_remove, desc="Removing"):
        filepath = os.path.join(images_dir, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                removed += 1
            except Exception as e:
                print(f"Error removing {filename}: {e}")
                errors += 1
        else:
            missing += 1
    
    print("\n=== Removal Results ===")
    print(f"Removed: {removed:,}")
    print(f"Already missing: {missing:,}")
    print(f"Errors: {errors:,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", required=True, help="Path to corrupt_images.txt")
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--max-pixels", type=int, default=None, 
                        help="Also remove images larger than this many pixels (default: don't check)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--workers", type=int, default=32, help="Workers for size check")
    
    args = parser.parse_args()
    
    remove_from_list(
        args.list_file,
        args.images_dir,
        dry_run=args.dry_run,
        max_pixels=args.max_pixels,
        workers=args.workers
    )

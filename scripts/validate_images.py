#!/usr/bin/env python3
"""
validate_images.py - Validate and remove corrupt images

Scans an image directory, attempts to open each file with PIL,
and removes any that are corrupt or can't be identified.

Usage:
    python scripts/validate_images.py \
        --images-dir /path/to/images \
        --dry-run  # Show what would be removed without deleting
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image(filepath):
    """
    Validate a single image file.
    Returns (filepath, is_valid, error_msg)
    """
    try:
        with Image.open(filepath) as img:
            # Force load to detect truncated files
            img.load()
            # Check if it has valid dimensions
            if img.size[0] == 0 or img.size[1] == 0:
                return filepath, False, "Zero dimensions"
        return filepath, True, None
    except Exception as e:
        return filepath, False, str(e)

def validate_images(images_dir, workers=32, dry_run=True, remove_corrupt=False):
    """Validate all images in directory."""
    
    # Get list of image files
    print(f"Scanning directory: {images_dir}")
    image_files = []
    for f in os.listdir(images_dir):
        filepath = os.path.join(images_dir, f)
        if os.path.isfile(filepath):
            image_files.append(filepath)
    
    print(f"Found {len(image_files):,} files to validate")
    
    valid_count = 0
    corrupt_count = 0
    corrupt_files = []
    
    print(f"Validating with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(validate_image, fp): fp for fp in image_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating"):
            filepath, is_valid, error = future.result()
            if is_valid:
                valid_count += 1
            else:
                corrupt_count += 1
                corrupt_files.append((filepath, error))
    
    print("\n=== Validation Results ===")
    print(f"Valid:   {valid_count:,} ({valid_count/len(image_files)*100:.1f}%)")
    print(f"Corrupt: {corrupt_count:,} ({corrupt_count/len(image_files)*100:.1f}%)")
    
    if corrupt_count > 0:
        print("\nSample corrupt files (first 10):")
        for fp, err in corrupt_files[:10]:
            print(f"  {os.path.basename(fp)}: {err}")
    
    if remove_corrupt and corrupt_count > 0:
        if dry_run:
            print(f"\n[DRY RUN] Would remove {corrupt_count:,} corrupt files")
        else:
            print(f"\nRemoving {corrupt_count:,} corrupt files...")
            removed = 0
            for fp, _ in tqdm(corrupt_files, desc="Removing"):
                try:
                    os.remove(fp)
                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {fp}: {e}")
            print(f"Removed {removed:,} files")
    
    # Write corrupt list to file for reference
    corrupt_list_path = os.path.join(os.path.dirname(images_dir), "corrupt_images.txt")
    with open(corrupt_list_path, "w") as f:
        for fp, err in corrupt_files:
            f.write(f"{os.path.basename(fp)}\t{err}\n")
    print(f"\nCorrupt file list saved to: {corrupt_list_path}")
    
    return valid_count, corrupt_count, corrupt_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    parser.add_argument("--remove", action="store_true", help="Actually remove corrupt files")
    
    args = parser.parse_args()
    
    validate_images(
        args.images_dir, 
        workers=args.workers, 
        dry_run=args.dry_run,
        remove_corrupt=args.remove
    )

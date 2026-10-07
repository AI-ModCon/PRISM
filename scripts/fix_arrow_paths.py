#!/usr/bin/env python3
"""
fix_arrow_paths.py - Permanently fix image paths in Arrow dataset files.

This script rewrites the Arrow dataset to update stale 'local_path' entries
to point to the correct location in the new dataset directory.

Usage:
    python scripts/fix_arrow_paths.py --dataset-dir /path/to/pixmo_cap --dry-run
    python scripts/fix_arrow_paths.py --dataset-dir /path/to/pixmo_cap
"""

import argparse
import os

from datasets import load_from_disk
from tqdm import tqdm


def repair_path(old_path: str, images_dir: str) -> str:
    """
    Repair a stale path by extracting the filename and pointing to new images dir.
    
    Args:
        old_path: The original (possibly stale) path from the Arrow file
        images_dir: The new images directory (e.g., /path/to/pixmo_cap/images)
    
    Returns:
        The repaired absolute path
    """
    if not old_path:
        return old_path
    
    # Check if path already points to the correct images_dir
    # Normalize paths to handle trailing slashes etc.
    if os.path.realpath(old_path).startswith(os.path.realpath(images_dir)):
        if os.path.exists(old_path):
            return old_path
    
    # Extract filename from the old path
    # Handle both "images/abc.jpg" and "/old/path/images/abc.jpg"
    if "images/" in old_path:
        filename = old_path.split("images/")[-1]
    else:
        filename = os.path.basename(old_path)
    
    new_path = os.path.join(images_dir, filename)
    return new_path


def fix_dataset_paths(dataset_dir: str, images_source_dir: str = None, dry_run: bool = False):
    """
    Fix all paths in the Arrow dataset.
    """
    if images_source_dir:
        images_dir = images_source_dir
    else:
        images_dir = os.path.join(dataset_dir, "images")
    
    print(f"Target Images Directory: {images_dir}")
    
    if not os.path.isdir(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    print(f"Loading dataset from: {dataset_dir}")
    ds = load_from_disk(dataset_dir)
    
    print(f"Dataset size: {len(ds):,} samples")
    print(f"Columns: {ds.column_names}")
    
    # Check if local_path column exists
    if 'local_path' not in ds.column_names:
        print("Warning: 'local_path' column not found. Checking for alternative columns...")
        print(f"Available columns: {ds.column_names}")
        return
    
    # Analyze current paths
    sample_paths = ds['local_path'][:10]
    print("\nSample current paths:")
    for p in sample_paths:
        print(f"  {p}")
    
    # Count repairs needed
    repairs_needed = 0
    already_valid = 0
    
    print("\nAnalyzing paths...")
    for path in tqdm(ds['local_path'], desc="Checking paths"):
        if path and os.path.exists(path):
            already_valid += 1
        else:
            repairs_needed += 1
    
    print("\nPath Analysis:")
    print(f"  Already valid: {already_valid:,}")
    print(f"  Repairs needed: {repairs_needed:,}")
    
    if repairs_needed == 0:
        print("\n✅ All paths are already valid. No repairs needed.")
        return
    
    if dry_run:
        print("\n[DRY RUN] Would repair paths. Run without --dry-run to apply changes.")
        # Show sample repairs
        print("\nSample repairs:")
        count = 0
        for path in ds['local_path']:
            if path and not os.path.exists(path):
                new_path = repair_path(path, images_dir)
                print(f"  {path}")
                print(f"    -> {new_path}")
                exists = "✅" if os.path.exists(new_path) else "❌"
                print(f"    {exists}")
                count += 1
                if count >= 5:
                    break
        return
    
    # Apply repairs
    print("\nApplying path repairs...")
    
    def repair_example(example):
        example['local_path'] = repair_path(example['local_path'], images_dir)
        return example
    
    ds_fixed = ds.map(repair_example, desc="Repairing paths")
    
    # Backup original
    backup_dir = dataset_dir + "_backup"
    if not os.path.exists(backup_dir):
        print(f"\nBacking up original to: {backup_dir}")
        os.rename(dataset_dir, backup_dir)
        os.makedirs(dataset_dir)
        # Copy images symlink or directory
        os.symlink(os.path.join(backup_dir, "images"), images_dir)
    
    # Save fixed dataset
    print(f"\nSaving fixed dataset to: {dataset_dir}")
    ds_fixed.save_to_disk(dataset_dir)
    
    # Verify
    print("\nVerifying repairs...")
    ds_verify = load_from_disk(dataset_dir)
    valid_count = sum(1 for p in ds_verify['local_path'] if p and os.path.exists(p))
    print(f"  Valid paths after repair: {valid_count:,} / {len(ds_verify):,}")
    
    print("\n✅ Done! Dataset paths have been permanently fixed.")


def filter_valid_samples(dataset_dir: str, images_source_dir: str = None, dry_run: bool = False):
    """
    Filter dataset to only include samples with valid, existing image paths.
    This removes samples where images are missing entirely.
    """
    if images_source_dir:
        images_dir = images_source_dir
    else:
        images_dir = os.path.join(dataset_dir, "images")
    
    print(f"Target Images Directory: {images_dir}")
    print(f"Loading dataset from: {dataset_dir}")
    ds = load_from_disk(dataset_dir)
    
    print(f"Original dataset size: {len(ds):,} samples")
    
    # Check for local_path column
    if 'local_path' not in ds.column_names:
        print("Error: 'local_path' column not found.")
        return
    
    # Count valid vs invalid
    print("\nAnalyzing image availability...")
    valid_indices = []
    missing_count = 0
    
    # Caching existence check for performance if needed, but OS cache helps.
    
    for i, path in enumerate(tqdm(ds['local_path'], desc="Checking images")):
        # Check against the INTENDED target path
        if path:
            # We want to check if the file exists AT THE NEW LOCATION
            # Or if the current path is already valid AND points to the right place
            
            # Construct where it SHOULD be
            if "images/" in path:
                suffix = path.split("images/")[-1]
            else:
                suffix = os.path.basename(path)
            
            target_path = os.path.join(images_dir, suffix)
            
            if os.path.exists(target_path):
                valid_indices.append(i)
            else:
                missing_count += 1
        else:
            missing_count += 1
    
    print("\nImage Analysis:")
    print(f"  Valid samples: {len(valid_indices):,}")
    print(f"  Missing images: {missing_count:,}")
    print(f"  Retention rate: {100*len(valid_indices)/len(ds):.1f}%")
    
    if dry_run:
        print("\n[DRY RUN] Would filter dataset. Run with --filter (not --dry-run) to apply.")
        return
    
    # Create filtered dataset
    print("\nFiltering dataset...")
    ds_filtered = ds.select(valid_indices)
    
    # Also repair paths in the filtered set
    print("Repairing paths in filtered set...")
    
    def repair_example(example):
        path = example['local_path']
        if path:
            if "images/" in path:
                suffix = path.split("images/")[-1]
            else:
                suffix = os.path.basename(path)
            example['local_path'] = os.path.join(images_dir, suffix)
        return example
    
    ds_fixed = ds_filtered.map(repair_example, desc="Repairing paths")
    
    # Backup and save
    backup_dir = dataset_dir + "_original"
    if not os.path.exists(backup_dir):
        print(f"\nBacking up original to: {backup_dir}")
        import shutil
        # Copy just the arrow files, not images
        os.makedirs(backup_dir)
        for f in os.listdir(dataset_dir):
            if f.endswith('.arrow') or f.endswith('.json'):
                shutil.copy2(os.path.join(dataset_dir, f), backup_dir)
    
    print(f"\nSaving filtered dataset to: {dataset_dir}")
    ds_fixed.save_to_disk(dataset_dir)
    
    print(f"\n✅ Done! Filtered from {len(ds):,} to {len(ds_fixed):,} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix Arrow dataset paths")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory (e.g., /flare/ModCon/ngetty/data/zone_a/pixmo_cap)"
    )
    parser.add_argument(
        "--images-source-dir",
        type=str,
        required=False,
        help="Path to the directory containing actual images (if different from dataset_dir/images)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze and show what would be changed"
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter out samples with missing images (recommended)"
    )
    
    args = parser.parse_args()
    
    if args.filter:
        filter_valid_samples(args.dataset_dir, args.images_source_dir, args.dry_run)
    else:
        fix_dataset_paths(args.dataset_dir, args.images_source_dir, args.dry_run)

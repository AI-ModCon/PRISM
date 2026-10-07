#!/usr/bin/env python3
"""
prepare_pixmo_dataset.py - Hydrate PixMo-Cap dataset (Download & Link)

This script creates a fully local version of the PixMo-Cap dataset by:
1. Loading the original Arrow files (with URLs).
2. Downloading missing images (or linking from an existing cache).
3. Saving a new Arrow dataset with valid 'local_path' entries.

Usage:
    python scripts/prepare_pixmo_dataset.py \
        --input-arrow-dir /path/to/hf/cache \
        --output-dir /path/to/my/dataset \
        --cache-image-dir /path/to/existing/images \
        --workers 16
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm


def get_filename_from_url(url):
    """
    Generate a filename from the URL. 
    PixMo often uses MD5 hashes in filenames, or we can just use basename.
    """
    if not url:
        return None
    # Just use basename to match typical behavior
    return os.path.basename(url)

# Optional global session for prepare_dataset to use
_SESSION = None

def process_item(item, output_images_dir, cache_dirs=None, verify_ssl=True):
    """
    Ensure image exists locally. Return local path or None.
    """
    if cache_dirs is None:
        cache_dirs = []
    url = item.get('image_url')
    if not url:
        return None, "No URL"

    filename = get_filename_from_url(url)
    target_path = os.path.join(output_images_dir, filename)
    
    # 1. Check Target
    if os.path.exists(target_path):
        return target_path, "Exists"
    
    # 2. Check Caches
    for cache_dir in cache_dirs:
        if not cache_dir:
            continue
        cache_path = os.path.join(cache_dir, filename)
        if os.path.exists(cache_path):
            # Symlink or Copy? Symlink is faster and saves space.
            try:
                os.symlink(cache_path, target_path)
                return target_path, "Linked"
            except OSError:
                # Fallback to copy if symlink fails (cross-fs)
                import shutil
                shutil.copy2(cache_path, target_path)
                return target_path, "Copied"

    # 3. Download
    try:
        getter = _SESSION.get if _SESSION else requests.get
        response = getter(url, timeout=10, verify=verify_ssl)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.content)
            return target_path, "Downloaded"
        else:
            return None, f"HTTP {response.status_code}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def prepare_dataset(input_arrow_dir, output_dir, cache_image_dirs, workers=8, hf_id=None, limit=None, split="train", verify_ssl=False):
    # Setup Dirs
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Initialize global session with pool size matching worker count
    global _SESSION
    _SESSION = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    _SESSION.mount('http://', adapter)
    _SESSION.mount('https://', adapter)

    if not verify_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    ds = None
    
    if hf_id:
        print(f"Loading dataset from HuggingFace: {hf_id} (split={split})")
        from datasets import load_dataset
        ds = load_dataset(hf_id, split=split)
    elif input_arrow_dir:
        print(f"Loading dataset from: {input_arrow_dir}")
        # Check if it's a directory with arrow files (HF Cache)
        if os.path.isdir(input_arrow_dir):
            import glob
            arrow_files = glob.glob(os.path.join(input_arrow_dir, "*.arrow"))
            if not arrow_files:
                raise FileNotFoundError(f"No .arrow files found in {input_arrow_dir}")
            print(f"Found {len(arrow_files)} Arrow shards. Loading...")
            # Load as Arrow dataset
            from datasets import load_dataset
            ds = load_dataset("arrow", data_files=arrow_files, split=split)
        else:
            # Fallback for single file
            from datasets import load_dataset
            ds = load_dataset("arrow", data_files=input_arrow_dir, split=split)
            
    if ds is None:
        raise ValueError("Failed to load dataset")

    if limit:
        print(f"Limiting to {limit} samples")
        ds = ds.select(range(min(limit, len(ds))))

    print(f"Dataset size: {len(ds):,} samples")
    
    # Check if we need to add local_path or if it exists
    
    stats = {"Exists": 0, "Linked": 0, "Copied": 0, "Downloaded": 0, "Failed": 0}
    
    print(f"Processing images with {workers} workers (SSL Verify: {verify_ssl})...")
    print(f"Cache Directories: {cache_image_dirs}")
    
    def task(item):
        path, status = process_item(item, output_images_dir, cache_image_dirs, verify_ssl=verify_ssl)
        return path, status

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Using list(tqdm(executor.map)) can be slow to initialize for 2M items.
        # Let's ensure it iterates lazily.
        for res in tqdm(executor.map(task, ds), total=len(ds), desc="Hydrating Images"):
            results.append(res[0])
            status = res[1]
            if "HTTP" in status or "Error" in status or "No URL" in status:
                stats["Failed"] += 1
            else:
                stats[status] = stats.get(status, 0) + 1

    print("\nHydration Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
        
    # Create new dataset with local_path
    print("\nCreating new Arrow dataset...")
    
    # Filter out failed items? Or keep them with None local_path?
    # Better to filter so training doesn't crash
    
    # Add column
    ds_with_path = ds.add_column("local_path", results)
    
    # Filter
    ds_final = ds_with_path.filter(lambda x: x['local_path'] is not None)
    print(f"Filtered Dataset: {len(ds_final):,} / {len(ds):,} ({len(ds_final)/len(ds)*100:.1f}%)")
    
    print(f"Saving to {output_dir}...")
    ds_final.save_to_disk(output_dir)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-arrow-dir", help="Directory containing Arrow files (optional if --hf-id is used)")
    parser.add_argument("--hf-id", help="HuggingFace Dataset ID (e.g. allenai/pixmo-cap) to load directly")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-image-dir", nargs="*", help="List of directories to check for existing images")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    parser.add_argument("--split", default="train", help="Dataset split to load")
    
    args = parser.parse_args()

    if not args.input_arrow_dir and not args.hf_id:
        parser.error("Must provide either --input-arrow-dir or --hf-id")
    
    # If cache_image_dirs is None, it defaults to empty list [] in function
    if args.cache_image_dir is None:
        args.cache_image_dir = []
        
    prepare_dataset(
        args.input_arrow_dir, 
        args.output_dir, 
        args.cache_image_dir, 
        args.workers,
        hf_id=args.hf_id,
        limit=args.limit,
        split=args.split
    )

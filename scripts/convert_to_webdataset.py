#!/usr/bin/env python3
"""
convert_to_webdataset.py - Convert image dataset to WebDataset format

Creates TAR shards optimized for distributed multi-node training:
- Each shard contains N images with captions
- Shards are named with zero-padded indices for easy splitting
- Includes manifest file for shard assignment

Usage:
    python scripts/convert_to_webdataset.py \
        --arrow-dir /path/to/arrow/dataset \
        --images-dir /path/to/images \
        --output-dir /path/to/webdataset \
        --images-per-shard 1000 \
        --workers 8

Output structure:
    output_dir/
        shards/
            pixmo-000000.tar
            pixmo-000001.tar
            ...
        manifest.json  (shard info for node assignment)
"""

import argparse
import glob
import io
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def create_sample_key(idx):
    """Create a unique key for each sample."""
    return f"{idx:08d}"

def process_sample(item, images_dir):
    """
    Process a single sample and return bytes for TAR.
    Returns: (key, image_bytes, ext, caption, metadata) or None if failed
    """
    local_path = item.get('local_path')
    caption = item.get('caption', '')
    image_url = item.get('image_url', '')
    
    if not local_path or not os.path.exists(local_path):
        return None
    
    try:
        # Read image bytes directly (preserve original format)
        with open(local_path, 'rb') as f:
            image_bytes = f.read()
        
        # Detect extension from file
        ext = os.path.splitext(local_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            ext = '.jpg'  # Default fallback
        ext = ext.lstrip('.')
        
        # Validate image can be opened
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            return None
        
        # Create metadata
        metadata = {
            'image_url': image_url,
            'local_path': local_path,
        }
        
        return image_bytes, ext, caption, metadata
        
    except Exception:
        return None

def write_shard(samples, shard_path, start_idx):
    """
    Write a list of samples to a TAR shard.
    
    Each sample in the TAR has:
    - {key}.{ext} - Image file
    - {key}.txt - Caption
    - {key}.json - Metadata
    """
    written = 0
    with tarfile.open(shard_path, 'w') as tar:
        for i, sample in enumerate(samples):
            if sample is None:
                continue
                
            image_bytes, ext, caption, metadata = sample
            key = create_sample_key(start_idx + i)
            
            # Add image
            img_info = tarfile.TarInfo(name=f"{key}.{ext}")
            img_info.size = len(image_bytes)
            tar.addfile(img_info, io.BytesIO(image_bytes))
            
            # Add caption
            caption_bytes = caption.encode('utf-8')
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(caption_bytes)
            tar.addfile(txt_info, io.BytesIO(caption_bytes))
            
            # Add metadata
            meta_bytes = json.dumps(metadata).encode('utf-8')
            meta_info = tarfile.TarInfo(name=f"{key}.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, io.BytesIO(meta_bytes))
            
            written += 1
    
    return written

def convert_to_webdataset(arrow_dir, images_dir, output_dir, images_per_shard=1000, workers=8):
    """Convert Arrow dataset to WebDataset shards."""
    
    # Create output directories
    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    
    # Load Arrow dataset
    print(f"Loading Arrow dataset from: {arrow_dir}")
    arrow_files = glob.glob(os.path.join(arrow_dir, "*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files in {arrow_dir}")
    
    ds = load_dataset("arrow", data_files=arrow_files, split="train")
    total_samples = len(ds)
    print(f"Dataset size: {total_samples:,} samples")
    
    # Calculate shard count
    num_shards = (total_samples + images_per_shard - 1) // images_per_shard
    print(f"Creating {num_shards:,} shards ({images_per_shard} images each)")
    
    # Process in batches
    manifest = {
        'dataset': 'pixmo-cap',
        'total_samples': total_samples,
        'images_per_shard': images_per_shard,
        'num_shards': num_shards,
        'shards': []
    }
    
    total_written = 0
    pbar = tqdm(total=total_samples, desc="Converting")
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * images_per_shard
        end_idx = min(start_idx + images_per_shard, total_samples)
        batch = ds.select(range(start_idx, end_idx))
        
        # Process samples in parallel
        samples = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_sample, item, images_dir) for item in batch]
            for future in as_completed(futures):
                samples.append(future.result())
        
        # Reorder to match original order (as_completed scrambles order)
        # For simplicity, just process sequentially since I/O is the bottleneck
        samples = [process_sample(item, images_dir) for item in batch]
        
        # Write shard
        shard_name = f"pixmo-{shard_idx:06d}.tar"
        shard_path = os.path.join(shards_dir, shard_name)
        written = write_shard(samples, shard_path, start_idx)
        
        shard_size = os.path.getsize(shard_path)
        manifest['shards'].append({
            'name': shard_name,
            'samples': written,
            'size_bytes': shard_size,
            'start_idx': start_idx,
        })
        
        total_written += written
        pbar.update(end_idx - start_idx)
    
    pbar.close()
    
    # Update manifest
    manifest['total_written'] = total_written
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n=== Conversion Complete ===")
    print(f"Total samples: {total_samples:,}")
    print(f"Written to shards: {total_written:,}")
    print(f"Skipped (corrupt/missing): {total_samples - total_written:,}")
    print(f"Output: {shards_dir}")
    print(f"Manifest: {manifest_path}")
    
    # Print shard distribution info for multi-node planning
    print("\n=== Multi-Node Distribution ===")
    print(f"Shards: {num_shards}")
    for n_nodes in [1, 2, 4, 8, 12]:
        shards_per_node = num_shards // n_nodes
        remainder = num_shards % n_nodes
        print(f"  {n_nodes:2d} nodes: {shards_per_node} shards/node" + 
              (f" (+{remainder} remainder)" if remainder else ""))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-dir", required=True, help="Directory with Arrow files")
    parser.add_argument("--images-dir", required=True, help="Directory with images (for validation)")
    parser.add_argument("--output-dir", required=True, help="Output directory for WebDataset")
    parser.add_argument("--images-per-shard", type=int, default=1000, help="Images per TAR shard")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for image processing")
    
    args = parser.parse_args()
    
    convert_to_webdataset(
        args.arrow_dir,
        args.images_dir,
        args.output_dir,
        images_per_shard=args.images_per_shard,
        workers=args.workers
    )

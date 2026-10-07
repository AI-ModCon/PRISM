#!/usr/bin/env python3
"""
stage_shards.py - Stage WebDataset shards to compute node local storage

Copies assigned shards from shared filesystem to local /tmp for fast I/O.

Usage (typically called from launch script):
    python scripts/stage_shards.py \
        --manifest /path/to/manifest.json \
        --shards-dir /path/to/shards \
        --local-dir /tmp/webdataset \
        --node-rank 0 \
        --num-nodes 4

This will copy shards 0, 4, 8, 12, ... to node 0
            shards 1, 5, 9, 13, ... to node 1, etc.
"""

import argparse
import json
import os
import shutil

from tqdm import tqdm


def get_shards_for_node(manifest, node_rank, num_nodes):
    """
    Assign shards to nodes using round-robin.
    Returns list of shard names assigned to this node.
    """
    all_shards = manifest['shards']
    assigned = []
    
    for i, shard in enumerate(all_shards):
        if i % num_nodes == node_rank:
            assigned.append(shard['name'])
    
    return assigned

def stage_shards(manifest_path, shards_dir, local_dir, node_rank, num_nodes):
    """Stage assigned shards to local storage."""
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    total_shards = manifest['num_shards']
    print(f"Total shards: {total_shards}")
    print(f"Node {node_rank + 1}/{num_nodes}")
    
    # Get assigned shards
    assigned = get_shards_for_node(manifest, node_rank, num_nodes)
    print(f"Assigned shards: {len(assigned)}")
    
    # Create local directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Copy shards
    # Copy shards in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 4-8 workers usually optimal for file I/O
    num_workers = min(8, len(assigned))
    print(f"Starting parallel copy with {num_workers} workers...")
    
    staged = []
    
    def copy_shard(shard_name):
        src = os.path.join(shards_dir, shard_name)
        dst = os.path.join(local_dir, shard_name)
        
        if os.path.exists(dst):
            return dst, False # Already exists
            
        if not os.path.exists(src):
            print(f"Warning: Shard not found: {src}")
            return None, False
            
        shutil.copy2(src, dst)
        return dst, True # Copied
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(copy_shard, name): name for name in assigned}
        
        for future in tqdm(as_completed(futures), total=len(assigned), desc="Staging"):
            result, copied = future.result()
            if result:
                staged.append(result)
    
    # Write local manifest for this node
    local_manifest = {
        'node_rank': node_rank,
        'num_nodes': num_nodes,
        'local_dir': local_dir,
        'shards': [os.path.basename(s) for s in staged],
        'total_samples_estimate': sum(
            s['samples'] for s in manifest['shards'] 
            if s['name'] in [os.path.basename(x) for x in staged]
        )
    }
    
    local_manifest_path = os.path.join(local_dir, "local_manifest.json")
    with open(local_manifest_path, 'w') as f:
        json.dump(local_manifest, f, indent=2)
    
    print("\n=== Staging Complete ===")
    print(f"Staged {len(staged)} shards to {local_dir}")
    print(f"Local manifest: {local_manifest_path}")
    
    # Return shard pattern for WebDataset
    shard_pattern = os.path.join(local_dir, f"pixmo-{{000000..{total_shards - 1:06d}}}.tar")
    print(f"\nWebDataset pattern: {shard_pattern}")
    
    return staged, local_manifest_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--shards-dir", required=True, help="Directory containing shards")
    parser.add_argument("--local-dir", required=True, help="Local destination directory")
    parser.add_argument("--node-rank", type=int, required=True, help="This node's rank (0-indexed)")
    parser.add_argument("--num-nodes", type=int, required=True, help="Total number of nodes")
    
    args = parser.parse_args()
    
    stage_shards(
        args.manifest,
        args.shards_dir,
        args.local_dir,
        args.node_rank,
        args.num_nodes
    )

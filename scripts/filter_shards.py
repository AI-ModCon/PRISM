#!/usr/bin/env python3
"""
filter_shards.py - Filter staged shards for round-robin distribution

Usage:
    python filter_shards.py --shards-dir /tmp/webdataset --node-rank 0 --num-nodes 4

Each node keeps only shards where: shard_index % num_nodes == node_rank
"""

import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser(description="Filter shards for round-robin distribution")
    parser.add_argument("--shards-dir", required=True, help="Directory containing staged shards")
    parser.add_argument("--node-rank", type=int, required=True, help="This node's rank (0-indexed)")
    parser.add_argument("--num-nodes", type=int, required=True, help="Total number of nodes")
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shards_dir, "*.tar")))
    kept, deleted = 0, 0
    
    for i, shard in enumerate(shards):
        if i % args.num_nodes != args.node_rank:
            os.remove(shard)
            deleted += 1
        else:
            kept += 1
    
    print(f"Node {args.node_rank}: Kept {kept}, Deleted {deleted} shards")

if __name__ == "__main__":
    main()

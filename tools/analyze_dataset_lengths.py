#!/usr/bin/env python3
"""
analyze_dataset_lengths.py - Analyze text sequence lengths across DAOS datasets

This script samples from different dataset groups and computes token length statistics
to understand why certain datasets result in slower training throughput.

Usage:
    # Must run on a compute node with DAOS mounted
    python tools/analyze_dataset_lengths.py --dataset-groups all --samples-per-dataset 100
"""

import argparse
import json
import logging
import os
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning(
        "transformers not available - will report character counts instead of tokens"
    )


def load_yaml_config(yaml_path):
    """Load YAML config without PyYAML (simple parser)."""
    # For simplicity, we'll hardcode the dataset paths
    # In production, use OmegaConf or PyYAML
    return None


DATASET_PATHS = {
    # pixmo group
    "pixmo_cap": "pixmo/pixmo_cap_webdataset",
    "pixmo_points": "pixmo/pixmo_points_webdataset",
    "pixmo_count": "pixmo/pixmo_count_webdataset",
    # s1mmalign group
    "arxiv": "s1mmalign/arxiv_webdataset",
    "biorxiv": "s1mmalign/biorxiv_webdataset",
    "nature_comunication": "s1mmalign/nature_comunication_webdataset",
    # nemotron group
    "wiki_en": "nemotron/wiki_en_webdataset",
    "wiki_de": "nemotron/wiki_de_webdataset",
    "sparsetables": "nemotron/sparsetables_webdataset",
    "plotqa_cot": "nemotron/plotqa_cot_webdataset",
    # cosyn group
    "cosyn_point": "cosyn/cosyn_point_webdataset",
}

DATASET_GROUPS = {
    "pixmo": ["pixmo_cap", "pixmo_points", "pixmo_count"],
    "s1mmalign": ["arxiv", "biorxiv", "nature_comunication"],
    "nemotron": ["wiki_en", "wiki_de", "sparsetables", "plotqa_cot"],
    "cosyn": ["cosyn_point"],
}


def process_sample_text(text, metadata):
    """
    Replicate the _process_sample logic from multi_webdataset.py to get actual training text.

    This is critical for accurate token counting - pointing datasets store only the label
    in the .txt file, but the training pipeline constructs full conversations from metadata.
    """
    caption = None

    # Format 1: Conversation format (cosyn_point style)
    # metadata has 'conversations': [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': '<points>...'}]
    if "conversations" in metadata and isinstance(metadata["conversations"], list):
        conversations = metadata["conversations"]
        if len(conversations) >= 1:
            # Format as "user: ... assistant: ..." for training
            parts = []
            for turn in conversations:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                parts.append(f"{role}: {content}")
            caption = "\n".join(parts)

    # Format 2: Points + Label format (pixmo-points style)
    # metadata has 'points': [{'x': ..., 'y': ...}, ...] and 'label': '...'
    # OR 'points': {'x': [...], 'y': [...]} (pixmo-count parallel array format)
    elif "points" in metadata and "label" in metadata:
        points = metadata["points"]
        label = metadata["label"]

        # Convert parallel array format to list of dicts
        # pixmo_count uses: {"x": [x1, x2, ...], "y": [y1, y2, ...]}
        if isinstance(points, dict) and "x" in points and "y" in points:
            x_coords = points["x"]
            y_coords = points["y"]
            if isinstance(x_coords, list) and isinstance(y_coords, list):
                points = [{"x": x, "y": y} for x, y in zip(x_coords, y_coords, strict=False)]

        if isinstance(points, list) and len(points) > 0:
            # Construct Molmo2-style conversation
            question = f"Point to {label}"

            # Format points as Molmo2 coords
            coords_parts = []
            for i, pt in enumerate(points):
                if isinstance(pt, dict):
                    x = pt.get("x", 0)
                    y = pt.get("y", 0)
                elif isinstance(pt, list | tuple) and len(pt) >= 2:
                    x, y = pt[0], pt[1]
                else:
                    continue

                # Normalize to 0-1000 range if not already
                if x <= 100 and y <= 100:
                    x = int(x * 10)
                    y = int(y * 10)
                else:
                    x = int(x)
                    y = int(y)

                coords_parts.append(f"1 {i + 1} {x} {y}")

            coords_str = ";".join(coords_parts)
            answer = f'<points coords="{coords_str}">{label}</points>'
            caption = f"user: {question}\nassistant: {answer}"

    # Format 3: Default caption format
    if caption is None:
        caption = text

    return caption if caption and caption.strip() else "An image."


def sample_from_shard(shard_path, max_samples=10):
    """Extract text samples from a WebDataset shard."""
    samples = []
    try:
        with tarfile.open(shard_path, "r") as tar:
            # Group files by key
            files_by_key = defaultdict(dict)
            for member in tar.getmembers():
                if member.isfile():
                    key = member.name.rsplit(".", 1)[0]
                    ext = member.name.rsplit(".", 1)[1] if "." in member.name else ""
                    files_by_key[key][ext] = member

            # Extract text and metadata for each sample
            for key, files in list(files_by_key.items())[:max_samples]:
                sample = {"key": key}
                raw_text = ""
                metadata = {}

                # Get raw text from .txt file
                if "txt" in files:
                    f = tar.extractfile(files["txt"])
                    if f:
                        raw_text = f.read().decode("utf-8", errors="replace")

                # Get metadata from .json file
                if "json" in files:
                    f = tar.extractfile(files["json"])
                    if f:
                        try:
                            metadata = json.loads(f.read().decode("utf-8"))
                        except Exception:
                            metadata = {}

                # CRITICAL: Process the sample the same way training does
                # This constructs full conversations for pointing datasets
                sample["text"] = process_sample_text(raw_text, metadata)
                sample["raw_text"] = raw_text  # Keep raw for comparison
                sample["metadata"] = metadata

                if sample["text"]:
                    samples.append(sample)

    except Exception as e:
        logger.warning(f"Error reading {shard_path}: {e}")

    return samples


def analyze_dataset(dataset_name, base_path, tokenizer=None, samples_per_dataset=100):
    """Analyze text lengths for a single dataset."""
    dataset_path = os.path.join(base_path, DATASET_PATHS.get(dataset_name, ""))
    shards_dir = os.path.join(dataset_path, "shards")

    if not os.path.exists(shards_dir):
        logger.warning(f"  {dataset_name}: shards dir not found at {shards_dir}")
        return None

    # Get shard files
    shard_files = sorted([f for f in os.listdir(shards_dir) if f.endswith(".tar")])
    if not shard_files:
        logger.warning(f"  {dataset_name}: no shard files found")
        return None

    # Sample from random shards
    random.seed(42)
    sample_shards = random.sample(shard_files, min(10, len(shard_files)))

    all_samples = []
    for shard_file in sample_shards:
        shard_path = os.path.join(shards_dir, shard_file)
        samples = sample_from_shard(
            shard_path, max_samples=samples_per_dataset // len(sample_shards) + 1
        )
        all_samples.extend(samples)
        if len(all_samples) >= samples_per_dataset:
            break

    all_samples = all_samples[:samples_per_dataset]

    if not all_samples:
        return None

    # Compute statistics
    char_lengths = [len(s["text"]) for s in all_samples]

    if tokenizer:
        token_lengths = []
        for s in all_samples:
            tokens = tokenizer(s["text"], return_tensors=None, truncation=False)
            token_lengths.append(len(tokens["input_ids"]))
    else:
        # Estimate tokens as chars / 4
        token_lengths = [c // 4 for c in char_lengths]

    # Get sample data types
    data_types = defaultdict(int)
    for s in all_samples:
        meta = s.get("metadata", {})
        if "conversations" in meta:
            data_types["conversation"] += 1
        elif "points" in meta and "label" in meta:
            data_types["pointing"] += 1
        else:
            data_types["caption"] += 1

    return {
        "name": dataset_name,
        "num_samples": len(all_samples),
        "char_min": min(char_lengths),
        "char_max": max(char_lengths),
        "char_mean": sum(char_lengths) / len(char_lengths),
        "char_median": sorted(char_lengths)[len(char_lengths) // 2],
        "token_min": min(token_lengths),
        "token_max": max(token_lengths),
        "token_mean": sum(token_lengths) / len(token_lengths),
        "token_median": sorted(token_lengths)[len(token_lengths) // 2],
        "token_p90": sorted(token_lengths)[int(len(token_lengths) * 0.9)],
        "token_p99": sorted(token_lengths)[int(len(token_lengths) * 0.99)],
        "data_types": dict(data_types),
        # Sample texts for inspection (show both raw and processed)
        "sample_texts": [
            {
                "raw": s.get("raw_text", "")[:100]
                + ("..." if len(s.get("raw_text", "")) > 100 else ""),
                "processed": s["text"][:200] + ("..." if len(s["text"]) > 200 else ""),
            }
            for s in all_samples[:3]
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze text sequence lengths across datasets"
    )
    parser.add_argument(
        "--daos-mount",
        default="/tmp/${USER}/AuroraGPT/prism_training_data",
        help="DAOS mount path",
    )
    parser.add_argument(
        "--dataset-groups",
        default="all",
        help="Comma-separated list of groups or 'all'",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=100,
        help="Number of samples to analyze per dataset",
    )
    parser.add_argument(
        "--tokenizer",
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Tokenizer to use for token counting",
    )
    args = parser.parse_args()

    # Expand environment variables
    base_path = os.path.expandvars(args.daos_mount)

    # Check if DAOS is mounted
    if not os.path.exists(base_path):
        logger.error(f"DAOS not mounted at {base_path}")
        logger.error("Run this script on a compute node with DAOS mounted")
        sys.exit(1)

    # Load tokenizer
    tokenizer = None
    if HAS_TRANSFORMERS:
        try:
            logger.info(f"Loading tokenizer: {args.tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer, trust_remote_code=True
            )
            logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")

    # Determine which datasets to analyze
    if args.dataset_groups == "all":
        groups = list(DATASET_GROUPS.keys())
    else:
        groups = [g.strip() for g in args.dataset_groups.split(",")]

    logger.info(f"\n{'=' * 80}")
    logger.info("SEQUENCE LENGTH ANALYSIS")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Groups: {groups}")
    logger.info(f"Samples per dataset: {args.samples_per_dataset}")
    logger.info(f"{'=' * 80}\n")

    results = []

    for group in groups:
        datasets = DATASET_GROUPS.get(group, [])
        logger.info(f"\n--- Group: {group} ({len(datasets)} datasets) ---")

        for dataset_name in datasets:
            logger.info(f"\nAnalyzing: {dataset_name}")
            result = analyze_dataset(
                dataset_name,
                base_path,
                tokenizer=tokenizer,
                samples_per_dataset=args.samples_per_dataset,
            )

            if result:
                results.append(result)
                logger.info(f"  Samples: {result['num_samples']}")
                logger.info(
                    f"  Tokens: min={result['token_min']}, max={result['token_max']}, "
                    f"mean={result['token_mean']:.1f}, median={result['token_median']}"
                )
                logger.info(f"  P90={result['token_p90']}, P99={result['token_p99']}")
                logger.info(f"  Data types: {result['data_types']}")
                sample = result["sample_texts"][0]
                logger.info(f"  Raw text: {sample['raw']}")
                logger.info(f"  Processed text: {sample['processed'][:100]}...")

    # Summary table
    logger.info(f"\n\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"\n{'Dataset':<25} {'Mean Tok':<10} {'Median':<10} {'P90':<10} {'P99':<10} {'Max':<10}"
    )
    logger.info("-" * 80)

    # Sort by mean token length
    results.sort(key=lambda x: x["token_mean"], reverse=True)

    for r in results:
        logger.info(
            f"{r['name']:<25} {r['token_mean']:<10.1f} {r['token_median']:<10} "
            f"{r['token_p90']:<10} {r['token_p99']:<10} {r['token_max']:<10}"
        )

    # Analysis
    logger.info(f"\n\n{'=' * 80}")
    logger.info("ANALYSIS")
    logger.info(f"{'=' * 80}")

    if results:
        pixmo_results = [r for r in results if r["name"].startswith("pixmo")]
        other_results = [r for r in results if not r["name"].startswith("pixmo")]

        if pixmo_results:
            pixmo_mean = sum(r["token_mean"] for r in pixmo_results) / len(
                pixmo_results
            )
            logger.info(f"\nPixmo average token length: {pixmo_mean:.1f}")

        if other_results:
            other_mean = sum(r["token_mean"] for r in other_results) / len(
                other_results
            )
            logger.info(f"Other datasets average token length: {other_mean:.1f}")

        if pixmo_results and other_results:
            ratio = other_mean / pixmo_mean
            logger.info(f"\nRatio (other/pixmo): {ratio:.2f}x")
            logger.info("\nThis explains the throughput difference!")
            logger.info(
                "Longer sequences = more compute in attention (O(n²)) and more gradient computation"
            )
            logger.info(
                f"If other datasets have {ratio:.1f}x longer sequences, expect ~{ratio:.1f}x slower backward pass"
            )


if __name__ == "__main__":
    main()

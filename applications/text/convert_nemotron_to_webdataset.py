#!/usr/bin/env python3
"""
convert_nemotron_to_webdataset.py - Convert Nemotron-VLM datasets to WebDataset format

Nemotron datasets use a JSONL + TAR media structure. This script:
1. Reads the JSONL conversation data
2. Extracts images from the media/*.tar shards
3. Creates unified WebDataset TAR shards with image + conversation text

Usage:
    # Convert a single dataset
    python applications/text/convert_nemotron_to_webdataset.py \
        --input-dir /flare/ModCon/ngetty/data/zone_a/Nemotron-VLM-Dataset-v2/wiki_en \
        --output-dir /flare/ModCon/ngetty/data/zone_a/nemotron_wiki_en_webdataset \
        --dataset-name nemotron-wiki-en

    # Convert all hydrated Nemotron datasets
    python applications/text/convert_nemotron_to_webdataset.py --convert-all
"""

import argparse
import glob
import io
import json
import logging
import os
import random
import tarfile

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_conversation_text(messages):
    """
    Extract text from Nemotron conversation format.
    Returns: (user_text, assistant_text, image_path)
    """
    user_text = ""
    assistant_text = ""
    image_path = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", [])

        for item in content:
            item_type = item.get("type", "")

            if item_type == "text":
                text = item.get("text", "")
                if role == "user":
                    user_text += text + " "
                elif role == "assistant":
                    assistant_text += text + " "

            elif item_type == "image":
                if image_path is None:
                    image_path = item.get("image", "")

    return user_text.strip(), assistant_text.strip(), image_path


def format_training_text(user_text, assistant_text, format_style="conversation"):
    """
    Format user/assistant text for training.
    """
    if format_style == "caption":
        return assistant_text
    elif format_style == "qa":
        return f"Question: {user_text}\nAnswer: {assistant_text}"
    else:
        return f"User: {user_text}\nAssistant: {assistant_text}"


def load_tar_index(tar_path):
    """
    Load image data from a TAR shard into memory.
    Returns dict: {filename: image_bytes}
    """
    images = {}
    try:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        name = os.path.basename(member.name)
                        images[name] = f.read()
    except Exception as e:
        logger.warning(f"Error reading {tar_path}: {e}")
    return images


def create_sample_key(idx):
    """Create a unique key for each sample."""
    return f"{idx:08d}"


def write_shard(samples, shard_path, dataset_name):
    """
    Write samples to a TAR shard.
    Each sample: (image_bytes, ext, text, metadata)
    """
    written = 0
    with tarfile.open(shard_path, "w") as tar:
        for i, sample in enumerate(samples):
            if sample is None:
                continue

            image_bytes, ext, text, metadata = sample
            key = create_sample_key(metadata.get("global_idx", i))

            if ext.lower() in ["jpeg"]:
                ext = "jpg"
            elif ext.lower() not in ["jpg", "png", "webp", "gif"]:
                ext = "jpg"

            # Add image
            img_info = tarfile.TarInfo(name=f"{key}.{ext}")
            img_info.size = len(image_bytes)
            tar.addfile(img_info, io.BytesIO(image_bytes))

            # Add text
            text_bytes = text.encode("utf-8")
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(text_bytes)
            tar.addfile(txt_info, io.BytesIO(text_bytes))

            # Add metadata
            meta_bytes = json.dumps(metadata).encode("utf-8")
            meta_info = tarfile.TarInfo(name=f"{key}.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, io.BytesIO(meta_bytes))

            written += 1

    return written


def convert_nemotron_dataset(
    input_dir,
    output_dir,
    dataset_name,
    samples_per_shard=1000,
    text_format="conversation",
    val_split=0.01,
):
    """
    Convert a single Nemotron dataset to WebDataset format.
    """
    os.makedirs(output_dir, exist_ok=True)
    shards_dir = os.path.join(output_dir, "shards")
    val_shards_dir = os.path.join(output_dir, "val_shards")
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(val_shards_dir, exist_ok=True)

    # Find JSONL file
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")
    jsonl_path = jsonl_files[0]
    logger.info(f"JSONL: {jsonl_path}")

    # Find media TAR shards
    media_dir = os.path.join(input_dir, "media")
    if not os.path.isdir(media_dir):
        raise FileNotFoundError(f"No media/ directory found in {input_dir}")

    tar_files = sorted(glob.glob(os.path.join(media_dir, "*.tar")))
    logger.info(f"Found {len(tar_files)} media TAR shards")

    # Load all images from TARs
    logger.info("Loading images from TAR shards...")
    all_images = {}
    for tar_path in tqdm(tar_files, desc="Loading TARs"):
        images = load_tar_index(tar_path)
        all_images.update(images)
    logger.info(f"Loaded {len(all_images)} images")

    # Load JSONL data
    logger.info("Loading JSONL data...")
    samples_data = []
    with open(jsonl_path) as f:
        for line in tqdm(f, desc="Reading JSONL"):
            try:
                data = json.loads(line.strip())
                samples_data.append(data)
            except json.JSONDecodeError:
                continue

    total_samples = len(samples_data)
    logger.info(f"Loaded {total_samples:,} conversation samples")

    # Process samples
    logger.info("Processing samples...")
    processed_samples = []
    stats = {"matched": 0, "no_image": 0, "missing_image": 0}

    for idx, data in enumerate(tqdm(samples_data, desc="Processing")):
        messages = data.get("messages", [])
        sample_id = data.get("id", str(idx))

        user_text, assistant_text, image_path = extract_conversation_text(messages)

        if not image_path:
            stats["no_image"] += 1
            continue

        image_filename = os.path.basename(image_path)
        image_bytes = all_images.get(image_filename)

        if image_bytes is None:
            stats["missing_image"] += 1
            continue

        try:
            img = Image.open(io.BytesIO(image_bytes))
            ext = img.format.lower() if img.format else "jpg"
            img.verify()
        except Exception:
            stats["missing_image"] += 1
            continue

        text = format_training_text(user_text, assistant_text, text_format)

        metadata = {
            "id": sample_id,
            "original_image": image_path,
            "global_idx": idx,
            "dataset": dataset_name,
        }

        processed_samples.append((image_bytes, ext, text, metadata))
        stats["matched"] += 1

    logger.info("Processing stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v:,}")

    if not processed_samples:
        logger.error("No samples processed!")
        return None

    # Shuffle and split
    random.seed(42)
    random.shuffle(processed_samples)

    val_size = int(len(processed_samples) * val_split)
    train_samples = processed_samples[val_size:]
    val_samples = processed_samples[:val_size]

    logger.info(f"Train: {len(train_samples):,}, Val: {len(val_samples):,}")

    # Write training shards
    num_train_shards = (len(train_samples) + samples_per_shard - 1) // samples_per_shard
    manifest = {
        "dataset": dataset_name,
        "source": "Nemotron-VLM-Dataset-v2",
        "total_samples": len(processed_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "samples_per_shard": samples_per_shard,
        "num_shards": num_train_shards,
        "text_format": text_format,
        "shards": [],
        "val_shards": [],
    }

    total_written = 0
    logger.info(f"Writing {num_train_shards} training shards...")

    for shard_idx in tqdm(range(num_train_shards), desc="Training shards"):
        start = shard_idx * samples_per_shard
        end = min(start + samples_per_shard, len(train_samples))
        batch = train_samples[start:end]

        shard_name = f"{dataset_name}-{shard_idx:06d}.tar"
        shard_path = os.path.join(shards_dir, shard_name)
        written = write_shard(batch, shard_path, dataset_name)

        manifest["shards"].append(
            {
                "name": shard_name,
                "samples": written,
                "size_bytes": os.path.getsize(shard_path),
            }
        )
        total_written += written

    # Write validation shards
    if val_samples:
        num_val_shards = max(
            1, (len(val_samples) + samples_per_shard - 1) // samples_per_shard
        )
        logger.info(f"Writing {num_val_shards} validation shards...")

        for shard_idx in range(num_val_shards):
            start = shard_idx * samples_per_shard
            end = min(start + samples_per_shard, len(val_samples))
            batch = val_samples[start:end]

            val_shard_idx = num_train_shards + shard_idx
            shard_name = f"{dataset_name}-{val_shard_idx:06d}.tar"
            shard_path = os.path.join(val_shards_dir, shard_name)
            written = write_shard(batch, shard_path, dataset_name)

            manifest["val_shards"].append(
                {
                    "name": shard_name,
                    "samples": written,
                    "size_bytes": os.path.getsize(shard_path),
                }
            )

    manifest["total_written"] = total_written
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Conversion Complete: {dataset_name}")
    logger.info(f"Total written: {total_written:,}")
    logger.info(f"Output: {output_dir}")

    return manifest


def convert_all_nemotron(base_dir, output_base_dir):
    """
    Convert all Nemotron datasets that have media/ folders.
    """
    hydrated_datasets = [
        "wiki_en",
        "wiki_de",
        "wiki_es",
        "wiki_fr",
        "wiki_it",
        "wiki_ja",
        "wiki_ko",
        "wiki_nl",
        "wiki_pt",
        "wiki_zh",
        "sparsetables",
        "plotqa_cot",
        "nights_cot",
        "ego_exo_learn",
        "perception_test_1",
        "perception_test_2",
        "perception_test_cot",
        "breakfast_actions",
    ]

    os.makedirs(output_base_dir, exist_ok=True)
    results = []

    for ds_name in hydrated_datasets:
        input_dir = os.path.join(base_dir, ds_name)
        if not os.path.isdir(input_dir):
            logger.warning(f"Skipping {ds_name}: directory not found")
            continue

        media_dir = os.path.join(input_dir, "media")
        if not os.path.isdir(media_dir):
            logger.warning(f"Skipping {ds_name}: no media/ folder")
            continue

        output_dir = os.path.join(output_base_dir, f"{ds_name}_webdataset")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Converting: {ds_name}")
        logger.info(f"{'=' * 60}")

        try:
            manifest = convert_nemotron_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                dataset_name=f"nemotron-{ds_name}",
                text_format="conversation",
            )
            if manifest:
                results.append((ds_name, manifest["total_written"], "success"))
            else:
                results.append((ds_name, 0, "no samples"))
        except Exception as e:
            logger.error(f"Error converting {ds_name}: {e}")
            results.append((ds_name, 0, str(e)[:50]))

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    total = 0
    for name, count, status in results:
        print(f"  {name:30s} {count:>8,} samples  [{status}]")
        total += count
    print(f"{'TOTAL':>40s} {total:>8,}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nemotron-VLM datasets to WebDataset format"
    )

    parser.add_argument("--input-dir", help="Input Nemotron dataset directory")
    parser.add_argument("--output-dir", help="Output WebDataset directory")
    parser.add_argument("--dataset-name", help="Name for output shards")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument(
        "--text-format",
        choices=["conversation", "qa", "caption"],
        default="conversation",
    )
    parser.add_argument("--val-split", type=float, default=0.01)

    parser.add_argument(
        "--convert-all",
        action="store_true",
        help="Convert all hydrated Nemotron datasets",
    )
    parser.add_argument(
        "--base-dir", default="/flare/ModCon/ngetty/data/zone_a/Nemotron-VLM-Dataset-v2"
    )
    parser.add_argument(
        "--output-base-dir",
        default="/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets",
    )

    args = parser.parse_args()

    if args.convert_all:
        convert_all_nemotron(args.base_dir, args.output_base_dir)
    elif args.input_dir and args.output_dir and args.dataset_name:
        convert_nemotron_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            samples_per_shard=args.samples_per_shard,
            text_format=args.text_format,
            val_split=args.val_split,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

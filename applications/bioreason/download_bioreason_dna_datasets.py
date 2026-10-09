"""Download BioReason DNA datasets (KEGG, variant effect) and save them to disk
as Arrow shards, so the StreamingMultimodalDataset local-path loader
(src/data/multimodal.py, the `*.arrow` fallback under the generic local-loading
branch) can pick them up without hitting the HF Hub at train time.

Usage:
    python applications/bioreason/download_bioreason_dna_datasets.py --dataset kegg --output-dir /path/to/output
    python applications/bioreason/download_bioreason_dna_datasets.py --dataset variant_effect_coding --output-dir /path/to/output
    python applications/bioreason/download_bioreason_dna_datasets.py --dataset variant_effect_non_snv --output-dir /path/to/output
    python applications/bioreason/download_bioreason_dna_datasets.py --dataset all --output-dir /path/to/output_root
"""

import argparse
import os

from datasets import load_dataset

DATASETS = {
    "kegg": {"hf_id": "wanglab/kegg", "trust_remote_code": True},
    "variant_effect_coding": {"hf_id": "wanglab/variant_effect_coding", "trust_remote_code": False},
    "variant_effect_non_snv": {"hf_id": "wanglab/variant_effect_non_snv", "trust_remote_code": False},
}


def download_one(key: str, output_dir: str, split: str = "train") -> None:
    info = DATASETS[key]
    hf_id = info["hf_id"]
    dest = os.path.join(output_dir, key)
    os.makedirs(dest, exist_ok=True)

    print(f"[{key}] Loading {hf_id} (split={split}) ...")
    kwargs = {"split": split}
    if info["trust_remote_code"]:
        kwargs["trust_remote_code"] = True
    ds = load_dataset(hf_id, **kwargs)

    print(f"[{key}] Saving {len(ds)} rows to {dest} ...")
    ds.save_to_disk(dest)
    print(f"[{key}] Done -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory. Each dataset is saved to <output-dir>/<dataset_key>/",
    )
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for key in keys:
        download_one(key, args.output_dir, split=args.split)


if __name__ == "__main__":
    main()

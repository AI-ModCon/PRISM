#!/usr/bin/env python3
"""Validate and extract the pinned AstroPRISM AION feature parity reference.

The released image embeddings are a parity/regression oracle only, not Stage 1A
raw-image model inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.astroprism_stage1a import Stage1AConfig, prepare_stage1a  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=REPO_ROOT / "docs/astroprism/data/overlap_candidates_646.csv.gz",
    )
    parser.add_argument(
        "--paired-root",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/hf/datasets/desi_x_ls_aion",
    )
    parser.add_argument(
        "--caption-root",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/hf/datasets/galaxy-descriptions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1a_candidates",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all checksum/schema/content checks without writing output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output after validation succeeds.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_stage1a(
        Stage1AConfig(
            roster_path=args.roster,
            paired_root=args.paired_root,
            caption_root=args.caption_root,
            output_dir=args.output,
            validate_only=args.validate_only,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

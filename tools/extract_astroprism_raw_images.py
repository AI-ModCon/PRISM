#!/usr/bin/env python3
"""Select the 646 AstroPRISM raw Legacy Survey image rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.astroprism_raw_images import (  # noqa: E402
    RAW_SOURCE_ETAG,
    RAW_SOURCE_LENGTH,
    RAW_SOURCE_URL,
    RawImageConfig,
    extract_raw_images,
)


def _is_http(source: str) -> bool:
    return urlsplit(source).scheme.lower() in {"http", "https"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=REPO_ROOT / "docs/astroprism/data/overlap_candidates_646.csv.gz",
    )
    parser.add_argument(
        "--source",
        default=RAW_SOURCE_URL,
        help="Official HTTP HDF5 URL or a local HDF5 path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/raw_images",
    )
    parser.add_argument(
        "--expected-source-length",
        type=int,
        default=None,
        help="Required for custom HTTP sources; the official URL uses its pinned length.",
    )
    parser.add_argument(
        "--expected-source-etag",
        default=None,
        help="Required for custom HTTP sources; include quotes if present in the ETag.",
    )
    parser.add_argument("--http-block-size", type=int, default=256 * 1024)
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--http-probe-attempts", type=int, default=3)
    parser.add_argument("--http-probe-backoff-seconds", type=float, default=1.0)
    parser.add_argument(
        "--caption-legacy-coordinate-tolerance-arcsec",
        type=float,
        default=1e-4,
        help="Maximum raw Legacy-to-caption roster coordinate separation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output after extraction succeeds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_is_official = args.source == RAW_SOURCE_URL
    if _is_http(args.source):
        expected_length = (
            RAW_SOURCE_LENGTH
            if source_is_official and args.expected_source_length is None
            else args.expected_source_length
        )
        expected_etag = (
            RAW_SOURCE_ETAG
            if source_is_official and args.expected_source_etag is None
            else args.expected_source_etag
        )
    else:
        expected_length = args.expected_source_length
        expected_etag = args.expected_source_etag

    summary = extract_raw_images(
        RawImageConfig(
            roster_path=args.roster,
            output_dir=args.output,
            source=args.source,
            overwrite=args.overwrite,
            expected_source_length=expected_length,
            expected_source_etag=expected_etag,
            http_block_size=args.http_block_size,
            http_timeout_seconds=args.http_timeout_seconds,
            http_probe_attempts=args.http_probe_attempts,
            http_probe_backoff_seconds=args.http_probe_backoff_seconds,
            caption_legacy_coordinate_tolerance_arcsec=(
                args.caption_legacy_coordinate_tolerance_arcsec
            ),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download immutable official DOCCI source objects, with resumable transfers.

The HF google/docci loader points to these publisher-hosted files. This script
downloads data only; it does not execute the dataset's remote loading script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

OBJECTS = {
    "docci_images.tar.gz": ("1714249456173037", 7_592_938_768),
    "docci_descriptions.jsonlines": ("1714384012999810", 11_000_214),
    "docci_metadata.jsonlines": ("1714384023714932", 162_767_377),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(directory, name, generation, expected_bytes):
    url = f"https://storage.googleapis.com/docci/data/{name}?generation={generation}"
    target = directory / name
    partial = directory / (name + ".partial")
    if not target.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_bytes:
            raise ValueError(f"Partial file exceeds source size: {partial}")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        if offset < expected_bytes:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as response:
                if offset and (
                    response.status != 206
                    or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                ):
                    raise ValueError("Server did not honor the resume range")
                if response.headers.get("x-goog-generation") != generation:
                    raise ValueError("Unexpected GCS object generation")
                (directory / (name + ".headers.json")).write_text(
                    json.dumps(dict(response.headers), indent=2) + "\n"
                )
                last_report = time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_report >= 20:
                            print(
                                json.dumps(
                                    {
                                        "file": name,
                                        "bytes": offset,
                                        "expected_bytes": expected_bytes,
                                    }
                                ),
                                flush=True,
                            )
                            last_report = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
        if partial.stat().st_size != expected_bytes:
            raise ValueError(f"Incomplete download: {partial}")
        partial.rename(target)
    if target.stat().st_size != expected_bytes:
        raise ValueError(f"Existing file has wrong size: {target}")
    result = {
        "file": name,
        "url": url,
        "generation": generation,
        "bytes": expected_bytes,
        "sha256": sha256_file(target),
    }
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    # Small annotation objects first, then the image payload.
    for name in ("docci_descriptions.jsonlines", "docci_metadata.jsonlines", "docci_images.tar.gz"):
        records.append(download(args.output_dir, name, *OBJECTS[name]))
    manifest = {
        "status": "downloaded",
        "dataset": "google/docci",
        "dataset_revision": "a0a43eaf34676ffd008fb6565dd8c2ba00d09100",
        "license": "CC-BY-4.0",
        "source": "https://google.github.io/docci/",
        "objects": records,
        "archive_integrity": "Requires full gzip/tar and image-decode validation during conversion",
    }
    (args.output_dir / "download.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

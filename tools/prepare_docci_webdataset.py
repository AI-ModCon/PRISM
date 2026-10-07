#!/usr/bin/env python3
"""Convert official DOCCI images/descriptions to audited, indexed WebDataset.

This CPU-only tool never extracts archive paths. Original JPEG bytes and full
descriptions are retained. The official qual_dev split becomes validation;
test and qual_test remain separate and are never added to training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

OFFICIAL_COUNTS = {"train": 9647, "test": 5000, "qual_dev": 100, "qual_test": 100}
SPLIT_MAP = {"train": "train", "qual_dev": "validation", "test": "test", "qual_test": "qual_test"}
SCHEMA = "prism.docci.webdataset.v1"
SOURCE_URL = "https://storage.googleapis.com/docci/data/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _archive_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe archive name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive name: {name!r}")
    return str(path)


def load_descriptions(path: Path, *, expected_counts=None) -> dict[str, dict]:
    rows = {}
    ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identifier, split = row.get("example_id"), row.get("split")
        filename, description = row.get("image_file"), row.get("description")
        if split not in SPLIT_MAP or not isinstance(identifier, str):
            raise ValueError(f"description line {line_number}: invalid id or split")
        if not re.fullmatch(re.escape(split) + r"_[0-9]+", identifier):
            raise ValueError(f"description line {line_number}: id/split mismatch")
        if filename != identifier + ".jpg" or _archive_name(filename) != filename:
            raise ValueError(f"description line {line_number}: unexpected image filename")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"description line {line_number}: empty caption")
        if identifier in ids or filename in rows:
            raise ValueError(f"duplicate description: {identifier}")
        ids.add(identifier)
        rows[filename] = row
    counts = dict(Counter(row["split"] for row in rows.values()))
    if not rows or (expected_counts is not None and counts != expected_counts):
        raise ValueError(f"description split counts {counts} do not match {expected_counts}")
    return rows


def load_metadata(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    """Keep human-relevant grouping fields, not bulky auxiliary model responses."""
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get("example_id")
            if identifier not in expected_ids or identifier in result:
                raise ValueError(f"metadata line {line_number}: unmatched or duplicate example id")
            cluster = row.get("cluster_id")
            if cluster is not None and not isinstance(cluster, (str, int)):
                raise ValueError(f"metadata line {line_number}: invalid cluster_id")
            entities = row.get("entity_tags", [])
            if not isinstance(entities, list):
                raise ValueError(f"metadata line {line_number}: entity_tags must be a list")
            result[identifier] = {
                "cluster_id": str(cluster) if cluster is not None else None,
                "entity_tags": entities,
                "image_width": row.get("image_width"),
                "image_height": row.get("image_height"),
            }
    if set(result) != expected_ids:
        raise ValueError(f"metadata missing {len(expected_ids - set(result))} described examples")
    return result


class _ShardWriter:
    def __init__(self, root: Path, split: str, max_bytes: int, max_records: int):
        self.root, self.split = root, split
        self.max_bytes, self.max_records = max_bytes, max_records
        self.number, self.count, self.total = 0, 0, 0
        self.handle = None
        self.path = None
        self.summaries = []

    def _close(self):
        if self.handle is not None:
            self.handle.close()
            self.summaries.append(
                {
                    "path": str(self.path.relative_to(self.root)),
                    "records": self.count,
                    "bytes": self.path.stat().st_size,
                    "sha256": _sha256(self.path),
                }
            )
            self.handle = None

    def add(self, key: str, image: bytes, caption: bytes, metadata: bytes) -> dict:
        # Include headers and tar block padding in the rollover estimate.
        required = sum(512 + ((len(b) + 511) // 512) * 512 for b in (image, caption, metadata))
        if self.handle is not None and (
            self.count >= self.max_records or self.handle.offset + required > self.max_bytes
        ):
            self._close()
        if self.handle is None:
            self.path = self.root / "shards" / self.split / f"docci-{self.number:06d}.tar"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = tarfile.open(self.path, "w", format=tarfile.USTAR_FORMAT)
            self.number += 1
            self.count = 0
        locator = None
        for suffix, content in (("jpg", image), ("txt", caption), ("json", metadata)):
            info = tarfile.TarInfo(f"{key}.{suffix}")
            info.size = len(content)
            info.mode = 0o644
            if suffix == "jpg":
                locator = {
                    "shard": str(self.path.relative_to(self.root)),
                    "member": info.name,
                    "header_offset": self.handle.offset,
                    "data_offset": self.handle.offset + 512,
                    "size": info.size,
                }
            self.handle.addfile(info, io.BytesIO(content))
        self.count += 1
        self.total += 1
        return locator


def convert_docci(
    descriptions: str | Path,
    images_archive: str | Path,
    output: str | Path,
    *,
    shard_bytes: int = 256 * 1024 * 1024,
    shard_records: int = 256,
    expected_counts: dict | None = OFFICIAL_COUNTS,
    metadata: str | Path | None = None,
) -> dict:
    """Validate the complete image/caption join and publish only a complete bundle.

    Cross-split decoded-pixel duplicates are fatal. Within-split duplicates
    are retained and counted. Related subjects/near duplicates cannot be
    inferred from the official description file and are not certified absent.
    ``expected_counts=None`` is reserved for tiny unit-test fixtures; the CLI
    always enforces the official split counts.
    """
    descriptions, images_archive, output = map(Path, (descriptions, images_archive, output))
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if shard_bytes <= 0 or shard_records <= 0:
        raise ValueError("shard limits must be positive")
    rows = load_descriptions(descriptions, expected_counts=expected_counts)
    metadata = Path(metadata) if metadata is not None else None
    metadata_rows = (
        load_metadata(metadata, {row["example_id"] for row in rows.values()})
        if metadata is not None
        else {}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    writers = {
        split: _ShardWriter(temporary, split, shard_bytes, shard_records)
        for split in SPLIT_MAP.values()
    }
    indexes = {split: [] for split in SPLIT_MAP.values()}
    seen, pixel_owners, within_duplicates, ignored = set(), {}, [], []
    related_owners, entity_owners = {}, {}
    archive_digest = hashlib.sha256()

    class HashingReader:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size=-1):
            data = self.handle.read(size)
            archive_digest.update(data)
            return data

    try:
        with images_archive.open("rb") as archive_file:
            reader = HashingReader(archive_file)
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                archive_names = set()
                for member in archive:
                    name = _archive_name(member.name)
                    if name in archive_names:
                        raise ValueError(f"duplicate archive member: {name}")
                    archive_names.add(name)
                    if member.isdir():
                        continue
                    if not member.isfile() or member.issparse():
                        raise ValueError(f"unsupported archive member type: {name}")
                    if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        ignored.append({"member": name, "bytes": member.size})
                        continue
                    if not name.startswith("images/") or name.count("/") != 1:
                        raise ValueError(f"unexpected image member: {name}")
                    filename = name.split("/", 1)[1]
                    if filename not in rows or filename in seen:
                        raise ValueError(f"image has no unique description: {name}")
                    if member.size <= 0 or member.size > 128 * 1024 * 1024:
                        raise ValueError(f"invalid image byte size: {name}: {member.size}")
                    image_bytes = archive.extractfile(member).read()
                    if len(image_bytes) != member.size:
                        raise ValueError(f"truncated image: {name}")
                    with Image.open(io.BytesIO(image_bytes)) as check:
                        if check.format != "JPEG":
                            raise ValueError(f"expected original JPEG: {name}")
                        check.verify()
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                        rgb.load()
                        pixel_digest = hashlib.sha256(
                            f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()
                        ).hexdigest()
                        width, height = rgb.size
                    row = rows[filename]
                    split, identifier = SPLIT_MAP[row["split"]], row["example_id"]
                    previous = pixel_owners.get(pixel_digest)
                    if previous is not None:
                        if previous[0] != split:
                            raise ValueError(
                                f"split leakage: identical pixels in {previous} "
                                f"and {(split, identifier)}"
                            )
                        within_duplicates.append(
                            {
                                "first": previous[1],
                                "duplicate": identifier,
                                "split": split,
                                "pixel_sha256": pixel_digest,
                            }
                        )
                    pixel_owners[pixel_digest] = (split, identifier)
                    image_hash = hashlib.sha256(image_bytes).hexdigest()
                    record = {
                        "schema": SCHEMA,
                        "id": identifier,
                        "task": "t2i",
                        "prompt": row["description"],
                        "source_images": [],
                        "split": split,
                        "source_split": row["split"],
                        "source_dataset": "google/docci",
                        "source_image": filename,
                        "source_archive_member": name,
                        "group_ids": [f"docci:{identifier}", f"pixels:{pixel_digest}"],
                        "image_sha256": image_hash,
                        "pixel_sha256": pixel_digest,
                        "width": width,
                        "height": height,
                        "license": "CC-BY-4.0",
                    }
                    if metadata_rows:
                        source_metadata = metadata_rows[identifier]
                        cluster = source_metadata["cluster_id"]
                        record["source_metadata"] = source_metadata
                        record["related_group_ids"] = (
                            [] if cluster is None else [f"docci-cluster:{cluster}"]
                        )
                        for group in record["related_group_ids"]:
                            related_owners.setdefault(group, set()).add(split)
                        for entity in source_metadata["entity_tags"]:
                            group = json.dumps(entity, sort_keys=True, ensure_ascii=False)
                            entity_owners.setdefault(group, set()).add(split)
                    locator = writers[split].add(
                        identifier,
                        image_bytes,
                        row["description"].encode("utf-8"),
                        _json_bytes(record),
                    )
                    indexes[split].append({**record, **locator})
                    seen.add(filename)
            # tarfile can finish before the gzip file's physical EOF.
            while reader.read(8 * 1024 * 1024):
                pass
        if seen != set(rows):
            missing = sorted(set(rows) - seen)
            raise ValueError(f"missing {len(missing)} described images: {missing[:10]}")
        for writer in writers.values():
            writer._close()
        index_details = {}
        for split, records in indexes.items():
            records.sort(key=lambda row: row["id"])
            path = temporary / f"{split}.jsonl"
            with path.open("wb") as handle:
                for record in records:
                    handle.write(_json_bytes(record))
            index_details[split] = {
                "path": path.name,
                "records": len(records),
                "sha256": _sha256(path),
            }
        summary = {
            "schema": SCHEMA,
            "complete": True,
            "source_dataset": "google/docci",
            "license": "CC-BY-4.0",
            "source_descriptions_url": SOURCE_URL + "docci_descriptions.jsonlines",
            "source_images_url": SOURCE_URL + "docci_images.tar.gz",
            "descriptions_sha256": _sha256(descriptions),
            "images_archive_sha256": archive_digest.hexdigest(),
            "official_split_counts": dict(Counter(r["split"] for r in rows.values())),
            "split_mapping": SPLIT_MAP,
            "indexes": index_details,
            "shards": [s for w in writers.values() for s in w.summaries],
            "validation": {
                "all_images_decoded": len(seen),
                "missing_images": 0,
                "cross_split_exact_pixel_duplicates": 0,
                "within_split_exact_pixel_duplicates": within_duplicates,
                "near_duplicate_and_subject_separation": "not established",
                "related_clusters_crossing_official_splits": [
                    {"group": key, "splits": sorted(splits)}
                    for key, splits in sorted(related_owners.items())
                    if len(splits) > 1
                ],
                "entity_tags_crossing_official_splits": [
                    {"entity": json.loads(key), "splits": sorted(splits)}
                    for key, splits in sorted(entity_owners.items())
                    if len(splits) > 1
                ],
                "related_cluster_policy": "preserve official splits; report overlap, do not claim subject-disjoint",
                "ignored_non_image_archive_members": ignored,
            },
            "optimization_splits": ["train"],
            "validation_splits": ["validation"],
            "sealed_test_splits": ["test", "qual_test"],
        }
        if metadata is not None:
            summary["metadata_sha256"] = _sha256(metadata)
            summary["source_metadata_url"] = SOURCE_URL + "docci_metadata.jsonlines"
            summary["metadata_fields_retained"] = [
                "cluster_id",
                "entity_tags",
                "image_width",
                "image_height",
            ]
        summary["data_fingerprint"] = hashlib.sha256(_json_bytes(summary)).hexdigest()
        (temporary / "conversion.json").write_bytes(_json_bytes(summary))
        os.rename(temporary, output)
        return summary
    except BaseException:
        for writer in writers.values():
            if writer.handle is not None:
                writer.handle.close()
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptions", required=True, type=Path)
    parser.add_argument("--images-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional official metadata for related-cluster/entity overlap auditing",
    )
    parser.add_argument("--shard-mb", type=int, default=256)
    parser.add_argument("--shard-records", type=int, default=256)
    args = parser.parse_args()
    result = convert_docci(
        args.descriptions,
        args.images_archive,
        args.output,
        shard_bytes=args.shard_mb * 1024 * 1024,
        shard_records=args.shard_records,
        metadata=args.metadata,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fingerprint": result["data_fingerprint"],
                "indexes": result["indexes"],
                "validation": result["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

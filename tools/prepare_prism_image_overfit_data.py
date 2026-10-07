"""Select bounded, original-caption PixMo pairs from existing local WebDataset shards.

No network, model loading, or training. Validation is disjoint from connector
training only: parent-pretraining overlap and unrecorded semantic duplicates
remain unknown. Source tar members are read without extractall or path reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

from PIL import Image, ImageOps, ImageStat

MAX_METADATA_BYTES = 32 * 1024**2
MAX_IMAGE_BYTES = 100 * 1024**2
MAX_MEMBER_BYTES = 8 * 1024**2
MAX_HEADERS = 30000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_groups(metadata: dict) -> list[str]:
    groups = []
    url = metadata.get("image_url")
    if isinstance(url, str) and url:
        parts = urlsplit(url)
        canonical = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))
        groups.append("source-url:" + digest(canonical.encode()))
        pieces = parts.path.strip("/").split("/")
        if parts.hostname in {"img.youtube.com", "i.ytimg.com"} and len(pieces) >= 2:
            if pieces[0] in {"vi", "vi_webp"}:
                groups.append("youtube-video:" + pieces[1])
        elif parts.hostname in {"youtube.com", "www.youtube.com", "youtu.be"}:
            video = parse_qs(parts.query).get("v", [None])[0]
            if parts.hostname == "youtu.be" and pieces:
                video = pieces[0]
            if video:
                groups.append("youtube-video:" + video)
        # Flickr sizes differ in suffix but share the numeric photo identifier.
        if parts.hostname and parts.hostname.endswith("staticflickr.com") and pieces:
            photo = pieces[-1].split("_")[0]
            if photo.isdigit():
                groups.append("flickr-photo:" + photo)
    for field in ("image_id", "video_id", "subject_id", "duplicate_group"):
        value = metadata.get(field)
        if isinstance(value, (str, int)) and str(value):
            groups.append(f"metadata-{field}:{value}")
    return sorted(set(groups))


def image_identity(data: bytes) -> tuple[str, int, tuple[float, ...], tuple[int, int]]:
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 24_000_000:
            raise ValueError("image exceeds the 24 megapixel decode bound")
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        pixel_hash = digest(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes())
        small = rgb.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
        pixels = list(small.getdata())
        dhash = 0
        for row in range(8):
            for col in range(8):
                dhash = (dhash << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
        mean = tuple(ImageStat.Stat(rgb.resize((16, 16))).mean)
        return pixel_hash, dhash, mean, rgb.size


def write_auxiliary_manifests(output: Path, rows: list[dict]) -> dict[str, str]:
    """Derive split and target-free parent diagnostic manifests from selected pairs."""
    manifests = {
        "train.jsonl": [row for row in rows if row["split"] == "train"],
        "validation.jsonl": [row for row in rows if row["split"] == "validation"],
        "parent-cases.jsonl": [
            {
                "id": row["id"],
                "prompt": "Describe the main objects in this image.",
                "source_images": [row["target_image"]],
            }
            for row in rows
            if row["split"] == "validation"
        ],
    }
    hashes = {}
    for name, records in manifests.items():
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode()
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite different manifest: {path}")
        path.write_bytes(content)
        hashes[name] = digest(content)
    return hashes


def prepare_pack(
    train_shard: Path | list[Path],
    validation_shard: Path,
    output: Path,
    *,
    train_count: int = 16,
    validation_count: int = 8,
    max_caption_words: int = 80,
    source_manifest: Path | None = None,
    excluded_keys: dict[str, str] | None = None,
) -> dict:
    train_shards = train_shard if isinstance(train_shard, list) else [train_shard]
    train_shards = [p.resolve(strict=True) for p in train_shards]
    if not train_shards or len(train_shards) > 8 or len(set(train_shards)) != len(train_shards):
        raise ValueError("provide one to eight distinct training shards")
    validation_shard = validation_shard.resolve(strict=True)
    if validation_shard in train_shards:
        raise ValueError("training and validation must use distinct source shards")
    if (
        any(p.parent.name != "shards" for p in train_shards)
        or validation_shard.parent.name != "val_shards"
    ):
        raise ValueError(
            "source shards must belong to explicit shards/ and val_shards/ directories"
        )
    if not (1 <= train_count <= 16 and 1 <= validation_count <= 8):
        raise ValueError("this diagnostic pack is bounded to 16 training and 8 validation pairs")
    if not 5 <= max_caption_words <= 80:
        raise ValueError("max_caption_words must be between 5 and 80")
    if output.exists():
        raise ValueError("output must be a new directory; existing packs are never overwritten")

    excluded_keys = excluded_keys or {}
    if any(
        not isinstance(key, str) or not isinstance(reason, str) or not reason.strip()
        for key, reason in excluded_keys.items()
    ):
        raise ValueError("excluded keys require explicit review reasons")
    selected = []
    selected_ids: set[str] = set()
    seen_groups: set[str] = set()
    seen_pixels: set[str] = set()
    seen_perceptual = []
    consumed_metadata = 0
    consumed_images = 0
    exclusions: dict[str, int] = {}
    sources = []

    def exclude(reason):
        exclusions[reason] = exclusions.get(reason, 0) + 1

    accepted_counts = {"train": 0, "validation": 0}
    selections = [("train", p, train_count) for p in train_shards]
    selections.append(("validation", validation_shard, validation_count))
    for split, shard, count in selections:
        if accepted_counts[split] >= count:
            continue
        stat = shard.stat()
        sources.append(
            {
                "path": str(shard),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "split": split,
            }
        )
        pending = {}
        accepted = accepted_counts[split]
        with tarfile.open(shard, "r:") as archive:
            for header_number, member in enumerate(archive, 1):
                if header_number > MAX_HEADERS:
                    raise ValueError("tar header scan budget exhausted")
                if not member.isfile():
                    continue
                suffix = Path(member.name).suffix.lower()
                if suffix not in IMAGE_SUFFIXES | {".txt", ".json"}:
                    continue
                key = str(Path(member.name).with_suffix(""))
                record = pending.setdefault(key, {})
                role = "image" if suffix in IMAGE_SUFFIXES else suffix[1:]
                if role in record:
                    raise ValueError(f"ambiguous repeated sample member: {key}/{role}")
                record[role] = member
                if set(record) != {"image", "txt", "json"}:
                    continue
                del pending[key]
                if key in excluded_keys:
                    exclude("explicit_quality_review")
                    continue
                if record["txt"].size > 65536 or record["json"].size > 65536:
                    exclude("metadata_member_too_large")
                    continue
                consumed_metadata += record["txt"].size + record["json"].size
                if consumed_metadata > MAX_METADATA_BYTES:
                    raise ValueError("metadata read budget exhausted")
                caption_bytes = archive.extractfile(record["txt"]).read()
                metadata_bytes = archive.extractfile(record["json"]).read()
                try:
                    caption = caption_bytes.decode("utf-8")
                    metadata = json.loads(metadata_bytes)
                except (ValueError, UnicodeError):
                    exclude("invalid_metadata")
                    continue
                words = len(caption.split())
                if not 5 <= words <= max_caption_words:
                    exclude("caption_word_budget")
                    continue
                if not isinstance(metadata, dict):
                    exclude("invalid_metadata")
                    continue
                groups = source_groups(metadata)
                if not groups:
                    exclude("missing_source_identity")
                    continue
                if seen_groups.intersection(groups):
                    exclude("source_or_subject_group_duplicate")
                    continue
                image_member = record["image"]
                if not 0 < image_member.size <= MAX_MEMBER_BYTES:
                    exclude("image_member_size")
                    continue
                consumed_images += image_member.size
                if consumed_images > MAX_IMAGE_BYTES:
                    raise ValueError("image read budget exhausted")
                image_bytes = archive.extractfile(image_member).read()
                try:
                    pixels, dhash, mean, dimensions = image_identity(image_bytes)
                except (OSError, ValueError, Image.DecompressionBombError):
                    exclude("image_decode_or_size")
                    continue
                if pixels in seen_pixels:
                    exclude("decoded_pixel_duplicate")
                    continue
                if any(
                    (dhash ^ h).bit_count() <= 4
                    and max(abs(a - b) for a, b in zip(mean, m, strict=True)) < 12
                    for h, m in seen_perceptual
                ):
                    exclude("perceptual_near_duplicate")
                    continue
                identifier = f"pixmo-{split}-{Path(key).name}"
                if identifier in selected_ids:
                    raise ValueError(f"duplicate selected sample ID: {identifier}")
                selected_ids.add(identifier)
                relative_image = f"assets/{identifier}{Path(image_member.name).suffix.lower()}"
                selected.append(
                    {
                        "row": {
                            "id": identifier,
                            "task": "t2i",
                            "prompt": caption,
                            "source_images": [],
                            "target_image": relative_image,
                            "split": split,
                            "group_ids": ["pixmo-sample:" + key, "pixels:" + pixels, *groups],
                        },
                        "image_bytes": image_bytes,
                        "caption_bytes": caption_bytes,
                        "metadata_bytes": metadata_bytes,
                        "provenance": {
                            "id": identifier,
                            "split": split,
                            "source_shard": str(shard),
                            "source_key": key,
                            "source_metadata": metadata,
                            "image_member": image_member.name,
                            "image_offset_data": image_member.offset_data,
                            "image_sha256": digest(image_bytes),
                            "image_content_sha256": pixels,
                            "image_dimensions": dimensions,
                            "dhash64": f"{dhash:016x}",
                            "caption_member": record["txt"].name,
                            "caption_offset_data": record["txt"].offset_data,
                            "caption_sha256": digest(caption_bytes),
                            "caption_words": words,
                            "metadata_member": record["json"].name,
                            "metadata_sha256": digest(metadata_bytes),
                        },
                    }
                )
                seen_groups.update(groups)
                seen_pixels.add(pixels)
                seen_perceptual.append((dhash, mean))
                accepted += 1
                if accepted == count:
                    break
        accepted_counts[split] = accepted
        if shard.stat().st_size != stat.st_size or shard.stat().st_mtime_ns != stat.st_mtime_ns:
            raise ValueError("source shard changed during selection")

    for split, count in (("train", train_count), ("validation", validation_count)):
        if accepted_counts[split] != count:
            raise ValueError(
                f"insufficient eligible {split} examples: {accepted_counts[split]}/{count}"
            )

    output.mkdir(parents=True)
    (output / "assets").mkdir()
    (output / "source_sidecars").mkdir()
    rows = [record["row"] for record in selected]
    for record in selected:
        (output / record["row"]["target_image"]).write_bytes(record["image_bytes"])
        identifier = record["row"]["id"]
        (output / "source_sidecars" / f"{identifier}.txt").write_bytes(record["caption_bytes"])
        (output / "source_sidecars" / f"{identifier}.json").write_bytes(record["metadata_bytes"])
    manifest_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    (output / "manifest.jsonl").write_bytes(manifest_bytes)
    report = {
        "schema_version": 1,
        "evidence_kind": "original_caption_connector_diagnostic_data",
        "selection": "first eligible members in train shard then validation shard archive order",
        "counts": {"train": train_count, "validation": validation_count},
        "max_caption_words": max_caption_words,
        "caption_policy": "original UTF-8 text bytes preserved; no rewriting or truncation",
        "held_out_scope": "connector training only; parent pretraining overlap unknown",
        "deduplication": "source URLs, recorded image/video/subject IDs, YouTube/Flickr IDs, decoded RGB SHA256, dHash distance <=4 with RGB means <12",
        "semantic_identity_limit": "unrecorded subjects and semantic duplicates cannot be proven absent",
        "sources": sources,
        "source_manifest_sha256": digest(source_manifest.read_bytes()) if source_manifest else None,
        "manifest_sha256": digest(manifest_bytes),
        "auxiliary_manifest_sha256": write_auxiliary_manifests(output, rows),
        "metadata_bytes_read": consumed_metadata,
        "image_bytes_read": consumed_images,
        "asset_bytes": sum(len(record["image_bytes"]) for record in selected),
        "exclusions": exclusions,
        "quality_review_exclusions": excluded_keys,
        "examples": [record["provenance"] for record in selected],
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", type=Path, required=True, action="append")
    parser.add_argument("--validation-shard", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--excluded-keys-json",
        type=Path,
        help="JSON object mapping reviewed source sample keys to rejection reasons",
    )
    parser.add_argument("--max-caption-words", type=int, default=80)
    args = parser.parse_args()
    report = prepare_pack(
        args.train_shard,
        args.validation_shard,
        args.output_dir,
        max_caption_words=args.max_caption_words,
        source_manifest=args.source_manifest,
        excluded_keys=json.loads(args.excluded_keys_json.read_text())
        if args.excluded_keys_json
        else None,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "counts",
                    "manifest_sha256",
                    "asset_bytes",
                    "image_bytes_read",
                    "exclusions",
                )
            }
        )
    )


if __name__ == "__main__":
    main()

"""Map-style generation dataset over audited, uncompressed WebDataset shards.

Indexes contain original captions and byte locators, so training neither
unpacks thousands of images nor walks tar archives for each minibatch. A
complete conversion audit and all split indexes are checked before selection.
The target can never become a PRISM vision input in this text-to-image loader.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps
from src.data.image_generation import vae_target_transform
from torch.utils.data import Dataset

SCHEMA = "prism.docci.webdataset.v1"


def _contained_path(value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "://" in value:
        raise ValueError(f"invalid relative dataset path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe dataset path: {value!r}")
    path = (root / value).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"dataset path is absent or escapes root: {value!r}")
    return path


@dataclass(frozen=True)
class WebDatasetImageRecord:
    id: str
    prompt: str
    split: str
    group_ids: tuple[str, ...]
    shard: Path
    member: str
    header_offset: int
    data_offset: int
    size: int
    image_sha256: str
    pixel_sha256: str
    width: int
    height: int
    task: str = "t2i"
    source_paths: tuple = ()
    source_ids: tuple = ()
    target_path: None = None


class ImageGenerationWebDataset(Dataset):
    def __init__(
        self,
        index: str | Path,
        *,
        target_size: tuple[int, int] | None = None,
        split: str | None = "train",
    ):
        self.index = Path(index).resolve()
        self.manifest = self.index
        root = self.index.parent
        summary = json.loads((root / "conversion.json").read_text(encoding="utf-8"))
        if summary.get("schema") != SCHEMA or summary.get("complete") is not True:
            raise ValueError("a completed DOCCI conversion audit is required")
        fingerprint = summary.get("data_fingerprint")
        body = {key: value for key, value in summary.items() if key != "data_fingerprint"}
        canonical = (json.dumps(body, sort_keys=True, ensure_ascii=False) + "\n").encode()
        if hashlib.sha256(canonical).hexdigest() != fingerprint:
            raise ValueError("conversion audit fingerprint mismatch")
        validation = summary.get("validation", {})
        if (
            validation.get("missing_images") != 0
            or validation.get("cross_split_exact_pixel_duplicates") != 0
        ):
            raise ValueError("conversion audit reports missing images or split leakage")
        self.validation_report = validation
        self.data_fingerprint = fingerprint
        self.target_size, self.split = target_size, split
        owners, ids, selected, shard_sizes = {}, set(), [], {}
        found_index = False
        shard_details = {row["path"]: row for row in summary["shards"]}
        for declared_split, detail in summary["indexes"].items():
            path = _contained_path(detail["path"], root)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != detail["sha256"]:
                raise ValueError(f"index hash mismatch: {path}")
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if len(rows) != detail["records"]:
                raise ValueError(f"index count mismatch: {path}")
            for row in rows:
                if (
                    row.get("schema") != SCHEMA
                    or row.get("task") != "t2i"
                    or row.get("source_images") != []
                ):
                    raise ValueError("only explicit, target-separated t2i records are supported")
                if row.get("split") != declared_split or row.get("id") in ids:
                    raise ValueError("duplicate id or split/index mismatch")
                if not isinstance(row.get("id"), str) or not row["id"]:
                    raise ValueError("record id must be nonempty")
                ids.add(row["id"])
                if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                    raise ValueError("record prompt must be nonempty")
                groups = row.get("group_ids")
                if (
                    not isinstance(groups, list)
                    or not groups
                    or any(not isinstance(g, str) or not g for g in groups)
                ):
                    raise ValueError("record group_ids must be nonempty strings")
                keys = [("group", group) for group in groups] + [("pixels", row["pixel_sha256"])]
                for key in keys:
                    previous = owners.get(key)
                    if previous is not None and previous != declared_split:
                        raise ValueError(f"split leakage for {key}: {previous}/{declared_split}")
                    owners[key] = declared_split
                shard = _contained_path(row["shard"], root)
                if row["shard"] not in shard_details:
                    raise ValueError("index references an unaudited shard")
                if shard not in shard_sizes:
                    shard_sizes[shard] = shard.stat().st_size
                    if shard_sizes[shard] != shard_details[row["shard"]]["bytes"]:
                        raise ValueError(f"shard byte-size mismatch: {shard}")
                offsets = [row.get(k) for k in ("header_offset", "data_offset", "size")]
                if any(type(v) is not int for v in offsets):
                    raise ValueError("tar offsets and size must be integers")
                header, data, size = offsets
                if (
                    header < 0
                    or header % 512
                    or data != header + 512
                    or size <= 0
                    or size > 128 * 1024 * 1024
                    or data + size > shard_sizes[shard]
                ):
                    raise ValueError("invalid or out-of-bounds tar member locator")
                if (
                    row.get("member") != row["id"] + ".jpg"
                    or "/" in row["member"]
                    or "\\" in row["member"]
                ):
                    raise ValueError("invalid tar image member name")
                if path == self.index and (split is None or declared_split == split):
                    selected.append(
                        WebDatasetImageRecord(
                            row["id"],
                            row["prompt"],
                            declared_split,
                            tuple(groups),
                            shard,
                            row["member"],
                            header,
                            data,
                            size,
                            row["image_sha256"],
                            row["pixel_sha256"],
                            row["width"],
                            row["height"],
                        )
                    )
            if path == self.index:
                found_index = True
        if not found_index or not selected:
            raise ValueError(f"no audited examples in {self.index} for split {split!r}")
        if validation.get("all_images_decoded") != len(ids):
            raise ValueError("conversion decoded-count does not cover all records")
        self.records = selected
        self.all_records = selected

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with record.shard.open("rb") as handle:
            handle.seek(record.header_offset)
            header = handle.read(512)
            try:
                info = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            except (tarfile.HeaderError, UnicodeError) as exc:
                raise ValueError(f"invalid indexed tar header: {record.id}") from exc
            if (
                not info.isfile()
                or info.issparse()
                or info.name != record.member
                or info.size != record.size
            ):
                raise ValueError(f"indexed tar header mismatch: {record.id}")
            image_bytes = handle.read(record.size)
        if (
            len(image_bytes) != record.size
            or hashlib.sha256(image_bytes).hexdigest() != record.image_sha256
        ):
            raise ValueError(f"indexed image checksum mismatch: {record.id}")
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.load()
            if rgb.size != (record.width, record.height):
                raise ValueError(f"indexed image dimensions mismatch: {record.id}")
            target = vae_target_transform(rgb, self.target_size)
        return {
            "id": record.id,
            "task": record.task,
            "prompt": record.prompt,
            "split": record.split,
            "group_ids": record.group_ids,
            "source_ids": (),
            "encoder_source_images": [],
            "reference_images": [],
            "target_image": target,
        }

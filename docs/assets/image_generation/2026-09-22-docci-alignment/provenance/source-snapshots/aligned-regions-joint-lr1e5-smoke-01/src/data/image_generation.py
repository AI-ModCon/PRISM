"""Explicit source/target data for image-output training.

JSONL rows use ``id``, ``task`` (t2i/edit/in_context), ``prompt``,
``source_images`` (an ordered list of local paths), ``target_image``, ``split``,
and ``group_ids`` (all subject/video/duplicate groups known for the example).
Paths are relative to the manifest. ``reference_paths`` and ``group`` are
accepted for compatibility with target-free evaluation manifests.

Source images have two distinct paths: an explicit input-encoder transform,
and RGB PIL images for the generator's own reference processor. Targets only
enter the VAE-normalized target tensor. Nothing implicitly copies a target to
the sources or treats an instruction as a supervised text answer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

_TASKS = {
    "t2i": "t2i",
    "text_to_image": "t2i",
    "edit": "edit",
    "editing": "edit",
    "in_context": "in_context",
    "multi_reference": "in_context",
}


@dataclass(frozen=True)
class ImageGenerationRecord:
    id: str
    task: str
    prompt: str
    source_paths: tuple[Path, ...]
    source_ids: tuple[str, ...]
    target_path: Path | None
    split: str
    group_ids: tuple[str, ...]


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _local_path(value: Any, base: Path, name: str, verify_files: bool) -> Path:
    text = _nonempty_string(value, name)
    if "://" in text:
        raise ValueError(f"{name} must be a local file, not a URL")
    path = (base / text).resolve()
    if verify_files and not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def load_image_generation_manifest(
    manifest: str | Path,
    *,
    require_targets: bool = True,
    verify_files: bool = True,
) -> list[ImageGenerationRecord]:
    """Read and validate row structure, without opening image pixels.

    Pass all resulting records to :func:`validate_manifest_splits` when train
    and validation/test examples live in separate manifests.
    """
    manifest = Path(manifest).resolve()
    records = []
    seen_ids: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each row must be an object")
                identifier = _nonempty_string(row.get("id", row.get("case_id")), "id")
                if identifier in seen_ids:
                    raise ValueError(f"duplicate example id: {identifier}")
                task = _TASKS.get(row.get("task"))
                if task is None:
                    raise ValueError("task must be t2i, edit, or in_context")
                prompt = _nonempty_string(row.get("prompt"), "prompt")
                split = _nonempty_string(row.get("split"), "split")
                group_ids = row.get("group_ids")
                if group_ids is None:
                    group = row.get("group_id", row.get("group"))
                    group_ids = [group] if group is not None else None
                if not isinstance(group_ids, list) or not group_ids:
                    raise ValueError("group_ids must identify at least one split group")
                groups = tuple(_nonempty_string(g, "group_id") for g in group_ids)
                if "source_images" in row and "reference_paths" in row:
                    raise ValueError("use source_images or reference_paths, not both")
                sources = row.get("source_images", row.get("reference_paths", []))
                if not isinstance(sources, list):
                    raise ValueError("source_images must be an ordered list")
                source_paths = tuple(
                    _local_path(p, manifest.parent, "source image", verify_files) for p in sources
                )
                if task == "t2i" and source_paths:
                    raise ValueError("t2i examples cannot contain source images")
                if task == "edit" and len(source_paths) != 1:
                    raise ValueError("edit examples require exactly one source image")
                if task == "in_context" and len(source_paths) < 2:
                    raise ValueError("in_context examples require at least two sources")
                source_ids = row.get("source_ids", sources)
                if not isinstance(source_ids, list) or len(source_ids) != len(source_paths):
                    raise ValueError("source_ids must align with the source image order")
                source_ids = tuple(_nonempty_string(s, "source_id") for s in source_ids)
                if len(set(source_ids)) != len(source_ids):
                    raise ValueError("source_ids must be unique within an example")
                target = row.get("target_image")
                if target is None and require_targets:
                    raise ValueError("target_image is required for training")
                target_path = (
                    _local_path(target, manifest.parent, "target image", verify_files)
                    if target is not None
                    else None
                )
                if target_path in source_paths:
                    raise ValueError("target_image must not also be a source image")
                records.append(
                    ImageGenerationRecord(
                        identifier,
                        task,
                        prompt,
                        source_paths,
                        source_ids,
                        target_path,
                        split,
                        groups,
                    )
                )
                seen_ids.add(identifier)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{manifest}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"manifest contains no examples: {manifest}")
    return records


def validate_manifest_splits(
    records: Iterable[ImageGenerationRecord], *, check_content: bool = True
) -> dict[str, str]:
    """Reject cross-split groups, paths, and exact image-content duplicates.

    Every known grouping identifier belongs in ``group_ids``. This check
    cannot discover semantic/near duplicates or subjects absent from metadata.
    Content fingerprints use decoded RGB pixels, catching exact duplicate
    images stored with different file metadata; they are not perceptual hashes.
    Returns path-to-content fingerprints for the run's data provenance.
    """
    ownership: dict[tuple[str, str], tuple[str, str]] = {}
    fingerprints: dict[Path, str] = {}
    for record in records:
        keys = [("id", record.id)] + [("group", g) for g in record.group_ids]
        paths = list(record.source_paths)
        if record.target_path is not None:
            paths.append(record.target_path)
        for path in paths:
            keys.append(("path", str(path)))
            if check_content:
                if path not in fingerprints:
                    with Image.open(path) as image:
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                        digest = hashlib.sha256()
                        digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode())
                        digest.update(rgb.tobytes())
                        fingerprints[path] = digest.hexdigest()
                keys.append(("image_content", fingerprints[path]))
        if check_content and record.target_path is not None:
            target_fingerprint = fingerprints[record.target_path]
            if any(fingerprints[p] == target_fingerprint for p in record.source_paths):
                raise ValueError(f"{record.id}: target pixels are identical to a source image")
        for key in keys:
            previous = ownership.get(key)
            if previous is not None and previous[0] != record.split:
                raise ValueError(
                    f"split leakage for {key[0]} {key[1]!r}: "
                    f"{previous[0]}/{previous[1]} and {record.split}/{record.id}"
                )
            ownership[key] = (record.split, record.id)
    return {str(path): value for path, value in fingerprints.items()}


def vae_target_transform(image: Image.Image, size: tuple[int, int] | None) -> torch.Tensor:
    """RGB CHW float32 in [-1, 1]; optional explicit resize uses (height, width).

    This is target pixel normalization, not VAE latent scaling. The image
    decoder owns its pinned codec's latent normalization and flow objective.
    """
    image = image.convert("RGB")
    if size is not None:
        height, width = size
        if height <= 0 or width <= 0:
            raise ValueError("target height and width must be positive")
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return pixels.reshape(image.height, image.width, 3).permute(2, 0, 1).float() / 127.5 - 1


class ImageGenerationDataset(Dataset):
    """Finite local manifest dataset; validates *all* splits before selection."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        source_transform: Callable[[Image.Image], torch.Tensor] | None,
        target_size: tuple[int, int] | None = None,
        split: str | None = "train",
        require_targets: bool = True,
        check_content: bool = True,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.all_records = load_image_generation_manifest(
            self.manifest, require_targets=require_targets
        )
        self.image_fingerprints = validate_manifest_splits(
            self.all_records, check_content=check_content
        )
        self.data_fingerprint = hashlib.sha256(
            self.manifest.read_bytes()
            + json.dumps(self.image_fingerprints, sort_keys=True).encode()
        ).hexdigest()
        self.records = [r for r in self.all_records if split is None or r.split == split]
        if not self.records:
            raise ValueError(f"no examples for split {split!r}")
        if source_transform is None and any(r.source_paths for r in self.records):
            raise ValueError("source_transform is required for image input-encoder processing")
        self.source_transform = source_transform
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_rgb(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        references = [self._read_rgb(path) for path in record.source_paths]
        encoded = []
        for image in references:
            values = self.source_transform(image.copy())  # type: ignore[misc]
            if not isinstance(values, torch.Tensor) or values.ndim != 3 or values.shape[0] != 3:
                raise ValueError("source_transform must return a [3, H, W] tensor")
            if not values.is_floating_point() or not torch.isfinite(values).all():
                raise ValueError("source_transform must return finite floating-point values")
            encoded.append(values)
        target = (
            vae_target_transform(self._read_rgb(record.target_path), self.target_size)
            if record.target_path is not None
            else None
        )
        return {
            "id": record.id,
            "task": record.task,
            "prompt": record.prompt,
            "split": record.split,
            "group_ids": record.group_ids,
            "source_ids": record.source_ids,
            "encoder_source_images": encoded,
            "reference_images": references,
            "target_image": target,
        }


class ImageGenerationCollator:
    """Build the target-separated ``forward_outputs`` argument bundle.

    The tokenizer must return input_ids and attention_mask. Truncation is
    disabled: silently dropping an instruction/source role is invalid here.
    Prompts are tokenized exactly as recorded. For an interleaved PRISM config,
    the manifest must supply the configured image marker pair at each intended
    reference position, in source-list order; the model compiler validates it.
    This collator does not invent image positions or apply a chat template.
    Multi-reference images use [B, N, 3, H, W] with a contiguous true-prefix
    image_mask; native references retain their per-example list ordering.
    """

    def __init__(self, tokenizer: Any, *, max_text_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty image-generation batch")
        for sample in samples:
            counts = {
                len(sample["encoder_source_images"]),
                len(sample["reference_images"]),
                len(sample["source_ids"]),
            }
            if len(counts) != 1:
                raise ValueError("encoder images, native references, and source IDs must align")
        tokens = self.tokenizer(
            [sample["prompt"] for sample in samples],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        text = tokens["input_ids"]
        attention_mask = tokens.get("attention_mask")
        if attention_mask is None or text.ndim != 2 or attention_mask.shape != text.shape:
            raise ValueError("tokenizer must return aligned 2D input_ids and attention_mask")
        if self.max_text_length is not None and text.shape[1] > self.max_text_length:
            raise ValueError("instruction exceeds max_text_length; truncation is not allowed")
        inputs: dict[str, Any] = {"text": text, "text_attention_mask": attention_mask}
        count = max(len(sample["reference_images"]) for sample in samples)
        source_mask = torch.zeros((len(samples), count), dtype=torch.bool)
        source_values = [image for s in samples for image in s["encoder_source_images"]]
        if source_values:
            shape, dtype = source_values[0].shape, source_values[0].dtype
            if any(image.shape != shape or image.dtype != dtype for image in source_values):
                raise ValueError("input-encoder source tensors must share shape and dtype")
            images = torch.zeros((len(samples), count, *shape), dtype=dtype)
            for index, sample in enumerate(samples):
                n_sources = len(sample["encoder_source_images"])
                if n_sources != len(sample["reference_images"]):
                    raise ValueError("encoder and native source images must align")
                if n_sources:
                    images[index, :n_sources] = torch.stack(sample["encoder_source_images"])
                    source_mask[index, :n_sources] = True
            inputs.update(image=images, image_mask=source_mask)
        targets: dict[str, Any] = {}
        output_specs: dict[str, Any] = {}
        target_values = [sample["target_image"] for sample in samples]
        if any(target is not None for target in target_values):
            if any(target is None for target in target_values):
                raise ValueError("batch cannot mix supervised and target-free examples")
            if any(target.shape != target_values[0].shape for target in target_values):
                raise ValueError("target shapes differ; use explicit target_size or shape buckets")
            targets["image"] = torch.stack(target_values)
            height, width = targets["image"].shape[-2:]
            output_specs["image"] = {"height": height, "width": width}
        return {
            "inputs": inputs,
            "targets": targets,
            "requested_outputs": ["image"],
            "output_specs": output_specs,
            "native_context": {
                "image": {
                    "reference_images": [list(s["reference_images"]) for s in samples],
                    "source_ids": [list(s["source_ids"]) for s in samples],
                    "source_mask": source_mask,
                }
            },
            "metadata": {
                "ids": [s["id"] for s in samples],
                "tasks": [s["task"] for s in samples],
                "splits": [s["split"] for s in samples],
                "group_ids": [list(s["group_ids"]) for s in samples],
            },
        }


def move_image_batch_to_device(value: Any, device: str | torch.device) -> Any:
    """Move nested tensors, preserving raw PIL references and metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_image_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_image_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_image_batch_to_device(item, device) for item in value)
    return value

"""Dataset contract for AstroPRISM Stage 1A raw-image alignment.

The dataset intentionally consumes the calibrated raw-image artifact rather
than the released AION embeddings. The latter remain a parity reference and
are only used here as the source of the matched caption records.

Each sample contains one ``float32[4, 160, 160]`` Legacy Survey image and a
Qwen chat sequence. Only tokens belonging to the final assistant response are
supervised; the system/user prefix is masked with ``-100``. The AION image
observation is supplied out of band under ``image`` and expands to 576 tokens
in the model, so it is included in the merged-length safety check.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

ASTROPRISM_IMAGE_SHAPE = (4, 160, 160)
ASTROPRISM_IMAGE_TOKENS = 576
ASTROPRISM_SOURCE_BANDS = ("des-g", "des-r", "des-i", "des-z")
ASTROPRISM_AION_BANDS = ("DES-G", "DES-R", "DES-I", "DES-Z")
IGNORE_INDEX = -100

DEFAULT_SYSTEM_PROMPT = (
    "You are an astronomy assistant. Describe only morphology and visual evidence "
    "supported by the supplied calibrated four-band galaxy image."
)
DEFAULT_USER_PROMPT = "Describe the galaxy in this calibrated four-band image."

_CAPTION_FIELD_BY_RELEASE_STATUS = {
    "released_reused": "original",
    "released_revised": "revised",
}


class AstroPRISMStage1ADatasetError(ValueError):
    """Raised when a Stage 1A artifact or sample violates its contract."""


@dataclass(frozen=True)
class AstroPRISMStage1ADatasetConfig:
    """Paths and safety settings for :class:`AstroPRISMStage1ADataset`.

    ``allow_unsafe_candidates`` exists for unit tests and data inspection. A
    training launch should leave it false so pending/unreviewed candidates and
    records without a source-level split cannot enter a batch.
    """

    raw_manifest_path: Path
    candidates_manifest_path: Path
    candidates_path: Path | None = None
    split: str | None = None
    max_merged_seq_length: int = 2048
    caption_field: str = "auto"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT
    allow_unsafe_candidates: bool = False
    accepted_statuses: tuple[str, ...] = ("accepted", "approved")
    accepted_caption_statuses: tuple[str, ...] = (
        "released_reused",
        "released_revised",
    )
    verify_file_hashes: bool = False


def _read_json(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise AstroPRISMStage1ADatasetError(f"{kind} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AstroPRISMStage1ADatasetError(f"cannot read {kind} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AstroPRISMStage1ADatasetError(f"{kind} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AstroPRISMStage1ADatasetError(f"candidate JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise AstroPRISMStage1ADatasetError(
                        f"candidate JSONL contains a blank row at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AstroPRISMStage1ADatasetError(
                        f"candidate JSONL line {line_number} must be an object"
                    )
                records.append(value)
    except json.JSONDecodeError as exc:
        raise AstroPRISMStage1ADatasetError(
            f"cannot parse candidate JSONL {path} at line {exc.lineno}: {exc.msg}"
        ) from exc
    return records


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_output_path(manifest_path: Path, output: dict[str, Any], *, name: str) -> Path:
    relative = output.get("path")
    if not isinstance(relative, str) or not relative:
        raise AstroPRISMStage1ADatasetError(f"manifest output {name!r} has no path")
    path = Path(relative)
    return path if path.is_absolute() else manifest_path.parent / path


def _coerce_token_ids(value: Any, *, context: str) -> list[int]:
    """Normalize common ``apply_chat_template`` return forms to one ID list."""

    if isinstance(value, dict):
        if "input_ids" not in value:
            raise AstroPRISMStage1ADatasetError(
                f"tokenizer returned a mapping without input_ids for {context}"
            )
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list | tuple):
        value = list(value[0])
    if not isinstance(value, list) or any(isinstance(item, list | tuple) for item in value):
        raise AstroPRISMStage1ADatasetError(
            f"tokenizer must return one flat token sequence for {context}"
        )
    try:
        token_ids = [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise AstroPRISMStage1ADatasetError(
            f"tokenizer returned non-integer token IDs for {context}"
        ) from exc
    if not token_ids:
        raise AstroPRISMStage1ADatasetError(f"tokenizer returned no token IDs for {context}")
    return token_ids


def _source_split_keys(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return every stable source identity whose split must remain atomic."""

    keys: list[tuple[str, str]] = []
    for field in ("source_group_id", "paired_targetid", "caption_object_id"):
        value = record.get(field)
        if value is not None and str(value).strip():
            keys.append((field, str(value)))
    return tuple(keys)


def _validate_source_split_integrity(records: list[dict[str, Any]]) -> None:
    assignments: dict[tuple[str, str], str] = {}
    for record in records:
        split = record.get("split")
        if split is None or not str(split).strip():
            continue
        split = str(split)
        for source_key in _source_split_keys(record):
            previous = assignments.setdefault(source_key, split)
            if previous != split:
                field, source_id = source_key
                raise AstroPRISMStage1ADatasetError(
                    f"source split leakage: {field}={source_id!r} appears in both "
                    f"{previous!r} and {split!r}"
                )


class AstroPRISMStage1ADataset(Dataset):
    """Map-style Stage 1A dataset backed by mmap-safe calibrated cutouts."""

    def __init__(
        self,
        config: AstroPRISMStage1ADatasetConfig,
        tokenizer: Any,
    ) -> None:
        self.config = config
        self.active_modalities = {"image"}
        self.tokenizer = tokenizer

        if config.max_merged_seq_length <= ASTROPRISM_IMAGE_TOKENS:
            raise AstroPRISMStage1ADatasetError(
                "max_merged_seq_length must exceed the fixed 576-token AION image prefix"
            )
        if not config.caption_field.strip():
            raise AstroPRISMStage1ADatasetError("caption_field must be non-empty")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise AstroPRISMStage1ADatasetError(
                "tokenizer must provide apply_chat_template for Qwen chat formatting"
            )

        self.raw_manifest_path = Path(config.raw_manifest_path)
        self.candidates_manifest_path = Path(config.candidates_manifest_path)
        self.raw_manifest = _read_json(self.raw_manifest_path, kind="raw-image manifest")
        self.candidates_manifest = _read_json(
            self.candidates_manifest_path, kind="candidate manifest"
        )
        self._validate_manifest_types()

        candidate_output = self.candidates_manifest.get("outputs", {}).get("candidates")
        if not isinstance(candidate_output, dict):
            raise AstroPRISMStage1ADatasetError("candidate manifest is missing outputs.candidates")
        self.candidates_path = (
            Path(config.candidates_path)
            if config.candidates_path is not None
            else _resolve_output_path(
                self.candidates_manifest_path, candidate_output, name="candidates"
            )
        )
        self._all_records = _read_jsonl(self.candidates_path)
        self._validate_candidate_file(candidate_output)

        image_output = self.raw_manifest.get("outputs", {}).get("image_array")
        if not isinstance(image_output, dict):
            raise AstroPRISMStage1ADatasetError("raw-image manifest is missing outputs.image_array")
        self.image_path = _resolve_output_path(
            self.raw_manifest_path, image_output, name="image_array"
        )
        self._images = self._open_and_validate_images(image_output)

        self._validate_roster_identity()
        self._validate_candidate_alignment()
        _validate_source_split_integrity(self._all_records)

        if config.split is None:
            self._indices = list(range(len(self._all_records)))
        else:
            self._indices = [
                index
                for index, record in enumerate(self._all_records)
                if record.get("split") == config.split
            ]
        if not self._indices:
            raise AstroPRISMStage1ADatasetError(
                f"no candidates found for requested split {config.split!r}"
            )
        self._validate_training_safety()

    def _validate_manifest_types(self) -> None:
        if self.raw_manifest.get("artifact_type") != "astroprism_raw_images":
            raise AstroPRISMStage1ADatasetError(
                "raw manifest artifact_type must be 'astroprism_raw_images'"
            )
        candidate_type = self.candidates_manifest.get("artifact_type")
        if candidate_type not in {
            "astroprism_stage1a_candidates",
            "astroprism_aion_feature_parity_reference",
        }:
            raise AstroPRISMStage1ADatasetError(
                f"candidate manifest has an unsupported artifact_type: {candidate_type!r}"
            )

    def _validate_candidate_file(self, output: dict[str, Any]) -> None:
        declared_count = output.get("row_count")
        if declared_count is not None and declared_count != len(self._all_records):
            raise AstroPRISMStage1ADatasetError(
                f"candidate row count mismatch: manifest declares {declared_count}, "
                f"JSONL contains {len(self._all_records)}"
            )
        if not self._all_records:
            raise AstroPRISMStage1ADatasetError("candidate JSONL is empty")
        if self.config.verify_file_hashes and output.get("sha256"):
            actual = _sha256_file(self.candidates_path)
            if actual != output["sha256"]:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate JSONL SHA-256 mismatch: expected {output['sha256']}, got {actual}"
                )

    def _open_and_validate_images(self, output: dict[str, Any]) -> np.ndarray:
        if not self.image_path.is_file():
            raise AstroPRISMStage1ADatasetError(
                f"raw image array does not exist: {self.image_path}"
            )
        try:
            images = np.load(self.image_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise AstroPRISMStage1ADatasetError(
                f"cannot mmap raw image array {self.image_path}: {exc}"
            ) from exc

        expected_shape = (len(self._all_records), *ASTROPRISM_IMAGE_SHAPE)
        if images.shape != expected_shape:
            raise AstroPRISMStage1ADatasetError(
                f"raw image array must have shape {expected_shape}, got {images.shape}"
            )
        if images.dtype != np.dtype("float32"):
            raise AstroPRISMStage1ADatasetError(
                f"raw image array must be float32, got {images.dtype}"
            )
        declared_shape = output.get("shape")
        if declared_shape is not None and list(images.shape) != declared_shape:
            raise AstroPRISMStage1ADatasetError(
                f"raw image shape disagrees with manifest: {images.shape} != {declared_shape}"
            )
        declared_dtype = output.get("dtype")
        if declared_dtype is not None and np.dtype(declared_dtype) != images.dtype:
            raise AstroPRISMStage1ADatasetError(
                f"raw image dtype disagrees with manifest: {images.dtype} != {declared_dtype}"
            )
        if self.config.verify_file_hashes and output.get("sha256"):
            actual = _sha256_file(self.image_path)
            if actual != output["sha256"]:
                raise AstroPRISMStage1ADatasetError(
                    f"raw image SHA-256 mismatch: expected {output['sha256']}, got {actual}"
                )
        return images

    def _validate_roster_identity(self) -> None:
        raw_roster = self.raw_manifest.get("roster", {})
        candidate_pin = self.candidates_manifest.get("pins", {}).get("roster_sha256")
        raw_sha = raw_roster.get("sha256") if isinstance(raw_roster, dict) else None
        if raw_sha is not None and candidate_pin is not None and raw_sha != candidate_pin:
            raise AstroPRISMStage1ADatasetError(
                f"raw/candidate roster SHA-256 mismatch: {raw_sha} != {candidate_pin}"
            )

    def _validate_candidate_alignment(self) -> None:
        raw_candidates = self.raw_manifest.get("candidates")
        if not isinstance(raw_candidates, list):
            raise AstroPRISMStage1ADatasetError("raw manifest is missing candidates list")
        if len(raw_candidates) != len(self._all_records):
            raise AstroPRISMStage1ADatasetError(
                f"raw/caption candidate count mismatch: {len(raw_candidates)} != "
                f"{len(self._all_records)}"
            )

        seen_indices: set[int] = set()
        for index, (raw_record, caption_record) in enumerate(
            zip(raw_candidates, self._all_records, strict=True)
        ):
            for kind, record in (("raw", raw_record), ("caption", caption_record)):
                if not isinstance(record, dict):
                    raise AstroPRISMStage1ADatasetError(
                        f"{kind} candidate at row {index} must be an object"
                    )
                if record.get("candidate_index") != index:
                    raise AstroPRISMStage1ADatasetError(
                        f"{kind} candidate row {index} has candidate_index="
                        f"{record.get('candidate_index')!r}; roster order must be preserved"
                    )
            if index in seen_indices:
                raise AstroPRISMStage1ADatasetError(f"duplicate candidate_index {index}")
            bands = raw_record.get("bands")
            if not isinstance(bands, dict):
                raise AstroPRISMStage1ADatasetError(
                    f"raw candidate {index} is missing verified band metadata"
                )
            source_bands = tuple(bands.get("source", ()))
            aion_bands = tuple(bands.get("aion", ()))
            if source_bands != ASTROPRISM_SOURCE_BANDS or aion_bands != ASTROPRISM_AION_BANDS:
                raise AstroPRISMStage1ADatasetError(
                    f"raw candidate {index} has unsupported band order: "
                    f"source={source_bands!r}, aion={aion_bands!r}"
                )
            seen_indices.add(index)
            for field in ("paired_targetid", "caption_object_id"):
                if str(raw_record.get(field)) != str(caption_record.get(field)):
                    raise AstroPRISMStage1ADatasetError(
                        f"candidate {index} {field} mismatch between raw and caption artifacts: "
                        f"{raw_record.get(field)!r} != {caption_record.get(field)!r}"
                    )

    def _validate_training_safety(self) -> None:
        """Reject unsafe records that would actually enter this dataset view.

        Split-integrity checks run over the complete manifest, but review gates
        apply to ``self._indices``. This permits a provenance-complete manifest
        to retain explicitly rejected or unassigned rows while a split-specific
        view consumes only accepted records. A ``split=None`` view remains strict
        because every record would enter the dataset.
        """

        if self.config.allow_unsafe_candidates:
            for index in self._indices:
                self._caption_text(self._all_records[index])
            return
        accepted = {status.casefold() for status in self.config.accepted_statuses}
        accepted_captions = {status.casefold() for status in self.config.accepted_caption_statuses}
        if not accepted:
            raise AstroPRISMStage1ADatasetError("accepted_statuses must be non-empty")
        if not accepted_captions:
            raise AstroPRISMStage1ADatasetError("accepted_caption_statuses must be non-empty")

        for index in self._indices:
            record = self._all_records[index]
            status = str(record.get("acceptance_status", "")).strip()
            if status.casefold() not in accepted:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} has acceptance_status={status!r}; expected one of "
                    f"{sorted(self.config.accepted_statuses)!r}. Pass "
                    "allow_unsafe_candidates=True only for inspection/testing."
                )
            split = record.get("split")
            if split is None or not str(split).strip():
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} has split=null; source-level splits are required. "
                    "Pass allow_unsafe_candidates=True only for inspection/testing."
                )
            caption = record.get("caption")
            caption_status = (
                str(caption.get("status", "")).strip() if isinstance(caption, dict) else ""
            )
            if caption_status.casefold() not in accepted_captions:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} has caption.status={caption_status!r}; expected one "
                    f"of {sorted(self.config.accepted_caption_statuses)!r}. Stage 1A only "
                    "trains on captions reviewed against the exact AION crop. Pass "
                    "allow_unsafe_candidates=True only for inspection/testing."
                )
            self._caption_text(record)

    def __len__(self) -> int:
        return len(self._indices)

    def _caption_text(self, record: dict[str, Any]) -> tuple[str, str]:
        caption = record.get("caption")
        if not isinstance(caption, dict):
            raise AstroPRISMStage1ADatasetError(
                f"candidate {record.get('candidate_index')} has no caption object"
            )
        requested_field = self.config.caption_field.strip()
        caption_status = str(caption.get("status", "")).strip().casefold()
        expected_field = _CAPTION_FIELD_BY_RELEASE_STATUS.get(caption_status)

        if requested_field.casefold() == "auto":
            if expected_field is not None:
                selected_field = expected_field
            elif self.config.allow_unsafe_candidates:
                selected_field = "original"
            else:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {record.get('candidate_index')} has caption.status="
                    f"{caption.get('status')!r}, which has no safe automatic caption mapping"
                )
        else:
            selected_field = requested_field
            if expected_field is not None and selected_field != expected_field:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {record.get('candidate_index')} caption.status="
                    f"{caption.get('status')!r} requires caption.{expected_field}, not "
                    f"caption.{selected_field}; use caption_field='auto'"
                )
            if expected_field is None and not self.config.allow_unsafe_candidates:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {record.get('candidate_index')} has caption.status="
                    f"{caption.get('status')!r}, which has no safe caption mapping; "
                    "explicit fields are permitted only with "
                    "allow_unsafe_candidates=True for inspection/testing"
                )

        value = caption.get(selected_field)
        if not isinstance(value, str) or not value.strip():
            raise AstroPRISMStage1ADatasetError(
                f"candidate {record.get('candidate_index')} has no non-empty "
                f"caption.{selected_field} selected from caption.status="
                f"{caption.get('status')!r}"
            )
        return value.strip(), selected_field

    def _tokenize_chat(self, target: str) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": self.config.user_prompt},
        ]
        full_messages = [*prefix_messages, {"role": "assistant", "content": target}]

        common = {"tokenize": True, "enable_thinking": False}
        try:
            prompt_value = self.tokenizer.apply_chat_template(
                prefix_messages,
                add_generation_prompt=True,
                **common,
            )
            full_value = self.tokenizer.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                **common,
            )
        except (TypeError, ValueError) as exc:
            raise AstroPRISMStage1ADatasetError(
                "tokenizer failed Qwen non-thinking chat formatting "
                "(apply_chat_template(..., enable_thinking=False))"
            ) from exc

        prompt_ids = _coerce_token_ids(prompt_value, context="chat prompt")
        full_ids = _coerce_token_ids(full_value, context="full chat")
        if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
            raise AstroPRISMStage1ADatasetError(
                "Qwen full chat is not prefixed by the generation prompt; cannot construct "
                "assistant-only labels safely"
            )

        merged_length = ASTROPRISM_IMAGE_TOKENS + len(full_ids)
        if merged_length > self.config.max_merged_seq_length:
            raise AstroPRISMStage1ADatasetError(
                f"merged sequence length {merged_length} exceeds "
                f"max_merged_seq_length={self.config.max_merged_seq_length} "
                f"({ASTROPRISM_IMAGE_TOKENS} image tokens + {len(full_ids)} text tokens); "
                "the dataset never truncates captions silently"
            )

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = IGNORE_INDEX
        if not torch.any(labels != IGNORE_INDEX):
            raise AstroPRISMStage1ADatasetError("assistant response produced no supervised tokens")
        return input_ids, labels

    def __getitem__(self, item: int) -> dict[str, Any]:
        source_index = self._indices[item]
        record = self._all_records[source_index]
        raw_record = self.raw_manifest["candidates"][source_index]

        # np.array(copy=True) avoids returning a writable torch view over a
        # read-only memmap and isolates AION's in-place preprocessing.
        image_array = np.array(self._images[source_index], dtype=np.float32, order="C", copy=True)
        if not np.isfinite(image_array).all():
            raise AstroPRISMStage1ADatasetError(
                f"candidate {source_index} raw image contains non-finite values"
            )
        image = torch.from_numpy(image_array)

        caption_text, selected_caption_field = self._caption_text(record)
        input_ids, labels = self._tokenize_chat(caption_text)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        provenance = {
            "candidate_index": source_index,
            "paired_targetid": str(record["paired_targetid"]),
            "caption_object_id": str(record["caption_object_id"]),
            "source_group_id": record.get("source_group_id"),
            "split": record.get("split"),
            "acceptance_status": record.get("acceptance_status"),
            "raw_row": raw_record.get("raw_row"),
            "bands": raw_record.get("bands"),
            "coordinates_deg": raw_record.get("coordinates_deg"),
            "raw_source": self.raw_manifest.get("source"),
            "caption_source": record.get("caption_source"),
            "paired_source": record.get("paired_source"),
            "roster_sha256": self.candidates_manifest.get("pins", {}).get("roster_sha256"),
            "caption_field": selected_caption_field,
            "caption_field_config": self.config.caption_field,
            "caption_status": record.get("caption", {}).get("status"),
            "chat_template": "qwen_non_thinking",
        }

        return {
            "image": image,
            "text": input_ids,
            "text_attention_mask": attention_mask,
            "labels": labels,
            "image_token_count": ASTROPRISM_IMAGE_TOKENS,
            "merged_length": ASTROPRISM_IMAGE_TOKENS + input_ids.numel(),
            "provenance": provenance,
        }


__all__ = [
    "ASTROPRISM_AION_BANDS",
    "ASTROPRISM_IMAGE_SHAPE",
    "ASTROPRISM_IMAGE_TOKENS",
    "ASTROPRISM_SOURCE_BANDS",
    "IGNORE_INDEX",
    "AstroPRISMStage1ADataset",
    "AstroPRISMStage1ADatasetConfig",
    "AstroPRISMStage1ADatasetError",
]

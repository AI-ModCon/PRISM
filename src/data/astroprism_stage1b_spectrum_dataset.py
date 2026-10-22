"""Spectrum-only AstroPRISM alignment dataset.

Stage 1B pairs raw observed-frame DESI spectra with the promoted, reviewed
spectrum-caption artifact. Source identities and train/validation/test splits
remain tied to the immutable 646-object overlap roster. The loader fails
closed unless the selected records carry the accepted review decision and the
split-specific training/evaluation policy produced by caption promotion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .astroprism_spectrum_contract import (
    ASTROPRISM_SPECTRUM_LENGTH,
    ASTROPRISM_SPECTRUM_TOKENS,
)
from .astroprism_stage1a_dataset import (
    IGNORE_INDEX,
    AstroPRISMStage1ADataset,
    AstroPRISMStage1ADatasetError,
    _coerce_token_ids,
    _read_json,
    _read_jsonl,
    _resolve_output_path,
    _sha256_file,
    _validate_source_split_integrity,
)

DEFAULT_SPECTRUM_SYSTEM_PROMPT = (
    "You are an astronomy assistant. Give a concise, objective description grounded "
    "only in the supplied observed-frame DESI spectrum. Do not infer image morphology "
    "or unsupported physical causes."
)
DEFAULT_SPECTRUM_USER_PROMPT = "Describe the supplied observed-frame DESI galaxy spectrum."

_ROLE_BY_SPLIT = {
    "train": "alignment_training",
    "validation": "alignment_validation",
    "test": "alignment_test",
}


@dataclass(frozen=True)
class AstroPRISMStage1BSpectrumDatasetConfig:
    raw_manifest_path: Path
    captions_manifest_path: Path
    captions_path: Path | None = None
    split: str | None = "train"
    max_merged_seq_length: int = 2048
    system_prompt: str = DEFAULT_SPECTRUM_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_SPECTRUM_USER_PROMPT
    accepted_caption_statuses: tuple[str, ...] = ("accepted",)
    verify_file_hashes: bool = False


class AstroPRISMStage1BSpectrumDataset(AstroPRISMStage1ADataset):
    """Mmap-backed raw-spectrum dataset with assistant-only caption labels."""

    def __init__(
        self,
        config: AstroPRISMStage1BSpectrumDatasetConfig,
        tokenizer: Any,
    ) -> None:
        # Do not call the image dataset initializer; only its Dataset base and
        # shared exception/helper contract are reused.
        self.config = config
        self.active_modalities = {"spectrum"}
        self.tokenizer = tokenizer

        if config.max_merged_seq_length <= ASTROPRISM_SPECTRUM_TOKENS:
            raise AstroPRISMStage1ADatasetError(
                "max_merged_seq_length must exceed the fixed 273-token AION spectrum prefix"
            )
        if not hasattr(tokenizer, "apply_chat_template"):
            raise AstroPRISMStage1ADatasetError(
                "tokenizer must provide apply_chat_template for Qwen chat formatting"
            )

        self.raw_manifest_path = Path(config.raw_manifest_path)
        self.captions_manifest_path = Path(config.captions_manifest_path)
        self.raw_manifest = _read_json(
            self.raw_manifest_path,
            kind="raw-spectrum manifest",
        )
        self.captions_manifest = _read_json(
            self.captions_manifest_path,
            kind="reviewed spectrum-caption manifest",
        )
        self._validate_manifest_types()

        captions_output = self.captions_manifest.get("outputs", {}).get("captions.jsonl")
        if not isinstance(captions_output, dict):
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest is missing outputs['captions.jsonl']"
            )
        self.captions_path = (
            Path(config.captions_path)
            if config.captions_path is not None
            else self.captions_manifest_path.parent / "captions.jsonl"
        )
        self._all_records = _read_jsonl(self.captions_path)
        self._validate_caption_file(captions_output)

        self._spectra: dict[str, np.ndarray] = {}
        for key, dtype in (
            ("spectrum_flux", np.dtype("float32")),
            ("spectrum_ivar", np.dtype("float32")),
            ("spectrum_mask", np.dtype("bool")),
            ("spectrum_wavelength", np.dtype("float32")),
        ):
            output = self.raw_manifest.get("outputs", {}).get(key)
            if not isinstance(output, dict):
                raise AstroPRISMStage1ADatasetError(
                    f"raw-spectrum manifest is missing outputs.{key}"
                )
            self._spectra[key] = self._open_and_validate_array(key, output, dtype)

        self._validate_roster_identity()
        self._validate_caption_alignment()
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
                f"no spectrum captions found for requested split {config.split!r}"
            )
        self._validate_training_safety()

    def _validate_manifest_types(self) -> None:
        if self.raw_manifest.get("artifact_type") != "astroprism_raw_spectra":
            raise AstroPRISMStage1ADatasetError(
                "raw manifest artifact_type must be 'astroprism_raw_spectra'"
            )
        spectrum_contract = self.raw_manifest.get("spectrum")
        if not isinstance(spectrum_contract, dict):
            raise AstroPRISMStage1ADatasetError(
                "raw-spectrum manifest is missing the spectrum contract"
            )
        expected_contract = {
            "length": ASTROPRISM_SPECTRUM_LENGTH,
            "aion_token_count": ASTROPRISM_SPECTRUM_TOKENS,
            "observed_frame": True,
            "mask_true_means_excluded": True,
            "aion_modality": "DESISpectrum",
        }
        for key, expected in expected_contract.items():
            actual = spectrum_contract.get(key)
            if actual != expected or type(actual) is not type(expected):
                raise AstroPRISMStage1ADatasetError(
                    "raw-spectrum manifest contract mismatch: "
                    f"spectrum.{key} must be {expected!r}, got {actual!r}"
                )

        caption_type = self.captions_manifest.get("artifact_type")
        if caption_type != "astroprism_spectrum_captions_accepted":
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest artifact_type must be "
                "'astroprism_spectrum_captions_accepted'"
            )
        if self.captions_manifest.get("role") != "reviewed_alignment_dataset":
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest role must be 'reviewed_alignment_dataset'"
            )
        review = self.captions_manifest.get("review")
        if not isinstance(review, dict) or review.get("status") != "accepted":
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest review.status must be 'accepted'"
            )
        if review.get("decision_source") != "user_instruction":
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest review.decision_source must be " "'user_instruction'"
            )
        policy = self.captions_manifest.get("training_policy")
        if not isinstance(policy, dict) or policy.get("dataset_ready") is not True:
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest training_policy.dataset_ready must be true"
            )
        if policy.get("gradient_training_splits") != ["train"]:
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest must restrict gradient training to ['train']"
            )
        evaluation_splits = policy.get("evaluation_splits")
        if not isinstance(evaluation_splits, list) or set(evaluation_splits) != {
            "validation",
            "test",
        }:
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest evaluation splits must be validation and test"
            )
        if policy.get("roles") != _ROLE_BY_SPLIT:
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest split roles do not match the Stage 1B contract"
            )

    def _validate_caption_file(self, output: dict[str, Any]) -> None:
        if not self._all_records:
            raise AstroPRISMStage1ADatasetError("spectrum-caption JSONL is empty")
        declared_count = self.captions_manifest.get("row_count")
        if declared_count != len(self._all_records):
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption row count mismatch: manifest declares "
                f"{declared_count!r}, JSONL contains {len(self._all_records)}"
            )
        declared_indices = self.captions_manifest.get("candidate_indices")
        observed_indices = [record.get("candidate_index") for record in self._all_records]
        if declared_indices != observed_indices:
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption manifest candidate_indices do not match captions.jsonl"
            )
        observed_splits = dict(Counter(record.get("split") for record in self._all_records))
        declared_splits = self.captions_manifest.get("split_counts")
        nested_splits = self.captions_manifest.get("splits", {}).get("row_counts")
        if declared_splits != observed_splits or nested_splits != observed_splits:
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption split counts do not match captions.jsonl"
            )
        declared_size = output.get("size_bytes")
        if declared_size is not None and declared_size != self.captions_path.stat().st_size:
            raise AstroPRISMStage1ADatasetError(
                "spectrum-caption JSONL size disagrees with its manifest"
            )
        if self.config.verify_file_hashes and output.get("sha256"):
            actual = _sha256_file(self.captions_path)
            if actual != output["sha256"]:
                raise AstroPRISMStage1ADatasetError(
                    "spectrum-caption JSONL SHA-256 mismatch: expected "
                    f"{output['sha256']}, got {actual}"
                )

    def _open_and_validate_array(
        self,
        key: str,
        output: dict[str, Any],
        expected_dtype: np.dtype[Any],
    ) -> np.ndarray:
        path = _resolve_output_path(self.raw_manifest_path, output, name=key)
        if not path.is_file():
            raise AstroPRISMStage1ADatasetError(f"raw spectrum array does not exist: {path}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise AstroPRISMStage1ADatasetError(
                f"cannot mmap raw spectrum array {path}: {exc}"
            ) from exc

        expected_shape = (len(self._all_records), ASTROPRISM_SPECTRUM_LENGTH)
        if array.shape != expected_shape:
            raise AstroPRISMStage1ADatasetError(
                f"{key} must have shape {expected_shape}, got {array.shape}"
            )
        if array.dtype != expected_dtype:
            raise AstroPRISMStage1ADatasetError(
                f"{key} must have dtype {expected_dtype}, got {array.dtype}"
            )
        if output.get("shape") is not None and list(array.shape) != output["shape"]:
            raise AstroPRISMStage1ADatasetError(
                f"{key} shape disagrees with manifest: {array.shape} != {output['shape']}"
            )
        if output.get("dtype") is not None and np.dtype(output["dtype"]) != array.dtype:
            raise AstroPRISMStage1ADatasetError(
                f"{key} dtype disagrees with manifest: {array.dtype} != {output['dtype']}"
            )
        if self.config.verify_file_hashes and output.get("sha256"):
            actual = _sha256_file(path)
            if actual != output["sha256"]:
                raise AstroPRISMStage1ADatasetError(
                    f"{key} SHA-256 mismatch: expected {output['sha256']}, got {actual}"
                )
        return array

    def _validate_roster_identity(self) -> None:
        raw_roster = self.raw_manifest.get("roster")
        caption_roster = self.captions_manifest.get("roster")
        if not isinstance(raw_roster, dict) or not isinstance(caption_roster, dict):
            raise AstroPRISMStage1ADatasetError(
                "raw and spectrum-caption manifests must both declare roster provenance"
            )
        raw_sha = raw_roster.get("sha256")
        caption_sha = caption_roster.get("sha256")
        if not isinstance(raw_sha, str) or not raw_sha:
            raise AstroPRISMStage1ADatasetError("raw-spectrum roster SHA-256 is missing")
        if not isinstance(caption_sha, str) or not caption_sha:
            raise AstroPRISMStage1ADatasetError("spectrum-caption roster SHA-256 is missing")
        if raw_sha != caption_sha:
            raise AstroPRISMStage1ADatasetError(
                f"raw/spectrum-caption roster SHA-256 mismatch: {raw_sha} != {caption_sha}"
            )
        expected_count = len(self._all_records)
        for label, roster in (("raw", raw_roster), ("spectrum-caption", caption_roster)):
            if roster.get("row_count") != expected_count:
                raise AstroPRISMStage1ADatasetError(
                    f"{label} roster row_count must be {expected_count}, "
                    f"got {roster.get('row_count')!r}"
                )

    def _validate_caption_alignment(self) -> None:
        raw_candidates = self.raw_manifest.get("candidates")
        if not isinstance(raw_candidates, list):
            raise AstroPRISMStage1ADatasetError("raw-spectrum manifest is missing candidates list")
        if len(raw_candidates) != len(self._all_records):
            raise AstroPRISMStage1ADatasetError(
                f"raw/spectrum-caption row count mismatch: {len(raw_candidates)} != "
                f"{len(self._all_records)}"
            )
        paired_ids: set[str] = set()
        caption_ids: set[str] = set()
        for index, (raw_record, caption_record) in enumerate(
            zip(raw_candidates, self._all_records, strict=True)
        ):
            for kind, record in (("raw", raw_record), ("caption", caption_record)):
                if not isinstance(record, dict) or record.get("candidate_index") != index:
                    raise AstroPRISMStage1ADatasetError(
                        f"{kind} candidate row {index} does not preserve roster order"
                    )
            for field in ("paired_targetid", "caption_object_id"):
                raw_value = raw_record.get(field)
                caption_value = caption_record.get(field)
                if not isinstance(caption_value, str) or not caption_value.strip():
                    raise AstroPRISMStage1ADatasetError(
                        f"spectrum-caption candidate {index} has no {field}"
                    )
                if str(raw_value) != caption_value:
                    raise AstroPRISMStage1ADatasetError(
                        f"candidate {index} {field} mismatch between raw and spectrum-caption "
                        f"artifacts: {raw_value!r} != {caption_value!r}"
                    )
            paired_id = caption_record["paired_targetid"]
            caption_id = caption_record["caption_object_id"]
            if paired_id in paired_ids or caption_id in caption_ids:
                raise AstroPRISMStage1ADatasetError(
                    f"spectrum-caption candidate {index} repeats an object identity"
                )
            paired_ids.add(paired_id)
            caption_ids.add(caption_id)
            source_group = caption_record.get("source_group_id")
            if not isinstance(source_group, str) or not source_group.strip():
                raise AstroPRISMStage1ADatasetError(
                    f"spectrum-caption candidate {index} has no source_group_id"
                )
            split = caption_record.get("split")
            if split not in _ROLE_BY_SPLIT:
                raise AstroPRISMStage1ADatasetError(
                    f"spectrum-caption candidate {index} has unsupported split {split!r}"
                )

    def _validate_training_safety(self) -> None:
        accepted_statuses = {
            status.strip().casefold()
            for status in self.config.accepted_caption_statuses
            if status.strip()
        }
        if not accepted_statuses:
            raise AstroPRISMStage1ADatasetError("accepted_caption_statuses must be non-empty")
        for index in self._indices:
            record = self._all_records[index]
            caption = record.get("spectrum_caption")
            status = str(caption.get("status", "")).strip() if isinstance(caption, dict) else ""
            if status.casefold() not in accepted_statuses:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} has spectrum_caption.status={status!r}; expected "
                    f"one of {sorted(self.config.accepted_caption_statuses)!r}"
                )
            review = record.get("review")
            if not isinstance(review, dict) or review.get("status") != "accepted":
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} review.status must be 'accepted'"
                )
            if review.get("decision_source") != "user_instruction":
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} review.decision_source must be 'user_instruction'"
                )
            policy = record.get("training_policy")
            if not isinstance(policy, dict) or policy.get("dataset_ready") is not True:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} training_policy.dataset_ready must be true"
                )
            split = record["split"]
            is_train = split == "train"
            if policy.get("gradient_training_allowed") is not is_train:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} gradient_training_allowed must be {is_train} "
                    f"for split {split!r}"
                )
            if policy.get("eligible_for_training") is not is_train:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} eligible_for_training must be {is_train} "
                    f"for split {split!r}"
                )
            if policy.get("eligible_for_evaluation") is not (not is_train):
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} eligible_for_evaluation must be {not is_train} "
                    f"for split {split!r}"
                )
            expected_role = _ROLE_BY_SPLIT[split]
            if policy.get("role") != expected_role:
                raise AstroPRISMStage1ADatasetError(
                    f"candidate {index} training_policy.role must be {expected_role!r}"
                )
            self._caption_text(record)

    def _caption_text(self, record: dict[str, Any]) -> tuple[str, str]:
        caption = record.get("spectrum_caption")
        if not isinstance(caption, dict):
            raise AstroPRISMStage1ADatasetError(
                f"candidate {record.get('candidate_index')} has no spectrum_caption object"
            )
        value = caption.get("original")
        if not isinstance(value, str) or not value.strip():
            raise AstroPRISMStage1ADatasetError(
                f"candidate {record.get('candidate_index')} has no non-empty "
                "spectrum_caption.original"
            )
        return value.strip(), "original"

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
                "tokenizer failed Qwen non-thinking chat formatting"
            ) from exc
        prompt_ids = _coerce_token_ids(prompt_value, context="chat prompt")
        full_ids = _coerce_token_ids(full_value, context="full chat")
        if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
            raise AstroPRISMStage1ADatasetError(
                "Qwen full chat is not prefixed by the generation prompt"
            )
        merged_length = ASTROPRISM_SPECTRUM_TOKENS + len(full_ids)
        if merged_length > self.config.max_merged_seq_length:
            raise AstroPRISMStage1ADatasetError(
                f"merged sequence length {merged_length} exceeds "
                f"max_merged_seq_length={self.config.max_merged_seq_length} "
                f"({ASTROPRISM_SPECTRUM_TOKENS} spectrum tokens + "
                f"{len(full_ids)} text tokens); refusing silent truncation"
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

        spectrum = {
            "flux": torch.from_numpy(
                np.array(
                    self._spectra["spectrum_flux"][source_index],
                    dtype=np.float32,
                    order="C",
                    copy=True,
                )
            ),
            "ivar": torch.from_numpy(
                np.array(
                    self._spectra["spectrum_ivar"][source_index],
                    dtype=np.float32,
                    order="C",
                    copy=True,
                )
            ),
            "mask": torch.from_numpy(
                np.array(
                    self._spectra["spectrum_mask"][source_index],
                    dtype=np.bool_,
                    order="C",
                    copy=True,
                )
            ),
            "wavelength": torch.from_numpy(
                np.array(
                    self._spectra["spectrum_wavelength"][source_index],
                    dtype=np.float32,
                    order="C",
                    copy=True,
                )
            ),
        }
        if not all(
            bool(torch.isfinite(spectrum[key]).all()) for key in ("flux", "ivar", "wavelength")
        ):
            raise AstroPRISMStage1ADatasetError(
                f"candidate {source_index} raw spectrum contains non-finite values"
            )
        if bool(torch.any(spectrum["ivar"] < 0)):
            raise AstroPRISMStage1ADatasetError(
                f"candidate {source_index} raw spectrum contains negative inverse variance"
            )
        if not bool(torch.all(torch.diff(spectrum["wavelength"]) > 0)):
            raise AstroPRISMStage1ADatasetError(
                f"candidate {source_index} wavelength is not strictly increasing"
            )
        if bool(torch.all(spectrum["mask"])):
            raise AstroPRISMStage1ADatasetError(
                f"candidate {source_index} spectrum has no usable samples"
            )

        caption_text, selected_caption_field = self._caption_text(record)
        input_ids, labels = self._tokenize_chat(caption_text)
        provenance = {
            "candidate_index": record["candidate_index"],
            "paired_targetid": record["paired_targetid"],
            "caption_object_id": record["caption_object_id"],
            "source_group_id": record["source_group_id"],
            "split": record["split"],
            "raw_row": raw_record.get("raw_row"),
            "coordinates_deg": raw_record.get("coordinates_deg"),
            "raw_source": self.raw_manifest.get("source"),
            "roster_sha256": self.captions_manifest["roster"]["sha256"],
            "caption_field": selected_caption_field,
            "caption_status": record["spectrum_caption"]["status"],
            "caption_review_status": record["review"]["status"],
            "caption_review_decision_source": record["review"]["decision_source"],
            "caption_training_role": record["training_policy"]["role"],
            "caption_supervision": "reviewed_spectrum_caption",
            "spectral_grounding_claimed": True,
            "chat_template": "qwen_non_thinking",
        }
        return {
            "spectrum": spectrum,
            "text": input_ids,
            "text_attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            "labels": labels,
            "spectrum_token_count": ASTROPRISM_SPECTRUM_TOKENS,
            "merged_length": ASTROPRISM_SPECTRUM_TOKENS + input_ids.numel(),
            "provenance": provenance,
        }


__all__ = [
    "ASTROPRISM_SPECTRUM_LENGTH",
    "ASTROPRISM_SPECTRUM_TOKENS",
    "DEFAULT_SPECTRUM_SYSTEM_PROMPT",
    "DEFAULT_SPECTRUM_USER_PROMPT",
    "AstroPRISMStage1BSpectrumDataset",
    "AstroPRISMStage1BSpectrumDatasetConfig",
]

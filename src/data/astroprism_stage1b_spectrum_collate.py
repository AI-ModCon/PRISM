"""Lockstep collation for AstroPRISM raw-spectrum/chat samples."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .astroprism_spectrum_contract import ASTROPRISM_SPECTRUM_LENGTH
from .astroprism_stage1a_dataset import IGNORE_INDEX
from .astroprism_stage1b_spectrum_dataset import ASTROPRISM_SPECTRUM_TOKENS


class AstroPRISMStage1BSpectrumCollator:
    """Stack four raw DESI fields and pad assistant-only chat labels."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_merged_seq_length: int,
        spectrum_token_count: int = ASTROPRISM_SPECTRUM_TOKENS,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("AstroPRISM spectrum alignment requires tokenizer.pad_token_id")
        if spectrum_token_count != ASTROPRISM_SPECTRUM_TOKENS:
            raise ValueError(
                "AstroPRISM spectrum alignment requires exactly "
                f"{ASTROPRISM_SPECTRUM_TOKENS} AION tokens, got {spectrum_token_count}"
            )
        if max_merged_seq_length <= spectrum_token_count:
            raise ValueError(
                "max_merged_seq_length must exceed the 273-token AION spectrum prefix"
            )
        self.pad_token_id = int(pad_token_id)
        self.max_merged_seq_length = int(max_merged_seq_length)
        self.spectrum_token_count = int(spectrum_token_count)
        self.ignore_index = int(ignore_index)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            return {}
        required = {
            "spectrum",
            "text",
            "text_attention_mask",
            "labels",
            "spectrum_token_count",
            "merged_length",
            "provenance",
        }
        spectrum_fields = {
            "flux": torch.float32,
            "ivar": torch.float32,
            "mask": torch.bool,
            "wavelength": torch.float32,
        }
        for index, sample in enumerate(batch):
            missing = required.difference(sample)
            if missing:
                raise ValueError(
                    f"spectrum alignment sample {index} is missing keys: {sorted(missing)}"
                )
            text = sample["text"]
            labels = sample["labels"]
            attention = sample["text_attention_mask"]
            if not all(isinstance(value, torch.Tensor) for value in (text, labels, attention)):
                raise TypeError(
                    f"spectrum alignment sample {index} text/labels/mask must be tensors"
                )
            if text.ndim != 1 or labels.shape != text.shape or attention.shape != text.shape:
                raise ValueError(
                    f"spectrum alignment sample {index} text, labels, and attention mask "
                    "must be aligned 1-D tensors"
                )
            if int(sample["spectrum_token_count"]) != self.spectrum_token_count:
                raise ValueError(
                    f"spectrum alignment sample {index} has spectrum_token_count="
                    f"{sample['spectrum_token_count']}; expected {self.spectrum_token_count}"
                )
            merged_length = self.spectrum_token_count + int(text.numel())
            if int(sample["merged_length"]) != merged_length:
                raise ValueError(
                    f"spectrum alignment sample {index} has inconsistent merged_length="
                    f"{sample['merged_length']}; expected {merged_length}"
                )
            if merged_length > self.max_merged_seq_length:
                raise ValueError(
                    f"spectrum alignment sample {index} merged length {merged_length} "
                    f"exceeds max_merged_seq_length={self.max_merged_seq_length}"
                )
            spectrum = sample["spectrum"]
            if not isinstance(spectrum, dict) or set(spectrum) != set(spectrum_fields):
                raise ValueError(
                    f"spectrum alignment sample {index} must contain exactly "
                    f"{sorted(spectrum_fields)}"
                )
            for field, dtype in spectrum_fields.items():
                value = spectrum[field]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"sample {index} spectrum.{field} must be a tensor")
                if value.shape != (ASTROPRISM_SPECTRUM_LENGTH,):
                    raise ValueError(
                        f"sample {index} spectrum.{field} must have shape "
                        f"({ASTROPRISM_SPECTRUM_LENGTH},), got {tuple(value.shape)}"
                    )
                if value.dtype != dtype:
                    raise ValueError(
                        f"sample {index} spectrum.{field} must have dtype {dtype}, "
                        f"got {value.dtype}"
                    )

        return {
            "spectrum": {
                field: torch.stack([sample["spectrum"][field] for sample in batch])
                for field in spectrum_fields
            },
            "text": pad_sequence(
                [sample["text"] for sample in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "text_attention_mask": pad_sequence(
                [sample["text_attention_mask"] for sample in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [sample["labels"] for sample in batch],
                batch_first=True,
                padding_value=self.ignore_index,
            ),
            "spectrum_token_count": self.spectrum_token_count,
            "merged_length": torch.tensor(
                [int(sample["merged_length"]) for sample in batch],
                dtype=torch.long,
            ),
            "provenance": [sample["provenance"] for sample in batch],
        }


__all__ = ["AstroPRISMStage1BSpectrumCollator"]

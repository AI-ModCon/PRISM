"""Lockstep collation for AstroPRISM Stage 1A image/chat samples."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .astroprism_stage1a_dataset import (
    ASTROPRISM_IMAGE_SHAPE,
    ASTROPRISM_IMAGE_TOKENS,
    IGNORE_INDEX,
)


class AstroPRISMStage1ACollator:
    """Pad Stage 1A chat samples without weakening their masking contract.

    The generic multimodal collators use one padding value for every 1-D
    tensor. Stage 1A requires three distinct values: the tokenizer pad ID for
    ``text``, zero for ``text_attention_mask``, and ``-100`` for ``labels``.
    This dedicated collator also rechecks the 576-token image-prefix budget and
    never truncates an oversized example.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_merged_seq_length: int,
        image_token_count: int = ASTROPRISM_IMAGE_TOKENS,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("AstroPRISM Stage 1A requires tokenizer.pad_token_id")
        if image_token_count != ASTROPRISM_IMAGE_TOKENS:
            raise ValueError(
                "AstroPRISM Stage 1A requires exactly "
                f"{ASTROPRISM_IMAGE_TOKENS} image tokens, got {image_token_count}"
            )
        if max_merged_seq_length <= image_token_count:
            raise ValueError("max_merged_seq_length must exceed the 576-token AION image prefix")
        self.pad_token_id = int(pad_token_id)
        self.max_merged_seq_length = int(max_merged_seq_length)
        self.image_token_count = int(image_token_count)
        self.ignore_index = int(ignore_index)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            return {}
        required = {
            "image",
            "text",
            "text_attention_mask",
            "labels",
            "image_token_count",
            "merged_length",
            "provenance",
        }
        for index, sample in enumerate(batch):
            missing = required.difference(sample)
            if missing:
                raise ValueError(f"Stage 1A sample {index} is missing keys: {sorted(missing)}")
            text = sample["text"]
            labels = sample["labels"]
            attention = sample["text_attention_mask"]
            if not all(isinstance(value, torch.Tensor) for value in (text, labels, attention)):
                raise TypeError(f"Stage 1A sample {index} text/labels/mask must be tensors")
            if text.ndim != 1 or labels.shape != text.shape or attention.shape != text.shape:
                raise ValueError(
                    f"Stage 1A sample {index} text, labels, and attention mask must be "
                    "aligned 1-D tensors"
                )
            if int(sample["image_token_count"]) != self.image_token_count:
                raise ValueError(
                    f"Stage 1A sample {index} has image_token_count="
                    f"{sample['image_token_count']}; expected {self.image_token_count}"
                )
            merged_length = self.image_token_count + int(text.numel())
            if int(sample["merged_length"]) != merged_length:
                raise ValueError(
                    f"Stage 1A sample {index} has inconsistent merged_length="
                    f"{sample['merged_length']}; expected {merged_length}"
                )
            if merged_length > self.max_merged_seq_length:
                raise ValueError(
                    f"Stage 1A sample {index} merged sequence length {merged_length} exceeds "
                    f"max_merged_seq_length={self.max_merged_seq_length}; refusing to truncate"
                )
            image = sample["image"]
            if not isinstance(image, torch.Tensor) or image.shape != ASTROPRISM_IMAGE_SHAPE:
                raise ValueError(
                    f"Stage 1A sample {index} image must be {ASTROPRISM_IMAGE_SHAPE}, got "
                    f"{getattr(image, 'shape', None)}"
                )
            if image.dtype != torch.float32:
                raise ValueError(
                    f"Stage 1A sample {index} image must be float32, got {image.dtype}"
                )

        return {
            "image": torch.stack([sample["image"] for sample in batch]),
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
            "image_token_count": self.image_token_count,
            "merged_length": torch.tensor(
                [int(sample["merged_length"]) for sample in batch], dtype=torch.long
            ),
            "provenance": [sample["provenance"] for sample in batch],
        }


__all__ = ["AstroPRISMStage1ACollator"]

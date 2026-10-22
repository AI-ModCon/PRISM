"""Modality-routing collator for joint AstroPRISM alignment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .astroprism_joint_dataset import ASTROPRISM_JOINT_MODALITIES
from .astroprism_stage1a_collate import AstroPRISMStage1ACollator
from .astroprism_stage1b_spectrum_collate import AstroPRISMStage1BSpectrumCollator


class AstroPRISMJointCollator:
    """Delegate a homogeneous batch to its existing modality collator.

    The explicit ``batch_modality`` tag is required on every input sample.  A
    mixed batch fails before either source collator or the model sees it.
    """

    active_modalities = {"image", "spectrum"}

    def __init__(
        self,
        image_collator: Callable[[list[dict[str, Any]]], dict[str, Any]],
        spectrum_collator: Callable[[list[dict[str, Any]]], dict[str, Any]],
    ) -> None:
        self._collators = {
            "image": image_collator,
            "spectrum": spectrum_collator,
        }

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        *,
        max_merged_seq_length: int,
    ) -> AstroPRISMJointCollator:
        return cls(
            AstroPRISMStage1ACollator(
                tokenizer,
                max_merged_seq_length=max_merged_seq_length,
            ),
            AstroPRISMStage1BSpectrumCollator(
                tokenizer,
                max_merged_seq_length=max_merged_seq_length,
            ),
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            return {}

        modalities: list[str] = []
        payload: list[dict[str, Any]] = []
        for index, sample in enumerate(batch):
            if not isinstance(sample, dict):
                raise TypeError(
                    f"joint sample {index} must be a dict, got {type(sample).__name__}"
                )
            modality = sample.get("batch_modality")
            if modality not in ASTROPRISM_JOINT_MODALITIES:
                raise ValueError(
                    f"joint sample {index} has invalid batch_modality={modality!r}; "
                    f"expected one of {list(ASTROPRISM_JOINT_MODALITIES)}"
                )
            if modality == "image" and "spectrum" in sample:
                raise ValueError(f"joint image sample {index} unexpectedly contains spectrum")
            if modality == "spectrum" and "image" in sample:
                raise ValueError(f"joint spectrum sample {index} unexpectedly contains image")
            modalities.append(modality)
            payload.append({key: value for key, value in sample.items() if key != "batch_modality"})

        unique_modalities = set(modalities)
        if len(unique_modalities) != 1:
            raise ValueError(
                "AstroPRISM joint batches must be modality-homogeneous; got "
                f"{modalities!r}"
            )
        modality = modalities[0]
        collated = self._collators[modality](payload)
        if not isinstance(collated, dict):
            raise TypeError(
                f"{modality} collator returned {type(collated).__name__}; expected a dict"
            )
        if "batch_modality" in collated and collated["batch_modality"] != modality:
            raise ValueError(
                f"{modality} collator returned conflicting "
                f"batch_modality={collated['batch_modality']!r}"
            )
        output = dict(collated)
        output["batch_modality"] = modality
        return output


__all__ = ["AstroPRISMJointCollator"]

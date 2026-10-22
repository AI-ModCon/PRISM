"""Map-style union of the reviewed AstroPRISM image and spectrum datasets.

The two source datasets retain their own validation and preprocessing
contracts.  This wrapper only provides a stable joint index space and tags
each sample with the modality that its batch must contain.  Homogeneous
batches are enforced by :mod:`astroprism_joint_batch_sampler` and rechecked by
:mod:`astroprism_joint_collate`.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from typing import Any

from torch.utils.data import Dataset

from .astroprism_stage1a_dataset import (
    AstroPRISMStage1ADataset,
    AstroPRISMStage1ADatasetConfig,
)
from .astroprism_stage1b_spectrum_dataset import (
    AstroPRISMStage1BSpectrumDataset,
    AstroPRISMStage1BSpectrumDatasetConfig,
)

ASTROPRISM_JOINT_MODALITIES = ("image", "spectrum")
ASTROPRISM_JOINT_SPLITS = frozenset({"train", "validation", "test"})


class AstroPRISMJointDatasetError(ValueError):
    """Raised when the two source datasets cannot form one joint view."""


@dataclass(frozen=True)
class AstroPRISMJointDatasetConfig:
    """Child dataset configs plus the single split exposed by the joint view.

    ``split`` is authoritative. :meth:`AstroPRISMJointDataset.from_config`
    copies it into both child configs, preventing a caller from accidentally
    pairing train examples from one modality with validation examples from the
    other.
    """

    image: AstroPRISMStage1ADatasetConfig
    spectrum: AstroPRISMStage1BSpectrumDatasetConfig
    split: str = "train"


class AstroPRISMJointDataset(Dataset):
    """Joint index space over image-caption and spectrum-caption examples.

    Image examples occupy ``[0, len(image_dataset))`` and spectrum examples
    follow them.  A shallow copy of each source sample is returned with a
    ``batch_modality`` tag; source dataset samples are never mutated.
    """

    active_modalities = {"image", "spectrum"}

    def __init__(
        self,
        image_dataset: Dataset,
        spectrum_dataset: Dataset,
        *,
        split: str,
    ) -> None:
        split = str(split).strip()
        if split not in ASTROPRISM_JOINT_SPLITS:
            raise AstroPRISMJointDatasetError(
                f"joint split must be one of {sorted(ASTROPRISM_JOINT_SPLITS)}, got {split!r}"
            )

        self.split = split
        self.image_dataset = image_dataset
        self.spectrum_dataset = spectrum_dataset
        self._validate_child("image", image_dataset)
        self._validate_child("spectrum", spectrum_dataset)

        image_count = len(image_dataset)
        spectrum_count = len(spectrum_dataset)
        if image_count <= 0 or spectrum_count <= 0:
            raise AstroPRISMJointDatasetError(
                "joint dataset requires at least one image and one spectrum example; "
                f"got image={image_count}, spectrum={spectrum_count}"
            )

        self._modality_indices = {
            "image": range(0, image_count),
            "spectrum": range(image_count, image_count + spectrum_count),
        }

    @classmethod
    def from_config(
        cls,
        config: AstroPRISMJointDatasetConfig,
        tokenizer: Any,
    ) -> AstroPRISMJointDataset:
        """Build both validated child datasets for ``config.split``."""

        split = str(config.split).strip()
        image_config = replace(config.image, split=split)
        spectrum_config = replace(config.spectrum, split=split)
        return cls(
            AstroPRISMStage1ADataset(image_config, tokenizer),
            AstroPRISMStage1BSpectrumDataset(spectrum_config, tokenizer),
            split=split,
        )

    def _validate_child(self, modality: str, dataset: Dataset) -> None:
        active_modalities = getattr(dataset, "active_modalities", None)
        if set(active_modalities or ()) != {modality}:
            raise AstroPRISMJointDatasetError(
                f"{modality} child must expose active_modalities={{{modality!r}}}, "
                f"got {active_modalities!r}"
            )

        child_config = getattr(dataset, "config", None)
        child_split = getattr(child_config, "split", None)
        if child_split != self.split:
            raise AstroPRISMJointDatasetError(
                f"{modality} child split must be {self.split!r}, got {child_split!r}"
            )

    @property
    def modality_indices(self) -> dict[str, range]:
        """Return immutable ranges in the joint index space by modality."""

        return dict(self._modality_indices)

    @property
    def modality_lengths(self) -> dict[str, int]:
        return {
            modality: len(indices)
            for modality, indices in self._modality_indices.items()
        }

    def modality_for_index(self, item: int) -> str:
        index = self._normalize_index(item)
        return "image" if index < len(self.image_dataset) else "spectrum"

    def _normalize_index(self, item: int) -> int:
        try:
            index = operator.index(item)
        except TypeError as exc:
            raise TypeError(f"joint dataset index must be an integer, got {type(item).__name__}") from exc
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"joint dataset index {item!r} is out of range")
        return index

    def __len__(self) -> int:
        return len(self.image_dataset) + len(self.spectrum_dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self._normalize_index(item)
        image_count = len(self.image_dataset)
        if index < image_count:
            modality = "image"
            sample = self.image_dataset[index]
        else:
            modality = "spectrum"
            sample = self.spectrum_dataset[index - image_count]

        if not isinstance(sample, dict):
            raise AstroPRISMJointDatasetError(
                f"{modality} child returned {type(sample).__name__}; expected a dict"
            )
        existing_tag = sample.get("batch_modality")
        if existing_tag is not None and existing_tag != modality:
            raise AstroPRISMJointDatasetError(
                f"{modality} child returned conflicting batch_modality={existing_tag!r}"
            )
        tagged = dict(sample)
        tagged["batch_modality"] = modality
        return tagged


__all__ = [
    "ASTROPRISM_JOINT_MODALITIES",
    "ASTROPRISM_JOINT_SPLITS",
    "AstroPRISMJointDataset",
    "AstroPRISMJointDatasetConfig",
    "AstroPRISMJointDatasetError",
]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import src.data.astroprism_joint_dataset as joint_dataset_module
from accelerate.data_loader import BatchSamplerShard, prepare_data_loader
from src.data.astroprism_joint_batch_sampler import AstroPRISMJointBatchSampler
from src.data.astroprism_joint_collate import AstroPRISMJointCollator
from src.data.astroprism_joint_dataset import (
    AstroPRISMJointDataset,
    AstroPRISMJointDatasetConfig,
    AstroPRISMJointDatasetError,
)
from src.data.astroprism_stage1a_dataset import AstroPRISMStage1ADatasetConfig
from src.data.astroprism_stage1b_spectrum_dataset import (
    AstroPRISMStage1BSpectrumDatasetConfig,
)
from torch.utils.data import DataLoader, Dataset

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]


@dataclass(frozen=True)
class _ChildConfig:
    split: str


class _ChildDataset(Dataset):
    def __init__(
        self,
        modality: str,
        count: int,
        *,
        split: str,
        config: Any | None = None,
    ) -> None:
        self.active_modalities = {modality}
        self.modality = modality
        self.config = config if config is not None else _ChildConfig(split=split)
        self.samples = [
            {modality: f"{modality}-{index}", "sample_id": f"{modality}-{index}"}
            for index in range(count)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def _joint(
    split: str,
    *,
    image_count: int = 7,
    spectrum_count: int = 5,
) -> AstroPRISMJointDataset:
    return AstroPRISMJointDataset(
        _ChildDataset("image", image_count, split=split),
        _ChildDataset("spectrum", spectrum_count, split=split),
        split=split,
    )


def _batch_modality(dataset: AstroPRISMJointDataset, batch: list[int]) -> str:
    modalities = {dataset.modality_for_index(index) for index in batch}
    assert len(modalities) == 1
    return modalities.pop()


def test_joint_dataset_exposes_both_modalities_and_tags_copies() -> None:
    dataset = _joint("train", image_count=3, spectrum_count=2)

    assert dataset.active_modalities == {"image", "spectrum"}
    assert dataset.modality_lengths == {"image": 3, "spectrum": 2}
    assert list(dataset.modality_indices["image"]) == [0, 1, 2]
    assert list(dataset.modality_indices["spectrum"]) == [3, 4]
    assert dataset[0]["batch_modality"] == "image"
    assert dataset[3]["batch_modality"] == "spectrum"
    assert dataset[-1]["sample_id"] == "spectrum-1"
    assert "batch_modality" not in dataset.image_dataset.samples[0]
    assert "batch_modality" not in dataset.spectrum_dataset.samples[0]


def test_joint_dataset_requires_children_on_the_configured_split() -> None:
    with pytest.raises(AstroPRISMJointDatasetError, match="spectrum child split"):
        AstroPRISMJointDataset(
            _ChildDataset("image", 2, split="train"),
            _ChildDataset("spectrum", 2, split="validation"),
            split="train",
        )


def test_from_config_applies_one_split_to_both_source_datasets(monkeypatch) -> None:
    image_config = AstroPRISMStage1ADatasetConfig(
        raw_manifest_path=Path("image-raw.json"),
        candidates_manifest_path=Path("image-captions.json"),
        split="test",
    )
    spectrum_config = AstroPRISMStage1BSpectrumDatasetConfig(
        raw_manifest_path=Path("spectrum-raw.json"),
        captions_manifest_path=Path("spectrum-captions.json"),
        split="validation",
    )
    captured: dict[str, Any] = {}

    def build_image(config, tokenizer):
        captured["image"] = (config, tokenizer)
        return _ChildDataset("image", 2, split=config.split, config=config)

    def build_spectrum(config, tokenizer):
        captured["spectrum"] = (config, tokenizer)
        return _ChildDataset("spectrum", 2, split=config.split, config=config)

    monkeypatch.setattr(joint_dataset_module, "AstroPRISMStage1ADataset", build_image)
    monkeypatch.setattr(
        joint_dataset_module,
        "AstroPRISMStage1BSpectrumDataset",
        build_spectrum,
    )
    tokenizer = object()

    dataset = AstroPRISMJointDataset.from_config(
        AstroPRISMJointDatasetConfig(
            image=image_config,
            spectrum=spectrum_config,
            split="train",
        ),
        tokenizer,
    )

    assert dataset.split == "train"
    assert captured["image"][0].split == "train"
    assert captured["spectrum"][0].split == "train"
    assert captured["image"][1] is tokenizer
    assert captured["spectrum"][1] is tokenizer


def test_joint_collator_routes_and_tags_homogeneous_batches() -> None:
    calls: list[tuple[str, list[dict[str, Any]]]] = []

    def image_collator(batch):
        calls.append(("image", batch))
        return {"sample_ids": [sample["sample_id"] for sample in batch]}

    def spectrum_collator(batch):
        calls.append(("spectrum", batch))
        return {"sample_ids": [sample["sample_id"] for sample in batch]}

    collator = AstroPRISMJointCollator(image_collator, spectrum_collator)
    output = collator(
        [
            {"image": "image-0", "sample_id": "image-0", "batch_modality": "image"},
            {"image": "image-1", "sample_id": "image-1", "batch_modality": "image"},
        ]
    )

    assert output == {
        "sample_ids": ["image-0", "image-1"],
        "batch_modality": "image",
    }
    assert calls[0][0] == "image"
    assert all("batch_modality" not in sample for sample in calls[0][1])


def test_joint_collator_fails_closed_on_mixed_modalities() -> None:
    collator = AstroPRISMJointCollator(lambda batch: {}, lambda batch: {})
    with pytest.raises(ValueError, match="modality-homogeneous"):
        collator(
            [
                {"image": "image-0", "batch_modality": "image"},
                {"spectrum": "spectrum-0", "batch_modality": "spectrum"},
            ]
        )


def test_train_sampler_is_balanced_deterministic_and_epoch_shuffled() -> None:
    dataset = _joint("train", image_count=61, spectrum_count=37)
    sampler = AstroPRISMJointBatchSampler(
        dataset,
        batch_size=2,
        seed=17,
        num_replicas=12,
    )

    epoch_zero = list(sampler)
    assert epoch_zero == list(sampler)
    assert len(epoch_zero) == len(sampler) == 72
    assert all(len(batch) == 2 for batch in epoch_zero)
    assert all(
        len({_batch_modality(dataset, batch) for batch in epoch_zero[start : start + 12]})
        == 1
        for start in range(0, len(epoch_zero), 12)
    )
    counts = {"image": 0, "spectrum": 0}
    for batch in epoch_zero:
        counts[_batch_modality(dataset, batch)] += 1
    assert counts == {"image": 36, "spectrum": 36}

    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert epoch_one == list(sampler)
    assert epoch_one != epoch_zero


def test_train_sampler_is_rank_synchronous_under_12_rank_accelerate_sharding() -> None:
    dataset = _joint("train", image_count=61, spectrum_count=37)
    sampler = AstroPRISMJointBatchSampler(
        dataset,
        batch_size=2,
        seed=29,
        num_replicas=12,
    )
    rank_batches = [
        list(
            BatchSamplerShard(
                sampler,
                num_processes=12,
                process_index=rank,
                split_batches=False,
                even_batches=True,
            )
        )
        for rank in range(12)
    ]

    assert {len(batches) for batches in rank_batches} == {6}
    step_modalities: list[str] = []
    for optimizer_step in range(6):
        modalities = {
            _batch_modality(dataset, rank_batches[rank][optimizer_step])
            for rank in range(12)
        }
        assert len(modalities) == 1
        step_modalities.append(modalities.pop())
    assert step_modalities.count("image") == 3
    assert step_modalities.count("spectrum") == 3



def test_prepared_accelerate_loader_propagates_epoch_to_joint_sampler() -> None:
    dataset = _joint("train", image_count=61, spectrum_count=37)
    sampler = AstroPRISMJointBatchSampler(
        dataset,
        batch_size=2,
        seed=43,
        num_replicas=12,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda samples: samples,
    )
    prepared = prepare_data_loader(
        loader,
        device=None,
        num_processes=12,
        process_index=0,
        split_batches=False,
        put_on_device=False,
    )

    prepared.set_epoch(0)
    epoch_zero = [
        [sample["sample_id"] for sample in batch]
        for batch in prepared
    ]
    assert sampler.epoch == 0

    prepared.set_epoch(4)
    epoch_four = [
        [sample["sample_id"] for sample in batch]
        for batch in prepared
    ]
    assert sampler.epoch == 4
    assert epoch_four != epoch_zero

def test_sampler_resolves_world_size_from_environment(monkeypatch) -> None:
    for variable in (
        "WORLD_SIZE",
        "ACCELERATE_NUM_PROCESSES",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "12")

    sampler = AstroPRISMJointBatchSampler(_joint("train"), batch_size=2)

    assert sampler.num_replicas == 12


@pytest.mark.parametrize("split", ["validation", "test"])
def test_evaluation_sampler_is_deterministic_and_never_repeats(split: str) -> None:
    dataset = _joint(split, image_count=5, spectrum_count=3)
    sampler = AstroPRISMJointBatchSampler(
        dataset,
        batch_size=2,
        seed=31,
        num_replicas=12,
    )

    batches = list(sampler)
    sampler.set_epoch(9)

    assert list(sampler) == batches
    assert len(batches) == len(sampler) == 5
    assert all(len({_batch_modality(dataset, batch)}) == 1 for batch in batches)
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(dataset)))


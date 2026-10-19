"""Rank-synchronous homogeneous batching for joint AstroPRISM alignment."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from .astroprism_joint_dataset import (
    ASTROPRISM_JOINT_MODALITIES,
    ASTROPRISM_JOINT_SPLITS,
    AstroPRISMJointDataset,
)

_WORLD_SIZE_ENVIRONMENT_VARIABLES = (
    "WORLD_SIZE",
    "ACCELERATE_NUM_PROCESSES",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
)


class AstroPRISMJointBatchSampler(Sampler[list[int]]):
    """Yield rank-synchronous image-only or spectrum-only batches.

    Training produces equal numbers of full batches for both modalities. The
    shorter modality is deterministically reshuffled and cycled as needed.
    Batches are emitted in blocks of ``num_replicas`` with one modality per
    block. With Accelerate whole-batch sharding (``split_batches=False``), all
    ranks therefore execute the same modality on each optimizer step.

    ``set_epoch`` selects a deterministic epoch-specific ordering. Validation
    and test never shuffle or repeat examples. Their final batches may be
    partial, so distributed evaluation should use ``even_batches=False`` to
    prevent Accelerate from filling a partial batch with another modality.
    """

    def __init__(
        self,
        dataset: AstroPRISMJointDataset,
        *,
        batch_size: int,
        seed: int = 0,
        drop_last: bool | None = None,
        num_replicas: int | None = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if dataset.split not in ASTROPRISM_JOINT_SPLITS:
            raise ValueError(f"unsupported joint dataset split {dataset.split!r}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = int(seed)
        self.split = dataset.split
        self.is_training = self.split == "train"
        self.drop_last = False if drop_last is None else bool(drop_last)
        self.num_replicas = self._resolve_num_replicas(num_replicas)
        self.shuffle = self.is_training
        self.balanced = self.is_training
        self.epoch = 0

        indices = dataset.modality_indices
        self._indices = {
            modality: tuple(indices[modality])
            for modality in ASTROPRISM_JOINT_MODALITIES
        }
        for modality, modality_indices in self._indices.items():
            if not modality_indices:
                raise ValueError(f"joint batch sampler requires non-empty {modality} indices")

        if self.is_training and self._training_blocks_per_modality() == 0:
            longest = max(len(indices) for indices in self._indices.values())
            raise ValueError(
                "drop_last=True would produce no complete distributed training block: "
                f"longest modality has {longest} examples, batch_size={self.batch_size}, "
                f"num_replicas={self.num_replicas}"
            )

    @staticmethod
    def _resolve_num_replicas(explicit: int | None) -> int:
        if explicit is not None:
            if not isinstance(explicit, int) or isinstance(explicit, bool) or explicit <= 0:
                raise ValueError(
                    f"num_replicas must be a positive integer, got {explicit!r}"
                )
            return explicit

        for variable in _WORLD_SIZE_ENVIRONMENT_VARIABLES:
            value = os.environ.get(variable)
            if value is None or not value.strip():
                continue
            try:
                resolved = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"distributed environment variable {variable} must be an integer, "
                    f"got {value!r}"
                ) from exc
            if resolved <= 0:
                raise ValueError(
                    f"distributed environment variable {variable} must be positive, "
                    f"got {value!r}"
                )
            return resolved
        return 1

    @property
    def sampler(self) -> AstroPRISMJointBatchSampler:
        """Expose the epoch hook through Accelerate's custom-sampler chain.

        ``DataLoaderShard.set_epoch`` traverses
        ``BatchSamplerShard.batch_sampler.sampler``. Returning this batch
        sampler lets that hook reach :meth:`set_epoch` without changing the
        batches consumed by PyTorch or Accelerate.
        """
        return self

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}")
        self.epoch = epoch

    def _training_blocks_per_modality(self) -> int:
        longest = max(len(indices) for indices in self._indices.values())
        samples_per_world_block = self.batch_size * self.num_replicas
        if self.drop_last:
            return longest // samples_per_world_block
        return math.ceil(longest / samples_per_world_block)

    def _training_batches_per_modality(self) -> int:
        return self._training_blocks_per_modality() * self.num_replicas

    def _epoch_seed(self, stream: int) -> int:
        return (
            self.seed
            + (self.epoch + 1) * 1_000_003
            + (stream + 1) * 97_409
        ) & ((1 << 63) - 1)

    def _draw_training_indices(
        self,
        indices: Sequence[int],
        *,
        count: int,
        stream: int,
    ) -> list[int]:
        rng = random.Random(self._epoch_seed(stream))
        drawn: list[int] = []
        while len(drawn) < count:
            cycle = list(indices)
            rng.shuffle(cycle)
            drawn.extend(cycle[: count - len(drawn)])
        return drawn

    def _training_batches(self) -> list[list[int]]:
        block_count = self._training_blocks_per_modality()
        batch_count = self._training_batches_per_modality()
        sample_count = batch_count * self.batch_size
        by_modality: dict[str, list[list[int]]] = {}
        for stream, modality in enumerate(ASTROPRISM_JOINT_MODALITIES):
            drawn = self._draw_training_indices(
                self._indices[modality],
                count=sample_count,
                stream=stream,
            )
            by_modality[modality] = [
                drawn[start : start + self.batch_size]
                for start in range(0, sample_count, self.batch_size)
            ]

        order_rng = random.Random(self._epoch_seed(len(ASTROPRISM_JOINT_MODALITIES)))
        batches: list[list[int]] = []
        for block_index in range(block_count):
            modality_pair = list(ASTROPRISM_JOINT_MODALITIES)
            if order_rng.getrandbits(1):
                modality_pair.reverse()
            start = block_index * self.num_replicas
            stop = start + self.num_replicas
            for modality in modality_pair:
                batches.extend(by_modality[modality][start:stop])
        return batches

    def _evaluation_batches(self) -> list[list[int]]:
        by_modality: dict[str, list[list[int]]] = {}
        for modality in ASTROPRISM_JOINT_MODALITIES:
            indices = self._indices[modality]
            batches = [
                list(indices[start : start + self.batch_size])
                for start in range(0, len(indices), self.batch_size)
            ]
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches.pop()
            by_modality[modality] = batches

        batches: list[list[int]] = []
        batch_count = max(len(value) for value in by_modality.values())
        for batch_index in range(batch_count):
            for modality in ASTROPRISM_JOINT_MODALITIES:
                modality_batches = by_modality[modality]
                if batch_index < len(modality_batches):
                    batches.append(modality_batches[batch_index])
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._training_batches() if self.is_training else self._evaluation_batches()
        yield from batches

    def __len__(self) -> int:
        if self.is_training:
            return 2 * self._training_batches_per_modality()

        total = 0
        for indices in self._indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


__all__ = ["AstroPRISMJointBatchSampler"]

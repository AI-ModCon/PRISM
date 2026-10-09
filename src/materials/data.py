"""Data pipeline for paired Materials Project text, CIF, and scalar targets."""

from __future__ import annotations

import csv
import hashlib
import math
import random
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]")


@dataclass(frozen=True)
class MaterialRecord:
    material_id: str
    text_path: Path
    cif_path: Path
    target: float


def build_manifest(
    materials_dir: str | Path,
    target_name: str = "band_gap",
    max_records: int | None = None,
    seed: int = 17,
) -> list[MaterialRecord]:
    """Join target rows to ``text/<id>.txt`` and ``bulk_data_full/<id>.cif``.

    Duplicate CSV rows with the same ID and target are collapsed. Conflicting
    duplicate targets fail loudly instead of silently choosing one value.
    """
    root = Path(materials_dir)
    target_path = root / "targets_material.csv"
    targets: dict[str, float] = {}
    with target_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"{target_path} must contain an 'id' column")
        if target_name not in reader.fieldnames:
            raise ValueError(
                f"Unknown target {target_name!r}; available columns: {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            material_id = row["id"].strip()
            raw_target = row[target_name].strip()
            if not material_id or not raw_target:
                continue
            try:
                target = float(raw_target)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {target_name!r} value on CSV row {row_number}: {raw_target!r}"
                ) from exc
            if not math.isfinite(target):
                continue
            if material_id in targets and targets[material_id] != target:
                raise ValueError(f"Conflicting targets for material ID {material_id!r}")
            targets[material_id] = target

    target_items = list(targets.items())
    if max_records is not None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        # Shuffle IDs before filesystem checks, then stop when enough complete
        # pairs have been found. This keeps smoke runs from stat-ing 139k files.
        random.Random(seed).shuffle(target_items)
    records = []
    for material_id, target in target_items:
        text_path = root / "text" / f"{material_id}.txt"
        cif_path = root / "bulk_data_full" / f"{material_id}.cif"
        if text_path.is_file() and cif_path.is_file():
            records.append(MaterialRecord(material_id, text_path, cif_path, target))
            if max_records is not None and len(records) >= max_records:
                break
    return sorted(records, key=lambda record: record.material_id)


def split_records(
    records: Sequence[MaterialRecord],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 17,
) -> tuple[list[MaterialRecord], list[MaterialRecord], list[MaterialRecord]]:
    """Make a reproducible split whose assignment is stable as the dataset grows."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to less than 1")
    train: list[MaterialRecord] = []
    val: list[MaterialRecord] = []
    test: list[MaterialRecord] = []
    val_cut = val_fraction
    test_cut = val_fraction + test_fraction
    for record in records:
        digest = hashlib.blake2b(
            f"{seed}:{record.material_id}".encode(), digest_size=8
        ).digest()
        value = int.from_bytes(digest, "big") / 2**64
        if value < val_cut:
            val.append(record)
        elif value < test_cut:
            test.append(record)
        else:
            train.append(record)
    if not train or not val or not test:
        raise ValueError("The requested split produced an empty train, validation, or test set")
    return train, val, test


class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, tokens: Sequence[str]):
        self.tokens = [self.PAD, self.UNK, *tokens]
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def fit(
        cls, records: Iterable[MaterialRecord], min_frequency: int = 2, max_size: int = 30_000
    ) -> Vocabulary:
        counts: Counter[str] = Counter()
        for record in records:
            counts.update(tokenize(record.text_path.read_text(encoding="utf-8")))
        tokens = [
            token
            for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_frequency
        ][: max(0, max_size - 2)]
        return cls(tokens)

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        ids = [self.token_to_id.get(token, 1) for token in tokenize(text)[:max_length]]
        return torch.tensor(ids or [1], dtype=torch.long)


class TextTokenizer(Protocol):
    """Minimal tokenizer interface used by :class:`MaterialsDataset`."""

    pad_token_id: int

    def encode(self, text: str, max_length: int) -> torch.Tensor: ...


class HuggingFaceTokenizer:
    """Adapt a Hugging Face tokenizer to the materials data pipeline."""

    def __init__(self, model_id: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Pretrained text encoding requires transformers; install requirements/base.txt"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(f"Tokenizer {model_id!r} has neither a pad nor EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = int(self.tokenizer.pad_token_id)

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        return torch.tensor(ids or [self.pad_token_id], dtype=torch.long)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cif_to_graph(
    cif_path: str | Path, cutoff: float = 5.0, max_neighbors: int = 16
) -> dict[str, torch.Tensor]:
    """Convert a CIF to a periodic radius graph using pymatgen.

    Atomic number is the categorical node input. Every periodic neighbor
    returned by pymatgen is retained, so multiple images of an atom can form
    distinct edges with distinct distances.
    """
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ImportError(
            "Crystal parsing requires pymatgen; install requirements/materials.txt"
        ) from exc

    with warnings.catch_warnings():
        # Materials Project CIFs commonly need harmless finite-precision
        # rounding; emitting this warning for every sample overwhelms job logs.
        warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")
        structure = Structure.from_file(str(cif_path), primitive=False)
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")
    center, neighbor, _images, distances = structure.get_neighbor_list(r=cutoff)
    # A pure radius graph can contain hundreds of periodic images per site.
    # Bound memory while retaining the chemically closest neighborhood.
    candidates: list[list[int]] = [[] for _ in structure]
    for edge_number, atom_index in enumerate(center):
        candidates[int(atom_index)].append(edge_number)
    keep = np.asarray(
        [
            edge_number
            for atom_edges in candidates
            for edge_number in sorted(atom_edges, key=distances.__getitem__)[:max_neighbors]
        ],
        dtype=np.int64,
    )
    center, neighbor, distances = center[keep], neighbor[keep], distances[keep]
    atomic_numbers = [
        site.specie.Z
        if site.is_ordered
        else max(
            site.species.items(),
            key=lambda element_and_occupancy: element_and_occupancy[1],
        )[0].Z
        for site in structure
    ]
    return {
        "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.long),
        "edge_index": torch.from_numpy(np.stack((center, neighbor))).long(),
        "edge_distance": torch.from_numpy(np.asarray(distances, dtype=np.float32)),
    }


class MaterialsDataset(Dataset):
    def __init__(
        self,
        records: Sequence[MaterialRecord],
        vocabulary: Vocabulary | TextTokenizer,
        max_text_length: int = 192,
        cutoff: float = 5.0,
        max_neighbors: int = 16,
        graph_cache_dir: str | Path | None = None,
    ):
        self.records = list(records)
        self.vocabulary = vocabulary
        self.max_text_length = max_text_length
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.graph_cache_dir = Path(graph_cache_dir) if graph_cache_dir else None
        if self.graph_cache_dir:
            self.graph_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def _graph(self, record: MaterialRecord) -> dict[str, torch.Tensor]:
        cache_path = None
        if self.graph_cache_dir:
            cache_path = self.graph_cache_dir / (
                f"{record.material_id}.cutoff-{self.cutoff:g}.neighbors-{self.max_neighbors}.pt"
            )
            if cache_path.is_file():
                return torch.load(cache_path, map_location="cpu", weights_only=True)
        graph = cif_to_graph(record.cif_path, self.cutoff, self.max_neighbors)
        if cache_path:
            temporary = cache_path.with_suffix(".tmp")
            torch.save(graph, temporary)
            temporary.replace(cache_path)
        return graph

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        text = record.text_path.read_text(encoding="utf-8")
        return {
            "material_id": record.material_id,
            "text_ids": self.vocabulary.encode(text, self.max_text_length),
            "text_pad_id": getattr(self.vocabulary, "pad_token_id", 0),
            "graph": self._graph(record),
            "target": torch.tensor(record.target, dtype=torch.float32),
        }


def collate_materials(samples: Sequence[dict]) -> dict[str, torch.Tensor | list[str]]:
    max_tokens = max(sample["text_ids"].numel() for sample in samples)
    pad_token_id = samples[0].get("text_pad_id", 0)
    if any(sample.get("text_pad_id", 0) != pad_token_id for sample in samples):
        raise ValueError("All samples in a batch must use the same text padding token")
    text_ids = torch.full((len(samples), max_tokens), pad_token_id, dtype=torch.long)
    text_mask = torch.zeros((len(samples), max_tokens), dtype=torch.bool)
    node_parts = []
    edge_parts = []
    distance_parts = []
    graph_batch = []
    node_offset = 0
    for graph_index, sample in enumerate(samples):
        ids = sample["text_ids"]
        text_ids[graph_index, : ids.numel()] = ids
        text_mask[graph_index, : ids.numel()] = True
        graph = sample["graph"]
        nodes = graph["atomic_numbers"]
        node_parts.append(nodes)
        edge_parts.append(graph["edge_index"] + node_offset)
        distance_parts.append(graph["edge_distance"])
        graph_batch.append(torch.full((nodes.numel(),), graph_index, dtype=torch.long))
        node_offset += nodes.numel()
    return {
        "material_id": [sample["material_id"] for sample in samples],
        "text_ids": text_ids,
        "text_mask": text_mask,
        "atomic_numbers": torch.cat(node_parts),
        "edge_index": torch.cat(edge_parts, dim=1),
        "edge_distance": torch.cat(distance_parts),
        "graph_batch": torch.cat(graph_batch),
        "target": torch.stack([sample["target"] for sample in samples]),
    }

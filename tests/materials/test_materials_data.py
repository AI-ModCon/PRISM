from pathlib import Path

import torch
from src.materials.data import MaterialRecord, build_manifest, collate_materials, split_records


def test_manifest_joins_modalities_and_collapses_identical_duplicates(tmp_path: Path):
    (tmp_path / "text").mkdir()
    (tmp_path / "bulk_data_full").mkdir()
    (tmp_path / "targets_material.csv").write_text(
        "id,formation_energy,total_energy,band_gap\n"
        "mp-1,-1,-2,1.5\n"
        "mp-1,-1,-2,1.5\n"
        "mp-2,-1,-2,2.0\n",
        encoding="utf-8",
    )
    (tmp_path / "text" / "mp-1.txt").write_text("A crystal", encoding="utf-8")
    (tmp_path / "bulk_data_full" / "mp-1.cif").touch()

    records = build_manifest(tmp_path)

    assert [(record.material_id, record.target) for record in records] == [("mp-1", 1.5)]


def test_hash_split_is_disjoint_and_repeatable(tmp_path: Path):
    records = [
        MaterialRecord(f"mp-{i}", tmp_path / "t", tmp_path / "c", float(i))
        for i in range(100)
    ]
    first = split_records(records, seed=9)
    second = split_records(records, seed=9)

    assert [[record.material_id for record in split] for split in first] == [
        [record.material_id for record in split] for split in second
    ]
    sets = [{record.material_id for record in split} for split in first]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(map(len, first)) == 100


def test_collate_uses_explicit_mask_when_pad_token_is_also_content():
    samples = [
        {
            "material_id": "mp-1",
            "text_ids": torch.tensor([7, 9]),
            "text_pad_id": 9,
            "graph": {
                "atomic_numbers": torch.tensor([6]),
                "edge_index": torch.tensor([[0], [0]]),
                "edge_distance": torch.tensor([1.0]),
            },
            "target": torch.tensor(1.0),
        },
        {
            "material_id": "mp-2",
            "text_ids": torch.tensor([8]),
            "text_pad_id": 9,
            "graph": {
                "atomic_numbers": torch.tensor([8]),
                "edge_index": torch.tensor([[0], [0]]),
                "edge_distance": torch.tensor([1.0]),
            },
            "target": torch.tensor(2.0),
        },
    ]

    batch = collate_materials(samples)

    assert batch["text_ids"].tolist() == [[7, 9], [8, 9]]
    assert batch["text_mask"].tolist() == [[True, True], [True, False]]

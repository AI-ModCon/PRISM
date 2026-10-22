from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.accept_astroprism_stage1a import (
    AcceptanceConfig,
    AcceptanceValidationError,
    accept_stage1a,
    sha256_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]


def _write_source(tmp_path: Path) -> tuple[Path, list[dict]]:
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for index in range(646):
        ra = index * 0.1
        if index == 536:
            ra = 535 * 0.1 + 1.0 / 3600.0
        records.append(
            {
                "schema_version": 1,
                "candidate_index": index,
                "paired_targetid": str(10_000 + index),
                "caption_object_id": f"caption-{index}",
                "coordinates_deg": {
                    "paired_ra": ra,
                    "paired_dec": 0.0,
                    "caption_ra": ra,
                    "caption_dec": 0.0,
                },
                "caption": {
                    "original": f"  Original caption {index} — preserved.\n",
                    "summary_original": f"Summary {index}",
                    "status": "unreviewed",
                },
                "image_embedding": {
                    "file": "image_embeddings.f16.npy",
                    "index": index,
                    "dtype": "float16",
                    "shape": [576, 768],
                },
                "acceptance_status": "pending_review",
                "source_group_id": None,
                "split": None,
            }
        )
    candidates = source / "candidates.jsonl"
    candidates.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    embeddings = source / "image_embeddings.f16.npy"
    embeddings.write_bytes(b"parity-reference")
    manifest = {
        "schema_version": 1,
        "artifact_type": "astroprism_stage1a_candidates",
        "pins": {"roster_sha256": "roster-digest"},
        "roster": {"row_count": 646, "sha256": "roster-digest"},
        "outputs": {
            "candidates": {
                "path": candidates.name,
                "row_count": 646,
                "size_bytes": candidates.stat().st_size,
                "sha256": sha256_file(candidates),
            },
            "image_embeddings": {
                "path": embeddings.name,
                "dtype": "float16",
                "shape": [646, 576, 768],
                "size_bytes": embeddings.stat().st_size,
                "sha256": sha256_file(embeddings),
            },
        },
        "validation": {"acceptance_status": "pending_review"},
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, records


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_promotes_deterministically_with_exact_grouped_splits(tmp_path: Path) -> None:
    source_manifest, originals = _write_source(tmp_path)
    source_candidates = source_manifest.parent / "candidates.jsonl"
    original_source_digest = sha256_file(source_candidates)

    first = tmp_path / "accepted-a"
    second = tmp_path / "accepted-b"
    first_summary = accept_stage1a(AcceptanceConfig(source_manifest, first))
    second_summary = accept_stage1a(AcceptanceConfig(source_manifest, second))

    assert (first / "candidates.jsonl").read_bytes() == (second / "candidates.jsonl").read_bytes()
    assert sha256_file(source_candidates) == original_source_digest
    assert first_summary["candidate_count"] == 646
    assert first_summary["group_count"] == 645
    assert first_summary["split_counts"] == {"test": 65, "train": 516, "validation": 65}
    assert first_summary["mapping_sha256"] == second_summary["mapping_sha256"]

    accepted = _read_jsonl(first / "candidates.jsonl")
    assert [record["candidate_index"] for record in accepted] == list(range(646))
    assert [record["caption"]["original"] for record in accepted] == [
        record["caption"]["original"] for record in originals
    ]
    assert all(record["acceptance_status"] == "accepted" for record in accepted)
    assert all(record["caption"]["status"] == "released_reused" for record in accepted)
    assert accepted[535]["source_group_id"] == accepted[536]["source_group_id"]
    assert accepted[535]["split"] == accepted[536]["split"]
    assert all(
        record["image_embedding"]["file"] == "../source/image_embeddings.f16.npy"
        for record in accepted
    )

    group_splits: dict[str, set[str]] = {}
    for record in accepted:
        group_splits.setdefault(record["source_group_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())

    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    output = manifest["outputs"]["candidates"]
    assert output["sha256"] == sha256_file(first / "candidates.jsonl")
    assert output["size_bytes"] == (first / "candidates.jsonl").stat().st_size
    assert manifest["splits"]["grouping_policy"] == "skycc60-v1"
    assert manifest["splits"]["overlap_radius_arcsec"] == 60.0
    assert manifest["splits"]["row_counts"] == {
        "test": 65,
        "train": 516,
        "validation": 65,
    }
    assert manifest["lineage"]["source_candidates"]["sha256"] == original_source_digest
    assert "validation" not in manifest


def test_refuses_source_hash_mismatch_without_writing_output(tmp_path: Path) -> None:
    source_manifest, _records = _write_source(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["outputs"]["candidates"]["sha256"] = "0" * 64
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "accepted"

    with pytest.raises(AcceptanceValidationError, match="SHA-256"):
        accept_stage1a(AcceptanceConfig(source_manifest, output))

    assert not output.exists()
    assert not list(tmp_path.glob(".accepted.tmp-*"))


def test_refuses_to_overwrite_or_repromote(tmp_path: Path) -> None:
    source_manifest, _records = _write_source(tmp_path)
    output = tmp_path / "accepted"
    accept_stage1a(AcceptanceConfig(source_manifest, output))

    with pytest.raises(FileExistsError, match="already exists"):
        accept_stage1a(AcceptanceConfig(source_manifest, output))

    source_candidates = source_manifest.parent / "candidates.jsonl"
    records = _read_jsonl(source_candidates)
    records[0]["acceptance_status"] = "accepted"
    source_candidates.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["outputs"]["candidates"]["sha256"] = sha256_file(source_candidates)
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AcceptanceValidationError, match="not pending_review"):
        accept_stage1a(AcceptanceConfig(source_manifest, tmp_path / "second"))

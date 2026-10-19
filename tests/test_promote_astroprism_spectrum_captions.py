from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.promote_astroprism_spectrum_captions import (
    PromotionConfig,
    PromotionValidationError,
    promote_spectrum_captions,
    sha256_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]

SPLIT_COUNTS = {"train": 2, "validation": 1, "test": 1}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _accepted_fixture(root: Path) -> tuple[Path, Path, list[dict]]:
    splits = ("train", "validation", "train", "test")
    rows = [
        {
            "candidate_index": index,
            "paired_targetid": f"target-{index}",
            "caption_object_id": f"caption-{index}",
            "source_group_id": f"group-{index}",
            "split": split,
            "acceptance_status": "accepted",
        }
        for index, split in enumerate(splits)
    ]
    candidates = root / "accepted/candidates.jsonl"
    _write_jsonl(candidates, rows)
    manifest = root / "accepted/manifest.json"
    _write_json(
        manifest,
        {
            "artifact_type": "astroprism_stage1a_candidates",
            "roster": {"row_count": 4, "sha256": "fixture-roster-sha256"},
            "outputs": {
                "candidates": {
                    "path": "candidates.jsonl",
                    "row_count": 4,
                    "size_bytes": candidates.stat().st_size,
                    "sha256": sha256_file(candidates),
                }
            },
            "splits": {"row_counts": SPLIT_COUNTS},
        },
    )
    return candidates, manifest, rows


def _bundle_fixture(
    root: Path, accepted: list[dict], indices: list[int], *, heldout: bool
) -> Path:
    root.mkdir(parents=True)
    records: list[dict] = []
    requests: list[dict] = []
    audits: list[dict] = []
    for index in indices:
        identity = accepted[index]
        teacher_reference = {
            "facts": [{"fact_id": "catalog.redshift", "value": 0.1 + index}],
            "catalog_context": {
                "redshift": 0.1 + index,
                "source": {
                    "path": f"/tmp/astroprism-fixture/cache/data/shard-{index}.parquet",
                    "shard": f"data/shard-{index}.parquet",
                    "row": index,
                },
            },
        }
        model_digest = f"model-{index}"
        payload_digest = f"payload-{index}"
        records.append(
            {
                "schema_version": 1,
                "artifact_type": "astroprism_spectrum_caption_candidate",
                **identity,
                "student_input": {
                    "modalities": ["spectrum"],
                    "raw_manifest": "/tmp/astroprism-fixture/raw_spectra/manifest.json",
                    "array_index": index,
                    "wavelength_frame": "observed_vacuum",
                    "spectrum_length": 7781,
                },
                "teacher_reference": teacher_reference,
                "spectrum_caption": {
                    "original": f"Grounded spectrum caption {index}.",
                    "revised": None,
                    "status": "pending_review",
                },
                "audit_sidecar": {
                    "claims": [
                        {"text": f"caption {index}", "fact_ids": ["catalog.redshift"]}
                    ],
                    "spectral_grounding_status": "candidate_unverified",
                },
                "generation": {
                    "method": "fixture-generation",
                    "model_input_sha256": model_digest,
                    "compact_payload_sha256": payload_digest,
                },
                "training_policy": {
                    "eligible_for_training": False,
                    "eligible_after_review": not heldout,
                    "role": (
                        "held_out_inspection_only"
                        if heldout
                        else "training_candidate_pending_review"
                    ),
                },
            }
        )
        requests.append(
            {
                "candidate_index": index,
                "paired_targetid": identity["paired_targetid"],
                "split": identity["split"],
                "model_input": {"fixture": index},
                "model_input_sha256": model_digest,
                "compact_payload_sha256": payload_digest,
            }
        )
        audits.append(
            {
                "candidate_index": index,
                "paired_targetid": identity["paired_targetid"],
                "caption_object_id": identity["caption_object_id"],
                "source_group_id": identity["source_group_id"],
                "split": identity["split"],
                "model_input_sha256": model_digest,
                "compact_payload_sha256": payload_digest,
                "teacher_reference": teacher_reference,
                "training_policy": {"eligible_for_training": False},
            }
        )
        plot = root / f"plots/candidate_{index:04d}.png"
        plot.parent.mkdir(exist_ok=True)
        plot.write_bytes(b"\x89PNG\r\n\x1a\n" + f"fixture-{index}".encode())

    _write_jsonl(root / "captions.jsonl", records)
    _write_jsonl(root / "generation_requests.jsonl", requests)
    _write_jsonl(root / "request_audit.jsonl", audits)
    (root / "review.md").write_text("Pending review fixture.\n", encoding="utf-8")
    outputs = {
        path.relative_to(root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }
    split_counts = {
        split: sum(accepted[index]["split"] == split for index in indices)
        for split in sorted({accepted[index]["split"] for index in indices})
    }
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "astroprism_spectrum_caption_pilot",
            "role": "review_only_not_training_eligible",
            "row_count": len(indices),
            "candidate_indices": indices,
            "held_out_candidate_indices": indices if heldout else [],
            "split_counts": split_counts,
            "implementation": {"fixture": True},
            "sources": {"paper": {"url": "https://example.test/paper"}},
            "outputs": outputs,
        },
    )
    return root


def _complete_fixture(tmp_path: Path, *, link_mode: str = "copy") -> PromotionConfig:
    candidates, accepted_manifest, accepted = _accepted_fixture(tmp_path)
    train = _bundle_fixture(tmp_path / "train", accepted, [0, 2], heldout=False)
    heldout = _bundle_fixture(tmp_path / "heldout", accepted, [1, 3], heldout=True)
    return PromotionConfig(
        output_dir=tmp_path / "promoted",
        accepted_candidates_path=candidates,
        accepted_manifest_path=accepted_manifest,
        train_bundle=train,
        heldout_bundle=heldout,
        expected_count=4,
        expected_split_counts=SPLIT_COUNTS,
        link_mode=link_mode,
    )


def test_promotes_all_splits_and_preserves_portable_provenance(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    source_hashes = {
        path: sha256_file(path)
        for root in (config.train_bundle, config.heldout_bundle)
        for path in Path(root).rglob("*")
        if path.is_file()
    }

    summary = promote_spectrum_captions(config)
    output = config.output_dir
    records = _read_jsonl(output / "captions.jsonl")

    assert summary["candidate_count"] == 4
    assert summary["gradient_training_count"] == 2
    assert summary["evaluation_count"] == 2
    assert [row["candidate_index"] for row in records] == [0, 1, 2, 3]
    assert all(row["spectrum_caption"]["status"] == "accepted" for row in records)
    assert all(row["review"]["status"] == "accepted" for row in records)
    assert all(row["review"]["decision_source"] == "user_instruction" for row in records)
    assert all(row["training_policy"]["dataset_ready"] is True for row in records)

    expected_roles = {
        "train": "alignment_training",
        "validation": "alignment_validation",
        "test": "alignment_test",
    }
    for row in records:
        is_train = row["split"] == "train"
        policy = row["training_policy"]
        assert policy["gradient_training_allowed"] is is_train
        assert policy["eligible_for_training"] is is_train
        assert policy["eligible_for_evaluation"] is (not is_train)
        assert policy["role"] == expected_roles[row["split"]]
        assert row["student_input"]["raw_manifest"] == "../raw_spectra/manifest.json"
        source = row["teacher_reference"]["catalog_context"]["source"]
        assert source["path"] == source["shard"]

    assert records[0]["generation"]["method"] == "fixture-generation"
    assert records[0]["audit_sidecar"]["claims"][0]["fact_ids"] == ["catalog.redshift"]
    assert _read_jsonl(output / "generation_requests.jsonl")[3]["model_input"] == {
        "fixture": 3
    }
    assert _read_jsonl(output / "request_audit.jsonl")[1]["teacher_reference"][
        "catalog_context"
    ]["source"]["path"] == "data/shard-1.parquet"

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}
    )
    assert "/lus/" not in serialized
    for path, digest in source_hashes.items():
        assert sha256_file(path) == digest

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "astroprism_spectrum_captions_accepted"
    assert manifest["row_count"] == 4
    assert manifest["split_counts"] == SPLIT_COUNTS
    assert manifest["roster"] == {"row_count": 4, "sha256": "fixture-roster-sha256"}
    assert manifest["review"]["decision_source"] == "user_instruction"
    assert manifest["outputs"]["captions.jsonl"]["sha256"] == sha256_file(
        output / "captions.jsonl"
    )
    checksums = (output / "SHA256SUMS").read_text(encoding="utf-8")
    assert "  manifest.json\n" in checksums
    assert "  captions.jsonl\n" in checksums
    assert "SHA256SUMS" not in checksums


def test_hardlink_mode_links_validated_plots(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path, link_mode="hardlink")

    promote_spectrum_captions(config)

    source = config.train_bundle / "plots/candidate_0000.png"
    promoted = config.output_dir / "plots/candidate_0000.png"
    assert source.stat().st_ino == promoted.stat().st_ino


def test_rejects_source_group_leakage_before_writing_output(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    rows = _read_jsonl(config.accepted_candidates_path)
    rows[1]["source_group_id"] = rows[0]["source_group_id"]
    _write_jsonl(config.accepted_candidates_path, rows)
    manifest = json.loads(config.accepted_manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["candidates"]["size_bytes"] = config.accepted_candidates_path.stat().st_size
    manifest["outputs"]["candidates"]["sha256"] = sha256_file(
        config.accepted_candidates_path
    )
    _write_json(config.accepted_manifest_path, manifest)

    with pytest.raises(PromotionValidationError, match="source groups cross splits"):
        promote_spectrum_captions(config)

    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".promoted.tmp-*"))


def test_rejects_plot_checksum_mismatch_without_mutating_sources(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    plot = config.heldout_bundle / "plots/candidate_0001.png"
    plot.write_bytes(plot.read_bytes() + b"tampered")

    with pytest.raises(PromotionValidationError, match="mismatch"):
        promote_spectrum_captions(config)

    assert not config.output_dir.exists()


def test_overwrite_is_atomic_when_revalidation_fails(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    promote_spectrum_captions(config)
    original_digest = sha256_file(config.output_dir / "captions.jsonl")
    plot = config.train_bundle / "plots/candidate_0000.png"
    plot.write_bytes(plot.read_bytes() + b"tampered")

    overwrite = PromotionConfig(**{**config.__dict__, "overwrite": True})
    with pytest.raises(PromotionValidationError, match="mismatch"):
        promote_spectrum_captions(overwrite)

    assert sha256_file(config.output_dir / "captions.jsonl") == original_digest
    assert not list(tmp_path.glob(".promoted.tmp-*"))

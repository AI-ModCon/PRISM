from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Features, Image
from PIL import Image as PILImage
from tools.prepare_astroprism_hf_dataset import (
    CAPTION_REPO_ID,
    CAPTION_REVISION,
    IMAGE_ARRAY_CONTRACT,
    PAIRED_REPO_ID,
    PAIRED_REVISION,
    SPECTRUM_ARRAY_CONTRACT,
    HFDatasetConfig,
    PackagingValidationError,
    _dataset_card,
    _validate_accepted_rows,
    prepare_hf_dataset,
    sha256_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]


def _png(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    PILImage.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _raw_fixture(
    root: Path,
    accepted: list[dict],
    contract: dict[str, tuple[np.dtype, tuple[int, ...]]],
    *,
    spectra: bool,
) -> Path:
    root.mkdir(parents=True)
    outputs = {}
    for offset, (key, (dtype, trailing_shape)) in enumerate(contract.items()):
        shape = (len(accepted), *trailing_shape)
        if key == "spectrum_wavelength":
            array = np.broadcast_to(np.arange(trailing_shape[0], dtype=np.float32), shape).copy()
        elif dtype.kind == "S":
            array = np.full(shape, b"band", dtype=dtype)
        elif dtype == np.dtype("bool"):
            array = np.zeros(shape, dtype=dtype)
        else:
            array = np.full(shape, offset + 1, dtype=dtype)
        path = root / f"{key}.npy"
        np.save(path, array, allow_pickle=False)
        outputs[key] = {
            "path": path.name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "mmap_safe": True,
            "source_dataset": f"source/{key}",
        }
    manifest = {
        "schema_version": 1,
        "artifact_type": "raw_spectra" if spectra else "raw_images",
        "candidates": [
            {
                "candidate_index": row["candidate_index"],
                "paired_targetid": row["paired_targetid"],
                "caption_object_id": row["caption_object_id"],
            }
            for row in accepted
        ],
        "outputs": outputs,
        "source": {
            "url": "https://example.test/raw.hdf5",
            "etag": '"fixture"',
            "length_bytes": 123,
            "hdf5_row_count": 3,
        },
        **(
            {
                "spectrum": {
                    "length": 7781,
                    "observed_frame": True,
                    "mask_true_means_excluded": True,
                }
            }
            if spectra
            else {"bands": {"source_labels_preserved": True}}
        ),
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _spectrum_bundle(root: Path, accepted: list[dict], indices: list[int]) -> Path:
    plots = root / "plots"
    plots.mkdir(parents=True)
    records = []
    for index in indices:
        source = accepted[index]
        records.append(
            {
                "candidate_index": index,
                "paired_targetid": source["paired_targetid"],
                "caption_object_id": source["caption_object_id"],
                "source_group_id": source["source_group_id"],
                "split": source["split"],
                "student_input": {
                    "wavelength_frame": "observed_vacuum",
                    "spectrum_length": 7781,
                },
                "spectrum_caption": {
                    "original": f"Grounded spectrum caption {index}.",
                    "status": "accepted",
                },
                "review": {
                    "status": "accepted",
                    "decision_source": "user_instruction",
                },
                "teacher_reference": {
                    "catalog_context": {"redshift": 0.1 + index, "redshift_error": 0.01}
                },
                "audit_sidecar": {"claims": [{"text": "grounded", "fact_ids": ["x"]}]},
                "generation": {"method": "fixture"},
                "training_policy": {
                    "dataset_ready": True,
                    "gradient_training_allowed": source["split"] == "train",
                    "eligible_for_training": source["split"] == "train",
                    "eligible_for_evaluation": source["split"] != "train",
                },
            }
        )
        (plots / f"candidate_{index:04d}.png").write_bytes(_png((32, 16), (10 + index, 20, 30)))
    captions = root / "captions.jsonl"
    _write_jsonl(captions, records)
    outputs = {
        "captions.jsonl": {
            "sha256": sha256_file(captions),
            "size_bytes": captions.stat().st_size,
        }
    }
    for path in plots.iterdir():
        outputs[f"plots/{path.name}"] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    _write_json(root / "manifest.json", {"row_count": len(records), "outputs": outputs})
    return root


def _complete_fixture(tmp_path: Path) -> HFDatasetConfig:
    caption_root = tmp_path / "caption_source"
    paired_root = tmp_path / "paired_source"
    (caption_root / "data").mkdir(parents=True)
    (paired_root / "data").mkdir(parents=True)
    caption_shard = caption_root / "data/captions.parquet"
    paired_shard = paired_root / "data/paired.parquet"
    captions = [f"Accepted image caption {index}." for index in range(3)]
    summaries = [f"Summary {index}." for index in range(3)]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "object_id": f"caption-{index}",
                    "survey": "legacy_dr10",
                    "ra": float(index),
                    "dec": -float(index),
                    "image": _png((160, 160), (index, 2 * index, 3 * index)),
                    "caption": captions[index],
                    "summary": summaries[index],
                }
                for index in range(3)
            ]
        ),
        caption_shard,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "object_id_provabgs": f"100{index}",
                    "ra_provabgs": float(index),
                    "dec_provabgs": -float(index),
                }
                for index in range(3)
            ]
        ),
        paired_shard,
    )
    caption_hash = sha256_file(caption_shard)
    paired_hash = sha256_file(paired_shard)
    splits = ("train", "validation", "test")
    accepted = []
    for index, split in enumerate(splits):
        accepted.append(
            {
                "schema_version": 1,
                "candidate_index": index,
                "paired_targetid": f"100{index}",
                "caption_object_id": f"caption-{index}",
                "coordinates_deg": {
                    "paired_ra": float(index),
                    "paired_dec": -float(index),
                    "caption_ra": float(index),
                    "caption_dec": -float(index),
                },
                "separation_arcsec": 0.0,
                "source_group_id": f"group-{index}",
                "split": split,
                "acceptance_status": "accepted",
                "caption": {
                    "original": captions[index],
                    "summary_original": summaries[index],
                    "status": "released_reused",
                },
                "caption_source": {
                    "repo_id": CAPTION_REPO_ID,
                    "revision": CAPTION_REVISION,
                    "shard": "data/captions.parquet",
                    "row": index,
                    "sha256": caption_hash,
                    "survey": "legacy_dr10",
                },
                "paired_source": {
                    "repo_id": PAIRED_REPO_ID,
                    "revision": PAIRED_REVISION,
                    "shard": "data/paired.parquet",
                    "row": index,
                    "sha256": paired_hash,
                },
            }
        )

    candidates = tmp_path / "accepted/candidates.jsonl"
    _write_jsonl(candidates, accepted)
    roster = tmp_path / "roster.csv.gz"
    with gzip.open(roster, "wt", encoding="utf-8", newline="") as handle:
        handle.write("candidate_index\n0\n1\n2\n")
    roster_hash = sha256_file(roster)
    accepted_manifest = tmp_path / "accepted/manifest.json"
    _write_json(
        accepted_manifest,
        {
            "outputs": {
                "candidates": {
                    "path": "candidates.jsonl",
                    "row_count": 3,
                    "sha256": sha256_file(candidates),
                }
            },
            "pins": {"roster_sha256": roster_hash},
            "splits": {
                "row_counts": {"train": 1, "validation": 1, "test": 1},
                "group_count": 3,
                "policy": "fixture-whole-group",
            },
        },
    )
    image_manifest = _raw_fixture(
        tmp_path / "raw_images", accepted, IMAGE_ARRAY_CONTRACT, spectra=False
    )
    spectrum_manifest = _raw_fixture(
        tmp_path / "raw_spectra", accepted, SPECTRUM_ARRAY_CONTRACT, spectra=True
    )
    spectrum_bundle = _spectrum_bundle(tmp_path / "spectrum_accepted", accepted, [0, 1, 2])
    return HFDatasetConfig(
        output_dir=tmp_path / "upload",
        accepted_candidates_path=candidates,
        accepted_manifest_path=accepted_manifest,
        raw_images_manifest_path=image_manifest,
        raw_spectra_manifest_path=spectrum_manifest,
        caption_root=caption_root,
        paired_root=paired_root,
        spectrum_caption_bundle=spectrum_bundle,
        roster_path=roster,
        expected_count=3,
        expected_split_counts={"train": 1, "validation": 1, "test": 1},
        expected_roster_sha256=roster_hash,
        link_mode="copy",
    )


def test_stages_portable_arrays_and_two_three_split_configs(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    summary = prepare_hf_dataset(config)
    output = config.output_dir

    assert summary["configs"] == {
        "image_caption": {"train": 1, "validation": 1, "test": 1},
        "spectrum_caption": {"train": 1, "validation": 1, "test": 1},
    }
    assert (output / "assets/raw/images/image_array.npy").is_file()
    assert (output / "assets/raw/spectra/spectrum_flux.npy").is_file()
    assert not list(output.rglob("*embedding*.npy"))
    roster = [
        json.loads(line) for line in (output / "metadata/candidates.jsonl").read_text().splitlines()
    ]
    assert [row["candidate_index"] for row in roster] == [0, 1, 2]
    assert all(row["raw_assets"]["array_index"] == row["candidate_index"] for row in roster)
    assert all("image_embedding" not in row for row in roster)

    image_features = Features.from_arrow_schema(
        pq.read_schema(output / "data/image_caption/train.parquet")
    )
    spectrum_features = Features.from_arrow_schema(
        pq.read_schema(output / "data/spectrum_caption/train.parquet")
    )
    assert isinstance(image_features["image"], Image)
    assert image_features["image"].decode is False
    assert isinstance(spectrum_features["spectrum_plot"], Image)
    assert spectrum_features["spectrum_plot"].decode is False

    spectrum_rows = {
        split: pq.read_table(output / f"data/spectrum_caption/{split}.parquet").to_pylist()[0]
        for split in ("train", "validation", "test")
    }
    assert all(row["caption_status"] == "accepted" for row in spectrum_rows.values())
    assert all(row["review_status"] == "accepted" for row in spectrum_rows.values())
    assert all(row["reviewed"] is True for row in spectrum_rows.values())
    assert all(row["dataset_ready"] is True for row in spectrum_rows.values())
    assert spectrum_rows["train"]["role"] == "alignment_training"
    assert spectrum_rows["train"]["training_allowed"] is True
    for split in ("validation", "test"):
        assert spectrum_rows[split]["role"] == "held_out_evaluation"
        assert spectrum_rows[split]["training_allowed"] is False
    assert not (output / "data/spectrum_caption_inspection").exists()
    assert "spectrum_caption_inspection" not in (output / "README.md").read_text()
    assert (output / "MANIFEST.json").is_file()
    assert (output / "SHA256SUMS").is_file()

    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            text = path.read_text(encoding="utf-8")
            assert "/lus/" not in text
            assert "_local/" not in text
            assert "BaseMM_PRISM" not in text


@pytest.mark.parametrize(
    ("section", "field", "value", "index", "match"),
    [
        ("spectrum_caption", "status", "pending_review", 0, "not accepted"),
        ("review", "status", "pending_review", 0, "lacks an accepted"),
        ("training_policy", "dataset_ready", False, 0, "not dataset_ready"),
        (
            "training_policy",
            "gradient_training_allowed",
            True,
            1,
            "does not match",
        ),
    ],
)
def test_rejects_invalid_spectrum_promotion(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    index: int,
    match: str,
) -> None:
    config = _complete_fixture(tmp_path)
    bundle = config.spectrum_caption_bundle
    captions_path = bundle / "captions.jsonl"
    rows = [json.loads(line) for line in captions_path.read_text().splitlines()]
    rows[index][section][field] = value
    _write_jsonl(captions_path, rows)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["outputs"]["captions.jsonl"]
    entry["sha256"] = sha256_file(captions_path)
    entry["size_bytes"] = captions_path.stat().st_size
    _write_json(manifest_path, manifest)

    with pytest.raises(PackagingValidationError, match=match):
        prepare_hf_dataset(config)


def test_rejects_source_group_split_leakage(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    rows = [json.loads(line) for line in config.accepted_candidates_path.read_text().splitlines()]
    rows[1]["source_group_id"] = rows[0]["source_group_id"]

    with pytest.raises(PackagingValidationError, match="source groups cross"):
        _validate_accepted_rows(
            rows,
            expected_count=3,
            expected_split_counts={"train": 1, "validation": 1, "test": 1},
            caption_repo_id=CAPTION_REPO_ID,
            caption_revision=CAPTION_REVISION,
            paired_repo_id=PAIRED_REPO_ID,
            paired_revision=PAIRED_REVISION,
        )


def test_dataset_card_documents_three_split_spectrum_policy() -> None:
    card = _dataset_card(split_counts={"train": 516, "validation": 65, "test": 65})

    assert "config_name: image_caption" in card
    assert "config_name: spectrum_caption" in card
    assert "data/spectrum_caption/validation.parquet" in card
    assert "data/spectrum_caption/test.parquet" in card
    assert "config_name: spectrum_caption_inspection" not in card
    assert "`reviewed=true`" in card
    assert "`dataset_ready=true`" in card

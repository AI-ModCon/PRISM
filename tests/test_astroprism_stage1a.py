from __future__ import annotations

import csv
import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from src.data.astroprism_stage1a import (
    IMAGE_EMBEDDING_SHAPE,
    Stage1AConfig,
    Stage1AValidationError,
    prepare_stage1a,
    sha256_file,
)

_ROSTER_FIELDS = [
    "paired_targetid",
    "caption_object_id",
    "paired_ra",
    "paired_dec",
    "caption_ra",
    "caption_dec",
    "separation_arcsec",
    "paired_candidate_count",
    "caption_candidate_count",
    "one_to_one_row_candidate",
    "unique_source_ids",
    "provisional_unambiguous_candidate",
    "paired_shard",
    "paired_row",
    "caption_shard",
    "caption_row",
]
_PAIRED_REVISION = "paired-test-revision"
_CAPTION_REVISION = "caption-test-revision"


def _fixed_list_array(values: np.ndarray, dtype: pa.DataType) -> pa.Array:
    rows, outer_size, inner_size = values.shape
    flat = pa.array(values.reshape(-1), type=dtype)
    inner = pa.FixedSizeListArray.from_arrays(flat, inner_size)
    outer = pa.FixedSizeListArray.from_arrays(inner, outer_size)
    assert len(outer) == rows
    return outer


def _write_metadata(root: Path, shard: str, revision: str) -> None:
    source = root / shard
    metadata = root / ".cache" / "huggingface" / "download" / f"{shard}.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"{revision}\n{sha256_file(source)}\n0\n", encoding="utf-8")


def _build_case(
    tmp_path: Path,
    *,
    paired_ids: list[str] | None = None,
    surveys: list[str] | None = None,
    caption_ras: list[float] | None = None,
    embedding_dtype: np.dtype = np.dtype("float16"),  # noqa: B008
    embedding_shape: tuple[int, int] = IMAGE_EMBEDDING_SHAPE,
    nonfinite: bool = False,
) -> Stage1AConfig:
    paired_root = tmp_path / "paired"
    caption_root = tmp_path / "captions"
    paired_shard = "data/paired.parquet"
    caption_shard = "data/captions.parquet"
    (paired_root / "data").mkdir(parents=True)
    (caption_root / "data").mkdir(parents=True)

    canonical_paired_ids = ["9007199254740993", "9007199254740995"]
    canonical_caption_ids = ["obj-a", "obj-b"]
    ras = [10.0, 30.0]
    decs = [20.0, -10.0]
    source_caption_ras = caption_ras or ras

    embeddings = np.empty((2, *embedding_shape), dtype=embedding_dtype)
    embeddings[0].fill(1.0)
    embeddings[1].fill(2.0)
    if nonfinite:
        embeddings[1, 0, 0] = np.nan
    arrow_dtype = pa.from_numpy_dtype(embedding_dtype)
    paired_table = pa.table(
        {
            "object_id_provabgs": paired_ids or canonical_paired_ids,
            "ra_provabgs": pa.array(ras, type=pa.float64()),
            "dec_provabgs": pa.array(decs, type=pa.float64()),
            "embedding_image": _fixed_list_array(embeddings, arrow_dtype),
            # Deliberately present: extraction must not copy this column.
            "embedding_spectrum": pa.array([[901.0], [902.0]], type=pa.list_(pa.float16())),
        }
    )
    pq.write_table(paired_table, paired_root / paired_shard)

    caption_table = pa.table(
        {
            "object_id": canonical_caption_ids,
            "survey": surveys or ["legacy_dr10", "legacy_dr10"],
            "ra": pa.array(source_caption_ras, type=pa.float64()),
            "dec": pa.array(decs, type=pa.float64()),
            "caption": ["  original caption A\n", "original caption B"],
            "summary": ["summary A", "summary B"],
        }
    )
    pq.write_table(caption_table, caption_root / caption_shard)
    _write_metadata(paired_root, paired_shard, _PAIRED_REVISION)
    _write_metadata(caption_root, caption_shard, _CAPTION_REVISION)

    # Roster order intentionally reverses both source tables.
    roster_rows = []
    for source_row in (1, 0):
        roster_rows.append(
            {
                "paired_targetid": canonical_paired_ids[source_row],
                "caption_object_id": canonical_caption_ids[source_row],
                "paired_ra": str(ras[source_row]),
                "paired_dec": str(decs[source_row]),
                "caption_ra": str(ras[source_row]),
                "caption_dec": str(decs[source_row]),
                "separation_arcsec": "0.0",
                "paired_candidate_count": "1",
                "caption_candidate_count": "1",
                "one_to_one_row_candidate": "True",
                "unique_source_ids": "True",
                "provisional_unambiguous_candidate": "True",
                "paired_shard": paired_shard,
                "paired_row": str(source_row),
                "caption_shard": caption_shard,
                "caption_row": str(source_row),
            }
        )
    roster = tmp_path / "roster.csv.gz"
    with gzip.open(roster, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ROSTER_FIELDS)
        writer.writeheader()
        writer.writerows(roster_rows)

    return Stage1AConfig(
        roster_path=roster,
        paired_root=paired_root,
        caption_root=caption_root,
        output_dir=tmp_path / "output",
        expected_count=2,
        expected_roster_sha256=sha256_file(roster),
        paired_revision=_PAIRED_REVISION,
        caption_revision=_CAPTION_REVISION,
        batch_size=1,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_extracts_image_only_in_deterministic_roster_order(tmp_path: Path):
    config = _build_case(tmp_path)

    summary = prepare_stage1a(config)

    assert summary["status"] == "prepared"
    embeddings = np.load(config.output_dir / "image_embeddings.f16.npy", mmap_mode="r")
    assert embeddings.shape == (2, *IMAGE_EMBEDDING_SHAPE)
    assert embeddings.dtype == np.float16
    assert np.all(embeddings[0] == np.float16(2.0))
    assert np.all(embeddings[1] == np.float16(1.0))

    records = _read_jsonl(config.output_dir / "candidates.jsonl")
    assert [record["candidate_index"] for record in records] == [0, 1]
    assert [record["paired_targetid"] for record in records] == [
        "9007199254740995",
        "9007199254740993",
    ]
    assert records[1]["caption"]["original"] == "  original caption A\n"
    assert all(record["acceptance_status"] == "pending_review" for record in records)
    assert all(record["source_group_id"] is None for record in records)
    assert all(record["split"] is None for record in records)
    assert "embedding_spectrum" not in json.dumps(records)

    manifest = json.loads((config.output_dir / "manifest.json").read_text())
    assert manifest["artifact_type"] == "astroprism_aion_feature_parity_reference"
    assert manifest["role"] == "parity_reference_only"
    assert manifest["spectra_included"] is False
    assert manifest["outputs"]["image_embeddings"]["sha256"] == sha256_file(
        config.output_dir / "image_embeddings.f16.npy"
    )
    assert manifest["sources"]["paired"][0]["sha256"] == sha256_file(
        config.paired_root / "data/paired.parquet"
    )


def test_validate_only_writes_nothing(tmp_path: Path):
    config = replace(_build_case(tmp_path), validate_only=True)

    summary = prepare_stage1a(config)

    assert summary["status"] == "validated"
    assert summary["spectra_included"] is False
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    ("case_kwargs", "message"),
    [
        ({"paired_ids": ["9007199254740993", "wrong-id"]}, "paired ID mismatch"),
        ({"surveys": ["legacy_dr10", "hsc"]}, "survey must be"),
        ({"caption_ras": [10.0, 30.01]}, "caption_ra mismatch"),
    ],
)
def test_rejects_identity_survey_and_coordinate_mismatches(
    tmp_path: Path, case_kwargs: dict, message: str
):
    config = _build_case(tmp_path, **case_kwargs)

    with pytest.raises(Stage1AValidationError, match=message):
        prepare_stage1a(config)

    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


@pytest.mark.parametrize(
    "case_kwargs",
    [
        {"embedding_dtype": np.dtype("float32")},
        {"embedding_shape": (575, 768)},
    ],
)
def test_rejects_wrong_embedding_schema(tmp_path: Path, case_kwargs: dict):
    config = _build_case(tmp_path, **case_kwargs)

    with pytest.raises(Stage1AValidationError, match="embedding_image must be"):
        prepare_stage1a(config)


def test_rejects_nonfinite_embedding(tmp_path: Path):
    config = _build_case(tmp_path, nonfinite=True)

    with pytest.raises(Stage1AValidationError, match="non-finite"):
        prepare_stage1a(config)


def test_source_hash_is_verified(tmp_path: Path):
    config = _build_case(tmp_path)
    with (config.paired_root / "data/paired.parquet").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(Stage1AValidationError, match="source SHA-256 mismatch"):
        prepare_stage1a(config)


def test_existing_output_requires_overwrite(tmp_path: Path):
    config = _build_case(tmp_path)
    prepare_stage1a(config)
    stale = config.output_dir / "stale"
    stale.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        prepare_stage1a(config)
    assert stale.exists()

    prepare_stage1a(replace(config, overwrite=True))
    assert not stale.exists()
    assert (config.output_dir / "manifest.json").is_file()

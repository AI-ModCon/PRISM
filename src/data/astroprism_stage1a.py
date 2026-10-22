"""Prepare the pinned AstroPRISM AION feature parity reference.

This module is intentionally offline.  It validates the already-downloaded
Hugging Face Parquet shards against their download metadata, selects rows by
the locators in the committed roster, and writes released AION image embeddings
only as a parity/regression oracle.  These precomputed features are not Stage 1A
model inputs: training starts from raw calibrated cutouts and runs the pinned
AION codec/shared encoder.  Released spectrum embeddings are never requested.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROSTER_SHA256 = "650935faa58953c9929e7379429bfb332a7486dab54544fedef64ca385abb3bc"
PAIRED_REPO_ID = "EiffL/desi_x_ls_aion"
PAIRED_REVISION = "57f7c7575260811ea076163bc743787d75b9aa49"
CAPTION_REPO_ID = "astronolan/galaxy-descriptions"
CAPTION_REVISION = "6890dc0c8fc793867c07a6ce4fce11e51c167d6e"
EXPECTED_CANDIDATES = 646
IMAGE_EMBEDDING_SHAPE = (576, 768)

_PAIRED_COLUMNS = (
    "object_id_provabgs",
    "ra_provabgs",
    "dec_provabgs",
    "embedding_image",
)
_CAPTION_COLUMNS = ("object_id", "survey", "ra", "dec", "caption", "summary")
_REQUIRED_ROSTER_COLUMNS = {
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
}


class Stage1AValidationError(ValueError):
    """Raised when a source asset violates the pinned Stage 1A contract."""


@dataclass(frozen=True)
class Stage1AConfig:
    """Inputs and validation pins for one extraction."""

    roster_path: Path
    paired_root: Path
    caption_root: Path
    output_dir: Path
    validate_only: bool = False
    overwrite: bool = False
    batch_size: int = 16
    expected_count: int = EXPECTED_CANDIDATES
    expected_roster_sha256: str | None = ROSTER_SHA256
    paired_revision: str = PAIRED_REVISION
    caption_revision: str = CAPTION_REVISION
    coordinate_tolerance_deg: float = 1e-9
    separation_tolerance_arcsec: float = 1e-6


@dataclass(frozen=True)
class _SourceFile:
    kind: str
    repo_id: str
    revision: str
    shard: str
    path: Path
    size_bytes: int
    sha256: str
    metadata_sha256: str

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "download_metadata_sha256": self.metadata_sha256,
        }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _open_roster(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _parse_int(value: str, *, field: str, index: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Stage1AValidationError(
            f"roster row {index}: {field} must be an integer, got {value!r}"
        ) from exc
    return parsed


def _parse_float(value: str, *, field: str, index: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage1AValidationError(
            f"roster row {index}: {field} must be a float, got {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise Stage1AValidationError(f"roster row {index}: {field} is not finite")
    return parsed


def _parse_true(value: str, *, field: str, index: int) -> None:
    if str(value).strip().lower() not in {"true", "1"}:
        raise Stage1AValidationError(f"roster row {index}: {field} must be true, got {value!r}")


def _read_roster(config: Stage1AConfig) -> tuple[list[dict[str, str]], str]:
    path = Path(config.roster_path)
    if not path.is_file():
        raise Stage1AValidationError(f"roster does not exist: {path}")

    roster_sha256 = sha256_file(path)
    if config.expected_roster_sha256 is not None and roster_sha256 != config.expected_roster_sha256:
        raise Stage1AValidationError(
            "roster SHA-256 mismatch: "
            f"expected {config.expected_roster_sha256}, got {roster_sha256}"
        )

    with _open_roster(path) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise Stage1AValidationError("roster contains duplicate column names")
        missing = sorted(_REQUIRED_ROSTER_COLUMNS - set(fieldnames))
        if missing:
            raise Stage1AValidationError(f"roster is missing required columns: {missing}")
        rows = list(reader)

    if len(rows) != config.expected_count:
        raise Stage1AValidationError(
            f"roster must contain exactly {config.expected_count} rows, got {len(rows)}"
        )

    paired_ids: set[str] = set()
    caption_ids: set[str] = set()
    paired_locators: set[tuple[str, int]] = set()
    caption_locators: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        paired_id = row["paired_targetid"]
        caption_id = row["caption_object_id"]
        if not paired_id or not caption_id:
            raise Stage1AValidationError(f"roster row {index}: identifiers must be nonempty")
        if paired_id in paired_ids:
            raise Stage1AValidationError(f"duplicate paired_targetid: {paired_id}")
        if caption_id in caption_ids:
            raise Stage1AValidationError(f"duplicate caption_object_id: {caption_id}")
        paired_ids.add(paired_id)
        caption_ids.add(caption_id)

        paired_row = _parse_int(row["paired_row"], field="paired_row", index=index)
        caption_row = _parse_int(row["caption_row"], field="caption_row", index=index)
        if paired_row < 0 or caption_row < 0:
            raise Stage1AValidationError(f"roster row {index}: source rows must be nonnegative")
        paired_locator = (row["paired_shard"], paired_row)
        caption_locator = (row["caption_shard"], caption_row)
        if paired_locator in paired_locators:
            raise Stage1AValidationError(f"duplicate paired locator: {paired_locator}")
        if caption_locator in caption_locators:
            raise Stage1AValidationError(f"duplicate caption locator: {caption_locator}")
        paired_locators.add(paired_locator)
        caption_locators.add(caption_locator)

        for field in ("paired_candidate_count", "caption_candidate_count"):
            if _parse_int(row[field], field=field, index=index) != 1:
                raise Stage1AValidationError(f"roster row {index}: {field} must equal 1")
        for field in (
            "one_to_one_row_candidate",
            "unique_source_ids",
            "provisional_unambiguous_candidate",
        ):
            _parse_true(row[field], field=field, index=index)

        paired_ra = _parse_float(row["paired_ra"], field="paired_ra", index=index)
        paired_dec = _parse_float(row["paired_dec"], field="paired_dec", index=index)
        caption_ra = _parse_float(row["caption_ra"], field="caption_ra", index=index)
        caption_dec = _parse_float(row["caption_dec"], field="caption_dec", index=index)
        separation = _parse_float(row["separation_arcsec"], field="separation_arcsec", index=index)
        if not 0.0 <= paired_ra < 360.0 or not 0.0 <= caption_ra < 360.0:
            raise Stage1AValidationError(f"roster row {index}: RA is outside [0, 360)")
        if not -90.0 <= paired_dec <= 90.0 or not -90.0 <= caption_dec <= 90.0:
            raise Stage1AValidationError(f"roster row {index}: declination is outside [-90, 90]")
        if not 0.0 <= separation <= 1.0:
            raise Stage1AValidationError(f"roster row {index}: separation_arcsec is outside [0, 1]")

    return rows, roster_sha256


def _safe_source_path(root: Path, shard: str) -> Path:
    pure = PurePosixPath(shard)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise Stage1AValidationError(f"unsafe source shard path: {shard!r}")
    root = root.resolve()
    path = (root / Path(*pure.parts)).resolve()
    if not path.is_relative_to(root):
        raise Stage1AValidationError(f"source shard escapes its root: {shard!r}")
    if not path.is_file():
        raise Stage1AValidationError(f"source shard does not exist: {path}")
    return path


def _source_file(*, kind: str, repo_id: str, revision: str, root: Path, shard: str) -> _SourceFile:
    root = Path(root)
    path = _safe_source_path(root, shard)
    metadata_path = root / ".cache" / "huggingface" / "download" / f"{shard}.metadata"
    if not metadata_path.is_file():
        raise Stage1AValidationError(f"missing Hugging Face download metadata: {metadata_path}")
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise Stage1AValidationError(f"malformed Hugging Face download metadata: {metadata_path}")
    observed_revision, expected_sha256 = lines[0].strip(), lines[1].strip().lower()
    if observed_revision != revision:
        raise Stage1AValidationError(
            f"revision mismatch for {shard}: expected {revision}, got {observed_revision}"
        )
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise Stage1AValidationError(
            f"download metadata for {shard} does not contain an LFS SHA-256"
        )
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise Stage1AValidationError(
            f"source SHA-256 mismatch for {shard}: expected {expected_sha256}, "
            f"got {observed_sha256}"
        )
    return _SourceFile(
        kind=kind,
        repo_id=repo_id,
        revision=revision,
        shard=shard,
        path=path,
        size_bytes=path.stat().st_size,
        sha256=observed_sha256,
        metadata_sha256=sha256_file(metadata_path),
    )


def _collect_sources(
    config: Stage1AConfig, rows: list[dict[str, str]]
) -> tuple[dict[str, _SourceFile], dict[str, _SourceFile]]:
    paired = {
        shard: _source_file(
            kind="paired",
            repo_id=PAIRED_REPO_ID,
            revision=config.paired_revision,
            root=Path(config.paired_root),
            shard=shard,
        )
        for shard in sorted({row["paired_shard"] for row in rows})
    }
    captions = {
        shard: _source_file(
            kind="caption",
            repo_id=CAPTION_REPO_ID,
            revision=config.caption_revision,
            root=Path(config.caption_root),
            shard=shard,
        )
        for shard in sorted({row["caption_shard"] for row in rows})
    }
    return paired, captions


def _require_columns(schema: pa.Schema, columns: tuple[str, ...], *, shard: str) -> None:
    missing = [name for name in columns if schema.get_field_index(name) < 0]
    if missing:
        raise Stage1AValidationError(f"{shard}: missing Parquet columns {missing}")


def _check_coordinate(
    actual: Any,
    expected_text: str,
    *,
    field: str,
    candidate_index: int,
    tolerance: float,
) -> float:
    if actual is None:
        raise Stage1AValidationError(f"candidate {candidate_index}: source {field} is null")
    value = float(actual)
    expected = float(expected_text)
    if not math.isfinite(value) or abs(value - expected) > tolerance:
        raise Stage1AValidationError(
            f"candidate {candidate_index}: {field} mismatch; expected {expected}, got {value}"
        )
    return value


def _fixed_embedding_type() -> pa.DataType:
    return pa.list_(pa.list_(pa.float16(), IMAGE_EMBEDDING_SHAPE[1]), IMAGE_EMBEDDING_SHAPE[0])


def _extract_embedding(array: pa.Array, local_index: int, *, candidate_index: int) -> np.ndarray:
    selected = array.slice(local_index, 1)
    if selected.null_count:
        raise Stage1AValidationError(f"candidate {candidate_index}: image embedding is null")
    flat = pc.list_flatten(pc.list_flatten(selected))
    if flat.null_count:
        raise Stage1AValidationError(
            f"candidate {candidate_index}: image embedding contains null values"
        )
    values = flat.to_numpy(zero_copy_only=False)
    if values.dtype != np.float16 or values.size != math.prod(IMAGE_EMBEDDING_SHAPE):
        raise Stage1AValidationError(
            f"candidate {candidate_index}: invalid image embedding buffer "
            f"({values.dtype}, {values.size} values)"
        )
    embedding = values.reshape(IMAGE_EMBEDDING_SHAPE)
    if not np.isfinite(embedding).all():
        raise Stage1AValidationError(
            f"candidate {candidate_index}: image embedding contains non-finite values"
        )
    if not np.any(embedding):
        raise Stage1AValidationError(f"candidate {candidate_index}: image embedding is all zero")
    return embedding


def _scan_paired(
    config: Stage1AConfig,
    rows: list[dict[str, str]],
    sources: dict[str, _SourceFile],
    output: np.memmap | None,
) -> tuple[list[dict[str, Any]], list[float]]:
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["paired_shard"]].append((index, row))

    found: list[dict[str, Any] | None] = [None] * len(rows)
    norms: list[float] = []
    for shard in sorted(grouped):
        source = sources[shard]
        parquet = pq.ParquetFile(source.path)
        _require_columns(parquet.schema_arrow, _PAIRED_COLUMNS, shard=shard)
        embedding_type = parquet.schema_arrow.field("embedding_image").type
        if embedding_type != _fixed_embedding_type():
            raise Stage1AValidationError(
                f"{shard}: embedding_image must be float16{IMAGE_EMBEDDING_SHAPE}, "
                f"got {embedding_type}"
            )
        if not pa.types.is_string(parquet.schema_arrow.field("object_id_provabgs").type):
            raise Stage1AValidationError(f"{shard}: object_id_provabgs must be a string")

        wanted = sorted(
            (
                _parse_int(row["paired_row"], field="paired_row", index=index),
                index,
                row,
            )
            for index, row in grouped[shard]
        )
        if wanted[-1][0] >= parquet.metadata.num_rows:
            raise Stage1AValidationError(
                f"{shard}: paired row {wanted[-1][0]} is outside {parquet.metadata.num_rows} rows"
            )

        pointer = 0
        row_offset = 0
        for batch in parquet.iter_batches(
            batch_size=config.batch_size, columns=list(_PAIRED_COLUMNS), use_threads=False
        ):
            batch_end = row_offset + len(batch)
            while pointer < len(wanted) and wanted[pointer][0] < batch_end:
                source_row, candidate_index, roster_row = wanted[pointer]
                if source_row < row_offset:
                    raise Stage1AValidationError(f"{shard}: duplicate or unordered row locator")
                local_index = source_row - row_offset
                source_id = batch.column(0)[local_index].as_py()
                if source_id != roster_row["paired_targetid"]:
                    raise Stage1AValidationError(
                        f"candidate {candidate_index}: paired ID mismatch; expected "
                        f"{roster_row['paired_targetid']!r}, got {source_id!r}"
                    )
                ra = _check_coordinate(
                    batch.column(1)[local_index].as_py(),
                    roster_row["paired_ra"],
                    field="paired_ra",
                    candidate_index=candidate_index,
                    tolerance=config.coordinate_tolerance_deg,
                )
                dec = _check_coordinate(
                    batch.column(2)[local_index].as_py(),
                    roster_row["paired_dec"],
                    field="paired_dec",
                    candidate_index=candidate_index,
                    tolerance=config.coordinate_tolerance_deg,
                )
                embedding = _extract_embedding(
                    batch.column(3), local_index, candidate_index=candidate_index
                )
                if output is not None:
                    output[candidate_index] = embedding
                norms.append(float(np.linalg.norm(embedding.astype(np.float32))))
                found[candidate_index] = {"ra": ra, "dec": dec}
                pointer += 1
            row_offset = batch_end
            if pointer == len(wanted):
                break
        if pointer != len(wanted):
            raise Stage1AValidationError(f"{shard}: not all paired locators were found")

    if any(item is None for item in found):
        raise Stage1AValidationError("not every roster candidate produced an image embedding")
    return [item for item in found if item is not None], norms


def _scan_captions(
    config: Stage1AConfig,
    rows: list[dict[str, str]],
    sources: dict[str, _SourceFile],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["caption_shard"]].append((index, row))

    found: list[dict[str, Any] | None] = [None] * len(rows)
    for shard in sorted(grouped):
        source = sources[shard]
        parquet = pq.ParquetFile(source.path)
        _require_columns(parquet.schema_arrow, _CAPTION_COLUMNS, shard=shard)
        for field in ("object_id", "survey", "caption", "summary"):
            if not pa.types.is_string(parquet.schema_arrow.field(field).type):
                raise Stage1AValidationError(f"{shard}: {field} must be a string")

        wanted = sorted(
            (
                _parse_int(row["caption_row"], field="caption_row", index=index),
                index,
                row,
            )
            for index, row in grouped[shard]
        )
        if wanted[-1][0] >= parquet.metadata.num_rows:
            raise Stage1AValidationError(
                f"{shard}: caption row {wanted[-1][0]} is outside {parquet.metadata.num_rows} rows"
            )

        pointer = 0
        row_offset = 0
        for batch in parquet.iter_batches(
            batch_size=max(config.batch_size, 128),
            columns=list(_CAPTION_COLUMNS),
            use_threads=False,
        ):
            batch_end = row_offset + len(batch)
            while pointer < len(wanted) and wanted[pointer][0] < batch_end:
                source_row, candidate_index, roster_row = wanted[pointer]
                if source_row < row_offset:
                    raise Stage1AValidationError(f"{shard}: duplicate or unordered row locator")
                local_index = source_row - row_offset
                source_id = batch.column(0)[local_index].as_py()
                survey = batch.column(1)[local_index].as_py()
                if source_id != roster_row["caption_object_id"]:
                    raise Stage1AValidationError(
                        f"candidate {candidate_index}: caption ID mismatch; expected "
                        f"{roster_row['caption_object_id']!r}, got {source_id!r}"
                    )
                if survey != "legacy_dr10":
                    raise Stage1AValidationError(
                        f"candidate {candidate_index}: survey must be 'legacy_dr10', got {survey!r}"
                    )
                ra = _check_coordinate(
                    batch.column(2)[local_index].as_py(),
                    roster_row["caption_ra"],
                    field="caption_ra",
                    candidate_index=candidate_index,
                    tolerance=config.coordinate_tolerance_deg,
                )
                dec = _check_coordinate(
                    batch.column(3)[local_index].as_py(),
                    roster_row["caption_dec"],
                    field="caption_dec",
                    candidate_index=candidate_index,
                    tolerance=config.coordinate_tolerance_deg,
                )
                caption = batch.column(4)[local_index].as_py()
                summary = batch.column(5)[local_index].as_py()
                if not isinstance(caption, str) or not caption.strip():
                    raise Stage1AValidationError(
                        f"candidate {candidate_index}: caption is null or empty"
                    )
                if not isinstance(summary, str) or not summary.strip():
                    raise Stage1AValidationError(
                        f"candidate {candidate_index}: summary is null or empty"
                    )
                found[candidate_index] = {
                    "ra": ra,
                    "dec": dec,
                    "caption": caption,
                    "summary": summary,
                }
                pointer += 1
            row_offset = batch_end
            if pointer == len(wanted):
                break
        if pointer != len(wanted):
            raise Stage1AValidationError(f"{shard}: not all caption locators were found")

    if any(item is None for item in found):
        raise Stage1AValidationError("not every roster candidate produced a caption")
    return [item for item in found if item is not None]


def _angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1_rad, dec1_rad, ra2_rad, dec2_rad = map(math.radians, (ra1, dec1, ra2, dec2))
    sin_dec = math.sin((dec2_rad - dec1_rad) / 2.0)
    sin_ra = math.sin((ra2_rad - ra1_rad) / 2.0)
    haversine = sin_dec**2 + math.cos(dec1_rad) * math.cos(dec2_rad) * sin_ra**2
    haversine = min(1.0, max(0.0, haversine))
    return math.degrees(2.0 * math.asin(math.sqrt(haversine))) * 3600.0


def _candidate_records(
    config: Stage1AConfig,
    rows: list[dict[str, str]],
    paired_values: list[dict[str, Any]],
    caption_values: list[dict[str, Any]],
    paired_sources: dict[str, _SourceFile],
    caption_sources: dict[str, _SourceFile],
) -> list[dict[str, Any]]:
    records = []
    for index, (row, paired, caption) in enumerate(
        zip(rows, paired_values, caption_values, strict=False)
    ):
        separation = _angular_separation_arcsec(
            paired["ra"], paired["dec"], caption["ra"], caption["dec"]
        )
        recorded_separation = float(row["separation_arcsec"])
        if separation > 1.0 + config.separation_tolerance_arcsec:
            raise Stage1AValidationError(
                f"candidate {index}: source separation exceeds 1 arcsec ({separation})"
            )
        if abs(separation - recorded_separation) > config.separation_tolerance_arcsec:
            raise Stage1AValidationError(
                f"candidate {index}: separation mismatch; expected {recorded_separation}, "
                f"got {separation}"
            )

        paired_source = paired_sources[row["paired_shard"]]
        caption_source = caption_sources[row["caption_shard"]]
        records.append(
            {
                "schema_version": 1,
                "candidate_index": index,
                "paired_targetid": row["paired_targetid"],
                "caption_object_id": row["caption_object_id"],
                "coordinates_deg": {
                    "paired_ra": paired["ra"],
                    "paired_dec": paired["dec"],
                    "caption_ra": caption["ra"],
                    "caption_dec": caption["dec"],
                },
                "separation_arcsec": separation,
                "matching": {
                    "paired_candidate_count": 1,
                    "caption_candidate_count": 1,
                    "one_to_one_row_candidate": True,
                    "unique_source_ids": True,
                    "provisional_unambiguous_candidate": True,
                },
                "paired_source": {
                    "repo_id": PAIRED_REPO_ID,
                    "revision": config.paired_revision,
                    "shard": row["paired_shard"],
                    "row": int(row["paired_row"]),
                    "sha256": paired_source.sha256,
                },
                "caption_source": {
                    "repo_id": CAPTION_REPO_ID,
                    "revision": config.caption_revision,
                    "shard": row["caption_shard"],
                    "row": int(row["caption_row"]),
                    "sha256": caption_source.sha256,
                    "survey": "legacy_dr10",
                },
                "image_embedding": {
                    "origin": "precomputed_release",
                    "file": "image_embeddings.f16.npy",
                    "index": index,
                    "dtype": "float16",
                    "shape": list(IMAGE_EMBEDDING_SHAPE),
                },
                "caption": {
                    "original": caption["caption"],
                    "summary_original": caption["summary"],
                    "status": "unreviewed",
                },
                "acceptance_status": "pending_review",
                "source_group_id": None,
                "split": None,
            }
        )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _commit_staging(staging: Path, output: Path, *, overwrite: bool) -> None:
    if not _path_exists(output):
        os.replace(staging, output)
        return
    if not overwrite:
        raise FileExistsError(f"output already exists: {output}")

    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    if backup.is_dir() and not backup.is_symlink():
        shutil.rmtree(backup)
    else:
        backup.unlink(missing_ok=True)


def prepare_stage1a(config: Stage1AConfig) -> dict[str, Any]:
    """Validate inputs and optionally write the Stage 1A candidate artifact.

    The returned dictionary is a concise summary suitable for CLI output.
    ``validate_only`` performs the full checksum/schema/content scan without
    creating or modifying the output directory.
    """

    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")

    output_dir = Path(config.output_dir)
    if not config.validate_only and _path_exists(output_dir) and not config.overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")

    rows, roster_sha256 = _read_roster(config)
    paired_sources, caption_sources = _collect_sources(config, rows)

    staging: Path | None = None
    embedding_output: np.memmap | None = None
    try:
        if not config.validate_only:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
            )
            embedding_output = np.lib.format.open_memmap(
                staging / "image_embeddings.f16.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(rows), *IMAGE_EMBEDDING_SHAPE),
            )

        paired_values, norms = _scan_paired(config, rows, paired_sources, embedding_output)
        caption_values = _scan_captions(config, rows, caption_sources)
        records = _candidate_records(
            config,
            rows,
            paired_values,
            caption_values,
            paired_sources,
            caption_sources,
        )

        if embedding_output is not None:
            embedding_output.flush()
            del embedding_output
            embedding_output = None

        summary = {
            "status": "validated" if config.validate_only else "prepared",
            "candidate_count": len(records),
            "roster_sha256": roster_sha256,
            "paired_revision": config.paired_revision,
            "caption_revision": config.caption_revision,
            "image_embedding_dtype": "float16",
            "image_embedding_shape": [len(records), *IMAGE_EMBEDDING_SHAPE],
            "spectra_included": False,
        }
        if config.validate_only:
            return summary

        assert staging is not None
        candidates_path = staging / "candidates.jsonl"
        embeddings_path = staging / "image_embeddings.f16.npy"
        _write_jsonl(candidates_path, records)
        manifest = {
            "schema_version": 1,
            "artifact_type": "astroprism_aion_feature_parity_reference",
            "role": "parity_reference_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pins": {
                "roster_sha256": roster_sha256,
                "paired": {"repo_id": PAIRED_REPO_ID, "revision": config.paired_revision},
                "captions": {
                    "repo_id": CAPTION_REPO_ID,
                    "revision": config.caption_revision,
                    "config": "default",
                    "split": "default",
                    "survey": "legacy_dr10",
                },
            },
            "roster": {
                "path": str(Path(config.roster_path)),
                "sha256": roster_sha256,
                "row_count": len(rows),
            },
            "sources": {
                "paired": [paired_sources[key].manifest_entry() for key in sorted(paired_sources)],
                "captions": [
                    caption_sources[key].manifest_entry() for key in sorted(caption_sources)
                ],
            },
            "outputs": {
                "candidates": {
                    "path": candidates_path.name,
                    "row_count": len(records),
                    "size_bytes": candidates_path.stat().st_size,
                    "sha256": sha256_file(candidates_path),
                },
                "image_embeddings": {
                    "path": embeddings_path.name,
                    "dtype": "float16",
                    "shape": [len(records), *IMAGE_EMBEDDING_SHAPE],
                    "size_bytes": embeddings_path.stat().st_size,
                    "sha256": sha256_file(embeddings_path),
                },
            },
            "validation": {
                "candidate_count": len(records),
                "coordinate_tolerance_deg": config.coordinate_tolerance_deg,
                "separation_tolerance_arcsec": config.separation_tolerance_arcsec,
                "image_embedding_l2_norm": {
                    "min": min(norms),
                    "median": float(np.median(norms)),
                    "max": max(norms),
                },
                "acceptance_status": "pending_review",
            },
            "spectra_included": False,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _commit_staging(staging, output_dir, overwrite=config.overwrite)
        staging = None
        summary["output_dir"] = str(output_dir)
        summary["manifest"] = str(output_dir / "manifest.json")
        return summary
    finally:
        if embedding_output is not None:
            del embedding_output
        if staging is not None and staging.exists():
            shutil.rmtree(staging)

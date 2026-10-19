"""Select the pinned AstroPRISM raw Legacy Survey image rows.

The official AION tutorial HDF5 is about 121 GB, but its datasets are
contiguous and the server supports HTTP byte ranges.  This module maps the
committed candidate roster through the exact PROVABGS identifier column and
copies only the selected image rows into mmap-friendly NPY files.  Spectrum
datasets are intentionally never read or written.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

import fsspec
import h5py
import numpy as np

RAW_SOURCE_URL = "https://users.flatironinstitute.org/~polymathic/data/provabgs_desi_ls.hdf5"
RAW_SOURCE_ETAG = '"6a62487d-1c2e3b0ad4"'
RAW_SOURCE_LENGTH = 121_034_705_620
ROSTER_SHA256 = "650935faa58953c9929e7379429bfb332a7486dab54544fedef64ca385abb3bc"
EXPECTED_CANDIDATES = 646

_PROVABGS_GROUP = "provabgs_desi_edr_sv3"
_EXPECTED_SOURCE_BANDS = ("des-g", "des-r", "des-i", "des-z")
_AION_BAND_MAPPING = {
    "des-g": "DES-G",
    "des-r": "DES-R",
    "des-i": "DES-I",
    "des-z": "DES-Z",
}
_REQUIRED_ROSTER_COLUMNS = {
    "paired_targetid",
    "caption_object_id",
    "paired_ra",
    "paired_dec",
    "caption_ra",
    "caption_dec",
}


@dataclass(frozen=True)
class _DatasetSpec:
    source_path: str
    filename: str
    dtype: np.dtype[Any]
    tail_shape: tuple[int, ...]
    finite: bool = False


_IMAGE_DATASETS = {
    "image_array": _DatasetSpec(
        "legacysurvey/image_array", "image_array.npy", np.dtype("float32"), (4, 160, 160), True
    ),
    "image_band": _DatasetSpec("legacysurvey/image_band", "image_band.npy", np.dtype("S5"), (4,)),
    "image_ivar": _DatasetSpec(
        "legacysurvey/image_ivar", "image_ivar.npy", np.dtype("float32"), (4, 160, 160), True
    ),
    "image_mask": _DatasetSpec(
        "legacysurvey/image_mask", "image_mask.npy", np.dtype("bool"), (160, 160)
    ),
    "image_scale": _DatasetSpec(
        "legacysurvey/image_scale", "image_scale.npy", np.dtype("float32"), (4,), True
    ),
    "object_mask": _DatasetSpec(
        "legacysurvey/object_mask", "object_mask.npy", np.dtype("uint8"), (160, 160)
    ),
}

_METADATA_DATASETS = {
    "provabgs_id": (f"{_PROVABGS_GROUP}/object_id", np.dtype("int64")),
    "provabgs_ra": (f"{_PROVABGS_GROUP}/ra", np.dtype("float64")),
    "provabgs_dec": (f"{_PROVABGS_GROUP}/dec", np.dtype("float64")),
    "desi_id": ("desi/object_id", np.dtype("int64")),
    "desi_ra": ("desi/ra", np.dtype("float64")),
    "desi_dec": ("desi/dec", np.dtype("float64")),
    "legacy_id": ("legacysurvey/object_id", np.dtype("S16")),
    "legacy_ra": ("legacysurvey/ra", np.dtype("float64")),
    "legacy_dec": ("legacysurvey/dec", np.dtype("float64")),
}


class RawImageValidationError(ValueError):
    """Raised when a source violates the pinned raw-image contract."""


@dataclass(frozen=True)
class RawImageConfig:
    """Inputs and validation pins for one selective raw-image extraction."""

    roster_path: Path
    output_dir: Path
    source: str | Path = RAW_SOURCE_URL
    overwrite: bool = False
    expected_count: int = EXPECTED_CANDIDATES
    expected_roster_sha256: str | None = ROSTER_SHA256
    expected_source_length: int | None = RAW_SOURCE_LENGTH
    expected_source_etag: str | None = RAW_SOURCE_ETAG
    http_block_size: int = 256 * 1024
    http_timeout_seconds: float = 30.0
    http_probe_attempts: int = 3
    http_probe_backoff_seconds: float = 1.0
    roster_coordinate_tolerance_arcsec: float = 0.1
    desi_coordinate_tolerance_arcsec: float = 1e-4
    legacy_coordinate_tolerance_arcsec: float = 1.0
    caption_legacy_coordinate_tolerance_arcsec: float = 1e-4


@dataclass(frozen=True)
class _SourceInfo:
    kind: str
    location: str
    length_bytes: int
    etag: str | None
    range_verified: bool
    status_code: int | None
    content_range: str | None
    local_mtime_ns: int | None = None


@dataclass(frozen=True)
class _Run:
    start: int
    stop: int
    candidate_indices: tuple[int, ...]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_http_source(source: str | Path) -> bool:
    return urlsplit(str(source)).scheme.lower() in {"http", "https"}


def _open_roster(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _parse_roster_float(value: str, *, field: str, index: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RawImageValidationError(
            f"roster row {index}: {field} must be a finite float, got {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise RawImageValidationError(f"roster row {index}: {field} is not finite")
    return parsed


def _parse_roster_id(value: str, *, index: int) -> int:
    if not re.fullmatch(r"[0-9]+", value or ""):
        raise RawImageValidationError(
            f"roster row {index}: paired_targetid must be an unsigned decimal integer"
        )
    parsed = int(value)
    if parsed > np.iinfo(np.int64).max:
        raise RawImageValidationError(
            f"roster row {index}: paired_targetid is outside signed int64"
        )
    return parsed


def _read_roster(config: RawImageConfig) -> tuple[list[dict[str, str]], list[int], str]:
    path = Path(config.roster_path)
    if not path.is_file():
        raise RawImageValidationError(f"roster does not exist: {path}")
    digest = sha256_file(path)
    if config.expected_roster_sha256 is not None and digest != config.expected_roster_sha256:
        raise RawImageValidationError(
            f"roster SHA-256 mismatch: expected {config.expected_roster_sha256}, got {digest}"
        )

    with _open_roster(path) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise RawImageValidationError("roster contains duplicate column names")
        missing = sorted(_REQUIRED_ROSTER_COLUMNS - set(fieldnames))
        if missing:
            raise RawImageValidationError(f"roster is missing required columns: {missing}")
        rows = list(reader)

    if len(rows) != config.expected_count:
        raise RawImageValidationError(
            f"roster must contain exactly {config.expected_count} rows, got {len(rows)}"
        )

    parsed_ids: list[int] = []
    seen: set[int] = set()
    caption_ids_seen: set[str] = set()
    for index, row in enumerate(rows):
        parsed_id = _parse_roster_id(row["paired_targetid"], index=index)
        if parsed_id in seen:
            raise RawImageValidationError(f"duplicate paired_targetid: {parsed_id}")
        seen.add(parsed_id)
        parsed_ids.append(parsed_id)

        ra = _parse_roster_float(row["paired_ra"], field="paired_ra", index=index)
        dec = _parse_roster_float(row["paired_dec"], field="paired_dec", index=index)
        caption_id = row["caption_object_id"]
        caption_ra = _parse_roster_float(row["caption_ra"], field="caption_ra", index=index)
        caption_dec = _parse_roster_float(row["caption_dec"], field="caption_dec", index=index)
        if not caption_id:
            raise RawImageValidationError(
                f"roster row {index}: caption_object_id must not be empty"
            )
        if caption_id in caption_ids_seen:
            raise RawImageValidationError(f"duplicate caption_object_id: {caption_id!r}")
        caption_ids_seen.add(caption_id)
        if not 0.0 <= ra < 360.0 or not 0.0 <= caption_ra < 360.0:
            raise RawImageValidationError(
                f"roster row {index}: paired_ra/caption_ra is outside [0, 360)"
            )
        if not -90.0 <= dec <= 90.0 or not -90.0 <= caption_dec <= 90.0:
            raise RawImageValidationError(
                f"roster row {index}: paired_dec/caption_dec is outside [-90, 90]"
            )
    return rows, parsed_ids, digest


def _probe_http_source(
    url: str,
    *,
    expected_length: int,
    expected_etag: str,
    timeout_seconds: float,
    attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> _SourceInfo:
    """Verify a one-byte HTTP range response and return pinned source metadata."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if not math.isfinite(backoff_seconds) or backoff_seconds < 0:
        raise ValueError("backoff_seconds must be nonnegative")

    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={"Range": "bytes=0-0", "If-Match": expected_etag},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = response.getcode()
                headers = response.headers
                body = response.read(2)
                final_url = response.geturl()
        except urllib.error.HTTPError as exc:
            raise RawImageValidationError(f"HTTP range probe returned status {exc.code}") from exc
        except OSError as exc:
            if attempt + 1 == attempts:
                raise RawImageValidationError(
                    f"HTTP range probe failed for {url} after {attempts} attempts: {exc}"
                ) from exc
            time.sleep(backoff_seconds * (2**attempt))
            continue
        break

    if status != 206:
        raise RawImageValidationError(f"HTTP range probe requires status 206, got {status}")
    if len(body) != 1:
        raise RawImageValidationError(
            f"HTTP range probe expected exactly one response byte, got {len(body)}"
        )
    content_range = headers.get("Content-Range")
    match = re.fullmatch(r"bytes 0-0/([0-9]+)", content_range or "")
    if match is None:
        raise RawImageValidationError(
            f"HTTP range probe returned invalid Content-Range: {content_range!r}"
        )
    observed_length = int(match.group(1))
    if observed_length != expected_length:
        raise RawImageValidationError(
            f"source length mismatch: expected {expected_length}, got {observed_length}"
        )
    content_length = headers.get("Content-Length")
    if content_length is not None and content_length != "1":
        raise RawImageValidationError(f"one-byte range returned Content-Length {content_length!r}")
    observed_etag = headers.get("ETag")
    if observed_etag != expected_etag:
        raise RawImageValidationError(
            f"source ETag mismatch: expected {expected_etag!r}, got {observed_etag!r}"
        )
    return _SourceInfo(
        kind="http",
        location=final_url,
        length_bytes=observed_length,
        etag=observed_etag,
        range_verified=True,
        status_code=status,
        content_range=content_range,
    )


def _inspect_source(config: RawImageConfig) -> _SourceInfo:
    if _is_http_source(config.source):
        if config.expected_source_length is None or config.expected_source_etag is None:
            raise ValueError("HTTP sources require expected_source_length and expected_source_etag")
        return _probe_http_source(
            str(config.source),
            expected_length=config.expected_source_length,
            expected_etag=config.expected_source_etag,
            timeout_seconds=config.http_timeout_seconds,
            attempts=config.http_probe_attempts,
            backoff_seconds=config.http_probe_backoff_seconds,
        )

    path = Path(config.source).resolve()
    if not path.is_file():
        raise RawImageValidationError(f"local HDF5 source does not exist: {path}")
    stat = path.stat()
    if config.expected_source_length is not None and stat.st_size != config.expected_source_length:
        raise RawImageValidationError(
            f"source length mismatch: expected {config.expected_source_length}, got {stat.st_size}"
        )
    if config.expected_source_etag is not None:
        raise ValueError("expected_source_etag must be None for a local HDF5 source")
    return _SourceInfo(
        kind="local",
        location=str(path),
        length_bytes=stat.st_size,
        etag=None,
        range_verified=False,
        status_code=None,
        content_range=None,
        local_mtime_ns=stat.st_mtime_ns,
    )


@contextmanager
def _open_hdf5(config: RawImageConfig) -> Iterator[h5py.File]:
    if _is_http_source(config.source):
        headers = {"If-Match": config.expected_source_etag}
        with fsspec.open(
            str(config.source),
            mode="rb",
            block_size=config.http_block_size,
            cache_type="none",
            headers=headers,
        ) as remote:
            with h5py.File(remote, "r") as handle:
                yield handle
        return

    with h5py.File(Path(config.source), "r") as handle:
        yield handle


def _require_dataset(
    handle: h5py.File,
    path: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    require_contiguous: bool,
) -> h5py.Dataset:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise RawImageValidationError(f"source is missing dataset {path!r}")
    dataset = handle[path]
    if dataset.shape != shape:
        raise RawImageValidationError(f"{path} has shape {dataset.shape}, expected {shape}")
    if dataset.dtype != dtype:
        raise RawImageValidationError(f"{path} has dtype {dataset.dtype}, expected {dtype}")
    if require_contiguous and (dataset.chunks is not None or dataset.compression is not None):
        raise RawImageValidationError(
            f"{path} must be contiguous and uncompressed for bounded HTTP range extraction"
        )
    return dataset


def _validate_schema(handle: h5py.File, *, require_contiguous: bool) -> int:
    id_path = _METADATA_DATASETS["provabgs_id"][0]
    if id_path not in handle or not isinstance(handle[id_path], h5py.Dataset):
        raise RawImageValidationError(f"source is missing dataset {id_path!r}")
    id_dataset = handle[id_path]
    if id_dataset.ndim != 1:
        raise RawImageValidationError(f"{id_path} must be one-dimensional")
    source_count = id_dataset.shape[0]

    for path, dtype in _METADATA_DATASETS.values():
        _require_dataset(
            handle,
            path,
            shape=(source_count,),
            dtype=dtype,
            require_contiguous=require_contiguous,
        )
    for spec in _IMAGE_DATASETS.values():
        _require_dataset(
            handle,
            spec.source_path,
            shape=(source_count, *spec.tail_shape),
            dtype=spec.dtype,
            require_contiguous=require_contiguous,
        )
    return source_count


def _build_raw_row_map(handle: h5py.File, roster_ids: list[int]) -> tuple[list[int], np.ndarray]:
    source_ids = np.asarray(handle[_METADATA_DATASETS["provabgs_id"][0]][:])
    unique_ids, counts = np.unique(source_ids, return_counts=True)
    duplicate_ids = unique_ids[counts != 1]
    if duplicate_ids.size:
        raise RawImageValidationError(
            f"PROVABGS object_id is not unique; first duplicate is {int(duplicate_ids[0])}"
        )
    lookup = {int(value): index for index, value in enumerate(source_ids)}
    missing = [value for value in roster_ids if value not in lookup]
    if missing:
        raise RawImageValidationError(
            f"{len(missing)} roster PROVABGS IDs are absent; first missing ID is {missing[0]}"
        )
    raw_rows = [lookup[value] for value in roster_ids]
    if len(set(raw_rows)) != len(raw_rows):
        raise RawImageValidationError("multiple roster IDs map to the same raw HDF5 row")
    return raw_rows, source_ids


def _contiguous_runs(raw_rows: list[int]) -> list[_Run]:
    ordered = sorted((raw_row, candidate_index) for candidate_index, raw_row in enumerate(raw_rows))
    if not ordered:
        return []
    runs: list[_Run] = []
    run_start = ordered[0][0]
    previous = ordered[0][0]
    indices = [ordered[0][1]]
    for raw_row, candidate_index in ordered[1:]:
        if raw_row == previous + 1:
            previous = raw_row
            indices.append(candidate_index)
            continue
        runs.append(_Run(run_start, previous + 1, tuple(indices)))
        run_start = previous = raw_row
        indices = [candidate_index]
    runs.append(_Run(run_start, previous + 1, tuple(indices)))
    return runs


def _coordinate(ra: Any, dec: Any, *, label: str, index: int) -> tuple[float, float]:
    ra_float, dec_float = float(ra), float(dec)
    if not math.isfinite(ra_float) or not 0.0 <= ra_float < 360.0:
        raise RawImageValidationError(f"candidate {index}: invalid {label} RA {ra_float}")
    if not math.isfinite(dec_float) or not -90.0 <= dec_float <= 90.0:
        raise RawImageValidationError(f"candidate {index}: invalid {label} Dec {dec_float}")
    return ra_float, dec_float


def _angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1_rad, dec1_rad, ra2_rad, dec2_rad = map(math.radians, (ra1, dec1, ra2, dec2))
    sin_dec = math.sin((dec2_rad - dec1_rad) / 2.0)
    sin_ra = math.sin((ra2_rad - ra1_rad) / 2.0)
    haversine = sin_dec**2 + math.cos(dec1_rad) * math.cos(dec2_rad) * sin_ra**2
    haversine = min(1.0, max(0.0, haversine))
    return math.degrees(2.0 * math.asin(math.sqrt(haversine))) * 3600.0


def _decode_legacy_id(value: np.bytes_, *, index: int) -> str:
    try:
        decoded = bytes(value).rstrip(b"\x00").decode("ascii")
    except UnicodeDecodeError as exc:
        raise RawImageValidationError(
            f"candidate {index}: Legacy Survey object_id is not ASCII"
        ) from exc
    if not decoded:
        raise RawImageValidationError(f"candidate {index}: Legacy Survey object_id is empty")
    return decoded


def _crosscheck_metadata(
    config: RawImageConfig,
    handle: h5py.File,
    rows: list[dict[str, str]],
    roster_ids: list[int],
    raw_rows: list[int],
    source_ids: np.ndarray,
) -> list[dict[str, Any]]:
    values: dict[str, np.ndarray] = {}
    selected_indices = np.asarray(raw_rows, dtype=np.intp)
    for key, (path, _) in _METADATA_DATASETS.items():
        if key == "provabgs_id":
            values[key] = source_ids[selected_indices]
        else:
            full_column = np.asarray(handle[path][:])
            values[key] = full_column[selected_indices]

    records: list[dict[str, Any]] = []
    legacy_ids_seen: set[str] = set()
    for index, (row, roster_id, raw_row) in enumerate(zip(rows, roster_ids, raw_rows, strict=True)):
        provabgs_id = int(values["provabgs_id"][index])
        desi_id = int(values["desi_id"][index])
        legacy_id = _decode_legacy_id(values["legacy_id"][index], index=index)
        if provabgs_id != roster_id:
            raise RawImageValidationError(
                f"candidate {index}: PROVABGS ID mismatch; expected {roster_id}, got {provabgs_id}"
            )
        if desi_id != provabgs_id:
            raise RawImageValidationError(
                f"candidate {index}: DESI ID {desi_id} does not match PROVABGS ID {provabgs_id}"
            )
        if legacy_id in legacy_ids_seen:
            raise RawImageValidationError(
                f"candidate {index}: duplicate Legacy Survey object_id {legacy_id!r}"
            )
        legacy_ids_seen.add(legacy_id)
        caption_id = row["caption_object_id"]
        if legacy_id != caption_id:
            raise RawImageValidationError(
                f"candidate {index}: Legacy Survey ID {legacy_id!r} does not match "
                f"caption_object_id {caption_id!r}"
            )

        provabgs_coord = _coordinate(
            values["provabgs_ra"][index],
            values["provabgs_dec"][index],
            label="PROVABGS",
            index=index,
        )
        desi_coord = _coordinate(
            values["desi_ra"][index],
            values["desi_dec"][index],
            label="DESI",
            index=index,
        )
        legacy_coord = _coordinate(
            values["legacy_ra"][index],
            values["legacy_dec"][index],
            label="Legacy Survey",
            index=index,
        )
        roster_coord = (
            _parse_roster_float(row["paired_ra"], field="paired_ra", index=index),
            _parse_roster_float(row["paired_dec"], field="paired_dec", index=index),
        )
        caption_coord = (
            _parse_roster_float(row["caption_ra"], field="caption_ra", index=index),
            _parse_roster_float(row["caption_dec"], field="caption_dec", index=index),
        )
        roster_separation = _angular_separation_arcsec(*roster_coord, *provabgs_coord)
        desi_separation = _angular_separation_arcsec(*provabgs_coord, *desi_coord)
        legacy_separation = _angular_separation_arcsec(*provabgs_coord, *legacy_coord)
        caption_legacy_separation = _angular_separation_arcsec(*caption_coord, *legacy_coord)
        if roster_separation > config.roster_coordinate_tolerance_arcsec:
            raise RawImageValidationError(
                f"candidate {index}: roster/PROVABGS coordinate separation "
                f"{roster_separation:.6g} arcsec exceeds "
                f"{config.roster_coordinate_tolerance_arcsec}"
            )
        if desi_separation > config.desi_coordinate_tolerance_arcsec:
            raise RawImageValidationError(
                f"candidate {index}: PROVABGS/DESI coordinate separation "
                f"{desi_separation:.6g} arcsec exceeds "
                f"{config.desi_coordinate_tolerance_arcsec}"
            )
        if legacy_separation > config.legacy_coordinate_tolerance_arcsec:
            raise RawImageValidationError(
                f"candidate {index}: PROVABGS/Legacy Survey coordinate separation "
                f"{legacy_separation:.6g} arcsec exceeds "
                f"{config.legacy_coordinate_tolerance_arcsec}"
            )
        if caption_legacy_separation > config.caption_legacy_coordinate_tolerance_arcsec:
            raise RawImageValidationError(
                f"candidate {index}: caption/Legacy Survey coordinate separation "
                f"{caption_legacy_separation:.6g} arcsec exceeds "
                f"{config.caption_legacy_coordinate_tolerance_arcsec}"
            )

        records.append(
            {
                "candidate_index": index,
                "paired_targetid": str(roster_id),
                "caption_object_id": caption_id,
                "raw_row": raw_row,
                "source_ids": {
                    "provabgs": str(provabgs_id),
                    "desi": str(desi_id),
                    "legacysurvey": legacy_id,
                },
                "coordinates_deg": {
                    "roster": {"ra": roster_coord[0], "dec": roster_coord[1]},
                    "caption": {"ra": caption_coord[0], "dec": caption_coord[1]},
                    "provabgs": {"ra": provabgs_coord[0], "dec": provabgs_coord[1]},
                    "desi": {"ra": desi_coord[0], "dec": desi_coord[1]},
                    "legacysurvey": {"ra": legacy_coord[0], "dec": legacy_coord[1]},
                },
                "coordinate_crosscheck_arcsec": {
                    "roster_to_provabgs": roster_separation,
                    "provabgs_to_desi": desi_separation,
                    "provabgs_to_legacysurvey": legacy_separation,
                    "caption_to_legacysurvey": caption_legacy_separation,
                },
                "acceptance_status": "pending_review",
                "split": None,
            }
        )
    return records


def _map_aion_bands(block: np.ndarray, candidate_indices: tuple[int, ...]) -> np.ndarray:
    mapped = np.empty(block.shape, dtype="S5")
    for local_index, candidate_index in enumerate(candidate_indices):
        source_names: list[str] = []
        for value in block[local_index]:
            try:
                source_name = bytes(value).rstrip(b"\x00").decode("ascii")
            except UnicodeDecodeError as exc:
                raise RawImageValidationError(
                    f"candidate {candidate_index}: image_band contains non-ASCII bytes"
                ) from exc
            if source_name not in _AION_BAND_MAPPING:
                raise RawImageValidationError(
                    f"candidate {candidate_index}: unsupported image band {source_name!r}"
                )
            source_names.append(source_name)
        if tuple(source_names) != _EXPECTED_SOURCE_BANDS:
            raise RawImageValidationError(
                f"candidate {candidate_index}: expected image bands in strict "
                f"des-g/des-r/des-i/des-z order, "
                f"got {source_names}"
            )
        mapped[local_index] = np.asarray(
            [_AION_BAND_MAPPING[name].encode("ascii") for name in source_names], dtype="S5"
        )
    return mapped


def _extract_datasets(
    handle: h5py.File,
    runs: list[_Run],
    candidate_count: int,
    staging: Path,
) -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray]:
    outputs: dict[str, dict[str, Any]] = {}
    source_bands: np.ndarray | None = None
    aion_bands: np.ndarray | None = None

    for key, spec in _IMAGE_DATASETS.items():
        output_path = staging / spec.filename
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=spec.dtype,
            shape=(candidate_count, *spec.tail_shape),
        )
        mapped_output: np.memmap | None = None
        mapped_path = staging / "aion_image_band.npy"
        if key == "image_band":
            mapped_output = np.lib.format.open_memmap(
                mapped_path,
                mode="w+",
                dtype=np.dtype("S5"),
                shape=(candidate_count, 4),
            )
        try:
            dataset = handle[spec.source_path]
            for run in runs:
                block = np.asarray(dataset[run.start : run.stop])
                if block.shape != (len(run.candidate_indices), *spec.tail_shape):
                    raise RawImageValidationError(
                        f"short or malformed read from {spec.source_path}: got {block.shape}"
                    )
                if spec.finite and not np.isfinite(block).all():
                    raise RawImageValidationError(
                        f"{spec.source_path} contains non-finite values in raw rows "
                        f"{run.start}:{run.stop}"
                    )
                destination = np.asarray(run.candidate_indices, dtype=np.intp)
                output[destination] = block
                if mapped_output is not None:
                    mapped_output[destination] = _map_aion_bands(block, run.candidate_indices)
            output.flush()
            if mapped_output is not None:
                mapped_output.flush()
        finally:
            del output
            if mapped_output is not None:
                del mapped_output

        outputs[key] = {
            "path": spec.filename,
            "source_dataset": spec.source_path,
            "dtype": str(spec.dtype),
            "shape": [candidate_count, *spec.tail_shape],
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "mmap_safe": True,
        }
        if key == "image_band":
            outputs["aion_image_band"] = {
                "path": mapped_path.name,
                "derived_from": spec.filename,
                "dtype": str(np.dtype("S5")),
                "shape": [candidate_count, 4],
                "size_bytes": mapped_path.stat().st_size,
                "sha256": sha256_file(mapped_path),
                "mmap_safe": True,
            }
            source_bands = np.load(output_path, mmap_mode="r")
            aion_bands = np.load(mapped_path, mmap_mode="r")

    assert source_bands is not None and aion_bands is not None
    # Materialize these tiny arrays before the temporary mmap objects are closed.
    return outputs, np.asarray(source_bands).copy(), np.asarray(aion_bands).copy()


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


def _verify_source_unchanged(config: RawImageConfig, original: _SourceInfo) -> None:
    if original.kind == "http":
        current = _inspect_source(config)
        if current.length_bytes != original.length_bytes or current.etag != original.etag:
            raise RawImageValidationError("remote source changed during extraction")
        return
    stat = Path(original.location).stat()
    if stat.st_size != original.length_bytes or stat.st_mtime_ns != original.local_mtime_ns:
        raise RawImageValidationError("local source changed during extraction")


def _decode_band_row(row: np.ndarray) -> list[str]:
    return [bytes(value).rstrip(b"\x00").decode("ascii") for value in row]


def extract_raw_images(config: RawImageConfig) -> dict[str, Any]:
    """Extract the roster's raw image rows into an atomic mmap-safe artifact."""

    if config.http_block_size < 1:
        raise ValueError("http_block_size must be positive")
    if config.http_timeout_seconds <= 0:
        raise ValueError("http_timeout_seconds must be positive")
    if config.http_probe_attempts < 1:
        raise ValueError("http_probe_attempts must be at least 1")
    if (
        not math.isfinite(config.http_probe_backoff_seconds)
        or config.http_probe_backoff_seconds < 0
    ):
        raise ValueError("http_probe_backoff_seconds must be nonnegative")
    for name in (
        "roster_coordinate_tolerance_arcsec",
        "desi_coordinate_tolerance_arcsec",
        "legacy_coordinate_tolerance_arcsec",
        "caption_legacy_coordinate_tolerance_arcsec",
    ):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be nonnegative")

    output_dir = Path(config.output_dir)
    if _path_exists(output_dir) and not config.overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")

    rows, roster_ids, roster_sha256 = _read_roster(config)
    source_info = _inspect_source(config)
    staging: Path | None = None
    try:
        with _open_hdf5(config) as handle:
            source_count = _validate_schema(handle, require_contiguous=source_info.kind == "http")
            raw_rows, source_ids = _build_raw_row_map(handle, roster_ids)
            runs = _contiguous_runs(raw_rows)
            records = _crosscheck_metadata(
                config,
                handle,
                rows,
                roster_ids,
                raw_rows,
                source_ids,
            )

            output_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
            )
            outputs, source_bands, aion_bands = _extract_datasets(handle, runs, len(rows), staging)

        _verify_source_unchanged(config, source_info)
        for index, record in enumerate(records):
            record["bands"] = {
                "source": _decode_band_row(source_bands[index]),
                "aion": _decode_band_row(aion_bands[index]),
            }

        manifest = {
            "schema_version": 1,
            "artifact_type": "astroprism_raw_images",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "roster": {
                "path": str(Path(config.roster_path)),
                "sha256": roster_sha256,
                "row_count": len(rows),
            },
            "source": {
                "kind": source_info.kind,
                "location": source_info.location,
                "url": source_info.location if source_info.kind == "http" else None,
                "etag": source_info.etag,
                "length_bytes": source_info.length_bytes,
                "http_range_verified": source_info.range_verified,
                "http_status": source_info.status_code,
                "content_range": source_info.content_range,
                "hdf5_row_count": source_count,
            },
            "selection": {
                "candidate_count": len(rows),
                "roster_order_preserved": True,
                "raw_rows_in_roster_order": raw_rows,
                "raw_row_min": min(raw_rows),
                "raw_row_max": max(raw_rows),
                "contiguous_run_count": len(runs),
                "http_cache_type": "none" if source_info.kind == "http" else None,
                "acceptance_status": "pending_review",
                "split": None,
            },
            "identity_tolerances_arcsec": {
                "roster_to_provabgs": config.roster_coordinate_tolerance_arcsec,
                "provabgs_to_desi": config.desi_coordinate_tolerance_arcsec,
                "provabgs_to_legacysurvey": config.legacy_coordinate_tolerance_arcsec,
                "caption_to_legacysurvey": (config.caption_legacy_coordinate_tolerance_arcsec),
            },
            "bands": {
                "source_labels_preserved": True,
                "aion_mapping": _AION_BAND_MAPPING,
                "aion_mapping_verified_for_every_candidate": True,
            },
            "outputs": outputs,
            "candidates": records,
            "spectra_included": False,
        }
        assert staging is not None
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _commit_staging(staging, output_dir, overwrite=config.overwrite)
        staging = None
        return {
            "status": "prepared",
            "candidate_count": len(rows),
            "source_length_bytes": source_info.length_bytes,
            "source_etag": source_info.etag,
            "contiguous_run_count": len(runs),
            "output_dir": str(output_dir),
            "manifest": str(output_dir / "manifest.json"),
            "spectra_included": False,
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)

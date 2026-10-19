#!/usr/bin/env python3
"""Build a portable Hugging Face dataset upload tree for AstroPRISM.

The package deliberately keeps the calibrated NumPy arrays monolithic.  Every
Parquet row retains ``candidate_index``, which is also the first-axis index into
each raw array.  RGB images and spectrum plots embedded in Parquet are review
previews only; they are not substitutes for the calibrated model inputs.

No source cache, released AION embedding, or local filesystem path is copied.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CANDIDATES = 646
EXPECTED_SPLIT_COUNTS = {"train": 516, "validation": 65, "test": 65}
ROSTER_SHA256 = "650935faa58953c9929e7379429bfb332a7486dab54544fedef64ca385abb3bc"
CAPTION_REPO_ID = "astronolan/galaxy-descriptions"
CAPTION_REVISION = "6890dc0c8fc793867c07a6ce4fce11e51c167d6e"
PAIRED_REPO_ID = "EiffL/desi_x_ls_aion"
PAIRED_REVISION = "57f7c7575260811ea076163bc743787d75b9aa49"
RAW_SOURCE_URL = "https://users.flatironinstitute.org/~polymathic/data/provabgs_desi_ls.hdf5"
ASTROSPECLM_PAPER = "https://arxiv.org/abs/2609.07102"

FORBIDDEN_PORTABLE_TEXT = ("/lus/", "_local/", "BaseMM_PRISM")
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}

IMAGE_ARRAY_CONTRACT = {
    "image_array": (np.dtype("float32"), (4, 160, 160)),
    "image_ivar": (np.dtype("float32"), (4, 160, 160)),
    "image_mask": (np.dtype("bool"), (160, 160)),
    "object_mask": (np.dtype("uint8"), (160, 160)),
    "image_scale": (np.dtype("float32"), (4,)),
    "image_band": (np.dtype("S5"), (4,)),
    "aion_image_band": (np.dtype("S5"), (4,)),
}
SPECTRUM_ARRAY_CONTRACT = {
    "spectrum_flux": (np.dtype("float32"), (7781,)),
    "spectrum_ivar": (np.dtype("float32"), (7781,)),
    "spectrum_mask": (np.dtype("bool"), (7781,)),
    "spectrum_wavelength": (np.dtype("float32"), (7781,)),
}


class PackagingValidationError(ValueError):
    """Raised when an input violates the portable dataset contract."""


@dataclass(frozen=True)
class HFDatasetConfig:
    """Inputs for one atomic dataset staging run."""

    output_dir: Path
    accepted_candidates_path: Path = (
        REPO_ROOT / "_local/astroprism/stage1a_accepted/candidates.jsonl"
    )
    accepted_manifest_path: Path = REPO_ROOT / "_local/astroprism/stage1a_accepted/manifest.json"
    raw_images_manifest_path: Path = REPO_ROOT / "_local/astroprism/raw_images/manifest.json"
    raw_spectra_manifest_path: Path = REPO_ROOT / "_local/astroprism/raw_spectra/manifest.json"
    caption_root: Path = REPO_ROOT / "_local/astroprism/hf/datasets/galaxy-descriptions"
    paired_root: Path = REPO_ROOT / "_local/astroprism/hf/datasets/desi_x_ls_aion"
    spectrum_caption_bundle: Path = (
        REPO_ROOT / "_local/astroprism/stage1b_spectrum_caption_accepted"
    )
    roster_path: Path = REPO_ROOT / "docs/astroprism/data/overlap_candidates_646.csv.gz"
    expected_count: int = EXPECTED_CANDIDATES
    expected_split_counts: dict[str, int] | None = None
    expected_roster_sha256: str | None = ROSTER_SHA256
    caption_repo_id: str = CAPTION_REPO_ID
    caption_revision: str = CAPTION_REVISION
    paired_repo_id: str = PAIRED_REPO_ID
    paired_revision: str = PAIRED_REVISION
    link_mode: str = "auto"
    overwrite: bool = False


@dataclass(frozen=True)
class _RawAsset:
    key: str
    source: Path
    destination: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    size_bytes: int
    source_dataset: str | None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PackagingValidationError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackagingValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackagingValidationError(f"{label} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PackagingValidationError(f"{label} does not exist: {path}")
    result: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise PackagingValidationError(f"{label} has a blank row at line {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PackagingValidationError(
                        f"{label} line {line_number} must be a JSON object"
                    )
                result.append(value)
    except json.JSONDecodeError as exc:
        raise PackagingValidationError(
            f"cannot parse {label} {path} at line {exc.lineno}: {exc.msg}"
        ) from exc
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            )


def _safe_child(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise PackagingValidationError(f"unsafe {label} path: {relative!r}")
    resolved_root = root.resolve()
    path = (resolved_root / Path(*pure.parts)).resolve()
    if not path.is_relative_to(resolved_root):
        raise PackagingValidationError(f"{label} escapes its root: {relative!r}")
    if not path.is_file():
        raise PackagingValidationError(f"{label} does not exist: {path}")
    return path


def _require_nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PackagingValidationError(f"{label} must be a non-empty string")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PackagingValidationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise PackagingValidationError(f"{label} must be finite")
    return result


def _validate_accepted_rows(
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
    expected_split_counts: dict[str, int],
    caption_repo_id: str,
    caption_revision: str,
    paired_repo_id: str,
    paired_revision: str,
) -> None:
    if len(rows) != expected_count:
        raise PackagingValidationError(
            f"accepted candidates must contain {expected_count} rows, got {len(rows)}"
        )
    paired_ids: set[str] = set()
    caption_ids: set[str] = set()
    paired_locators: set[tuple[str, int]] = set()
    caption_locators: set[tuple[str, int]] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    observed_splits: Counter[str] = Counter()

    for expected_index, row in enumerate(rows):
        if row.get("candidate_index") != expected_index:
            raise PackagingValidationError(
                f"accepted row {expected_index} has candidate_index="
                f"{row.get('candidate_index')!r}; direct raw-array indexing would be unsafe"
            )
        paired_id = _require_nonempty_text(
            row.get("paired_targetid"), label=f"candidate {expected_index} paired_targetid"
        )
        caption_id = _require_nonempty_text(
            row.get("caption_object_id"), label=f"candidate {expected_index} caption_object_id"
        )
        if paired_id in paired_ids or caption_id in caption_ids:
            raise PackagingValidationError(
                f"candidate {expected_index} repeats a paired or caption identifier"
            )
        paired_ids.add(paired_id)
        caption_ids.add(caption_id)

        if row.get("acceptance_status") != "accepted":
            raise PackagingValidationError(f"candidate {expected_index} is not accepted")
        caption = row.get("caption")
        if not isinstance(caption, dict):
            raise PackagingValidationError(f"candidate {expected_index} has no caption object")
        _require_nonempty_text(
            caption.get("original"), label=f"candidate {expected_index} caption.original"
        )
        _require_nonempty_text(
            caption.get("summary_original"),
            label=f"candidate {expected_index} caption.summary_original",
        )
        if caption.get("status") != "released_reused":
            raise PackagingValidationError(
                f"candidate {expected_index} caption status is not released_reused"
            )

        split = row.get("split")
        if split not in expected_split_counts:
            raise PackagingValidationError(
                f"candidate {expected_index} has unsupported split {split!r}"
            )
        observed_splits[split] += 1
        group = _require_nonempty_text(
            row.get("source_group_id"), label=f"candidate {expected_index} source_group_id"
        )
        group_splits[group].add(split)

        coordinates = row.get("coordinates_deg")
        if not isinstance(coordinates, dict):
            raise PackagingValidationError(
                f"candidate {expected_index} has no coordinates_deg object"
            )
        for field in ("paired_ra", "paired_dec", "caption_ra", "caption_dec"):
            _finite_float(coordinates.get(field), label=f"candidate {expected_index} {field}")
        separation = _finite_float(
            row.get("separation_arcsec"),
            label=f"candidate {expected_index} separation_arcsec",
        )
        if not 0.0 <= separation <= 1.0:
            raise PackagingValidationError(
                f"candidate {expected_index} separation_arcsec is outside [0, 1]"
            )

        for kind, repo_id, revision, locator_set in (
            ("caption", caption_repo_id, caption_revision, caption_locators),
            ("paired", paired_repo_id, paired_revision, paired_locators),
        ):
            source = row.get(f"{kind}_source")
            if not isinstance(source, dict):
                raise PackagingValidationError(
                    f"candidate {expected_index} has no {kind}_source object"
                )
            if source.get("repo_id") != repo_id or source.get("revision") != revision:
                raise PackagingValidationError(
                    f"candidate {expected_index} {kind} source pin does not match the package pin"
                )
            shard = _require_nonempty_text(
                source.get("shard"), label=f"candidate {expected_index} {kind} shard"
            )
            try:
                source_row = int(source["row"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PackagingValidationError(
                    f"candidate {expected_index} {kind} row is invalid"
                ) from exc
            if source_row < 0 or (shard, source_row) in locator_set:
                raise PackagingValidationError(
                    f"candidate {expected_index} has an invalid or duplicate {kind} locator"
                )
            locator_set.add((shard, source_row))
            digest = _require_nonempty_text(
                source.get("sha256"), label=f"candidate {expected_index} {kind} SHA-256"
            )
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise PackagingValidationError(
                    f"candidate {expected_index} {kind} SHA-256 is malformed"
                )

    if dict(observed_splits) != expected_split_counts:
        raise PackagingValidationError(
            f"accepted split counts are {dict(observed_splits)}, expected {expected_split_counts}"
        )
    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if leaking_groups:
        raise PackagingValidationError(
            f"source groups cross train/held-out splits: {leaking_groups[:5]}"
        )


def _validate_accepted_manifest(
    path: Path,
    manifest: dict[str, Any],
    candidates_path: Path,
    rows: list[dict[str, Any]],
    expected_split_counts: dict[str, int],
    roster_sha256: str,
) -> None:
    output = manifest.get("outputs", {}).get("candidates", {})
    if output.get("sha256") != sha256_file(candidates_path):
        raise PackagingValidationError("accepted candidate SHA-256 disagrees with its manifest")
    if output.get("row_count") != len(rows):
        raise PackagingValidationError("accepted candidate row count disagrees with its manifest")
    if manifest.get("splits", {}).get("row_counts") != expected_split_counts:
        raise PackagingValidationError("accepted manifest split counts are not authoritative")
    if manifest.get("pins", {}).get("roster_sha256") != roster_sha256:
        raise PackagingValidationError("accepted manifest roster pin mismatch")
    recorded_groups = manifest.get("splits", {}).get("group_count")
    actual_groups = len({row["source_group_id"] for row in rows})
    if recorded_groups != actual_groups:
        raise PackagingValidationError(
            f"accepted manifest records {recorded_groups} groups, found {actual_groups}"
        )
    if path.parent == candidates_path.parent and output.get("path") != candidates_path.name:
        raise PackagingValidationError("accepted manifest points to a different candidate file")


def _row_groups_for_indices(parquet: Any, indices: list[int]) -> dict[int, list[tuple[int, int]]]:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    sorted_indices = sorted(indices)
    pointer = 0
    offset = 0
    for group_index in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        end = offset + count
        while pointer < len(sorted_indices) and sorted_indices[pointer] < end:
            source_row = sorted_indices[pointer]
            if source_row < offset:
                raise PackagingValidationError("source Parquet row locators are unordered")
            grouped[group_index].append((source_row, source_row - offset))
            pointer += 1
        offset = end
    if pointer != len(sorted_indices):
        raise PackagingValidationError(
            f"source row {sorted_indices[pointer]} is outside the Parquet file"
        )
    return grouped


def _read_selected_parquet_rows(
    path: Path, indices: list[int], columns: list[str]
) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - deployment environment check
        raise RuntimeError("pyarrow is required to stage AstroPRISM") from exc

    parquet = pq.ParquetFile(path)
    missing = [name for name in columns if parquet.schema_arrow.get_field_index(name) < 0]
    if missing:
        raise PackagingValidationError(f"{path.name} is missing Parquet columns {missing}")
    selected: dict[int, dict[str, Any]] = {}
    for group_index, wanted in _row_groups_for_indices(parquet, indices).items():
        table = parquet.read_row_group(group_index, columns=columns, use_threads=False)
        for source_row, local_row in wanted:
            selected[source_row] = table.slice(local_row, 1).to_pylist()[0]
    return selected


def _validate_png(
    value: Any,
    *,
    label: str,
    expected_size: tuple[int, int] | None = None,
    expected_mode: str | None = None,
) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise PackagingValidationError(f"{label} is not non-empty PNG bytes")
    try:
        from PIL import Image as PILImage
    except ImportError as exc:  # pragma: no cover - deployment environment check
        raise RuntimeError("Pillow is required to validate embedded previews") from exc
    try:
        with PILImage.open(io.BytesIO(value)) as image:
            image.verify()
        with PILImage.open(io.BytesIO(value)) as image:
            if image.format != "PNG":
                raise PackagingValidationError(f"{label} must be PNG, got {image.format}")
            if expected_size is not None and image.size != expected_size:
                raise PackagingValidationError(f"{label} must be {expected_size}, got {image.size}")
            if expected_mode is not None and image.mode != expected_mode:
                raise PackagingValidationError(
                    f"{label} must use {expected_mode}, got {image.mode}"
                )
            if image.width < 1 or image.height < 1:
                raise PackagingValidationError(f"{label} has invalid dimensions")
    except PackagingValidationError:
        raise
    except Exception as exc:
        raise PackagingValidationError(f"{label} is not a valid PNG: {exc}") from exc
    return value


def _validate_source_rows(
    rows: list[dict[str, Any]],
    *,
    caption_root: Path,
    paired_root: Path,
) -> dict[int, bytes]:
    checksum_cache: dict[Path, str] = {}
    previews: dict[int, bytes] = {}

    def verified_source(root: Path, shard: str, expected_hashes: set[str], label: str) -> Path:
        if len(expected_hashes) != 1:
            raise PackagingValidationError(f"conflicting {label} hashes for {shard}")
        source = _safe_child(root, shard, label=f"{label} shard")
        observed = checksum_cache.setdefault(source, sha256_file(source))
        expected = next(iter(expected_hashes))
        if observed != expected:
            raise PackagingValidationError(
                f"{label} source SHA-256 mismatch for {shard}: expected {expected}, got {observed}"
            )
        return source

    caption_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paired_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        caption_groups[row["caption_source"]["shard"]].append(row)
        paired_groups[row["paired_source"]["shard"]].append(row)

    caption_columns = ["object_id", "survey", "ra", "dec", "image", "caption", "summary"]
    for shard, group in sorted(caption_groups.items()):
        source = verified_source(
            caption_root,
            shard,
            {row["caption_source"]["sha256"] for row in group},
            "caption",
        )
        source_rows = _read_selected_parquet_rows(
            source, [int(row["caption_source"]["row"]) for row in group], caption_columns
        )
        for accepted in group:
            index = int(accepted["candidate_index"])
            source_row = source_rows[int(accepted["caption_source"]["row"])]
            if source_row["object_id"] != accepted["caption_object_id"]:
                raise PackagingValidationError(f"candidate {index} caption source ID mismatch")
            if source_row["survey"] != "legacy_dr10":
                raise PackagingValidationError(
                    f"candidate {index} caption survey is not legacy_dr10"
                )
            coordinates = accepted["coordinates_deg"]
            for source_field, accepted_field in (("ra", "caption_ra"), ("dec", "caption_dec")):
                observed = _finite_float(
                    source_row[source_field], label=f"candidate {index} source {source_field}"
                )
                expected = float(coordinates[accepted_field])
                if abs(observed - expected) > 1e-9:
                    raise PackagingValidationError(
                        f"candidate {index} caption source {source_field} mismatch"
                    )
            if source_row["caption"] != accepted["caption"]["original"]:
                raise PackagingValidationError(f"candidate {index} source caption text mismatch")
            if source_row["summary"] != accepted["caption"]["summary_original"]:
                raise PackagingValidationError(f"candidate {index} source caption summary mismatch")
            previews[index] = _validate_png(
                source_row["image"],
                label=f"candidate {index} source image",
                expected_size=(160, 160),
                expected_mode="RGB",
            )

    paired_columns = ["object_id_provabgs", "ra_provabgs", "dec_provabgs"]
    for shard, group in sorted(paired_groups.items()):
        source = verified_source(
            paired_root,
            shard,
            {row["paired_source"]["sha256"] for row in group},
            "paired",
        )
        source_rows = _read_selected_parquet_rows(
            source, [int(row["paired_source"]["row"]) for row in group], paired_columns
        )
        for accepted in group:
            index = int(accepted["candidate_index"])
            source_row = source_rows[int(accepted["paired_source"]["row"])]
            if str(source_row["object_id_provabgs"]) != accepted["paired_targetid"]:
                raise PackagingValidationError(f"candidate {index} paired source ID mismatch")
            coordinates = accepted["coordinates_deg"]
            for source_field, accepted_field in (
                ("ra_provabgs", "paired_ra"),
                ("dec_provabgs", "paired_dec"),
            ):
                observed = _finite_float(
                    source_row[source_field], label=f"candidate {index} source {source_field}"
                )
                expected = float(coordinates[accepted_field])
                if abs(observed - expected) > 1e-9:
                    raise PackagingValidationError(
                        f"candidate {index} paired source {source_field} mismatch"
                    )

    if set(previews) != set(range(len(rows))):
        raise PackagingValidationError("not every accepted row produced an image preview")
    return previews


def _validate_raw_arrays(
    manifest_path: Path,
    manifest: dict[str, Any],
    accepted_rows: list[dict[str, Any]],
    contract: dict[str, tuple[np.dtype[Any], tuple[int, ...]]],
    *,
    destination_root: str,
) -> list[_RawAsset]:
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(accepted_rows):
        raise PackagingValidationError(f"{manifest_path} candidate count mismatch")
    for index, (raw, accepted) in enumerate(zip(candidates, accepted_rows, strict=True)):
        for field in ("candidate_index", "paired_targetid", "caption_object_id"):
            if raw.get(field) != accepted.get(field):
                raise PackagingValidationError(
                    f"candidate {index} raw/accepted identity mismatch for {field}"
                )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(contract):
        raise PackagingValidationError(
            f"{manifest_path} outputs must be exactly {sorted(contract)}"
        )
    assets: list[_RawAsset] = []
    for key, (expected_dtype, trailing_shape) in contract.items():
        entry = outputs[key]
        if not isinstance(entry, dict):
            raise PackagingValidationError(f"raw output {key} has no manifest entry")
        source = _safe_child(manifest_path.parent, entry.get("path", ""), label=key)
        expected_shape = (len(accepted_rows), *trailing_shape)
        try:
            array = np.load(source, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise PackagingValidationError(f"cannot load raw array {source}: {exc}") from exc
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise PackagingValidationError(
                f"{key} must have shape {expected_shape} and dtype {expected_dtype}, "
                f"got {array.shape} and {array.dtype}"
            )
        if (
            entry.get("shape") != list(expected_shape)
            or np.dtype(entry.get("dtype")) != expected_dtype
        ):
            raise PackagingValidationError(f"{key} header disagrees with its source manifest")
        if entry.get("size_bytes") != source.stat().st_size:
            raise PackagingValidationError(f"{key} size disagrees with its source manifest")
        digest = sha256_file(source)
        if entry.get("sha256") != digest:
            raise PackagingValidationError(f"{key} SHA-256 disagrees with its source manifest")

        if array.dtype.kind == "f":
            for start in range(0, len(array), 32):
                chunk = np.asarray(array[start : start + 32])
                if not np.isfinite(chunk).all():
                    raise PackagingValidationError(f"{key} contains non-finite values")
                if key.endswith("_ivar") and np.any(chunk < 0):
                    raise PackagingValidationError(f"{key} contains negative inverse variance")
        if key == "spectrum_wavelength" and np.any(np.diff(array, axis=1) <= 0):
            raise PackagingValidationError("spectrum_wavelength is not strictly increasing")

        destination = f"{destination_root}/{source.name}"
        assets.append(
            _RawAsset(
                key=key,
                source=source,
                destination=destination,
                dtype=str(array.dtype),
                shape=tuple(int(value) for value in array.shape),
                sha256=digest,
                size_bytes=source.stat().st_size,
                source_dataset=entry.get("source_dataset"),
            )
        )
    return assets


def _copy_or_link(source: Path, destination: Path, *, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"auto", "hardlink", "copy"}:
        raise ValueError("link_mode must be one of: auto, hardlink, copy")
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def _portable_raw_manifest(
    *,
    kind: str,
    source_manifest: dict[str, Any],
    assets: list[_RawAsset],
    row_count: int,
) -> dict[str, Any]:
    source = source_manifest.get("source", {})
    portable_source = {
        key: source[key]
        for key in ("url", "etag", "length_bytes", "hdf5_row_count")
        if key in source
    }
    return {
        "schema_version": 1,
        "artifact_type": f"astroprism_portable_raw_{kind}",
        "row_count": row_count,
        "array_index_contract": "candidate_index is the first-axis index in every array",
        "source": portable_source,
        **({"bands": source_manifest.get("bands")} if kind == "images" else {}),
        **({"spectrum": source_manifest.get("spectrum")} if kind == "spectra" else {}),
        "outputs": {
            asset.key: {
                "path": Path(asset.destination).name,
                "dtype": asset.dtype,
                "shape": list(asset.shape),
                "size_bytes": asset.size_bytes,
                "sha256": asset.sha256,
                "mmap_safe": True,
                **(
                    {"source_dataset": asset.source_dataset}
                    if asset.source_dataset is not None
                    else {}
                ),
            }
            for asset in assets
        },
    }


def _validate_bundle_manifest(bundle: Path, records_path: Path, row_count: int) -> dict[str, Any]:
    manifest = _read_json(bundle / "manifest.json", label="spectrum caption bundle manifest")
    if manifest.get("row_count") != row_count:
        raise PackagingValidationError(f"{bundle} manifest row count mismatch")
    entry = manifest.get("outputs", {}).get("captions.jsonl")
    if not isinstance(entry, dict) or entry.get("sha256") != sha256_file(records_path):
        raise PackagingValidationError(f"{bundle} caption checksum mismatch")
    return manifest


def _load_spectrum_bundle(
    bundle: Path,
    accepted_rows: list[dict[str, Any]],
    *,
    expected_indices: set[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, bytes]]:
    records_path = bundle / "captions.jsonl"
    records = _read_jsonl(records_path, label="spectrum caption candidates")
    manifest = _validate_bundle_manifest(bundle, records_path, len(records))
    by_index: dict[int, dict[str, Any]] = {}
    plots: dict[int, bytes] = {}
    for record in records:
        try:
            index = int(record["candidate_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PackagingValidationError(
                "spectrum caption has an invalid candidate_index"
            ) from exc
        if index in by_index or index not in expected_indices:
            raise PackagingValidationError(
                f"spectrum bundle has unexpected or duplicate candidate {index}"
            )
        accepted = accepted_rows[index]
        if record.get("split") != accepted["split"]:
            raise PackagingValidationError(f"candidate {index} spectrum split mismatch")
        for field in ("paired_targetid", "caption_object_id", "source_group_id"):
            if record.get(field) != accepted.get(field):
                raise PackagingValidationError(
                    f"candidate {index} spectrum/accepted identity mismatch for {field}"
                )
        caption = record.get("spectrum_caption")
        if not isinstance(caption, dict) or caption.get("status") != "accepted":
            raise PackagingValidationError(f"candidate {index} spectrum caption is not accepted")
        _require_nonempty_text(caption.get("original"), label=f"candidate {index} spectrum caption")
        review = record.get("review")
        if (
            not isinstance(review, dict)
            or review.get("status") != "accepted"
            or review.get("decision_source") != "user_instruction"
        ):
            raise PackagingValidationError(
                f"candidate {index} spectrum caption lacks an accepted user-directed review"
            )
        policy = record.get("training_policy")
        training_allowed = accepted["split"] == "train"
        if not isinstance(policy, dict) or policy.get("dataset_ready") is not True:
            raise PackagingValidationError(
                f"candidate {index} spectrum caption is not dataset_ready"
            )
        if policy.get("gradient_training_allowed") is not training_allowed:
            raise PackagingValidationError(
                f"candidate {index} gradient_training_allowed does not match its split"
            )
        if policy.get("eligible_for_training") is not training_allowed:
            raise PackagingValidationError(
                f"candidate {index} eligible_for_training does not match its split"
            )
        if policy.get("eligible_for_evaluation") is not (not training_allowed):
            raise PackagingValidationError(
                f"candidate {index} eligible_for_evaluation does not match its split"
            )

        relative_plot = f"plots/candidate_{index:04d}.png"
        plot_path = _safe_child(bundle, relative_plot, label="spectrum plot")
        plot_bytes = _validate_png(plot_path.read_bytes(), label=f"candidate {index} spectrum plot")
        plot_entry = manifest.get("outputs", {}).get(relative_plot)
        if not isinstance(plot_entry, dict) or plot_entry.get("sha256") != sha256_bytes(plot_bytes):
            raise PackagingValidationError(f"candidate {index} spectrum plot checksum mismatch")
        by_index[index] = record
        plots[index] = plot_bytes

    if set(by_index) != expected_indices:
        missing = sorted(expected_indices - set(by_index))
        raise PackagingValidationError(
            f"spectrum bundle is incomplete; missing candidate indices {missing[:10]}"
        )
    return by_index, plots


def _portable_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: source[key]
        for key in ("repo_id", "revision", "shard", "row", "sha256", "survey")
        if key in source
    }


def _portable_candidate(row: dict[str, Any]) -> dict[str, Any]:
    index = int(row["candidate_index"])
    return {
        "schema_version": 1,
        "candidate_index": index,
        "paired_targetid": row["paired_targetid"],
        "caption_object_id": row["caption_object_id"],
        "source_group_id": row["source_group_id"],
        "split": row["split"],
        "coordinates_deg": row["coordinates_deg"],
        "separation_arcsec": row["separation_arcsec"],
        "acceptance_status": row["acceptance_status"],
        "caption": row["caption"],
        "caption_source": _portable_source(row["caption_source"]),
        "paired_source": _portable_source(row["paired_source"]),
        "raw_assets": {
            "image_manifest": "assets/raw/images/MANIFEST.json",
            "spectrum_manifest": "assets/raw/spectra/MANIFEST.json",
            "array_index": index,
        },
    }


def _image_parquet_rows(
    rows: list[dict[str, Any]], previews: dict[int, bytes], split: str
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["split"] != split:
            continue
        index = int(row["candidate_index"])
        result.append(
            {
                "candidate_index": index,
                "paired_targetid": row["paired_targetid"],
                "caption_object_id": row["caption_object_id"],
                "source_group_id": row["source_group_id"],
                "split": split,
                "image": {"bytes": previews[index], "path": None},
                "caption": row["caption"]["original"],
                "caption_summary": row["caption"]["summary_original"],
                "caption_status": row["caption"]["status"],
                "training_allowed": split == "train",
                "role": "alignment_training" if split == "train" else "held_out_evaluation",
                "raw_array_index": index,
                "raw_image_manifest": "assets/raw/images/MANIFEST.json",
                "paired_ra_deg": float(row["coordinates_deg"]["paired_ra"]),
                "paired_dec_deg": float(row["coordinates_deg"]["paired_dec"]),
                "caption_ra_deg": float(row["coordinates_deg"]["caption_ra"]),
                "caption_dec_deg": float(row["coordinates_deg"]["caption_dec"]),
                "separation_arcsec": float(row["separation_arcsec"]),
                "caption_source_repo": row["caption_source"]["repo_id"],
                "caption_source_revision": row["caption_source"]["revision"],
                "caption_source_shard": row["caption_source"]["shard"],
                "caption_source_row": int(row["caption_source"]["row"]),
            }
        )
    return result


def _spectrum_parquet_rows(
    records: dict[int, dict[str, Any]],
    plots: dict[int, bytes],
    *,
    split: str,
) -> list[dict[str, Any]]:
    result = []
    for index in sorted(records):
        record = records[index]
        if record["split"] != split:
            continue
        teacher = record.get("teacher_reference", {})
        catalog = teacher.get("catalog_context", {}) if isinstance(teacher, dict) else {}
        redshift = catalog.get("redshift")
        redshift_error = catalog.get("redshift_error")
        claims = record.get("audit_sidecar", {}).get("claims", [])
        generation = record.get("generation", {})
        result.append(
            {
                "candidate_index": index,
                "paired_targetid": record["paired_targetid"],
                "caption_object_id": record["caption_object_id"],
                "source_group_id": record["source_group_id"],
                "split": split,
                "spectrum_plot": {"bytes": plots[index], "path": None},
                "caption": record["spectrum_caption"]["original"],
                "caption_status": "accepted",
                "review_status": "accepted",
                "reviewed": True,
                "dataset_ready": True,
                "training_allowed": split == "train",
                "role": "alignment_training" if split == "train" else "held_out_evaluation",
                "raw_array_index": index,
                "raw_spectrum_manifest": "assets/raw/spectra/MANIFEST.json",
                "wavelength_frame": record.get("student_input", {}).get(
                    "wavelength_frame", "observed_vacuum"
                ),
                "spectrum_length": int(
                    record.get("student_input", {}).get("spectrum_length", 7781)
                ),
                "catalog_redshift": None if redshift is None else float(redshift),
                "catalog_redshift_error": (
                    None if redshift_error is None else float(redshift_error)
                ),
                "claims_json": json.dumps(
                    claims, sort_keys=True, ensure_ascii=False, allow_nan=False
                ),
                "generation_json": json.dumps(
                    generation, sort_keys=True, ensure_ascii=False, allow_nan=False
                ),
            }
        )
    return result


def _write_image_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        from datasets import Dataset, Features, Image, Value, disable_progress_bars
    except ImportError as exc:  # pragma: no cover - deployment environment check
        raise RuntimeError("datasets is required to write Hugging Face Parquet features") from exc

    disable_progress_bars()
    features = Features(
        {
            "candidate_index": Value("int32"),
            "paired_targetid": Value("string"),
            "caption_object_id": Value("string"),
            "source_group_id": Value("string"),
            "split": Value("string"),
            "image": Image(decode=False),
            "caption": Value("string"),
            "caption_summary": Value("string"),
            "caption_status": Value("string"),
            "training_allowed": Value("bool"),
            "role": Value("string"),
            "raw_array_index": Value("int32"),
            "raw_image_manifest": Value("string"),
            "paired_ra_deg": Value("float64"),
            "paired_dec_deg": Value("float64"),
            "caption_ra_deg": Value("float64"),
            "caption_dec_deg": Value("float64"),
            "separation_arcsec": Value("float64"),
            "caption_source_repo": Value("string"),
            "caption_source_revision": Value("string"),
            "caption_source_shard": Value("string"),
            "caption_source_row": Value("int32"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows, features=features).to_parquet(str(path), compression="zstd")
    _validate_hf_image_feature(path, "image")


def _write_spectrum_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        from datasets import Dataset, Features, Image, Value, disable_progress_bars
    except ImportError as exc:  # pragma: no cover - deployment environment check
        raise RuntimeError("datasets is required to write Hugging Face Parquet features") from exc

    disable_progress_bars()
    features = Features(
        {
            "candidate_index": Value("int32"),
            "paired_targetid": Value("string"),
            "caption_object_id": Value("string"),
            "source_group_id": Value("string"),
            "split": Value("string"),
            "spectrum_plot": Image(decode=False),
            "caption": Value("string"),
            "caption_status": Value("string"),
            "review_status": Value("string"),
            "reviewed": Value("bool"),
            "dataset_ready": Value("bool"),
            "training_allowed": Value("bool"),
            "role": Value("string"),
            "raw_array_index": Value("int32"),
            "raw_spectrum_manifest": Value("string"),
            "wavelength_frame": Value("string"),
            "spectrum_length": Value("int32"),
            "catalog_redshift": Value("float64"),
            "catalog_redshift_error": Value("float64"),
            "claims_json": Value("string"),
            "generation_json": Value("string"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows, features=features).to_parquet(str(path), compression="zstd")
    _validate_hf_image_feature(path, "spectrum_plot")


def _validate_hf_image_feature(path: Path, field: str) -> None:
    try:
        import pyarrow.parquet as pq
        from datasets import Features, Image
    except ImportError as exc:  # pragma: no cover - deployment environment check
        raise RuntimeError(
            "datasets and pyarrow are required to validate Parquet features"
        ) from exc
    features = Features.from_arrow_schema(pq.read_schema(path))
    feature = features.get(field)
    if not isinstance(feature, Image) or feature.decode is not False:
        raise PackagingValidationError(
            f"{path} did not preserve {field} as datasets.Image(decode=False)"
        )


def _dataset_card(
    *,
    split_counts: dict[str, int],
) -> str:
    configs = [
        "- config_name: image_caption\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: data/image_caption/train.parquet\n"
        "  - split: validation\n"
        "    path: data/image_caption/validation.parquet\n"
        "  - split: test\n"
        "    path: data/image_caption/test.parquet",
        "- config_name: spectrum_caption\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: data/spectrum_caption/train.parquet\n"
        "  - split: validation\n"
        "    path: data/spectrum_caption/validation.parquet\n"
        "  - split: test\n"
        "    path: data/spectrum_caption/test.parquet",
    ]
    return f"""---
license: other
language:
- en
task_categories:
- image-to-text
tags:
- astronomy
- multimodal
- spectra
- private-research-dataset
pretty_name: AstroPRISM image and spectrum caption pairs
configs:
{chr(10).join(configs)}
---

# AstroPRISM

Private research staging data for aligning PRISM with astronomical images and spectra. The
package contains {sum(split_counts.values())} object-level candidates with a fixed,
source-group-safe split ({split_counts["train"]} train / {split_counts["validation"]}
validation / {split_counts["test"]} test).

## Configs and training status

| Config | Splits | Rows | Training status |
|---|---|---:|---|
| `image_caption` | train, validation, test | {sum(split_counts.values())} | Only train rows are allowed; captions are accepted weak supervision. |
| `spectrum_caption` | train, validation, test | {sum(split_counts.values())} | All captions are accepted, reviewed, and dataset-ready; only train rows allow gradient training. |

Every spectrum-caption row has `caption_status=accepted`, `review_status=accepted`,
`reviewed=true`, and `dataset_ready=true`. Dataset readiness means the pair is approved for its
assigned split; it does not grant gradient-training permission. Only train rows have
`training_allowed=true`. Validation and test rows have `role=held_out_evaluation` and
`training_allowed=false` so they remain clean evaluation supervision.

The embedded `image` and `spectrum_plot` columns are inspection previews. Model inputs are the
calibrated arrays in `assets/raw/images/` and `assets/raw/spectra/`. Released AION embeddings,
download caches, and encoder weights are intentionally excluded.

## Direct raw-array indexing

`candidate_index` and `raw_array_index` are identical. They index axis 0 of every monolithic
NumPy array. This remains true in all Parquet splits; indices are deliberately not renumbered.

```python
from huggingface_hub import hf_hub_download
import numpy as np

path = hf_hub_download(
    repo_id="Sand33p/AstroPRISM",
    repo_type="dataset",
    filename="assets/raw/images/image_array.npy",
)
images = np.load(path, mmap_mode="r")
image = images[row["candidate_index"]]  # float32 [4, 160, 160]
```

Load pair metadata with an authenticated Hugging Face session:

```python
from datasets import load_dataset

images = load_dataset("Sand33p/AstroPRISM", "image_caption")
spectra = load_dataset("Sand33p/AstroPRISM", "spectrum_caption")
```

See `metadata/candidates.jsonl` for the authoritative cross-modal identity and split mapping,
`provenance/SOURCES.json` for immutable upstream pins, `LICENSES.md` for source-specific terms,
and `SHA256SUMS` for integrity verification.

## Important limitations

- Image descriptions are synthetic weak supervision and can contain unsupported interpretation.
- Spectrum captions were generated from deterministic measured facts and accepted by user instruction.
- Spectrum flux units were not retained explicitly in the extracted local arrays.
- Object-level source groups must stay in their assigned split across every derived task.
- Raw scientific data retain their original survey terms and required acknowledgements.
"""


def _licenses() -> str:
    return """# Licensing and acknowledgements

This compilation does not apply one blanket license to all payloads. Upstream terms continue to
govern each component.

- Caption text and RGB review previews come from `astronolan/galaxy-descriptions` at revision
  `6890dc0c8fc793867c07a6ce4fce11e51c167d6e`. Its dataset card releases descriptions and
  embeddings under MIT while stating that underlying Legacy Surveys data retain the original
  data-use policy and required acknowledgements:
  https://huggingface.co/datasets/astronolan/galaxy-descriptions
- Cross-match metadata and source locators come from `EiffL/desi_x_ls_aion` at revision
  `57f7c7575260811ea076163bc743787d75b9aa49`, whose dataset card is CC-BY-4.0:
  https://huggingface.co/datasets/EiffL/desi_x_ls_aion
- Calibrated image and spectrum arrays were selected from the official AION tutorial HDF5 source.
  They derive from Legacy Surveys, DESI, PROVABGS, and Multimodal Universe data; retain all
  original data-use policies, citations, and acknowledgements.
- Generated spectrum captions and plots are research artifacts. They do not change the terms of
  the underlying observations or catalog values.

Users are responsible for reviewing current upstream policies before redistribution or
publication. Cite the Legacy Surveys, DESI/PROVABGS, Multimodal Universe, AION-1, AION-Search,
and AstroSpecLM sources as applicable.
"""


def _sources_manifest(
    accepted_manifest: dict[str, Any],
    raw_images_manifest: dict[str, Any],
    raw_spectra_manifest: dict[str, Any],
    roster_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "roster": {
            "path": "provenance/overlap_candidates_646.csv.gz",
            "sha256": roster_sha256,
        },
        "caption_source": {
            "repo_id": CAPTION_REPO_ID,
            "revision": CAPTION_REVISION,
            "config": "default",
            "split": "default",
            "survey": "legacy_dr10",
        },
        "paired_source": {
            "repo_id": PAIRED_REPO_ID,
            "revision": PAIRED_REVISION,
        },
        "raw_observation_source": {
            "url": raw_images_manifest.get("source", {}).get("url", RAW_SOURCE_URL),
            "etag": raw_images_manifest.get("source", {}).get("etag"),
            "length_bytes": raw_images_manifest.get("source", {}).get("length_bytes"),
            "image_manifest_source_sha256": None,
            "spectrum_manifest_source_sha256": None,
        },
        "split_policy": accepted_manifest.get("splits", {}),
        "spectrum_caption_method": {
            "reference": ASTROSPECLM_PAPER,
            "status": "synthetic_pending_human_review",
        },
        "raw_source_consistency": (
            raw_images_manifest.get("source", {}).get("etag")
            == raw_spectra_manifest.get("source", {}).get("etag")
            and raw_images_manifest.get("source", {}).get("length_bytes")
            == raw_spectra_manifest.get("source", {}).get("length_bytes")
        ),
    }


def _all_file_entries(root: Path, *, exclude: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = exclude or set()
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        result[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _write_integrity_files(
    root: Path,
    *,
    config_counts: dict[str, dict[str, int]],
    link_actions: dict[str, str],
) -> None:
    payload = _all_file_entries(root, exclude={"MANIFEST.json", "SHA256SUMS"})
    manifest = {
        "schema_version": 1,
        "artifact_type": "astroprism_huggingface_dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "visibility_intent": "private",
        "array_index_contract": "candidate_index == raw_array_index == first-axis array index",
        "configs": config_counts,
        "raw_asset_staging_method": link_actions,
        "excluded": [
            "released AION image embeddings",
            "released AION spectrum embeddings",
            "Hugging Face download caches",
            "model weights",
        ],
        "files": payload,
        "integrity_note": "MANIFEST.json and SHA256SUMS are excluded from this self-listing.",
    }
    _write_json(root / "MANIFEST.json", manifest)
    checksummed = _all_file_entries(root, exclude={"SHA256SUMS"})
    lines = [f"{entry['sha256']}  {name}" for name, entry in sorted(checksummed.items())]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_portability(root: Path) -> None:
    forbidden_bytes = [item.encode("utf-8") for item in FORBIDDEN_PORTABLE_TEXT]
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if any(part == ".cache" for part in Path(relative).parts):
            raise PackagingValidationError(f"cache path leaked into package: {relative}")
        if "embedding" in path.name.lower() and path.suffix == ".npy":
            raise PackagingValidationError(f"released embedding leaked into package: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.suffix.lower() == ".parquet":
            content = path.read_bytes()
            for forbidden, encoded in zip(FORBIDDEN_PORTABLE_TEXT, forbidden_bytes, strict=True):
                if encoded in content:
                    raise PackagingValidationError(
                        f"non-portable text {forbidden!r} leaked into {relative}"
                    )


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


def prepare_hf_dataset(config: HFDatasetConfig) -> dict[str, Any]:
    """Validate sources and atomically stage a portable Hugging Face dataset tree."""

    expected_splits = config.expected_split_counts or dict(EXPECTED_SPLIT_COUNTS)
    if sum(expected_splits.values()) != config.expected_count:
        raise ValueError("expected split counts must sum to expected_count")
    if config.link_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError("link_mode must be one of: auto, hardlink, copy")

    output = Path(config.output_dir).resolve()
    if _path_exists(output) and not config.overwrite:
        raise FileExistsError(f"output already exists: {output}")

    roster = Path(config.roster_path)
    if not roster.is_file():
        raise PackagingValidationError(f"roster does not exist: {roster}")
    roster_sha256 = sha256_file(roster)
    if config.expected_roster_sha256 is not None and roster_sha256 != config.expected_roster_sha256:
        raise PackagingValidationError(
            f"roster SHA-256 mismatch: expected {config.expected_roster_sha256}, got {roster_sha256}"
        )
    with gzip.open(roster, "rt", encoding="utf-8", newline="") as handle:
        roster_rows = sum(1 for _line in handle) - 1
    if roster_rows != config.expected_count:
        raise PackagingValidationError(
            f"roster must contain {config.expected_count} data rows, got {roster_rows}"
        )

    accepted_rows = _read_jsonl(Path(config.accepted_candidates_path), label="accepted candidates")
    _validate_accepted_rows(
        accepted_rows,
        expected_count=config.expected_count,
        expected_split_counts=expected_splits,
        caption_repo_id=config.caption_repo_id,
        caption_revision=config.caption_revision,
        paired_repo_id=config.paired_repo_id,
        paired_revision=config.paired_revision,
    )
    accepted_manifest = _read_json(
        Path(config.accepted_manifest_path), label="accepted candidate manifest"
    )
    _validate_accepted_manifest(
        Path(config.accepted_manifest_path),
        accepted_manifest,
        Path(config.accepted_candidates_path),
        accepted_rows,
        expected_splits,
        roster_sha256,
    )

    image_previews = _validate_source_rows(
        accepted_rows,
        caption_root=Path(config.caption_root),
        paired_root=Path(config.paired_root),
    )
    raw_images_manifest = _read_json(
        Path(config.raw_images_manifest_path), label="raw image manifest"
    )
    raw_spectra_manifest = _read_json(
        Path(config.raw_spectra_manifest_path), label="raw spectrum manifest"
    )
    image_assets = _validate_raw_arrays(
        Path(config.raw_images_manifest_path),
        raw_images_manifest,
        accepted_rows,
        IMAGE_ARRAY_CONTRACT,
        destination_root="assets/raw/images",
    )
    spectrum_assets = _validate_raw_arrays(
        Path(config.raw_spectra_manifest_path),
        raw_spectra_manifest,
        accepted_rows,
        SPECTRUM_ARRAY_CONTRACT,
        destination_root="assets/raw/spectra",
    )

    spectrum_indices = {int(row["candidate_index"]) for row in accepted_rows}
    spectrum_records, spectrum_plots = _load_spectrum_bundle(
        Path(config.spectrum_caption_bundle),
        accepted_rows,
        expected_indices=spectrum_indices,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    committed = False
    try:
        link_actions: dict[str, str] = {}
        for asset in [*image_assets, *spectrum_assets]:
            destination = staging / asset.destination
            action = _copy_or_link(asset.source, destination, mode=config.link_mode)
            link_actions[asset.destination] = action
            if sha256_file(destination) != asset.sha256:
                raise PackagingValidationError(
                    f"staged raw asset checksum mismatch: {asset.destination}"
                )

        _write_json(
            staging / "assets/raw/images/MANIFEST.json",
            _portable_raw_manifest(
                kind="images",
                source_manifest=raw_images_manifest,
                assets=image_assets,
                row_count=len(accepted_rows),
            ),
        )
        _write_json(
            staging / "assets/raw/spectra/MANIFEST.json",
            _portable_raw_manifest(
                kind="spectra",
                source_manifest=raw_spectra_manifest,
                assets=spectrum_assets,
                row_count=len(accepted_rows),
            ),
        )
        _write_jsonl(
            staging / "metadata/candidates.jsonl",
            (_portable_candidate(row) for row in accepted_rows),
        )
        _copy_or_link(
            roster,
            staging / "provenance/overlap_candidates_646.csv.gz",
            mode=config.link_mode,
        )
        sources = _sources_manifest(
            accepted_manifest,
            raw_images_manifest,
            raw_spectra_manifest,
            roster_sha256,
        )
        sources["raw_observation_source"]["image_manifest_source_sha256"] = sha256_file(
            Path(config.raw_images_manifest_path)
        )
        sources["raw_observation_source"]["spectrum_manifest_source_sha256"] = sha256_file(
            Path(config.raw_spectra_manifest_path)
        )
        _write_json(staging / "provenance/SOURCES.json", sources)

        for split in ("train", "validation", "test"):
            pair_rows = _image_parquet_rows(accepted_rows, image_previews, split)
            if len(pair_rows) != expected_splits[split]:
                raise PackagingValidationError(f"image Parquet row count mismatch for {split}")
            _write_image_parquet(staging / f"data/image_caption/{split}.parquet", pair_rows)

        spectrum_counts: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            pair_rows = _spectrum_parquet_rows(
                spectrum_records,
                spectrum_plots,
                split=split,
            )
            if len(pair_rows) != expected_splits[split]:
                raise PackagingValidationError(f"spectrum Parquet row count mismatch for {split}")
            spectrum_counts[split] = len(pair_rows)
            _write_spectrum_parquet(
                staging / f"data/spectrum_caption/{split}.parquet",
                pair_rows,
            )

        (staging / "README.md").write_text(
            _dataset_card(split_counts=expected_splits),
            encoding="utf-8",
        )
        (staging / "LICENSES.md").write_text(_licenses(), encoding="utf-8")
        config_counts = {
            "image_caption": dict(expected_splits),
            "spectrum_caption": dict(spectrum_counts),
        }
        _write_integrity_files(staging, config_counts=config_counts, link_actions=link_actions)
        _validate_portability(staging)
        _commit_staging(staging, output, overwrite=config.overwrite)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)

    return {
        "status": "prepared",
        "output": str(output),
        "configs": config_counts,
        "candidate_count": len(accepted_rows),
        "raw_array_index": "candidate_index",
        "spectrum_caption_count": len(spectrum_records),
        "spectrum_captions_reviewed": True,
        "manifest": str(output / "MANIFEST.json"),
        "sha256sums": str(output / "SHA256SUMS"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/hf_upload/AstroPRISM",
    )
    parser.add_argument(
        "--accepted-candidates",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1a_accepted/candidates.jsonl",
    )
    parser.add_argument(
        "--accepted-manifest",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1a_accepted/manifest.json",
    )
    parser.add_argument(
        "--raw-images-manifest",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/raw_images/manifest.json",
    )
    parser.add_argument(
        "--raw-spectra-manifest",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/raw_spectra/manifest.json",
    )
    parser.add_argument(
        "--caption-root",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/hf/datasets/galaxy-descriptions",
    )
    parser.add_argument(
        "--paired-root",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/hf/datasets/desi_x_ls_aion",
    )
    parser.add_argument(
        "--spectrum-caption-bundle",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1b_spectrum_caption_accepted",
        help="Complete promoted train/validation/test spectrum-caption bundle.",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=REPO_ROOT / "docs/astroprism/data/overlap_candidates_646.csv.gz",
    )
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_hf_dataset(
        HFDatasetConfig(
            output_dir=args.output,
            accepted_candidates_path=args.accepted_candidates,
            accepted_manifest_path=args.accepted_manifest,
            raw_images_manifest_path=args.raw_images_manifest,
            raw_spectra_manifest_path=args.raw_spectra_manifest,
            caption_root=args.caption_root,
            paired_root=args.paired_root,
            spectrum_caption_bundle=args.spectrum_caption_bundle,
            roster_path=args.roster,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

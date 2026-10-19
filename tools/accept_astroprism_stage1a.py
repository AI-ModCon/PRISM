#!/usr/bin/env python3
"""Promote reviewed AstroPRISM Stage 1A candidates into a training artifact.

The source candidate artifact remains immutable. This tool copies its 646
records in roster order, marks unchanged captions as released, groups nearby
cutouts with the deterministic 'skycc60-v1' policy, and assigns exact
516/65/65 train/validation/test splits without separating a group.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CANDIDATES = 646
OVERLAP_RADIUS_ARCSEC = 60.0
GROUP_POLICY = "skycc60-v1"
SPLIT_POLICY = "sha256-greedy-whole-group-v1"
SPLIT_RANK_SEED = "astroprism-stage1a-split-v1"
SPLIT_COUNTS = {"train": 516, "validation": 65, "test": 65}
ACCEPTANCE_BASIS = "user_directive_assume_reviewed_and_accepted"


class AcceptanceValidationError(ValueError):
    """Raised when promotion would violate the accepted-artifact contract."""


@dataclass(frozen=True)
class AcceptanceConfig:
    """Inputs for deterministic Stage 1A candidate promotion."""

    source_manifest_path: Path
    output_dir: Path
    overwrite: bool = False
    expected_count: int = EXPECTED_CANDIDATES
    split_counts: dict[str, int] | None = None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise AcceptanceValidationError(f"{kind} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceValidationError(f"cannot read {kind} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcceptanceValidationError(f"{kind} must contain a JSON object: {path}")
    return value


def _resolve_output(manifest_path: Path, output: dict[str, Any], *, name: str) -> Path:
    value = output.get("path")
    if not isinstance(value, str) or not value:
        raise AcceptanceValidationError(f"source manifest output {name!r} has no path")
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AcceptanceValidationError(f"source candidates do not exist: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise AcceptanceValidationError(
                        f"source candidates contain a blank row at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AcceptanceValidationError(
                        f"source candidate line {line_number} must be an object"
                    )
                records.append(value)
    except json.JSONDecodeError as exc:
        raise AcceptanceValidationError(
            f"cannot parse source candidates {path} at line {exc.lineno}: {exc.msg}"
        ) from exc
    return records


def _paired_coordinate(record: dict[str, Any], *, index: int) -> tuple[float, float]:
    coordinates = record.get("coordinates_deg")
    if not isinstance(coordinates, dict):
        raise AcceptanceValidationError(f"candidate {index} has no coordinates_deg object")
    try:
        ra = float(coordinates["paired_ra"])
        dec = float(coordinates["paired_dec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AcceptanceValidationError(
            f"candidate {index} has invalid paired_ra/paired_dec"
        ) from exc
    if not math.isfinite(ra) or not math.isfinite(dec):
        raise AcceptanceValidationError(f"candidate {index} coordinates must be finite")
    if not 0.0 <= ra < 360.0 or not -90.0 <= dec <= 90.0:
        raise AcceptanceValidationError(f"candidate {index} coordinates are out of range")
    return math.radians(ra), math.radians(dec)


def _separation_arcsec(first: tuple[float, float], second: tuple[float, float]) -> float:
    ra1, dec1 = first
    ra2, dec2 = second
    haversine = (
        math.sin((dec2 - dec1) / 2.0) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2.0) ** 2
    )
    return 2.0 * math.asin(min(1.0, math.sqrt(haversine))) * 180.0 / math.pi * 3600.0


def _validate_source_records(
    records: list[dict[str, Any]], *, expected_count: int
) -> list[tuple[float, float]]:
    if len(records) != expected_count:
        raise AcceptanceValidationError(
            f"source artifact must contain exactly {expected_count} rows, got {len(records)}"
        )
    paired_ids: set[str] = set()
    caption_ids: set[str] = set()
    coordinates: list[tuple[float, float]] = []
    for index, record in enumerate(records):
        if record.get("candidate_index") != index:
            raise AcceptanceValidationError(
                f"candidate row {index} has candidate_index={record.get('candidate_index')!r}; "
                "roster order must be preserved"
            )
        paired_id = str(record.get("paired_targetid", "")).strip()
        caption_id = str(record.get("caption_object_id", "")).strip()
        if not paired_id or paired_id in paired_ids:
            raise AcceptanceValidationError(
                f"candidate {index} has an empty or duplicate paired_targetid"
            )
        if not caption_id or caption_id in caption_ids:
            raise AcceptanceValidationError(
                f"candidate {index} has an empty or duplicate caption_object_id"
            )
        paired_ids.add(paired_id)
        caption_ids.add(caption_id)

        caption = record.get("caption")
        if not isinstance(caption, dict):
            raise AcceptanceValidationError(f"candidate {index} has no caption object")
        original = caption.get("original")
        if not isinstance(original, str) or not original.strip():
            raise AcceptanceValidationError(f"candidate {index} has no non-empty caption.original")
        if str(record.get("acceptance_status", "")).strip() != "pending_review":
            raise AcceptanceValidationError(
                f"candidate {index} is not pending_review; refusing to re-promote it"
            )
        if str(caption.get("status", "")).strip() != "unreviewed":
            raise AcceptanceValidationError(
                f"candidate {index} caption is not unreviewed; refusing to re-promote it"
            )
        if record.get("source_group_id") is not None or record.get("split") is not None:
            raise AcceptanceValidationError(
                f"candidate {index} already has a source group or split"
            )
        image_embedding = record.get("image_embedding")
        if not isinstance(image_embedding, dict) or not image_embedding.get("file"):
            raise AcceptanceValidationError(
                f"candidate {index} has no image_embedding.file provenance"
            )
        coordinates.append(_paired_coordinate(record, index=index))
    return coordinates


def _source_groups(
    records: list[dict[str, Any]], coordinates: list[tuple[float, float]]
) -> list[tuple[str, tuple[int, ...]]]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    for first in range(len(records)):
        for second in range(first + 1, len(records)):
            if _separation_arcsec(coordinates[first], coordinates[second]) <= OVERLAP_RADIUS_ARCSEC:
                union(first, second)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)

    groups: list[tuple[str, tuple[int, ...]]] = []
    for members in components.values():
        identities = sorted(f"desi:{records[index]['paired_targetid']}" for index in members)
        payload = ("\n".join(identities) + "\n").encode("utf-8")
        group_id = f"{GROUP_POLICY}:{hashlib.sha256(payload).hexdigest()[:24]}"
        groups.append((group_id, tuple(sorted(members))))
    return groups


def _assign_splits(
    groups: list[tuple[str, tuple[int, ...]]], split_counts: dict[str, int]
) -> dict[str, str]:
    ranked = sorted(
        groups,
        key=lambda item: (
            hashlib.sha256(f"{SPLIT_RANK_SEED}\0{item[0]}".encode()).digest(),
            item[0],
        ),
    )
    remaining = ranked
    assignments: dict[str, str] = {}
    for split in ("test", "validation"):
        capacity = split_counts[split]
        deferred: list[tuple[str, tuple[int, ...]]] = []
        for group_id, members in remaining:
            if len(members) <= capacity:
                assignments[group_id] = split
                capacity -= len(members)
            else:
                deferred.append((group_id, members))
        if capacity:
            raise AcceptanceValidationError(
                f"whole-group allocation could not fill {split!r}; {capacity} rows remain"
            )
        remaining = deferred

    for group_id, _members in remaining:
        assignments[group_id] = "train"
    train_rows = sum(
        len(members) for group_id, members in groups if assignments[group_id] == "train"
    )
    if train_rows != split_counts["train"]:
        raise AcceptanceValidationError(
            f"whole-group allocation produced {train_rows} train rows, "
            f"expected {split_counts['train']}"
        )
    return assignments


def _mapping_sha256(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        f"{index}\t{record['source_group_id']}\t{record['split']}\n"
        for index, record in enumerate(records)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _commit_staging(staging: Path, output: Path, *, overwrite: bool) -> None:
    if not output.exists() and not output.is_symlink():
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


def accept_stage1a(config: AcceptanceConfig) -> dict[str, Any]:
    """Write a separate accepted artifact and return its reproducibility summary."""

    split_counts = dict(config.split_counts or SPLIT_COUNTS)
    if set(split_counts) != {"train", "validation", "test"}:
        raise ValueError("split_counts must define train, validation, and test")
    if any(not isinstance(count, int) or count < 0 for count in split_counts.values()):
        raise ValueError("split counts must be nonnegative integers")
    if sum(split_counts.values()) != config.expected_count:
        raise ValueError("split counts must sum to expected_count")

    source_manifest_path = Path(config.source_manifest_path).resolve()
    output_dir = Path(config.output_dir).resolve()
    if (output_dir.exists() or output_dir.is_symlink()) and not config.overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")
    source_manifest = _read_json(source_manifest_path, kind="source manifest")
    if source_manifest.get("artifact_type") not in {
        "astroprism_stage1a_candidates",
        "astroprism_aion_feature_parity_reference",
    }:
        raise AcceptanceValidationError("source manifest has an unsupported artifact_type")
    outputs = source_manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("candidates"), dict):
        raise AcceptanceValidationError("source manifest is missing outputs.candidates")
    source_candidates_path = _resolve_output(
        source_manifest_path, outputs["candidates"], name="candidates"
    ).resolve()
    source_candidates_sha256 = sha256_file(source_candidates_path)
    expected_sha256 = outputs["candidates"].get("sha256")
    if expected_sha256 and source_candidates_sha256 != expected_sha256:
        raise AcceptanceValidationError("source candidate SHA-256 does not match its manifest")
    records = _read_jsonl(source_candidates_path)
    declared_count = outputs["candidates"].get("row_count")
    if declared_count is not None and declared_count != len(records):
        raise AcceptanceValidationError("source candidate row count does not match its manifest")
    coordinates = _validate_source_records(records, expected_count=config.expected_count)

    image_output = outputs.get("image_embeddings")
    if not isinstance(image_output, dict):
        raise AcceptanceValidationError("source manifest is missing outputs.image_embeddings")
    image_embeddings_path = _resolve_output(
        source_manifest_path, image_output, name="image_embeddings"
    ).resolve()
    if not image_embeddings_path.is_file():
        raise AcceptanceValidationError(
            f"source parity embeddings do not exist: {image_embeddings_path}"
        )

    groups = _source_groups(records, coordinates)
    assignments = _assign_splits(groups, split_counts)
    accepted = copy.deepcopy(records)
    parity_locator = Path(os.path.relpath(image_embeddings_path, start=output_dir)).as_posix()
    for group_id, members in groups:
        split = assignments[group_id]
        for index in members:
            record = accepted[index]
            record["acceptance_status"] = "accepted"
            record["caption"]["status"] = "released_reused"
            record["source_group_id"] = group_id
            record["split"] = split
            record["image_embedding"]["file"] = parity_locator

    observed_counts = Counter(record["split"] for record in accepted)
    if observed_counts != Counter(split_counts):
        raise AssertionError(f"internal split-count mismatch: {observed_counts}")
    for before, after in zip(records, accepted, strict=True):
        if before["caption"]["original"] != after["caption"]["original"]:
            raise AssertionError("caption.original changed during promotion")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        candidates_path = staging / "candidates.jsonl"
        _write_jsonl(candidates_path, accepted)
        group_sizes = Counter(len(members) for _group_id, members in groups)
        group_split_counts = Counter(assignments.values())
        manifest = {
            "schema_version": 1,
            "artifact_type": "astroprism_stage1a_candidates",
            "role": "accepted_training_candidates",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "acceptance": {
                "basis": ACCEPTANCE_BASIS,
                "accepted_count": len(accepted),
                "acceptance_status": "accepted",
                "caption_policy": "unchanged_original",
                "caption_status": "released_reused",
            },
            "lineage": {
                "source_manifest": {
                    "path": Path(
                        os.path.relpath(source_manifest_path, start=output_dir)
                    ).as_posix(),
                    "size_bytes": source_manifest_path.stat().st_size,
                    "sha256": sha256_file(source_manifest_path),
                },
                "source_candidates": {
                    "path": Path(
                        os.path.relpath(source_candidates_path, start=output_dir)
                    ).as_posix(),
                    "row_count": len(records),
                    "size_bytes": source_candidates_path.stat().st_size,
                    "sha256": source_candidates_sha256,
                },
            },
            "pins": copy.deepcopy(source_manifest.get("pins", {})),
            "roster": copy.deepcopy(source_manifest.get("roster", {})),
            "sources": copy.deepcopy(source_manifest.get("sources", {})),
            "parity_reference": {
                "path": parity_locator,
                "dtype": image_output.get("dtype"),
                "shape": image_output.get("shape"),
                "size_bytes": image_output.get("size_bytes"),
                "sha256": image_output.get("sha256"),
            },
            "splits": {
                "policy": SPLIT_POLICY,
                "rank_seed": SPLIT_RANK_SEED,
                "allocation_order": ["test", "validation", "train_remainder"],
                "grouping_policy": GROUP_POLICY,
                "grouping_coordinates": "coordinates_deg.paired_ra/paired_dec",
                "overlap_radius_arcsec": OVERLAP_RADIUS_ARCSEC,
                "row_counts": dict(sorted(observed_counts.items())),
                "group_counts": dict(sorted(group_split_counts.items())),
                "group_count": len(groups),
                "group_size_histogram": {
                    str(size): count for size, count in sorted(group_sizes.items())
                },
                "mapping_sha256": _mapping_sha256(accepted),
            },
            "outputs": {
                "candidates": {
                    "path": candidates_path.name,
                    "row_count": len(accepted),
                    "size_bytes": candidates_path.stat().st_size,
                    "sha256": sha256_file(candidates_path),
                }
            },
        }
        _write_json(staging / "manifest.json", manifest)
        _commit_staging(staging, output_dir, overwrite=config.overwrite)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)

    return {
        "status": "accepted",
        "output_dir": str(output_dir),
        "candidate_count": len(accepted),
        "group_count": len(groups),
        "split_counts": dict(sorted(observed_counts.items())),
        "mapping_sha256": _mapping_sha256(accepted),
        "candidates_sha256": manifest["outputs"]["candidates"]["sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1a_candidates/manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1a_accepted",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing accepted artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = accept_stage1a(
        AcceptanceConfig(
            source_manifest_path=args.source_manifest,
            output_dir=args.output,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

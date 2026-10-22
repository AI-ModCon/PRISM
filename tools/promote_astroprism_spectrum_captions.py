#!/usr/bin/env python3
"""Promote reviewed AstroPRISM spectrum captions into one portable artifact.

The source generation bundles are immutable review artifacts.  This tool
validates their manifests and sidecars, overlays the user's review decision,
and atomically writes a single accepted bundle in candidate-roster order.
Only the train split is eligible for gradient updates; validation and test are
accepted inputs for their assigned evaluation splits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CANDIDATES = 646
EXPECTED_SPLIT_COUNTS = {"train": 516, "validation": 65, "test": 65}
REVIEW_DECISION_SOURCE = "user_instruction"
REVIEW_INSTRUCTION = (
    "Generate the 130 held-out spectrum captions separately, add them as validation and test, "
    "and mark all spectrum captions as reviewed and ready for the alignment pipeline."
)
ROLE_BY_SPLIT = {
    "train": "alignment_training",
    "validation": "alignment_validation",
    "test": "alignment_test",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class PromotionValidationError(ValueError):
    """Raised when promotion would violate the accepted-caption contract."""


@dataclass(frozen=True)
class PromotionConfig:
    """Inputs for one atomic Stage 1B spectrum-caption promotion."""

    output_dir: Path
    accepted_candidates_path: Path = (
        REPO_ROOT / "_local/astroprism/stage1a_accepted/candidates.jsonl"
    )
    accepted_manifest_path: Path = REPO_ROOT / "_local/astroprism/stage1a_accepted/manifest.json"
    train_bundle: Path = REPO_ROOT / "_local/astroprism/stage1b_spectrum_caption_train_all"
    heldout_bundle: Path = REPO_ROOT / "_local/astroprism/stage1b_spectrum_caption_heldout_all"
    expected_count: int = EXPECTED_CANDIDATES
    expected_split_counts: dict[str, int] | None = None
    raw_manifest_reference: str = "../raw_spectra/manifest.json"
    review_instruction: str = REVIEW_INSTRUCTION
    link_mode: str = "auto"
    overwrite: bool = False


@dataclass(frozen=True)
class _Bundle:
    label: str
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    records: dict[int, dict[str, Any]]
    requests: dict[int, dict[str, Any]]
    audits: dict[int, dict[str, Any]]
    plots: dict[int, Path]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PromotionValidationError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionValidationError(f"{label} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PromotionValidationError(f"{label} does not exist: {path}")
    result: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise PromotionValidationError(
                        f"{label} contains a blank line at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PromotionValidationError(
                        f"{label} line {line_number} must be a JSON object"
                    )
                result.append(value)
    except json.JSONDecodeError as exc:
        raise PromotionValidationError(
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


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromotionValidationError(f"{label} must be a non-empty string")
    return value


def _safe_member(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise PromotionValidationError(f"unsafe {label} path: {relative!r}")
    resolved_root = root.resolve()
    path = (resolved_root / Path(*pure.parts)).resolve()
    if not path.is_relative_to(resolved_root):
        raise PromotionValidationError(f"{label} escapes its bundle: {relative!r}")
    if not path.is_file():
        raise PromotionValidationError(f"{label} does not exist: {path}")
    return path


def _validate_output_entry(
    root: Path, manifest: dict[str, Any], relative: str, *, label: str
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise PromotionValidationError(f"{label} manifest has no outputs object")
    entry = outputs.get(relative)
    if not isinstance(entry, dict):
        raise PromotionValidationError(f"{label} manifest does not declare {relative}")
    path = _safe_member(root, relative, label=f"{label} output")
    if entry.get("size_bytes") != path.stat().st_size:
        raise PromotionValidationError(f"{label} size mismatch for {relative}")
    digest = sha256_file(path)
    if entry.get("sha256") != digest:
        raise PromotionValidationError(f"{label} SHA-256 mismatch for {relative}")
    return path


def _validate_all_manifest_outputs(
    root: Path, manifest: dict[str, Any], *, label: str
) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise PromotionValidationError(f"{label} manifest has no outputs")
    for relative in outputs:
        if not isinstance(relative, str):
            raise PromotionValidationError(f"{label} manifest output names must be strings")
        _validate_output_entry(root, manifest, relative, label=label)


def _index_rows(
    rows: list[dict[str, Any]], expected_indices: set[int], *, label: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            index = int(row["candidate_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PromotionValidationError(f"{label} has an invalid candidate_index") from exc
        if index in indexed:
            raise PromotionValidationError(f"{label} repeats candidate {index}")
        if index not in expected_indices:
            raise PromotionValidationError(f"{label} has unexpected candidate {index}")
        indexed[index] = row
    if set(indexed) != expected_indices:
        missing = sorted(expected_indices - set(indexed))
        raise PromotionValidationError(f"{label} is missing candidates {missing[:10]}")
    return indexed


def _validate_accepted_candidates(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    candidates_path: Path,
    *,
    expected_count: int,
    expected_split_counts: dict[str, int],
) -> None:
    if len(rows) != expected_count:
        raise PromotionValidationError(
            f"accepted candidate roster has {len(rows)} rows, expected {expected_count}"
        )
    outputs = manifest.get("outputs")
    candidate_entry = outputs.get("candidates") if isinstance(outputs, dict) else None
    if not isinstance(candidate_entry, dict):
        raise PromotionValidationError("accepted manifest has no outputs.candidates")
    if candidate_entry.get("row_count") != len(rows):
        raise PromotionValidationError("accepted candidate row count does not match its manifest")
    if candidate_entry.get("size_bytes") != candidates_path.stat().st_size:
        raise PromotionValidationError("accepted candidate size does not match its manifest")
    if candidate_entry.get("sha256") != sha256_file(candidates_path):
        raise PromotionValidationError("accepted candidate SHA-256 does not match its manifest")
    declared_splits = manifest.get("splits", {}).get("row_counts")
    if declared_splits != expected_split_counts:
        raise PromotionValidationError("accepted manifest split counts do not match the contract")

    observed: Counter[str] = Counter()
    paired_ids: set[str] = set()
    caption_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    for expected_index, row in enumerate(rows):
        if row.get("candidate_index") != expected_index:
            raise PromotionValidationError(
                f"accepted row {expected_index} is not in candidate roster order"
            )
        if row.get("acceptance_status") != "accepted":
            raise PromotionValidationError(f"accepted candidate {expected_index} is not accepted")
        paired_id = _require_text(
            row.get("paired_targetid"), label=f"candidate {expected_index} paired_targetid"
        )
        caption_id = _require_text(
            row.get("caption_object_id"), label=f"candidate {expected_index} caption_object_id"
        )
        if paired_id in paired_ids or caption_id in caption_ids:
            raise PromotionValidationError(
                f"accepted candidate {expected_index} repeats an object identity"
            )
        paired_ids.add(paired_id)
        caption_ids.add(caption_id)
        split = row.get("split")
        if split not in expected_split_counts:
            raise PromotionValidationError(
                f"accepted candidate {expected_index} has unsupported split {split!r}"
            )
        observed[split] += 1
        group = _require_text(
            row.get("source_group_id"), label=f"candidate {expected_index} source_group_id"
        )
        group_splits[group].add(split)
    if dict(observed) != expected_split_counts:
        raise PromotionValidationError(
            f"accepted split counts are {dict(observed)}, expected {expected_split_counts}"
        )
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if leaking:
        raise PromotionValidationError(f"source groups cross splits: {leaking[:5]}")


def _validate_bundle_record(
    record: dict[str, Any],
    request: dict[str, Any],
    audit: dict[str, Any],
    accepted: dict[str, Any],
    *,
    label: str,
) -> None:
    index = int(accepted["candidate_index"])
    split = accepted["split"]
    for kind, item in (("caption", record), ("request", request), ("audit", audit)):
        if item.get("split") != split:
            raise PromotionValidationError(f"{label} {kind} split mismatch for candidate {index}")
        if item.get("paired_targetid") != accepted["paired_targetid"]:
            raise PromotionValidationError(
                f"{label} {kind} paired_targetid mismatch for candidate {index}"
            )
    for field in ("caption_object_id", "source_group_id"):
        for kind, item in (("caption", record), ("audit", audit)):
            if item.get(field) != accepted[field]:
                raise PromotionValidationError(
                    f"{label} {kind} {field} mismatch for candidate {index}"
                )

    caption = record.get("spectrum_caption")
    if not isinstance(caption, dict) or caption.get("status") != "pending_review":
        raise PromotionValidationError(
            f"{label} spectrum caption {index} must still be pending_review"
        )
    _require_text(caption.get("original"), label=f"candidate {index} spectrum caption")
    policy = record.get("training_policy")
    if not isinstance(policy, dict) or policy.get("eligible_for_training") is not False:
        raise PromotionValidationError(
            f"{label} candidate {index} must be training-ineligible before promotion"
        )
    expected_after_review = split == "train"
    if policy.get("eligible_after_review") is not expected_after_review:
        raise PromotionValidationError(
            f"{label} candidate {index} has an invalid pre-promotion split policy"
        )

    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise PromotionValidationError(f"{label} candidate {index} has no generation object")
    for digest_field in ("model_input_sha256", "compact_payload_sha256"):
        values = (
            generation.get(digest_field),
            request.get(digest_field),
            audit.get(digest_field),
        )
        if not isinstance(values[0], str) or len(set(values)) != 1:
            raise PromotionValidationError(
                f"{label} {digest_field} provenance mismatch for candidate {index}"
            )
    if record.get("teacher_reference") != audit.get("teacher_reference"):
        raise PromotionValidationError(
            f"{label} teacher-reference provenance mismatch for candidate {index}"
        )
    claims = record.get("audit_sidecar", {}).get("claims")
    if not isinstance(claims, list):
        raise PromotionValidationError(f"{label} candidate {index} has no audit claims")


def _load_bundle(
    root: Path,
    accepted_rows: list[dict[str, Any]],
    *,
    label: str,
    expected_indices: set[int],
    expected_split_counts: dict[str, int],
) -> _Bundle:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, label=f"{label} manifest")
    if manifest.get("artifact_type") != "astroprism_spectrum_caption_pilot":
        raise PromotionValidationError(f"{label} has unsupported artifact_type")
    if manifest.get("role") != "review_only_not_training_eligible":
        raise PromotionValidationError(f"{label} is not an immutable review bundle")
    if manifest.get("row_count") != len(expected_indices):
        raise PromotionValidationError(f"{label} manifest row count mismatch")
    if manifest.get("split_counts") != expected_split_counts:
        raise PromotionValidationError(f"{label} manifest split counts mismatch")

    _validate_all_manifest_outputs(root, manifest, label=label)
    caption_path = _validate_output_entry(root, manifest, "captions.jsonl", label=label)
    request_path = _validate_output_entry(
        root, manifest, "generation_requests.jsonl", label=label
    )
    audit_path = _validate_output_entry(root, manifest, "request_audit.jsonl", label=label)
    _validate_output_entry(root, manifest, "review.md", label=label)

    caption_rows = _read_jsonl(caption_path, label=f"{label} captions")
    request_rows = _read_jsonl(request_path, label=f"{label} generation requests")
    audit_rows = _read_jsonl(audit_path, label=f"{label} request audits")
    records = _index_rows(caption_rows, expected_indices, label=f"{label} captions")
    requests = _index_rows(request_rows, expected_indices, label=f"{label} generation requests")
    audits = _index_rows(audit_rows, expected_indices, label=f"{label} request audits")

    ordered_indices = [int(row["candidate_index"]) for row in caption_rows]
    if manifest.get("candidate_indices") != ordered_indices:
        raise PromotionValidationError(f"{label} candidate_indices do not match captions")
    expected_heldout = sorted(
        index for index in expected_indices if accepted_rows[index]["split"] != "train"
    )
    if sorted(manifest.get("held_out_candidate_indices", [])) != expected_heldout:
        raise PromotionValidationError(f"{label} held-out candidate list mismatch")

    plots: dict[int, Path] = {}
    for index in sorted(expected_indices):
        record = records[index]
        request = requests[index]
        audit = audits[index]
        _validate_bundle_record(
            record, request, audit, accepted_rows[index], label=label
        )
        relative = f"plots/candidate_{index:04d}.png"
        plot = _validate_output_entry(root, manifest, relative, label=label)
        with plot.open("rb") as handle:
            if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                raise PromotionValidationError(f"{label} candidate {index} plot is not PNG")
        plots[index] = plot

    return _Bundle(
        label=label,
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        records=records,
        requests=requests,
        audits=audits,
        plots=plots,
    )


def _is_absolute_local_path(value: str) -> bool:
    return value.startswith("/") or WINDOWS_ABSOLUTE_PATH.match(value) is not None


def _portable_copy(
    value: Any, *, key: str | None = None, parent: dict[str, Any] | None = None
) -> Any:
    """Deep-copy JSON data while replacing machine-local absolute paths."""

    if isinstance(value, dict):
        return {
            child_key: _portable_copy(child, key=child_key, parent=value)
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_portable_copy(item) for item in value]
    if isinstance(value, str) and _is_absolute_local_path(value):
        if key == "raw_manifest":
            return "../raw_spectra/manifest.json"
        if parent is not None and isinstance(parent.get("shard"), str):
            return parent["shard"]
        return Path(value).name
    return copy.deepcopy(value)


def _promoted_record(
    source: dict[str, Any], *, split: str, source_bundle: str, review_instruction: str
) -> dict[str, Any]:
    record = _portable_copy(source)
    record["spectrum_caption"]["status"] = "accepted"
    record["review"] = {
        "status": "accepted",
        "decision_source": REVIEW_DECISION_SOURCE,
        "instruction": review_instruction,
        "source_bundle": source_bundle,
        "source_caption_status": "pending_review",
    }
    is_train = split == "train"
    record["training_policy"] = {
        "dataset_ready": True,
        "gradient_training_allowed": is_train,
        "eligible_for_training": is_train,
        "eligible_for_evaluation": not is_train,
        "role": ROLE_BY_SPLIT[split],
        "restrictions": (
            []
            if is_train
            else [
                "Use only in the assigned evaluation split.",
                "Do not use for gradient updates, SFT, reward training, or prompt tuning.",
            ]
        ),
    }
    return record


def _copy_or_link(source: Path, destination: Path, *, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if mode not in {"auto", "hardlink"}:
        raise ValueError("link_mode must be 'auto', 'hardlink', or 'copy'")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        if mode == "hardlink":
            raise
        shutil.copy2(source, destination)
        return "copy"


def _portable_lineage(bundle: _Bundle) -> dict[str, Any]:
    outputs = bundle.manifest["outputs"]
    return {
        "artifact_type": bundle.manifest["artifact_type"],
        "role": bundle.manifest["role"],
        "row_count": bundle.manifest["row_count"],
        "split_counts": bundle.manifest["split_counts"],
        "manifest_sha256": bundle.manifest_sha256,
        "captions_sha256": outputs["captions.jsonl"]["sha256"],
        "generation_requests_sha256": outputs["generation_requests.jsonl"]["sha256"],
        "request_audit_sha256": outputs["request_audit.jsonl"]["sha256"],
        "implementation": _portable_copy(bundle.manifest.get("implementation", {})),
        "paper": _portable_copy(bundle.manifest.get("sources", {}).get("paper", {})),
    }


def _mapping_sha256(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        f"{row['candidate_index']}\t{row['source_group_id']}\t{row['split']}\n"
        for row in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _roster_provenance(accepted_manifest: dict[str, Any], *, row_count: int) -> dict[str, Any]:
    roster = accepted_manifest.get("roster")
    if not isinstance(roster, dict):
        roster = {}
    digest = roster.get("sha256")
    if not isinstance(digest, str) or not digest:
        digest = accepted_manifest.get("pins", {}).get("roster_sha256")
    if not isinstance(digest, str) or not digest:
        raise PromotionValidationError("accepted manifest does not pin the candidate roster")
    declared_count = roster.get("row_count", row_count)
    if declared_count != row_count:
        raise PromotionValidationError("accepted manifest roster row count mismatch")
    return {"row_count": row_count, "sha256": digest}


def _review_markdown(split_counts: dict[str, int], instruction: str) -> str:
    return f"""# AstroPRISM accepted spectrum captions

All {sum(split_counts.values())} spectrum captions were explicitly accepted by user instruction.
The source captions and their generation/audit provenance are preserved in this bundle.

Review decision: `accepted`

Decision source: `{REVIEW_DECISION_SOURCE}`

Instruction: {instruction}

| Split | Rows | Gradient updates | Pipeline role |
|---|---:|---|---|
| train | {split_counts['train']} | yes | `{ROLE_BY_SPLIT['train']}` |
| validation | {split_counts['validation']} | no | `{ROLE_BY_SPLIT['validation']}` |
| test | {split_counts['test']} | no | `{ROLE_BY_SPLIT['test']}` |

Validation and test captions are reviewed and dataset-ready, but remain evaluation-only.
"""


def _output_entries(root: Path, *, exclude: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        result[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _write_sha256sums(root: Path) -> None:
    entries = _output_entries(root, exclude={"SHA256SUMS"})
    (root / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {relative}\n" for relative, entry in entries.items()),
        encoding="utf-8",
    )


def _assert_portable_json(value: Any, *, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_portable_json(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_portable_json(child, location=f"{location}[{index}]")
    elif isinstance(value, str) and _is_absolute_local_path(value):
        raise PromotionValidationError(f"absolute local path remains at {location}: {value}")


def _validate_portability(root: Path) -> None:
    for path in sorted(root.rglob("*.json")):
        _assert_portable_json(
            json.loads(path.read_text(encoding="utf-8")), location=path.relative_to(root).as_posix()
        )
    for path in sorted(root.rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            _assert_portable_json(
                json.loads(line),
                location=f"{path.relative_to(root).as_posix()}:{line_number}",
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


def promote_spectrum_captions(config: PromotionConfig) -> dict[str, Any]:
    """Validate, promote, and atomically publish the accepted caption bundle."""

    split_counts = dict(config.expected_split_counts or EXPECTED_SPLIT_COUNTS)
    if set(split_counts) != set(ROLE_BY_SPLIT):
        raise ValueError("expected_split_counts must define train, validation, and test")
    if any(not isinstance(count, int) or count < 0 for count in split_counts.values()):
        raise ValueError("split counts must be nonnegative integers")
    if sum(split_counts.values()) != config.expected_count:
        raise ValueError("split counts must sum to expected_count")
    if config.link_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError("link_mode must be 'auto', 'hardlink', or 'copy'")
    review_instruction = _require_text(
        config.review_instruction, label="review_instruction"
    ).strip()
    raw_reference = _require_text(
        config.raw_manifest_reference, label="raw_manifest_reference"
    )
    if _is_absolute_local_path(raw_reference):
        raise ValueError("raw_manifest_reference must be portable, not absolute")

    output = Path(config.output_dir).resolve()
    if (output.exists() or output.is_symlink()) and not config.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    accepted_path = Path(config.accepted_candidates_path).resolve()
    accepted_manifest_path = Path(config.accepted_manifest_path).resolve()
    train_root = Path(config.train_bundle).resolve()
    heldout_root = Path(config.heldout_bundle).resolve()
    for source_root in (train_root, heldout_root):
        if output == source_root or output.is_relative_to(source_root):
            raise ValueError("output_dir must not be inside a source bundle")

    accepted_manifest = _read_json(accepted_manifest_path, label="accepted manifest")
    accepted_rows = _read_jsonl(accepted_path, label="accepted candidates")
    _validate_accepted_candidates(
        accepted_rows,
        accepted_manifest,
        accepted_path,
        expected_count=config.expected_count,
        expected_split_counts=split_counts,
    )
    train_indices = {
        int(row["candidate_index"]) for row in accepted_rows if row["split"] == "train"
    }
    heldout_indices = set(range(config.expected_count)) - train_indices
    train = _load_bundle(
        train_root,
        accepted_rows,
        label="train bundle",
        expected_indices=train_indices,
        expected_split_counts={"train": split_counts["train"]},
    )
    heldout = _load_bundle(
        heldout_root,
        accepted_rows,
        label="held-out bundle",
        expected_indices=heldout_indices,
        expected_split_counts={
            "test": split_counts["test"],
            "validation": split_counts["validation"],
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    committed = False
    try:
        promoted: list[dict[str, Any]] = []
        requests: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        link_actions: Counter[str] = Counter()
        for accepted in accepted_rows:
            index = int(accepted["candidate_index"])
            source_bundle = train if index in train.records else heldout
            record = _promoted_record(
                source_bundle.records[index],
                split=accepted["split"],
                source_bundle="train" if source_bundle is train else "heldout",
                review_instruction=review_instruction,
            )
            record["student_input"]["raw_manifest"] = raw_reference
            promoted.append(record)
            requests.append(_portable_copy(source_bundle.requests[index]))
            audits.append(_portable_copy(source_bundle.audits[index]))
            action = _copy_or_link(
                source_bundle.plots[index],
                staging / f"plots/candidate_{index:04d}.png",
                mode=config.link_mode,
            )
            link_actions[action] += 1

        if [row["candidate_index"] for row in promoted] != list(range(config.expected_count)):
            raise AssertionError("internal promotion ordering invariant failed")
        _write_jsonl(staging / "captions.jsonl", promoted)
        _write_jsonl(staging / "generation_requests.jsonl", requests)
        _write_jsonl(staging / "request_audit.jsonl", audits)
        (staging / "review.md").write_text(
            _review_markdown(split_counts, review_instruction), encoding="utf-8"
        )

        outputs = _output_entries(staging, exclude={"manifest.json", "SHA256SUMS"})
        manifest = {
            "schema_version": 1,
            "artifact_type": "astroprism_spectrum_captions_accepted",
            "role": "reviewed_alignment_dataset",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": len(promoted),
            "split_counts": split_counts,
            "candidate_indices": list(range(config.expected_count)),
            "roster": _roster_provenance(
                accepted_manifest, row_count=len(accepted_rows)
            ),
            "review": {
                "status": "accepted",
                "decision_source": REVIEW_DECISION_SOURCE,
                "instruction": review_instruction,
                "accepted_count": len(promoted),
            },
            "training_policy": {
                "dataset_ready": True,
                "gradient_training_splits": ["train"],
                "evaluation_splits": ["validation", "test"],
                "roles": ROLE_BY_SPLIT,
            },
            "splits": {
                "row_counts": split_counts,
                "mapping_sha256": _mapping_sha256(promoted),
                "source_groups_cross_splits": False,
            },
            "lineage": {
                "accepted_candidates": {
                    "row_count": len(accepted_rows),
                    "sha256": sha256_file(accepted_path),
                    "manifest_sha256": sha256_file(accepted_manifest_path),
                },
                "train_bundle": _portable_lineage(train),
                "heldout_bundle": _portable_lineage(heldout),
            },
            "storage": {
                "plot_actions": dict(sorted(link_actions.items())),
                "raw_manifest_reference": raw_reference,
            },
            "outputs": outputs,
        }
        _write_json(staging / "manifest.json", manifest)
        _validate_portability(staging)
        _write_sha256sums(staging)
        _commit_staging(staging, output, overwrite=config.overwrite)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)

    return {
        "status": "accepted",
        "output_dir": str(output),
        "candidate_count": len(promoted),
        "split_counts": split_counts,
        "gradient_training_count": split_counts["train"],
        "evaluation_count": split_counts["validation"] + split_counts["test"],
        "captions_sha256": manifest["outputs"]["captions.jsonl"]["sha256"],
        "manifest": str(output / "manifest.json"),
        "sha256sums": str(output / "SHA256SUMS"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1b_spectrum_caption_accepted",
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
        "--train-bundle",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1b_spectrum_caption_train_all",
    )
    parser.add_argument(
        "--heldout-bundle",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1b_spectrum_caption_heldout_all",
    )
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = promote_spectrum_captions(
        PromotionConfig(
            output_dir=args.output,
            accepted_candidates_path=args.accepted_candidates,
            accepted_manifest_path=args.accepted_manifest,
            train_bundle=args.train_bundle,
            heldout_bundle=args.heldout_bundle,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect caption-feature alignment evidence; never loads weights or caches.

Usage: python pack_feature_alignment.py ROOT RUN_NAME [--verify-checkpoint]
Final acceptance is evaluated only for a completed run with an independently
stream-hashed terminal checkpoint. Feature loss is not image generation accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# A direct script invocation resolves sibling helpers regardless of working dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_run import (
    MAX_ARCHIVE_INPUT_BYTES,
    MAX_FILE_BYTES,
    contained,
    json_bytes,
    load_indexes,
    read_jsonl,
    safe_name,
    sha256_file,
)

KIND = "real_checkpoint_connector_native_feature_alignment"
SUFFIXES = {".json", ".jsonl", ".py", ".sh", ".pbs", ".txt", ".log", ".md", ".png", ".pdf"}


def validate_cache_audits(report, records):
    audits = report.get("cache_audit", {})
    complete, valid = True, True
    details = {}
    for split in ("train", "validation"):
        selection = report["selection"][split]
        expected = {entry["id"] for entry in selection}
        rows = audits.get(split, [])
        identifiers = [row["id"] for row in rows]
        complete &= len(identifiers) == len(expected) and set(identifiers) == expected
        by_id = {row["id"]: row for row in records[split]}
        for row in rows:
            start, end = row["content_span"]
            valid &= (
                row["id"] in expected
                and row["split"] == split
                and row.get("target_pixels_read") is False
                and all(
                    row.get(key) is True
                    for key in (
                        "exact_formatted_input_match",
                        "exact_input_ids_match",
                        "exact_input_masks_match",
                        "actual_native_forward_inputs_verified",
                    )
                )
                and 0 <= start < end <= row["total_tokens"]
                and end - start == row["content_tokens"]
                and row["template_tokens"] > 0
                and row["content_tokens"] + row["template_tokens"] == row["total_tokens"]
                and row["prompt_sha256"]
                == hashlib.sha256(by_id[row["id"]]["prompt"].encode()).hexdigest()
                and all(
                    re.fullmatch(r"[0-9a-f]{64}", row.get(key, ""))
                    for key in (
                        "formatted_prompt_sha256",
                        "input_ids_sha256",
                        "input_mask_sha256",
                        "prism_hidden_sha256",
                        "native_features_sha256",
                        "teacher_normalized_sha256",
                    )
                )
            )
        details[split] = {
            "selected_ids": len(expected),
            "audited_ids": len(rows),
            "content_tokens": sum(row["content_tokens"] for row in rows),
            "template_tokens": sum(row["template_tokens"] for row in rows),
        }
    return bool(complete), bool(valid), details


def pack(root, name, *, source_root=None, output=None, verify_checkpoint=False):
    root = Path(root).resolve(strict=True)
    name = safe_name(name)
    run = contained(root, "runs/" + name)
    source_root = (
        Path(source_root).resolve(strict=True) if source_root else root / "prism-feature-alignment"
    )
    report_raw = (run / "report.json").read_bytes()
    report = json.loads(report_raw)
    if report.get("evidence_kind") not in (KIND, "fixture_only_connector_native_feature_alignment"):
        raise ValueError(
            "This collector accepts only the distinct caption-feature alignment schema"
        )
    _, records = load_indexes(report)
    members = {f"runs/{name}/report.json": report_raw}
    skipped = []

    def add(path, archive_name=None):
        if path.is_symlink() or not path.is_file() or path.suffix not in SUFFIXES:
            return
        if any(
            part.lower() in {"cache", "caches", "optimizer", "checkpoints", "weights"}
            for part in path.relative_to(root).parts
        ):
            return
        if path.stat().st_size > MAX_FILE_BYTES:
            skipped.append(str(path))
            return
        archive_name = archive_name or str(path.relative_to(root))
        members.setdefault(archive_name, path.read_bytes())

    for directory in (run, contained(root, "jobs/" + name)):
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                add(path)
    provenance = root / "provenance"
    if provenance.exists():
        for path in sorted(provenance.iterdir()):
            if "feature-alignment" in path.name or path.name in {
                "pack_feature_alignment.py",
                "pack_run.py",
            }:
                add(path)
    sources = dict(report.get("source_sha256", {}))
    sources["tools/align_prism_image_conditioning.py"] = report["runner_sha256"]
    source_checks = {}
    for relative, expected in sources.items():
        path = contained(source_root, relative)
        if not path.is_file():
            source_checks[relative] = False
            continue
        raw = path.read_bytes()
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("Unexpectedly large source file")
        source_checks[relative] = hashlib.sha256(raw).hexdigest() == expected
        members["prism-feature-alignment/" + relative] = raw
    steps, partial_lines = read_jsonl(members.get(f"runs/{name}/steps.jsonl", b""))
    settings = report["settings"]
    finished = report.get("status") == "completed"
    selected = {
        split: {row["id"] for row in report["selection"][split]}
        for split in ("train", "validation")
    }
    selections_valid = all(
        len(selected[split]) == len(report["selection"][split])
        and all(
            records[split][row["index"]]["id"] == row["id"] for row in report["selection"][split]
        )
        for split in selected
    )
    seen = [identifier for row in steps for identifier in row["ids"]]
    audit_complete, audits_valid, cache_counts = validate_cache_audits(report, records)
    checks = {
        "real_feature_alignment_evidence": report.get("evidence_kind") == KIND,
        "connector_only_scope": report.get("training_scope") == ["decoders.image.connector"],
        "content_alignment_objective": report.get("objective", {}).get("name")
        == "native_caption_rmsnorm_content_mse",
        "not_flow_or_image_quality_evidence": report.get("flow_training_performed") is False
        and report.get("quality_benchmark") is False,
        "target_pixels_not_read": report.get("target_pixels_read") is False,
        "feature_cache_not_saved": report.get("cache_saved") is False,
        "chat_format": report.get("prompt_format") == "chat",
        "native_normalization_frozen": report.get("caption_normalization", {}).get("frozen") is True
        and bool(report.get("caption_normalization", {}).get("state_sha256")),
        "negative_anchor_not_trained": report.get("negative_anchor_trained") is False,
        "logged_batch_sizes": all(len(row["ids"]) == settings["batch_size"] for row in steps),
        "selection_matches_dataset": selections_valid,
        "selected_training_count": len(selected["train"]) == settings["train_subset_size"],
        "heldout_count": len(selected["validation"]) == settings["validation_count"],
        "training_heldout_disjoint": not (selected["train"] & selected["validation"]),
        "updates_use_selected_training_ids": set(seen) <= selected["train"],
        "recorded_native_input_and_content_mask_audits": audits_valid,
        "feature_audit_covers_selection": audit_complete,
        "recorded_source_files_unchanged": bool(source_checks) and all(source_checks.values()),
        "finite_alignment_losses_and_gradients": all(
            all(
                math.isfinite(row[key])
                for key in (
                    "content_mse",
                    "content_cosine",
                    "template_mse",
                    "template_cosine",
                    "gradient_norm_before_clip",
                )
            )
            and row["gradient_norm_before_clip"] >= 0
            for row in steps
        ),
    }
    observations = {
        "report_status": report.get("status"),
        "phase": report.get("phase"),
        "reported_steps": report.get("completed_steps"),
        "logged_steps": len(steps),
        "observed_exposures": len(seen),
        "observed_unique_training_ids": len(set(seen)),
        "expected_terminal_exposures": settings["steps"] * settings["batch_size"],
        "cache_audit_counts": cache_counts,
        "partial_jsonl_lines_skipped": partial_lines,
        "checkpoint_sha256_independently_verified": False,
    }
    if finished:
        checks.update(
            requested_steps_completed=report["completed_steps"] == settings["steps"]
            and [row["step"] for row in steps] == list(range(1, settings["steps"] + 1)),
            requested_exposure_completed=len(seen) == settings["steps"] * settings["batch_size"],
            all_selected_training_ids_exposed=set(seen) == selected["train"],
            frozen_state_unchanged=report.get("frozen_state_unchanged") is True
            and bool(report.get("frozen_hashes_before"))
            and report.get("frozen_hashes_before") == report.get("frozen_hashes_after"),
            connector_state_changed=report.get("connector_state_changed") is True
            and bool(report.get("connector_hashes_before"))
            and set(report.get("connector_hashes_before", {}))
            == set(report.get("connector_hashes_after", {}))
            and report.get("connector_hashes_before") != report.get("connector_hashes_after"),
        )
        exposure = report.get("sample_exposure", {})
        counts = Counter(seen)
        checks["exact_per_id_exposures"] = (
            exposure.get("examples_seen") == len(seen)
            and exposure.get("unique_examples_seen") == len(counts)
            and exposure.get("per_id_counts")
            == {identifier: counts[identifier] for identifier in selected["train"]}
        )
        evaluations = report.get("evaluations", [])
        checks["terminal_feature_evaluation"] = (
            bool(evaluations)
            and evaluations[-1]["step"] == settings["steps"]
            and all(
                evaluations[-1]["splits"][split]["count"] == len(selected[split])
                and {row["id"] for row in evaluations[-1]["splits"][split]["examples"]}
                == selected[split]
                for split in selected
            )
        )
        checkpoints = report.get("checkpoints", [])
        checks["terminal_checkpoint_recorded"] = (
            bool(checkpoints)
            and checkpoints[-1]["step"] == settings["steps"]
            and checkpoints[-1].get("evidence_kind") == report["evidence_kind"]
        )
        if checks["terminal_checkpoint_recorded"]:
            checkpoint = checkpoints[-1]
            path = run / safe_name(Path(checkpoint["path"]).name)
            observations["terminal_checkpoint"] = {**checkpoint, "bytes": path.stat().st_size}
            if verify_checkpoint:
                checks["terminal_checkpoint_sha256"] = sha256_file(path) == checkpoint["sha256"]
                observations["checkpoint_sha256_independently_verified"] = checks[
                    "terminal_checkpoint_sha256"
                ]
    evaluate_final = finished and verify_checkpoint
    acceptance = {
        "schema_version": 1,
        "evidence_kind": report["evidence_kind"],
        "final_acceptance_evaluated": evaluate_final,
        "final_acceptance_passed": all(checks.values()) if evaluate_final else None,
        "checks": checks,
        "observations": observations,
        "source_root": str(source_root),
        "source_identity_checks": source_checks,
        "interpretation": "Feature-alignment execution and artifact checks only. Runtime token/content-mask audit records are validated, not recomputed with a model. No flow training or image quality claim.",
    }
    summary = {
        "run_name": name,
        "evidence_kind": report["evidence_kind"],
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "report_status": report.get("status"),
        "completed_steps": report.get("completed_steps"),
        "report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "observed_exposures": len(seen),
        "selected_training_ids": len(selected["train"]),
        "heldout_ids": len(selected["validation"]),
        "checkpoint_or_cache_bytes_included": False,
        "skipped_oversize_files": skipped,
        "final_acceptance_passed": acceptance["final_acceptance_passed"],
    }
    members[f"provenance/{name}-feature-alignment-acceptance.json"] = json_bytes(acceptance)
    members[f"runs/{name}/collection-summary.json"] = json_bytes(summary)
    total = sum(len(value) for value in members.values())
    if total > MAX_ARCHIVE_INPUT_BYTES:
        raise ValueError("Feature-alignment evidence exceeds archive byte limit")
    output = Path(output).resolve() if output else root / (name + "-evidence.tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for relative, payload in sorted(members.items()):
            contained(root, relative)
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(payload), 0o644
            archive.addfile(info, io.BytesIO(payload))
    temporary.replace(output)
    return {
        **summary,
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "uncompressed_bytes": total,
        "members": len(members),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-checkpoint", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            pack(
                args.root,
                args.run_name,
                source_root=args.source_root,
                output=args.output,
                verify_checkpoint=args.verify_checkpoint,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

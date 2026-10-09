#!/usr/bin/env python3
"""Snapshot modest DOCCI experiment evidence without importing models.

Usage: python pack_run.py RUN_ROOT RUN_NAME [--verify-checkpoint]

Works while a run is active. Interim snapshots never imply final acceptance.
The archive excludes weights, optimizer states, temporary files and tar shards.
Only gallery targets are read from the audited WebDataset byte locators.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SMALL_SUFFIXES = {
    ".json",
    ".jsonl",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".txt",
    ".log",
    ".pbs",
    ".sh",
    ".py",
    ".md",
}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_INPUT_BYTES = 384 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe artifact name: {value!r}")
    return value


def contained(root, relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or "\\" in str(value):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Path escapes root: {relative!r}")
    return path


def read_jsonl(payload):
    """A running writer may be between writing a line and its terminal newline."""
    lines = payload.splitlines(keepends=True)
    rows, skipped = [], 0
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if position == len(lines) - 1 and not line.endswith(b"\n"):
                skipped += 1
            else:
                raise
    return rows, skipped


def load_indexes(report):
    indexes, records = {}, {}
    for split in ("train", "validation"):
        path = Path(report["settings"][split + "_index"]).resolve(strict=True)
        raw = path.read_bytes()
        expected = report.get("index_sha256", {}).get(split) or report.get(split + "_index_sha256")
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{split} index SHA changed from recorded run")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len({row["id"] for row in rows}) != len(rows) or any(
            row["split"] != split for row in rows
        ):
            raise ValueError(f"Duplicate IDs or wrong splits in {split} index")
        indexes[split], records[split] = path, rows
    if {row["id"] for row in records["train"]} & {row["id"] for row in records["validation"]}:
        raise ValueError("Training and validation IDs overlap")
    return indexes, records


def sample_identity(sample, report, records):
    if "id" in sample:
        split = sample["split"]
        row = next(row for row in records[split] if row["id"] == sample["id"])
        return split, row
    match = re.fullmatch(r"(train|validation)-(\d+)", sample["case_id"])
    if match is None:
        raise ValueError("Unknown diagnostic gallery case ID")
    split, position = match.group(1), int(match.group(2))
    selection = report["selection"][split][position]
    row = records[split][selection["index"]]
    if row["id"] != selection["id"] or sample["prompt"] != row["prompt"]:
        raise ValueError("Diagnostic gallery case no longer matches its selected caption")
    return split, row


def indexed_target(index, row):
    identifier = safe_name(row["id"])
    if row["member"] != identifier + ".jpg":
        raise ValueError("Unexpected target member name")
    header, offset, size = (row[key] for key in ("header_offset", "data_offset", "size"))
    if any(type(value) is not int for value in (header, offset, size)) or (
        header < 0 or header % 512 or offset != header + 512 or not 0 < size <= MAX_FILE_BYTES
    ):
        raise ValueError("Invalid indexed target byte locator")
    shard = contained(index.parent, row["shard"])
    if offset + size > shard.stat().st_size:
        raise ValueError("Target locator exceeds shard size")
    with shard.open("rb") as stream:
        stream.seek(header)
        info = tarfile.TarInfo.frombuf(stream.read(512), "utf-8", "strict")
        if not info.isfile() or info.issparse() or info.name != row["member"] or info.size != size:
            raise ValueError("Indexed target header differs from recorded tar member")
        payload = stream.read(size)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
        raise ValueError("Extracted target checksum mismatch")
    return payload


def is_diagnostic(report):
    return "conditioning_diagnostic" in report.get("evidence_kind", "") or (
        "flow_controls" in report and report.get("training_performed") is False
    )


def acceptance(report, steps, records, samples, *, verify_checkpoint):
    finished = report.get("status") == "completed"
    diagnostic = is_diagnostic(report)
    checks = {
        "real_checkpoint_evidence": report.get("evidence_kind", "").startswith("real_checkpoint_"),
        "report_completed": finished,
        "train_validation_ids_disjoint": True,
    }
    observations = {
        "status": report.get("status"),
        "phase": report.get("phase"),
        "samples_collected": len(samples),
    }
    latent_groups = {}
    for sample in samples:
        latent_groups.setdefault(sample.get("id", sample.get("case_id")), set()).add(
            sample["initial_latent_sha256"]
        )
    if samples:
        checks.update(
            gallery_initial_noise_replayed=all(
                len(values) == 1 for values in latent_groups.values()
            ),
            generation_target_free=all(
                sample.get("sampling_target_free", sample.get("target_free")) is True
                for sample in samples
            ),
        )
    if diagnostic:
        controls = report.get("flow_controls", [])
        checks.update(
            no_training_performed=report.get("training_performed") is False,
            recorded_flow_inputs_present=all(
                value.get("actual_inputs", {}).get("noisy_latent_sha256")
                and value.get("actual_inputs", {}).get("timestep_sha256")
                for value in controls
            ),
            finite_flow_controls=all(
                all(
                    math.isfinite(route[key])
                    for key in ("matched", "wrong", "wrong_minus_matched", "prediction_change_mse")
                )
                for value in controls
                for route in value["routes"].values()
            ),
        )
        observations["flow_evaluations_collected"] = len(controls)
        if finished:
            expected = sum(len(rows) for rows in report["selection"].values()) * len(
                report["settings"]["flow_timesteps"]
            )
            checks["all_requested_flow_controls"] = len(controls) == expected
            checks["native_pretrained_state_unchanged"] = (
                report.get("native_pretrained_state_unchanged") is True
            )
    else:
        train_ids = {row["id"] for row in records["train"]}
        seen = [
            identifier
            for step in steps
            for batch in step["microbatches"]
            for identifier in batch["ids"]
        ]
        checks.update(
            training_ids_only=bool(seen) and set(seen) <= train_ids,
            finite_losses_and_group_gradients=all(
                math.isfinite(step["loss"])
                and all(
                    math.isfinite(detail["gradient_norm_before_clip"])
                    and detail["nonzero_gradient_tensors"] > 0
                    for detail in step["optimizer_diagnostics"]["groups"].values()
                )
                for step in steps
            ),
            complete_dense_scope=report.get("trainable_parameter_counts")
            == {"connector": 4200448, "diffusion": 3967161400},
        )
        observations.update(
            completed_steps=report.get("completed_steps"),
            step_log_rows=len(steps),
            examples_consumed=len(seen),
            unique_training_examples=len(set(seen)),
        )
        selected = report.get("train_selection")
        if selected:
            selected_ids = {row["id"] for row in selected}
            checks["training_within_selected_pool"] = set(seen) <= selected_ids
            observations["selected_example_count"] = len(selected_ids)
        if finished:
            requested = report["settings"]["steps"]
            # Resumed jobs can have only this invocation's updates in steps.jsonl.
            start = report.get("resume_step", 0)
            if steps:
                start = steps[0]["step"] - 1
            checks["all_requested_updates_recorded"] = report.get(
                "completed_steps"
            ) == requested and [row["step"] for row in steps] == list(
                range(start + 1, requested + 1)
            )
            checks["both_groups_changed"] = report.get("trainable_groups_changed") == {
                "connector": True,
                "diffusion": True,
            }
            if not report["settings"].get("resume"):
                progress = report.get("sample_exposure") or report.get("sampler_state") or {}
                if "examples_seen" in progress:
                    checks["sample_exposure_matches_steps"] = progress["examples_seen"] == len(seen)
                else:
                    observations["sample_exposure_check"] = "not recorded by this runner version"
                exposure = (report.get("sample_exposure") or {}).get("per_id_counts")
                if exposure is not None:
                    checks["per_id_exposure_matches_steps"] = all(
                        exposure.get(identifier, 0) == count
                        for identifier, count in Counter(seen).items()
                    ) and sum(exposure.values()) == len(seen)
            if not report["settings"].get("final_validation_count"):
                final = report["evaluations"][-1]
                checks["full_final_validation"] = bool(final.get("full_validation")) and final[
                    "splits"
                ]["validation"]["count"] == len(records["validation"])
            if report.get("checkpoints"):
                checkpoint = report["checkpoints"][-1]
                path = Path(checkpoint["path"])
                detail = {
                    **checkpoint,
                    "bytes": path.stat().st_size,
                    "independently_verified": False,
                }
                checks["terminal_checkpoint_step"] = checkpoint["step"] == requested
                if verify_checkpoint:
                    digest = sha256_file(path)
                    detail["independently_verified"] = digest == checkpoint["sha256"]
                    checks["terminal_checkpoint_sha256"] = detail["independently_verified"]
                observations["checkpoint"] = detail
            else:
                checks["terminal_checkpoint_present"] = False
    if finished:
        checks["frozen_state_unchanged"] = (
            report.get("frozen_state_unchanged") is True
            and bool(report.get("frozen_hashes_before"))
            and report.get("frozen_hashes_before") == report.get("frozen_hashes_after")
        )
    return {
        "schema_version": 1,
        "final_acceptance_evaluated": finished,
        "final_acceptance_passed": all(bool(value) for value in checks.values())
        if finished
        else None,
        "checks": {key: bool(value) for key, value in checks.items()},
        "observations": observations,
        "interpretation": "Artifact integrity and execution checks only; no generation quality or benchmark qualification.",
    }


def pack_run(root, name, output=None, *, verify_checkpoint=False, source_root=None):
    root = Path(root).resolve(strict=True)
    source_root = Path(source_root).resolve(strict=True) if source_root else root / "prism"
    name = safe_name(name)
    run = contained(root, "runs/" + name)
    report_raw = (run / "report.json").read_bytes()
    report = json.loads(report_raw)
    indexes, records = load_indexes(report)
    members = {"runs/" + name + "/report.json": report_raw}
    ignored = []

    def add(path, arcname=None, expected_sha=None):
        relative = arcname or str(path.relative_to(root))
        if relative in members:
            return
        if path.is_symlink() or not path.is_file() or path.suffix not in SMALL_SUFFIXES:
            return
        if path.stat().st_size > MAX_FILE_BYTES:
            ignored.append({"path": str(path), "reason": "exceeds_small_artifact_limit"})
            return
        raw = path.read_bytes()
        if expected_sha and hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError(f"Artifact checksum changed: {path}")
        members[relative] = raw

    for base in (run, contained(root, "jobs/" + name), root / "provenance"):
        if base.exists():
            for path in sorted(base.rglob("*")):
                if base == run and "targets" in path.relative_to(base).parts:
                    continue
                if path.is_file() and not any(
                    part.startswith(".") for part in path.relative_to(base).parts
                ):
                    add(path)
    steps, skipped_lines = read_jsonl(members.get("runs/" + name + "/steps.jsonl", b""))
    collected_samples, target_rows, case_targets = [], {}, {}
    for sample in report.get("samples", []):
        # Use the run-local basename, never an arbitrary path embedded in JSON.
        path = run / safe_name(Path(sample["path"]).name)
        add(path, expected_sha=sample["sha256"])
        if not path.is_file() or sha256_file(path) != sample["sha256"]:
            raise ValueError("Recorded generated image is missing or changed")
        split, row = sample_identity(sample, report, records)
        if sample.get("prompt") != row["prompt"]:
            raise ValueError("Gallery caption differs from its dataset row")
        target_rows[row["id"]] = row
        if sample.get("case_id"):
            case_targets[sample["case_id"]] = {"split": split, "id": row["id"]}
        collected_samples.append(sample)
    for identifier, row in target_rows.items():
        members[f"runs/{name}/targets/{safe_name(identifier)}.jpg"] = indexed_target(
            indexes[row["split"]], row
        )
    members[f"runs/{name}/gallery-targets.json"] = json_bytes(
        {
            "source": "Google DOCCI",
            "license": "CC-BY-4.0",
            "source_url": "https://google.github.io/docci/",
            "records": list(target_rows.values()),
            "case_targets": case_targets,
            "extraction": "Exact indexed tar member bytes; header/name/size/image SHA256 verified.",
        }
    )
    source_checks = {}
    source_hashes = dict(report.get("source_sha256", {}))
    runner = (
        "tools/diagnose_prism_image_conditioning.py"
        if is_diagnostic(report)
        else "tools/train_prism_image_diffusion.py"
    )
    if report.get("runner_sha256"):
        source_hashes[runner] = report["runner_sha256"]
    for relative, digest in source_hashes.items():
        path = contained(source_root, relative)
        if path.is_file():
            raw = path.read_bytes()
            matches = hashlib.sha256(raw).hexdigest() == digest
            source_checks[relative] = matches
            # Preserve actual bytes even if changed, explicitly recording mismatch.
            members["prism/" + relative] = raw
        else:
            source_checks[relative] = None
    audit = acceptance(
        report, steps, records, collected_samples, verify_checkpoint=verify_checkpoint
    )
    audit["source_identity_checks"] = source_checks
    audit["source_root"] = str(source_root)
    audit["checks"]["recorded_source_files_unchanged"] = bool(source_checks) and all(
        value is True for value in source_checks.values()
    )
    if audit["final_acceptance_evaluated"]:
        audit["final_acceptance_passed"] = all(audit["checks"].values())
    summary = {
        "run_name": name,
        "source_root": str(source_root),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "report_status": report.get("status"),
        "report_phase": report.get("phase"),
        "completed_steps": report.get("completed_steps"),
        "duration_seconds": report.get("duration_seconds"),
        "data_fingerprint": report.get("data_fingerprint"),
        "samples": len(collected_samples),
        "target_count": len(target_rows),
        "partial_jsonl_lines_skipped_for_checks": skipped_lines,
        "checkpoint_bytes_included": False,
        "ignored_oversize_files": ignored,
        "final_acceptance_passed": audit["final_acceptance_passed"],
    }
    members[f"provenance/{name}-acceptance-checks.json"] = json_bytes(audit)
    members[f"runs/{name}/collection-summary.json"] = json_bytes(summary)
    total = sum(len(value) for value in members.values())
    if total > MAX_ARCHIVE_INPUT_BYTES:
        raise ValueError(f"Evidence payload exceeds bounded limit: {total} bytes")
    output = Path(output).resolve() if output else root / (name + "-evidence.tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for relative, payload in sorted(members.items()):
            namepath = PurePosixPath(relative)
            if namepath.is_absolute() or ".." in namepath.parts:
                raise ValueError("Unsafe archive member")
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument(
        "--source-root", type=Path, help="Explicit executed source checkout; defaults to ROOT/prism"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            pack_run(
                args.root,
                args.run_name,
                args.output,
                verify_checkpoint=args.verify_checkpoint,
                source_root=args.source_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

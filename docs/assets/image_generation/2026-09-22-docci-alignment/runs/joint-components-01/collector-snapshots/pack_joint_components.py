#!/usr/bin/env python3
"""Collect bounded component-factorial evidence without importing or archiving weights.

A completed run is accepted only with independently streamed checkpoint digests,
verified executed source bytes, and the strict standalone artifact audit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_run import (
    MAX_ARCHIVE_INPUT_BYTES,
    MAX_FILE_BYTES,
    SMALL_SUFFIXES,
    contained,
    indexed_target,
    json_bytes,
    load_indexes,
    safe_name,
    sample_identity,
    sha256_file,
)


def pack(root, name, *, source_root, verify_checkpoints=False, output=None):
    root = Path(root).resolve(strict=True)
    source_root = Path(source_root).resolve(strict=True)
    name = safe_name(name)
    run = contained(root, "runs/" + name)
    report_raw = (run / "report.json").read_bytes()
    report = json.loads(report_raw)
    members = {f"runs/{name}/report.json": report_raw}
    prefix = f"runs/{name}/"

    def add(path, archive_name, expected=None):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.suffix not in SMALL_SUFFIXES:
            raise ValueError(f"Missing, symlinked, or unsupported artifact: {path}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Artifact exceeds bounded size: {path}")
        raw = path.read_bytes()
        if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Artifact digest changed: {path}")
        value = PurePosixPath(archive_name)
        if value.is_absolute() or ".." in value.parts or "\\" in archive_name:
            raise ValueError("Unsafe archive member")
        if archive_name in members and members[archive_name] != raw:
            raise ValueError("Conflicting archive member")
        members[archive_name] = raw
        return raw

    indexes, records = load_indexes(report)
    targets, case_targets = {}, {}
    for sample in report.get("samples", []):
        basename = safe_name(Path(sample["path"]).name)
        add(run / basename, prefix + basename, sample["sha256"])
        split, row = sample_identity(sample, report, records)
        if sample.get("prompt") != row["prompt"]:
            raise ValueError("Sample prompt differs from indexed caption")
        targets[row["id"]] = row
        case_targets[sample["case_id"]] = {"split": split, "id": row["id"]}
    for identifier, row in targets.items():
        members[prefix + "targets/" + safe_name(identifier) + ".jpg"] = indexed_target(
            indexes[row["split"]], row
        )
    members[prefix + "gallery-targets.json"] = json_bytes(
        {
            "source": "Google DOCCI",
            "license": "CC-BY-4.0",
            "source_url": "https://google.github.io/docci/",
            "records": list(targets.values()),
            "case_targets": case_targets,
            "extraction": "Exact indexed tar member bytes; header/name/size/image SHA256 verified.",
        }
    )
    for split, selections in report.get("selection", {}).items():
        for selected in selections:
            row = records[split][selected["index"]]
            if row["id"] != selected["id"]:
                raise ValueError("Selected ID differs from indexed record")

    sources = dict(report.get("source_sha256", {}))
    runner = "tools/diagnose_prism_joint_components.py"
    if report.get("runner_sha256"):
        if runner in sources and sources[runner] != report["runner_sha256"]:
            raise ValueError("Runner and source digest disagree")
        sources[runner] = report["runner_sha256"]
    source_checks = {}
    for relative, digest in sources.items():
        raw = add(contained(source_root, relative), "prism/" + relative, digest)
        source_checks[relative] = hashlib.sha256(raw).hexdigest() == digest

    checkpoints = {}
    for label, key in (
        ("region", "alignment_checkpoint"),
        ("joint", "joint_checkpoint"),
    ):
        lineage = report.get(key)
        if not lineage:
            continue
        checkpoint = Path(lineage["checkpoint"]).resolve(strict=True)
        source_report_path = checkpoint.parent / "report.json"
        source_raw = add(
            source_report_path,
            prefix + label + "-source-report.json",
            lineage["report_sha256"],
        )
        source_report = json.loads(source_raw)
        matching = [
            row
            for row in source_report.get("checkpoints", [])
            if row.get("sha256") == lineage["sha256"]
            and row.get("step") == lineage["step"]
        ]
        if (
            len(matching) != 1
            or source_report.get("status") != "completed"
            or source_report.get("completed_steps") != lineage["step"]
        ):
            raise ValueError(
                f"{label} source report does not bind a completed terminal checkpoint"
            )
        actual = sha256_file(checkpoint) if verify_checkpoints else None
        if actual is not None and actual != lineage["sha256"]:
            raise ValueError(f"{label} checkpoint digest differs")
        checkpoints[label] = {
            "checkpoint": str(checkpoint),
            "expected_sha256": lineage["sha256"],
            "actual_sha256": actual,
            "independently_verified": actual == lineage["sha256"],
            "report_sha256": hashlib.sha256(source_raw).hexdigest(),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "archived_report": label + "-source-report.json",
        }

    collectors = {}
    for basename in (
        "pack_joint_components.py",
        "render_joint_components.py",
        "pack_run.py",
    ):
        raw = add(
            Path(__file__).resolve().parent / basename,
            prefix + "collector-snapshots/" + basename,
        )
        collectors[basename] = hashlib.sha256(raw).hexdigest()
    jobs = contained(root, "jobs/" + name)
    if jobs.exists():
        for path in sorted(jobs.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix in SMALL_SUFFIXES
            ):
                add(path, str(path.relative_to(root)))
    for path in sorted(run.iterdir()):
        if path.is_file() and path.suffix in (".log", ".txt") and not path.is_symlink():
            add(path, prefix + path.name)
    evidence = {
        "schema_version": 1,
        "run_name": name,
        "source_root": str(source_root),
        "report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "source_identity_checks": source_checks,
        "recorded_source_files_unchanged": bool(source_checks)
        and all(source_checks.values()),
        "checkpoint_checks": checkpoints,
        "collector_sha256": collectors,
        "checkpoint_bytes_included": False,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    members[prefix + "collection-evidence.json"] = json_bytes(evidence)
    from render_joint_components import audit_payload

    if report.get("status") == "completed":
        # This reads only the already bounded archive payload and rejects missing proofs.
        audit = audit_payload(report_raw, members, prefix)
    else:
        audit = {
            "artifact_audit_passed": False,
            "final_acceptance_evaluated": False,
            "status": report.get("status"),
            "reason": "Run has not completed; partial evidence is not acceptance.",
        }
    members[prefix + "component-artifact-audit.json"] = json_bytes(audit)
    summary = {
        **evidence,
        "report_status": report.get("status"),
        "final_acceptance_passed": audit.get("artifact_audit_passed") is True,
        "samples": len(report.get("samples", [])),
        "target_count": len(targets),
    }
    members[prefix + "collection-summary.json"] = json_bytes(summary)
    total = sum(len(raw) for raw in members.values())
    if total > MAX_ARCHIVE_INPUT_BYTES:
        raise ValueError("Archive exceeds bounded evidence size")
    output = Path(output).resolve() if output else root / (name + "-evidence.tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    with tarfile.open(temp, "w:gz") as archive:
        for relative, raw in sorted(members.items()):
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(raw), 0o644
            archive.addfile(info, io.BytesIO(raw))
    temp.replace(output)
    return {
        **summary,
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "archive_bytes": output.stat().st_size,
        "members": len(members),
        "uncompressed_bytes": total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            pack(
                args.root,
                args.run_name,
                source_root=args.source_root,
                verify_checkpoints=args.verify_checkpoints,
                output=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

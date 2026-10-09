#!/usr/bin/env python3
"""Fetch HPC artifacts for CI runs (local or SSH-remote scheduler)."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch HPC job artifacts")
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument(
        "--dispatch-file",
        default="",
        help="Optional dispatch payload for remote fetch metadata.",
    )
    return parser


def _copy_local_artifacts(artifact_dir: str, destination: Path) -> None:
    source = Path(artifact_dir)
    if not source.exists():
        return
    subprocess.run(
        ["cp", "-R", str(source), str(destination / "job_artifacts")],
        check=False,
        capture_output=True,
        text=True,
    )


def _copy_remote_artifacts(dispatch: dict, artifact_dir: str, destination: Path) -> None:
    host = dispatch.get("scheduler_host", "")
    if not host:
        return

    user = dispatch.get("scheduler_user", "")
    target = f"{user}@{host}" if user else host
    remote_src = f"{target}:{artifact_dir}/"
    local_dst = str(destination / "job_artifacts")
    scp_cmd = ["scp", "-r", "-q"]
    ssh_key = dispatch.get("ssh_key", "")
    if ssh_key:
        scp_cmd.extend(["-i", ssh_key])
    ssh_port = dispatch.get("ssh_port")
    if isinstance(ssh_port, int) and ssh_port > 0:
        scp_cmd.extend(["-P", str(ssh_port)])
    scp_cmd.extend([remote_src, local_dst])
    subprocess.run(scp_cmd, check=False, capture_output=True, text=True)


def main() -> int:
    args = build_parser().parse_args()

    result_path = Path(args.result_file)
    result = json.loads(result_path.read_text())
    dispatch = json.loads(Path(args.dispatch_file).read_text()) if args.dispatch_file else {}

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = result.get("artifact_dir", "")
    if artifact_dir:
        if dispatch.get("scheduler_host", ""):
            _copy_remote_artifacts(dispatch, artifact_dir, artifacts_dir)
        else:
            _copy_local_artifacts(artifact_dir, artifacts_dir)

    summary = (
        f"Platform: {result['platform']}\n"
        f"Job ID: {result['job_id']}\n"
        f"Status: {result['status']}\n"
        f"Backend: {result['backend']}\n"
        f"Duration: {result['duration_sec']}s\n"
        f"Artifact Dir: {artifact_dir or 'n/a'}\n"
    )
    (artifacts_dir / "summary.log").write_text(summary)
    (artifacts_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"Artifacts generated in {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

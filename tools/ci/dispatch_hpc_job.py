#!/usr/bin/env python3
"""Dispatch Aurora/Perlmutter CI jobs in dry-run or real scheduler mode."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _runtime_to_hms(max_runtime_min: int) -> str:
    hours = max_runtime_min // 60
    minutes = max_runtime_min % 60
    return f"{hours:02d}:{minutes:02d}:00"


def _parse_job_id(platform: str, stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".strip()
    if platform == "perlmutter":
        # sbatch --parsable returns numeric job id (sometimes with suffix).
        first = text.splitlines()[0].strip() if text else ""
        if re.fullmatch(r"\d+(?:[._][\w-]+)?", first):
            return first
        match = re.search(r"Submitted batch job (\d+)", text)
        if match:
            return match.group(1)
    else:
        # qsub usually returns "<jobid>[.<server>]".
        first = text.splitlines()[0].strip() if text else ""
        if re.fullmatch(r"[A-Za-z0-9._-]+", first):
            return first
    raise RuntimeError(f"Unable to parse job id for platform={platform}. Output:\n{text}")


def _ssh_prefix(args: argparse.Namespace) -> list[str]:
    if not args.scheduler_host:
        return []

    target = (
        f"{args.scheduler_user}@{args.scheduler_host}"
        if args.scheduler_user
        else args.scheduler_host
    )
    prefix = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
    if args.ssh_key:
        prefix.extend(["-i", args.ssh_key])
    if args.ssh_port:
        prefix.extend(["-p", str(args.ssh_port)])
    prefix.append(target)
    return prefix


def _build_job_script(args: argparse.Namespace) -> str:
    escaped_command = shlex.quote(args.command)
    shell_name = "PBS_JOBID" if args.platform == "aurora" else "SLURM_JOB_ID"
    return f"""#!/bin/bash
set -euo pipefail

ART_ROOT="{args.artifacts_root}"
mkdir -p "$ART_ROOT"
JOB_RUNTIME_ID="${{{shell_name}:-unknown}}"
JOB_DIR="$ART_ROOT/{args.platform}-$JOB_RUNTIME_ID"
mkdir -p "$JOB_DIR"
LOG_FILE="$JOB_DIR/job.log"
RESULT_FILE="$JOB_DIR/job_result.json"
START_TS="$(date +%s)"

STATUS="failed"
EXIT_CODE=1
cleanup() {{
  EXIT_CODE=$?
  END_TS="$(date +%s)"
  DURATION="$((END_TS-START_TS))"
  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="success"
  else
    STATUS="failed"
  fi
  cat > "$RESULT_FILE" <<JSON
{{
  "platform": "{args.platform}",
  "job_id": "$JOB_RUNTIME_ID",
  "status": "$STATUS",
  "exit_code": $EXIT_CODE,
  "duration_sec": $DURATION
}}
JSON
  exit "$EXIT_CODE"
}}
trap cleanup EXIT

echo "[CI-HPC] Platform: {args.platform}" | tee -a "$LOG_FILE"
echo "[CI-HPC] Label: {args.job_label}" | tee -a "$LOG_FILE"
echo "[CI-HPC] Command: {args.command}" | tee -a "$LOG_FILE"

bash -lc {escaped_command} >> "$LOG_FILE" 2>&1
"""


def _submit_command(args: argparse.Namespace) -> list[str]:
    hms = _runtime_to_hms(args.max_runtime_min)
    if args.platform == "aurora":
        cmd = ["qsub", "-q", args.queue, "-N", args.job_label, "-l", f"walltime={hms}"]
        if args.account:
            cmd.extend(["-A", args.account])
        cmd.append("-")
        return cmd

    # perlmutter / slurm
    cmd = [
        "sbatch",
        "--parsable",
        "-q",
        args.queue,
        "-J",
        args.job_label,
        "-t",
        hms,
    ]
    if args.account:
        cmd.extend(["-A", args.account])
    cmd.append("-")
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch Aurora/Perlmutter CI jobs")
    parser.add_argument("--platform", choices=["aurora", "perlmutter"], required=True)
    parser.add_argument("--design-id", default="CI-SMOKE")
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--account", default="")
    parser.add_argument("--command", default="hostname && date && echo ci-smoke")
    parser.add_argument("--job-label", default="ci-smoke")
    parser.add_argument("--max-runtime-min", type=int, default=30)
    parser.add_argument("--ci-mode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scheduler-host", default="")
    parser.add_argument("--scheduler-user", default="")
    parser.add_argument("--ssh-key", default="")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument(
        "--artifacts-root",
        default="$HOME/prism_ci_artifacts",
        help="Artifact root path visible from scheduler environment.",
    )
    parser.add_argument("--emit-json", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    now = datetime.now(timezone.utc)
    dry_job_id = f"dryrun-{args.platform}-{now.strftime('%Y%m%d%H%M%S')}"

    payload = {
        "platform": args.platform,
        "job_id": dry_job_id,
        "status": "submitted",
        "submitted_at": now.isoformat(),
        "design_id": args.design_id,
        "queue": args.queue,
        "account": args.account,
        "command": args.command,
        "job_label": args.job_label,
        "max_runtime_min": args.max_runtime_min,
        "ci_mode": args.ci_mode,
        "dry_run": args.dry_run,
        "submission_mode": "dry_run" if args.dry_run else "real",
        "scheduler_system": "pbs" if args.platform == "aurora" else "slurm",
        "scheduler_host": args.scheduler_host,
        "scheduler_user": args.scheduler_user,
        "ssh_key": args.ssh_key,
        "ssh_port": args.ssh_port,
        "artifacts_root": args.artifacts_root,
    }

    if not args.dry_run:
        script = _build_job_script(args)
        submit_cmd = _submit_command(args)
        full_cmd = _ssh_prefix(args) + submit_cmd
        proc = subprocess.run(
            full_cmd,
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(
                "Scheduler submission failed.\n"
                f"Command: {' '.join(full_cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        job_id = _parse_job_id(args.platform, proc.stdout, proc.stderr)
        payload["job_id"] = job_id
        payload["scheduler_submit_command"] = full_cmd
        payload["scheduler_submit_stdout"] = proc.stdout.strip()
        payload["scheduler_submit_stderr"] = proc.stderr.strip()
        payload["artifact_dir"] = f"{args.artifacts_root}/{args.platform}-{job_id}"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.emit_json:
        print(json.dumps(payload))
    else:
        mode = "dry-run" if args.dry_run else "real"
        print(f"Dispatch ({mode}) contract written: {output}")
        if not args.dry_run:
            print(f"Submitted job id: {payload['job_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

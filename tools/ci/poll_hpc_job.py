#!/usr/bin/env python3
"""Poll Aurora/Perlmutter job status in dry-run or real scheduler mode."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RUNNING_STATES = {
    "Q",
    "H",
    "R",
    "B",
    "E",
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
}
SUCCESS_STATES = {"F", "C", "COMPLETED"}
FAILED_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poll Aurora/Perlmutter CI jobs")
    parser.add_argument("--dispatch-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--emit-json", action="store_true")
    parser.add_argument("--poll-interval-sec", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    return parser


def _ssh_prefix(dispatch: dict) -> list[str]:
    host = dispatch.get("scheduler_host", "")
    if not host:
        return []

    user = dispatch.get("scheduler_user", "")
    target = f"{user}@{host}" if user else host
    prefix = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
    ssh_key = dispatch.get("ssh_key", "")
    if ssh_key:
        prefix.extend(["-i", ssh_key])
    ssh_port = dispatch.get("ssh_port")
    if isinstance(ssh_port, int) and ssh_port > 0:
        prefix.extend(["-p", str(ssh_port)])
    prefix.append(target)
    return prefix


def _run_scheduler_query(dispatch: dict, command: list[str]) -> subprocess.CompletedProcess:
    full = _ssh_prefix(dispatch) + command
    return subprocess.run(full, capture_output=True, text=True, check=False)


def _poll_aurora(dispatch: dict) -> tuple[str | None, int | None, str]:
    job_id = dispatch["job_id"]
    proc = _run_scheduler_query(dispatch, ["qstat", "-xf", job_id])
    text = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        return None, None, text.strip()

    state_match = re.search(r"job_state\s*=\s*([A-Z])", text)
    exit_match = re.search(r"exit_status\s*=\s*(-?\d+)", text)
    state = state_match.group(1) if state_match else None
    exit_code = int(exit_match.group(1)) if exit_match else None
    return state, exit_code, text.strip()


def _poll_perlmutter(dispatch: dict) -> tuple[str | None, int | None, str]:
    job_id = dispatch["job_id"]
    proc = _run_scheduler_query(
        dispatch,
        ["sacct", "-j", job_id, "--format=State,ExitCode", "--parsable2", "--noheader"],
    )
    if proc.returncode == 0 and proc.stdout.strip():
        for line in proc.stdout.splitlines():
            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 2:
                continue
            state = parts[0].split()[0]
            exit_raw = parts[1]
            code = None
            if ":" in exit_raw:
                raw = exit_raw.split(":", 1)[0]
                if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
                    code = int(raw)
            return state, code, proc.stdout.strip()

    fallback = _run_scheduler_query(dispatch, ["squeue", "-j", job_id, "--noheader", "-o", "%T"])
    if fallback.returncode == 0 and fallback.stdout.strip():
        state = fallback.stdout.strip().splitlines()[0].split()[0]
        return state, None, fallback.stdout.strip()
    return None, None, f"{proc.stdout}\n{proc.stderr}\n{fallback.stdout}\n{fallback.stderr}".strip()


def _state_to_status(state: str | None, exit_code: int | None) -> str | None:
    if state is None:
        return None

    normalized = state.upper()
    if normalized in RUNNING_STATES:
        return None
    if normalized in SUCCESS_STATES:
        if exit_code in (None, 0):
            return "success"
        return "failed"
    if normalized in FAILED_STATES:
        return "failed"
    return None


def _read_remote_result(dispatch: dict) -> tuple[int, int, str]:
    artifact_dir = dispatch.get("artifact_dir", "")
    if not artifact_dir:
        return 0, 0, ""

    cat_cmd = ["cat", f"{artifact_dir}/job_result.json"]
    proc = _run_scheduler_query(dispatch, cat_cmd)
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0, 0, ""

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return 0, 0, ""

    duration = payload.get("duration_sec", 0)
    if isinstance(duration, int) and duration >= 0:
        derived_duration = duration
    else:
        derived_duration = 0

    if payload.get("status") == "success":
        return derived_duration, 1, ""
    return derived_duration, 0, f"exit_code={payload.get('exit_code', 'unknown')}"


def main() -> int:
    args = build_parser().parse_args()
    dispatch = json.loads(Path(args.dispatch_file).read_text())

    platform = dispatch["platform"]
    backend = "xccl" if platform == "aurora" else "nccl"
    now = datetime.now(timezone.utc)

    if dispatch.get("dry_run", False):
        payload = {
            "platform": platform,
            "job_id": dispatch["job_id"],
            "git_sha": "contract-mode",
            "status": "success",
            "duration_sec": 60,
            "backend": backend,
            "steps_completed": 2,
            "tests_passed": 3,
            "error_summary": "",
            "completed_at": now.isoformat(),
            "artifact_dir": dispatch.get("artifact_dir", ""),
        }
    else:
        start = time.monotonic()
        terminal_status = None
        last_state = None
        last_exit = None
        scheduler_trace = ""

        while time.monotonic() - start < args.timeout_sec:
            if platform == "aurora":
                state, exit_code, trace = _poll_aurora(dispatch)
            else:
                state, exit_code, trace = _poll_perlmutter(dispatch)
            scheduler_trace = trace or scheduler_trace
            last_state = state if state is not None else last_state
            last_exit = exit_code if exit_code is not None else last_exit

            mapped = _state_to_status(state, exit_code)
            if mapped is not None:
                terminal_status = mapped
                break
            time.sleep(max(1, args.poll_interval_sec))

        if terminal_status is None:
            terminal_status = "timeout"

        duration_sec, tests_passed, result_hint = _read_remote_result(dispatch)
        if duration_sec <= 0:
            duration_sec = int(time.monotonic() - start)

        error_summary = ""
        if terminal_status != "success":
            state_repr = last_state if last_state is not None else "unknown"
            exit_repr = last_exit if last_exit is not None else "unknown"
            error_summary = f"state={state_repr}, exit_code={exit_repr}"
            if result_hint:
                error_summary = f"{error_summary}, {result_hint}"
            elif scheduler_trace:
                error_summary = f"{error_summary}, scheduler_trace={scheduler_trace[:300]}"

        payload = {
            "platform": platform,
            "job_id": dispatch["job_id"],
            "git_sha": dispatch.get("git_sha", "unknown"),
            "status": terminal_status,
            "duration_sec": duration_sec,
            "backend": backend,
            "steps_completed": 1 if terminal_status in {"success", "failed"} else 0,
            "tests_passed": tests_passed,
            "error_summary": error_summary,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "artifact_dir": dispatch.get("artifact_dir", ""),
            "scheduler_state": last_state if last_state is not None else "",
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.emit_json:
        print(json.dumps(payload))
    else:
        print(f"Poll contract written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

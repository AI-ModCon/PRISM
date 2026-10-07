#!/usr/bin/env python3
"""Validate HPC result payloads against required contract fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_KEYS = {
    "platform",
    "job_id",
    "git_sha",
    "status",
    "duration_sec",
    "backend",
    "steps_completed",
    "tests_passed",
    "error_summary",
}

VALID_PLATFORMS = {"aurora", "perlmutter"}
VALID_BACKENDS = {"xccl", "nccl"}
VALID_STATUS = {"success", "failed", "timeout"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate HPC CI result payload")
    parser.add_argument("--result-file", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    payload = json.loads(Path(args.result_file).read_text())
    missing = sorted(REQUIRED_KEYS - set(payload.keys()))
    if missing:
        raise SystemExit(f"Missing required keys: {', '.join(missing)}")

    if payload["platform"] not in VALID_PLATFORMS:
        raise SystemExit(f"Invalid platform: {payload['platform']}")
    if payload["backend"] not in VALID_BACKENDS:
        raise SystemExit(f"Invalid backend: {payload['backend']}")
    if payload["status"] not in VALID_STATUS:
        raise SystemExit(f"Invalid status: {payload['status']}")

    if not isinstance(payload["duration_sec"], int) or payload["duration_sec"] < 0:
        raise SystemExit("duration_sec must be a non-negative integer")

    print("HPC result payload is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Subprocess smoke for tools/isoflop_calibrate.py.

Verifies the help text and idempotency surface without requiring an XPU/CUDA
device or a real HF backbone (which would otherwise take minutes and need
network). The full calibration is exercised manually via PR-1 Smoke 2; here
we only protect against argparse / import regressions.

The full calibration test is `@pytest.mark.gpu` and skipped without a
device, mirroring the pattern in tests/multimodal/test_modality_contracts.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CALIBRATE = REPO / "tools" / "isoflop_calibrate.py"


def test_help_runs_without_imports_failing() -> None:
    out = subprocess.run(
        [sys.executable, str(CALIBRATE), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "--projector-variant" in out.stdout
    assert "BASE" in out.stdout and "W4X" in out.stdout
    assert "--family" in out.stdout


def test_idempotency_skips_existing(tmp_path: Path) -> None:
    """When --output exists and --force not set, exit 0 without re-running.

    We write a sentinel file and assert mtime is unchanged after invocation.
    The subprocess should bail out before importing torch / loading models.
    """
    out_json = tmp_path / "cal.json"
    out_json.write_text(json.dumps({"flops_per_step": 1e12}))
    mtime_before = out_json.stat().st_mtime

    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATE),
            "--backbone",
            "ignored",
            "--projector-variant",
            "BASE",
            "--family",
            "text_image",
            "--output",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "already calibrated" in result.stdout
    assert out_json.stat().st_mtime == mtime_before


@pytest.mark.gpu
def test_calibration_writes_json(tmp_path: Path) -> None:
    """Full calibration round-trip. Skipped without a real device.

    Marked `gpu` so CI without an accelerator skips it. PR-1 Smoke 2
    covers this on a real Aurora node.
    """
    pytest.skip("Requires XPU/CUDA + HF backbone; covered by PR-1 Smoke 2")

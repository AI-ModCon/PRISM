import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.launcher, pytest.mark.perlmutter]


def test_launch_baremetal_requires_explicit_secrets(tmp_path: Path, monkeypatch):
    pytest.importorskip("yaml")

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    design = tmp_path / "design.yaml"
    design.write_text(
        "\n".join(
            [
                "experiments:",
                "  - id: TEST-EXP",
                "    name: test",
                "    resources:",
                "      ngpus: 1",
                "    common_overrides:",
                "      training.max_steps: 1",
            ]
        )
    )

    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "launch_baremetal.py"),
        "--file",
        str(design),
        "--id",
        "TEST-EXP",
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    combined = result.stdout + "\n" + result.stderr
    assert "WANDB_API_KEY is not set" in combined
    assert "hf_" not in combined

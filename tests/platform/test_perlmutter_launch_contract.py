import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.launcher, pytest.mark.perlmutter]


def test_perlmutter_launcher_dry_run(tmp_path: Path):
    pytest.importorskip("yaml")

    design = tmp_path / "design.yaml"
    design.write_text(
        "\n".join(
            [
                "experiments:",
                "  - id: TEST-PERLMUTTER",
                "    name: perlmutter test",
                "    resources:",
                "      ngpus: 2",
                "    common_overrides:",
                "      training.max_steps: 2",
            ]
        )
    )

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "launch_perlmutter.py"),
            "--file",
            str(design),
            "--id",
            "TEST_PERM_RUN",
            "--design",
            "TEST-PERLMUTTER",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"

    generated = None
    for line in result.stdout.splitlines():
        if line.startswith("Generated Run Script: "):
            generated = line.split(": ", 1)[1].strip()
            break

    assert generated is not None
    script_text = (tmp_path / generated).read_text()
    assert "#SBATCH" in script_text
    assert "srun" in script_text
    assert "${{" not in script_text

"""Focused safety tests for the generic Aurora launcher."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tools import launch_aurora

pytestmark = [pytest.mark.launcher, pytest.mark.aurora]


def _write_design(path: Path) -> Path:
    design = path / "design.yaml"
    backbone = path / "local-backbone"
    backbone.mkdir(exist_ok=True)
    design.write_text(
        "\n".join(
            [
                "experiments:",
                "  - id: TEST-LAUNCHER",
                '    name: "launcher safety test"',
                "    resources:",
                "      ngpus: 1",
                '      walltime: "00:10:00"',
                "    common_overrides:",
                "      training.max_steps: 1",
                f'      model.backbone_id: "{backbone}"',
            ]
        ),
        encoding="utf-8",
    )
    return design


def _launcher_argv(
    design: Path,
    packed_env: Path,
    *,
    batch: bool,
    dry_run: bool,
) -> list[str]:
    argv = [
        "launch_aurora.py",
        "--file",
        str(design),
        "--design",
        "TEST-LAUNCHER",
        "--id",
        "TEST-LAUNCHER",
        "--nodes",
        "1",
        "--packed-env",
        str(packed_env),
    ]
    if batch:
        argv.append("--batch")
    if dry_run:
        argv.append("--dry-run")
    return argv


def _generated_script(tmp_path: Path) -> str:
    scripts = list((tmp_path / "jobs").glob("run_aurora_TEST-LAUNCHER_*"))
    assert len(scripts) == 1
    return scripts[0].read_text(encoding="utf-8")


def test_packed_env_uses_documented_nested_venv_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tarball holds .venv-deepspeed/, so activate must look inside it."""
    design = _write_design(tmp_path)
    packed_env = tmp_path / "deepspeed_env.tar.gz"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        _launcher_argv(design, packed_env, batch=True, dry_run=True),
    )

    launch_aurora.main()

    script = _generated_script(tmp_path)
    nested_activate = "$LOCAL_ENV/.venv-deepspeed/bin/activate"
    assert nested_activate in script
    assert "source $LOCAL_ENV/bin/activate" not in script
    assert 'VIRTUAL_ENV=\\"$LOCAL_ENV/.venv-deepspeed\\"' in script
    assert '[ "$ENV_TARBALL" -nt "$MARKER_FILE" ]' in script
    assert 'tar -xzf "$ENV_TARBALL" -C "$LOCAL_ENV"' in script
    # The wait must test the same staleness condition as the unpack: with a
    # stale marker on disk, an existence-only wait lets other ranks activate
    # the environment while rank 0 is deleting it.
    wait_line = next(
        line for line in script.splitlines() if line.startswith("while [ ! -f")
    )
    assert '[ "$ENV_TARBALL" -nt "$MARKER_FILE" ]' in wait_line


def test_batch_mode_submits_generated_script_with_qsub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--batch must qsub the PBS script, not execute it on the login node."""
    design = _write_design(tmp_path)
    packed_env = tmp_path / "deepspeed_env.tar.gz"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="12345.aurora-pbs-0001\n",
            stderr="",
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        _launcher_argv(design, packed_env, batch=True, dry_run=False),
    )
    monkeypatch.setattr(launch_aurora.subprocess, "run", fake_run)

    launch_aurora.main()

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == "qsub"
    assert Path(command[1]).name.startswith("run_aurora_TEST-LAUNCHER_")
    assert kwargs == {"check": True, "capture_output": True, "text": True}

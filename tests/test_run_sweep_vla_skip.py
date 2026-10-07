"""Lock in `tools/run_sweep.py` bidirectional skip rules for VLA preset / designs.

The skip block in `tools/run_sweep.py` constrains VLA bidirectionally:

  - `vla` preset only runs against VLA designs (PRISM-AURORA-ZONE-A-VLA-CALVIN…)
  - VLA designs only run when the `vla` preset is selected (otherwise the
    non-VLA trainer would receive a CALVIN batch and crash)

Both directions matter — a stale test that only checks one would let a
regression that drops the other slip in. We invoke the CLI in `--print`
mode and assert the output's launch-command count.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SWEEP = REPO_ROOT / "tools" / "run_sweep.py"


def _run(preset: str, designs: str) -> tuple[int, str, str]:
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SWEEP),
            "--preset",
            preset,
            "--designs",
            designs,
            "--storage",
            "daos",
            "--print",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    return result.returncode, result.stdout, result.stderr


def _count_launch_cmds(stdout: str) -> int:
    """Count printed launcher invocations (one per non-skipped sweep cell)."""
    return sum(
        1
        for line in stdout.splitlines()
        if "launch_aurora" in line and "--design" in line
    )


def test_vla_preset_on_vla_design_emits_one_command():
    rc, out, err = _run("vla", "PRISM-AURORA-ZONE-A-VLA-CALVIN-SMOKE")
    assert rc == 0, f"non-zero exit: {err}"
    assert _count_launch_cmds(out) == 1


def test_vla_preset_on_non_vla_design_is_skipped():
    rc, out, err = _run("vla", "PRISM-MODALITY-SMOKE-1N")
    assert rc == 0, f"non-zero exit: {err}"
    assert _count_launch_cmds(out) == 0
    assert "vla preset is only valid against VLA designs" in err


def test_non_vla_preset_on_vla_design_is_skipped():
    rc, out, err = _run("all6", "PRISM-AURORA-ZONE-A-VLA-CALVIN-SMOKE")
    assert rc == 0, f"non-zero exit: {err}"
    assert _count_launch_cmds(out) == 0
    assert "VLA design requires the vla preset" in err


def test_text_only_on_modality_smoke_still_skipped():
    """text_only ↔ PRISM-MODALITY-SMOKE-1N: skipped (frozen backbone)."""
    rc, out, err = _run("text_only", "PRISM-MODALITY-SMOKE-1N")
    assert rc == 0, f"non-zero exit: {err}"
    assert _count_launch_cmds(out) == 0
    assert "frozen-backbone design has no trainable params" in err


def test_text_only_on_textonly_design_emits_one_command():
    """text_only ↔ PRISM-MODALITY-SMOKE-TEXTONLY-1N: routes through (backbone unfrozen)."""
    rc, out, err = _run("text_only", "PRISM-MODALITY-SMOKE-TEXTONLY-1N")
    assert rc == 0, f"non-zero exit: {err}"
    assert _count_launch_cmds(out) == 1


def test_non_textonly_preset_on_textonly_design_is_skipped():
    """text_image ↔ PRISM-MODALITY-SMOKE-TEXTONLY-1N: skipped (TEXTONLY is text_only-only)."""
    rc, out, err = _run("text_image", "PRISM-MODALITY-SMOKE-TEXTONLY-1N")
    assert rc == 0, f"non-zero exit: {err}"
    assert _count_launch_cmds(out) == 0
    assert "TEXTONLY design is only meaningful with the text_only preset" in err


def test_dataset_overrides_yaml_is_injected_per_preset():
    """run_sweep.py must add `data=per_modality_smoke/<preset>` to the launcher
    cmd when the matching yaml exists; otherwise the dataloader never sees
    the per-cell dataset_overrides and falls back to the global config (which
    activates only ts_qa regardless of model.modalities)."""
    rc, out, err = _run("text_image", "PRISM-MODALITY-SMOKE-1N")
    assert rc == 0, f"non-zero exit: {err}"
    assert "+data=per_modality_smoke/text_image" in out, (
        f"Expected per_modality_smoke wiring in launcher cmd:\n{out}"
    )


def test_mixed_presets_filter_per_cell():
    """Multi-preset run drops VLA-incompatible cells only, keeps the rest."""
    rc, out, err = _run("vla,text_image", "PRISM-AURORA-ZONE-A-VLA-CALVIN-SMOKE")
    assert rc == 0, f"non-zero exit: {err}"
    # vla preset is valid here (1 cmd); text_image is skipped (VLA design).
    assert _count_launch_cmds(out) == 1
    assert "VLA design requires the vla preset" in err


@pytest.mark.parametrize(
    "design",
    [
        "PRISM-AURORA-ZONE-A-VLA-CALVIN",
        "PRISM-AURORA-ZONE-A-VLA-CALVIN-DEFAULT",
        "PRISM-AURORA-ZONE-A-VLA-CALVIN-WEB",
        "PRISM-AURORA-ZONE-A-VLA-CALVIN-WEB-SMOKE",
    ],
)
def test_all_vla_design_variants_match_skip_rule(design):
    """Every VLA design id must match the `is_vla_design` predicate."""
    rc, out, err = _run("vla", design)
    assert rc == 0, f"non-zero exit: {err}"
    assert _count_launch_cmds(out) == 1, (
        f"{design} did not match the VLA skip-rule predicate: stderr={err}"
    )


# --- DL_NUM_WORKERS env scoping ---
#
# build_launcher_cmd previously mutated os.environ["DL_NUM_WORKERS"]="0" for
# non-image cells. In a mixed sweep (e.g. text_image,text_ts,text_image),
# the second text_image cell inherited DL_NUM_WORKERS=0 from the prior
# text_ts cell — silently losing parallel data loading. The fix returns
# the per-cell env via extra_env (a dict) instead.

def test_build_launcher_cmd_returns_extra_env_for_nonimage_cells():
    """Non-image cells must return DL_NUM_WORKERS=0 in extra_env, not mutate os.environ."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.run_sweep import build_launcher_cmd

    os.environ.pop("DL_NUM_WORKERS", None)
    cmd, extra_env = build_launcher_cmd(
        launcher="tools/launch_aurora.py",
        design="PRISM-MODALITY-SMOKE-1N",
        preset_name="text_ts",
        modalities=["text", "time_series"],
        sweep_id="sweepX",
        extra_args=[],
        dry_run=True,
    )
    assert extra_env == {"DL_NUM_WORKERS": "0"}, (
        f"text_ts cell must set DL_NUM_WORKERS=0 in extra_env, got {extra_env}"
    )
    assert os.environ.get("DL_NUM_WORKERS") is None, (
        "build_launcher_cmd must not mutate os.environ — that leaks to later cells"
    )


def test_build_launcher_cmd_image_cell_has_empty_extra_env():
    """Image cells must NOT set DL_NUM_WORKERS in extra_env."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.run_sweep import build_launcher_cmd

    cmd, extra_env = build_launcher_cmd(
        launcher="tools/launch_aurora.py",
        design="PRISM-MODALITY-SMOKE-1N",
        preset_name="text_image",
        modalities=["text", "image"],
        sweep_id="sweepX",
        extra_args=[],
        dry_run=True,
    )
    assert extra_env == {}, (
        f"text_image cell must have empty extra_env, got {extra_env}"
    )


def test_build_launcher_cmd_does_not_leak_between_cells():
    """Critical regression guard: running a non-image cell must not change
    the env seen by a subsequent image cell. This is the bug the fix closes —
    the prior implementation mutated os.environ['DL_NUM_WORKERS'] = '0' and
    every image cell after the first non-image cell inherited single-process
    data loading silently."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.run_sweep import build_launcher_cmd

    os.environ.pop("DL_NUM_WORKERS", None)
    # Simulate: text_ts (non-image, sets env) → text_image (image, must NOT see env)
    build_launcher_cmd(
        launcher="tools/launch_aurora.py",
        design="PRISM-MODALITY-SMOKE-1N",
        preset_name="text_ts",
        modalities=["text", "time_series"],
        sweep_id="sweepX",
        extra_args=[],
        dry_run=True,
    )
    assert os.environ.get("DL_NUM_WORKERS") is None, (
        "After non-image cell, parent os.environ must NOT carry DL_NUM_WORKERS"
    )

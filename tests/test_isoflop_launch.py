"""Tests for tools/isoflop_launch.py.

Uses a stub launcher (a tiny shell script) that just records its argv so
the test can assert structure without actually submitting PBS jobs. Mirrors
the subprocess-fixture pattern in tests/test_run_sweep.py.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH = REPO_ROOT / "tools" / "isoflop_launch.py"


def _make_stub_launcher(path: Path, argv_log: Path) -> None:
    """Write a tiny launcher stub that records argv + env-flag presence.

    Written in Python because isoflop_launch.py hardcodes `sys.executable`
    as the launcher interpreter (the real launcher is launch_aurora_daos.py).
    """
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"with open({str(argv_log)!r}, 'a') as _f:\n"
        "    _f.write('ARGV: ' + ' '.join(sys.argv[1:]) + '\\n')\n"
        "    _f.write('DL_NUM_WORKERS=' + os.environ.get('DL_NUM_WORKERS', 'UNSET') + '\\n')\n"
        "    _f.write('---\\n')\n"
        "sys.exit(0)\n"
    )
    path.chmod(0o755)


def _make_plan(tmp_path: Path, family: str = "text_image", n_cells: int = 2) -> Path:
    cells = []
    for i in range(n_cells):
        cells.append({
            "run_id": f"ISO-{family}-OLMO3-1B-BASE-1e18-s{i}",
            "phase": "M3", "family": family,
            "modalities": ["text", "image"] if family == "text_image" else ["text", "time_series"],
            "regime": "projector_only",
            "backbone": "OLMO3-1B",
            "backbone_hf_id": "allenai/Olmo-3-1B-Instruct",
            "projector_variant": "BASE",
            "projector_hidden_mult": 1, "projector_num_layers": 2,
            "budget_flops": 1e18,
            "seed": i, "nodes": 1, "batch_size": 8,
            "max_steps": 100, "max_seq_length": 2048,
            "tokens_total": 10000,
            "calibration_json": "/tmp/fake_cal.json",
            "design": "PRISM-IMAGE-ONLY-1N",
            "status": "planned",
        })
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump({
        "manifest_version": 1, "family": family,
        "regime": "projector_only", "cells": cells,
    }))
    return plan_path


def _run_launch(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(LAUNCH), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=60,
    )


def test_help_runs():
    r = _run_launch("--help")
    assert r.returncode == 0
    assert "--plan" in r.stdout


def test_print_mode_shows_cmd_per_cell(tmp_path: Path):
    """Launch invocation must use the plan's (rescaled) max_steps, NOT
    --target-flops. Post-feedback-PR: dropping --target-flops here was
    THE critical fix — without it, launch_aurora_daos.py would re-derive
    max_steps from raw cal_fps and undo the plan's cal/runtime rescale.
    """
    plan = _make_plan(tmp_path, n_cells=3)
    csv_path = tmp_path / "exp.csv"
    r = _run_launch("--plan", str(plan), "--csv", str(csv_path), "--print")
    assert r.returncode == 0, r.stderr
    # 3 commands printed
    cmds = [line for line in r.stdout.splitlines() if "launch_aurora_daos.py" in line]
    assert len(cmds) == 3
    for c in cmds:
        # CRITICAL: --target-flops MUST NOT be in the cmd — the launcher
        # would re-derive max_steps from raw cal_fps and undo the plan's
        # rescale (see review feedback on PR #98).
        assert "--target-flops" not in c, c
        # --max-steps MUST be passed with the plan's value (cells have
        # max_steps=100 per _make_plan fixture).
        assert "--max-steps 100" in c, c
        # --calibration-json still propagated for the trainer's _FlopCounter
        assert "--calibration-json" in c
        assert "model.projector_hidden_mult=1" in c
        assert "training.eval_enabled=true" in c
    # CSV should NOT have been written in --print mode
    assert not csv_path.exists()


def test_dry_run_passes_through_to_launcher(tmp_path: Path):
    """`--dry-run` should add `--dry-run` to each launcher cmd."""
    plan = _make_plan(tmp_path, n_cells=1)
    r = _run_launch("--plan", str(plan), "--csv", str(tmp_path / "exp.csv"),
                    "--print", "--dry-run")
    assert r.returncode == 0, r.stderr
    cmds = [line for line in r.stdout.splitlines() if "launch_aurora_daos.py" in line]
    assert "--dry-run" in cmds[0]


def test_idempotent_skips_done_cells(tmp_path: Path):
    """Cells with status=done in CSV must be skipped without --retry-failed."""
    plan = _make_plan(tmp_path, n_cells=2)
    csv_path = tmp_path / "exp.csv"
    # Pre-seed the CSV with the FIRST cell marked done.
    from tools.isoflop_launch import CSV_COLUMNS
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        first_id = "ISO-text_image-OLMO3-1B-BASE-1e18-s0"
        row = {c: "" for c in CSV_COLUMNS}
        row.update({"run_id": first_id, "family": "text_image", "status": "done"})
        w.writerow(row)
    r = _run_launch("--plan", str(plan), "--csv", str(csv_path), "--print")
    assert r.returncode == 0, r.stderr
    # Only s1 (second cell) should appear; s0 was skipped
    assert "BASE-1e18-s0" not in "\n".join(line for line in r.stdout.splitlines() if "launch" in line)
    assert "BASE-1e18-s1" in r.stdout


def test_filter_narrows_cells(tmp_path: Path):
    plan = _make_plan(tmp_path, n_cells=2)
    r = _run_launch(
        "--plan", str(plan), "--csv", str(tmp_path / "exp.csv"),
        "--print", "--filter", "seed=0",
    )
    assert r.returncode == 0
    cmds = [line for line in r.stdout.splitlines() if "launch_aurora_daos.py" in line]
    assert len(cmds) == 1
    assert "s0" in cmds[0]


def test_text_ts_family_sets_dl_num_workers(tmp_path: Path):
    """Non-image families must get DL_NUM_WORKERS=0 in the env prefix."""
    plan = _make_plan(tmp_path, family="text_ts", n_cells=1)
    r = _run_launch("--plan", str(plan), "--csv", str(tmp_path / "exp.csv"), "--print")
    assert r.returncode == 0, r.stderr
    assert "DL_NUM_WORKERS=0" in r.stdout
    assert "training.data_num_workers=0" in r.stdout


def test_dry_run_leaves_status_planned(tmp_path: Path):
    """Under --dry-run, status must stay 'planned' so a subsequent real
    launch picks the cell up. Flipping to 'running' would orphan it
    (the launcher exits 0 without submitting and the trainer never runs)."""
    plan = _make_plan(tmp_path, n_cells=1)
    csv_path = tmp_path / "exp.csv"
    argv_log = tmp_path / "argv.log"
    stub = tmp_path / "stub_launcher.sh"
    _make_stub_launcher(stub, argv_log)
    r = _run_launch(
        "--plan", str(plan), "--csv", str(csv_path),
        "--launcher", str(stub), "--dry-run",
    )
    assert r.returncode == 0, r.stderr
    # CSV row exists and is still planned (not running, not done).
    from tools.isoflop_launch import CSV_COLUMNS  # noqa: F401
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "planned"
    assert rows[0]["launched_at"] == ""


def test_real_launch_flips_status_running(tmp_path: Path):
    """Real (non-dry) launch must flip status to 'running' before invoking
    the launcher so a crash mid-launch leaves a trail."""
    plan = _make_plan(tmp_path, n_cells=1)
    csv_path = tmp_path / "exp.csv"
    argv_log = tmp_path / "argv.log"
    stub = tmp_path / "stub_launcher.sh"
    _make_stub_launcher(stub, argv_log)
    r = _run_launch(
        "--plan", str(plan), "--csv", str(csv_path),
        "--launcher", str(stub),
    )
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    # Stub exits 0 without writing perf.jsonl, so collector hasn't seen
    # it yet — status should be running (NOT done; the collector owns that).
    assert rows[0]["status"] == "running"
    assert rows[0]["launched_at"] != ""
    # Cell-derived projector knobs from the plan should be preserved.
    assert rows[0]["projector_hidden_mult"] == "1"
    assert rows[0]["projector_num_layers"] == "2"


def test_merge_order_cell_wins_for_structural_columns(tmp_path: Path):
    """A planned-then-relaunched cell must not lose structural columns to
    stale empties in `existing`. Regression test for the prior
    {**cell, **existing} ordering."""
    plan = _make_plan(tmp_path, n_cells=1)
    csv_path = tmp_path / "exp.csv"
    from tools.isoflop_launch import CSV_COLUMNS
    # Pre-seed CSV with a row that has structural empties but a stored note.
    run_id = "ISO-text_image-OLMO3-1B-BASE-1e18-s0"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        empty = {c: "" for c in CSV_COLUMNS}
        empty.update({
            "run_id": run_id, "status": "planned",
            "notes": "manual-note-should-survive",
            # All structural fields blank — exercises the merge order.
        })
        w.writerow(empty)

    argv_log = tmp_path / "argv.log"
    stub = tmp_path / "stub_launcher.sh"
    _make_stub_launcher(stub, argv_log)
    r = _run_launch(
        "--plan", str(plan), "--csv", str(csv_path),
        "--launcher", str(stub),
    )
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    # Cell-derived fields must NOT be the stale empties.
    assert row["family"] == "text_image"
    assert row["backbone"] == "OLMO3-1B"
    assert row["projector_variant"] == "BASE"
    assert row["nodes"] == "1"
    # And the manually-edited note must survive the relaunch.
    assert row["notes"] == "manual-note-should-survive"


def test_launch_uses_plan_max_steps_not_target_flops_recompute(tmp_path: Path):
    """End-to-end assertion of the post-feedback rescale-bypass fix.

    The plan rescales `cal_fps` → `runtime_fps` and writes a rescaled
    `max_steps` into the cell. The launcher must use THAT step count,
    not re-derive it from `target_flops / cal_fps`. Critical because
    the entire rescale fix is defeated if the launcher re-derives.

    We construct a plan where cell.max_steps differs sharply from
    `budget_flops / calibration.flops_per_step` and confirm the launcher
    cmd carries `--max-steps <cell.max_steps>` (not the raw quotient).
    """
    # Synthetic plan: budget=1e18, cell.max_steps=42 (a value that has
    # no natural relation to any cal_fps). A re-derived launcher would
    # pick something like 1e18/2e15 = 500, which is what we MUST NOT see.
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump({
        "manifest_version": 1, "family": "text_image",
        "regime": "projector_only",
        "cells": [{
            "run_id": "ISO-PLAN-RESCALE-TEST",
            "phase": "M3", "family": "text_image",
            "modalities": ["text", "image"], "regime": "projector_only",
            "backbone": "OLMO3-1B",
            "backbone_hf_id": "allenai/Olmo-3-1B-Instruct",
            "projector_variant": "BASE",
            "projector_hidden_mult": 1, "projector_num_layers": 2,
            "budget_flops": 1.0e18, "seed": 0, "nodes": 1,
            "batch_size": 8,
            "max_steps": 42,  # ← intentionally not budget/cal_fps
            "max_seq_length": 2048, "tokens_total": 10000,
            "calibration_json": "/tmp/fake_cal.json",
            "design": "PRISM-IMAGE-ONLY-1N",
            "status": "planned",
        }],
    }))
    csv_path = tmp_path / "exp.csv"
    r = _run_launch("--plan", str(plan_path), "--csv", str(csv_path), "--print")
    assert r.returncode == 0, r.stderr
    cmd_line = next(
        line for line in r.stdout.splitlines() if "launch_aurora_daos.py" in line
    )
    # The launcher cmd MUST carry the plan's --max-steps, not re-derive.
    assert "--max-steps 42" in cmd_line, cmd_line
    # And MUST NOT have --target-flops (which would trigger the launcher's
    # raw cal_fps recompute).
    assert "--target-flops" not in cmd_line, cmd_line
    # eval_every_n_steps must be max(1, 42 // 10) = 4, NOT based on any
    # stale pre-rescale step count.
    assert "training.eval_every_n_steps=4" in cmd_line, cmd_line


def test_launch_propagates_runtime_fps_when_present(tmp_path: Path):
    """When the plan stamps `runtime_fps` on a cell, the launcher cmd must
    carry `--runtime-flops-per-step <runtime_fps>` so the trainer's
    _FlopCounter accumulates in rescaled FLOPs (matching `budget_flops`)
    instead of raw cal FPS (which would undercount by `rescale_factor`).
    See PR feedback on PR #98.
    """
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump({
        "manifest_version": 1, "family": "text_image",
        "regime": "projector_only",
        "cells": [{
            "run_id": "ISO-RFPS-PROPAGATE",
            "phase": "M3", "family": "text_image",
            "modalities": ["text", "image"], "regime": "projector_only",
            "backbone": "OLMO3-1B",
            "backbone_hf_id": "allenai/Olmo-3-1B-Instruct",
            "projector_variant": "BASE",
            "projector_hidden_mult": 1, "projector_num_layers": 2,
            "budget_flops": 1.0e18, "seed": 0, "nodes": 1,
            "batch_size": 8,
            "max_steps": 10,
            "max_seq_length": 2048, "tokens_total": 10000,
            "calibration_json": "/tmp/fake_cal.json",
            "calibration_fps": 2.0e15,
            "runtime_fps": 9.6e16,          # 48x rescale
            "rescale_factor": 48.0,
            "design": "PRISM-IMAGE-ONLY-1N",
            "status": "planned",
        }],
    }))
    csv_path = tmp_path / "exp.csv"
    r = _run_launch("--plan", str(plan_path), "--csv", str(csv_path), "--print")
    assert r.returncode == 0, r.stderr
    cmd_line = next(
        line for line in r.stdout.splitlines() if "launch_aurora_daos.py" in line
    )
    assert "--runtime-flops-per-step" in cmd_line, cmd_line
    # Numeric value: 9.6e16 formatted with 6-decimal scientific notation.
    assert "9.600000e+16" in cmd_line, cmd_line


def test_launch_propagates_seed_per_cell(tmp_path: Path):
    """Every cell cmd must carry `system.seed=<cell.seed>`. Stage A IsoFLOP's
    variance-floor measurement runs BASE@C with seeds {0,1,2,3} and the
    replicas would be bit-identical without per-cell seed plumbing (pre-A).
    """
    plan = _make_plan(tmp_path, n_cells=3)  # seeds = 0, 1, 2
    csv_path = tmp_path / "exp.csv"
    r = _run_launch("--plan", str(plan), "--csv", str(csv_path), "--print")
    assert r.returncode == 0, r.stderr
    cmds = [line for line in r.stdout.splitlines() if "launch_aurora_daos.py" in line]
    assert len(cmds) == 3
    # Each cell's cmd has system.seed=<cell.seed> matching its run_id suffix.
    for cmd, expected_seed in zip(cmds, (0, 1, 2), strict=True):
        assert f"system.seed={expected_seed}" in cmd, (expected_seed, cmd)


def test_launcher_arg_passthrough(tmp_path: Path):
    """`--launcher-arg X --launcher-arg Y` forwards X and Y verbatim into
    every cell's launcher cmd, spliced BEFORE the Hydra k=v overrides
    (so they're parsed as launcher flags, not Hydra unknown_args).

    Required for Stage A IsoFLOP: `launch_aurora_web.py --webdataset-dir
    /flare/.../pixmo_cap_webdataset` must reach the launcher for shard
    staging to fire, and isoflop_launch.py doesn't know about that flag.
    """
    plan = _make_plan(tmp_path, n_cells=2)
    csv_path = tmp_path / "exp.csv"
    r = _run_launch(
        "--plan", str(plan), "--csv", str(csv_path), "--print",
        "--launcher-arg=--webdataset-dir",
        "--launcher-arg=/flare/test/pixmo",
        "--launcher-arg=--shared-hf-home",
        "--launcher-arg=/flare/test/hub",
    )
    assert r.returncode == 0, r.stderr
    cmds = [line for line in r.stdout.splitlines() if "launch_aurora_daos.py" in line]
    assert len(cmds) == 2
    for cmd in cmds:
        assert "--webdataset-dir /flare/test/pixmo" in cmd, cmd
        assert "--shared-hf-home /flare/test/hub" in cmd, cmd
        # Spliced BEFORE Hydra overrides — the launcher arg should appear
        # before the first 'model.foo=bar' token.
        idx_webdataset = cmd.find("--webdataset-dir")
        idx_hydra = cmd.find("model.projector_hidden_mult=")
        assert idx_webdataset < idx_hydra, (
            f"--webdataset-dir must appear before Hydra overrides, got:\n{cmd}"
        )


def test_launch_omits_runtime_fps_when_absent_in_plan(tmp_path: Path):
    """Back-compat: a plan from before PR #98's rescale fix has no
    `runtime_fps` field on cells. The launcher cmd must still build (no
    crash) but must not carry --runtime-flops-per-step.
    """
    # _make_plan does NOT stamp runtime_fps — older plan shape
    plan = _make_plan(tmp_path, n_cells=1)
    csv_path = tmp_path / "exp.csv"
    r = _run_launch("--plan", str(plan), "--csv", str(csv_path), "--print")
    assert r.returncode == 0, r.stderr
    cmd_line = next(
        line for line in r.stdout.splitlines() if "launch_aurora_daos.py" in line
    )
    assert "--runtime-flops-per-step" not in cmd_line, cmd_line

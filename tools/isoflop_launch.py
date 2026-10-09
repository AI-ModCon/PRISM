#!/usr/bin/env python3
"""Launch IsoFLOP cells from a `tools/isoflop_plan.py` YAML manifest.

For each cell in the plan, invokes `tools/launch_aurora_daos.py` with
`--target-flops` (computed from the budget + calibration JSON) and the
projector knobs. Tracks per-cell status in `experiments.csv` so the
sweep is idempotent: cells already `running` / `done` are skipped unless
`--retry-failed` is passed.

Mirrors `tools/run_sweep.py` (288 lines) structurally: per-cell `extra_env`
scoping prevents `DL_NUM_WORKERS=0` (for non-image cells) from leaking
into the next cell's subprocess.

Usage:
    python tools/isoflop_launch.py \\
      --plan /tmp/iso-plan-text_image.yaml \\
      --print           # show cmds without invoking
    python tools/isoflop_launch.py --plan /tmp/iso-plan-text_image.yaml --dry-run
    python tools/isoflop_launch.py --plan /tmp/iso-plan-text_image.yaml
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER = REPO_ROOT / "tools" / "launch_aurora_daos.py"

# experiments.csv schema, pinned so column drift surfaces loudly in
# both this file and isoflop_collect.py (which mirrors the same tuple).
CSV_COLUMNS: tuple[str, ...] = (
    "run_id", "phase", "family", "regime", "backbone", "projector_variant",
    "projector_hidden_mult", "projector_num_layers",
    "budget_flops", "seed", "nodes", "batch_size", "grad_accum",
    "max_seq_length", "max_steps", "n_total_params", "n_active_params",
    "n_trainable_params", "flops_per_step", "cumulative_flops",
    "d_compute_tokens", "d_label_tokens", "d_modality_tokens",
    "samples_per_sec", "tokens_per_sec", "seq_p50", "seq_p95", "seq_p99",
    "padding_ratio", "loss_main", "loss_caption",
    "loss_pointing", "loss_count", "loss_ts_qa", "loss_graph_qa",
    "loss_source",  # must mirror tools/isoflop_collect.py:CSV_COLUMNS
    "loss_stability",  # sample stdev over the last `window` eval rows
    "downstream_metric", "status", "job_id", "launched_at",
    "completed_at", "wandb_url", "notes",
)


def _load_plan(path: Path) -> dict[str, Any]:
    with open(path) as f:
        plan = yaml.safe_load(f)
    if not isinstance(plan, dict) or "cells" not in plan:
        raise ValueError(f"{path}: not a plan manifest (missing 'cells')")
    return plan


def _load_csv(path: Path) -> dict[str, dict[str, str]]:
    """Read experiments.csv into {run_id: row_dict}.

    Returns empty dict if the file doesn't exist yet — the launcher will
    create it from CSV_COLUMNS.
    """
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is not None and set(reader.fieldnames) != set(CSV_COLUMNS):
            extra = set(reader.fieldnames) - set(CSV_COLUMNS)
            missing = set(CSV_COLUMNS) - set(reader.fieldnames)
            print(
                f"WARNING: {path}: column schema drift. extra={extra} missing={missing}",
                file=sys.stderr,
            )
        for row in reader:
            rid = row.get("run_id")
            if rid:
                rows[rid] = row
    return rows


def _write_csv(path: Path, rows: dict[str, dict[str, str]]) -> None:
    """Rewrite experiments.csv from the current row map.

    PR-2 owns CSV writes (PR-1 emits only perf.jsonl). Sort by run_id for
    a deterministic diff and to make grep-based inspection easier.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for rid in sorted(rows):
            # Ensure every column is present so DictWriter doesn't barf.
            row = {col: rows[rid].get(col, "") for col in CSV_COLUMNS}
            writer.writerow(row)


def _cell_to_csv_row(cell: dict[str, Any]) -> dict[str, str]:
    """Convert a plan cell into an initial CSV row (mostly empty until collect).

    Includes projector_hidden_mult/_num_layers from the plan so the row is
    self-describing even before the collector parses the startup_param_count
    event from perf.jsonl.
    """
    return {
        "run_id": str(cell["run_id"]),
        "phase": str(cell.get("phase", "")),
        "family": str(cell.get("family", "")),
        "regime": str(cell.get("regime", "")),
        "backbone": str(cell.get("backbone", "")),
        "projector_variant": str(cell.get("projector_variant", "")),
        "projector_hidden_mult": str(cell.get("projector_hidden_mult", "")),
        "projector_num_layers": str(cell.get("projector_num_layers", "")),
        "budget_flops": _fmt_float(cell.get("budget_flops")),
        "seed": str(cell.get("seed", "")),
        "nodes": str(cell.get("nodes", "")),
        "batch_size": str(cell.get("batch_size", "")),
        "grad_accum": str(cell.get("grad_accum", "")),
        "max_seq_length": str(cell.get("max_seq_length", "")),
        "max_steps": str(cell.get("max_steps", "")),
        "status": "planned",
    }


def _fmt_float(v: Any) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.6e}"
    except (TypeError, ValueError):
        return str(v)


def _matches_filter(cell: dict[str, Any], filters: list[str]) -> bool:
    """`--filter family=text_image` etc. AND across all flags."""
    for f in filters:
        if "=" not in f:
            continue
        k, v = f.split("=", 1)
        if str(cell.get(k.strip(), "")) != v.strip():
            return False
    return True


def build_launcher_cmd(
    cell: dict[str, Any],
    launcher: Path,
    sweep_id: str,
    dry_run: bool,
    extra_launcher_args: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build the launch_aurora_daos.py invocation for one cell.

    Returns `(cmd, extra_env)`. Caller must merge `extra_env` per-subprocess
    (NOT into os.environ) — mirrors the run_sweep.py pattern that prevents
    DL_NUM_WORKERS=0 from leaking into the next cell.
    """
    run_id = str(cell["run_id"])
    family = str(cell.get("family", ""))
    modalities = cell.get("modalities") or []
    modalities_str = ",".join(str(m) for m in modalities)
    # IMPORTANT: pass --max-steps (not --target-flops) so the launcher uses
    # the plan's *rescaled* step count, not its own raw cal_fps-based
    # derivation. Without this, the launcher's --target-flops handler at
    # `tools/launch_aurora_daos.py:424` would recompute
    # `max_steps = target / cal_fps` from the raw JSON, undoing the plan's
    # cal/runtime FLOPs rescale (see PR feedback on PR #98). The calibration
    # JSON is still passed via --calibration-json so the trainer's
    # _FlopCounter gets the env-var export, but the step count is now
    # authoritatively set by the plan.
    cell_max_steps = int(cell["max_steps"])
    cmd: list[str] = [
        sys.executable,
        str(launcher),
        "--id", run_id,
        "--design", str(cell["design"]),
        "--nodes", str(cell["nodes"]),
        "--max-steps", str(cell_max_steps),
        "--calibration-json", str(cell["calibration_json"]),
    ]
    # Pass the plan's rescaled runtime FPS so the trainer's _FlopCounter
    # accumulates in rescaled FLOPs. Without this, perf.jsonl's
    # cumulative_flops would undercount the plan's budget_flops by
    # `rescale_factor` (cal_fps vs runtime_fps) — the fit's x-axis
    # wouldn't match the budget axis. Falls back gracefully when an old
    # plan without `runtime_fps` is loaded (the trainer just uses raw
    # cal_fps as before).
    if cell.get("runtime_fps") is not None:
        cmd.extend(["--runtime-flops-per-step", f"{float(cell['runtime_fps']):.6e}"])
    # Hydra overrides — pin projector knobs, modalities, backbone, sweep
    # metadata, and the eval gate so the collector has loss_<family> rows.
    cmd.extend([
        f"model.projector_hidden_mult={int(cell['projector_hidden_mult'])}",
        f"model.projector_num_layers={int(cell['projector_num_layers'])}",
        f"model.modalities=[{modalities_str}]",
        f"exp.sweep_id={sweep_id}",
        f"exp.preset={family}",
        "training.eval_enabled=true",
        # Eval cadence: ~10 evals per cell. Cell.max_steps may be tiny on
        # smokes, so floor at 1 to avoid `step % 0`. Uses the same
        # cell_max_steps the launcher will see — so eval fires 10× per
        # actual run, not 10× per pre-rescale-overshoot step count.
        f"training.eval_every_n_steps={max(1, cell_max_steps // 10)}",
        # Stage A variance-floor: each replica needs a distinct RNG seed so
        # the BASE@C-fixed seed-replicas actually diverge. The plan emits
        # seed=0..N-1; the trainer reads system.seed in src/train.py.
        f"system.seed={int(cell.get('seed', 0))}",
    ])
    if cell.get("backbone_hf_id"):
        cmd.append(f"model.backbone_id={cell['backbone_hf_id']}")

    # Forward repeatable --launcher-arg passthrough (e.g.
    # `--launcher-arg --webdataset-dir --launcher-arg /flare/.../pixmo_cap_webdataset`
    # for Stage A IsoFLOP runs that need shard staging). Inserted BEFORE the
    # Hydra overrides so they appear as launcher flags, not unknown_args.
    if extra_launcher_args:
        # Insert after the fixed launcher flags (--id, --design, --nodes,
        # --max-steps, --calibration-json, --runtime-flops-per-step) and
        # before the Hydra k=v overrides. Find the first Hydra-style entry
        # (contains '=' and doesn't start with '-') and splice before it.
        first_hydra = next(
            (i for i, x in enumerate(cmd) if "=" in x and not x.startswith("-")),
            len(cmd),
        )
        cmd[first_hydra:first_hydra] = [str(a) for a in extra_launcher_args]

    extra_env: dict[str, str] = {}
    # Non-image families: HF IterableDatasets report n_shards=1, which
    # causes all but one dataloader worker to silently stop. Force
    # single-process loading per the run_sweep.py pattern.
    if family in ("text_ts", "text_graph", "text_table", "text_geometry", "text_only"):
        cmd.append("training.data_num_workers=0")
        extra_env["DL_NUM_WORKERS"] = "0"

    if dry_run:
        cmd.append("--dry-run")
    return cmd, extra_env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, help="YAML manifest from tools/isoflop_plan.py")
    p.add_argument(
        "--csv",
        default=None,
        help="experiments.csv path. Defaults to <scaling-study-dir>/experiments.csv",
    )
    p.add_argument(
        "--scaling-study-dir",
        default="/flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/",
    )
    p.add_argument(
        "--launcher",
        default=str(DEFAULT_LAUNCHER),
        help="Path to launch_aurora_daos.py (override for testing)",
    )
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Filter cells (key=value). Repeatable. e.g. --filter family=text_image",
    )
    p.add_argument("--sweep-id", default=None,
                   help="Group label propagated to exp.sweep_id. Auto-generated if omitted.")
    p.add_argument("--print", action="store_true",
                   help="Print commands without invoking the launcher")
    p.add_argument("--dry-run", action="store_true",
                   help="Pass --dry-run to launcher (generates qsub script, no submit)")
    p.add_argument("--retry-failed", action="store_true",
                   help="Also re-run cells whose status is 'failed'")
    p.add_argument(
        "--launcher-arg",
        action="append",
        default=[],
        help="Repeatable. Forwarded verbatim into every cell's launcher cmd "
             "(inserted before the Hydra k=v overrides). Use to pass "
             "launcher-specific flags isoflop_launch doesn't know about, e.g. "
             "`--launcher-arg --webdataset-dir --launcher-arg /flare/.../pixmo_cap_webdataset` "
             "for Stage A IsoFLOP runs that need shard staging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    plan_path = Path(args.plan)
    plan = _load_plan(plan_path)
    cells = plan.get("cells", [])

    csv_path = Path(args.csv) if args.csv else (
        Path(args.scaling_study_dir) / "experiments.csv"
    )
    rows = _load_csv(csv_path)

    sweep_id = args.sweep_id or (
        _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    launcher = Path(args.launcher)
    print(f"# isoflop_launch sweep_id={sweep_id}  plan={plan_path}", file=sys.stderr)
    print(f"# {len(cells)} cells in plan; csv={csv_path}", file=sys.stderr)

    failures = 0
    skipped_idempotent = 0
    launched = 0
    for cell in cells:
        if not _matches_filter(cell, args.filter):
            continue
        run_id = str(cell["run_id"])
        existing = rows.get(run_id, {})
        status = (existing.get("status") or "").strip()
        if status in ("running", "done") and not args.retry_failed:
            skipped_idempotent += 1
            print(f"# skip {run_id} (status={status})", file=sys.stderr)
            continue
        if status == "failed" and not args.retry_failed:
            skipped_idempotent += 1
            print(f"# skip {run_id} (status=failed; pass --retry-failed to rerun)",
                  file=sys.stderr)
            continue

        cmd, extra_env = build_launcher_cmd(
            cell, launcher, sweep_id, args.dry_run,
            extra_launcher_args=args.launcher_arg,
        )

        # XCCL cleanup so the next cell starts from a clean state. The
        # in-launcher heredoc also cleans on the compute node; this is the
        # local-side cleanup mirroring run_sweep.py convention.
        tmpdir = f"/tmp/xccl_{os.environ.get('USER', 'unknown')}"
        try:
            subprocess.run(["rm", "-rf", tmpdir], check=False, capture_output=True)
        except Exception:  # noqa: BLE001
            pass

        env_prefix = " ".join(f"{k}={v}" for k, v in extra_env.items())
        full = (env_prefix + " " if env_prefix else "") + " ".join(shlex.quote(c) for c in cmd)
        print(full)
        if args.print:
            continue

        # Initial row: cell fields win for structural columns; preserve
        # any human-edited fields from `existing` (notes, downstream_metric)
        # by layering them on top last but only after re-pinning status.
        new_row = {**existing, **_cell_to_csv_row(cell)}
        # Under --dry-run the launcher exits 0 after writing qsub.sh
        # without submitting; flipping status=running here would orphan
        # the cell (a subsequent real launch would skip it as "running").
        # Leave status=planned so the next non-dry invocation picks it up.
        if args.dry_run:
            new_row["status"] = "planned"
            rows[run_id] = new_row
            _write_csv(csv_path, rows)
        else:
            # status=running so a crash mid-launch leaves a trail.
            new_row["status"] = "running"
            new_row["launched_at"] = _dt.datetime.utcnow().isoformat() + "Z"
            rows[run_id] = new_row
            _write_csv(csv_path, rows)

        cell_env = {**os.environ, **extra_env} if extra_env else None
        try:
            subprocess.run(cmd, check=True, env=cell_env)
            launched += 1
        except subprocess.CalledProcessError as e:
            print(f"FAILED ({e.returncode}): {run_id}", file=sys.stderr)
            failures += 1
            # On dry-run failure leave status=planned (no real attempt made).
            if not args.dry_run:
                rows[run_id]["status"] = "failed"
                rows[run_id]["completed_at"] = _dt.datetime.utcnow().isoformat() + "Z"
                _write_csv(csv_path, rows)
            continue
        # Note: even on a successful real launch we DON'T flip status to
        # "done" here — only the collector does that, after parsing
        # perf.jsonl. The trainer process may still be running.

    print(
        f"# isoflop_launch summary: launched={launched} skipped={skipped_idempotent} failures={failures}",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

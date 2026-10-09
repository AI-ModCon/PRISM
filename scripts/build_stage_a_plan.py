#!/usr/bin/env python3
"""Build the Stage A IsoFLOP round 1 plan manifest for OLMo-3 1B.

Stage A round 1 needs 18 cells:
  - Main capacity ladder: 3 budgets × 5 variants × 1 seed (seed=0) = 15 cells
  - Variance-floor measurement: BASE@3e17 × seeds {1,2,3} = 3 cells

The 18th cell is "seed=0 of BASE@3e17" from the main grid; the variance-floor
needs *4* distinct seeds of BASE@3e17, so seeds 1-3 extend the main grid.

This helper invokes `tools/isoflop_plan.py` twice (main grid and
seed-replicas) and merges the cell lists, deduping by run_id. Output is a
single manifest that `tools/isoflop_launch.py` can consume directly.

Why a helper instead of `--seeds 4` in one call: `--seeds 4` would produce
4 seeds for EVERY (budget × variant) combination → 60 cells = ~5× the
compute. The seed-replica budget is specifically pinned to the lowest
budget × BASE variant.

Usage:
    python scripts/build_stage_a_plan.py \\
      --calibration-dir /flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/calibration/ \\
      --output /flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/stage_a_round1_olmo3_1b.yaml \\
      --nodes-override 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_TOOL = REPO_ROOT / "tools" / "isoflop_plan.py"


def _run_plan(args: list[str]) -> Path:
    """Invoke isoflop_plan.py, return the path it wrote."""
    out_path = Path(args[args.index("--output") + 1])
    result = subprocess.run(
        [sys.executable, str(PLAN_TOOL), *args],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"isoflop_plan.py failed (rc={result.returncode})")
    sys.stderr.write(result.stderr)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--calibration-dir",
        default="/flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/calibration/",
        help="Where to read OLMO3-1B-<VARIANT>-text_image-projector_only.json from.",
    )
    p.add_argument(
        "--output", required=True,
        help="Path to write the merged Stage A round 1 manifest.",
    )
    p.add_argument(
        "--nodes-override", type=int, default=4,
        help="Per-cell node count (default 4 for round 1 wall-clock speed).",
    )
    p.add_argument(
        "--runtime-batch-size", type=int, default=8,
        help="Per-rank batch size at runtime. Default 8 matches PRISM-IMAGE-ONLY-*.",
    )
    p.add_argument(
        "--runtime-seq-len", type=int, default=2048,
        help="Per-sample seq length. Default 2048 matches launcher default.",
    )
    p.add_argument(
        "--runtime-ranks-per-node", type=int, default=12,
        help="Aurora ppn (12 XPU tiles).",
    )
    p.add_argument(
        "--budgets", default="3e17,1e18,3e18",
        help="Comma-separated FLOP budgets for the main grid.",
    )
    p.add_argument(
        "--variants", default="BASE,W2X,W4X,D2X,D4X",
        help="Projector variants for the main grid.",
    )
    p.add_argument(
        "--variance-floor-budget", default="3e17",
        help="Budget at which the BASE variance-floor seed-replicas run.",
    )
    p.add_argument(
        "--variance-floor-seeds", type=int, default=3,
        help="Number of additional BASE seed-replicas (seeds 1..N). "
             "The main grid contributes seed=0, giving N+1 distinct seeds total.",
    )
    p.add_argument(
        "--skip-backbone-check", action="store_true",
        help="Pass through to isoflop_plan.py. Use when backbone staging will "
             "happen between plan generation and launch.",
    )
    p.add_argument(
        "--design-override", default="PRISM-IMAGE-ONLY-4N",
        help="Override `design` in every cell. _BACKBONE_TABLE['OLMO3-1B']['design'] "
             "defaults to PRISM-IMAGE-ONLY-1N (1-node LR/warmup tuning), but at "
             "--nodes-override 4 we want the 4N-tuned design (LR=7e-4, warmup=400). "
             "Set to '' to keep the table default.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    shared = [
        "--family", "text_image",
        "--backbones", "OLMO3-1B",
        "--calibration-dir", args.calibration_dir,
        "--nodes-override", str(args.nodes_override),
        "--runtime-batch-size", str(args.runtime_batch_size),
        "--runtime-seq-len", str(args.runtime_seq_len),
        "--runtime-ranks-per-node", str(args.runtime_ranks_per_node),
    ]
    if args.skip_backbone_check:
        shared.append("--skip-backbone-check")

    with tempfile.TemporaryDirectory(prefix="stage_a_plan_") as td:
        td_path = Path(td)

        # Sub-plan 1: main grid (seed=0 across all budgets × variants)
        main_path = td_path / "main.yaml"
        _run_plan([
            *shared,
            "--budgets", args.budgets,
            "--projector-variants", args.variants,
            "--seeds", "1",   # seed=0 only
            "--output", str(main_path),
        ])

        # Sub-plan 2: variance-floor (BASE@<vf-budget> for seeds 0..N).
        # We then drop seed=0 (already in main grid) and keep seeds 1..N.
        vf_path = td_path / "vf.yaml"
        _run_plan([
            *shared,
            "--budgets", args.variance_floor_budget,
            "--projector-variants", "BASE",
            "--seeds", str(args.variance_floor_seeds + 1),
            "--output", str(vf_path),
        ])

        main_plan = yaml.safe_load(main_path.read_text())
        vf_plan = yaml.safe_load(vf_path.read_text())

    # Filter variance-floor to seeds 1..N (drop seed=0; it's in main grid).
    vf_extra = [c for c in vf_plan["cells"] if int(c["seed"]) >= 1]

    # Dedupe by run_id (defensive — should already be disjoint).
    seen = {c["run_id"] for c in main_plan["cells"]}
    deduped_vf = [c for c in vf_extra if c["run_id"] not in seen]

    merged_cells = main_plan["cells"] + deduped_vf

    # Apply design override post-merge. isoflop_plan.py picks design from
    # _BACKBONE_TABLE which is hard-coded to PRISM-IMAGE-ONLY-1N for OLMO3-1B,
    # but the 4-node run needs PRISM-IMAGE-ONLY-4N (LR=7e-4, warmup=400).
    # Leaving the LR/warmup at 1N tuning would silently miscalibrate every
    # cell — IsoFLOP loss curves require consistent + appropriate hyperparams.
    if args.design_override:
        for c in merged_cells:
            c["design"] = args.design_override
    expected = (
        len(args.budgets.split(",")) * len(args.variants.split(","))
        + args.variance_floor_seeds
    )
    if len(merged_cells) != expected:
        print(
            f"WARNING: expected {expected} merged cells, got {len(merged_cells)} "
            f"(main={len(main_plan['cells'])}, vf_extra={len(deduped_vf)})",
            file=sys.stderr,
        )

    merged_manifest = {
        "manifest_version": 1,
        "generated_at": main_plan.get("generated_at"),
        "family": "text_image",
        "regime": main_plan.get("regime", "projector_only"),
        "calibration_dir": main_plan.get("calibration_dir"),
        "stage": "A",
        "round": 1,
        "backbone": "OLMO3-1B",
        "notes": (
            "Stage A round 1: 15-cell main grid (3 budgets × 5 variants × seed=0) "
            f"+ {args.variance_floor_seeds} BASE@{args.variance_floor_budget} seed-replicas "
            f"(seeds 1..{args.variance_floor_seeds}) for variance-floor measurement. The variance-floor "
            "analysis groups these with seed=0 of BASE@vf-budget from the main grid, "
            f"giving {args.variance_floor_seeds + 1} total replicas."
        ),
        "cells": merged_cells,
    }
    skipped = list(main_plan.get("skipped", [])) + list(vf_plan.get("skipped", []))
    if skipped:
        merged_manifest["skipped"] = skipped

    with open(out, "w") as f:
        yaml.safe_dump(merged_manifest, f, sort_keys=False)

    print(f"[stage_a_plan] wrote {len(merged_cells)} cells to {out}")
    print(f"  main grid: {len(main_plan['cells'])} cells (seed=0)")
    print(f"  variance-floor: {len(deduped_vf)} cells (BASE@{args.variance_floor_budget}, "
          f"seeds 1..{args.variance_floor_seeds})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

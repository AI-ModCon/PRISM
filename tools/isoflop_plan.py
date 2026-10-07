#!/usr/bin/env python3
"""Generate an IsoFLOP run manifest from calibration JSONs + budget list.

For each `(backbone, projector_variant, budget, seed)` in the cross-product:
  1. Look up `<calibration-dir>/{backbone}-{variant}-{family}-projector_only.json`.
  2. Compute `max_steps = round(budget / cal["flops_per_step"])`.
  3. Compute `tokens_total = max_steps * global_batch_size * cal["mean_seq_len"]`.
  4. Emit a YAML row matching the 36-col `experiments.csv` schema.

Backbones get the budget filter from `_budgets_for_backbone()` per PLAN.md §3.3:
  - 1B class (OLMo-3 1B, Q3-0.6B–1.7B): full ladder.
  - 7B–8B class: from 1e19 up.
  - 14B+: top two budgets only.

Missing calibration → warn + skip (not fatal — partial manifests are fine).

Downstream: `tools/isoflop_launch.py` consumes the YAML row by row, mirroring
`tools/run_sweep.py`.

Usage:
    python tools/isoflop_plan.py \\
      --family text_image \\
      --budgets 3e17,1e18,3e18 \\
      --backbones OLMO3-1B \\
      --projector-variants BASE,W2X,W4X,D2X,D4X \\
      --calibration-dir /flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/calibration/ \\
      --output /tmp/iso-plan-text_image.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.projector import ModalityProjector  # noqa: E402

# Mapping from short backbone label to (hf_id, design_id_template, nodes).
# `design_id_template` is the prism_designs.yaml entry the launcher will
# resolve; nodes is the PBS node count to request. Values mirror PLAN.md
# §3.3 and the existing OLMo configs in experiments/prism_designs.yaml.
#
# IMPORTANT: hf_id MUST match a model that's been staged into a HuggingFace
# cache dir on /flare — the trainer loads with `local_files_only=True`.
# The OLMo-3 1B/7B Instruct checkpoints aren't on /flare as of 2026-05-27,
# so the table uses the OLMo-1B-0724-hf / OLMo-7B-0724-hf bases (the
# checkpoints the existing PRISM-IMAGE-ONLY designs already wire). Keep
# the OLMO3-* label keys to leave room for a later swap once the OLMo-3
# Instruct weights are staged; `validate_backbone_caches` below catches
# any divergence at plan time so we don't discover the gap mid-launch.
# 32B OLMo doesn't ship in either generation, so OLMO3-32B is intentionally
# absent — re-add once a 32B-class backbone is staged.
_BACKBONE_TABLE: dict[str, dict[str, Any]] = {
    "OLMO3-1B": {
        "hf_id": "allenai/OLMo-1B-0724-hf",
        "design": "PRISM-IMAGE-ONLY-1N",  # 1-node DDP design as default
        "nodes": 1,
    },
    "OLMO3-7B": {
        "hf_id": "allenai/OLMo-7B-0724-hf",
        "design": "PRISM-IMAGE-ONLY-2N",
        "nodes": 2,
    },
}

# HF cache roots to probe when validating backbone availability. Order
# matches the launcher's `--hf-fallback-dirs` default (personal first, then
# shared), so plan time and run time agree on which copy will be picked.
_HF_CACHE_ROOTS: tuple[str, ...] = (
    "/flare/ModCon/ngetty/huggingface/hub",
    "/flare/ModCon/sandeep/hub",
)


def _hf_cache_dir_name(hf_id: str) -> str:
    """`allenai/OLMo-1B-0724-hf` → `models--allenai--OLMo-1B-0724-hf`."""
    return "models--" + hf_id.replace("/", "--")


def _backbone_is_staged(hf_id: str) -> bool:
    """True iff `hf_id` has a non-empty `snapshots/` dir in some HF cache root.

    Probing just the top-level `models--<...>/` directory isn't enough — HF
    can leave a partial layout (e.g. `.no_exist/<rev>/model.safetensors`
    markers; see the Tapas shared-hub incident in memory) that satisfies
    `is_dir()` but still fails `from_pretrained(local_files_only=True)`.
    Requiring `snapshots/` to exist and contain at least one revision dir
    catches the common broken-snapshot case cheaply.
    """
    dirname = _hf_cache_dir_name(hf_id)
    for root in _HF_CACHE_ROOTS:
        snapshots = Path(root) / dirname / "snapshots"
        if snapshots.is_dir() and any(snapshots.iterdir()):
            return True
    return False


_FAMILY_TO_MODALITIES: dict[str, list[str]] = {
    "text_image": ["text", "image"],
    "text_ts": ["text", "time_series"],
    "text_graph": ["text", "graph"],
}


def _budgets_for_backbone(backbone: str, requested: list[float]) -> list[float]:
    """Filter the requested budget list down to what fits the backbone.

    Per PLAN.md §3.3:
      - 1B class: full ladder
      - 7B/8B class: budgets >= 1e19
      - 14B+: top two budgets only
    Unknown backbones default to the full ladder (caller's responsibility).
    """
    if backbone.endswith("-1B") or backbone.endswith("-0.6B") or backbone.endswith("-1.7B"):
        return list(requested)
    if backbone.endswith("-7B") or backbone.endswith("-8B") or backbone.endswith("-4B"):
        return [b for b in requested if b >= 1e19]
    if backbone.endswith("-14B") or backbone.endswith("-32B"):
        sorted_b = sorted(requested)
        return sorted_b[-2:]
    return list(requested)


def _calibration_path(
    cal_dir: Path,
    backbone: str,
    variant: str,
    family: str,
    regime: str = "projector_only",
) -> Path:
    """Mirror the naming convention from tools/isoflop_calibrate.py docs."""
    return cal_dir / f"{backbone}-{variant}-{family}-{regime}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scaling-study-dir",
        default="/flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/",
        help="Root of the IsoFLOP scaling study (defaults to the canonical /flare path)",
    )
    p.add_argument(
        "--calibration-dir",
        default=None,
        help="Directory holding calibration JSONs. Defaults to <scaling-study-dir>/calibration/",
    )
    p.add_argument("--family", required=True, choices=list(_FAMILY_TO_MODALITIES.keys()))
    p.add_argument(
        "--budgets",
        required=True,
        help="Comma-separated FLOP budgets (e.g. '3e17,1e18,3e18,1e19,3e19')",
    )
    p.add_argument(
        "--backbones",
        required=True,
        help=f"Comma-separated backbone labels from {{{','.join(_BACKBONE_TABLE.keys())}}}",
    )
    p.add_argument(
        "--projector-variants",
        default="BASE,W2X,W4X,D2X,D4X",
        help="Comma-separated variants from {BASE,W2X,W4X,D2X,D4X}",
    )
    p.add_argument("--seeds", type=int, default=1, help="Number of seed replicas per cell")
    p.add_argument("--global-batch-size", type=int, default=None,
                   help="Override global batch size used in tokens_total. Defaults to "
                        "cal['batch_size'] * nodes * 12 (Aurora ppn=12)")
    p.add_argument("--regime", default="projector_only",
                   choices=["projector_only", "encoder_projector", "e2e"])
    p.add_argument("--phase", default="M3",
                   help="Phase label written to each cell (M3, M4, M5a, …)")
    p.add_argument("--skip-backbone-check", action="store_true",
                   help="Skip the plan-time check that backbones are staged in "
                        "an HF cache. Use when staging will happen between "
                        "plan and launch.")
    # IsoFLOP cal/runtime FLOP rescale (PR-4 fix). The calibrator measures
    # per-rank FLOPs at its own (bs, sl) defaults; the real run is multi-rank
    # at typically different (bs, sl). Without rescaling, `max_steps =
    # target/cal_fps` over/under-sizes by ~50x. We rescale cal_fps to the
    # runtime config before dividing.
    p.add_argument("--runtime-batch-size", type=int, default=8,
                   help="Per-rank batch size at runtime (matches design's "
                        "training.batch_size). Default 8 matches PRISM-IMAGE-ONLY-*.")
    p.add_argument("--runtime-seq-len", type=int, default=2048,
                   help="Per-sample seq length at runtime. Default 2048 matches "
                        "the launcher --max-seq-length default.")
    p.add_argument("--runtime-ranks-per-node", type=int, default=12,
                   help="MPI ranks per Aurora node (12 XPU tiles by default).")
    p.add_argument("--no-runtime-rescale", action="store_true",
                   help="Skip the cal/runtime FLOPs rescale. Use only if you "
                        "explicitly want the raw calibration max_steps math.")
    p.add_argument("--nodes-override", type=int, default=None,
                   help="Override _BACKBONE_TABLE[*]['nodes'] for ALL backbones in "
                        "this plan (e.g. --nodes-override 4 to run every cell on 4 "
                        "nodes). The runtime-rescale math already auto-scales "
                        "max_steps via runtime_ranks_total = ranks_per_node * nodes, "
                        "so cells get fewer steps at higher node counts (same total "
                        "FLOPs, more parallel work per step).")
    p.add_argument("--output", required=True, help="YAML manifest path to write")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cal_dir = Path(args.calibration_dir) if args.calibration_dir else (
        Path(args.scaling_study_dir) / "calibration"
    )
    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    variants = [x.strip() for x in args.projector_variants.split(",") if x.strip()]
    unknown_variants = [v for v in variants if v not in ModalityProjector.VARIANT_MAP]
    if unknown_variants:
        print(
            f"ERROR: unknown variants {unknown_variants}. "
            f"Known: {list(ModalityProjector.VARIANT_MAP)}",
            file=sys.stderr,
        )
        return 2

    unknown = [b for b in backbones if b not in _BACKBONE_TABLE]
    if unknown:
        print(
            f"ERROR: unknown backbones {unknown}. Add them to _BACKBONE_TABLE.",
            file=sys.stderr,
        )
        return 2

    # Catch the unstaged-backbone gap at plan time so we don't discover it
    # mid-launch when the trainer's local_files_only=True call fails. Skip
    # the check when --skip-backbone-check is passed (useful for offline
    # plan generation where you'll stage the model before launch).
    if not args.skip_backbone_check:
        unstaged: list[str] = []
        for b in backbones:
            hf_id = _BACKBONE_TABLE[b]["hf_id"]
            if not _backbone_is_staged(hf_id):
                unstaged.append(f"{b} -> {hf_id}")
        if unstaged:
            print(
                "ERROR: backbones not staged in any HF cache "
                f"({list(_HF_CACHE_ROOTS)}):\n  "
                + "\n  ".join(unstaged)
                + "\nStage the model(s) or pass --skip-backbone-check to bypass.",
                file=sys.stderr,
            )
            return 2

    family_modalities = _FAMILY_TO_MODALITIES[args.family]

    cells: list[dict[str, Any]] = []
    skipped: list[str] = []
    # Track backbone_id mismatches separately: if EVERY would-be cell is
    # skipped because the calibration was measured against a different
    # backbone than the table claims, downstream tooling would silently
    # consume an empty plan and run zero cells. Surface that as a hard error.
    mismatches: list[str] = []
    for backbone in backbones:
        b_meta = dict(_BACKBONE_TABLE[backbone])
        if args.nodes_override is not None:
            if args.nodes_override < 1:
                print(
                    f"ERROR: --nodes-override must be >= 1 (got {args.nodes_override})",
                    file=sys.stderr,
                )
                return 2
            b_meta["nodes"] = int(args.nodes_override)
        backbone_budgets = _budgets_for_backbone(backbone, budgets)
        if not backbone_budgets:
            skipped.append(f"{backbone}: no budgets fit (requested={budgets})")
            continue
        for variant in variants:
            cal_path = _calibration_path(cal_dir, backbone, variant, args.family, args.regime)
            if not cal_path.exists():
                skipped.append(f"{backbone}-{variant}: calibration not found at {cal_path}")
                continue
            try:
                with open(cal_path) as f:
                    cal = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                skipped.append(f"{backbone}-{variant}: cannot read {cal_path}: {e}")
                continue
            fps = cal.get("flops_per_step")
            try:
                fps = float(fps)
            except (TypeError, ValueError):
                skipped.append(f"{backbone}-{variant}: flops_per_step not numeric")
                continue
            if fps <= 0:
                skipped.append(f"{backbone}-{variant}: flops_per_step <= 0")
                continue
            # Calibration JSON's backbone_id must match the table's hf_id —
            # otherwise the plan would launch against a different model than
            # the FLOPs were measured against, biasing max_steps. Warn-and-skip
            # rather than hard-fail: the user may have intentionally moved
            # backbones and the calibration is just stale.
            cal_backbone = cal.get("backbone_id")
            if cal_backbone and cal_backbone != b_meta["hf_id"]:
                msg = (
                    f"{backbone}-{variant}: calibration backbone_id={cal_backbone!r} "
                    f"!= table hf_id={b_meta['hf_id']!r}; re-calibrate or fix "
                    f"_BACKBONE_TABLE"
                )
                skipped.append(msg)
                mismatches.append(msg)
                # Surface eagerly to stderr — these are almost always a bug
                # the user wants to see now, not buried in manifest['skipped'].
                print(f"WARNING: {msg}", file=sys.stderr)
                continue
            # Rescale cal_fps from calibration config (per-rank, cal_bs/sl)
            # to runtime config (per-rank-equivalent across all ranks at
            # runtime bs/sl). The trainer's per-step FLOPs scale linearly in
            # batch_size, seq_len, and rank count (each rank does its own
            # forward+backward on its own micro-batch). Without this, a
            # cal at BS=4/SL=1024/1-rank vs runtime BS=8/SL=2048/12-ranks
            # under-estimates real per-step FLOPs by 48x.
            cal_bs = int(cal.get("batch_size", 8))
            cal_sl = int(cal.get("seq_len", 2048))
            cal_ranks = int(cal.get("n_ranks", 1))
            if args.no_runtime_rescale:
                runtime_fps = fps
                rescale_factor = 1.0
            else:
                runtime_ranks_total = (
                    int(args.runtime_ranks_per_node) * int(b_meta["nodes"])
                )
                rescale_factor = (
                    (args.runtime_batch_size / cal_bs)
                    * (args.runtime_seq_len / cal_sl)
                    * (runtime_ranks_total / cal_ranks)
                )
                runtime_fps = fps * rescale_factor
            for budget in backbone_budgets:
                for seed in range(args.seeds):
                    max_steps = max(1, round(budget / runtime_fps))
                    bs = int(args.runtime_batch_size)
                    # Default global batch = per-rank * ranks/node * nodes
                    gbs = args.global_batch_size or (
                        bs * int(args.runtime_ranks_per_node) * int(b_meta["nodes"])
                    )
                    seq_len = float(args.runtime_seq_len)
                    tokens_total = int(max_steps * gbs * seq_len)
                    run_id = (
                        f"ISO-{args.family}-{backbone}-{variant}-"
                        f"{budget:.0e}-s{seed}".replace("+", "")
                    )
                    cell = {
                        "run_id": run_id,
                        "phase": args.phase,
                        "family": args.family,
                        "modalities": list(family_modalities),
                        "regime": args.regime,
                        "backbone": backbone,
                        "backbone_hf_id": b_meta["hf_id"],
                        "projector_variant": variant,
                        # Source of truth = ModalityProjector.VARIANT_MAP, not
                        # the calibration JSON (which is just a witness). This
                        # protects against stale or hand-edited JSONs.
                        "projector_hidden_mult": ModalityProjector.VARIANT_MAP[variant][0],
                        "projector_num_layers": ModalityProjector.VARIANT_MAP[variant][1],
                        "budget_flops": float(budget),
                        "seed": int(seed),
                        "nodes": int(b_meta["nodes"]),
                        "batch_size": bs,
                        "max_steps": int(max_steps),
                        "max_seq_length": int(args.runtime_seq_len),
                        "tokens_total": tokens_total,
                        "calibration_json": str(cal_path),
                        "calibration_fps": float(fps),
                        "runtime_fps": float(runtime_fps),
                        "rescale_factor": float(rescale_factor),
                        "design": b_meta["design"],
                        "status": "planned",
                    }
                    cells.append(cell)

    # All-mismatch guard: if we'd emit zero cells AND at least one cell was
    # rejected for backbone_id mismatch, the user almost certainly has a
    # stale _BACKBONE_TABLE or stale calibrations. Don't write an empty
    # manifest that downstream `isoflop_launch.py` would silently no-op on.
    if not cells and mismatches:
        print(
            f"ERROR: every cell was skipped due to calibration backbone_id "
            f"mismatch ({len(mismatches)} variant(s)). Re-run calibration "
            f"or update _BACKBONE_TABLE in tools/isoflop_plan.py.",
            file=sys.stderr,
        )
        return 2

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "family": args.family,
        "regime": args.regime,
        "calibration_dir": str(cal_dir),
        "cells": cells,
    }
    if skipped:
        manifest["skipped"] = skipped

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"[isoflop_plan] wrote {len(cells)} cells to {out}")
    if skipped:
        print(f"[isoflop_plan] skipped {len(skipped)} entries:", file=sys.stderr)
        for s in skipped:
            print(f"  - {s}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

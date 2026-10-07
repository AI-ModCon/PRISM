#!/usr/bin/env python3
"""Launch debug Time-series Aurora jobs from an experiment YAML.

This script is designed for the repository's experiments/*.yaml files. If a YAML is
provided, it loads the experiment definitions and launches every matching design.
If a single --design is supplied, it launches just that design.

Supported patterns include:
  - PRISM-*-TIMEOMNI
  - PRISM-*-INTERN-S2
  - PRISM-*-INTERN-S2-397B

The intended workflow is:
  1) submit a 1-node hold job (see tools/hold_1n_debug_timeseries.sh)
  2) run this script against an experiments/*.yaml file
  3) let it iterate through all matching time-series designs in that YAML
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

DEFAULT_WEBDATASET_DIR = "/flare/ModCon/pemami/data/SciTS-processed"
DEFAULT_WANDB_PROJECT = "goose"
DEFAULT_WANDB_ENTITY = "pemami"
TIME_SERIES_DESIGN_RE = re.compile(
    r"PRISM-.*(?:TIMEOMNI|INTERN-S2(?:-397B)?)",
    re.IGNORECASE,
)


def _exp_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


def _iter_experiment_designs(yaml_path: Path) -> list[str]:
    data = _load_yaml(yaml_path)
    experiments = data.get("experiments") or []
    if not isinstance(experiments, list):
        raise ValueError(f"Expected 'experiments' list in {yaml_path}")

    designs: list[str] = []
    for item in experiments:
        if not isinstance(item, dict):
            continue
        design_id = str(item.get("id") or "").strip()
        if not design_id:
            continue
        if TIME_SERIES_DESIGN_RE.fullmatch(design_id):
            designs.append(design_id)
    return designs


def _coerce_designs(args: argparse.Namespace) -> list[str]:
    if args.design:
        return [args.design]
    if not args.yaml:
        raise SystemExit("Either --design or --yaml must be provided.")
    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        raise SystemExit(f"Experiment YAML not found: {yaml_path}")
    return _iter_experiment_designs(yaml_path)


def _build_command(args: argparse.Namespace, design_id: str) -> list[str]:
    cmd: list[str] = [
        "python3",
        "tools/launch_aurora_web.py",
        "--id",
        args.run_prefix + design_id if args.run_prefix else f"{design_id}-DEBUG",
        "--design",
        design_id,
        "--nodes",
        str(args.nodes),
        "--dist-strategy",
        args.dist_strategy,
        "--fsdp-sharding",
        args.fsdp_sharding,
        "--webdataset-dir",
        args.webdataset_dir,
        "--webdataset-modality",
        "time_series",
        "--wandb-project",
        args.wandb_project,
        f"wandb.entity={args.wandb_entity}",
        "+training.data_num_workers=1",
        f"++training.max_steps={args.max_steps}",
        f"++training.eval_every_n_steps={args.eval_every_n_steps}",
        f"++training.save_every_n_steps={args.save_every_n_steps}",
        f"++training.viz_every_n_steps={args.viz_every_n_steps}",
        "++training.eval_enabled=true",
    ]

    if args.head_node:
        cmd.extend(["--hosts", args.head_node, "--run-via-ssh"])
    if getattr(args, "resume_from_checkpoint", None):
        cmd.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])
    if getattr(args, "resume_weights_only", None):
        cmd.extend(["--resume-weights-only", args.resume_weights_only])
    if args.use_shared_venv:
        cmd.append("--use-shared-venv")
    if args.packed_env:
        cmd.extend(["--packed-env", args.packed_env])
    if args.debug:
        cmd.append("++training.verbosity=DEBUG")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _render_shell_command(cmd: list[str]) -> str:
    inner = " ".join(shlex.quote(part) for part in cmd)
    return "module load frameworks && source .venv-deepspeed/bin/activate && " + inner


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--design",
        default=None,
        help="Single design to run, e.g. PRISM-QWEN3-0-6B-TIMEOMNI. If omitted, --yaml is required.",
    )
    parser.add_argument(
        "--yaml",
        default=None,
        help="Experiment YAML under experiments/ to scan. All matching time-series designs in it will run.",
    )
    parser.add_argument(
        "--run-prefix",
        default="DEBUG-",
        help="Prefix used when constructing run IDs for launched jobs.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes per design.",
    )
    parser.add_argument(
        "--head-node",
        default=None,
        help="Head node hostname for --hosts/--run-via-ssh on the reserved node.",
    )
    parser.add_argument(
        "--run-via-ssh",
        action="store_true",
        help="Pass --run-via-ssh to launch_aurora_web.py.",
    )
    parser.add_argument(
        "--webdataset-dir",
        default=DEFAULT_WEBDATASET_DIR,
        help="WebDataset directory containing SciTS data.",
    )
    parser.add_argument(
        "--dist-strategy",
        choices=["ddp", "fsdp", "hsdp"],
        default="hsdp",
        help="Aurora distributed strategy.",
    )
    parser.add_argument(
        "--fsdp-sharding",
        choices=["full_shard", "shard_grad_op", "hybrid_shard", "no_shard"],
        default="shard_grad_op",
        help="FSDP sharding strategy.",
    )
    parser.add_argument(
        "--packed-env",
        default="deepspeed_env.tar.gz",
        help="Packed env tarball passed to the Aurora launcher.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=DEFAULT_WANDB_ENTITY,
        help="Weights & Biases entity name.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Short debug run length.",
    )
    parser.add_argument(
        "--eval-every-n-steps",
        type=int,
        default=250,
        help="Eval cadence for the debug run.",
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=500,
        help="Checkpoint cadence for the debug run.",
    )
    parser.add_argument(
        "--viz-every-n-steps",
        type=int,
        default=250,
        help="Visualization cadence for the debug run.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to checkpoint directory for full resume: model, optimizer, scheduler.",
    )
    parser.add_argument(
        "--resume-weights-only",
        default=None,
        help="Path to checkpoint directory to load model weights from with a fresh optimizer.",
    )
    parser.add_argument(
        "--use-shared-venv",
        action="store_true",
        help="Use the shared local venv instead of unpacking the tarball.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add explicit debug verbosity to the Hydra config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact commands without launching them.",
    )
    args = parser.parse_args(argv)

    if args.run_via_ssh and not args.head_node:
        parser.error("--run-via-ssh requires --head-node")

    if args.design is None and args.yaml is None:
        parser.error("Either --design or --yaml must be provided.")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    design_ids = _coerce_designs(args)

    if not design_ids:
        print(f"No matching time-series designs found in {args.yaml or args.design}", file=sys.stderr)
        return 1

    for design_id in design_ids:
        cmd = _build_command(args, design_id)
        rendered = _render_shell_command(cmd)
        print(f"# design={design_id}")
        print(rendered)
        if args.dry_run:
            continue

        env = os.environ.copy()
        try:
            subprocess.run(rendered, shell=True, executable="/bin/bash", env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Failed: {design_id} (exit={exc.returncode})", file=sys.stderr)
            return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

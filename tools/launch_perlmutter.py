#!/usr/bin/env python3
"""Generate or execute PRISM Perlmutter launch scripts in a CI-safe way."""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
from pathlib import Path

import yaml


def _find_design(data: dict, design_id: str) -> tuple[dict, dict]:
    for exp in data.get("experiments", []):
        if exp.get("id") == design_id:
            common = exp.get("common_overrides", {}).copy()
            common.update(exp.get("overrides", {}))
            return exp, common
        for variant in exp.get("variants", []):
            if variant.get("id") == design_id:
                common = exp.get("common_overrides", {}).copy()
                common.update(exp.get("overrides", {}))
                common.update(variant.get("overrides", {}))
                return variant, common
    raise ValueError(f"Experiment ID '{design_id}' not found")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch PRISM experiments on Perlmutter")
    parser.add_argument("--file", default="experiments/prism_designs.yaml")
    parser.add_argument("--id", required=True, help="Run ID / Job name")
    parser.add_argument("--design", default=None, help="Design ID in yaml; defaults to --id")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--partition", default="debug")
    parser.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT", ""))
    parser.add_argument("--time", default="00:30:00")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args, unknown_args = parser.parse_known_args()

    design_id = args.design if args.design else args.id

    with open(args.file) as f:
        data = yaml.safe_load(f)

    exp_cfg, overrides = _find_design(data, design_id)
    resources = exp_cfg.get("resources", {})

    for item in unknown_args:
        if "=" in item and not item.startswith("--"):
            key, value = item.split("=", 1)
            overrides[key] = value

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    logs_dir = Path("logs") / design_id
    jobs_dir = Path("jobs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    ngpus = int(resources.get("ngpus", overrides.get("resources.ngpus", 4)))
    overrides_list = [f"{k}={v}" for k, v in overrides.items()]
    overrides_str = " \\\n    ".join(overrides_list)

    script_path = jobs_dir / f"run_perlmutter_{args.id}_{date_str}_{time_str}.sh"
    script = f"""#!/bin/bash
#SBATCH -J {args.id}
#SBATCH -N {args.nodes}
#SBATCH -q {args.partition}
#SBATCH -t {args.time}
#SBATCH --output={logs_dir}/%x-%j.out

set -euo pipefail

module load pytorch/2.8.0

export MASTER_ADDR=${{MASTER_ADDR:-$(hostname)}}
export MASTER_PORT=${{MASTER_PORT:-29500}}

srun -l -u --gpus-per-node={ngpus} \\
    python -m accelerate.commands.launch \\
    --multi_gpu \\
    --num_machines={args.nodes} \\
    --num_processes={max(1, args.nodes * ngpus)} \\
    src/train.py \\
    {overrides_str}
"""

    script_path.write_text(script)
    os.chmod(script_path, 0o755)

    print(f"Generated Run Script: {script_path}")

    if args.submit and not args.dry_run:
        cmd = ["sbatch"]
        if args.account:
            cmd.extend(["-A", args.account])
        cmd.append(str(script_path))
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            return result.returncode
        print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

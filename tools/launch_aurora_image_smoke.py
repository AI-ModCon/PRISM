#!/usr/bin/env python3
"""Generate/review a one-node, one-XPU OmniGen2 smoke job before submission."""

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


def render_job(args):
    for key in ("project", "queue", "name"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", getattr(args, key)):
            raise ValueError(f"Invalid PBS {key}")
    if args.minutes < 1 or args.minutes > 15 or args.steps < 1 or args.steps > 4:
        raise ValueError("Smoke budget must be 1–15 minutes and 1–4 denoising steps")
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in ("repo", "venv", "checkpoint", "upstream", "cases", "output_dir", "job_dir")
    }

    def q(key):
        return shlex.quote(str(paths[key]))

    worker = paths["job_dir"] / "worker.sh"
    command = [
        str(paths["venv"] / "bin/python"),
        str(paths["repo"] / "tools/validate_image_decoder.py"),
        "--reference",
        "--smoke",
        "--checkpoint",
        str(paths["checkpoint"]),
        "--upstream",
        str(paths["upstream"]),
        "--cases",
        str(paths["cases"]),
        "--output-dir",
        str(paths["output_dir"]),
        "--device",
        "xpu",
        "--dtype",
        "bfloat16",
        "--steps",
        str(args.steps),
    ]
    commands = [shlex.join(command)]
    repeatability_reference = getattr(args, "repeatability_reference", None)
    if repeatability_reference:
        if getattr(args, "with_parity", False):
            raise ValueError("Choose parity or repeatability, not both")
        command = [
            str(paths["venv"] / "bin/python"),
            str(paths["repo"] / "tools/diagnose_image_decoder_repeatability.py"),
            "--checkpoint", str(paths["checkpoint"]),
            "--upstream", str(paths["upstream"]),
            "--cases", str(paths["cases"]),
            "--reference-dir", str(Path(repeatability_reference).resolve()),
            "--output-dir", str(paths["output_dir"]),
            "--device", "xpu", "--dtype", "bfloat16", "--steps", str(args.steps),
        ]
        fresh = list(command)
        fresh[fresh.index("--output-dir") + 1] = str(paths["output_dir"]) + "-fresh"
        fresh += ["--prior-run", str(paths["output_dir"])]
        commands = [shlex.join(command), shlex.join(fresh)]
    if getattr(args, "with_parity", False):
        parity_command = list(command)
        parity_command[parity_command.index("--reference")] = "--parity"
        parity_command[parity_command.index("--output-dir") + 1] = (
            str(paths["output_dir"]) + "-parity"
        )
        parity_command += [
            "--parity-mode",
            "full_pipeline",
            "--reference-dir",
            str(paths["output_dir"]),
        ]
        commands.append(shlex.join(parity_command))
    probe = shlex.join([
        str(paths["venv"] / "bin/python"), "-u", "-c",
        "import json, os, torch; "
        "count = torch.xpu.device_count(); "
        "print(json.dumps({'xpu_device_count': count, 'torch': str(torch.__version__), "
        "'device_selection': {key: os.environ.get(key) for key in "
        "('ZE_FLAT_DEVICE_HIERARCHY', 'ZE_AFFINITY_MASK', 'ONEAPI_DEVICE_SELECTOR')}}), flush=True); "
        "assert count == 1, 'Smoke requires exactly one visible XPU tile'",
    ])
    # Framework module precedes activation; single-rank placement needs no
    # distributed rendezvous or shared nodefile from a previous allocation.
    worker_text = f"""#!/bin/bash
set -eo pipefail
module use /soft/modulefiles
module load frameworks/2025.3.1
set -u
source {q("venv")}/bin/activate
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Aurora's FLAT hierarchy indexes tiles directly, as in the production launcher.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
unset SYCL_DEVICE_FILTER
export PYTHONPATH={q("repo")}:{q("upstream")}:${{PYTHONPATH:-}}
cd {q("repo")}
{probe}
{chr(10).join(commands)}
"""
    pbs = f"""#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:{args.minutes:02d}:00
#PBS -l filesystems=home:flare
#PBS -q {args.queue}
#PBS -A {args.project}
#PBS -N {args.name}
#PBS -j oe
#PBS -k doe
set -eo pipefail
: "${{PBS_JOBID:?Run this script via qsub}}"
module use /soft/modulefiles
module load frameworks/2025.3.1
set -u
cd {q("repo")}
mpiexec -n 1 --ppn 1 bash {shlex.quote(str(worker))}
"""
    return pbs, worker_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "venv", "checkpoint", "upstream", "cases", "output-dir", "job-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--project", default="ModCon")
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--name", default="prism-image-smoke")
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2)
    followup = parser.add_mutually_exclusive_group()
    followup.add_argument(
        "--with-parity",
        action="store_true",
        help="Run a second smoke through the PRISM reference adapter within the same walltime",
    )
    followup.add_argument(
        "--repeatability-reference", type=Path,
        help="Diagnose native/adapter repeatability in two fresh processes using this reference run",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    pbs, worker = render_job(args)
    # Dry-run needs no remote assets; it permits review on a workstation.
    if args.dry_run:
        print(
            json.dumps(
                {"pbs": pbs, "worker": worker, "nodes": 1, "processes": 1, "xpu_tiles": 1}, indent=2
            )
        )
        return 0
    for name in ("repo", "venv", "checkpoint", "upstream", "cases"):
        if not getattr(args, name).exists():
            raise ValueError(f"Missing staged {name}: {getattr(args, name)}")
    if args.output_dir.exists():
        raise ValueError("Output directory must be new")
    if args.with_parity and Path(str(args.output_dir) + "-parity").exists():
        raise ValueError("Parity output directory must be new")
    if args.repeatability_reference:
        if not args.repeatability_reference.is_dir():
            raise ValueError("Repeatability reference must be a completed reference directory")
        if Path(str(args.output_dir) + "-fresh").exists():
            raise ValueError("Fresh-process diagnostic output directory must be new")
    args.job_dir.mkdir(parents=True, exist_ok=False)
    # Metadata/file-integrity checks belong on the login node before spending
    # an allocation. This path never imports models or allocates an accelerator.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(args.repo.resolve()), str(args.upstream.resolve()), environment.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [
            str(args.venv.resolve() / "bin/python"),
            str(args.repo.resolve() / "tools/validate_image_decoder.py"),
            "--preflight",
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--upstream",
            str(args.upstream.resolve()),
            "--output-dir",
            str(args.job_dir.resolve() / "preflight"),
        ],
        check=True,
        env=environment,
    )
    for name, body in (("job.pbs", pbs), ("worker.sh", worker)):
        path = args.job_dir / name
        path.write_text(body)
        subprocess.run(["bash", "-n", str(path)], check=True)
    result = subprocess.run(
        ["qsub", str((args.job_dir / "job.pbs").resolve())],
        check=True,
        text=True,
        capture_output=True,
    )
    (args.job_dir / "submission.json").write_text(
        json.dumps(
            {
                "job_id": result.stdout.strip(),
                "nodes": 1,
                "processes": 1,
                "xpu_tiles": 1,
                "minutes": args.minutes,
                "steps": args.steps,
                "with_parity": args.with_parity,
                "repeatability_reference": str(args.repeatability_reference.resolve()) if args.repeatability_reference else None,
            },
            indent=2,
        )
        + "\n"
    )
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

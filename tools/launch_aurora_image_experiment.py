#!/usr/bin/env python3
"""Review and launch one-node image diagnostics or bounded training pilots.

This launcher executes an explicit JSON argv through a single allocated XPU tile.
It accepts only explicit project diagnostic/pilot entry points and never upgrades
any result to P0/P1/P2 acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

ENTRYPOINTS = {
    "diagnose_image_decoder_repeatability.py",
    "validate_prism_image_parent.py",
    "overfit_prism_image_connector.py",
    "train_prism_image_connector.py",
    "train_prism_image_diffusion.py",
    "diagnose_prism_image_conditioning.py",
    "diagnose_prism_joint_components.py",
    "align_prism_image_conditioning.py",
}


def require_repeatability(path, require_prior=False):
    """Fail closed before training; this bounded check is not P0/P1 acceptance."""
    report = json.loads(Path(path).read_text())
    if (
        report.get("status") != "completed"
        or report.get("evidence_kind") != "real_checkpoint_diagnostic"
        or report.get("fixture") is not False
    ):
        raise ValueError("A completed real-checkpoint numerical diagnostic is required")
    checks = ["native_repeat", "adapter_vs_native", "native_after_adapter"]
    if require_prior:
        checks.append("prior_process")
    for name in checks:
        if report.get("comparisons", {}).get(name, {}).get("passed") is not True:
            raise ValueError(f"Numerical pre-check failed: {name}; training was not started")
    print("Bounded numerical pre-check passed; P0/P1 qualification remains separate", flush=True)


def _option(argv, flag):
    if argv.count(flag) != 1:
        raise ValueError(f"Exactly one explicit {flag} is required")
    index = argv.index(flag)
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ValueError(f"Missing value for {flag}")
    return argv[index + 1]


def _read_command(path, *, pre_command=False):
    argv = json.loads(Path(path).read_text())
    if (
        not isinstance(argv, list)
        or len(argv) < 3
        or not all(isinstance(value, str) and value for value in argv)
    ):
        raise ValueError("Command file must be a nonempty JSON argv array")
    if argv[0] not in ENTRYPOINTS:
        raise ValueError("Only bounded image experiment entry points are allowed")
    if pre_command and argv[0] != "diagnose_image_decoder_repeatability.py":
        raise ValueError("Pre-command must use diagnose_image_decoder_repeatability.py")
    output = Path(_option(argv, "--output-dir"))
    if _option(argv, "--device") != "xpu":
        raise ValueError("Exactly one XPU tile is required")
    if not output.is_absolute():
        raise ValueError("Output directory must be absolute")
    return argv, output


def _read_commands(args):
    command, output = _read_command(args.command_file)
    pre_path = getattr(args, "pre_command_file", None)
    pre_command, pre_output = (
        _read_command(pre_path, pre_command=True) if pre_path else (None, None)
    )
    if pre_output is not None and pre_output.resolve() == output.resolve():
        raise ValueError("Pre-command and main command need distinct output directories")
    return command, output, pre_command, pre_output


def render_job(args, *, commands=None):
    for key in ("project", "queue", "name"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", getattr(args, key)):
            raise ValueError(f"Invalid PBS {key}")
    if args.queue not in ("debug", "capacity") or not 5 <= args.minutes <= 1440:
        raise ValueError("Experiment permits debug/capacity and 5–1440 minutes")
    paths = {
        key: Path(getattr(args, key)).resolve() for key in ("repo", "venv", "upstream", "job_dir")
    }
    argv, output, pre_argv, pre_output = commands or _read_commands(args)
    if args.minutes > 60 and (
        args.queue != "capacity" or argv[0] != "train_prism_image_diffusion.py"
    ):
        raise ValueError("Runs over 60 minutes require dense diffusion training in capacity")
    command = [
        str(paths["venv"] / "bin/python"),
        "-u",
        str(paths["repo"] / "tools" / argv[0]),
        *argv[1:],
    ]
    pre_command = (
        [
            str(paths["venv"] / "bin/python"),
            "-u",
            str(paths["repo"] / "tools" / pre_argv[0]),
            *pre_argv[1:],
        ]
        if pre_argv is not None
        else None
    )
    command_rows = []
    if pre_command is not None:
        command_rows.append(pre_command)
        command_rows.append(
            [
                str(paths["venv"] / "bin/python"),
                "-u",
                "-c",
                "import sys; from tools.launch_aurora_image_experiment import require_repeatability; "
                "require_repeatability(sys.argv[1], sys.argv[2] == 'yes')",
                str(pre_output / "manifest.json"),
                "yes" if "--prior-run" in pre_argv else "no",
            ]
        )
    command_rows.append(command)
    command_lines = "\n".join(shlex.join(row) for row in command_rows)
    # Dense diffusion pilots offload FP32 AdamW work to the allocated host.
    # Explicit thread counts avoid single-threaded multi-billion-parameter updates.
    cpu_threads = (
        "export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32\n"
        if argv[0] == "train_prism_image_diffusion.py"
        else ""
    )
    cpu_binding = (
        " --cpu-bind depth --depth 32" if argv[0] == "train_prism_image_diffusion.py" else ""
    )

    def q(name):
        return shlex.quote(str(paths[name]))

    probe = shlex.join(
        [
            str(paths["venv"] / "bin/python"),
            "-u",
            "-c",
            "import torch; n=torch.xpu.device_count(); print({'xpu_tiles':n,'torch':str(torch.__version__)},flush=True); assert n==1",
        ]
    )
    worker = f"""#!/bin/bash
set -eo pipefail
module use /soft/modulefiles
module load frameworks/2025.3.1
set -u
source {q("venv")}/bin/activate
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
unset SYCL_DEVICE_FILTER
export PYTHONPATH={q("repo")}:{q("upstream")}:${{PYTHONPATH:-}}
{cpu_threads.rstrip()}
cd {q("repo")}
{probe}
{command_lines}
"""
    pbs = f"""#!/bin/bash
#PBS -l select=1
#PBS -l walltime={args.minutes // 60:02d}:{args.minutes % 60:02d}:00
#PBS -l filesystems=home:flare
#PBS -q {args.queue}
#PBS -A {args.project}
#PBS -N {args.name}
#PBS -j oe
#PBS -k doe
set -eo pipefail
: "${{PBS_JOBID:?Run via qsub}}"
module use /soft/modulefiles
module load frameworks/2025.3.1
set -u
cd {q("repo")}
mpiexec -n 1 --ppn 1{cpu_binding} bash {shlex.quote(str(paths["job_dir"] / "worker.sh"))}
"""
    return pbs, worker, argv, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "venv", "upstream", "job-dir", "command-file"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--pre-command-file",
        type=Path,
        help="Repeatability diagnostic runs first; failed execution or exact comparisons stop training",
    )
    parser.add_argument("--project", default="ModCon")
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--name", default="prism-image-experiment")
    parser.add_argument("--minutes", type=int, default=15)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    commands = _read_commands(args)
    pbs, worker, command, output = render_job(args, commands=commands)
    _, _, pre_command, pre_output = commands
    if args.dry_run:
        print(
            json.dumps(
                {
                    "pbs": pbs,
                    "worker": worker,
                    "nodes": 1,
                    "processes": 1,
                    "xpu_tiles": 1,
                    "pre_command": pre_command,
                },
                indent=2,
            )
        )
        return 0
    for name in ("repo", "venv", "upstream"):
        getattr(args, name).resolve(strict=True)
    script = args.repo / "tools" / command[0]
    script.resolve(strict=True)
    pre_script = args.repo / "tools" / pre_command[0] if pre_command else None
    if pre_script is not None:
        pre_script.resolve(strict=True)
    for run_output in (output, pre_output):
        if run_output is not None and run_output.exists():
            raise ValueError(
                "Both output directories must be new; use --resume with a new output directory"
            )
    # Resolve local data/checkpoint arguments without importing torch on the login node.
    path_flags = (
        "--checkpoint",
        "--upstream",
        "--cases",
        "--reference-dir",
        "--prior-run",
        "--model-config",
        "--tokenizer",
        "--source-processor",
        "--manifest",
        "--train-manifest",
        "--validation-manifest",
        "--train-index",
        "--validation-index",
        "--repeatability-report",
        "--resume",
        "--connector-checkpoint",
        "--init-joint-checkpoint",
        "--init-alignment-checkpoint",
        "--joint-checkpoint",
        "--alignment-checkpoint",
        "--transformer-checkpoint-dir",
    )
    for run_command in (command, pre_command):
        if run_command is not None:
            for flag in path_flags:
                if flag in run_command:
                    argument_path = Path(_option(run_command, flag))
                    # The guarded pre-command creates this report in the same
                    # allocation. Every other input must already exist.
                    if (
                        run_command is command
                        and flag == "--repeatability-report"
                        and pre_output is not None
                        and argument_path.resolve() == (pre_output / "manifest.json").resolve()
                    ):
                        continue
                    argument_path.resolve(strict=True)
    args.job_dir.mkdir(parents=True, exist_ok=False)
    for name, body in (("job.pbs", pbs), ("worker.sh", worker)):
        path = args.job_dir / name
        path.write_text(body)
        subprocess.run(["bash", "-n", str(path)], check=True)
    record = {
        "command": command,
        "output_dir": str(output),
        "command_file_sha256": hashlib.sha256(args.command_file.read_bytes()).hexdigest(),
        "entrypoint_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "minutes": args.minutes,
        "queue": args.queue,
        "nodes": 1,
        "processes": 1,
        "xpu_tiles": 1,
    }
    if pre_command is not None:
        record.update(
            pre_command=pre_command,
            pre_output_dir=str(pre_output),
            pre_command_file_sha256=hashlib.sha256(args.pre_command_file.read_bytes()).hexdigest(),
            pre_entrypoint_sha256=hashlib.sha256(pre_script.read_bytes()).hexdigest(),
        )
    (args.job_dir / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    result = subprocess.run(
        ["qsub", str((args.job_dir / "job.pbs").resolve())],
        text=True,
        capture_output=True,
        check=True,
    )
    record["job_id"] = result.stdout.strip()
    (args.job_dir / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

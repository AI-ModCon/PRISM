#!/usr/bin/env python3
"""Review and submit bounded trained-PRISM parent or generator smoke jobs."""

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


def render_job(args):
    for name in ("project", "queue", "name"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", getattr(args, name)):
            raise ValueError(f"Invalid PBS {name}")
    if not 1 <= args.minutes <= 15 or not 1 <= args.steps <= 4:
        raise ValueError("Smoke budget requires 1–15 minutes and 1–4 steps")
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "repo",
            "venv",
            "upstream",
            "model_config",
            "checkpoint",
            "tokenizer",
            "source_processor",
            "cases",
            "output_dir",
            "job_dir",
        )
    }
    executable = (
        "validate_prism_image_parent.py"
        if args.mode == "parent"
        else "smoke_prism_image_training.py"
    )
    command = [str(paths["venv"] / "bin/python"), "-u", str(paths["repo"] / "tools" / executable)]
    for name in ("model_config", "checkpoint", "tokenizer", "source_processor", "output_dir"):
        command += ["--" + name.replace("_", "-"), str(paths[name])]
    command += [
        "--cases" if args.mode == "parent" else "--manifest",
        str(paths["cases"]),
        "--device",
        "xpu",
        "--dtype",
        "bfloat16",
    ]
    command += (
        ["--max-new-tokens", "16"]
        if args.mode == "parent"
        else [
            "--steps",
            str(args.steps),
            "--sampling-steps",
            "2",
            "--height",
            "256",
            "--width",
            "256",
        ]
    )

    def q(name):
        return shlex.quote(str(paths[name]))

    probe = shlex.join(
        [
            str(paths["venv"] / "bin/python"),
            "-u",
            "-c",
            "import torch; n=torch.xpu.device_count(); print({'xpu_tiles':n,'torch':str(torch.__version__)},flush=True); assert n==1, 'Exactly one XPU tile required'",
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
cd {q("repo")}
{probe}
{shlex.join(command)}
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
: "${{PBS_JOBID:?Run via qsub}}"
module use /soft/modulefiles
module load frameworks/2025.3.1
set -u
cd {q("repo")}
mpiexec -n 1 --ppn 1 bash {shlex.quote(str(paths["job_dir"] / "worker.sh"))}
"""
    return pbs, worker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo",
        "venv",
        "upstream",
        "model-config",
        "checkpoint",
        "tokenizer",
        "source-processor",
        "cases",
        "output-dir",
        "job-dir",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--mode", choices=("parent", "connector-smoke"), required=True)
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--project", default="ModCon")
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--name", default="prism-parent-smoke")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    pbs, worker = render_job(args)
    if args.dry_run:
        print(
            json.dumps(
                {"pbs": pbs, "worker": worker, "nodes": 1, "processes": 1, "xpu_tiles": 1}, indent=2
            )
        )
        return 0
    for name in (
        "repo",
        "venv",
        "upstream",
        "model_config",
        "checkpoint",
        "tokenizer",
        "source_processor",
        "cases",
    ):
        getattr(args, name).resolve(strict=True)
    if args.output_dir.exists():
        raise ValueError("Output directory must be new")
    # Metadata-only preflight: actual model loading belongs on allocated nodes.
    config = json.loads(args.model_config.read_text())
    for field in ("llm_backbone_id", "image_encoder_id"):
        if not Path(config[field]).is_dir():
            raise ValueError(f"Stage an explicit local {field} before submission")
    if args.mode == "connector-smoke":
        image_config = config["decoder_configs"]["image"]
        generator_config = image_config.get("generator") or image_config
        generator = Path(generator_config["model_id"])
        if not (generator / "prism_checkpoint_provenance.json").is_file():
            raise ValueError("Staged OmniGen2 checkpoint needs its verified file inventory")
    args.job_dir.mkdir(parents=True, exist_ok=False)
    for name, body in (("job.pbs", pbs), ("worker.sh", worker)):
        path = args.job_dir / name
        path.write_text(body)
        subprocess.run(["bash", "-n", str(path)], check=True)
    result = subprocess.run(
        ["qsub", str((args.job_dir / "job.pbs").resolve())],
        text=True,
        capture_output=True,
        check=True,
    )
    (args.job_dir / "submission.json").write_text(
        json.dumps(
            {
                "job_id": result.stdout.strip(),
                "mode": args.mode,
                "minutes": args.minutes,
                "nodes": 1,
                "processes": 1,
                "xpu_tiles": 1,
            },
            indent=2,
        )
        + "\n"
    )
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

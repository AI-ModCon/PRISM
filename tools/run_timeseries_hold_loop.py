#!/usr/bin/env python3
"""Automated hold runner for Aurora PRISM Time-Series training and evaluation.

Coordinates:
  1. Submitting 1-node PBS hold jobs (tools/hold_1n_debug_timeseries.sh).
  2. Monitoring PBS queue until the hold job starts running (state R).
  3. Launching training against the allocated node via SSH using tools/timeseries_launch.py.
  4. Monitoring walltime: when within safety margin (e.g. 10 minutes of the 1-hour walltime),
     preemptively stopping the training run, submitting a new hold job, and resuming training
     from the latest checkpoint.
  5. Once training completes (target max_steps reached), automatically running the PRISM
     Universal Evaluator (tools/universal_evaluator.py) on all SciTS validation shards.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import yaml

DEFAULT_HOLD_SCRIPT = "tools/hold_1n_debug_timeseries.sh"
DEFAULT_WEBDATASET_DIR = "/flare/ModCon/pemami/data/SciTS-processed"
DEFAULT_VAL_SHARDS_DIR = "/flare/ModCon/pemami/data/SciTS-processed/val_shards"
DEFAULT_WANDB_PROJECT = "goose"
DEFAULT_WANDB_ENTITY = "pemami"
DEFAULT_SAFETY_MARGIN_MINUTES = 10
DEFAULT_WALLTIME_SECONDS = 3600  # 1 hour
DEFAULT_POLL_INTERVAL = 15.0

TIME_SERIES_DESIGN_RE = re.compile(
    r"PRISM-.*(?:TIMEOMNI|INTERN-S2(?:-397B)?)",
    re.IGNORECASE,
)

KNOWN_BACKBONES = {
    "PRISM-QWEN3-0-6B-INTERN-S2-397B": "Qwen/Qwen3-0.6B",
    "PRISM-QWEN3-0-6B-TIMEOMNI": "Qwen/Qwen3-0.6B",
    "PRISM-QWEN3-0-6B-INTERN-S2": "Qwen/Qwen3-0.6B",
    "PRISM-OLMO-1B-TIMEOMNI": "allenai/OLMo-1B-0724-hf",
    "PRISM-OLMO-1B-INTERN-S2-397B": "allenai/OLMo-1B-0724-hf",
    "PRISM-OLMO-1B-INTERN-S2": "allenai/OLMo-1B-0724-hf",
    "PRISM-OLMO3-7B-INSTRUCT-TSQA": "allenai/Olmo-3-7B-Instruct",
}


@dataclasses.dataclass
class JobStatus:
    job_id: str
    state: str  # 'Q', 'R', 'H', 'E', 'F', 'UNKNOWN'
    exec_host: str | None = None
    elapsed_walltime: int = 0
    total_walltime: int = 3600
    remaining_walltime: int = 3600


def parse_job_id(output: str) -> str:
    """Extract job ID string from qsub output."""
    raw = output.strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""
    # Usually the first line or only line is the job ID
    return lines[0].split()[0]


def parse_time_str(time_s: str | None) -> int:
    """Parse [[HH:]MM:]SS time format to total seconds."""
    if not time_s:
        return 0
    parts = time_s.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return int(parts[0])
    except ValueError:
        return 0
    return 0


def parse_qstat_output(raw_output: str) -> JobStatus:
    """Parse `qstat -f <job_id>` output into structured JobStatus."""
    if not raw_output or not raw_output.strip():
        return JobStatus(job_id="", state="F", elapsed_walltime=0, total_walltime=0, remaining_walltime=0)

    job_id = ""
    state = "UNKNOWN"
    exec_host = None
    elapsed = 0
    total = DEFAULT_WALLTIME_SECONDS

    # Merge continuation lines (lines without '=')
    lines = [raw_line.rstrip() for raw_line in raw_output.splitlines() if raw_line.strip()]
    merged_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("job id:") or "=" in stripped:
            merged_lines.append(stripped)
        elif merged_lines:
            merged_lines[-1] += " " + stripped

    for line in merged_lines:
        if line.lower().startswith("job id:"):
            job_id = line.split(":", 1)[1].strip()
        elif "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key == "job_state":
                state = val
            elif key == "exec_host":
                # Formats: "x4212c0s0b0n0/0*208" or "x4212c0s0b0n0"
                first_node = val.split("+")[0].split("/")[0].strip()
                if first_node:
                    exec_host = first_node
            elif key == "resources_used.walltime":
                elapsed = parse_time_str(val)
            elif key == "Resource_List.walltime":
                total = parse_time_str(val) or DEFAULT_WALLTIME_SECONDS

    remaining = max(0, total - elapsed)
    return JobStatus(
        job_id=job_id,
        state=state,
        exec_host=exec_host,
        elapsed_walltime=elapsed,
        total_walltime=total,
        remaining_walltime=remaining,
    )


def submit_hold_job(hold_script: str, extra_args: list[str] | None = None) -> str:
    """Submit the hold job script via qsub."""
    cmd = ["qsub"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(hold_script)

    print(f"[PBS] Submitting hold script: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = parse_job_id(proc.stdout)
    print(f"[PBS] Job submitted successfully: {job_id}")
    return job_id


def query_job_status(job_id: str) -> JobStatus:
    """Query job status from PBS via qstat -f."""
    if not job_id:
        return JobStatus(job_id="", state="UNKNOWN")
    try:
        proc = subprocess.run(
            ["qstat", "-f", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            # Job is no longer in active queue
            return JobStatus(job_id=job_id, state="F", elapsed_walltime=0, total_walltime=0, remaining_walltime=0)
        status = parse_qstat_output(proc.stdout)
        if not status.job_id:
            status.job_id = job_id
        return status
    except Exception as e:
        print(f"[PBS] Warning: failed to query qstat for {job_id}: {e}", file=sys.stderr)
        return JobStatus(job_id=job_id, state="UNKNOWN")


def delete_job(job_id: str) -> None:
    """Delete / release a PBS job via qdel."""
    if not job_id:
        return
    print(f"[PBS] Releasing/Deleting job {job_id}...")
    try:
        subprocess.run(["qdel", job_id], capture_output=True, text=True, check=False)
    except Exception as e:
        print(f"[PBS] Warning: qdel {job_id} failed: {e}", file=sys.stderr)


def wait_for_job_start(
    job_id: str,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = 3600.0,
    nodefile_path: Path | None = None,
) -> JobStatus:
    """Poll qstat until the job state becomes 'R' (running)."""
    print(f"[PBS] Waiting for job {job_id} to start running...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = query_job_status(job_id)
        if status.state == "R":
            # If exec_host wasn't found in qstat -f, check nodefile
            if not status.exec_host and nodefile_path and nodefile_path.exists():
                node_content = nodefile_path.read_text().strip()
                if node_content:
                    status.exec_host = node_content.splitlines()[0].strip()
            if status.exec_host:
                print(f"[PBS] Job {job_id} is RUNNING on head node: {status.exec_host}")
                return status
        elif status.state in ("E", "F"):
            raise RuntimeError(f"Job {job_id} finished or exited before starting (state={status.state}).")

        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out waiting for job {job_id} to enter running state.")


def find_latest_checkpoint(
    design_id: str,
    run_prefix: str = "DEBUG-",
    outputs_root: Path | None = None,
) -> tuple[Path | None, int]:
    """Find the highest step checkpoint directory for a given design."""
    if outputs_root is None:
        outputs_root = Path.cwd() / "outputs"

    run_id = f"{run_prefix}{design_id}" if run_prefix else design_id
    candidate_roots = [
        outputs_root / run_id,
        outputs_root / f"{design_id}-DEBUG",
        outputs_root / design_id,
    ]

    latest_step = 0
    latest_dir: Path | None = None

    for root in candidate_roots:
        if not root.exists():
            continue
        # Search all step_* folders
        for step_dir in root.glob("**/checkpoints/step_*"):
            if not step_dir.is_dir():
                continue
            # Ensure it contains model weights or training state
            has_state = (
                (step_dir / "training_state.json").exists()
                or (step_dir / "model.safetensors").exists()
                or (step_dir / "pytorch_model.bin").exists()
            )
            if not has_state:
                continue

            match = re.search(r"step_(\d+)", step_dir.name)
            if match:
                step_val = int(match.group(1))
                if step_val > latest_step:
                    latest_step = step_val
                    latest_dir = step_dir

    return latest_dir, latest_step


def resolve_backbone_id(design_id: str, yaml_path: Path | None = None) -> str:
    """Resolve the HuggingFace backbone ID for a given design."""
    if design_id in KNOWN_BACKBONES:
        return KNOWN_BACKBONES[design_id]

    if "QWEN3" in design_id.upper():
        return "Qwen/Qwen3-0.6B"
    elif "OLMO-1B" in design_id.upper():
        return "allenai/OLMo-1B-0724-hf"
    elif "OLMO3-7B" in design_id.upper():
        return "allenai/Olmo-3-7B-Instruct"

    if yaml_path and yaml_path.exists():
        try:
            with yaml_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            for item in data.get("experiments", []):
                if item.get("id") == design_id:
                    overrides = item.get("common_overrides", {})
                    if "model.backbone_id" in overrides:
                        return overrides["model.backbone_id"]
        except Exception:
            pass

    return "allenai/OLMo-7B-0724-hf"


def should_preempt(remaining_walltime: int, safety_margin_seconds: int = 600) -> bool:
    """Determine if execution should be preempted to launch the next hold."""
    return remaining_walltime <= safety_margin_seconds


def build_timeseries_launch_command(
    design_id: str,
    head_node: str,
    resume_from_checkpoint: str | None = None,
    max_steps: int = 5000,
    eval_every_n_steps: int = 100,
    save_every_n_steps: int = 25,
    viz_every_n_steps: int = 100,
    run_prefix: str = "DEBUG-",
    dist_strategy: str = "hsdp",
    fsdp_sharding: str = "shard_grad_op",
    webdataset_dir: str = DEFAULT_WEBDATASET_DIR,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    use_shared_venv: bool = False,
    debug: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Construct command to launch training via tools/timeseries_launch.py."""
    cmd = [
        "python",
        "tools/timeseries_launch.py",
        "--design",
        design_id,
        "--head-node",
        head_node,
        "--run-via-ssh",
        "--run-prefix",
        run_prefix,
        "--max-steps",
        str(max_steps),
        "--eval-every-n-steps",
        str(eval_every_n_steps),
        "--save-every-n-steps",
        str(save_every_n_steps),
        "--viz-every-n-steps",
        str(viz_every_n_steps),
        "--dist-strategy",
        dist_strategy,
        "--fsdp-sharding",
        fsdp_sharding,
        "--webdataset-dir",
        webdataset_dir,
        "--wandb-project",
        wandb_project,
        "--wandb-entity",
        wandb_entity,
    ]

    if resume_from_checkpoint:
        cmd.extend(["--resume-from-checkpoint", str(resume_from_checkpoint)])
    if use_shared_venv:
        cmd.append("--use-shared-venv")
    if debug:
        cmd.append("--debug")
    if dry_run:
        cmd.append("--dry-run")

    return cmd


def build_evaluator_command(
    checkpoint_path: str,
    backbone_id: str,
    val_shards_dir: str = DEFAULT_VAL_SHARDS_DIR,
    mode: str = "verify_timeseries_scits",
    limit: int = 0,
) -> str:
    """Build shell command to run PRISM Universal Evaluator on SciTS validation shards."""
    parts = [
        "export PYTHONNOUSERSITE=1 && unset PYTHONPATH &&",
        f"PRISM_VAL_SHARDS_DIR={shlex.quote(val_shards_dir)}",
        "python tools/universal_evaluator.py",
        f"--checkpoint {shlex.quote(checkpoint_path)}",
        f"--mode {shlex.quote(mode)}",
        "--validation",
        f"--limit {limit}",
        f"--backbone {shlex.quote(backbone_id)}",
    ]
    return " ".join(parts)


def execute_training_segment(
    cmd: str,
    max_duration_sec: float,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> int:
    """Execute a training command in a subprocess, gracefully terminating before walltime expires."""
    print(f"\n[Training] Launching training process (max allowed duration: {max_duration_sec:.0f}s):")
    print(f"  {cmd}\n")

    start_time = time.time()
    env = os.environ.copy()

    # Launch process in its own process group so signals propagate to children (ssh, bash, python)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        env=env,
        preexec_fn=os.setsid,
    )

    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                print(f"[Training] Process completed with exit code: {ret}")
                return ret

            elapsed = time.time() - start_time
            if elapsed >= max_duration_sec:
                print(f"\n[Training] Time limit reached ({elapsed:.0f}s >= {max_duration_sec:.0f}s). Preempting for next hold...")
                try:
                    # Send SIGINT to allow graceful checkpointing
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    for _ in range(30):
                        if proc.poll() is not None:
                            break
                        time.sleep(1)
                    if proc.poll() is None:
                        # Escalate to SIGTERM
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        time.sleep(5)
                    if proc.poll() is None:
                        # Force kill if needed
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception as e:
                    print(f"[Training] Error during process shutdown: {e}", file=sys.stderr)
                return -1

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user. Terminating training process...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except Exception:
            pass
        raise


def execute_evaluator(cmd: str) -> int:
    """Run evaluator command in the current environment."""
    print(f"\n{'='*80}\n[Universal Evaluator] Running SciTS Evaluation:\n  {cmd}\n{'='*80}\n")
    full_cmd = f"module load frameworks && source .venv-deepspeed/bin/activate && {cmd}"
    try:
        proc = subprocess.run(
            full_cmd,
            shell=True,
            executable="/bin/bash",
            check=True,
        )
        return proc.returncode
    except subprocess.CalledProcessError as exc:
        print(f"[Universal Evaluator] Failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode


def resolve_designs(args: argparse.Namespace) -> list[str]:
    """Resolve target designs from CLI or experiment YAML."""
    if getattr(args, "designs", None):
        return args.designs
    if not getattr(args, "yaml", None):
        return []
    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Experiment YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    designs = []
    for item in data.get("experiments", []):
        if not isinstance(item, dict):
            continue
        design_id = str(item.get("id") or "").strip()
        if design_id and TIME_SERIES_DESIGN_RE.fullmatch(design_id):
            designs.append(design_id)
    return designs


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--design",
        dest="single_design",
        default=None,
        help="Single PRISM design ID (e.g. PRISM-QWEN3-0-6B-INTERN-S2-397B).",
    )
    group.add_argument(
        "--designs",
        dest="multi_designs",
        default=None,
        help="Comma-separated list of PRISM design IDs.",
    )
    parser.add_argument(
        "--yaml",
        default=None,
        help="Experiment YAML under experiments/ to scan for time-series designs.",
    )
    parser.add_argument(
        "--hold-script",
        default=DEFAULT_HOLD_SCRIPT,
        help=f"PBS script used to reserve a node (default: {DEFAULT_HOLD_SCRIPT}).",
    )
    parser.add_argument(
        "--safety-margin-minutes",
        type=float,
        default=DEFAULT_SAFETY_MARGIN_MINUTES,
        help=f"Minutes before hold expiration to preempt and renew hold (default: {DEFAULT_SAFETY_MARGIN_MINUTES}).",
    )
    parser.add_argument(
        "--walltime-seconds",
        type=int,
        default=DEFAULT_WALLTIME_SECONDS,
        help=f"Expected hold job walltime in seconds (default: {DEFAULT_WALLTIME_SECONDS}).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Polling interval in seconds (default: {DEFAULT_POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Target total training steps across all hold iterations (default: 5000).",
    )
    parser.add_argument(
        "--eval-every-n-steps",
        type=int,
        default=100,
        help="Evaluation cadence during training.",
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=25,
        help="Checkpoint save cadence.",
    )
    parser.add_argument(
        "--viz-every-n-steps",
        type=int,
        default=100,
        help="Visualization cadence.",
    )
    parser.add_argument(
        "--run-prefix",
        default="DEBUG-",
        help="Prefix used for run ID / outputs directory.",
    )
    parser.add_argument(
        "--dist-strategy",
        default="hsdp",
        choices=["ddp", "fsdp", "hsdp"],
        help="Distributed strategy for Aurora.",
    )
    parser.add_argument(
        "--fsdp-sharding",
        default="shard_grad_op",
        choices=["full_shard", "shard_grad_op", "hybrid_shard", "no_shard"],
        help="FSDP sharding mode.",
    )
    parser.add_argument(
        "--webdataset-dir",
        default=DEFAULT_WEBDATASET_DIR,
        help="WebDataset directory containing SciTS training shards.",
    )
    parser.add_argument(
        "--val-shards-dir",
        default=DEFAULT_VAL_SHARDS_DIR,
        help="Directory containing SciTS held-out validation shards.",
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
        "--eval-mode",
        default="verify_timeseries_scits",
        choices=["verify_timeseries_scits", "run_eval"],
        help="Universal evaluator mode to run at completion (default: verify_timeseries_scits).",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="Limit of validation samples to evaluate (default: 0 = all validation shards).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training SciTS evaluation.",
    )
    parser.add_argument(
        "--use-shared-venv",
        action="store_true",
        help="Use shared virtualenv instead of unpacking tarball.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug verbosity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions and commands without submitting jobs or running training.",
    )
    parser.add_argument(
        "--existing-job-id",
        default=None,
        help="Attach to an already submitted or running PBS hold job ID.",
    )

    args = parser.parse_args(argv)

    # Coerce designs list
    if args.single_design:
        args.designs = [args.single_design]
    elif args.multi_designs:
        args.designs = [d.strip() for d in args.multi_designs.split(",") if d.strip()]
    elif args.yaml:
        args.designs = resolve_designs(args)
    else:
        parser.error("Must specify --design, --designs, or --yaml")

    args.safety_margin_seconds = int(args.safety_margin_minutes * 60)
    return args


def run_hold_loop(args: argparse.Namespace, outputs_root: Path | None = None) -> int:
    """Main orchestration loop: holds nodes, runs/resumes training, evaluates SciTS validation shards."""
    if outputs_root is None:
        outputs_root = Path.cwd() / "outputs"

    designs = args.designs
    if not designs:
        print("No matching PRISM designs to run.", file=sys.stderr)
        return 1

    print("=== PRISM Time-Series Hold Runner Starting ===")
    print(f"Target Designs: {designs}")
    print(f"Target Max Steps: {args.max_steps}")
    print(f"Safety Margin: {args.safety_margin_minutes} minutes ({args.safety_margin_seconds}s)")
    print(f"Hold Script: {args.hold_script}")

    current_job_id = args.existing_job_id
    nodefile_path = Path("logs/hold_1n_timeseries_debug_nodefile.txt")

    for design_idx, design_id in enumerate(designs):
        print(f"\n{'='*80}")
        print(f"Processing Design [{design_idx + 1}/{len(designs)}]: {design_id}")
        print(f"{'='*80}")

        while True:
            # Check current checkpoint status
            latest_ckpt_dir, current_step = find_latest_checkpoint(
                design_id=design_id,
                run_prefix=args.run_prefix,
                outputs_root=outputs_root,
            )

            if current_step >= args.max_steps:
                print(f"[Design {design_id}] Reached target max_steps ({current_step} >= {args.max_steps}). Training complete!")
                break

            print(f"[Design {design_id}] Current progress: step {current_step} / {args.max_steps}")
            if latest_ckpt_dir:
                print(f"[Design {design_id}] Will resume from checkpoint: {latest_ckpt_dir}")

            # Ensure we have an active hold job
            job_status: JobStatus | None = None
            if current_job_id:
                job_status = query_job_status(current_job_id)
                if job_status.state not in ("Q", "R"):
                    print(f"[PBS] Existing job {current_job_id} is no longer active (state={job_status.state}). Requesting new hold.")
                    current_job_id = None
                    job_status = None

            if not current_job_id:
                if args.dry_run:
                    print(f"[DRY-RUN] Would submit hold job: qsub {args.hold_script}")
                    current_job_id = "DRY_RUN_JOB_001"
                    job_status = JobStatus(job_id=current_job_id, state="R", exec_host="dry-run-node", remaining_walltime=3600)
                else:
                    current_job_id = submit_hold_job(args.hold_script)
                    job_status = wait_for_job_start(
                        job_id=current_job_id,
                        poll_interval=args.poll_interval,
                        nodefile_path=nodefile_path,
                    )
            elif job_status and job_status.state != "R":
                if not args.dry_run:
                    job_status = wait_for_job_start(
                        job_id=current_job_id,
                        poll_interval=args.poll_interval,
                        nodefile_path=nodefile_path,
                    )

            # Check remaining walltime on current hold
            if job_status is None:
                job_status = query_job_status(current_job_id)

            remaining_sec = job_status.remaining_walltime
            head_node = job_status.exec_host or "localhost"

            print(f"[PBS] Hold Job {current_job_id}: Head Node={head_node}, Remaining Walltime={remaining_sec}s ({remaining_sec/60:.1f} min)")

            if should_preempt(remaining_sec, args.safety_margin_seconds):
                print(f"[PBS] Hold job {current_job_id} is expiring soon (remaining {remaining_sec}s <= margin {args.safety_margin_seconds}s).")
                print("[PBS] Preemptively cycling to a fresh hold job...")
                if not args.dry_run:
                    delete_job(current_job_id)
                current_job_id = None
                continue

            # Calculate safe duration to run before we hit the safety margin
            max_training_duration = max(0, remaining_sec - args.safety_margin_seconds)
            print(f"[Runner] Allocated training segment duration: {max_training_duration:.0f}s ({max_training_duration/60:.1f} min)")

            # Build and execute training command
            cmd_parts = build_timeseries_launch_command(
                design_id=design_id,
                head_node=head_node,
                resume_from_checkpoint=str(latest_ckpt_dir) if latest_ckpt_dir else None,
                max_steps=args.max_steps,
                eval_every_n_steps=args.eval_every_n_steps,
                save_every_n_steps=args.save_every_n_steps,
                viz_every_n_steps=args.viz_every_n_steps,
                run_prefix=args.run_prefix,
                dist_strategy=args.dist_strategy,
                fsdp_sharding=args.fsdp_sharding,
                webdataset_dir=args.webdataset_dir,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                use_shared_venv=args.use_shared_venv,
                debug=args.debug,
                dry_run=args.dry_run,
            )
            training_cmd = "module load frameworks && source .venv-deepspeed/bin/activate && " + " ".join(cmd_parts)

            if args.dry_run:
                print(f"[DRY-RUN] Would execute: {training_cmd}")
                break

            segment_ret = execute_training_segment(
                cmd=training_cmd,
                max_duration_sec=max_training_duration,
                poll_interval=args.poll_interval,
            )

            # Check if training completed or was preempted
            latest_ckpt_dir, current_step = find_latest_checkpoint(
                design_id=design_id,
                run_prefix=args.run_prefix,
                outputs_root=outputs_root,
            )

            if current_step >= args.max_steps or segment_ret == 0:
                print(f"[Design {design_id}] Training completed at step {current_step}!")
                break
            else:
                print(f"[Runner] Segment ended at step {current_step}. Cycling hold job for resume...")
                if current_job_id:
                    delete_job(current_job_id)
                    current_job_id = None

        # Post-training: Run Universal Evaluator on SciTS Validation Shards
        if not args.skip_eval:
            print(f"\n{'='*80}")
            print(f"Running Universal Evaluator on SciTS Validation Shards for {design_id}")
            print(f"{'='*80}")

            final_ckpt, final_step = find_latest_checkpoint(
                design_id=design_id,
                run_prefix=args.run_prefix,
                outputs_root=outputs_root,
            )
            if not final_ckpt and not args.dry_run:
                print(f"[Universal Evaluator] Error: No checkpoint found for {design_id} in {outputs_root}", file=sys.stderr)
                continue

            ckpt_arg = str(final_ckpt) if final_ckpt else f"outputs/{args.run_prefix}{design_id}/checkpoints/step_{args.max_steps}"
            backbone_id = resolve_backbone_id(design_id, Path(args.yaml) if args.yaml else None)

            eval_cmd = build_evaluator_command(
                checkpoint_path=ckpt_arg,
                backbone_id=backbone_id,
                val_shards_dir=args.val_shards_dir,
                mode=args.eval_mode,
                limit=args.eval_limit,
            )

            if args.dry_run:
                print(f"[DRY-RUN] Would execute evaluator: {eval_cmd}")
            else:
                eval_ret = execute_evaluator(eval_cmd)
                if eval_ret != 0:
                    print(f"[Universal Evaluator] Warning: evaluation returned non-zero code {eval_ret}", file=sys.stderr)

    # Clean up hold job at the end
    if current_job_id and not args.dry_run:
        delete_job(current_job_id)

    print("\n[Hold Runner] All designs completed successfully.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return run_hold_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())

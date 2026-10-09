#!/usr/bin/env python3
"""
launch_aurora_daos.py - Launch PRISM with DAOS-backed WebDataset

Simplified launcher that reads WebDataset shards directly from DAOS
instead of staging to /tmp. This eliminates per-job copy overhead.

Key differences from launch_aurora_web.py:
- No shard staging to /tmp (reads directly from DAOS mount)
- Uses libpil4dfs.so for kernel-bypass I/O
- Requests daos_user_fs in PBS filesystems
- Much faster job startup (no stagger delays)

Usage:
    python tools/launch_aurora_daos.py \
        --id PRISM-AURORA-DAOS \
        --packed-env deepspeed_env.tar.gz \
        --nodes 2
"""

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch_common import (  # noqa: E402
    DEFAULT_IMAGE_ENCODER_ID,
    hf_cache_dir,
    load_dotenv,
    load_model_group_config,
    lookup_experiment,
    resolve_hydra_override,
    unique_hf_cache_dirs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Launch PRISM with DAOS WebDataset support on Aurora"
    )
    parser.add_argument(
        "--file",
        default="experiments/prism_designs.yaml",
        help="Experiment Design YAML file",
    )
    parser.add_argument("--id", required=True, help="Run ID / Job Name")
    parser.add_argument(
        "--design", help="Experiment Design ID (from YAML). Defaults to same as --id"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print script instead of executing"
    )
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--batch", action="store_true", help="Generate Batch Script (PBS)"
    )
    parser.add_argument("--project", default="AuroraGPT", help="Project Allocation")
    parser.add_argument("--queue", default="debug", help="Queue Name")
    parser.add_argument("--walltime", default=None, help="Walltime (overrides config)")
    parser.add_argument(
        "--packed-env",
        default="deepspeed_env.tar.gz",
        help="Path to packed env tarball",
    )
    parser.add_argument(
        "--prism-dir",
        default=os.getcwd(),
        help="Path to PRISM checkout (default: current working directory)",
    )
    parser.add_argument(
        "--hf-fallback-dirs",
        default="/flare/ModCon/ngetty/huggingface/hub,/flare/ModCon/sandeep/hub",
        help="Comma-separated list of HuggingFace cache directories to fall back to "
        "if a model isn't on DAOS. Searched in order.",
    )

    # DAOS Configuration
    parser.add_argument(
        "--daos-pool",
        default="AuroraGPT",
        help="DAOS pool name",
    )
    parser.add_argument(
        "--daos-container",
        default="prism_training_data",
        help="DAOS container name for training data",
    )
    parser.add_argument(
        "--daos-models-container",
        default="prism_models",
        help="DAOS container name for model weights (faster multi-node staging)",
    )
    parser.add_argument(
        "--daos-dataset",
        default=None,
        help="Single dataset path within DAOS container (legacy, use --dataset-groups instead)",
    )
    parser.add_argument(
        "--dataset-groups",
        default="all",
        help="Dataset groups to use: 'all', 'pixmo', 's1mmalign', 'nemotron', 'cosyn', or comma-separated list",
    )
    parser.add_argument(
        "--dataset-config",
        default="src/conf/data/daos_datasets.yaml",
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--dataset-proportions",
        default=None,
        help="Override dataset proportions (0.0-1.0). Format: 'dataset1:0.1,dataset2:0.2'. "
        "Controls what fraction of each dataset's shards to use. Useful for faster startup.",
    )
    parser.add_argument(
        "--finite-webdataset",
        action="store_true",
        help="Disable WebDataset resampling. Each iterator consumes assigned shards once; "
        "the trainer restarts the iterator for the next shuffled epoch.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=16,
        help="CPUs per rank for DataLoader workers",
    )

    # DDP mode
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        help="Use Accelerate instead of native DDP",
    )
    parser.add_argument(
        "--deepspeed",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Use DeepSpeed via Accelerate with ZeRO stage 1, 2, or 3. "
        "Stage 1 shards optimizer states, stage 2 also shards gradients. "
        "Stage 3 shards params+grads+optimizer (comparable to FSDP). "
        "Selects scripts/accelerate_configs/deepspeed_zero{1,2,3}.yaml.",
    )
    parser.add_argument(
        "--debug-no-sync",
        action="store_true",
        help="Run backward WITHOUT DDP AllReduce to measure pure compute time",
    )
    parser.add_argument(
        "--no-pil4dfs",
        action="store_true",
        help="Disable libpil4dfs.so interception library (workaround for Python/webdataset compatibility)",
    )
    parser.add_argument(
        "--use-shared-venv",
        action="store_true",
        help="Source the shared venv from VENV_PATH in .env instead of extracting "
        "the tarball to /tmp on every node. Requires VENV_PATH to be set in .env "
        "and the venv to have been built via tools/build_aurora_env.sh.",
    )
    parser.add_argument(
        "--find-unused-params",
        action="store_true",
        help="Enable DDP find_unused_parameters (disables static_graph optimization, required for dynamic models)",
    )

    # === GPU Hierarchy ===
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Use COMPOSITE GPU hierarchy: 2 tiles per card become 1 logical device with ~128GB HBM. "
        "Launches 6 ranks/node instead of 12. Enables DDP for 7B+ models without FSDP.",
    )

    # === Distributed Strategy Selection ===
    parser.add_argument(
        "--dist-strategy",
        choices=["ddp", "fsdp", "hsdp"],
        default="ddp",
        help="Distributed training strategy: ddp (default), fsdp (better for 7B+), hsdp (hybrid)",
    )
    parser.add_argument(
        "--fsdp-sharding",
        choices=["full_shard", "shard_grad_op", "hybrid_shard", "no_shard"],
        default="full_shard",
        help="FSDP sharding strategy (only used with --dist-strategy fsdp)",
    )
    parser.add_argument(
        "--fsdp-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for FSDP (saves GPU memory, slower)",
    )
    parser.add_argument(
        "--ddp-bucket-mb",
        type=int,
        default=50,
        help="DDP gradient bucket size in MB (larger = fewer AllReduces)",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for backbone (significant speedup, slow first step)",
    )
    parser.add_argument(
        "--grad-ckpt-freq",
        type=int,
        default=1,
        help="Gradient checkpoint frequency: 1=every layer (default), 2=every other, 0=disabled",
    )
    # === HSDP perf knobs (validated by scaling-study/investigation/REPORT.md) ===
    parser.add_argument(
        "--fsdp-no-sync-accum",
        action="store_true",
        help="Opt FSDP/HSDP into no_sync() on non-final accumulation microbatches "
        "(FSDP_NO_SYNC_ACCUM=1). Only safe when 2x grad HBM fits — validated for "
        "Qwen3-0.6B + SigLIP2; for 7B+ models leave OFF. Pair with "
        "--gradient-accumulation-steps>=2 for the +68 percent 10N HSDP win.",
    )
    parser.add_argument(
        "--prism-disable-perf-probes",
        action="store_true",
        help="Skip the every-50-steps 1-element AllReduce probe (PRISM_DISABLE_PERF_PROBES=1). "
        "The probe is cheap but its torch.xpu.synchronize() drains in-flight "
        "collectives and pollutes the next logging window. Enable for clean perf runs.",
    )
    parser.add_argument(
        "--grad-norm-interval",
        type=int,
        default=50,
        help="Encoder/projector grad-norm logging cadence (GRAD_NORM_INTERVAL). "
        "Each fire walks every named param's grad with .norm().item() — host syncs "
        "pollute throughput. Set to 0 for clean perf runs, larger for less noise.",
    )

    # === Sequence Length Bucketing ===
    parser.add_argument(
        "--use-bucketing",
        action="store_true",
        help="Enable sequence length bucketing for mixed-length datasets. "
        "Buffers samples and sorts by length to minimize padding waste.",
    )
    parser.add_argument(
        "--bucket-buffer-size",
        type=int,
        default=2000,
        help="Number of samples to buffer before sorting by length (larger = better bucketing, default: 2000)",
    )
    parser.add_argument(
        "--bucket-num-buckets",
        type=int,
        default=8,
        help="Number of length buckets for grouping similar sequences (more = tighter grouping, default: 8)",
    )
    parser.add_argument(
        "--use-bucketed-collator",
        action="store_true",
        dest="use_bucketed_collator",
        help="Sort samples by length within each batch to reduce padding (default: on)",
    )
    parser.add_argument(
        "--no-bucketed-collator",
        action="store_false",
        dest="use_bucketed_collator",
        help="Disable within-batch sorting by length",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization (default: 2048, Molmo uses 2560)",
    )
    parser.add_argument(
        "--enable-all-modalities",
        action="store_true",
        help="Set ENABLE_ALL_MODALITIES=1 in the training env. Activates all "
        "skip=true datasets whose modality is in model.modalities and falls "
        "back to dummy tensors when real shards are missing. Off by default.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps. When omitted, the Hydra/design value is used. "
        "Useful for short smoke runs (e.g. --max-steps 50).",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=None,
        help="IsoFLOP budget in total FLOPs (e.g. 3e18). Computes max_steps from "
        "--calibration-json's flops_per_step. Mutually exclusive with --max-steps. "
        "When set, also exports CALIBRATION_JSON=<path> so the trainer's _FlopCounter "
        "can attribute cumulative FLOPs in perf.jsonl.",
    )
    parser.add_argument(
        "--calibration-json",
        type=str,
        default=None,
        help="Path to a JSON written by tools/isoflop_calibrate.py. Required when "
        "--target-flops is set; ignored otherwise.",
    )
    parser.add_argument(
        "--runtime-flops-per-step",
        type=float,
        default=None,
        help="Plan's runtime-rescaled per-step FLOPs (cal_fps * (rbs/cbs)*(rsl/csl)*"
        "(rranks/cranks)). When set, exported to the trainer as "
        "RUNTIME_FLOPS_PER_STEP so the _FlopCounter accumulates in "
        "rescaled FLOPs and `cumulative_flops` in perf.jsonl matches the "
        "plan's `budget_flops`. Populated by tools/isoflop_launch.py from "
        "the cell's `runtime_fps`. Without this, cumulative_flops would "
        "undercount the budget by `rescale_factor` (~48x default).",
    )

    # === Profiling & Debugging ===
    parser.add_argument(
        "--enable-profiler",
        action="store_true",
        help="Enable PyTorch profiler for detailed performance analysis",
    )
    parser.add_argument(
        "--profiler-steps",
        type=str,
        default="5,10,15",
        help="Comma-separated step numbers to profile (default: 5,10,15)",
    )
    parser.add_argument(
        "--ccl-debug",
        action="store_true",
        help="Enable CCL debug logging for communication analysis",
    )
    parser.add_argument(
        "--daos-debug",
        action="store_true",
        help="Enable DAOS I/O tracing for storage analysis",
    )
    parser.add_argument(
        "--per-rank-timing",
        action="store_true",
        help="Enable per-rank timing instrumentation for straggler detection",
    )
    parser.add_argument(
        "--ddp-debug",
        action="store_true",
        help="Enable DDP_DEBUG=1 for verbose per-microbatch prints (hang diagnosis)",
    )
    parser.add_argument(
        "--fsdp-production-mode",
        action="store_true",
        help="Skip non-essential torch.xpu.synchronize() calls (FSDP_PRODUCTION_MODE=1)",
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Enable production mode for both native DDP/FSDP and ZoneATrainer "
        "(sets FSDP_PRODUCTION_MODE=1 and PRISM_PRODUCTION_MODE=1) for fair "
        "throughput comparisons across distributed strategies.",
    )
    parser.add_argument(
        "--step-watchdog-timeout",
        type=int,
        default=0,
        help="Per-step watchdog timeout in seconds (0=disabled). Prints stack trace if step exceeds timeout.",
    )
    parser.add_argument(
        "--viz-interval",
        type=int,
        default=None,
        help="Visualize predictions every N steps (default: 500, use lower values for short runs)",
    )

    # === Checkpoint & Resume ===
    parser.add_argument(
        "--resume-weights-only",
        type=str,
        default=None,
        help="Path to checkpoint directory to load model weights from (fresh optimizer). "
        "Use for stage transitions, e.g., projector-only -> E2E fine-tuning. "
        "Example: outputs/PROJECTOR-STAGE1/2026-02-19/12-00-00/checkpoints/step_500",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory for full native resume: model, optimizer, "
        "scheduler, W&B run id, and deterministic finite-dataloader replay.",
    )

    # === Multi-Node Interactive Support ===
    parser.add_argument(
        "--hosts",
        type=str,
        default=None,
        help="Comma-separated list of hostnames for multi-node interactive jobs (e.g., 'x4418c4s4b0n0,x4418c7s7b0n0'). "
        "Bypasses PBS_NODEFILE requirement.",
    )
    parser.add_argument(
        "--run-via-ssh",
        action="store_true",
        help="Execute the script on the first host via SSH (for running from UAN)",
    )
    parser.add_argument(
        "--pbs-jobid",
        default=None,
        help="PBS job ID to forward via SSH so multi-node mpiexec can attach to "
        "the PALS shepherd (e.g. '8429838.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov'). "
        "Required for multi-node --run-via-ssh when PBS_JOBID is not already set in "
        "the UAN environment. Falls back to $PBS_JOBID.",
    )

    # Logging
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="WandB project name (overrides config default)",
    )
    parser.add_argument(
        "--suppress-warnings",
        action="store_true",
        dest="suppress_warnings",
        help="Suppress Python warnings to clean up logs (default: on)",
    )
    parser.add_argument(
        "--no-suppress-warnings",
        action="store_false",
        dest="suppress_warnings",
        help="Show Python warnings in logs",
    )

    parser.set_defaults(use_bucketed_collator=True, suppress_warnings=True)
    args, unknown_args = parser.parse_known_args()

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error(f"--max-steps must be positive, got {args.max_steps}")

    if args.runtime_flops_per_step is not None and args.runtime_flops_per_step <= 0:
        parser.error(
            f"--runtime-flops-per-step must be positive, got {args.runtime_flops_per_step}"
        )
    if args.resume_from_checkpoint and args.resume_weights_only:
        parser.error(
            "--resume-from-checkpoint and --resume-weights-only are mutually exclusive"
        )

    # IsoFLOP `--target-flops` handling. Mutually exclusive with --max-steps —
    # otherwise the user has two competing answers for "how many steps?".
    # When set, read the calibration JSON's flops_per_step and convert.
    target_flops_steps: int | None = None
    if args.target_flops is not None:
        if args.target_flops <= 0:
            parser.error(f"--target-flops must be positive, got {args.target_flops}")
        if args.max_steps is not None:
            parser.error(
                "--target-flops and --max-steps are mutually exclusive. "
                "Either pass a FLOP budget OR a step count, not both."
            )
        if not args.calibration_json:
            parser.error("--target-flops requires --calibration-json <path>")
        cal_path = os.path.abspath(args.calibration_json)
        if not os.path.isfile(cal_path):
            parser.error(f"--calibration-json not found: {cal_path}")
        try:
            with open(cal_path) as _cf:
                _cal = json.load(_cf)
        except (OSError, json.JSONDecodeError) as _e:
            parser.error(f"--calibration-json unreadable: {cal_path}: {_e}")
        _fps = _cal.get("flops_per_step")
        try:
            _fps = float(_fps)
        except (TypeError, ValueError):
            parser.error(
                f"--calibration-json {cal_path}: flops_per_step not numeric ({_fps!r})"
            )
        if _fps <= 0:
            parser.error(
                f"--calibration-json {cal_path}: flops_per_step must be > 0, got {_fps}"
            )
        target_flops_steps = max(1, round(args.target_flops / _fps))
        args.calibration_json = cal_path  # canonicalize for downstream export
        print(
            f"[isoflop] --target-flops={args.target_flops:.3e} / flops_per_step={_fps:.3e} "
            f"-> max_steps={target_flops_steps}"
        )
        # Loud warning: --target-flops here uses RAW cal_fps and skips the
        # cal/runtime rescale that tools/isoflop_plan.py applies. For
        # multi-rank runs at runtime bs/sl != cal bs/sl, max_steps will be
        # off by `rescale_factor` (~48x default). The supported IsoFLOP
        # entrypoint is tools/isoflop_launch.py which passes --max-steps
        # from the rescaled plan instead.
        print(
            "[isoflop] WARNING: --target-flops uses RAW cal_fps (no runtime rescale). "
            "For IsoFLOP sweeps prefer tools/isoflop_launch.py (which passes the "
            "plan's rescaled --max-steps + --runtime-flops-per-step). Use "
            "--target-flops directly only when cal config == runtime config.",
            file=sys.stderr,
        )
    elif args.calibration_json is not None:
        # Calibration without --target-flops is the "env-var only" path:
        # the launcher exports CALIBRATION_JSON into the qsub script so
        # the trainer's _FlopCounter can attribute cumulative FLOPs, but
        # max_steps is supplied separately (via --max-steps or the design
        # default). This is the mode tools/isoflop_launch.py uses post-PR
        # for the cal/runtime rescale fix — the plan owns the step count,
        # the launcher just propagates the cal handle. Validate the file
        # is readable so a typo doesn't reach the trainer silently.
        cal_path = os.path.abspath(args.calibration_json)
        if not os.path.isfile(cal_path):
            parser.error(f"--calibration-json not found: {cal_path}")
        try:
            with open(cal_path) as _cf:
                _ = json.load(_cf)
        except (OSError, json.JSONDecodeError) as _e:
            parser.error(f"--calibration-json unreadable: {cal_path}: {_e}")
        args.calibration_json = cal_path  # canonicalize for downstream export

    # Resolve shared-venv path from .env if requested.
    env_config = load_dotenv()
    shared_venv_path = env_config.get("VENV_PATH") or os.environ.get("VENV_PATH", "")
    if args.use_shared_venv and not shared_venv_path:
        sys.exit(
            "ERROR: --use-shared-venv requires VENV_PATH to be set in .env. "
            "Build the shared venv with: "
            "VENV_PATH=/flare/ModCon/$USER/prism-envs/py3.12 "
            "bash tools/build_aurora_env.sh"
        )

    # DAOS paths
    daos_pool = args.daos_pool
    daos_container = args.daos_container
    daos_models_container = args.daos_models_container
    # DAOS mount paths - include username for consistency with setup_daos_container.sh
    username = os.environ.get("USER")
    if not username:
        sys.exit("ERROR: $USER is not set; cannot derive per-user DAOS mount path")
    daos_mount = f"/tmp/{username}/{daos_pool}/{daos_container}"
    daos_models_mount = f"/tmp/{username}/{daos_pool}/{daos_models_container}"

    # Multi-dataset or legacy single-dataset mode
    use_multi_dataset = args.daos_dataset is None
    dataset_groups = args.dataset_groups
    dataset_config = args.dataset_config
    dataset_proportions = args.dataset_proportions or ""

    if use_multi_dataset:
        # Multi-dataset mode: use dataset groups configuration
        daos_shards_path = daos_mount  # Base path, multi_webdataset handles subdirs
        daos_val_path = daos_mount
        daos_manifest = None
        print("DAOS Configuration (Multi-Dataset Mode):")
        print(f"  Pool: {daos_pool}")
        print(f"  Data Container: {daos_container}")
        print(f"  Models Container: {daos_models_container}")
        print(f"  Data Mount: {daos_mount}")
        print(f"  Models Mount: {daos_models_mount}")
        print(f"  Dataset Groups: {dataset_groups}")
        print(f"  Config: {dataset_config}")
        if dataset_proportions:
            print(f"  Proportion Overrides: {dataset_proportions}")
    else:
        # Legacy single-dataset mode
        daos_dataset = args.daos_dataset
        daos_shards_path = f"{daos_mount}/{daos_dataset}/shards"
        daos_val_path = f"{daos_mount}/{daos_dataset}/val_shards"
        daos_manifest = f"{daos_mount}/{daos_dataset}/manifest.json"  # noqa: F841
        print("DAOS Configuration (Single Dataset Mode):")
        print(f"  Pool: {daos_pool}")
        print(f"  Data Container: {daos_container}")
        print(f"  Models Container: {daos_models_container}")
        print(f"  Dataset: {daos_dataset}")
        print(f"  Data Mount: {daos_mount}")
        print(f"  Models Mount: {daos_models_mount}")

    # Output Management
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "outputs", args.id, date_str, time_str)

    # Load Experiment Spec
    design_id = args.design if args.design else args.id
    target_exp, parent_exp, final_overrides = lookup_experiment(args.file, design_id)

    print(f"Found Experiment: {target_exp.get('name', design_id)}")

    # Apply default_dataset_groups from design if user didn't explicitly set --dataset-groups
    design_default_groups = (
        parent_exp.get("default_dataset_groups") if parent_exp else None
    )
    if design_default_groups and args.dataset_groups == "all":
        args.dataset_groups = design_default_groups
        dataset_groups = args.dataset_groups
        print(f"Using design default dataset groups: {dataset_groups}")

    # DeepSpeed validation: --deepspeed implies --use-accelerate and overrides
    # FSDP/HSDP because DeepSpeed handles its own sharding internally.
    if args.deepspeed is not None:
        args.use_accelerate = True
        if args.dist_strategy in ("fsdp", "hsdp"):
            print(
                f"WARNING: --deepspeed overrides --dist-strategy {args.dist_strategy}. "
                f"DeepSpeed ZeRO-{args.deepspeed} handles sharding internally."
            )
            args.dist_strategy = "ddp"
        # ZeRO-3 does AllGather collectives like FSDP and is exposed to the
        # same DAOS-17499 hang via libpil4dfs. Auto-enable --no-pil4dfs unless
        # the operator passed it explicitly.
        if args.deepspeed == 3 and not args.no_pil4dfs:
            print(
                "INFO: --deepspeed 3 auto-enables --no-pil4dfs "
                "(ZeRO-3 AllGather + libpil4dfs hangs per DAOS-17499)."
            )
            args.no_pil4dfs = True
        print(f"DeepSpeed ZeRO-{args.deepspeed} mode via Accelerate")

    # Resources
    resources = target_exp.get("resources", {"ngpus": 12})
    ngpus = resources.get("ngpus", 12)

    # COMPOSITE mode: 2 tiles merge into 1 logical device -> 6 devices/node
    if args.composite:
        ngpus = 6
        if args.dist_strategy in ("fsdp", "hsdp"):
            print(
                f"WARNING: --composite overrides --dist-strategy {args.dist_strategy} -> ddp "
                "(COMPOSITE gives 128GB/device, FSDP not needed)"
            )
            args.dist_strategy = "ddp"
        print("COMPOSITE mode: 6 devices/node (128GB HBM each), DDP only")

    print(f"Resources: {ngpus} XPUs per node, {args.nodes} nodes")

    # Proxy Setup
    proxy_env = """
# proxy settings
if [[ ! "${HOSTNAME}" =~ aurora-uan ]]; then
    export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
    export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
    export http_proxy="http://proxy.alcf.anl.gov:3128"
    export https_proxy="http://proxy.alcf.anl.gov:3128"
    export ftp_proxy="http://proxy.alcf.anl.gov:3128"
    export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
fi
"""

    if "HF_TOKEN" in os.environ:
        # Reference by name so generated job scripts never contain the token
        # value. Batch jobs intentionally do not forward it implicitly because
        # PBS exposes forwarded variables in job metadata.
        proxy_env += '\nexport HF_TOKEN="$HF_TOKEN"'
    else:
        print("Warning: HF_TOKEN not found. External models might fail to load.")

    # Forward DL_NUM_WORKERS from the caller's environment. tools/run_sweep.py
    # sets this to 0 for non-image cells so HF IterableDatasets with
    # n_shards < world_size don't get their workers silenced. Without this
    # forwarding, the sweep cell's `bash -lc` heredoc loses the var and
    # the bucketed-collator path (src/train.py) defaults to 4 workers,
    # which surfaces as "Stream X is persistently empty after 5 restarts".
    dl_num_workers = os.environ.get("DL_NUM_WORKERS") or env_config.get("DL_NUM_WORKERS")
    if dl_num_workers:
        proxy_env += f'\nexport DL_NUM_WORKERS="{dl_num_workers}"'

    # Forward the optional path-override env vars from .env (see
    # .env.template). src/config.py and src/encoders/geometry.py read these
    # directly via os.environ.get(...) with placeholder fallbacks, so unless
    # they're exported into the job's env they never reach the compute node.
    for var in (
        "PRISM_CALVIN_ROOT",
        "PRISM_AURORAGPT_2B_CHECKPOINT",
        "PRISM_OLMO1B_INTERLEAVED_TOKENIZER",
        "PRISM_WALRUS_WEIGHTS_PATH",
    ):
        val = os.environ.get(var) or env_config.get(var)
        if val:
            proxy_env += f'\nexport {var}="{val}"'

    # Build prism directory
    prism_dir = args.prism_dir
    model_group = resolve_hydra_override(final_overrides, unknown_args, "model", None)
    model_group_config = load_model_group_config(prism_dir, model_group)

    # Handle overrides
    overrides_list = []
    for k, v in final_overrides.items():
        # Handle list values (e.g., model.modalities: [text, image])
        # Convert Python list to Hydra-compatible format: '[item1,item2]'
        # Must be quoted for bash and have no spaces for Hydra parser
        if isinstance(v, list):
            list_str = "[" + ",".join(str(item) for item in v) + "]"
            overrides_list.append(f"'{k}={list_str}'")
        else:
            overrides_list.append(f"{k}={v}")

    backbone_id = resolve_hydra_override(
        final_overrides,
        unknown_args,
        "model.backbone_id",
        model_group_config.get("backbone_id", "allenai/OLMo-7B-0724-hf"),
    )
    image_encoder_id = resolve_hydra_override(
        final_overrides,
        unknown_args,
        "model.image_encoder_id",
        model_group_config.get("image_encoder_id", DEFAULT_IMAGE_ENCODER_ID),
    )
    image_processor_id = resolve_hydra_override(
        final_overrides,
        unknown_args,
        "model.image_processor_id",
        model_group_config.get("image_processor_id"),
    )
    image_model_dirs = unique_hf_cache_dirs([image_encoder_id, image_processor_id])
    image_model_stage_entries = "\n".join(f'    "{d}"' for d in image_model_dirs)

    # Detect local filesystem paths vs HuggingFace model IDs.
    # Local paths (starting with /) are loaded directly by transformers and
    # do not need staging through the HF cache directory structure.
    hf_model_dir = hf_cache_dir(backbone_id)
    if hf_model_dir is None:
        print(f"Model staging: {backbone_id} (local path, no staging needed)")
    else:
        print(f"Model staging: {backbone_id} -> {hf_model_dir}")
    if image_encoder_id:
        print(f"Image encoder staging: {image_encoder_id} -> {hf_cache_dir(image_encoder_id)}")
    if image_processor_id and image_processor_id != image_encoder_id:
        print(
            f"Image processor staging: {image_processor_id} -> "
            f"{hf_cache_dir(image_processor_id)}"
        )

    if "training.device" not in final_overrides:
        overrides_list.append("training.device=xpu")

    overrides_list.append(f"exp.id={args.id}")
    overrides_list.append(f"hydra.run.dir={output_dir}")

    if args.wandb_project:
        overrides_list.append(f"wandb.project={args.wandb_project}")
        overrides_list.append("wandb.mode=online")

    if args.viz_interval is not None:
        overrides_list.append(f"training.viz_every_n_steps={args.viz_interval}")

    def _set_override(key: str, value: object) -> None:
        """Append `key=value` to overrides_list, removing any prior `key=`.

        Without this, launcher-injected overrides (e.g. training.max_steps
        from --max-steps or --target-flops) leave the design's value in
        place too. Hydra last-wins resolves it but the cmd line gets a
        confusing duplicate entry — mirrors the CLI-conflict block below.
        """
        prior = [
            i for i, existing in enumerate(overrides_list)
            if existing.split("=", 1)[0].strip("'") == key
        ]
        for idx in sorted(prior, reverse=True):
            old = overrides_list.pop(idx)
            print(
                f"INFO: launcher override '{key}={value}' replaces design value '{old}'"
            )
        overrides_list.append(f"{key}={value}")

    if args.max_steps is not None:
        _set_override("training.max_steps", args.max_steps)
    elif target_flops_steps is not None:
        # --target-flops path: convert the budget into max_steps via the
        # calibrator JSON above. The trainer reads CALIBRATION_JSON from
        # the env (exported below) to fold cumulative FLOPs into perf rows.
        _set_override("training.max_steps", target_flops_steps)

    if args.resume_weights_only:
        ckpt_path = os.path.abspath(args.resume_weights_only)
        overrides_list.append(f"training.resume_weights_only={ckpt_path}")
        print(f"Resume (weights-only): {ckpt_path}")
    if args.resume_from_checkpoint:
        ckpt_path = os.path.abspath(args.resume_from_checkpoint)
        overrides_list.append(f"training.resume_from_checkpoint={ckpt_path}")
        print(f"Resume (full state): {ckpt_path}")

    # CLI overrides — these take precedence over design overrides.
    # Detect and warn about conflicts: if a CLI arg sets a key already defined
    # by the experiment design, the CLI value wins (last-write-wins in Hydra),
    # but we warn loudly so users know their design values are being overridden.
    # We also remove the earlier design value to avoid confusing duplicate entries.
    for arg in unknown_args:
        if "=" in arg and not arg.startswith("--"):
            cli_key = arg.split("=", 1)[0]
            # Check for conflict with existing overrides (from design or launcher)
            conflict_indices = [
                i
                for i, existing in enumerate(overrides_list)
                if existing.split("=", 1)[0].strip("'") == cli_key
            ]
            if conflict_indices:
                for idx in conflict_indices:
                    old_val = overrides_list[idx]
                    print(
                        f"WARNING: CLI override '{arg}' replaces design value '{old_val}'. "
                        f"Remove the CLI arg if you want the experiment design value."
                    )
                # Remove earlier duplicate(s) so the override list is clean
                for idx in sorted(conflict_indices, reverse=True):
                    overrides_list.pop(idx)
            overrides_list.append(arg)

    overrides_str = " ".join(overrides_list)

    # Handle explicit hosts for multi-node interactive
    explicit_hosts_export = ""
    if args.hosts:
        explicit_hosts_export = f'export EXPLICIT_HOSTS="{args.hosts}"\n'
        print(f"Multi-node interactive: Using explicit hosts: {args.hosts}")

    # Pick the Accelerate config file and assemble the DeepSpeed env-export
    # block in Python so the heredoc below stays readable.
    if args.deepspeed is not None:
        accelerate_config_name = f"deepspeed_zero{args.deepspeed}.yaml"
        deepspeed_export_lines = [
            f"export DEEPSPEED_ZERO_STAGE={args.deepspeed}",
            "export ACCELERATE_USE_DEEPSPEED=true",
            f"export ACCELERATE_DEEPSPEED_ZERO_STAGE={args.deepspeed}",
        ]
        if args.deepspeed == 3:
            deepspeed_export_lines.append(
                "export ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL=true"
            )
        deepspeed_exports = "\n".join(deepspeed_export_lines)
    else:
        accelerate_config_name = "aurora_ddp.yaml"
        deepspeed_exports = ""
    accelerate_config_path = (
        f"{prism_dir}/scripts/accelerate_configs/{accelerate_config_name}"
    )

    # Build the venv extraction + activation blocks. Branched on
    # --use-shared-venv so we either source the venv on /flare directly or
    # extract the tarball to /tmp on every node (the original behavior).
    if args.use_shared_venv:
        env_unpack_block = (
            "# --- Environment Setup (Shared Venv) ---\n"
            "# Using shared venv from /flare; no per-node tarball extraction.\n"
            f'echo "Using shared venv: {shared_venv_path}"'
        )
        # CRITICAL: do NOT sed the shared activate file — mutating it would
        # corrupt other users' jobs sharing the same venv. Source directly.
        activate_block = f'source "{shared_venv_path}/bin/activate"'
        hf_cache_block = (
            'export HF_HOME="/tmp/huggingface"\n'
            'export TRANSFORMERS_CACHE="/tmp/huggingface/hub"\n'
            'export HF_HUB_CACHE="/tmp/huggingface/hub"\n'
            'export HF_DATASETS_CACHE="/tmp/huggingface/datasets"'
        )
    else:
        env_unpack_block = (
            "# --- Environment Unpack ---\n"
            "# Extract venv from Lustre tarball to local /tmp (each node does this independently)\n"
            f'export ENV_TARBALL="{os.path.abspath(args.packed_env)}"\n'
            'export LOCAL_ENV="/tmp/deepspeed_env"\n'
            'export MARKER_FILE="$LOCAL_ENV/.env_ready_$LOCAL_RANK"\n'
            "\n"
            "# Only rank 0 checks and performs extraction; other ranks just wait for marker file\n"
            'if [ "$LOCAL_RANK" == "0" ]; then\n'
            "    # Re-extract if: no marker exists, OR tarball is newer than the marker (env was repacked)\n"
            '    if [ ! -f "$MARKER_FILE" ] || [ "$ENV_TARBALL" -nt "$MARKER_FILE" ]; then\n'
            '        if [ -f "$MARKER_FILE" ]; then\n'
            '            echo "Rank $RANK: Tarball is newer than cached env, re-extracting..."\n'
            "        fi\n"
            '        echo "Rank $RANK (local 0): Extracting venv to $LOCAL_ENV..."\n'
            "        # Clean stale env to avoid leftover files from previous packs\n"
            "        rm -rf $LOCAL_ENV\n"
            "        mkdir -p $LOCAL_ENV\n"
            "        tar -xzf $ENV_TARBALL -C $LOCAL_ENV\n"
            "        # Signal to other local ranks that extraction is complete\n"
            "        for i in $(seq 0 $((LOCAL_WORLD_SIZE - 1))); do\n"
            '            touch "$LOCAL_ENV/.env_ready_$i"\n'
            "        done\n"
            '        echo "Rank $RANK: Venv extraction complete"\n'
            "    fi\n"
            "fi\n"
            "# Wait for local rank 0 to finish extraction\n"
            'while [ ! -f "$MARKER_FILE" ]; do sleep 1; done'
        )
        activate_block = (
            "# Patch VIRTUAL_ENV in activate script to point to /tmp extraction (not Lustre)\n"
            'sed -i "s|VIRTUAL_ENV=.*|VIRTUAL_ENV=\\"$LOCAL_ENV/.venv-deepspeed\\"|" '
            "$LOCAL_ENV/.venv-deepspeed/bin/activate\n"
            "source $LOCAL_ENV/.venv-deepspeed/bin/activate"
        )
        hf_cache_block = (
            'export HF_HOME="/tmp/huggingface"\n'
            'export TRANSFORMERS_CACHE="/tmp/huggingface/hub"\n'
            'export HF_HUB_CACHE="/tmp/huggingface/hub"\n'
            'export HF_DATASETS_CACHE="/tmp/huggingface/datasets"'
        )

    # Main Command with DAOS
    cmd = textwrap.dedent(f"""
        cd {prism_dir}

        # --- Environment Variables ---
        export ACCELERATE_CONFIG_FILE={accelerate_config_path}
{deepspeed_exports}

# --- Multi-Node Interactive Support ---
{explicit_hosts_export}

# --- CCL/Multi-Node Configuration (Aurora Best Practices) ---
export ZE_FLAT_DEVICE_HIERARCHY={"COMPOSITE" if args.composite else "FLAT"}
export MPICH_GPU_SUPPORT_ENABLED=1
# CRITICAL: Use 'none' launcher and 'ofi' transport to avoid MPI re-initialization inside mpiexec
# Using pmix/mpi causes "Fatal error in internal_Init_thread: Other MPI error"
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0

# === CCL SCALING OPTIMIZATIONS (Critical for 7B+ models) ===
# Worker count: MUST be 1 for multi-node.
# CCL_WORKER_COUNT=4 causes a 48x AllGather bandwidth regression on multi-node
# (111 GiB/s → 2.3 GiB/s). Root-caused in torchtune/GRPO benchmarks Apr 2026.
# NOTE: CCL_WORKER_COUNT=8 causes pthread_create error 22 (EINVAL)
export CCL_WORKER_COUNT=1
# ring is safe for AllReduce (no regression)
export CCL_ALLREDUCE=ring
# CCL_REDUCE_SCATTER=ring REMOVED: causes 63x ReduceScatter regression on multi-node
# mpiexec (138 GiB/s → 1.9 GiB/s). CCL default algorithm runs at full speed.
# Root-caused in torchtune/GRPO benchmarks Apr 2026.
# Enable chunking for large messages (16MB chunks)
export CCL_CHUNK_SIZE=16777216
# Slingshot/CXI network optimizations
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072

# --- General Settings ---
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS={args.cpus_per_task}
export OMP_NUM_THREADS={args.cpus_per_task}
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
{"export PYTHONWARNINGS=ignore" if args.suppress_warnings else ""}
{"# Using Accelerate (user requested)" if args.use_accelerate else "export USE_NATIVE_DDP=1"}
{"export DEBUG_NO_SYNC=1" if args.debug_no_sync else ""}
{"export NO_PIL4DFS=1" if args.no_pil4dfs else ""}
{"export PRISM_DDP_FIND_UNUSED=1" if args.find_unused_params else ""}
{f'export CALIBRATION_JSON="{args.calibration_json}"' if args.calibration_json is not None else ""}
{f'export RUNTIME_FLOPS_PER_STEP="{args.runtime_flops_per_step:.6e}"' if args.runtime_flops_per_step is not None else ""}
export TMPDIR=/tmp

# --- Sequence Length Bucketing (improves throughput for mixed-length datasets) ---
{"export USE_BUCKETING=1" if args.use_bucketing else "# Bucketing disabled"}
{"export BUCKET_BUFFER_SIZE=" + str(args.bucket_buffer_size) if args.use_bucketing else ""}
{"export BUCKET_NUM_BUCKETS=" + str(args.bucket_num_buckets) if args.use_bucketing else ""}
export USE_BUCKETED_COLLATOR={"1" if args.use_bucketed_collator else "0"}
export MAX_SEQ_LENGTH={args.max_seq_length}
{"export ENABLE_ALL_MODALITIES=1" if args.enable_all_modalities else ""}

# --- Profiling & Debugging ---
{"export ENABLE_PROFILER=1" if args.enable_profiler else ""}
{"export PROFILER_STEPS=" + args.profiler_steps if args.enable_profiler else ""}
{"export CCL_LOG_LEVEL=info" if args.ccl_debug else ""}
{"export CCL_SCHED_DUMP=1" if args.ccl_debug else ""}
{"export D_LOG_MASK=WARN" if args.daos_debug else ""}
{"export DAOS_DEBUG=1" if args.daos_debug else ""}
{"export PER_RANK_TIMING=1" if args.per_rank_timing else ""}
{"export DDP_DEBUG=1" if args.ddp_debug else ""}
{"export FSDP_PRODUCTION_MODE=1" if args.fsdp_production_mode or args.benchmark_mode else ""}
{"export PRISM_PRODUCTION_MODE=1" if args.fsdp_production_mode or args.benchmark_mode else ""}
{"export STEP_WATCHDOG_TIMEOUT=" + str(args.step_watchdog_timeout) if args.step_watchdog_timeout > 0 else ""}

# === Distributed Strategy Configuration ===
export DIST_STRATEGY="{args.dist_strategy}"  # ddp, fsdp, or hsdp
export FSDP_SHARDING="{args.fsdp_sharding}"  # full_shard, shard_grad_op, etc.
{"export FSDP_CPU_OFFLOAD=1" if args.fsdp_cpu_offload else ""}
export DDP_BUCKET_CAP_MB={args.ddp_bucket_mb}
{"export TORCH_COMPILE=1" if args.torch_compile else ""}
export GRAD_CKPT_FREQ={args.grad_ckpt_freq}
# HSDP perf knobs (see scaling-study/investigation/REPORT.md)
{"export FSDP_NO_SYNC_ACCUM=1" if args.fsdp_no_sync_accum else ""}
{"export PRISM_DISABLE_PERF_PROBES=1" if args.prism_disable_perf_probes else ""}
export GRAD_NORM_INTERVAL={args.grad_norm_interval}

# --- DAOS Configuration ---
export DAOS_POOL="{daos_pool}"
export DAOS_CONT="{daos_container}"
export DAOS_MODELS_CONT="{daos_models_container}"
export DAOS_MOUNT="{daos_mount}"
export DAOS_MODELS_MOUNT="{daos_models_mount}"
export NUM_NODES={args.nodes}

# --- Multi-Dataset Configuration ---
export USE_MULTI_DATASET={"1" if use_multi_dataset else "0"}
export DATASET_GROUPS="{dataset_groups}"
export DATASET_CONFIG="{prism_dir}/{dataset_config}"
export DATASET_PROPORTIONS="{dataset_proportions}"
export WEBDATASET_RESAMPLED={"0" if args.finite_webdataset else "1"}
{"" if use_multi_dataset else f'export DAOS_SHARDS_PATH="{daos_shards_path}"'}
{"" if use_multi_dataset else f'export DAOS_VAL_PATH="{daos_val_path}"'}

# --- Master Addr/Port (Use HSN for high-speed Slingshot network) ---
# Support explicit --hosts for multi-node interactive jobs (no PBS_NODEFILE)
if [ -n "$EXPLICIT_HOSTS" ]; then
    # Use explicit hosts provided via --hosts argument
    MASTER_HOST=$(echo "$EXPLICIT_HOSTS" | cut -d',' -f1)
    echo "Using explicit hosts: $EXPLICIT_HOSTS"
elif [ ! -z "$PBS_NODEFILE" ] && [ -f "$PBS_NODEFILE" ]; then
    MASTER_HOST=$(head -n 1 $PBS_NODEFILE)
else
    MASTER_HOST=$(hostname)
fi
# CRITICAL: Use HSN interface for multi-node communication (200+ Gb/s vs 1 Gb/s Ethernet)
# Strip any existing domain suffix before adding HSN domain (avoid double-suffix)
MASTER_HOST_BASE=$(echo "$MASTER_HOST" | cut -d'.' -f1)
export MASTER_ADDR="${{MASTER_HOST_BASE}}.hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=$((20000 + RANDOM % 20000))

echo "Master: $MASTER_ADDR:$MASTER_PORT (HSN)"
echo "Output Dir: {output_dir}"
echo "DAOS Pool: $DAOS_POOL"
echo "  Data Container: $DAOS_CONT -> $DAOS_MOUNT"
echo "  Models Container: $DAOS_MODELS_CONT -> $DAOS_MODELS_MOUNT"
{'echo "Using Multi-Dataset Mode: $DATASET_GROUPS"' if use_multi_dataset else 'echo "DAOS Shards: $DAOS_SHARDS_PATH"'}

# --- Cleanup ---
echo "Cleaning up stale python processes..."
pkill -u $USER -f "python src/train.py" || true
sleep 2

# --- Mount DAOS on all compute nodes ---
# CRITICAL: Mount on local node first (where this script runs), then on remote nodes
echo "Mounting DAOS containers..."

# Get list of hosts
LOCAL_HOST=$(hostname | cut -d'.' -f1)
if [ -n "$EXPLICIT_HOSTS" ]; then
    ALL_HOSTS="$EXPLICIT_HOSTS"
elif [ -f "$PBS_NODEFILE" ]; then
    ALL_HOSTS=$(cat $PBS_NODEFILE | sort -u | tr '\n' ',' | sed 's/,$//')
else
    ALL_HOSTS=$(hostname)
fi
echo "Nodes: $ALL_HOSTS (local: $LOCAL_HOST)"

# Mount locally first (required for verification)
echo "Mounting DAOS on local node ($LOCAL_HOST)..."

# Mount data container locally
if mount | grep -q "dfuse.*$DAOS_MOUNT"; then
    echo "  [$LOCAL_HOST] Data already mounted"
else
    mkdir -p $DAOS_MOUNT
    dfuse -m $DAOS_MOUNT --pool $DAOS_POOL --cont $DAOS_CONT --disable-wb-cache &
    DFUSE_DATA_PID=$!
fi

# Mount models container locally
if mount | grep -q "dfuse.*$DAOS_MODELS_MOUNT"; then
    echo "  [$LOCAL_HOST] Models already mounted"
else
    mkdir -p $DAOS_MODELS_MOUNT
    dfuse -m $DAOS_MODELS_MOUNT --pool $DAOS_POOL --cont $DAOS_MODELS_CONT --disable-wb-cache &
    DFUSE_MODELS_PID=$!
fi

# Wait for local mounts to complete
sleep 5

# Verify local data mount (required)
if [ ! -d "$DAOS_MOUNT" ] || ! ls "$DAOS_MOUNT" &>/dev/null || [ -z "$(ls -A $DAOS_MOUNT 2>/dev/null)" ]; then
    echo "ERROR: DAOS data mount at $DAOS_MOUNT is empty or not mounted!"
    echo "Make sure your job was submitted with: -l filesystems=flare:home:daos_user_fs"
    exit 1
fi
echo "  [$LOCAL_HOST] Data mount OK: $(ls $DAOS_MOUNT | head -3 | tr '\n' ' ')"

# Check local models mount (optional)
if [ -d "$DAOS_MODELS_MOUNT" ] && ls "$DAOS_MODELS_MOUNT" &>/dev/null && [ -n "$(ls -A $DAOS_MODELS_MOUNT 2>/dev/null)" ]; then
    echo "  [$LOCAL_HOST] Models mount OK"
    export DAOS_MODELS_AVAILABLE=1
else
    echo "  [$LOCAL_HOST] Models mount not available - will fall back to /flare staging"
    export DAOS_MODELS_AVAILABLE=0
fi

# Mount on other nodes (in parallel via SSH)
for HOST in $(echo "$ALL_HOSTS" | tr ',' ' '); do
    HOST_BASE=$(echo "$HOST" | cut -d'.' -f1)
    if [ "$HOST_BASE" != "$LOCAL_HOST" ]; then
        echo "Mounting DAOS on $HOST_BASE..."
        ssh $HOST "module use /soft/modulefiles 2>/dev/null; module load daos 2>/dev/null; \
            mkdir -p $DAOS_MOUNT $DAOS_MODELS_MOUNT; \
            mount | grep -q \"dfuse.*$DAOS_MOUNT\" || dfuse -m $DAOS_MOUNT --pool $DAOS_POOL --cont $DAOS_CONT --disable-wb-cache 2>/dev/null & \
            mount | grep -q \"dfuse.*$DAOS_MODELS_MOUNT\" || dfuse -m $DAOS_MODELS_MOUNT --pool $DAOS_POOL --cont $DAOS_MODELS_CONT --disable-wb-cache 2>/dev/null & \
            sleep 3; \
            if ls $DAOS_MOUNT &>/dev/null; then echo \"  [$HOST_BASE] Data mount OK\"; else echo \"  [$HOST_BASE] Data mount FAILED\"; fi" &
    fi
done
wait
sleep 2

echo "DAOS data container mounted at $DAOS_MOUNT"
echo "DAOS models available: $DAOS_MODELS_AVAILABLE"

# --- Create hostfile for multi-node interactive jobs ---
HOSTFILE_ARG=""
if [ -n "$EXPLICIT_HOSTS" ]; then
    # Create hostfile from explicit hosts list
    HOSTFILE="/tmp/prism_hostfile_$$"
    echo "$EXPLICIT_HOSTS" | tr ',' '\n' > "$HOSTFILE"
    HOSTFILE_ARG="--hostfile $HOSTFILE"
    echo "Created hostfile: $HOSTFILE"
    cat "$HOSTFILE"
fi

# List contents with timeout to detect hung DAOS mounts
echo "Verifying DAOS mount contents (30s timeout)..."
if ! timeout 30 ls -la $DAOS_MOUNT/; then
    echo "ERROR: DAOS mount appears hung - ls timed out after 30s"
    echo "This may indicate DAOS connectivity issues with this node"
    fusermount3 -u $DAOS_MOUNT 2>/dev/null || true
    exit 1
fi
echo "DAOS mount verified successfully"
if [ "$USE_MULTI_DATASET" == "1" ]; then
    echo "Multi-dataset mode - listing dataset groups:"
    for group in pixmo s1mmalign cosyn nemotron; do
        if [ -d "$DAOS_MOUNT/$group" ]; then
            echo "  $group: $(ls $DAOS_MOUNT/$group | wc -l) datasets"
        fi
    done
else
    ls $DAOS_SHARDS_PATH/ | head -5
fi

# --- Execution ---
# CRITICAL: --no-vni is REQUIRED for DAOS compatibility (prevents NA_HOSTUNREACH errors).
# Without it, DAOS RPCs over libfabric fail with NA_HOSTUNREACH and the job hangs.
mpiexec $HOSTFILE_ARG -n {args.nodes * ngpus} -ppn {ngpus} --no-vni --cpu-bind depth --depth {args.cpus_per_task} bash -lc '
# CRITICAL: Load modules BEFORE activating venv
# - frameworks: Ensures IPEX and XPU libraries are properly configured
#   Without this, importing intel_extension_for_pytorch crashes with std::bad_alloc
# - daos: Sets DAOS_AGENT_DRPC_DIR for proper DAOS client communication
# Re-export ZE_FLAT_DEVICE_HIERARCHY inside mpiexec worker.
# bash -lc may reset env on some systems; explicit re-export ensures correctness.
export ZE_FLAT_DEVICE_HIERARCHY={"COMPOSITE" if args.composite else "FLAT"}

module use /soft/modulefiles
# Pinned to 2025.3.1: matches what the shared venv was built against and what
# the XCCL pre-flight gate validated 2026-05-23 (24-rank allreduce passed
# cleanly, the April 2026 ccl_check_usm_pointers regression is gone).
# No 2>/dev/null: if the pin is removed/renamed we want a loud failure here,
# not cryptic import errors hundreds of lines downstream.
module load frameworks/2025.3.1
module load daos 2>/dev/null
# NOTE: the LD_LIBRARY_PATH workaround for py3.10 h5py is no longer needed on
# 2025.3.1 (py3.12); the system h5py is path-correct.

# CRITICAL: Re-export CCL settings AFTER module loads.
# `module load frameworks` overrides CCL_PROCESS_LAUNCHER to pmix (its default),
# clobbering our "none" setting. Re-exporting here ensures our values take effect.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
# Disable deprecated hostname sharing (HSDP init_device_mesh enables it internally;
# this prevents the CCL_OFI_ENABLE_HOSTNAME_SHARING deprecation warning)
export CCL_OFI_ENABLE_HOSTNAME_SHARING=0

# Get distributed environment variables with fallback chains
# CRITICAL: For multi-node, WORLD_SIZE must be total ranks across all nodes
# NUM_NODES is passed from the job script, LOCAL_WORLD_SIZE is ranks per node
export LOCAL_WORLD_SIZE=${{PMI_LOCAL_SIZE:-${{PMIX_LOCAL_SIZE:-${{PALS_LOCAL_SIZE:-{ngpus}}}}}}}
export NUM_NODES=${{NUM_NODES:-1}}

# Calculate WORLD_SIZE: prefer PMI_SIZE if set, otherwise compute from nodes * ppn
if [ -n "$PMI_SIZE" ] && [ "$PMI_SIZE" -gt 0 ] 2>/dev/null; then
    export WORLD_SIZE=$PMI_SIZE
elif [ -n "$PALS_SIZE" ] && [ "$PALS_SIZE" -gt 0 ] 2>/dev/null; then
    export WORLD_SIZE=$PALS_SIZE
else
    # Compute from NUM_NODES * LOCAL_WORLD_SIZE
    export WORLD_SIZE=$((NUM_NODES * LOCAL_WORLD_SIZE))
fi

export RANK=${{PMI_RANK:-${{PMIX_RANK:-${{PALS_RANKID:-0}}}}}}
export LOCAL_RANK=${{PMI_LOCAL_RANK:-${{PMIX_LOCAL_RANK:-${{PALS_LOCAL_RANKID:-0}}}}}}

# Set GPU affinity - each rank sees only its assigned GPU tile as xpu:0
export ZE_AFFINITY_MASK=$LOCAL_RANK

NODE_RANK=$((RANK / LOCAL_WORLD_SIZE))
echo "DEBUG: Rank=$RANK, Local=$LOCAL_RANK, NodeRank=$NODE_RANK, World=$WORLD_SIZE, NumNodes=$NUM_NODES"

{env_unpack_block}

# --- Model Staging ---
# DAOS models container provides fast multi-node access without per-node copying
# Fall back to /flare staging only if DAOS models container not available
# DAOS models are stored in hub/ subdirectory for HuggingFace compatibility
export DAOS_MODELS_HUB="$DAOS_MODELS_MOUNT/hub"
HF_FALLBACK_DIRS=({" ".join(f'"{d.strip()}"' for d in args.hf_fallback_dirs.split(",") if d.strip())})
export LOCAL_HF_HOME="/tmp/huggingface/hub"
export MODEL_DIR="{hf_model_dir or ""}"
export MODEL_MARKER="$LOCAL_HF_HOME/.model_ready"

MODELS_TO_STAGE=(
{image_model_stage_entries}
    "models--google--tapas-base"
    "models--Salesforce--moirai-2.0-R-small"
    "models--polymathic-ai--walrus"
    "models--HuggingFaceTB--SmolLM2-360M-Instruct"
)
# Only stage backbone if the model dir is a HuggingFace model ID
if [ -n "$MODEL_DIR" ]; then
    MODELS_TO_STAGE=("$MODEL_DIR" "${{MODELS_TO_STAGE[@]}}")
fi

if [ "$LOCAL_RANK" == "0" ]; then
    mkdir -p $LOCAL_HF_HOME
    if [ ! -f "$MODEL_MARKER" ]; then
        if [ "$DAOS_MODELS_AVAILABLE" == "1" ] && [ -d "$DAOS_MODELS_HUB" ]; then
            # DAOS models available - symlink instead of copy (FAST!)
            echo "Rank $RANK (local 0): Linking models from DAOS ($DAOS_MODELS_HUB)..."
            for MODEL in "${{MODELS_TO_STAGE[@]}}"; do
                if [ -d "$DAOS_MODELS_HUB/$MODEL" ] && [ ! -e "$LOCAL_HF_HOME/$MODEL" ]; then
                    echo "  Linking $MODEL from DAOS..."
                    ln -s "$DAOS_MODELS_HUB/$MODEL" "$LOCAL_HF_HOME/$MODEL"
                else
                    # Try case-insensitive match (e.g., OLMo vs Olmo)
                    # NOTE: Use ls+grep instead of find to avoid hangs on DAOS dfuse mounts
                    FOUND_NAME=$(ls "$DAOS_MODELS_HUB" 2>/dev/null | grep -Fix "$MODEL" | head -1)
                    if [ -n "$FOUND_NAME" ] && [ ! -e "$LOCAL_HF_HOME/$MODEL" ]; then
                        echo "  Linking $MODEL from DAOS (case-insensitive: $FOUND_NAME)..."
                        ln -s "$DAOS_MODELS_HUB/$FOUND_NAME" "$LOCAL_HF_HOME/$MODEL"
                    else
                        FOUND_FALLBACK=""
                        for FBDIR in "${{HF_FALLBACK_DIRS[@]}}"; do
                            if [ -d "$FBDIR/$MODEL" ] && [ ! -e "$LOCAL_HF_HOME/$MODEL" ]; then
                                echo "  $MODEL not on DAOS, copying from $FBDIR..."
                                cp -r "$FBDIR/$MODEL" "$LOCAL_HF_HOME/"
                                FOUND_FALLBACK=1
                                break
                            fi
                        done
                        if [ -z "$FOUND_FALLBACK" ] && [ ! -e "$LOCAL_HF_HOME/$MODEL" ]; then
                            echo "  WARNING: $MODEL not found in any source!"
                        fi
                    fi
                fi
            done
            echo "Rank $RANK: Model linking complete (DAOS)"
        else
            # Fall back to copying from /flare
            echo "Rank $RANK (local 0): DAOS models not available, staging from /flare..."
            for MODEL in "${{MODELS_TO_STAGE[@]}}"; do
                FOUND_FALLBACK=""
                for FBDIR in "${{HF_FALLBACK_DIRS[@]}}"; do
                    if [ -d "$FBDIR/$MODEL" ] && [ ! -d "$LOCAL_HF_HOME/$MODEL" ]; then
                        echo "  Copying $MODEL from $FBDIR..."
                        cp -r "$FBDIR/$MODEL" "$LOCAL_HF_HOME/"
                        FOUND_FALLBACK=1
                        break
                    fi
                done
                if [ -z "$FOUND_FALLBACK" ] && [ ! -d "$LOCAL_HF_HOME/$MODEL" ]; then
                    echo "  WARNING: $MODEL not found in any source!"
                fi
            done
            echo "Rank $RANK: Model staging complete (/flare)"
        fi
        touch "$MODEL_MARKER"
    fi
fi
while [ ! -f "$MODEL_MARKER" ]; do sleep 2; done

# --- Activate Environment ---
export PYTHONNOUSERSITE=1
{activate_block}

# --- PRISM_BUILD_INFO manifest check ---
# Confirm the venv frameworks_module + python_realpath match the
# currently-loaded values. Catches the silent-staleness failure mode where
# ALCF flips the default frameworks python out from under a stale venv.
# Runs before any modality import — IPEX errors otherwise dump 40+ lines of
# unparseable C++ symbol noise.
#
# IMPORTANT: in tarball mode, do NOT trust $VIRTUAL_ENV. The outer launch
# script may have run `source .venv-deepspeed/bin/activate` at the top and
# set VIRTUAL_ENV to the repo-root .venv-deepspeed. The sed-then-source
# dance for the /tmp activate inside mpiexec does NOT always re-export
# VIRTUAL_ENV when bash -lc runs a new login shell, so the manifest read
# can hit the stale repo venv. Resolve from $LOCAL_ENV explicitly for
# tarball mode; shared-venv mode keeps $VIRTUAL_ENV (which is set by the
# explicit source on a known shared path).
# NOTE: comments inside this mpiexec heredoc MUST NOT contain apostrophes
# (single quotes). The heredoc is opened with a single quote; any unescaped
# apostrophe terminates the string and the remaining script body executes
# in the outer shell instead of on every rank (see launcher_smoke_harness_bug
# memory entry / PR #33).
if [ -n "${{LOCAL_ENV:-}}" ] && [ -f "$LOCAL_ENV/.venv-deepspeed/PRISM_BUILD_INFO" ]; then
    BUILD_INFO="$LOCAL_ENV/.venv-deepspeed/PRISM_BUILD_INFO"
else
    BUILD_INFO="$VIRTUAL_ENV/PRISM_BUILD_INFO"
fi
if [ -f "$BUILD_INFO" ]; then
    expected_fw=$(grep -E "^frameworks_module:" "$BUILD_INFO" | awk -F": " "{{print \\$2}}")
    expected_py=$(grep -E "^python_realpath:" "$BUILD_INFO" | awk -F": " "{{print \\$2}}")
    actual_fw="frameworks/${{LMOD_FAMILY_FRAMEWORKS_VERSION:-UNKNOWN}}"
    actual_py=$(readlink -f "$(which python)" 2>/dev/null)
    if [ -n "$expected_fw" ] && [ "$expected_fw" != "$actual_fw" ]; then
        echo "ERROR: venv manifest mismatch (frameworks_module)"
        echo "  expected: $expected_fw"
        echo "  actual:   $actual_fw"
        echo "Rebuild venv against currently-loaded $actual_fw."
        exit 1
    fi
    if [ -n "$expected_py" ] && [ "$expected_py" != "$actual_py" ]; then
        echo "ERROR: venv manifest mismatch (python_realpath)"
        echo "  expected: $expected_py"
        echo "  actual:   $actual_py"
        echo "Rebuild venv against currently-loaded $actual_fw."
        exit 1
    fi
fi

export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
{hf_cache_block}

# CRITICAL: Force offline mode to prevent HF Hub metadata operations
# which cause segfaults due to concurrent file access on multi-node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Verify critical dependencies ---
# These should be in the packed env (via setup_deepspeed_env.sh).
# If missing, the env needs to be rebuilt and repacked.
# NOTE: This runs inside bash -lc (single-quoted), so we use a heredoc
# to avoid single-quote nesting issues.
if [ "$LOCAL_RANK" == "0" ]; then
    echo "Verifying packed env dependencies..."
    echo "  python: $(which python)"
    VERIFY_SCRIPT="/tmp/prism_verify_env_$$.py"
    cat > "$VERIFY_SCRIPT" << VERIFY_EOF
import sys
print(f"  prefix: {{sys.prefix}}")
print(f"  sys.path ({{len(sys.path)}} entries):")
for i, p in enumerate(sys.path[:8]):
    print(f"    [{{i}}] {{p}}")
if len(sys.path) > 8:
    print(f"    ... and {{len(sys.path) - 8}} more")

errors = []

try:
    import webdataset
    print(f"  webdataset: {{webdataset.__version__}} ({{webdataset.__file__}})")
except ImportError as e:
    errors.append(f"webdataset: {{e}}")

try:
    import typer
    print(f"  typer: OK")
except ImportError as e:
    errors.append(f"typer: {{e}}")

try:
    import transformers
    from packaging.version import Version
    v = Version(transformers.__version__)
    loc = transformers.__file__
    if v < Version("4.57.6"):
        errors.append(f"transformers {{transformers.__version__}} < 4.57.6 (no OLMo-3 support, loaded from {{loc}})")
    else:
        print(f"  transformers: {{transformers.__version__}} ({{loc}})")
except ImportError as e:
    errors.append(f"transformers: {{e}}")

if errors:
    print()
    for err in errors:
        print(f"  FAILED: {{err}}")
    print()
    print("Rebuild: bash tools/setup_deepspeed_env.sh && tar -czf deepspeed_env.tar.gz .venv-deepspeed")
    sys.exit(1)
print("  All dependency checks passed.")
VERIFY_EOF
    python "$VERIFY_SCRIPT" || exit 1
    rm -f "$VERIFY_SCRIPT"
fi

# --- DAOS Interception Library for kernel-bypass I/O ---
# This significantly improves read performance by bypassing the kernel.
# D_IL_COMPATIBLE=1 is a workaround for Python compatibility issues with pil4dfs.
# NOTE: libpil4dfs can hang FSDP AllGather collectives (DAOS-17499). Use
# --no-pil4dfs (or export NO_PIL4DFS=1) for FSDP jobs.
if [ -z "$NO_PIL4DFS" ]; then
    export D_IL_COMPATIBLE=1
    export LD_PRELOAD=/usr/lib64/libpil4dfs.so
    PIL4DFS_STATUS="enabled"
else
    echo "WARNING: libpil4dfs.so DISABLED (--no-pil4dfs flag set)"
    PIL4DFS_STATUS="DISABLED"
fi

# --- Point WebDataset to DAOS mount ---
if [ "$USE_MULTI_DATASET" == "1" ]; then
    echo "Rank $RANK/$WORLD_SIZE (Node: $NODE_RANK, Local: $LOCAL_RANK/$LOCAL_WORLD_SIZE)"
    echo "  Multi-dataset mode: $DATASET_GROUPS"
    echo "  DAOS mount: $DAOS_MOUNT"
    echo "  Interception library (pil4dfs): $PIL4DFS_STATUS"
else
    # Legacy single-dataset mode
    export WEBDATASET_LOCAL_PATH="$DAOS_SHARDS_PATH"
    export WEBDATASET_VAL_PATH="$DAOS_VAL_PATH"
    echo "Rank $RANK/$WORLD_SIZE (Node: $NODE_RANK, Local: $LOCAL_RANK/$LOCAL_WORLD_SIZE)"
    echo "  Reading shards directly from DAOS: $DAOS_SHARDS_PATH"
    echo "  Interception library (pil4dfs): $PIL4DFS_STATUS"
fi

python src/train.py {overrides_str}
'

# --- Cleanup DAOS mounts ---
echo "Cleaning up DAOS mounts..."
clean-dfuse.sh ${{DAOS_POOL}}:${{DAOS_CONT}} || true
clean-dfuse.sh ${{DAOS_POOL}}:${{DAOS_MODELS_CONT}} || true
""")

    # Generate Script
    logs_dir = os.path.join(os.getcwd(), "logs", design_id)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Logs directory: {logs_dir}")

    if args.batch:
        wt = args.walltime if args.walltime else resources.get("walltime", "01:00:00")
        # IMPORTANT: Request daos_user_fs filesystem
        header = f"""#!/bin/bash -l
#PBS -l select={args.nodes}
#PBS -l walltime={wt}
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -q {args.queue}
#PBS -A {args.project}
#PBS -k doe
#PBS -j oe
#PBS -N {args.id}
#PBS -o {logs_dir}/
#PBS -e {logs_dir}/
"""
    else:
        header = "#!/bin/bash"

    run_script_content = textwrap.dedent(f"""{header}
# Aurora DAOS Launch Script
# Experiment: {design_id}
# Generated by tools/launch_aurora_daos.py
#
# Key differences from launch_aurora_web.py:
#   - Reads shards directly from DAOS (no staging to /tmp)
#   - Uses libpil4dfs.so for kernel-bypass I/O
#   - Requests daos_user_fs in PBS filesystems
#   - Much faster job startup

# --- Module Load ---
module use /soft/modulefiles
module load frameworks/2025.3.1
module load hdf5
module load daos

# --- Activate Venv ---
if [ -d ".venv-deepspeed" ]; then
    source .venv-deepspeed/bin/activate
fi

{proxy_env}

{cmd}
""")

    mode = "batch" if args.batch else "interactive"
    script_name = f"run_aurora_daos_{args.id}_{date_str}_{time_str}_{mode}.sh"
    script_name = script_name.replace("/", "_")

    jobs_folder = "jobs"
    os.makedirs(jobs_folder, exist_ok=True)
    script_path = os.path.join(jobs_folder, script_name)

    with open(script_path, "w") as f:
        f.write(run_script_content)

    os.chmod(script_path, 0o755)
    print(f"Generated Run Script: {script_path}")

    if args.dry_run:
        print("\n--- Script generated (dry run, not executing) ---")
    else:
        if args.batch:
            print("Submitting batch job via qsub...")
            try:
                result = subprocess.run(
                    ["qsub", script_path],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"Job submitted: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"qsub failed: {e.stderr}")
        else:
            # Interactive mode: run directly or via SSH
            if args.run_via_ssh and args.hosts:
                first_host = args.hosts.split(",")[0]
                print(f"Executing via SSH on {first_host}...")
                # Forward PBS_JOBID so multi-node mpiexec can attach to the PALS shepherd.
                # Without it, PALS sees "job NONE" on the remote node and fails (exit 127).
                pbs_jobid = args.pbs_jobid or os.environ.get("PBS_JOBID", "")
                if not pbs_jobid and args.nodes > 1:
                    print(
                        "ERROR: --pbs-jobid not set for multi-node SSH launch. "
                        "mpiexec will see 'job NONE' on the remote node and fail "
                        "with exit 127. Pass --pbs-jobid <id> (find it via "
                        "'qstat -u $USER') and re-run."
                    )
                    sys.exit(2)
                # shlex.quote on the value protects against jobids containing
                # whitespace or shell metacharacters (defensive — real PBS jobids
                # are dot-separated alnum, but the flag is operator-supplied).
                pbs_env = (
                    f"PBS_JOBID={shlex.quote(pbs_jobid)} " if pbs_jobid else ""
                )
                ssh_cmd = f"ssh {first_host} 'cd {os.getcwd()} && {pbs_env}bash {script_path}'"
                # Echo the full command before invoking — this path has a long
                # history of silent failures (PALS jobid, libpil4dfs, apostrophe
                # quoting), and the printed command is the fastest debug aid.
                print(f"  ssh_cmd: {ssh_cmd}")
                try:
                    subprocess.run(ssh_cmd, shell=True, check=True)
                except KeyboardInterrupt:
                    print("\nExecution Interrupted by User.")
                except subprocess.CalledProcessError as e:
                    print(f"\nExecution Failed with exit code {e.returncode}.")
            else:
                print("Executing...")
                try:
                    subprocess.run([f"./{script_path}"], check=True)
                except KeyboardInterrupt:
                    print("\nExecution Interrupted by User.")
                except subprocess.CalledProcessError as e:
                    print(f"\nExecution Failed with exit code {e.returncode}.")


if __name__ == "__main__":
    main()

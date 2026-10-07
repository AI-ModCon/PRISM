#!/usr/bin/env python3
"""
launch_aurora_web.py - Launch PRISM with WebDataset shard staging

Extends launch_aurora.py with:
- WebDataset manifest-based shard staging to /tmp
- Round-robin shard distribution across nodes
- Automatic config override to use local shards

Usage:
    python tools/launch_aurora_web.py \
        --id PRISM-AURORA-ZONE-A \
        --packed-env deepspeed_env.tar.gz \
        --webdataset-dir /flare/ModCon/ngetty/data/zone_a/pixmo_cap_webdataset \
        --nodes 2
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import textwrap

import yaml

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
    parser = argparse.ArgumentParser(description="Launch PRISM with WebDataset support on Aurora")
    parser.add_argument(
        "--file",
        default="experiments/prism_designs.yaml",
        help="Experiment Design YAML file",
    )
    parser.add_argument("--id", required=True, help="Run ID / Job Name")
    parser.add_argument(
        "--design", help="Experiment Design ID (from YAML). Defaults to same as --id"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print script instead of executing")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--batch", action="store_true", help="Generate Batch Script (PBS)"
    )
    parser.add_argument("--project", default="ModCon", help="Project Allocation")
    parser.add_argument("--queue", default="debug", help="Queue Name")
    parser.add_argument("--walltime", default=None, help="Walltime (overrides config)")
    parser.add_argument(
        "--packed-env",
        default="deepspeed_env.tar.gz",
        help="Path to packed env tarball",
    )
    parser.add_argument(
        "--prism-dir",
        default=None,
        help="Path to PRISM checkout (default: $PRISM_DIR from .env, else cwd). "
        "Overrides .env PRISM_DIR.",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="HuggingFace cache root (default: $HF_HOME from .env, else $HOME/.cache/huggingface). "
        "Overrides .env HF_HOME.",
    )
    parser.add_argument(
        "--shared-hf-home",
        default=None,
        help="Shared HuggingFace hub on /flare to stage models from (default: $SHARED_HF_HOME from .env, "
        "else $HF_HOME/hub). Overrides .env SHARED_HF_HOME.",
    )

    # WebDataset-specific args
    parser.add_argument(
        "--webdataset-dir",
        default=None,
        help="Path to WebDataset directory (contains shards/ and manifest.json). "
        "Required for WebDataset-backed designs; optional for designs that read "
        "data from a different path (e.g. VLA → training.calvin_root).",
    )
    parser.add_argument(
        "--dataset-groups",
        default=None,
        help="Use MultiWebDataset directly from a Lustre root instead of staging a "
        "single --webdataset-dir. Example: --dataset-groups pixmo.",
    )
    parser.add_argument(
        "--dataset-config",
        default="src/conf/data/lustre_datasets.yaml",
        help="Dataset configuration YAML for --dataset-groups mode.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root directory for --dataset-groups mode. Defaults to dastr.mount_base "
        "or daos.mount_base from --dataset-config.",
    )
    parser.add_argument(
        "--finite-webdataset",
        action="store_true",
        help="Disable WebDataset resampling in --dataset-groups mode. Each iterator "
        "consumes assigned shards once; the trainer can restart at finite epoch boundaries.",
    )
    parser.add_argument(
        "--stage-dataset-groups-local",
        action="store_true",
        help="For --dataset-groups, stage a sampled subset of group shards to "
        "--local-shards-dir and train from a mirrored multi-dataset tree.",
    )
    parser.add_argument(
        "--stage-max-shards-per-node",
        type=int,
        default=0,
        help="Optional final cap on total dataset-group shards staged per node. "
        "Default 0 means no total cap.",
    )
    parser.add_argument(
        "--stage-shards-per-dataset-per-node",
        type=int,
        default=12,
        help="Maximum shards to stage per active dataset per node when "
        "--stage-dataset-groups-local is set. Default 12 gives each local "
        "rank one shard for datasets with enough shards.",
    )
    parser.add_argument(
        "--local-shards-dir",
        default="/tmp/webdataset",
        help="Local directory on compute node for staged shards",
    )
    parser.add_argument(
        "--webdataset-modality",
        default="image",
        help="Modality of data in --webdataset-dir (e.g. 'image', 'time_series'). "
        "Sets WEBDATASET_LOCAL_MODALITY so the staged shards override applies to "
        "the correct modality stream. Default: image.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=16,
        help="CPUs per rank for DataLoader workers (like VJEPA2)",
    )

    # DDP mode - Native DDP is default (better scaling on Aurora)
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        help="Use Accelerate instead of native DDP (for DeepSpeed/ZeRO features)",
    )
    parser.add_argument(
        "--debug-no-sync",
        action="store_true",
        help="Run backward WITHOUT DDP AllReduce to measure pure compute time",
    )
    parser.add_argument(
        "--use-shared-venv",
        action="store_true",
        help="Use shared venv from VENV_PATH in .env instead of unpacking tarball. Requires VENV_PATH to be set.",
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
        "--grad-ckpt-freq",
        type=int,
        default=1,
        help="Gradient checkpoint frequency: 1=every layer (default), 2=every other, 0=disabled",
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

    # === Sequence Length & Bucketing ===
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization (default: 2048, Molmo uses 2560)",
    )
    parser.add_argument(
        "--webdataset-shuffle-buffer",
        type=int,
        default=32768,
        help="Sample-level WebDataset shuffle buffer for multi-dataset streams.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps. When omitted, the Hydra/design value is used. "
        "Useful for short smoke runs (e.g. --max-steps 50).",
    )
    # IsoFLOP env-var-only path (mirrors launch_aurora_daos.py:460). Stage A
    # uses launch_aurora_web.py for Lustre runs; without these the trainer's
    # _FlopCounter has no calibration handle and cumulative_flops/d_*_tokens
    # show up empty in perf.jsonl. No --target-flops here on purpose — the
    # supported orchestration is tools/isoflop_launch.py passing pre-rescaled
    # --max-steps + --runtime-flops-per-step from the plan.
    parser.add_argument(
        "--calibration-json",
        type=str,
        default=None,
        help="Path to a JSON written by tools/isoflop_calibrate.py. When set, "
        "exported as CALIBRATION_JSON in the qsub script so the trainer's "
        "_FlopCounter can attribute cumulative FLOPs in perf.jsonl. "
        "Validated at submit time so a typo doesn't reach the trainer silently.",
    )
    parser.add_argument(
        "--runtime-flops-per-step",
        type=float,
        default=None,
        help="Plan's runtime-rescaled per-step FLOPs (from isoflop_plan.py's "
        "rescale_factor). Exported as RUNTIME_FLOPS_PER_STEP so the _FlopCounter "
        "accumulates in rescaled FLOPs and cumulative_flops matches the plan's "
        "budget_flops. Mirrors launch_aurora_daos.py.",
    )
    parser.add_argument(
        "--enable-all-modalities",
        action="store_true",
        help="Set ENABLE_ALL_MODALITIES=1 in the training env. Activates all "
        "skip=true datasets whose modality is in model.modalities and falls "
        "back to dummy tensors when real shards are missing. Off by default.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory for full native resume: model, optimizer, "
        "scheduler, W&B run id, and deterministic finite-dataloader replay.",
    )
    parser.add_argument(
        "--resume-weights-only",
        type=str,
        default=None,
        help="Path to checkpoint directory to load model weights from with a fresh optimizer.",
    )
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
        default=True,
        help="Sort samples by length within each batch to reduce padding (default: on)",
    )
    parser.add_argument(
        "--no-bucketed-collator",
        action="store_false",
        dest="use_bucketed_collator",
        help="Disable within-batch sorting by length",
    )

    # Logging fixes
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="WandB project name (overrides config default)",
    )
    parser.add_argument(
        "--suppress-warnings",
        action="store_true",
        default=True,
        help="Suppress Python warnings to clean up logs (default: True)",
    )

    args, unknown_args = parser.parse_known_args()
    cli_overrides = {}
    for arg in unknown_args:
        if "=" in arg and not arg.startswith("--"):
            key, value = arg.split("=", 1)
            cli_overrides[key.strip()] = value.strip()

    if args.use_accelerate and args.dist_strategy != "ddp":
        print("Warning: --dist-strategy is ignored when --use-accelerate is enabled.")

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error(f"--max-steps must be positive, got {args.max_steps}")
    if args.webdataset_shuffle_buffer <= 0:
        parser.error(
            f"--webdataset-shuffle-buffer must be positive, got {args.webdataset_shuffle_buffer}"
        )

    if args.runtime_flops_per_step is not None and args.runtime_flops_per_step <= 0:
        parser.error(
            f"--runtime-flops-per-step must be positive, got {args.runtime_flops_per_step}"
        )
    if args.resume_from_checkpoint and args.resume_weights_only:
        parser.error(
            "--resume-from-checkpoint and --resume-weights-only are mutually exclusive"
        )

    # IsoFLOP calibration JSON validation (env-var-only path; see arg help).
    # Canonicalize the path so the env-var export in the qsub heredoc is
    # absolute (compute-node cwd may differ from UAN cwd).
    if args.calibration_json is not None:
        cal_path = os.path.abspath(args.calibration_json)
        if not os.path.isfile(cal_path):
            parser.error(f"--calibration-json not found: {cal_path}")
        try:
            with open(cal_path) as _cf:
                _ = json.load(_cf)
        except (OSError, json.JSONDecodeError) as _e:
            parser.error(f"--calibration-json unreadable: {cal_path}: {_e}")
        args.calibration_json = cal_path

    use_multi_dataset = args.dataset_groups is not None
    if use_multi_dataset and args.webdataset_dir is not None:
        parser.error("--dataset-groups and --webdataset-dir are mutually exclusive")
    if args.stage_dataset_groups_local and not use_multi_dataset:
        parser.error("--stage-dataset-groups-local requires --dataset-groups")
    train_use_multi_dataset = use_multi_dataset

    dataset_config_abs = ""
    dataset_root = ""
    train_dataset_root = ""
    if use_multi_dataset:
        dataset_config_abs = os.path.abspath(args.dataset_config)
        if not os.path.isfile(dataset_config_abs):
            parser.error(f"--dataset-config not found: {dataset_config_abs}")
        with open(dataset_config_abs) as f:
            dataset_config_data = yaml.safe_load(f) or {}
        dataset_root = (
            args.dataset_root
            or dataset_config_data.get("dastr", {}).get("mount_base")
            or dataset_config_data.get("daos", {}).get("mount_base")
        )
        if not dataset_root:
            parser.error(
                "--dataset-root is required when --dataset-config has no "
                "dastr.mount_base or daos.mount_base"
            )
        dataset_root = os.path.abspath(os.path.expandvars(dataset_root))
        if not os.path.isdir(dataset_root):
            parser.error(f"--dataset-root not found: {dataset_root}")
        train_dataset_root = (
            args.local_shards_dir if args.stage_dataset_groups_local else dataset_root
        )
        mode = (
            "dataset-group node-local staging mode"
            if args.stage_dataset_groups_local
            else "Lustre multi-dataset mode"
        )
        print(f"WebDataset: {mode}")
        print(f"  Groups: {args.dataset_groups}")
        print(f"  Config: {dataset_config_abs}")
        print(f"  Root: {dataset_root}")
        if args.stage_dataset_groups_local:
            print(f"  Train root: {train_dataset_root}")

    # 1. Validate WebDataset directory (skipped when --webdataset-dir is omitted,
    # e.g. for VLA designs that read CALVIN from training.calvin_root instead)
    if args.webdataset_dir is not None:
        manifest_path = os.path.join(args.webdataset_dir, "manifest.json")
        shards_dir = os.path.join(args.webdataset_dir, "shards")

        if not os.path.exists(manifest_path):
            print(f"Error: manifest.json not found at {manifest_path}")
            sys.exit(1)
        if not os.path.isdir(shards_dir):
            print(f"Error: shards/ directory not found at {shards_dir}")
            sys.exit(1)

        with open(manifest_path) as f:
            manifest = json.load(f)

        num_shards = manifest["num_shards"]
        print(f"WebDataset: {num_shards} shards, {manifest.get('total_written', '?')} samples")
        print(f"Distribution: {num_shards // args.nodes} shards/node (round-robin)")

        # Held-out split, if the dataset ships one (SciTS does: val_shards/).
        _vsd = os.path.join(args.webdataset_dir, "val_shards")
        if os.path.isdir(_vsd):
            _nval = len([f for f in os.listdir(_vsd) if f.endswith(".tar")])
            val_shards_dir = _vsd
            print(f"Validation: {_nval} shard(s) at {_vsd}")
        else:
            val_shards_dir = ""
            print(f"Validation: no val_shards/ under {args.webdataset_dir} — val/loss will be unavailable")
    else:
        manifest_path = ""
        val_shards_dir = ""
        shards_dir = ""
        if not use_multi_dataset:
            print("WebDataset: --webdataset-dir not provided; shard staging will be skipped.")

    # 2. Output Management
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "outputs", args.id, date_str, time_str)

    env_config = load_dotenv()

    # Defaults from .env (with fallbacks). CLI flags resolved later (need args).
    default_venv_path = env_config.get("VENV_PATH") or os.environ.get("VENV_PATH", "")

    # Validate --use-shared-venv
    if args.use_shared_venv and not default_venv_path:
        print("Error: --use-shared-venv requires VENV_PATH to be set in .env or environment")
        sys.exit(1)

    # 3. Load Experiment Spec
    design_id = args.design if args.design else args.id
    target_exp, _parent_exp, final_overrides = lookup_experiment(args.file, design_id)

    print(f"Found Experiment: {target_exp.get('name', design_id)}")

    effective_task = cli_overrides.get("training.task", final_overrides.get("training.task"))
    task_name = (
        str(effective_task).strip().strip('"').strip("'") if effective_task is not None else ""
    )
    is_vla_task = task_name == "vla_calvin"
    if is_vla_task:
        if not args.use_accelerate:
            print(
                "Detected VLA task; forcing Accelerate mode (native DDP/FSDP are unsupported for VLA)."
            )
            args.use_accelerate = True
        if args.dist_strategy != "ddp":
            print("Warning: --dist-strategy is ignored for VLA runs (Accelerate path only).")

    intern_s2_markers = [design_id, target_exp.get("name", "")]
    intern_s2_markers.extend(str(key) for key in final_overrides)
    intern_s2_markers.extend(str(value) for value in final_overrides.values())
    intern_s2_markers.extend(str(key) for key in cli_overrides)
    intern_s2_markers.extend(str(value) for value in cli_overrides.values())
    requires_intern_s2_transformers = any(
        "intern_s2" in marker.lower() or "intern-s2" in marker.lower()
        for marker in intern_s2_markers
    )

    # 4. Resources
    resources = target_exp.get("resources", {"ngpus": 12})
    ngpus = resources.get("ngpus", 12)
    print(f"Resources: {ngpus} XPUs per node, {args.nodes} nodes")

    # 5. Proxy Setup
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

    # Secrets (HF_TOKEN, WANDB_API_KEY) are read from .env ON THE NODE.
    #
    # The previous `export HF_TOKEN="$HF_TOKEN"` form could not work: qsub is
    # invoked without -V and the script carries no #PBS -V, so PBS Pro does not
    # propagate the submitting shell's environment. On the node "$HF_TOKEN"
    # expanded to the empty string. Embedding the literal value instead would
    # write plaintext secrets into every jobs/*.sh on a group-readable
    # filesystem, so read them from .env at runtime instead — .env is already
    # this project's secret store and is gitignored.
    _secrets_block = r'''
# --- secrets from .env (PBS does not propagate the submitting environment) ---
# Only these keys are taken, so the rest of .env cannot clobber values that are
# exported further down (HF_HOME, VENV_PATH, ...).
PRISM_ENV_FILE="__ENVFILE__"
if [ -f "$PRISM_ENV_FILE" ]; then
    for _k in HF_TOKEN WANDB_API_KEY; do
        _v=$(sed -n "s/^${_k}=//p" "$PRISM_ENV_FILE" | tail -1)
        _v="${_v%\"}"; _v="${_v#\"}"
        if [ -n "$_v" ]; then export "${_k}=$_v"; fi
    done
    unset _k _v
fi
'''.replace("__ENVFILE__", os.path.abspath(".env"))
    proxy_env += _secrets_block

    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not found in .env. External models might fail to load.")
    if not os.environ.get("WANDB_API_KEY"):
        print(
            "Warning: WANDB_API_KEY not found in .env. WandB logging will be "
            "skipped (training still runs, tracking is silently lost)."
        )

    # Forward DL_NUM_WORKERS from caller env or .env (matches launch_aurora.py +
    # launch_aurora_daos.py). tools/run_sweep.py scopes this per-cell to 0 for
    # non-image cells; without forwarding it the heredoc's `bash -lc` strips
    # it and the bucketed-collator path defaults to 4 workers — surfacing as
    # "Stream X is persistently empty after 5 restarts" on the sweep cell.
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

    # 6. Command Construction
    # Precedence: --prism-dir CLI > PRISM_DIR from .env > cwd
    if args.prism_dir is not None:
        prism_dir = args.prism_dir
    elif env_config.get("PRISM_DIR"):
        prism_dir = env_config["PRISM_DIR"]
    else:
        prism_dir = os.getcwd()
    print(f"PRISM dir: {prism_dir}")
    model_group = resolve_hydra_override(final_overrides, unknown_args, "model", None)
    model_group_config = load_model_group_config(prism_dir, model_group)

    # Resolve HF paths: --hf-home CLI > HF_HOME from .env > $HOME/.cache/huggingface
    if args.hf_home is not None:
        default_hf_home = args.hf_home
    elif env_config.get("HF_HOME"):
        default_hf_home = env_config["HF_HOME"]
    else:
        default_hf_home = os.path.expanduser("~/.cache/huggingface")
    # --shared-hf-home CLI > SHARED_HF_HOME from .env > $HF_HOME/hub
    if args.shared_hf_home is not None:
        default_shared_hf_home = args.shared_hf_home
    elif env_config.get("SHARED_HF_HOME"):
        default_shared_hf_home = env_config["SHARED_HF_HOME"]
    else:
        default_shared_hf_home = os.path.join(default_hf_home, "hub")
    print(f"HF home: {default_hf_home}")
    print(f"Shared HF home (model staging source): {default_shared_hf_home}")
    if not os.path.isdir(default_shared_hf_home):
        print(
            f"ERROR: --shared-hf-home {default_shared_hf_home} does not exist. "
            f"Model staging would fail mid-job and waste the queue allocation. "
            f"Pass --shared-hf-home <path> or set SHARED_HF_HOME in .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    proxy_env += f'\nexport HF_HOME="{default_hf_home}"'

    # IsoFLOP: export calibration handle + runtime FPS so the trainer's
    # _FlopCounter (src/training/perf_log.py) can attribute cumulative FLOPs
    # in perf.jsonl. Both are no-ops on non-IsoFLOP runs.
    if args.calibration_json is not None:
        proxy_env += f'\nexport CALIBRATION_JSON="{args.calibration_json}"'
    if args.runtime_flops_per_step is not None:
        proxy_env += f'\nexport RUNTIME_FLOPS_PER_STEP="{args.runtime_flops_per_step:.6e}"'

    # Handle overrides
    overrides_list = []
    for k, v in final_overrides.items():
        # Handle list values (e.g., model.modalities: [text, image])
        # Convert Python list to Hydra-compatible format: '[item1,item2]'
        # Must be quoted for bash and have no spaces for Hydra parser
        if isinstance(v, list):
            list_str = "[" + ",".join(str(item) for item in v) + "]"
            overrides_list.append(f"'{k}={list_str}'")
        elif v is None:
            overrides_list.append(f"{k}=null")
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

    # WandB project override (explicit is safer than relying on config defaults)
    if args.wandb_project:
        overrides_list.append(f"wandb.project={args.wandb_project}")
        overrides_list.append("wandb.mode=online")

    if args.max_steps is not None:
        overrides_list.append(f"training.max_steps={args.max_steps}")
    if args.resume_from_checkpoint:
        overrides_list.append(
            f"training.resume_from_checkpoint={os.path.abspath(args.resume_from_checkpoint)}"
        )
    if args.resume_weights_only:
        overrides_list.append(
            f"training.resume_weights_only={os.path.abspath(args.resume_weights_only)}"
        )

    # CLI overrides
    def _override_key(override: str) -> str:
        return override.split("=", 1)[0].strip("'")

    for arg in unknown_args:
        if "=" in arg and not arg.startswith("--"):
            cli_key = arg.split("=", 1)[0]
            conflict_indices = [
                i
                for i, existing in enumerate(overrides_list)
                if _override_key(existing) == cli_key
            ]
            for idx in sorted(conflict_indices, reverse=True):
                old = overrides_list.pop(idx)
                print(f"WARNING: CLI override '{arg}' replaces design value '{old}'.")
            overrides_list.append(arg)

    overrides_str = " ".join(overrides_list)

    # 7. Build explicit hosts export if provided
    explicit_hosts_export = ""
    if args.hosts:
        explicit_hosts_export = f'export EXPLICIT_HOSTS="{args.hosts}"\n'
        print(f"Multi-node interactive: Using explicit hosts: {args.hosts}")

    # Precompute the activate block. Python 3.10 disallows backslashes inside
    # f-string expression parts, so build this here (plain strings) and slot
    # the result into the cmd f-string below.
    if args.use_shared_venv:
        activate_block = f'source "{default_venv_path}/bin/activate"'
    else:
        # Mirror the DAOS launcher: tarball top-level is `.venv-deepspeed/`,
        # so patch VIRTUAL_ENV to the /tmp extraction path before sourcing.
        activate_block = (
            'sed -i "s|VIRTUAL_ENV=.*|VIRTUAL_ENV=\\"$LOCAL_ENV/.venv-deepspeed\\"|" '
            "$LOCAL_ENV/.venv-deepspeed/bin/activate\n"
            "source $LOCAL_ENV/.venv-deepspeed/bin/activate"
        )

    # Precompute the WebDataset shard-staging block. When --webdataset-dir was
    # omitted (e.g. VLA designs that read CALVIN from /flare directly), skip
    # the round-robin staging and just touch the marker so the wait-loop on
    # other ranks immediately passes.
    if args.stage_dataset_groups_local:
        shard_staging_block = (
            'if [ "$LOCAL_RANK" == "0" ]; then\n'
            "    STAGGER_DELAY=$((NODE_RANK * 10))\n"
            '    echo "Rank $RANK (Local 0): Waiting ${STAGGER_DELAY}s before staging dataset-group shards..."\n'
            "    sleep $STAGGER_DELAY\n"
            '    echo "Rank $RANK (Local 0): Staging dataset-group shards for node $NODE_RANK..."\n'
            '    rm -rf "$LOCAL_SHARDS_DIR"\n'
            '    mkdir -p "$LOCAL_SHARDS_DIR"\n'
            f"    python {prism_dir}/scripts/stage_dataset_groups.py \\\n"
            '        --config "$DATASET_CONFIG" \\\n'
            '        --groups "$DATASET_GROUPS" \\\n'
            '        --dataset-root "$DATASET_SOURCE_ROOT" \\\n'
            '        --local-dir "$LOCAL_SHARDS_DIR" \\\n'
            "        --node-rank $NODE_RANK \\\n"
            "        --num-nodes $NUM_NODES \\\n"
            f"        --max-shards-per-dataset-per-node {args.stage_shards_per_dataset_per_node} \\\n"
            f"        --max-shards-per-node {args.stage_max_shards_per_node}\n"
            '    touch "$SHARD_MARKER"\n'
            '    echo "Rank $RANK (Local 0): Dataset-group shard staging complete."\n'
            "fi"
        )
    elif args.webdataset_dir is not None:
        shard_staging_block = (
            'if [ "$LOCAL_RANK" == "0" ]; then\n'
            '    if [ ! -f "$SHARD_MARKER" ]; then\n'
            "        # Stagger start times to reduce Lustre contention (30s per node)\n"
            "        STAGGER_DELAY=$((NODE_RANK * 30))\n"
            '        echo "Rank $RANK (Local 0): Waiting ${STAGGER_DELAY}s before staging shards..."\n'
            "        sleep $STAGGER_DELAY\n"
            "\n"
            '        echo "Rank $RANK (Local 0): Staging shards for node $NODE_RANK..."\n'
            "        mkdir -p $LOCAL_SHARDS_DIR\n"
            "\n"
            f"        python {prism_dir}/scripts/stage_shards.py \\\n"
            '            --manifest "$WEBDATASET_MANIFEST" \\\n'
            '            --shards-dir "$WEBDATASET_SHARDS_DIR" \\\n'
            '            --local-dir "$LOCAL_SHARDS_DIR" \\\n'
            "            --node-rank $NODE_RANK \\\n"
            "            --num-nodes $NUM_NODES\n"
            "\n"
            '        touch "$SHARD_MARKER"\n'
            '        echo "Rank $RANK (Local 0): Shard staging complete."\n'
            "    else\n"
            '        echo "Rank $RANK (Local 0): Shards already staged."\n'
            "    fi\n"
            "fi"
        )
    else:
        # Mirror the staging-branch discipline: only LOCAL_RANK 0 touches the
        # marker, so the per-rank wait-loop below is symmetric across branches
        # and we never assume tmpfs idempotency if LOCAL_SHARDS_DIR moves.
        shard_staging_block = (
            'if [ "$LOCAL_RANK" == "0" ]; then\n'
            '    echo "Rank $RANK (Local 0): --webdataset-dir not provided; '
            'skipping shard staging."\n'
            '    mkdir -p "$LOCAL_SHARDS_DIR"\n'
            '    touch "$SHARD_MARKER"\n'
            "fi"
        )

    if requires_intern_s2_transformers:
        intern_s2_transformers_preflight = textwrap.dedent("""\
            # --- Intern-S2 transformers preflight ---
            if [ "$LOCAL_RANK" == "0" ]; then
                python -c "from transformers.modeling_rope_utils import RopeParameters; from transformers.configuration_utils import PreTrainedConfig, layer_type_validation; import transformers; print(transformers.__version__)" 2>/dev/null || {
                    echo "ERROR: Intern-S2 Preview requires transformers>=5.2.0 in the active training environment."
                    echo "  Active python: $(which python)"
                    echo "  A DEFAULT Aurora venv will not satisfy this: transformers>=5.2.0 and the"
                    echo "  system vLLM (transformers<5,>=4.56.0) are mutually exclusive, so the"
                    echo "  default build stays on the system 4.57.6 and leaves Intern-S2 disabled."
                    echo "  Fix shared venv mode: build a SEPARATE venv with the overlay --"
                    echo "    VENV_PATH=<new-path> bash tools/build_aurora_env.sh --intern-s2"
                    echo "  then set VENV_PATH to it and launch with --use-shared-venv."
                    echo "  Fix tarball mode: rebuild deepspeed_env.tar.gz from an --intern-s2 venv."
                    echo "  Use --use-shared-venv only after module load frameworks and venv activation are verified."
                    exit 1
                }
            fi
            """)
    else:
        intern_s2_transformers_preflight = ""

    # 8. Main Command with WebDataset Staging
    cmd = textwrap.dedent(f"""
        cd {prism_dir}

        # --- Environment Variables ---
        # Native DDP is default; only use Accelerate if explicitly requested
        export ACCELERATE_CONFIG_FILE={prism_dir}/scripts/accelerate_configs/aurora_ddp.yaml

# --- Multi-Node Interactive Support ---
{explicit_hosts_export}

# --- CCL/Multi-Node Configuration (Aurora Best Practices) ---
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
# CRITICAL: Use 'none' launcher and 'ofi' transport to avoid MPI re-initialization inside mpiexec
# Using pmix/mpi causes "Fatal error in internal_Init_thread: Other MPI error"
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0

# === CCL SCALING OPTIMIZATIONS (Critical for 7B+ models) ===
# Worker count: MUST be 1 for multi-node Aurora. Higher counts have caused
# large-message collective regressions in production benchmarks.
export CCL_WORKER_COUNT=1
# Use ring algorithm for large gradient AllReduce.
export CCL_ALLREDUCE=ring
# Do not force ReduceScatter. CCL defaults perform better on current Aurora.
# Enable chunking for large messages (16MB chunks)
export CCL_CHUNK_SIZE=16777216
# Slingshot/CXI network optimizations
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072

# --- General Settings (VJEPA2 pattern) ---
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS={args.cpus_per_task}
export OMP_NUM_THREADS={args.cpus_per_task}
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1  # Fix log buffering delays
{"export PYTHONWARNINGS=ignore  # Suppress noisy warnings" if args.suppress_warnings else ""}
{"# Using Accelerate (user requested)" if args.use_accelerate else "export USE_NATIVE_DDP=1  # Default: Native PyTorch DDP (best scaling on Aurora)"}
{"export DEBUG_NO_SYNC=1  # Measure bare backward without DDP AllReduce" if args.debug_no_sync else ""}
export TMPDIR=/tmp

# === Distributed Strategy Configuration ===
export DIST_STRATEGY="{args.dist_strategy}"  # ddp, fsdp, or hsdp
export FSDP_SHARDING="{args.fsdp_sharding}"  # full_shard, shard_grad_op, etc.
{"export FSDP_CPU_OFFLOAD=1" if args.fsdp_cpu_offload else ""}
export DDP_BUCKET_CAP_MB={args.ddp_bucket_mb}
export GRAD_CKPT_FREQ={args.grad_ckpt_freq}
{"export FSDP_PRODUCTION_MODE=1" if args.fsdp_production_mode or args.benchmark_mode else ""}
{"export PRISM_PRODUCTION_MODE=1" if args.fsdp_production_mode or args.benchmark_mode else ""}
# HSDP perf knobs (see scaling-study/investigation/REPORT.md)
{"export FSDP_NO_SYNC_ACCUM=1" if args.fsdp_no_sync_accum else ""}
{"export PRISM_DISABLE_PERF_PROBES=1" if args.prism_disable_perf_probes else ""}
export GRAD_NORM_INTERVAL={args.grad_norm_interval}

# --- Sequence Length & Bucketing (improves throughput for mixed-length datasets) ---
export MAX_SEQ_LENGTH={args.max_seq_length}
{"export USE_BUCKETING=1" if args.use_bucketing else "# Bucketing disabled"}
{"export BUCKET_BUFFER_SIZE=" + str(args.bucket_buffer_size) if args.use_bucketing else ""}
{"export BUCKET_NUM_BUCKETS=" + str(args.bucket_num_buckets) if args.use_bucketing else ""}
export USE_BUCKETED_COLLATOR={"1" if args.use_bucketed_collator else "0"}
{"export ENABLE_ALL_MODALITIES=1" if args.enable_all_modalities else ""}

# --- WebDataset Configuration ---
export WEBDATASET_SHARDS_DIR="{shards_dir}"
export WEBDATASET_MANIFEST="{manifest_path}"
# Held-out shards for val/loss. The trainer's manifest-based discovery only
# fires for USE_MULTI_DATASET=1, so --webdataset-dir runs previously trained
# with no validation at all. Read straight from Lustre (rank 0 only, a few
# shards) rather than staging to /tmp.
export PRISM_VAL_SHARDS_DIR="{val_shards_dir}"
export LOCAL_SHARDS_DIR="{args.local_shards_dir}"
export NUM_NODES={args.nodes}
export USE_MULTI_DATASET={"1" if train_use_multi_dataset else "0"}
export DATASET_ROOT="{train_dataset_root}"
export DATASET_SOURCE_ROOT="{dataset_root}"
export DATASET_GROUPS="{args.dataset_groups or ''}"
export DATASET_CONFIG="{dataset_config_abs}"
export DATASET_PROPORTIONS=""
export WEBDATASET_LOCAL_MODALITY=image
export WEBDATASET_RESAMPLED={"0" if args.finite_webdataset else "1"}
export WEBDATASET_PARTITION_BY={"local" if args.stage_dataset_groups_local else "global"}
export WEBDATASET_SHUFFLE_BUFFER={args.webdataset_shuffle_buffer}

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
echo "WebDataset Shards: {shards_dir}"

# --- Cleanup ---
echo "Cleaning up stale python processes..."
pkill -u $USER -f "python src/train.py" || true
sleep 2

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

# --- Execution ---
# CPU binding: --cpu-bind depth --depth matches VJEPA2 for DataLoader performance
mpiexec $HOSTFILE_ARG -n {args.nodes * ngpus} -ppn {ngpus} --cpu-bind depth --depth {args.cpus_per_task} bash -lc '
# CRITICAL: Load modules BEFORE activating venv
# - frameworks: Ensures IPEX and XPU libraries are properly configured
#   Without this, importing intel_extension_for_pytorch crashes with std::bad_alloc
# Pinned to 2025.3.1: matches what the shared venv was built against and
# what the XCCL pre-flight gate validated. No 2>/dev/null — if the pin
# is removed/renamed we want a loud failure here, not cryptic import
# errors hundreds of lines downstream.
module use /soft/modulefiles
module load frameworks/2025.3.1

# CRITICAL: Re-export CCL settings AFTER module load. The frameworks module
# re-asserts its own defaults (notably CCL_OP_SYNC=1 in 2025.3.1) and silently
# clobbers values set in the outer shell. Mirrors the post-module-load block
# in launch_aurora_daos.py. See scaling-study/investigation/REPORT.md Bug 3.
# NOTE: this comment is inside the single-quoted mpiexec heredoc — keep it
# apostrophe-free (an unescaped single quote terminates the string; see the
# fuller warning further down in this same heredoc).
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring

# Get distributed environment variables with fallback chains
# CRITICAL: For multi-node, WORLD_SIZE must be total ranks across all nodes
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
{"# --- Environment Setup (Shared Venv) ---" + chr(10) + "# Using shared venv - no tarball unpacking needed" + chr(10) + 'echo "Using shared venv: ' + default_venv_path + '"' if args.use_shared_venv else '''# --- Environment Unpack ---
export ENV_TARBALL="''' + os.path.abspath(args.packed_env) + '''"
export LOCAL_ENV="/tmp/deepspeed_env"
export MARKER_FILE="$LOCAL_ENV/env_ready"

# Re-extract if: no marker exists, OR tarball is newer than the marker (env was repacked)
if [ "$LOCAL_RANK" == "0" ]; then
    if [ ! -f "$MARKER_FILE" ] || [ "$ENV_TARBALL" -nt "$MARKER_FILE" ]; then
        if [ -f "$MARKER_FILE" ]; then
            echo "Rank $RANK (Local 0): Tarball is newer than cached env, re-extracting..."
        fi
        rm -rf $LOCAL_ENV
        mkdir -p $LOCAL_ENV
        echo "Rank $RANK (Local 0): Unpacking environment to $LOCAL_ENV..."
        tar -xzf "$ENV_TARBALL" -C $LOCAL_ENV
        touch "$MARKER_FILE"
        echo "Rank $RANK (Local 0): Environment ready."
    else
        echo "Rank $RANK (Local 0): Environment already unpacked."
    fi
fi
while [ ! -f "$MARKER_FILE" ]; do sleep 1; done'''}

{"# --- Model Cache (shared filesystem) ---" + chr(10) + 'export HF_HOME="' + default_hf_home + '"' + chr(10) + 'echo "  HF_HOME set to: $HF_HOME (shared filesystem)"' if args.use_shared_venv else '''# --- Model Copy ---
export SHARED_HF_ROOT="''' + default_hf_home + '''"
export SHARED_HF_HOME="''' + default_shared_hf_home + '''"
export LOCAL_HF_ROOT="/tmp/huggingface"
export LOCAL_HF_HOME="/tmp/huggingface/hub"
export MODEL_DIR="''' + (hf_model_dir or "") + '''"
export MODEL_MARKER="$LOCAL_HF_HOME/model_ready"

# List of all models to stage (backbone + encoders/processors)
MODELS_TO_STAGE=(
''' + image_model_stage_entries + '''
    "models--google--tapas-base"             # Table Encoder
    "models--Salesforce--moirai-2.0-R-small" # Time Series Encoder
    "models--polymathic-ai--walrus"          # Geometry Encoder
)
if [ -n "$MODEL_DIR" ]; then
    MODELS_TO_STAGE=("$MODEL_DIR" "${MODELS_TO_STAGE[@]}")
fi

# Extracted Intern-S2 checkpoints live directly under HF_HOME rather than in
# the Hugging Face Hub cache. Link them into node-local HF_HOME when present.
STANDALONE_ARTIFACTS=(
    "intern-s2-preview-timeseries"
    "intern-s2-preview-397b-timeseries"
)

if [ "$LOCAL_RANK" == "0" ]; then
    mkdir -p "$LOCAL_HF_HOME"
    for ARTIFACT in "${STANDALONE_ARTIFACTS[@]}"; do
        if [ -d "$SHARED_HF_ROOT/$ARTIFACT" ]; then
            ln -sfn "$SHARED_HF_ROOT/$ARTIFACT" "$LOCAL_HF_ROOT/$ARTIFACT"
            echo "  Linked standalone artifact $ARTIFACT"
        fi
    done
    if [ ! -f "$MODEL_MARKER" ]; then
        echo "Rank $RANK (Local 0): Staging models to /tmp..."
        for MODEL in "${MODELS_TO_STAGE[@]}"; do
            if [ -d "$SHARED_HF_HOME/$MODEL" ] && [ ! -d "$LOCAL_HF_HOME/$MODEL" ]; then
                echo "  Copying $MODEL..."
                cp -r "$SHARED_HF_HOME/$MODEL" "$LOCAL_HF_HOME/"
            fi
        done
        touch "$MODEL_MARKER"
        echo "Rank $RANK (Local 0): Model staging complete."
    else
        echo "Rank $RANK (Local 0): Models already staged."
    fi
fi
while [ ! -f "$MODEL_MARKER" ]; do sleep 2; done'''}

# --- WebDataset Shard Staging (Node-Local) ---
export SHARD_MARKER="$LOCAL_SHARDS_DIR/shards_ready"
export SHARD_PATTERN_FILE="$LOCAL_SHARDS_DIR/shard_pattern.txt"

{shard_staging_block}
while [ ! -f "$SHARD_MARKER" ]; do sleep 2; done

# --- Activate Environment ---
export PYTHONNOUSERSITE=1
{activate_block}

# --- PRISM_BUILD_INFO manifest check ---
# Read the venv manifest and confirm frameworks_module + python_realpath
# match the currently-loaded values. Catches the silent-staleness failure
# mode where ALCF flips the default frameworks python (py3.10 -> py3.12)
# but the venv keeps pointing at the old interpreter. 5-line check that
# saves the 40+ lines of unparseable IPEX C++ symbol noise.
#
# IMPORTANT: in tarball mode, do NOT trust $VIRTUAL_ENV. The outer launch
# script may have run `source .venv-deepspeed/bin/activate` at the top and
# set VIRTUAL_ENV to the repo-root .venv-deepspeed (a legacy dev convenience).
# The sed-then-source dance for the /tmp activate happens inside the mpiexec
# heredoc but under Aurora bash -lc invocation the outer-shell VIRTUAL_ENV
# can persist, causing the manifest read to hit the stale repo venv manifest
# instead of the freshly-extracted one in /tmp. Resolve from $LOCAL_ENV
# explicitly for tarball mode; shared-venv mode keeps $VIRTUAL_ENV (which
# is set by the explicit source on a known shared path).
# NOTE: comments inside this mpiexec heredoc MUST NOT contain apostrophes
# (single quotes). The heredoc is opened with a single quote; any unescaped
# apostrophe terminates the string and the remaining script body executes
# in the outer shell instead of on every rank (see memory entry
# launcher_smoke_harness_bug.md / PR #33).
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
    # Only compare if the manifest had the field — older venvs predate it.
    if [ -n "$expected_fw" ] && [ "$expected_fw" != "$actual_fw" ]; then
        echo "ERROR: venv manifest mismatch (frameworks_module)"
        echo "  expected: $expected_fw"
        echo "  actual:   $actual_fw"
        echo "Rebuild venv against currently-loaded frameworks/$LMOD_FAMILY_FRAMEWORKS_VERSION."
        exit 1
    fi
    if [ -n "$expected_py" ] && [ "$expected_py" != "$actual_py" ]; then
        echo "ERROR: venv manifest mismatch (python_realpath)"
        echo "  expected: $expected_py"
        echo "  actual:   $actual_py"
        echo "Rebuild venv against currently-loaded frameworks/$LMOD_FAMILY_FRAMEWORKS_VERSION."
        exit 1
    fi
fi

# --- Install webdataset if missing (Rank 0 per node only, with sync) ---
export WDS_MARKER="/tmp/webdataset_ready_${{NODE_RANK}}"
if [ "$LOCAL_RANK" == "0" ]; then
    python -c "import webdataset" >/dev/null 2>&1 || {{
        echo "Rank $RANK: Installing webdataset..."
        pip install --quiet webdataset 2>/dev/null
    }}
    touch "$WDS_MARKER"
fi
while [ ! -f "$WDS_MARKER" ]; do sleep 1; done

export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
{'export HF_HOME="' + default_hf_home + '"' + chr(10) + 'export TRANSFORMERS_CACHE="' + default_shared_hf_home + '"' + chr(10) + 'export HF_HUB_CACHE="' + default_shared_hf_home + '"' + chr(10) + 'export HF_DATASETS_CACHE="' + default_hf_home + '/datasets"' if args.use_shared_venv else '''export HF_HOME="/tmp/huggingface"
export TRANSFORMERS_CACHE="/tmp/huggingface/hub"
export HF_HUB_CACHE="/tmp/huggingface/hub"
export HF_DATASETS_CACHE="/tmp/huggingface/datasets"'''}

# CRITICAL: Force offline mode to prevent HF Hub metadata operations
# which cause segfaults due to concurrent file access on multi-node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Verify critical dependencies ---
# webdataset should be in the packed env (requirements/deepspeed.txt)
if [ "$LOCAL_RANK" == "0" ]; then
    python -c "import webdataset" 2>/dev/null || {{
        echo "ERROR: webdataset missing from packed env."
        echo "Rebuild the env with: bash tools/setup_deepspeed_env.sh && tar -czf deepspeed_env.tar.gz .venv-deepspeed"
        exit 1
    }}
fi

{intern_s2_transformers_preflight}

# Override dataset path to use local staged shards
export WEBDATASET_LOCAL_PATH="${{LOCAL_SHARDS_DIR}}"
export WEBDATASET_LOCAL_MODALITY="{args.webdataset_modality}"

# In Accelerate mode on Aurora, each rank sees exactly one XPU tile due
# ZE_AFFINITY_MASK. LOCAL_RANK may still be 0..(ppn-1), which causes
# Accelerate/DDP to target invalid device indices. Normalize LOCAL_RANK to 0
# just before launching Python while preserving the per-rank ZE mask.
if [ "${{USE_NATIVE_DDP:-0}}" != "1" ] && [ "${{USE_NATIVE_FSDP:-0}}" != "1" ]; then
    export ACCELERATE_ORIG_LOCAL_RANK="${{LOCAL_RANK}}"
    export LOCAL_RANK=0
    export PMI_LOCAL_RANK=0
    export PMIX_LOCAL_RANK=0
    export PALS_LOCAL_RANKID=0
    export OMPI_COMM_WORLD_LOCAL_RANK=0
    echo "Accelerate mode: normalized LOCAL_RANK to 0 (orig=$ACCELERATE_ORIG_LOCAL_RANK)"
fi

echo "Rank $RANK/$WORLD_SIZE (Node: $NODE_RANK, Local: $LOCAL_RANK/$LOCAL_WORLD_SIZE)"
echo "  Using local shards from: $LOCAL_SHARDS_DIR"

python src/train.py {overrides_str}
'
""")

    # 8. Generate Script
    # Create logs directory organized by design ID
    logs_dir = os.path.join(os.getcwd(), "logs", design_id)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Logs directory: {logs_dir}")

    if args.batch:
        wt = args.walltime if args.walltime else resources.get("walltime", "01:00:00")
        header = f"""#!/bin/bash -l
#PBS -l select={args.nodes}
#PBS -l walltime={wt}
#PBS -l filesystems=home:flare
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
# Aurora WebDataset Launch Script
# Experiment: {design_id}
# Generated by tools/launch_aurora_web.py

# --- Module Load ---
module load frameworks/2025.3.1
module load hdf5

# --- Activate Venv ---
if [ -d ".venv-deepspeed" ]; then
    source .venv-deepspeed/bin/activate
fi

{proxy_env}

{cmd}
""")

    mode = "batch" if args.batch else "interactive"
    script_name = f"run_aurora_web_{args.id}_{date_str}_{time_str}_{mode}.sh"
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
            # Batch mode: submit via qsub
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
                # Exit non-zero: a failed qsub previously printed the error but
                # still returned 0, so batch drivers submitting many cells in a
                # loop reported success while nothing reached the queue.
                print(f"qsub failed: {e.stderr}")
                sys.exit(1)
        else:
            # Interactive mode: run directly or via SSH
            if args.run_via_ssh and args.hosts:
                # Execute via SSH to first host (for running from UAN)
                first_host = args.hosts.split(",")[0]
                abs_script_path = os.path.abspath(script_path)
                print(f"Executing via SSH on {first_host}...")
                print(f"  Script: {abs_script_path}")
                try:
                    subprocess.run(["ssh", "-t", first_host, f"bash {abs_script_path}"], check=True)
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

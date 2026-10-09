#!/usr/bin/env python3
"""launch_polaris.py — submit/run PRISM on Polaris (NVIDIA A100, CUDA).

Counterpart to launch_aurora_web.py for ALCF's NVIDIA cluster. Same flag
surface where it makes sense (--design / --nodes / --batch /
--dist-strategy / --use-accelerate / --max-steps / --dry-run); Aurora-only
flags (ZE_AFFINITY_MASK, CCL_*, DAOS, HSN suffix) are dropped or replaced
with their NCCL/CUDA equivalents.

Polaris specifics this launcher hardcodes:
  - 4× A100 per node (auto-detected via `nvidia-smi -L` at runtime; the
    --gpus-per-node flag overrides for debugging)
  - NCCL backend (set via DIST_BACKEND=nccl env var read by src.training.distributed)
  - mpiexec -ppn 4 --cpu-bind depth -d 16 (matches ALCF best-practice guide)
  - `conda/2025-09-28` + venv overlay built by tools/setup_polaris_env.sh
  - /eagle filesystem (no DAOS / /flare)
  - Outbound HF downloads via proxy.alcf.anl.gov:3128

Usage (dry-run):
    python tools/launch_polaris.py --id POLARIS-DDP-1N \\
        --design PRISM-IMAGE-ONLY-2N --nodes 1 --batch --dry-run

Usage (submit a batch job):
    python tools/launch_polaris.py --id POLARIS-DDP-1N \\
        --design PRISM-IMAGE-ONLY-2N --nodes 1 --batch \\
        --queue debug --max-steps 50

Usage (interactive, from inside a qsub -I shell on a compute node):
    python tools/launch_polaris.py --id POLARIS-DDP-1N \\
        --design PRISM-IMAGE-ONLY-2N --nodes 1
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch_common import load_dotenv, lookup_experiment  # noqa: E402

POLARIS_DEFAULT_GPUS_PER_NODE = 4
POLARIS_PROXY = "http://proxy.alcf.anl.gov:3128"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch PRISM on Polaris (NVIDIA A100, CUDA, PBS).",
    )
    # === Core ===
    p.add_argument("--file", default="experiments/prism_designs.yaml")
    p.add_argument("--id", required=True, help="Run ID / PBS job name")
    p.add_argument(
        "--design",
        default=None,
        help="Design ID in YAML; defaults to --id when omitted.",
    )
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument(
        "--gpus-per-node",
        type=int,
        default=POLARIS_DEFAULT_GPUS_PER_NODE,
        help="Override GPUs per node (Polaris has 4× A100).",
    )

    # === PBS ===
    p.add_argument("--batch", action="store_true", help="Emit a PBS script + qsub it")
    p.add_argument("--project", default="ModCon")
    p.add_argument("--queue", default="debug", help="debug | debug-scaling | prod")
    p.add_argument(
        "--walltime",
        default=None,
        help="HH:MM:SS; defaults to the design's resources.walltime, "
        "else 01:00:00. Capped per queue by Polaris policy.",
    )
    p.add_argument(
        "--filesystems",
        default="home:eagle",
        help="PBS -l filesystems= value. /flare is not mounted on Polaris.",
    )

    # === Run mode ===
    p.add_argument("--dry-run", action="store_true", help="Write script but do not run / submit")
    p.add_argument(
        "--use-accelerate",
        action="store_true",
        help="Use Accelerate launch (required for DeepSpeed ZeRO). Default is native DDP/FSDP.",
    )
    p.add_argument(
        "--deepspeed-zero",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="DeepSpeed ZeRO stage. Non-zero forces --use-accelerate and sets "
        "DEEPSPEED_ZERO_STAGE so distributed.py picks the env-only init path.",
    )

    # === Distributed strategy (mirrors launch_aurora_web.py surface) ===
    p.add_argument(
        "--dist-strategy",
        choices=["ddp", "fsdp", "hsdp"],
        default="ddp",
    )
    p.add_argument(
        "--fsdp-sharding",
        choices=["full_shard", "shard_grad_op", "hybrid_shard", "no_shard"],
        default="full_shard",
    )
    p.add_argument("--fsdp-cpu-offload", action="store_true")
    p.add_argument("--ddp-bucket-mb", type=int, default=50)
    p.add_argument("--grad-ckpt-freq", type=int, default=1)
    p.add_argument(
        "--fsdp-production-mode",
        action="store_true",
        help="Skip non-essential CUDA syncs (FSDP_PRODUCTION_MODE=1).",
    )
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=None)

    # === Data ===
    p.add_argument(
        "--webdataset-dir",
        default=None,
        help="WebDataset directory with shards/ + manifest.json on /eagle. "
        "Required for real-data designs; omit to use the design's own data path "
        "or --enable-all-modalities dummy fallback.",
    )
    p.add_argument(
        "--enable-all-modalities",
        action="store_true",
        help="Set ENABLE_ALL_MODALITIES=1 so missing datasets fall back to dummy tensors. "
        "Use for compute-only smoke runs without staging real shards.",
    )
    p.add_argument(
        "--use-bucketing",
        action="store_true",
        help="Sequence length bucketing (helps mixed-length datasets).",
    )
    p.add_argument(
        "--dataset-groups",
        default=None,
        help="Comma-separated dataset groups to activate (e.g. pixmo,cosyn). "
        "Default lets train.py / Hydra config pick. Required when the design's "
        "model.modalities doesn't intersect with the default dataset's "
        "modalities (e.g. PRISM-IMAGE-ONLY-2N + ts_qa default → "
        "missing_modality_start_end_token_indices crash on first batch). "
        "Implies USE_MULTI_DATASET=1 and DAOS_MOUNT=<--dataset-root> in the env.",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        help="Filesystem root the dataset YAML's relative paths are resolved against. "
        "Polaris equivalent of Aurora's DAOS_MOUNT. Used together with "
        "--dataset-groups. Example: /eagle/ModCon/ngetty/datasets/",
    )
    p.add_argument(
        "--dataset-config",
        default=None,
        help="Override DATASET_CONFIG (the dataset-group YAML path). Default "
        "uses src/conf/data/daos_datasets.yaml which has Aurora paths; on "
        "Polaris point at a copy under src/conf/data/ that has paths matching "
        "your /eagle layout.",
    )

    # === Environment ===
    p.add_argument(
        "--prism-dir",
        default=None,
        help="PRISM checkout. Defaults to $PRISM_DIR from .env, else cwd.",
    )
    p.add_argument(
        "--prism-venv",
        default=None,
        help="Path to the Polaris venv. Defaults to $PRISM_VENV from .env, "
        "else <prism-dir>/.venv-polaris.",
    )
    p.add_argument(
        "--conda-module",
        default="conda/2025-09-28",
        help="ALCF conda module to load before activating the venv.",
    )
    p.add_argument(
        "--hf-home",
        default=None,
        help="HF cache root. Defaults to $HF_HOME, else $HOME/.cache/huggingface.",
    )
    p.add_argument(
        "--no-proxy",
        action="store_true",
        help="Skip setting the ALCF http(s) proxy (e.g. for fully-offline runs).",
    )
    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=16,
        help="--cpu-bind depth depth value (ALCF best practice: 16).",
    )

    # === Logging ===
    p.add_argument(
        "--wandb-project",
        default=None,
        help="WandB project (overrides config). Online unless WANDB_MODE is set.",
    )
    return p


def main() -> int:
    parser = _build_parser()
    args, unknown_args = parser.parse_known_args()

    # Stash Hydra-style overrides (key=value) for forwarding to train.py.
    cli_overrides: list[str] = []
    for raw in unknown_args:
        if "=" in raw and not raw.startswith("--"):
            cli_overrides.append(raw)
        else:
            print(f"Warning: ignoring unknown argument {raw!r}")

    # On Polaris DeepSpeed runs through native train.py + env-only init.
    # mpiexec spawns the per-GPU processes; train.py's Accelerator()
    # picks up the rank tuple from PMI_*/PALS_* and DEEPSPEED_ZERO_STAGE
    # tells setup_distributed() to take the env-only init path.
    # `accelerate launch` is NOT needed (and would double-spawn under
    # mpiexec) — just exporting DEEPSPEED_ZERO_STAGE is sufficient.

    design_id = args.design if args.design else args.id
    target_exp, _parent, design_overrides = lookup_experiment(args.file, design_id)
    print(f"Found Experiment: {target_exp.get('name', design_id)}")

    # === Resolve dirs ===
    # Skip .env on Polaris by default — the .env in the Aurora repo points at
    # /flare paths that don't exist here. Pass --env-file to opt in.
    env_config = {}
    if os.path.isfile(".env") and "/eagle" not in os.getcwd():
        env_config = load_dotenv()

    # When --prism-dir isn't given, prefer the launcher's own location over
    # cwd. That way `ssh polaris "cd /eagle/.../BaseMM_PRISM && \
    # python tools/launch_polaris.py ..."` Just Works from a different host.
    if args.prism_dir:
        prism_dir = args.prism_dir
    elif env_config.get("PRISM_DIR"):
        prism_dir = env_config["PRISM_DIR"]
    else:
        prism_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prism_venv = (
        args.prism_venv
        or env_config.get("PRISM_VENV")
        or os.path.join(prism_dir, ".venv-polaris")
    )
    hf_home = (
        args.hf_home
        or env_config.get("HF_HOME")
        or os.path.expanduser("~/.cache/huggingface")
    )
    print(f"PRISM dir : {prism_dir}")
    print(f"PRISM venv: {prism_venv}")
    print(f"HF home   : {hf_home}")
    print(f"Conda mod : {args.conda_module}")
    print(f"GPUs/node : {args.gpus_per_node}")
    print(f"Nodes     : {args.nodes}")

    # === Build Hydra overrides ===
    # Force training.device=cuda on Polaris regardless of the design value
    # (Aurora designs hardcode "xpu"). Done before merging CLI overrides so
    # the user can still bypass with a literal `training.device=cpu` if they
    # want to debug on a login node.
    override_pairs: list[str] = []
    for k, v in design_overrides.items():
        if isinstance(v, list):
            # Hydra list syntax `[a,b,c]` with NO surrounding quotes. The
            # outer mpiexec wrapper is `bash -lc '...'`, so a quoted
            # `'k=[a,b]'` would prematurely close the single-quoted block
            # and silently drop the rest of the train.py args on all ranks
            # (see launcher_smoke_harness_bug memory note).
            list_str = "[" + ",".join(str(item) for item in v) + "]"
            override_pairs.append(f"{k}={list_str}")
        else:
            override_pairs.append(f"{k}={v}")
    override_pairs.append("training.device=cuda")  # win over the design default
    override_pairs.append(f"exp.id={args.id}")
    if args.max_steps is not None:
        override_pairs.append(f"training.max_steps={args.max_steps}")
    if args.wandb_project:
        override_pairs.append(f"wandb.project={args.wandb_project}")
        override_pairs.append("wandb.mode=online")
    override_pairs.extend(cli_overrides)

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "outputs", args.id, date_str, time_str)
    override_pairs.append(f"hydra.run.dir={output_dir}")
    overrides_str = " ".join(override_pairs)

    # === Distributed env ===
    dist_env_lines = [
        f"export DIST_STRATEGY={args.dist_strategy}",
        f"export FSDP_SHARDING={args.fsdp_sharding}",
        f"export DDP_BUCKET_CAP_MB={args.ddp_bucket_mb}",
        f"export GRAD_CKPT_FREQ={args.grad_ckpt_freq}",
        f"export MAX_SEQ_LENGTH={args.max_seq_length}",
        "export DIST_BACKEND=nccl",  # read by src/training/distributed.py:_select_backend
    ]
    if args.fsdp_cpu_offload:
        dist_env_lines.append("export FSDP_CPU_OFFLOAD=1")
    if args.fsdp_production_mode:
        dist_env_lines.append("export FSDP_PRODUCTION_MODE=1")
        dist_env_lines.append("export PRISM_PRODUCTION_MODE=1")
    if args.use_bucketing:
        dist_env_lines.append("export USE_BUCKETING=1")
    if args.enable_all_modalities:
        dist_env_lines.append("export ENABLE_ALL_MODALITIES=1")
    if args.dataset_groups:
        dist_env_lines.append(f"export DATASET_GROUPS={args.dataset_groups}")
        dist_env_lines.append("export USE_MULTI_DATASET=1")
        if args.dataset_root:
            # train.py's multi-dataset path reads DAOS_MOUNT regardless of
            # filesystem — on Polaris it's just an /eagle path.
            dist_env_lines.append(f"export DAOS_MOUNT={args.dataset_root}")
        else:
            print(
                "WARN: --dataset-groups was given but --dataset-root was not. "
                "Multi-dataset path needs DAOS_MOUNT; falling back to whatever "
                "the env already sets (likely will not work).",
                file=sys.stderr,
            )
    if args.dataset_config:
        dist_env_lines.append(f"export DATASET_CONFIG={args.dataset_config}")
    if args.deepspeed_zero:
        # ZoneATrainer (the Accelerate-backed trainer) reads
        # ACCELERATE_USE_DEEPSPEED + ACCELERATE_CONFIG_FILE to switch on
        # DeepSpeed. setup_distributed() also reads DEEPSPEED_ZERO_STAGE
        # to take the env-only init path (avoids mpi4py re-init crash).
        ds_config = os.path.join(
            prism_dir, "scripts", "accelerate_configs",
            f"deepspeed_zero{args.deepspeed_zero}.yaml",
        )
        dist_env_lines.extend([
            f"export DEEPSPEED_ZERO_STAGE={args.deepspeed_zero}",
            "export ACCELERATE_USE_DEEPSPEED=true",
            f"export ACCELERATE_DEEPSPEED_ZERO_STAGE={args.deepspeed_zero}",
            f"export ACCELERATE_CONFIG_FILE={ds_config}",
        ])
        if args.deepspeed_zero == 3:
            dist_env_lines.append(
                "export ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL=true"
            )
        # DON'T set USE_NATIVE_DDP — train.py needs to fall through to
        # ZoneATrainer (Accelerator) for DeepSpeed to engage. Without
        # USE_NATIVE_DDP, setup_distributed() is bypassed for the env-only
        # init too, BUT DEEPSPEED_ZERO_STAGE keeps the mpi4py path off.
    elif not args.use_accelerate:
        dist_env_lines.append("export USE_NATIVE_DDP=1")

    if args.webdataset_dir is not None:
        if not os.path.isdir(args.webdataset_dir):
            print(
                f"ERROR: --webdataset-dir {args.webdataset_dir} does not exist on this host. "
                "On Polaris, shards must live under /eagle.",
                file=sys.stderr,
            )
            return 1
        dist_env_lines.append(f"export WEBDATASET_LOCAL_PATH={args.webdataset_dir}")

    # === Logs dir ===
    logs_dir = os.path.join(os.getcwd(), "logs", design_id)
    os.makedirs(logs_dir, exist_ok=True)

    # === Launch verb ===
    # Native DDP/FSDP and DeepSpeed both run as `python src/train.py` here.
    # mpiexec has already spawned the ranks; src/train.py uses either
    # setup_distributed (native) or Accelerator() (Accelerate/DeepSpeed) and
    # reads PMI_*/PALS_* env vars set by mpiexec. Wrapping with
    # `accelerate launch` would double-spawn under mpiexec.
    launch_line = f"python src/train.py {overrides_str}"
    if args.use_accelerate and not args.deepspeed_zero:
        # If the user genuinely needs accelerate's launcher (not DeepSpeed),
        # they likely also need --num_processes 1 because mpiexec already
        # owns the rank fan-out. Use venv-local accelerate to avoid the
        # base-conda transformers shadowing.
        launch_line = textwrap.dedent(f"""
            python -m accelerate.commands.launch \\
                --num_processes 1 \\
                --machine_rank $RANK \\
                --main_process_ip $MASTER_ADDR \\
                --main_process_port $MASTER_PORT \\
                --mixed_precision bf16 \\
                --dynamo_backend no \\
                src/train.py {overrides_str}
            """).strip()

    # === HPC script body ===
    # Order matters: mpiexec sets PALS_*/PMI_*; the rank shell then derives
    # WORLD_SIZE/RANK/LOCAL_RANK before importing torch. CUDA_VISIBLE_DEVICES
    # is pinned to LOCAL_RANK per the ALCF guidance ("you must set it BEFORE
    # mpi4py is imported").
    proxy_block = "" if args.no_proxy else textwrap.dedent(f"""
        export HTTP_PROXY="{POLARIS_PROXY}"
        export HTTPS_PROXY="{POLARIS_PROXY}"
        export http_proxy="{POLARIS_PROXY}"
        export https_proxy="{POLARIS_PROXY}"
        export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
        """).strip()

    # Reference HF_TOKEN by name rather than interpolating the value into
    # the generated jobs/*.sh — that file persists with mode 0o755 on /eagle
    # and a literal token would leak to anyone with read access. PBS -V /
    # mpiexec --envall propagate the variable from the submitting shell.
    hf_token_block = (
        'export HF_TOKEN="$HF_TOKEN"'
        if "HF_TOKEN" in os.environ
        else '# HF_TOKEN not set — public models only'
    )

    dist_env_block = "\n".join(dist_env_lines)

    cmd = textwrap.dedent(f"""
        cd {prism_dir}

        # === Modules + venv ===
        module use /soft/modulefiles
        module load {args.conda_module}
        conda activate base
        if [ -d "{prism_venv}" ]; then
            source "{prism_venv}/bin/activate"
        else
            echo "ERROR: PRISM venv not found at {prism_venv}." >&2
            echo "Build it with: bash tools/setup_polaris_env.sh" >&2
            exit 1
        fi

        # === Caches ===
        export HF_HOME="{hf_home}"
        export TRANSFORMERS_CACHE="{hf_home}/hub"
        export HF_HUB_CACHE="{hf_home}/hub"
        {hf_token_block}

        # === Proxy ===
        {proxy_block}

        # === Distributed strategy / data ===
{textwrap.indent(dist_env_block, "        ")}

        # === Master rendezvous ===
        # PBS_NODEFILE is set inside a PBS allocation (batch via -V or
        # interactive via qsub -I). If we're running outside one, mpiexec
        # has nowhere to dispatch — fail fast with a clear message rather
        # than letting `--hostfile /dev/null` produce an opaque error.
        if [ -z "${{PBS_NODEFILE:-}}" ] || [ ! -f "${{PBS_NODEFILE}}" ]; then
            echo "ERROR: PBS_NODEFILE is unset — no active PBS allocation." >&2
            echo "       Submit with --batch, or run inside qsub -I first." >&2
            exit 1
        fi
        MASTER_HOST=$(head -n1 "$PBS_NODEFILE")
        NNODES=$(wc -l < "$PBS_NODEFILE")
        export MASTER_ADDR="$MASTER_HOST"
        export MASTER_PORT=$((20000 + RANDOM % 20000))
        echo "Master: $MASTER_ADDR:$MASTER_PORT   Nodes: $NNODES"

        # === NCCL config on Polaris ===
        # The ALCF-recommended AWS-OFI-NCCL plugin (NCCL_NET="AWS Libfabric")
        # delivers 2-3× on some collectives but is fragile: it segfaults
        # mid-init on conda/2025-09-28 + torch 2.8.0 + PRISM (observed
        # 2026-05). The ALCF NCCL doc explicitly warns about hangs with
        # Megatron-DeepSpeed. For now stay on the default backend
        # (sockets/IB) which is slower but stable. Re-enable with:
        #   export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
        #   export NCCL_NET="AWS Libfabric" NCCL_NET_GDR_LEVEL=PHB
        #   export NCCL_CROSS_NIC=1 NCCL_COLLNET_ENABLE=1
        # if your run is stable with it.
        export NCCL_IB_DISABLE=0
        # NCCL_DEBUG=INFO is verbose but essential when chasing hangs:
        # uncomment for the first run on a new env.
        export NCCL_DEBUG=WARN

        # === Sanity prints ===
        echo "Output dir : {output_dir}"
        echo "Hosts:"
        cat "$PBS_NODEFILE"

        # Drop stale python processes from any prior failed run, on every
        # allocated node (a head-only pkill leaves zombies on the others
        # that collide with the fresh ranks for the same GPU).
        mpiexec --envall -n "$NNODES" --ppn 1 --hostfile "$PBS_NODEFILE" \\
            bash -c 'pkill -u $USER -f "python src/train.py" || true'
        sleep 2

        NGPU_PER_HOST={args.gpus_per_node}
        NRANKS=$((NNODES * NGPU_PER_HOST))

        mpiexec --verbose --envall \\
            -n "$NRANKS" --ppn "$NGPU_PER_HOST" \\
            --hostfile "$PBS_NODEFILE" \\
            --cpu-bind depth --depth {args.cpus_per_task} \\
            bash -lc '
        # === Per-rank shell ===
        # `bash -lc` is a login shell — it sources ~/.bash_profile which on
        # ALCF resets cwd to $HOME. cd back to the repo before importing
        # anything (src/train.py uses `from src.data...` which requires
        # repo root on sys.path, and the file is loaded relative to cwd).
        cd {prism_dir}

        # mpiexec exports PMI_* / PALS_*; derive the rank tuple from them.
        export LOCAL_WORLD_SIZE=${{PMI_LOCAL_SIZE:-${{PALS_LOCAL_SIZE:-{args.gpus_per_node}}}}}
        export WORLD_SIZE=${{PMI_SIZE:-${{PALS_SIZE:-1}}}}
        export RANK=${{PMI_RANK:-${{PALS_RANKID:-0}}}}
        export LOCAL_RANK=${{PMI_LOCAL_RANK:-${{PALS_LOCAL_RANKID:-0}}}}
        NODE_RANK=$((RANK / LOCAL_WORLD_SIZE))
        export NODE_RANK

        # CRITICAL on Polaris: pin CUDA_VISIBLE_DEVICES BEFORE any import of
        # mpi4py/torch. The ALCF guide stresses this — setting it after MPI
        # init silently leaves all 4 GPUs visible and breaks NCCL topology.
        export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

        echo "DEBUG: Rank=$RANK Local=$LOCAL_RANK Node=$NODE_RANK World=$WORLD_SIZE GPU=$CUDA_VISIBLE_DEVICES on $(hostname)"
        {launch_line}
        '
        """).strip()

    if args.batch:
        wt = (
            args.walltime
            or target_exp.get("resources", {}).get("walltime")
            or "01:00:00"
        )
        # Polaris debug + debug-scaling queues both cap at 1:00:00. The
        # design YAMLs were sized for Aurora 6h jobs, so a plain default
        # will overshoot and qsub will reject the submission. Clamp here
        # rather than failing at submit time.
        def _hms_to_sec(s: str) -> int:
            parts = [int(p) for p in s.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0)
            h, m, sec = parts[-3], parts[-2], parts[-1]
            return h * 3600 + m * 60 + sec
        queue_caps = {"debug": 3600, "debug-scaling": 3600, "preemptable": 72 * 3600}
        cap = queue_caps.get(args.queue)
        if cap and _hms_to_sec(wt) > cap:
            print(
                f"Walltime {wt} exceeds {args.queue} queue cap (1h); clipping to 01:00:00. "
                f"Pass --walltime explicitly to override.",
                file=sys.stderr,
            )
            wt = "01:00:00"
        header = textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -l select={args.nodes}:system=polaris
            #PBS -l place=scatter
            #PBS -l walltime={wt}
            #PBS -l filesystems={args.filesystems}
            #PBS -q {args.queue}
            #PBS -A {args.project}
            #PBS -k doe
            #PBS -j oe
            #PBS -N {args.id}
            #PBS -o {logs_dir}/
            #PBS -e {logs_dir}/
            """)
    else:
        header = "#!/bin/bash -l\n"

    script_text = f"{header}\n# Polaris PRISM launch — design={design_id} nodes={args.nodes}\n# Generated by tools/launch_polaris.py at {date_str}T{time_str}\n\n{cmd}\n"

    mode = "batch" if args.batch else "interactive"
    jobs_dir = os.path.join(os.getcwd(), "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    script_path = os.path.join(
        jobs_dir,
        f"run_polaris_{args.id}_{date_str}_{time_str}_{mode}.sh".replace("/", "_"),
    )
    with open(script_path, "w") as f:
        f.write(script_text)
    os.chmod(script_path, 0o755)
    print(f"Generated run script: {script_path}")

    if args.dry_run:
        print("--- Dry run: script written but not submitted ---")
        return 0

    if args.batch:
        print("Submitting via qsub...")
        # Forward HF_TOKEN by reference (-v VAR with no value picks it up
        # from the submitting shell's env). The script body references
        # $HF_TOKEN, never the literal value, so this avoids writing the
        # token into the persisted job script on /eagle.
        qsub_cmd = ["qsub"]
        if "HF_TOKEN" in os.environ:
            qsub_cmd += ["-v", "HF_TOKEN"]
        qsub_cmd.append(script_path)
        try:
            result = subprocess.run(
                qsub_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = result.stdout.strip()
            print(f"Submitted: {job_id}")
            print(f"  Logs: {logs_dir}/")
            print(f"  Watch with: tail -f {logs_dir}/{args.id}.o<jobnum>")
        except subprocess.CalledProcessError as e:
            print(f"qsub failed: {e.stderr}", file=sys.stderr)
            return 1
    else:
        # Interactive: execute directly. The script expects to run inside an
        # active qsub -I shell (PBS_NODEFILE must be readable).
        if "PBS_NODEFILE" not in os.environ:
            print(
                "WARNING: PBS_NODEFILE is not set — falling back to single-host mode. "
                "For real interactive runs, qsub -I first.",
                file=sys.stderr,
            )
        try:
            subprocess.run([script_path], check=True)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except subprocess.CalledProcessError as e:
            print(f"Run failed (exit {e.returncode}).", file=sys.stderr)
            return e.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

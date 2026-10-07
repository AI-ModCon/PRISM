import argparse
import datetime
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch_common import load_dotenv, lookup_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Launch PRISM Experiments on Aurora")
    parser.add_argument(
        "--file", default="experiments/prism_designs.yaml", help="Experiment Design YAML file"
    )
    parser.add_argument(
        "--id", required=True, help="Run ID / Job Name (used for output dir, PBS name)"
    )
    parser.add_argument(
        "--design",
        help="Experiment Design ID (from YAML). Defaults to same as --id. "
        "Lets tools/run_sweep.py pass a stable design id while --id varies per cell.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print script instead of executing")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--batch", action="store_true", help="Generate Batch Script (PBS)")
    parser.add_argument("--project", default="ModCon", help="Project Allocation")
    parser.add_argument("--queue", default="debug", help="Queue Name")
    parser.add_argument("--walltime", default=None, help="Walltime (overrides config)")
    parser.add_argument(
        "--packed-env", default="deepspeed_env.tar.gz", help="Path to packed env tarball"
    )
    parser.add_argument(
        "--native-ddp", action="store_true", help="Use native PyTorch DDP (bypasses Accelerate)"
    )
    parser.add_argument(
        "--native-fsdp", action="store_true", help="Use native PyTorch FSDP (bypasses Accelerate)"
    )
    parser.add_argument(
        "--use-shared-venv",
        action="store_true",
        help="Source the shared venv from VENV_PATH in .env instead of unpacking "
        "the tarball to /tmp on every node. Requires VENV_PATH to be set in .env. "
        "Before this PR, VENV_PATH alone triggered shared-venv mode silently; the "
        "flag makes the intent explicit and matches the other Aurora launchers.",
    )
    # GPU count is now derived from experiment config if available, else default to 12
    args, unknown_args = parser.parse_known_args()

    # 1. Output Management
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "outputs", args.id, date_str, time_str)

    env_config = load_dotenv()

    # Defaults from .env or hardcoded fallbacks
    default_prism_dir = env_config.get("PRISM_DIR", "/flare/ModCon/ngetty/BaseMM_PRISM")
    default_hf_home = env_config.get("HF_HOME", "/flare/ModCon/sandeep")
    env_config.get("PROJECT_ALLOCATION", "ModCon")
    env_config.get("QUEUE_NAME", "debug")
    env_config.get("ENV_TARBALL", "deepspeed_env.tar.gz")
    default_shared_hf_home = env_config.get("SHARED_HF_HOME", "/flare/ModCon/sandeep/hub")
    # Shared-venv path: opted into via --use-shared-venv. Reading VENV_PATH from
    # .env without the flag would silently switch behavior, which masked debug
    # confusion in older versions. Now the flag is required.
    env_venv_path = env_config.get("VENV_PATH", "")
    if args.use_shared_venv and not env_venv_path:
        sys.exit(
            "ERROR: --use-shared-venv requires VENV_PATH to be set in .env. "
            "Build the shared venv with: "
            "VENV_PATH=/flare/ModCon/$USER/prism-envs/py3.12 "
            "bash tools/build_aurora_env.sh"
        )
    if env_venv_path and not args.use_shared_venv:
        # Pre-PR-#68 behavior: VENV_PATH alone silently switched to shared-venv mode.
        # Warn so users migrating from old .env files notice the flag is now required.
        print(
            f"WARNING: VENV_PATH={env_venv_path} is set in .env but --use-shared-venv "
            "was not passed. Falling back to tarball mode. Pass --use-shared-venv to "
            "use the shared venv (this is a behavior change from older versions).",
            file=sys.stderr,
        )
    default_venv_path = env_venv_path if args.use_shared_venv else ""

    # 2. Load Experiment Spec
    design_id = args.design if args.design else args.id
    target_exp, _parent_exp, final_overrides = lookup_experiment(args.file, design_id)

    print(f"Found Experiment: {target_exp.get('name', design_id)}")

    # 3. Resources
    resources = target_exp.get("resources", {"ngpus": 12})
    ngpus = resources.get("ngpus", 12)
    print(f"Resources: {ngpus} GPUs/XPUs per node")

    # 4. Proxy Setup for Aurora
    proxy_url = env_config.get("PROXY_URL", "http://proxy.alcf.anl.gov:3128")
    no_proxy_list = env_config.get(
        "NO_PROXY",
        "admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov",
    )

    proxy_env = f"""
# proxy settings
if [[ ! "${{HOSTNAME}}" =~ aurora-uan ]]; then
    export HTTP_PROXY="{proxy_url}"
    export HTTPS_PROXY="{proxy_url}"
    export http_proxy="{proxy_url}"
    export https_proxy="{proxy_url}"
    export ftp_proxy="{proxy_url}"
    export no_proxy="{no_proxy_list}"
fi
"""

    # Propagate Hugging Face Credentials
    if "HF_TOKEN" in os.environ:
        proxy_env += f'\nexport HF_TOKEN="{os.environ["HF_TOKEN"]}"'
    else:
        print(
            "Warning: HF_TOKEN not found in current environment. External models might fail to load."
        )

    # Force HF_HOME to shared parent directory (or from .env)
    proxy_env += f'\\nexport HF_HOME="{default_hf_home}"'

    # Forward env vars the launcher heredoc needs but mpiexec's `bash -lc`
    # would otherwise strip. For each, prefer the caller's os.environ over
    # the .env file so per-cell overrides (set by tools/run_sweep.py via
    # subprocess env=...) take precedence over project defaults.
    #
    # HF_DATASETS_CACHE: HuggingFace's datasets library writes a lockfile
    # to HF_HOME/datasets/ by default, which is often read-only on shared
    # hub directories. Sweep cells then fail with EACCES on load_dataset().
    #
    # DL_NUM_WORKERS: tools/run_sweep.py sets this to 0 for non-image cells
    # so HF IterableDatasets with n_shards < world_size don't get their
    # workers silenced. Source asymmetry between the two vars was a footgun.
    #
    # PRISM_CALVIN_ROOT / PRISM_AURORAGPT_2B_CHECKPOINT /
    # PRISM_OLMO1B_INTERLEAVED_TOKENIZER / PRISM_WALRUS_WEIGHTS_PATH:
    # optional overrides for placeholder path defaults in src/config.py and
    # src/encoders/geometry.py (see .env.template) — read via os.environ.get
    # on the compute node, so unset here means the placeholder is used.
    for var in (
        "HF_DATASETS_CACHE",
        "DL_NUM_WORKERS",
        "PRISM_CALVIN_ROOT",
        "PRISM_AURORAGPT_2B_CHECKPOINT",
        "PRISM_OLMO1B_INTERLEAVED_TOKENIZER",
        "PRISM_WALRUS_WEIGHTS_PATH",
    ):
        val = os.environ.get(var) or env_config.get(var)
        if val:
            proxy_env += f'\\nexport {var}="{val}"'

    # 5. Command Construction
    prism_dir = default_prism_dir

    # Handle overrides
    overrides_list = []

    # From YAML
    for k, v in final_overrides.items():
        overrides_list.append(f"{k}={v}")

    # Extract backbone for local copy
    backbone_id = final_overrides.get("model.backbone_id", "allenai/OLMo-7B-0724-hf")
    hf_model_dir = "models--" + backbone_id.replace("/", "--")
    print(f"Detected backbone_id: {backbone_id}")
    print(f"Targeting HF Cache Dir: {hf_model_dir}")

    # Default Aurora Overrides (if not already present in YAML, though YAML should control)
    if "training.device" not in final_overrides:
        overrides_list.append("training.device=xpu")

    # Ensure ID is passed
    overrides_list.append(f"exp.id={args.id}")

    overrides_list.append(f"hydra.run.dir={output_dir}")

    # Add CLI overrides
    for arg in unknown_args:
        if "=" in arg and not arg.startswith("--"):
            overrides_list.append(arg)
        else:
            print(
                f"Warning: Ignoring unknown argument '{arg}'. (Overrides should be format key=value)"
            )

    overrides_str = " ".join(overrides_list)

    # --- Build distributed mode export ---
    if args.native_fsdp:
        dist_mode_export = "export USE_NATIVE_FSDP=1  # Native PyTorch FSDP"
    elif args.native_ddp:
        dist_mode_export = "export USE_NATIVE_DDP=1  # Native PyTorch DDP (bypasses Accelerate)"
    else:
        dist_mode_export = "# Using Accelerate (default)"

    # --- Build conditional bash sections (avoids nested f-string syntax issues) ---
    if default_venv_path:
        # Shared venv mode: skip tarball, source venv directly
        env_setup_section = (
            "# --- Environment Setup (Shared Venv) ---\n# Using shared venv - no tarball needed\n"
        )
        activate_line = 'source "' + default_venv_path + '/bin/activate"'
        model_section = (
            "# --- Model Cache (shared filesystem) ---\n"
            'export HF_HOME="' + default_hf_home + '"\n'
            'echo "  HF_HOME set to: $HF_HOME (shared filesystem)"\n'
        )
    else:
        # Tarball mode: unpack env to /tmp
        env_tarball_abs = os.path.abspath(args.packed_env)
        env_setup_section = (
            "# --- Environment Setup (Unpack Tarball) ---\n"
            'export ENV_TARBALL="' + env_tarball_abs + '"\n'
            'export LOCAL_ENV="/tmp/deepspeed_env"\n'
            'export MARKER_FILE="$LOCAL_ENV/env_ready"\n'
            "\n"
            "# Rank 0 on each node takes care of unpacking\n"
            'if [ "$LOCAL_RANK" == "0" ]; then\n'
            '    if [ ! -f "$MARKER_FILE" ]; then\n'
            '         echo "Rank $RANK (Local 0): Unpacking $ENV_TARBALL to $LOCAL_ENV..."\n'
            "         mkdir -p $LOCAL_ENV\n"
            "         tar -xzf $ENV_TARBALL -C $LOCAL_ENV\n"
            "         if [ $? -ne 0 ]; then\n"
            '             echo "ERROR: Failed to unpack $ENV_TARBALL"\n'
            "             exit 1\n"
            "         fi\n"
            "         touch $MARKER_FILE\n"
            '         echo "Rank $RANK (Local 0): Unpacking complete."\n'
            "    else\n"
            '         echo "Rank $RANK (Local 0): Environment already unpacked."\n'
            "    fi\n"
            "fi\n"
            "\n"
            "# Wait for unpack completion\n"
            'while [ ! -f "$MARKER_FILE" ]; do\n'
            "    sleep 1\n"
            "done\n"
        )
        activate_line = "source $LOCAL_ENV/bin/activate"
        model_section = (
            "# --- Model Copy (Local /tmp) ---\n"
            'export SHARED_HF_HOME="' + default_shared_hf_home + '"\n'
            'export LOCAL_HF_HOME="/tmp/huggingface/hub"\n'
            'export MODEL_DIR="' + hf_model_dir + '"\n'
            'export MODEL_MARKER="$LOCAL_HF_HOME/model_ready"\n'
            "\n"
            'if [ "$LOCAL_RANK" == "0" ]; then\n'
            "    mkdir -p $LOCAL_HF_HOME\n"
            "    \n"
            '    if [ ! -f "$MODEL_MARKER" ]; then\n'
            '        if [ -d "$SHARED_HF_HOME/$MODEL_DIR" ]; then\n'
            '            echo "Rank $RANK (Local 0): Copying model $MODEL_DIR to /tmp..."\n'
            '            cp -r "$SHARED_HF_HOME/$MODEL_DIR" "$LOCAL_HF_HOME/"\n'
            '            touch "$MODEL_MARKER"\n'
            '            echo "Rank $RANK (Local 0): Model copy complete."\n'
            "        else\n"
            '            echo "Rank $RANK (Local 0): WARNING - Model not found. Relying on download/cache."\n'
            '            touch "$MODEL_MARKER"\n'
            "        fi\n"
            "    else\n"
            '         echo "Rank $RANK (Local 0): Model already present in /tmp."\n'
            "    fi\n"
            "fi\n"
            "\n"
            "# Wait for model copy\n"
            'while [ ! -f "$MODEL_MARKER" ]; do\n'
            "    sleep 2\n"
            "done\n"
            "\n"
            "# Point HF_HOME to the local tmp location\n"
            'export HF_HOME="/tmp/huggingface"\n'
            'echo "  HF_HOME set to: $HF_HOME"\n'
        )

    cmd = f"""
cd {prism_dir}

# --- Environment Variables ---
export ACCELERATE_CONFIG_FILE={prism_dir}/scripts/accelerate_configs/aurora_deepspeed.yaml
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_OP_SYNC=1
export NUMEXPR_MAX_THREADS=512
export NUMEXPR_NUM_THREADS=128

# --- Distributed Mode ---
{dist_mode_export}

# --- Master Addr/Port (Multi-Node Auto-Detect) ---
if [ ! -z "$PBS_NODEFILE" ]; then
    export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
else
    export MASTER_ADDR=$(hostname)
fi

export MASTER_PORT=$((20000 + RANDOM % 20000))

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Output Dir: {output_dir}"

# --- Cleanup Stale Processes ---
echo "Cleaning up stale python processes..."
pkill -u $USER -f "python src/train.py" || true
sleep 2

# --- Execution ---
mpiexec -n {args.nodes * ngpus} bash -lc '
# Load frameworks inside mpiexec so each worker sees IPEX/XPU symbols.
# Pinned to 2025.3.1 (matches shared-venv build target and the validated
# XCCL config). No 2>/dev/null — loud failure beats cryptic import errors.
module use /soft/modulefiles
module load frameworks/2025.3.1

export WORLD_SIZE=${{PMI_SIZE:-${{PMIX_SIZE:-${{PALS_SIZE:-${{OMPI_COMM_WORLD_SIZE:-12}}}}}}}}
export RANK=${{PMI_RANK:-${{PMIX_RANK:-${{PALS_RANKID:-${{OMPI_COMM_WORLD_RANK:-0}}}}}}}}
export LOCAL_RANK=${{PMI_LOCAL_RANK:-${{PMIX_LOCAL_RANK:-${{PALS_LOCAL_RANKID:-${{OMPI_COMM_WORLD_LOCAL_RANK:-0}}}}}}}}
export LOCAL_WORLD_SIZE=${{PMI_LOCAL_SIZE:-${{PMIX_LOCAL_SIZE:-${{PALS_LOCAL_SIZE:-${{OMPI_COMM_WORLD_LOCAL_SIZE:-12}}}}}}}}

echo "DEBUG: Rank=$RANK, Local=$LOCAL_RANK, World=$WORLD_SIZE"

{env_setup_section}
# Activate
export PYTHONNOUSERSITE=1
{activate_line}

# --- PRISM_BUILD_INFO manifest check ---
# Confirm the venv frameworks_module + python_realpath match the
# currently-loaded values. Catches silent staleness from ALCF flipping
# the default frameworks python out from under an old venv.
#
# IMPORTANT: in tarball mode, do NOT trust $VIRTUAL_ENV. The outer launch
# script may have run `source .venv-deepspeed/bin/activate` and set
# VIRTUAL_ENV to the repo-root .venv-deepspeed. The sed-then-source dance
# for the /tmp activate inside mpiexec does NOT always re-export
# VIRTUAL_ENV when bash -lc runs a new login shell. Resolve from $LOCAL_ENV
# explicitly for tarball mode; shared-venv mode keeps $VIRTUAL_ENV.
# NOTE: comments inside this mpiexec heredoc MUST NOT contain apostrophes
# (single quotes). The heredoc is opened with a single quote; any unescaped
# apostrophe terminates the string and the remaining script body executes
# in the outer shell (see launcher_smoke_harness_bug memory entry / PR #33).
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
        echo "ERROR: venv manifest mismatch (frameworks_module): expected $expected_fw, got $actual_fw"
        echo "Rebuild venv against currently-loaded $actual_fw."
        exit 1
    fi
    if [ -n "$expected_py" ] && [ "$expected_py" != "$actual_py" ]; then
        echo "ERROR: venv manifest mismatch (python_realpath): expected $expected_py, got $actual_py"
        echo "Rebuild venv against currently-loaded $actual_fw."
        exit 1
    fi
fi

export MASTER_ADDR='"$MASTER_ADDR"'
export MASTER_PORT='"$MASTER_PORT"'

{model_section}
echo "Rank $RANK/$WORLD_SIZE (Local: $LOCAL_RANK/$LOCAL_WORLD_SIZE)"

python src/train.py {overrides_str}
'
"""

    # 6. Generate Script Content
    if args.batch:
        # Batch Mode - PBS Headers
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
"""
    else:
        # Interactive Mode
        header = "#!/bin/bash"

    run_script_content = f"""{header}
# Aurora Launch Script
# Experiment: {args.id}
# Generated by tools/launch_aurora.py

# --- Module Load ---
module load frameworks/2025.3.1
module load hdf5

# --- Activate Venv ---
if [ -d ".venv-deepspeed" ]; then
    source .venv-deepspeed/bin/activate
fi

{proxy_env}

{cmd}
"""

    mode = "batch" if args.batch else "interactive"
    script_name = f"run_aurora_{args.id}_{date_str}_{time_str}_{mode}.sh"
    # helper to clean up slashes if ID has them
    script_name = script_name.replace("/", "_")

    jobs_folder = "jobs"
    os.makedirs(jobs_folder, exist_ok=True)
    script_path = os.path.join(jobs_folder, script_name)

    # Write script
    with open(script_path, "w") as f:
        f.write(run_script_content)

    os.chmod(script_path, 0o755)
    print(f"Generated Run Script: {script_path}")

    if args.dry_run:
        # print("\n--- SCRIPT CONTENT ---")
        # print(run_script_content)
        pass
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

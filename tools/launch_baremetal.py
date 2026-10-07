import argparse
import datetime
import os
import subprocess
import sys

import yaml

# --- Force Cache Migration ---
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/raid/smadireddy/.cache/huggingface"
print(f"[Launcher] Using HF_HOME: {os.environ['HF_HOME']}")
# ------------------------------

# Wrapper for 8 GPU Pinned (Inserted into script)
PINNED_WRAPPER = """
# Wrapper called once per rank by Accelerate
pin_and_run() {
  case "${LOCAL_RANK:-0}" in
    0) CPUS="24-31,216-223";   NUMA=3  ;;
    1) CPUS="24-31,216-223";   NUMA=3  ;;
    2) CPUS="72-79,264-271";   NUMA=9  ;;
    3) CPUS="72-79,264-271";   NUMA=9  ;;
    4) CPUS="120-127,312-319"; NUMA=15 ;;
    5) CPUS="120-127,312-319"; NUMA=15 ;;
    6) CPUS="168-175,360-367"; NUMA=21 ;;
    7) CPUS="168-175,360-367"; NUMA=21 ;;
    *) echo "bad LOCAL_RANK=${LOCAL_RANK}"; exit 1 ;;
  esac

  echo "Rank ${LOCAL_RANK}: Pinning to CPUs $CPUS (NUMA $NUMA)"
  exec numactl --physcpubind="$CPUS" --membind="$NUMA" \
    .venv/bin/python -u src/train.py "$@"
}
export -f pin_and_run
"""


def generate_command(launcher, ngpus, topology, experiment_id, variant_id, overrides_str):
    cmd_args = f"""exp.id="{experiment_id}" \\
    exp.variant="{variant_id}" \\
    {overrides_str}"""

    if launcher == "torchrun":
        cmd = f""".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node={ngpus} \\
    src/train.py \\
    {cmd_args}"""
        if topology == "4gpu_nvlink":
            return f"export CUDA_VISIBLE_DEVICES=0,1,2,3\n{cmd}"
        return cmd

    elif launcher == "accelerate":
        base_launch = f""".venv/bin/accelerate launch \\
    --multi_gpu \\
    --num_machines 1 \\
    --num_processes {ngpus} \\
    --mixed_precision bf16 \\
    --dynamo_backend no"""

        if topology == "8gpu_pinned":
            # NO PYTHON mode for accelerate to call bash wrapper
            return f"""{PINNED_WRAPPER}

{base_launch} \\
    --main_process_port 29500 \\
    --no_python \\
    bash -lc 'pin_and_run "$@"' bash \\
    {cmd_args}"""

        elif topology == "4gpu_nvlink":
            return f"""export CUDA_VISIBLE_DEVICES=4,5,6,7
{base_launch} \\
    src/train.py \\
    {cmd_args}"""

        elif topology == "4gpu_fsdp":
            # FSDP Launch Flags
            fsdp_launch = """.venv/bin/accelerate launch \\
    --use_fsdp \\
    --fsdp_sharding_strategy FULL_SHARD \\
    --fsdp_offload_params false \\
    --fsdp_state_dict_type FULL_STATE_DICT \\
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \\
    --fsdp_transformer_layer_cls_to_wrap OlmoDecoderLayer \\
    --num_machines 1 \\
    --num_processes 4 \\
    --mixed_precision bf16 \\
    --dynamo_backend no"""

            return f"""export CUDA_VISIBLE_DEVICES=4,5,6,7
{fsdp_launch} \\
    src/train.py \\
    {cmd_args}"""

        elif topology == "1gpu_debug":
            return f"""export CUDA_VISIBLE_DEVICES=4
.venv/bin/accelerate launch \\
    --num_processes 1 \\
    --mixed_precision bf16 \\
    --dynamo_backend no \\
    src/train.py \\
    {cmd_args}"""

        elif topology == "4gpu_deepspeed":
            return f"""export CUDA_VISIBLE_DEVICES=4,5,6,7
.venv-deepspeed/bin/accelerate launch \\
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \\
    src/train.py \\
    {cmd_args}"""

        elif topology == "4gpu_deepspeed_zero3":
            return f"""export CUDA_VISIBLE_DEVICES=4,5,6,7
.venv-deepspeed/bin/accelerate launch \\
    --config_file scripts/accelerate_configs/deepspeed_zero3.yaml \\
    src/train.py \\
    {cmd_args}"""

        else:  # Standard
            return f"""{base_launch} \\
    src/train.py \\
    {cmd_args}"""

    else:
        raise ValueError(f"Unknown launcher: {launcher}")


def main():
    parser = argparse.ArgumentParser(description="Launch PRISM Experiments on Bare Metal")
    parser.add_argument(
        "--file", default="experiments/prism_designs.yaml", help="Experiment Design YAML file"
    )
    parser.add_argument("--id", required=True, help="Experiment ID to run")
    parser.add_argument(
        "--launcher", default="accelerate", choices=["accelerate", "torchrun"], help="Launcher"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print script instead of executing")
    args, unknown_args = parser.parse_known_args()

    # ... (Same parsing logic) ...
    # Check for CLI overrides (key=value)
    cli_overrides = []
    for arg in unknown_args:
        if "=" in arg and not arg.startswith("--"):
            cli_overrides.append(arg)
        else:
            print(
                f"Warning: Ignoring unknown argument '{arg}'. (Overrides should be format key=value)"
            )

    if not os.path.exists(args.file):
        print(f"Error: Design file '{args.file}' not found.")
        sys.exit(1)

    # 1. Load Spec
    with open(args.file) as f:
        data = yaml.safe_load(f)

    # 2. Find Experiment
    target_exp = None
    final_overrides = {}

    for exp in data["experiments"]:
        if exp.get("id") == args.id:
            target_exp = exp
            final_overrides = exp.get("common_overrides", {}).copy()
            final_overrides.update(exp.get("overrides", {}))
            break
        if "variants" in exp:
            parent_overrides = exp.get("common_overrides", {}).copy()
            parent_overrides.update(exp.get("overrides", {}))
            for variant in exp["variants"]:
                if variant.get("id") == args.id:
                    target_exp = variant
                    final_overrides = parent_overrides.copy()
                    final_overrides.update(variant.get("overrides", {}))
                    break
        if target_exp:
            break

    if not target_exp:
        print(f"Error: Experiment ID '{args.id}' not found")
        sys.exit(1)

    print(f"Found Experiment: {target_exp.get('name', args.id)}")

    # 3. Resources & Topology
    resources = target_exp.get("resources", {"ngpus": 4, "walltime": "04:00:00"})
    topology = resources.get("topology", "standard")
    print(f"Topology: {topology} | GPUs: {resources['ngpus']}")

    # 4. Construct Overrides
    overrides_list = []
    for k, v in final_overrides.items():
        overrides_list.append(f"{k}={v}")
    if cli_overrides:
        overrides_list.extend(cli_overrides)

    # 5. Output Management
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "outputs", args.id, date_str, time_str)
    os.makedirs(output_dir, exist_ok=True)
    overrides_list.append(f"hydra.run.dir={output_dir}")
    overrides_str = " \\\n    ".join(overrides_list)

    # 6. Generate Command
    command = generate_command(
        args.launcher,
        resources["ngpus"],
        topology,
        args.id,
        f"{date_str}_{time_str}",
        overrides_str,
    )

    # 7. Generate Run Script
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        print("Error: WANDB_API_KEY is not set. Export it before launching.")
        sys.exit(2)

    # Extract HF_TOKEN if present in overrides
    # Default to user provided token if not in env
    hf_token = os.environ.get("HF_TOKEN")  # Get from environment first, no default
    if not hf_token:  # If not found in environment or empty
        # Check overrides for HF_TOKEN
        for k, v in final_overrides.items():
            if k == "env.HF_TOKEN":
                hf_token = v
                break
    if not hf_token:
        print(
            "Error: HF_TOKEN is not set (and not provided via env.HF_TOKEN override). "
            "Export HF_TOKEN before launching."
        )
        sys.exit(2)

    run_script_content = f"""#!/bin/bash
# PRISM Bare Metal Launch Script
# Experiment: {args.id}
# Topology: {topology}
# Generated by tools/launch_baremetal.py

# --- Environment ---
export WANDB_MODE=online
export WANDB_DIR="{os.path.join(os.getcwd(), 'wandb')}"
export WANDB_API_KEY="{wandb_key}"
export HF_TOKEN="{hf_token}"
export HF_HOME="{os.environ.get('HF_HOME', '/raid/smadireddy/.cache/huggingface')}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- Command ---
echo "Starting Training..."
echo "Output Dir: {output_dir}"

{command}
"""

    script_name = f"run_exp_{args.id}.sh"
    script_name = script_name.replace("/", "_")

    # Output to jobs folder
    jobs_folder = "jobs"
    os.makedirs(jobs_folder, exist_ok=True)
    script_path = os.path.join(jobs_folder, script_name)

    with open(script_path, "w") as f:
        f.write(run_script_content)

    os.chmod(script_path, 0o755)
    print(f"Generated Run Script: {script_path}")

    if args.dry_run:
        print("\n--- SCRIPT CONTENT ---")
        print(run_script_content)
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

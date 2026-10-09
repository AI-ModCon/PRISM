> [!WARNING]
> This document may be outdated. Refer to [aurora_operations.md](../platforms/aurora_operations.md) for current distributed training guidance.

# DeepSpeed Distributed Training Guide

PRISM supports DeepSpeed (ZeRO-2 and ZeRO-3) for efficient distributed training on multi-GPU setups. This is fully integrated into the `launch_baremetal.py` tool.

## 1. Prerequisites (Separate Environment)

Due to dependency conflicts between standard PyTorch and DeepSpeed's specific CUDA compilation requirements, we use a **dedicated virtual environment** for DeepSpeed runs.

### Setup (Automated)

We provide a helper script to build the environment correctly:

```bash
bash tools/setup_deepspeed_env.sh
```

### Setup (Manual)

If you prefer manual control:

```bash
# 1. Create separate venv
python -m venv .venv-deepspeed
source .venv-deepspeed/bin/activate

# 2. Install Dependencies
pip install -r requirements/deepspeed.txt
```

## 2. Configuration (`prism_designs.yaml`)

To use DeepSpeed, simply set the `topology` field in your experiment design resources. The launcher handles the rest.

### Supported Topologies

- `4gpu_deepspeed`: Standard ZeRO-2 (Optimizer State Sharding). Efficient for 4xGPUs.
- `4gpu_deepspeed_zero3`: ZeRO-3 (Param + Grad + Optimizer Sharding). For fitting massive models.

### Example Design

Edit `experiments/prism_designs.yaml`:

```yaml
experiments:
  - id: PRISM-ZONE-A-DEEPSPEED
    resources:
      ngpus: 4
      topology: "4gpu_deepspeed" # <--- Triggers DeepSpeed ZeRO-2
    variants:
      - id: PRISM-DS-TEST
        overrides: { training.batch_size: 4 }
```

## 3. Under the Hood

When you select a DeepSpeed topology, `tools/launch_baremetal.py` automatically:

1.  Switches to `.venv-deepspeed`.
2.  Selects the appropriate Accelerate config file from `scripts/accelerate_configs/`.
    - `scripts/accelerate_configs/deepspeed_zero2.yaml`
    - `scripts/accelerate_configs/deepspeed_zero3.yaml`

### Customizing DeepSpeed Config

If you need to change DeepSpeed settings (e.g., Offload to CPU), edit the corresponding YAML in `scripts/accelerate_configs/`.

## 4. Launching Training

No changes are needed to the launch command. Just reference the experiment ID defined above.

```bash
# Deactivate standard venv first (recommended)
deactivate

# Launch (The script internally calls .venv-deepspeed/bin/python)
python tools/launch_baremetal.py --id PRISM-ZONE-A-DEEPSPEED
```

## 5. Troubleshooting

- **JIT Compilation Errors**: If DeepSpeed fails to compile ops, ensure `nvcc --version` matches the PyTorch CUDA version.
- **OOM on ZeRO-2**: Try switching to `4gpu_deepspeed_zero3` to shard model parameters.
- **"No such file .venv-deepspeed"**: You must create this environment manually first!

## PRISM on Perlmutter

This document is under development and will be expanded as more workflows on Perlmutter are tested.

Author: Patrick Emami (Patrick.Emami@nlr.gov)
Last update: 2/27/2026

## Verified setup 

The verified setup is:

- 1 NVIDIA A100 80GB or 40GB GPU
- PyTorch 2.8.0
- transformers 4.56.2
- cudatoolkit 12.9
- deepspeed 0.17.6

### Getting started

1. Allocate a GPU node on Perlmutter in interactive mode
2. Activate the pre-built PyTorch 2.8.0 module: `module load pytorch/2.8.0`
3. If the first time, install the required Python packages: `pip install -r requirements/perlmutter.txt`

`WANDB_PROJECT` environment variable should be set.

### Encoder-alignment training

#### Current 2/28/2026

- tools/launch_perlmutter.py

#### Old

An older `run_zone_a.py` entry point was used here before the unified launcher. It was removed rather than archived -- `scripts/legacy/` does not exist -- and is superseded by `tools/launch_perlmutter.py` above. `git log` retains it.

1. *Verified* encoder-alignment training with OLMo-7B, non-interleaved model config: `prism-olmo-ts-7b`. The ts_caption dataset `ChengsenWang/TSQA` should be the only dataset with `"skip": false` in the `dataset_configs.json` config file.
2. *Experimental* Interleaved QA training model config: `prism-olmo-ts-1b-interleaved`. The ts_qa dataset `ChatTSRepo/ChatTS-Training-Dataset` should be the only dataset with `"skip": false` in the `dataset_configs.json` config file.


## Troubleshooting

Fix a file lock hang by setting these env variables:

```
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export DEEPSPEED_TRITON_CACHE_DIR="$SCRATCH/ds_triton_cache"
```

## Environment variables

### Core Distributed Setup

| Variable | Purpose | Default |
|----------|---------|---------|
| `USE_NATIVE_DDP` | `"1"` to use env-var-only init (avoids mpi4py/XCCL conflicts) | `"0"` |
| `USE_NATIVE_FSDP` | `"1"` to use native FSDP mode | `"0"` |
| `RANK` / `PMI_RANK` / `PALS_RANKID` | Global rank | `0` |
| `WORLD_SIZE` / `PMI_SIZE` / `PALS_SIZE` | Total number of ranks | `1` |
| `LOCAL_RANK` / `PMI_LOCAL_RANK` / `PALS_LOCAL_RANKID` | Rank within the node | `0` |
| `LOCAL_WORLD_SIZE` | Ranks per node (used by HSDP) | `12` |
| `MASTER_ADDR` | Master node address (for `tcp://` init) | `"localhost"` |
| `MASTER_PORT` | Master node port | `"29500"` |
| `ZE_AFFINITY_MASK` | When set, each rank sees only its GPU as `xpu:0` | (unset) |

### Distribution Strategy

| Variable | Purpose | Default |
|----------|---------|---------|
| `DIST_STRATEGY` | `"ddp"`, `"fsdp"`, or `"hsdp"` | `"ddp"` |
| `FSDP_SHARDING` | `"full_shard"`, `"shard_grad_op"`, `"no_shard"`, `"hybrid_shard"` | `"full_shard"` |
| `FSDP_CPU_OFFLOAD` | `"1"` to enable CPU offload in FSDP | `"0"` |

### DDP-Specific

| Variable | Purpose | Default |
|----------|---------|---------|
| `DDP_BUCKET_CAP_MB` | Gradient bucket size in MB | `"25"` |
| `PRISM_DDP_FIND_UNUSED` | `"1"` to enable `find_unused_parameters` (disables `static_graph`) | `"0"` |

### Performance & Debugging

| Variable | Purpose | Default |
|----------|---------|---------|
| `TORCH_COMPILE` | `"1"` to enable `torch.compile` before wrapping | `"0"` |
| `TORCH_COMPILE_BACKEND` | Compile backend (e.g., `"inductor"`) | `"inductor"` |
| `GRAD_CKPT_FREQ` | Gradient checkpointing frequency (`0`=off, `1`=every layer, `2`=every other) | `"1"` |
| `DEBUG_SYNC` | `"1"` to synchronize XPU after every forward/backward phase | `"0"` |
| `ENABLE_PROFILER` | `"1"` to enable PyTorch profiler | `"0"` |
| `PROFILER_STEPS` | Comma-separated steps to profile | `"5,10,15"` |
| `PER_RANK_TIMING` | `"1"` to enable per-rank straggler detection | `"0"` |

### Data Pipeline

| Variable | Purpose | Default |
|----------|---------|---------|
| `USE_MULTI_DATASET` | `"1"` to use `MultiWebDatasetWrapper` from DAOS | `"0"` |
| `DAOS_MOUNT` | DAOS mount point path | (unset) |
| `DATASET_GROUPS` | Comma-separated dataset groups (e.g., `"pixmo,cosyn"`) | `"all"` |
| `DATASET_CONFIG` | Path to DAOS dataset YAML config | `"src/conf/data/daos_datasets.yaml"` |
| `DATASET_PROPORTIONS` | Override mixing weights (e.g., `"pixmo:0.5,cosyn:0.3"`) | `""` |
| `USE_BUCKETING` | `"1"` to use length-sorted bucketed batching | `"0"` |
| `BUCKET_BUFFER_SIZE` | Buffer size for bucketed batching | `"1000"` |
| `USE_BUCKETED_COLLATOR` | `"1"` to use `BucketedCollator` | `"1"` |
| `MAX_SEQ_LENGTH` | Hard cap on sequence length in collator | `"2048"` |
| `LOCAL_SHARDS_DIR` | Path to local webdataset `.tar` shards | `""` |
| `HF_TOKEN` | HuggingFace token for streaming datasets | (unset) |

### Logging & External Services

| Variable | Purpose | Default |
|----------|---------|---------|
| `WANDB_MODE` | WandB mode (`"online"`, `"offline"`, `"disabled"`) | set from config |
| `WANDB_ENTITY` | WandB entity/team | set from config |
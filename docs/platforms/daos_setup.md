# DAOS Setup for PRISM Training

This guide covers setting up DAOS storage for PRISM multimodal training on Aurora.

For runtime DAOS operations during training, see [aurora_operations.md](aurora_operations.md#daos-operations-runtime).
For DAOS-related performance results, see [scaling_study.md](../results/scaling_study.md#daos-vs-webdataset-throughput-with-static_graph-fix).

## Overview

DAOS provides high-performance storage with up to 30 TB/s throughput. By storing WebDataset shards on DAOS, we eliminate per-job staging overhead (copying from /flare to /tmp).

## Prerequisites

- DAOS pool: `AuroraGPT` (already allocated)
- Pool space: 253 TB free on NVMe tier
- Containers: `prism_training_data` (datasets), `prism_models` (HuggingFace weights)
- Job must include `-l filesystems=daos_user_fs` (required for DAOS agent)

## Quick Start

### 1. Initial Setup (One-Time)

```bash
# Load DAOS module
module use /soft/modulefiles
module load daos

# Create container
./scripts/setup_daos_container.sh create

# Mount on login node
./scripts/setup_daos_container.sh mount
./scripts/setup_daos_models.sh mount
# Copy initial dataset (pixmo_cap - already in WebDataset format)
./scripts/setup_daos_container.sh copy-pixmo

# Unmount when done
./scripts/setup_daos_container.sh unmount
```

### 2. Prepare Additional Datasets

**Dataset Status:**

| Dataset | Hydrated? | Format | Action Needed |
|---------|-----------|--------|---------------|
| pixmo_cap | ✅ Yes | WebDataset | Ready - copy to DAOS |
| CoSyn-point | ✅ Yes | Parquet w/ bytes | Convert to WebDataset |
| pixmo-points | ❌ No | URLs only | Hydrate first, then convert |
| pixmo-count | ❌ No | URLs only | Hydrate first, then convert |

**For already-hydrated datasets (CoSyn-point):**
```bash
# Convert directly to WebDataset
python scripts/convert_parquet_to_webdataset.py \
    --input-dir /flare/ModCon/ngetty/data/zone_a/CoSyn-point/data \
    --output-dir /flare/ModCon/ngetty/data/zone_a/CoSyn-point_webdataset \
    --dataset-name cosyn-point
```

**For non-hydrated datasets (pixmo-points, pixmo-count):**
```bash
# Step 1: Hydrate (download images) - pixmo-count is smaller, good for testing
python scripts/hydrate_pixmo_pointing.py \
    --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count/data \
    --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count-hydrated \
    --workers 32

# Step 2: Convert hydrated dataset to WebDataset
python scripts/convert_parquet_to_webdataset.py \
    --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count-hydrated \
    --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-count_webdataset \
    --dataset-name pixmo-count

# For pixmo-points (1.2M images - will take hours)
python scripts/hydrate_pixmo_pointing.py \
    --input-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points/data \
    --output-dir /flare/ModCon/ngetty/data/zone_a/pixmo-points-hydrated \
    --workers 64
```

**Then copy to DAOS:**
```bash
./scripts/setup_daos_container.sh mount
cp -r /flare/ModCon/ngetty/data/zone_a/CoSyn-point_webdataset \
    /tmp/$USER/AuroraGPT/prism_training_data/
./scripts/setup_daos_container.sh unmount
```

### 3. Launch Training with DAOS

```bash
# Generate and submit job
python tools/launch_aurora_daos.py \
    --id PRISM-DAOS-TEST \
    --design PRISM-AURORA-ZONE-A \
    --packed-env deepspeed_env.tar.gz \
    --nodes 2 \
    --batch \
    --queue debug
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/setup_daos_container.sh` | Create/mount/manage data container |
| `scripts/setup_daos_models.sh` | Create/mount/manage models container |
| `scripts/daos_mount_helper.sh` | Unified DAOS mount helper (used by launchers) |
| `scripts/hydrate_pixmo_pointing.py` | Download images for pixmo-points/count |
| `scripts/convert_parquet_to_webdataset.py` | Convert Parquet datasets to WebDataset |
| `tools/launch_aurora_daos.py` | Launch jobs with DAOS backend |
| `src/conf/data/daos_datasets.yaml` | Dataset group configuration |

## Architecture

```
DAOS Pool: AuroraGPT
├── Container: prism_training_data
│   ├── pixmo/                     # Pixmo datasets (WebDataset format)
│   │   ├── pixmo_cap_webdataset/
│   │   │   ├── manifest.json
│   │   │   ├── shards/
│   │   │   │   ├── pixmo-000000.tar
│   │   │   │   └── ... (613 shards)
│   │   │   └── val_shards/
│   │   ├── pixmo_points_webdataset/
│   │   └── pixmo_count_webdataset/
│   ├── s1mmalign/                 # S1MMA alignment datasets
│   ├── cosyn/                     # CoSyn datasets
│   │   └── cosyn_point_webdataset/
│   └── nemotron/                  # Nemotron datasets
│
└── Container: prism_models
    └── hub/                       # HuggingFace cache structure
        ├── models--allenai--OLMo-1B-0724-hf/
        ├── models--allenai--OLMo-7B-0724-hf/
        ├── models--google--siglip2-base-patch16-224/
        └── ...
```

### Dataset Groups

Datasets are organized by group for selective loading via `--dataset-groups`:

| Group | Contents | Use Case |
|-------|----------|----------|
| `pixmo` | Pixmo-cap, pixmo-points, pixmo-count | Projector training (fast) |
| `projector` | pixmo + s1mmalign + cosyn + nemotron | Full projector dataset mix |
| `all` | All available datasets | Mixed training |
| `quick` | Small subset | Debug/testing |

## Models Container (prism_models)

The `prism_models` container stores HuggingFace model weights on DAOS for fast, shared access across nodes without Lustre staging.

### Setup

```bash
# Create and mount the models container
./scripts/setup_daos_models.sh mount

# Models are accessed at:
# /tmp/${USER}/AuroraGPT/prism_models/hub/
```

### Symlink Strategy

The launcher creates symlinks from the standard HuggingFace cache to the DAOS mount:
```bash
# Created automatically by launch_aurora_daos.py:
/tmp/huggingface/hub/ -> /tmp/${USER}/AuroraGPT/prism_models/hub/
```

This lets `from_pretrained()` find models via the normal HF cache path without modification. Loading from DAOS is instant vs. 4.5 min for a Lustre copy of 7B weights.

### Uploading New Models

```bash
# Mount the models container
./scripts/setup_daos_models.sh mount

# Copy model from Lustre cache to DAOS (preserving HF directory structure)
cp -r /flare/ModCon/ngetty/huggingface/hub/models--<org>--<model> \
    /tmp/${USER}/AuroraGPT/prism_models/hub/

# Unmount
fusermount3 -u /tmp/${USER}/AuroraGPT/prism_models
```

### Case-Sensitive Model Names

HuggingFace model IDs are case-sensitive but some model directories on DAOS may have inconsistent casing. The launcher includes a `find -iname` fallback for case-insensitive lookup when the exact directory name doesn't match.

---

## PBS Job Configuration

The launch script automatically adds:
```bash
#PBS -l filesystems=home:flare:daos_user_fs
```

This ensures DAOS agent is available on compute nodes.

## Performance Features

1. **Interception Library**: `LD_PRELOAD=/usr/lib64/libpil4dfs.so`
   - Kernel-bypass I/O for improved read performance
   - Automatically set by launch script

2. **No Staging**: Shards read directly from DAOS mount
   - Eliminates 30s × N node stagger delay
   - Saves /tmp space on compute nodes

3. **Erasure Coding**: `EC_16P3GX` with 3-way redundancy
   - Protects against up to 3 server failures
   - Optimal for large sequential reads

## Troubleshooting

### Check DAOS Status
```bash
module load daos
daos pool query AuroraGPT
daos cont list AuroraGPT
```

### Verify Mount
```bash
mount | grep dfuse
ls /tmp/$USER/AuroraGPT/prism_training_data/
ls /tmp/$USER/AuroraGPT/prism_models/hub/
```

### Container Health
```bash
daos container get-prop AuroraGPT prism_training_data
```

If status shows "UNCLEAN":
```bash
daos cont set-prop AuroraGPT prism_training_data status:HEALTHY
daos fs check --flags=evict AuroraGPT prism_training_data
```

### `libpil4dfs.so` Issues (CRITICAL)

The DAOS interception library (`LD_PRELOAD=/usr/lib64/libpil4dfs.so`) provides kernel-bypass I/O for DAOS reads. However, it also intercepts file descriptor operations used by XCCL/OFI transport, which causes:

- **FSDP AllGather hangs** — collectives never complete
- **Python subprocess issues** — interferes with process management

**Fix**: Always use `--no-pil4dfs` when running with FSDP or when experiencing unexplained hangs. This is the default in production configs. See DAOS bug [DAOS-17499](https://daosio.atlassian.net/browse/DAOS-17499).

### `glob.glob()` Hangs on dfuse Mounts

Python's `glob.glob()` can hang or be extremely slow on dfuse-mounted directories due to metadata overhead.

**Fix**: Use `os.listdir()` + manual filtering instead:
```python
# BAD: Can hang on dfuse
shards = glob.glob("/tmp/AuroraGPT/prism_training_data/pixmo/*.tar")

# GOOD: Works reliably on dfuse
shard_dir = "/tmp/AuroraGPT/prism_training_data/pixmo"
shards = [os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".tar")]
```

### "DAOS agent not found"

```
Failed to connect to /var/run/daos_agent/daos_agent.sock
```
Job was not submitted with `-l filesystems=daos_user_fs`. Re-submit with this flag.

### Slow Model Loading (4+ min)

Models container not mounted. Mount it:
```bash
./scripts/setup_daos_models.sh mount
```
Without the models container, the launcher falls back to copying from Lustre (`/flare/ModCon/ngetty/huggingface/hub/`), which takes ~4.5 min for 7B weights.

## Comparison: Lustre Staging vs DAOS

| Metric | Lustre + /tmp Staging | DAOS Direct |
|--------|----------------------|-------------|
| Startup time (8 nodes) | ~4 min | ~30 sec |
| /tmp usage per node | ~40 GB | 0 GB |
| I/O bandwidth | Limited by contention | Up to 30 TB/s |
| Data redundancy | None (single copy) | 3-way erasure coding |

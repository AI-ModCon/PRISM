---
name: prism-daos-storage
description: >
  Use DAOS storage for PRISM on Aurora. Use when creating/mounting DAOS
  containers, staging datasets or model weights to DAOS, declaring the PBS
  filesystems directive, or debugging a DAOS-related hang (FSDP AllGather stall,
  libpil4dfs). Triggers: "DAOS container", "setup_daos", "--no-pil4dfs",
  "daos_user_fs", "AllGather hang", "prism_training_data / prism_models",
  "mount the data on Aurora".
compatibility: Aurora; requires DAOS pool access (not all contributors have it).
metadata:
  version: "1.0"
  project: prism
  facility: alcf
---

# PRISM — DAOS Storage (Aurora)

DAOS is the **fastest** storage path for PRISM (data + model weights on NVMe),
but it's optional — contributors without pool access use Lustre/`/flare` via the
`webdataset-staged` or `lustre` launchers (see prism-launching-jobs). DAOS has
not been exercised for non-vision modalities.

## Layout

- **Pool:** `AuroraGPT` (pre-allocated).
- **Containers:** `prism_training_data` (datasets), `prism_models` (HF weights).

## Setup (login node)

```bash
module use /soft/modulefiles
module load daos

./scripts/setup_daos_container.sh create     # one-time container create
./scripts/setup_daos_container.sh mount      # mount data container
./scripts/setup_daos_models.sh mount         # mount models container
./scripts/setup_daos_container.sh copy-pixmo # stage pixmo (already WebDataset)
./scripts/setup_daos_container.sh unmount    # when done
```

Stage other datasets by mounting, then `cp -r` from `/flare` into
`/tmp/$USER/AuroraGPT/prism_training_data/`, then unmounting. For models,
symlink `/tmp/huggingface/hub/` → `/tmp/$USER/AuroraGPT/prism_models/hub/`
(instant vs ~4.5 min Lustre copy).

## Launch against DAOS

```bash
python tools/launch_aurora_daos.py \
    --id MY-RUN --design PRISM-OLMO3-E2E-PROD \
    --nodes 2 --batch --queue debug-scaling \
    --dist-strategy fsdp --no-pil4dfs
```

The launcher auto-adds `#PBS -l filesystems=home:flare:daos_user_fs` for compute-
node DAOS agent access.

## The two rules that prevent lost days

1. **`--no-pil4dfs` is mandatory for FSDP over DAOS.** `libpil4dfs.so`
   (`LD_PRELOAD`) does kernel-bypass DAOS I/O but intercepts the file
   descriptors XCCL/OFI uses → **FSDP AllGather hangs** and subprocess breakage
   (DAOS bug **DAOS-17499**). The DAOS launcher auto-enables `--no-pil4dfs` for
   ZeRO-3; set it explicitly for any FSDP run. If a job hangs at the first
   collective with no error, this is the first suspect.

2. **Don't request `daos_user_fs` when DAOS is down.** Default the PBS
   `filesystems=` to `home:flare`; only add `daos_user_fs` when the job actually
   reads DAOS. Otherwise the job queues *indefinitely* whenever the DAOS service
   is unavailable — a silent, confusing stall.

## Related gotcha

`glob.glob()` hangs on dfuse/DAOS mounts — PRISM's data code uses `os.listdir()`
instead. If you write shard-discovery code touching a DAOS path, do the same. See
prism-data-pipeline.

## See also

- Deep reference: [`docs/platforms/daos_setup.md`](../../platforms/daos_setup.md) — containers, data
  prep, model staging, full troubleshooting.
- [prism-launching-jobs](../prism-launching-jobs/SKILL.md), [prism-data-pipeline](../prism-data-pipeline/SKILL.md),
  [prism-distributed-strategy](../prism-distributed-strategy/SKILL.md).

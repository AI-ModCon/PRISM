---
name: prism-launching-jobs
description: >
  Launch PRISM training jobs on Aurora. Use when submitting a training run,
  choosing a launcher / storage backend (DAOS vs Lustre vs WebDataset-staged),
  setting key flags (--design, --max-seq-length, --fsdp-production-mode,
  --grad-ckpt-freq), running an interactive two-terminal session, picking a PBS
  queue, or budgeting walltime. Triggers: "launch a job", "submit training",
  "which launcher", "launch_aurora", "--design", "debug-scaling", "run a smoke".
compatibility: Aurora login node (UAN). Never run train.py / mpiexec directly.
metadata:
  version: "1.0"
  project: prism
  facility: alcf
---

# PRISM — Launching Training Jobs (Aurora)

**You are on a login node (UAN): no GPU, no XPU, no MPI.** Never run `train.py`,
`mpiexec`, or GPU code directly — always go through a launcher under `tools/`.
For generic PBS/Aurora background, see the installed `pbs` and `aurora` skills;
this skill is about PRISM's launchers specifically. For queue-time discipline,
see `hpc-iteration-discipline`.

## 1. Pick the launcher by where the data lives

The single decision is **storage backend**. Use the unified dispatcher and let
`--storage` pick the per-backend launcher (it `exec`s the right one and forwards
all remaining flags verbatim):

```bash
python tools/launch_aurora_unified.py --storage {daos|lustre|webdataset-staged} ...
```

| `--storage` | Backing launcher | When |
|-------------|------------------|------|
| `daos` | `tools/launch_aurora_daos.py` | Data + models on a DAOS container. Fastest; Neil's VLM-scaling setup. Requires DAOS access. |
| `webdataset-staged` | `tools/launch_aurora_web.py` | Data on Lustre/`/flare`; launcher stages WebDataset shards to `/tmp` per node. |
| `lustre` | `tools/launch_aurora.py` | Generic; no DAOS, no staging. |

> **DAOS is not the default.** Contributors without DAOS access, and
> non-vision modalities, use `webdataset-staged` or `lustre`. You can call a
> per-backend launcher directly; `--help` shows its full flag surface.

## 2. Canonical commands

```bash
# DAOS-backed E2E (production shape)
python tools/launch_aurora_daos.py \
    --id MY-RUN --design PRISM-OLMO3-E2E-PROD \
    --nodes 2 --batch --queue debug-scaling \
    --dist-strategy fsdp --fsdp-sharding full_shard \
    --max-seq-length 1024 \
    --dataset-groups projector --use-bucketing \
    --no-pil4dfs --fsdp-production-mode

# WebDataset / Lustre fallback (no DAOS)
python tools/launch_aurora_web.py \
    --id MY-RUN --design PRISM-IMAGE-ONLY-7B \
    --nodes 2 --batch --queue debug-scaling \
    --webdataset-dir /flare/<project>/path/to/shards

# Inspect the generated PBS script WITHOUT submitting
python tools/launch_aurora_daos.py --id TEST --dry-run --nodes 1
```

Always `--dry-run` a new config first and read the generated PBS script.

## 3. Load-bearing flags (defaults bite)

| Flag | Why it matters |
|------|----------------|
| `--design` | Selects from `experiments/prism_designs.yaml`. See prism-configuration. |
| `--max-seq-length 1024` | **Critical for E2E** — 2048 OOMs. Default is 2048. |
| `--fsdp-production-mode` | Skips non-essential `synchronize()` calls (+6.6%). |
| `--grad-ckpt-freq` | Defaults to 1 (every layer) — **required for seq=1024**. Do NOT set to 2 (OOM). |
| `--no-pil4dfs` | **Mandatory for FSDP over DAOS** (libpil4dfs → AllGather hang, DAOS-17499). Auto-enabled for ZeRO-3. See prism-daos-storage. |
| `--use-bucketing` | Sequence-length bucketing → ~3× throughput on mixed-length data. |
| `--dist-strategy {ddp,fsdp,hsdp}` | See prism-distributed-strategy for the choice + BS ceilings. |

## 4. Interactive jobs need two terminals

1. **Terminal 1** — get the PBS allocation (`qsub -I ...` or a hold script under
   `tools/hold_*.sh`).
2. **Terminal 2** — from UAN, run the launcher with `--hosts <node>
   --run-via-ssh` (derive `<node>` from `qstat -f <jobid> | grep exec_host`).

**Never manually build SSH/mpiexec commands** — the launcher handles PALS rank
derivation, `ZE_AFFINITY_MASK`, CCL env, and GPU binding.

## 5. Node ownership & queue rules (hard-won)

- **Only use nodes from a job YOU submitted this session.** Verify the job ID
  before every launch. Never derive a node from a shared file like
  `hold_one_nodefile.txt` — it persists across jobs; use `qstat -f <jobid> |
  grep exec_host` instead.
- **Queue:** `debug` / `debug-scaling` for smokes; `capacity` for long jobs
  <256 nodes; `prod` **requires a 256-node minimum** and will reject a 4-node
  job.
- **Walltime:** scale from *(per-cell timing × cells)* and pad ~1.5×, not 4–6×.
  Oversized walltimes hurt scheduling and mask hangs.
- **DAOS down?** Default PBS `filesystems=home:flare`; only add `daos_user_fs`
  when the job actually reads DAOS, or it queues indefinitely.

## 6. Gotchas

- Killing a hung job: `pkill -9 python3; sleep 10` to clear stale XCCL state.
- Generated PBS/heredoc scripts have been silently broken by a stray apostrophe
  in a comment (only rank 0 starts). Run `bash -n` on the generated script
  before submitting. See the `shell-quoting-traps` skill.
- Launching from a worktree? PBS scripts need a `${PBS_O_WORKDIR:-...}` fallback
  and correct `LAUNCHER_PRISM_DIR` for the venv tarball — hardcoded bare-clone
  paths fail in ~1s.

## See also

- Deep reference: [`docs/platforms/aurora_operations.md`](../../platforms/aurora_operations.md) — full
  launcher reference, env-var table, session debugging.
- [prism-daos-storage](../prism-daos-storage/SKILL.md), [prism-configuration](../prism-configuration/SKILL.md),
  [prism-distributed-strategy](../prism-distributed-strategy/SKILL.md).
- Generic installed skills: `pbs`, `aurora`, `hpc-iteration-discipline`, `shell-quoting-traps`.

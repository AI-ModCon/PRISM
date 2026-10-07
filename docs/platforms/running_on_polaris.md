# PRISM on Polaris (NVIDIA A100, CUDA)

Polaris is ALCF's NVIDIA A100 system — 4× A100 (40GB) per node, PBSPro
scheduler, /eagle lustre filesystem. This page covers the CUDA codepath
counterpart to [Aurora operations](aurora_operations.md).

> Status (May 2026): launcher + env build are in place; smokes in progress.
> Throughput numbers below will be filled in as the first runs land.

## Verified setup

- Polaris (40-rack A100 system, 4× A100 40GB per node, NVLink intra-node,
  Slingshot 11 inter-node)
- ALCF `conda/2025-09-28` module (torch + CUDA + NCCL + DeepSpeed + Accelerate)
- venv overlay built by `tools/setup_polaris_env.sh`
- Project: `ModCon`; queues: `debug` (1-2 nodes), `debug-scaling` (1-10 nodes)
- Project storage: `/eagle/ModCon/ngetty/` mirrors the Aurora `/flare/ModCon/ngetty/`
  layout (`envs/ models/ datasets/ logs/ repos/ jobs/`)

## One-time setup

```bash
ssh polaris
cd /eagle/ModCon/ngetty/repos
git clone git@github.com:AI-ModCon/BaseMM_PRISM.git    # or rsync from Aurora
cd BaseMM_PRISM

# Build the venv on top of the conda base module (uses uv if available)
PRISM_VENV=/eagle/ModCon/ngetty/envs/.venv-polaris \
    bash tools/setup_polaris_env.sh
```

The build script:
- Loads `conda/2025-09-28` and activates the base env
- Verifies torch/CUDA/DeepSpeed/transformers/accelerate import cleanly
- Creates the venv with `--system-site-packages` (so we inherit the
  ALCF-built CUDA-aware torch / NCCL)
- Installs `requirements/polaris.txt` (Hydra, webdataset, modality
  dependencies, mpi4py)
- Installs `src/libs/walrus` editable
- Records build provenance in `$PRISM_VENV/PRISM_BUILD_INFO`

## Launching training

`tools/launch_polaris.py` mirrors `tools/launch_aurora_web.py`'s flag
surface where it makes sense — `--design`, `--nodes`, `--batch`,
`--dist-strategy`, `--use-accelerate`, `--max-steps`, `--dry-run`. The
Aurora-only flags (DAOS, CCL, ZE_AFFINITY_MASK, HSN suffix) are dropped
or replaced with NCCL/CUDA equivalents.

```bash
# Dry-run inspection (writes a PBS script, doesn't submit)
python tools/launch_polaris.py \
    --id POLARIS-DDP-1N --design PRISM-IMAGE-ONLY-2N \
    --nodes 1 --batch --queue debug --max-steps 50 \
    --enable-all-modalities \
    --dry-run

# Submit a batch job
python tools/launch_polaris.py \
    --id POLARIS-DDP-1N --design PRISM-IMAGE-ONLY-2N \
    --nodes 1 --batch --queue debug --max-steps 50 \
    --enable-all-modalities

# DeepSpeed ZeRO-2 (forces --use-accelerate)
python tools/launch_polaris.py \
    --id POLARIS-DSZ2-1N --design PRISM-IMAGE-ONLY-2N \
    --nodes 1 --batch --queue debug --deepspeed-zero 2 --max-steps 50
```

The launcher always sets `training.device=cuda` on the Hydra CLI to win
over the design YAML's `training.device: "xpu"` default.

## Distributed backend selection

`src/training/distributed.py` picks the torch.distributed backend
automatically:

| Accelerator | Backend |
|---|---|
| Aurora XPU available | `xccl` |
| NVIDIA CUDA available | `nccl` |
| neither | `gloo` |

Override with `DIST_BACKEND=<backend>` if you need to force one (e.g.
`gloo` for CPU debugging). The Polaris launcher exports
`DIST_BACKEND=nccl` explicitly so behavior doesn't depend on the
order modules are loaded.

## Data on /eagle

Polaris doesn't mount `/flare`; shards must be staged to `/eagle`. The
recommended layout:

```
/eagle/ModCon/ngetty/
  datasets/<dataset>/manifest.json   # WebDataset manifest
  datasets/<dataset>/shards/*.tar
  models/<hf-org--hf-model>/         # HF snapshot dirs
  envs/.venv-polaris/                # built by setup_polaris_env.sh
  repos/BaseMM_PRISM/                # checkout
  logs/<DESIGN>/                     # PBS stdout/stderr
```

For one-off shard transfer from Aurora, the simplest path is `scp`
through the user's UAN session (Globus is appropriate for full-dataset
moves but overkill for 1-2 shards). Example: from an Aurora UAN with a
live ControlMaster to Polaris,

```bash
scp /flare/ModCon/ngetty/data/zone_a/pixmo_cap_webdataset/shards/pixmo-00000{0,1}.tar \
    polaris:/eagle/ModCon/ngetty/datasets/pixmo_cap_smoke/shards/
```

## Throughput comparison vs. Aurora

Effective batch = `batch_size × grad_accum × world_size`. PRISM-IMAGE-ONLY-2N
uses `batch_size=8, grad_accum=4`. Polaris has 4× A100/node, Aurora has 12
XPU tiles/node.

All matched runs: PRISM-IMAGE-ONLY-2N (OLMo-1B + SigLIP2), pixmo_cap
WebDataset, native DDP, `--fsdp-production-mode --max-steps 30`. Eff.
batch = `8 × grad_accum=4 × world_size`. Aurora uses 12 XPU tiles/node;
Polaris uses 4 A100s/node, so eff. batch differs (Aurora does 3× more
samples per step).

| Nodes | System | Ranks/node | Eff. batch | Samp/s | Per-rank | Loss @ step 30 |
|---|---|---|---|---|---|---|
| 1 | Aurora (XPU) | 12 | 384 | **316.5** | 26.4 | 1.50 |
| 1 | Polaris (A100) | 4 | 128 | **289.1** | 72.3 | 9.25 |
| 2 | Aurora (XPU) | 12 | 768 | **514.8** | 21.5 | 1.88 |
| 2 | Polaris (A100) | 4 | 256 | **549.1** | 68.6 | 7.26 |
| 1 | Polaris DS-Z2 | 4 | 128 | **125.0** | 31.3 | 8.94 |

(Aurora jobs `8505044`/`8505045`, Polaris jobs `7166239`/`7166244`/`7166254`,
all run 2026-05-23.)

**Per-rank reading:** Polaris A100 is ~2.7× faster than an Aurora Max
1550 tile (72 vs 26 samp/s), which tracks the BF16 hardware ratio
(~312 vs ~104 TFLOPS). **Per-node reading:** Aurora wins 1N (316 vs 289)
because 12 tiles × 26 > 4 GPUs × 72 by ~9%; Polaris edges Aurora at 2N
(549 vs 515) because Polaris's communication scaling is cleaner (95% vs
81% efficiency) on a small-trainable-params workload (5.7M params,
collective-bound). Aurora's per-node BF16 advantage shows up at 7B+ E2E
where compute and HBM dominate — Aurora hits 122 samp/s on OLMo-3-7B
2N HSDP, which a single Polaris node would not fit at all (4×40GB <
shard requirements).

The loss values diverge because the two systems consume different
effective batches per step (Aurora 384/768 vs Polaris 128/256) so they
see different total tokens by step 30. For convergence comparison match
total tokens-seen, not step counts.

A100 40GB on Polaris caps batch sizes lower than Aurora's 64GB Max 1550
tiles; expect smaller `training.batch_size` headroom on Polaris for the
same design. Loss is converging on the smoke (10.85 → 9.25 from step 10
→ step 30), confirming the data + model + loss path is correct.

The first round of smokes (2026-05-23) shows the path is functional:

```
# 1-node, native DDP (job 7166239)
Step 10: Loss 10.8484 [THROUGHPUT] 236.3 samples/sec (effective batch: 128)
Step 20: Loss 10.5628 [THROUGHPUT] 274.3 samples/sec
Step 30: Loss  9.2531 [THROUGHPUT] 289.1 samples/sec

# 2-node, native DDP (job 7166244)
Step 10: Loss 10.4482 [THROUGHPUT] 451.7 samples/sec (effective batch: 256)
Step 20: Loss  9.9907 [THROUGHPUT] 520.7 samples/sec
Step 30: Loss  7.2594 [THROUGHPUT] 549.1 samples/sec
                     # 95% of perfect-linear (2 × 289 = 578)

# 1-node, DeepSpeed ZeRO-2 (job 7166254)
Training Zone A: 100%|██████| 30/30 [00:31<00:00, 1.02s/it, loss=8.94]
                     # ~125 samp/s — DS overhead dominates at <100M trainable
                     # params. Numbers should flip for 7B+ E2E.
```

## Polaris-specific gotchas

- **`/flare` is not mounted.** Anything in the Aurora `.env`
  pointing at `/flare/...` (`PRISM_DIR`, `HF_HOME`, `SHARED_HF_HOME`,
  `VENV_PATH`) needs an `--<flag>` override or a separate `.env`. The
  launcher skips `.env` when cwd is under `/eagle` to avoid this trap.
- **`CUDA_VISIBLE_DEVICES` must be set before any `mpi4py` import.** If
  the rank shell sets it after MPI init, NCCL silently uses the wrong
  topology. The Polaris launcher sets it inside the per-rank `bash -lc`
  *before* invoking `python`.
- **Per-rank `CUDA_VISIBLE_DEVICES=$LOCAL_RANK` pins one GPU per rank.**
  With that pin, the visible GPU is always `cuda:0` from torch's view,
  so `torch.cuda.set_device(local_rank)` would raise "invalid device
  ordinal". `setup_distributed()` auto-detects single-device CVD and
  uses `cuda:0` + `set_device(0)` in that case. (Same shape as
  Aurora's `ZE_AFFINITY_MASK` handling for XPU.)
- **`init_process_group(device_id=…)` is REQUIRED for NCCL** when each
  rank only sees one CUDA device. Without it NCCL emits a
  `"using GPU 0 as device used by this process is currently unknown ...
  can potentially cause a hang"` warning and the first collective
  reliably hangs or fails with `ncclInvalidUsage`. `setup_distributed`
  passes `device_id=torch.device("cuda:0")` automatically on the
  NCCL+CUDA path. On XPU/xccl it does the opposite — passing
  `device_id` triggers DataLoader-worker deadlocks.
- **The per-rank shell is `bash -lc` (login).** Login shells source
  `~/.bash_profile` which on ALCF resets cwd to `$HOME`. The launcher
  emits an explicit `cd $prism_dir` inside the per-rank block so
  `from src.X` imports resolve. Aurora launchers don't need this
  because their per-rank shell isn't `-l`.
- **rsync `--exclude='data'` is too greedy.** It matches both the
  intended repo-root `BaseMM_PRISM/data` AND `BaseMM_PRISM/src/data`.
  Use anchored excludes (`--exclude='/data'`) when staging the repo
  to Polaris.
- **Outbound network is via proxy.** Set
  `HTTP_PROXY=http://proxy.alcf.anl.gov:3128` for HF downloads. The
  launcher exports these by default; use `--no-proxy` for offline runs.
- **Pre-cache HF models.** Compute nodes go through the proxy too,
  which is rate-limited and intermittently blocks. Download models
  on a login node first:
  ```bash
  HF_HOME=/eagle/ModCon/ngetty/cache/huggingface \
      python -c 'from huggingface_hub import snapshot_download; \
                 snapshot_download("allenai/OLMo-1B-0724-hf"); \
                 snapshot_download("google/siglip2-base-patch16-224")'
  ```
- **Conda `transformers 4.53.3` is too old.** PRISM's `src/model.py`
  passes `dtype=` to `AutoModelForCausalLM.from_pretrained`, which only
  exists from transformers 4.55+. `setup_polaris_env.sh` installs
  `transformers==4.57.6` + `tokenizers<=0.23` into the venv with
  `--no-deps` so they shadow the conda base without disturbing torch.
- **walrus + the_well must be installed `--no-deps`.** Both transitively
  require `torch>=2.1` and a plain `pip install` will resolve a fresh
  torch wheel into the venv, shadowing the ALCF-built CUDA-aware torch
  and breaking NCCL. The setup script bakes in `--no-deps`.
- **uv pip surprises.** `uv pip install` doesn't honor
  `include-system-site-packages` the way plain pip does — it will
  install transitively-required `torch` into the venv even when the
  base already has it. Use plain `python -m pip` for Polaris.
- **NCCL backend is required for `mpi`-backed Polaris compute.** The
  conda module's torch has `gloo`, `mpi`, `nccl` backends, but `mpi`
  on Polaris is CUDA-unaware (HPE Cray MPICH lacks CUDA-aware MPI for
  PyTorch ProcessGroupMPI). Use `nccl` for any GPU collective.
- **DeepSpeed init**: launcher sets `DEEPSPEED_ZERO_STAGE=<n>` so
  `setup_distributed()` takes the env-only init path (avoids mpi4py
  re-initializing MPI after DS does).
- **Walltime clipping**: Aurora designs default to 6h walltimes; the
  Polaris debug/debug-scaling queues cap at 1h. The launcher
  auto-clips and warns when it does.

## Queue policy reminders

- `debug` — 1-2 nodes, 5min-1h, 24 nodes max shared
- `debug-scaling` — 1-10 nodes, 5min-1h, **1 job per user**
- `prod` — routing queue; 10-496 nodes; 5min-24h

See the [Polaris running-jobs guide](https://docs.alcf.anl.gov/polaris/running-jobs/)
for the full queue table.

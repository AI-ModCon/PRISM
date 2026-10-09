# PRISM on Perlmutter (NERSC, NVIDIA A100, CUDA)

## Launcher

`tools/launch_perlmutter.py` — loads module `pytorch/2.8.0`.

## Environment

- Install once: `pip install -r requirements/perlmutter.txt`.
- Verified stack: torch 2.8.0, transformers 4.56.2, cudatoolkit 12.9,
  deepspeed 0.17.6.
- Triton cache dirs are needed on compute nodes: set `TRITON_CACHE_DIR` and
  `DEEPSPEED_TRITON_CACHE_DIR`.

## Running

- Allocate an interactive GPU node.
- `USE_NATIVE_DDP=1` for env-only init (avoids mpi4py conflicts).
- `DIST_STRATEGY` chooses ddp/fsdp/hsdp.
- `LOCAL_WORLD_SIZE=12` (per-node GPU count on the target partition).

Backend auto-selects NCCL on CUDA. See
[`docs/platforms/running_on_perlmutter.md`](../../../platforms/running_on_perlmutter.md).

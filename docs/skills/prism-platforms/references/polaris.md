# PRISM on Polaris (NVIDIA A100, CUDA, PBS)

## Launcher

`tools/launch_polaris.py` — mirrors Aurora's `launch_aurora_web.py`, PBSPro
scheduler. Flags include `--design`, `--nodes`, `--batch`, `--dist-strategy`,
`--use-accelerate`, `--max-steps`, `--dry-run`, `--deepspeed-zero`.

## Environment

`tools/setup_polaris_env.sh`:
- Loads `conda/2025-09-28`.
- Creates a venv with `--system-site-packages` (inherits the conda CUDA torch).
- Installs `requirements/polaris.txt` (Hydra, webdataset, modality deps, mpi4py).
- Records provenance in `$PRISM_VENV/PRISM_BUILD_INFO`.

## Storage

`/eagle/ModCon/ngetty/` mirrors the Aurora `/flare` layout (`envs/ models/
datasets/ logs/ repos/ jobs/`). `datasets/` holds WebDataset manifests + `.tar`
shards. **No DAOS on `/eagle`** — use the WebDataset/Lustre-style path.

## Differences from Aurora

- Distributed backend: **NCCL** (launcher exports `DIST_BACKEND=nccl`).
- Proxy required for outbound: `HTTP_PROXY=http://proxy.alcf.anl.gov:3128`.
- Login shell `-l` requires an explicit `cd` in job scripts.
- Queue cap is one hour (vs Aurora's longer defaults) — size smokes accordingly.

See [`docs/platforms/running_on_polaris.md`](../../../platforms/running_on_polaris.md) for the full
walkthrough.

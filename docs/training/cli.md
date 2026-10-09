# PRISM CLI Reference

The `prism` command is a unified CLI that replaces 50+ scattered scripts with a single discoverable interface. Built with [Typer](https://typer.tiangolo.com/), it provides `--help` at every level.

## Installation

```bash
# From the project root
pip install -r requirements/base.txt
pip install -e . --no-deps

# Verify
prism --help
```

This registers the `prism` console script via the entry point in `pyproject.toml`. Requires Python 3.10+ and `typer>=0.9.0`.

On Aurora, Polaris, or Perlmutter, **do not** use a plain `pip install` — it
resolves a fresh PyTorch wheel that shadows the system-provided,
accelerator-aware build. Build the environment with the platform scripts in
`tools/` first, then register the console script with the `--no-deps` install
above (on Aurora, `tools/install_prism_entry_point.sh` does exactly that).

> **Note:** If you are on Aurora, activate the venv first:
> ```bash
> source "$VENV_PATH/bin/activate"   # e.g. /flare/<project>/$USER/prism-envs/py3.12
> ```

## Command Overview

| Command | Description | Underlying Script |
|---------|-------------|-------------------|
| `prism train` | Hydra-based training with config composition | `src/train.py` |
| `prism launch` | Generate and submit HPC jobs | `tools/launch_*.py` |
| `prism data` | Data download, conversion, validation, staging | `src/data/`, `scripts/` |
| `prism eval` | Model evaluation and throughput benchmarks | `tools/universal_evaluator.py` |
| `prism analyze` | Model and dataset analysis | `src/config.py`, `tools/analyze_dataset_lengths.py` |
| `prism serve` | Gradio UI, FastAPI server, demo | `src/ui/`, `src/api/` |

---

## prism train

Run PRISM training with Hydra config composition. All positional arguments are passed as Hydra overrides.

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--multirun` | `-m` | `false` | Enable Hydra multirun sweep |
| `--config-path` | | `src/conf` | Path to Hydra config directory |
| `--config-name` | | `config` | Name of the Hydra config file |
| `--list-presets` | | `false` | List available model presets |

### Examples

```bash
# Basic training with a model preset
prism train model=prism_olmo3_7b training=zone_a

# Override batch size and learning rate
prism train model=prism_olmo3_7b training.batch_size=8 training.learning_rate=1e-4

# Hyperparameter sweep
prism train --multirun training.learning_rate=1e-4,1e-5,1e-6

# List all available model presets
prism train --list-presets
```

---

## prism launch

Generate and optionally submit HPC launch scripts. Any arguments after `--` are forwarded as Hydra overrides to `src/train.py`.

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--platform` | `-p` | *required* | Target platform: `perlmutter`, `aurora`, `aurora-daos`, `aurora-web`, `baremetal` |
| `--id` | | *required* | Experiment / job ID |
| `--file` | `-f` | `experiments/prism_designs.yaml` | Experiment design YAML |
| `--design` | | same as `--id` | Design ID in YAML |
| `--nodes` | `-n` | `1` | Number of nodes |
| `--gpus` | `-g` | | Accepted for compatibility; currently ignored (no launcher parses it) |
| `--partition` / `--queue` | `-q` | from `.env` or `debug` | Queue/partition name |
| `--account` | `-A` | from `.env` | SLURM account / PBS project |
| `--time` | `-t` | `00:30:00` | Walltime |
| `--dry-run` | | `false` | Print script without executing/submitting |
| `--submit` | | `false` | Submit the job (sbatch/qsub) |
| `--native-ddp` | | `false` | Use native PyTorch DDP |
| `--native-fsdp` | | `false` | Use native PyTorch FSDP |

### DAOS-Specific Options (`aurora-daos`)

| Option | Default | Description |
|--------|---------|-------------|
| `--daos-pool` | `AuroraGPT` | DAOS pool name |
| `--daos-container` | `prism_training_data` | DAOS container for training data |
| `--daos-models-container` | `prism_models` | DAOS container for model weights |
| `--dataset-groups` | `all` | Dataset groups: `all`, `pixmo`, `s1mmalign`, `nemotron`, `cosyn`, or comma-separated |
| `--dataset-config` | `src/conf/data/daos_datasets.yaml` | Dataset configuration YAML |
| `--dataset-proportions` | | Override dataset proportions, e.g. `dataset1:0.1,dataset2:0.2` |

### Aurora-Web Options (`aurora-web`)

| Option | Default | Description |
|--------|---------|-------------|
| `--webdataset-dir` | *required* | Path to WebDataset directory |
| `--packed-env` | from `.env` or `deepspeed_env.tar.gz` | Packed env tarball path |

### Supported Platforms

| Platform | Scheduler | Launcher Script |
|----------|-----------|-----------------|
| `perlmutter` | SLURM (sbatch) | `tools/launch_perlmutter.py` |
| `aurora` | PBS (qsub) | `tools/launch_aurora.py` |
| `aurora-daos` | PBS (qsub) | `tools/launch_aurora_daos.py` |
| `aurora-web` | PBS (qsub) | `tools/launch_aurora_web.py` |
| `baremetal` | local | `tools/launch_baremetal.py` |

### Examples

```bash
# Aurora DAOS — submit a 2-node batch job
prism launch --platform aurora-daos --id PRISM-AURORA-ZONE-A-1B --nodes 2 --submit

# Aurora DAOS — dry run to inspect generated script
prism launch --platform aurora-daos --id PRISM-AURORA-ZONE-A-1B --nodes 2 --dry-run

# Aurora DAOS — specific dataset group
prism launch --platform aurora-daos --id MY-RUN --nodes 1 --submit \
    --dataset-groups pixmo

# Aurora WebDataset
prism launch --platform aurora-web --id PRISM-IMAGE-ONLY \
    --webdataset-dir /flare/<project>/<user>/data/zone_a/pixmo_cap_webdataset \
    --nodes 2 --submit

# Perlmutter — 4 nodes with Hydra overrides
prism launch --platform perlmutter --id my-exp --nodes 4 --submit \
    -- model=prism_olmo3_7b training.batch_size=4

# Baremetal — local run
prism launch --platform baremetal --id local-test
```

---

## prism data

Data pipeline utilities with four subcommands.

### prism data download

Download datasets for a specific training zone.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--zone` | `-z` | *required* | Training zone: `a`, `b`, or `c` |
| `--output-dir` | `-o` | `data` | Output directory |
| `--num-samples` | `-n` | `500` | Number of samples to download |

```bash
prism data download --zone a --output-dir data/zone_a --num-samples 1000
prism data download --zone b --output-dir data/zone_b
prism data download --zone c --output-dir data/zone_c
```

### prism data convert

Convert datasets to WebDataset format.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--source-format` | `-s` | `arrow` | Source format: `arrow`, `parquet`, `s1mmalign`, `nemotron`, `arxiv` |
| `--output-dir` | `-o` | *required* | Output directory |
| `--input-dir` | `-i` | | Input directory |
| `--arrow-dir` | | | Directory with Arrow files |
| `--images-dir` | | | Directory with images |
| `--dataset-name` | | | Name prefix for shards |
| `--images-per-shard` | | `1000` | Images per TAR shard |
| `--workers` | `-w` | `8` | Parallel workers |
| `--val-split` | | `0.01` | Validation split ratio (parquet only) |

```bash
prism data convert --source-format parquet --input-dir data/raw --output-dir data/wds
prism data convert --source-format s1mmalign --input-dir data/arxiv --output-dir data/wds --workers 16
```

### prism data validate

Validate image files and optionally remove corrupt ones.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--images-dir` | | *required* | Directory containing images |
| `--workers` | `-w` | `32` | Number of parallel workers |
| `--dry-run` | | `false` | Show what would be removed |
| `--remove` | | `false` | Actually remove corrupt files |

```bash
prism data validate --images-dir data/images --dry-run
prism data validate --images-dir data/images --remove --workers 64
```

### prism data stage

Stage WebDataset shards to node-local storage for distributed training.

| Option | Default | Description |
|--------|---------|-------------|
| `--manifest` | *required* | Path to manifest.json |
| `--shards-dir` | *required* | Directory containing shards |
| `--local-dir` | *required* | Local destination directory |
| `--node-rank` | *required* | This node's rank (0-indexed) |
| `--num-nodes` | *required* | Total number of nodes |

```bash
prism data stage --manifest data/manifest.json --shards-dir /shared/shards \
    --local-dir /tmp/shards --node-rank 0 --num-nodes 4
```

---

## prism eval

Model evaluation and throughput benchmarking.

### prism eval run

Run model evaluation using the universal evaluator.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--checkpoint` | `-c` | *required* | Path to model checkpoint |
| `--mode` | `-m` | `run_eval` | Eval mode: `run_eval`, `inspect_train`, `inspect_eval`, `verify_image`, `verify_timeseries_scits`, `verify_timeseries_interleave` |
| `--limit` | `-l` | `0` | Limit number of samples (0 = default) |
| `--save-dir` | | | Directory to save outputs |
| `--visualize` | | `false` | Enable visualization plotting |
| `--viz-dir` | | | Directory to save visualizations |
| `--exhaustive` | | `false` | Iterate ALL datasets |
| `--data-only` | | `false` | Skip model loading |
| `--modality` | | | Filter by modality name (e.g. `Vision`) |
| `--backbone` | | `allenai/OLMo-7B-0724-hf` | Backbone model ID |
| `--validation` | | `false` | Use validation data |

```bash
# Standard evaluation
prism eval run --checkpoint outputs/step_10000

# Inspect training data batches
prism eval run --checkpoint outputs/step_10000 --mode inspect_train --limit 50

# Evaluate with visualizations
prism eval run --checkpoint outputs/step_10000 --visualize --viz-dir eval_plots/

# Vision-only evaluation
prism eval run --checkpoint outputs/step_10000 --modality Vision
```

### prism eval throughput

Run throughput benchmarks.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--test` | | `all` | Comma-separated tests or `all` |
| `--model` | | | Model size: `1b` or `7b` |
| `--seq-len` | | `256` | Sequence length |
| `--num-steps` | | `10` | Number of benchmark steps |
| `--output` | `-o` | `benchmark_results.json` | Output JSON file |

```bash
prism eval throughput --model 1b --seq-len 512 --num-steps 20
prism eval throughput --model 7b --output results_7b.json
```

---

## prism analyze

Model and dataset analysis tools.

### prism analyze model

Analyze model configuration: parameter counts, projector dimensions, encoder sizes.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--preset` | `-p` | | Model preset name |
| `--list` | `-l` | `false` | List all available presets |

```bash
# List all presets
prism analyze model --list

# Analyze a specific preset
prism analyze model --preset prism-olmo3-7b
prism analyze model --preset prism-auroragpt-2b
```

### prism analyze dataset

Analyze dataset length distributions and statistics.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--daos-mount` | | | DAOS mount path for dataset |
| `--dataset-groups` | `-g` | `all` | Comma-separated dataset groups or `all` |
| `--samples` | `-n` | `100` | Samples per dataset to analyze |
| `--tokenizer` | `-t` | `allenai/OLMo-2-1124-7B-Instruct` | Tokenizer for length analysis |

```bash
prism analyze dataset --dataset-groups pixmo --samples 500
prism analyze dataset --daos-mount /tmp/data --tokenizer allenai/OLMo-7B-0724-hf
```

---

## prism serve

Serve models for interactive inference or API access.

For high-throughput batched inference via the in-tree vLLM plugin (image+text and time-series on XPU, 3,336 / 4,036 tok/s respectively on a single tile), see [`inference_vllm.md`](../evaluation/inference_vllm.md).

### prism serve ui

Launch the Gradio Chat UI.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--checkpoint` | `-c` | | Path to model checkpoint |
| `--port` | `-p` | `7860` | Server port |
| `--device` | `-d` | `auto` | Device: `auto`, `cuda`, `xpu`, `mps`, `cpu` |
| `--share` | | `false` | Create a public Gradio share link |

```bash
prism serve ui --checkpoint outputs/step_10000 --device xpu
prism serve ui --checkpoint outputs/step_10000 --share --port 7861
```

### prism serve api

Launch the FastAPI server (OpenAI-compatible).

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--port` | `-p` | `8000` | Server port |
| `--host` | | `0.0.0.0` | Server host |

```bash
prism serve api --port 8000
prism serve api --host 127.0.0.1 --port 9000
```

---

## Configuration (.env)

The CLI reads defaults from a `.env` file in the project root. These values are also propagated to subprocess environments.

| Key | Default | Used By | Description |
|-----|---------|---------|-------------|
| `VENV_PATH` | | `launch` | Path to virtual environment (skip tarball if set) |
| `PRISM_DIR` | cwd | `launch` | Project root directory |
| `HF_HOME` | | `launch` | HuggingFace cache directory |
| `HF_TOKEN` | | `launch` | HuggingFace authentication token |
| `SHARED_HF_HOME` | | `launch` | Shared HuggingFace hub path |
| `PROJECT_ALLOCATION` | | `launch --account` | Default PBS project / SLURM account |
| `QUEUE_NAME` | `debug` | `launch --queue` | Default queue/partition |
| `ENV_TARBALL` | `deepspeed_env.tar.gz` | `launch --packed-env` | Packed environment tarball |
| `PROXY_URL` | | `launch` | HTTP proxy for compute nodes |
| `WANDB_API_KEY` | | `train` | Weights & Biases API key |

---

## Equivalence Guide

Mapping from old script commands to the unified CLI:

| Old Command | New CLI Command |
|-------------|-----------------|
| `python tools/launch_aurora_daos.py --id RUN --nodes 2 --batch` | `prism launch --platform aurora-daos --id RUN --nodes 2 --submit` |
| `python tools/launch_aurora_web.py --id RUN --webdataset-dir /path --nodes 2` | `prism launch --platform aurora-web --id RUN --webdataset-dir /path --nodes 2` |
| `python tools/launch_perlmutter.py --id RUN --nodes 4 --submit` | `prism launch --platform perlmutter --id RUN --nodes 4 --submit` |
| `python tools/launch_baremetal.py --id RUN` | `prism launch --platform baremetal --id RUN` |
| `python src/train.py model=prism_olmo3_7b training=zone_a` | `prism train model=prism_olmo3_7b training=zone_a` |
| `python tools/universal_evaluator.py --checkpoint ckpt` | `prism eval run --checkpoint ckpt` |
| `python tools/benchmark_throughput.py --model 1b` | `prism eval throughput --model 1b` |
| `python src/ui/app.py` | `prism serve ui` |
| `python src/api/server.py` | `prism serve api` |
| `python src/data/download_zone_a_data.py` | `prism data download --zone a` |
| `python scripts/convert_to_webdataset.py --output-dir out` | `prism data convert --source-format arrow --output-dir out` |
| `python scripts/validate_images.py --images-dir imgs` | `prism data validate --images-dir imgs` |
| `python scripts/stage_shards.py --manifest m.json ...` | `prism data stage --manifest m.json ...` |
| `python tools/analyze_dataset_lengths.py` | `prism analyze dataset` |

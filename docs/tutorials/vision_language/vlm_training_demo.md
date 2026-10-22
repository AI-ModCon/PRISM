# PRISM tutorial: train a vision-language projector

A short Aurora walkthrough from environment setup to a W&B-logged
vision-language training run. The goal is to show the normal PRISM workflow:
install the repo, download models and data, convert data if needed, then launch
training through PRISM's existing launcher.

No separate training script is used.

## What you will build

A small image-caption model that projects frozen SigLIP2 image features into a
frozen OLMo language backbone. Only the PRISM image projector trains.

The demo uses:

- language backbone: `allenai/OLMo-1B-0724-hf`
- image encoder: `google/siglip2-base-patch16-224`
- model config: `src/conf/model/prism_image_only.yaml`
- training recipe: `src/conf/training/projector_only.yaml`
- data format: image-caption WebDataset shards on Lustre
- logging: W&B online mode

## Before you start

You need an Aurora allocation, W&B credentials, and network access on the login
node for Hugging Face model and dataset downloads.

The required data input is a WebDataset directory with this layout:

```text
$WDS_DIR/
  manifest.json
  shards/
    *.tar
```

`allenai/pixmo-cap` is a good public captioning source for this demo. PRISM
includes an application script to download PixMo-Cap images locally, then the
standard `prism data convert` command writes WebDataset shards.

## 1. Clone PRISM

Set site-specific paths first:

```bash
export PROJECT_ALLOCATION=YOUR_ALLOCATION
export QUEUE_NAME=debug
export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WORK_ROOT=/flare/${PROJECT_ALLOCATION}/${USER}

export PRISM_DIR=$WORK_ROOT/PRISM
export DEMO_ROOT=$WORK_ROOT/prism-vlm-demo
export HF_HOME=$DEMO_ROOT/hf
export SHARED_HF_HOME=$HF_HOME/hub
export HF_HUB_CACHE=$SHARED_HF_HOME
export TRANSFORMERS_CACHE=$SHARED_HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PIXMO_LOCAL_DIR=$DEMO_ROOT/pixmo-cap-local
export WDS_DIR=$DEMO_ROOT/pixmo-cap-webdataset

mkdir -p "$WORK_ROOT" "$DEMO_ROOT" "$HF_HOME" "$SHARED_HF_HOME" "$HF_DATASETS_CACHE"
cd "$WORK_ROOT"
git clone --recursive https://github.com/AI-ModCon/PRISM.git
cd "$PRISM_DIR"
```

Keep the virtual environment, model cache, and data outside the git checkout.

## 2. Install

On Aurora:

```bash
module use /opt/aurora/26.26.0/modulefiles
module load oneapi/release/2025.3.1
module load mpich/prd/5.0.0.aurora_test.51a9474
module load frameworks/2025.3.1
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

bash tools/setup_deepspeed_env.sh
source .venv-deepspeed/bin/activate
python -m pip install --no-deps -e .
tar -czf deepspeed_env.tar.gz .venv-deepspeed
```

If you open a new shell later, reload the Aurora modules before activating the
environment:

```bash
module use /opt/aurora/26.26.0/modulefiles
module load oneapi/release/2025.3.1
module load mpich/prd/5.0.0.aurora_test.51a9474
module load frameworks/2025.3.1
source .venv-deepspeed/bin/activate
```

Create `.env` for the launchers:

```bash
cp .env.template .env
cat >> .env <<EOF
PRISM_DIR=$PRISM_DIR
HF_HOME=$HF_HOME
SHARED_HF_HOME=$SHARED_HF_HOME
HF_HUB_CACHE=$HF_HUB_CACHE
TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
HF_DATASETS_CACHE=$HF_DATASETS_CACHE
PROJECT_ALLOCATION=$PROJECT_ALLOCATION
QUEUE_NAME=$QUEUE_NAME
ENV_TARBALL=$PRISM_DIR/deepspeed_env.tar.gz
EOF
```

Log in to W&B:

```bash
wandb login
chmod 600 .env
read -s WANDB_API_KEY
printf "\nWANDB_API_KEY=%s\n" "$WANDB_API_KEY" >> .env
```

The Aurora PBS launcher reads `WANDB_API_KEY` from `.env` on the compute nodes.

## 3. Download Models

```bash
huggingface-cli download allenai/OLMo-1B-0724-hf
huggingface-cli download google/siglip2-base-patch16-224
```

## 4. Prepare Data

Download a PixMo-Cap subset and write a local Arrow dataset with downloaded
image files:

```bash
python applications/vision_language/prepare_pixmo_dataset.py \
  --hf-id allenai/pixmo-cap \
  --output-dir "$PIXMO_LOCAL_DIR" \
  --limit 250000 \
  --workers 64
```

Convert it to WebDataset shards:

```bash
prism data convert \
  --source-format arrow \
  --arrow-dir "$PIXMO_LOCAL_DIR" \
  --images-dir "$PIXMO_LOCAL_DIR/images" \
  --output-dir "$WDS_DIR" \
  --images-per-shard 100 \
  --workers 8
```

If you already have compatible WebDataset shards instead, set:

```bash
export WDS_DIR=/path/to/image_caption_webdataset
```

For other image-caption datasets, `prism data convert` also accepts Parquet
records with embedded images or local image paths:

```bash
prism data convert \
  --source-format parquet \
  --input-dir /path/to/parquet_files \
  --dataset-name image-caption-demo \
  --output-dir "$WDS_DIR" \
  --images-per-shard 100 \
  --workers 8
```

For a two-node run, use at least 24 shards so every Aurora rank has data to read.
For finite, no-repeat training, also make sure the number of usable samples in
`manifest.json` is at least:

```text
required_samples = nodes * 12 * local_batch * max_steps
```

The launch commands below use two nodes, 400 steps, DDP local batch 10, and
HSDP local batch 16. That means the same WebDataset should contain at least
`2 * 12 * 16 * 400 = 153600` usable samples if you want both launches to run
without repeating data. Check this before submitting:

```bash
python - <<'PY'
import json
import os

manifest = os.path.join(os.environ["WDS_DIR"], "manifest.json")
with open(manifest) as f:
    data = json.load(f)

usable = int(data.get("total_written") or data.get("total_samples") or 0)
required = 2 * 12 * 16 * 400
print(f"usable_samples={usable:,}")
print(f"required_samples_for_hsdp_demo={required:,}")
if usable < required:
    raise SystemExit("Increase --limit and rebuild the WebDataset before training.")
PY
```

## 5. Launch Training

The default launch path uses DDP. The second launch script switches to HSDP
and increases the local batch size. For this comparison to be meaningful,
first choose the largest local batch that runs under DDP, then use HSDP for
the first larger local batch that DDP cannot run.

For example, if DDP runs at local batch 10 but fails at local batch 16, and
HSDP runs at local batch 16, set:

```bash
export DEMO_MAX_STEPS=400
export DDP_LOCAL_BATCH=10
export HSDP_LOCAL_BATCH=16
```

If DDP also runs at `HSDP_LOCAL_BATCH`, raise both values and repeat the short
smoke test. Otherwise, the comparison does not show a batch-size benefit from
HSDP.

### DDP launch script

```bash
python tools/launch_aurora_web.py \
  --file experiments/prism_designs.yaml \
  --id "PRISM-VLM-DEMO-OLMO1B-SIGLIP2-BASE-DDP-BS${DDP_LOCAL_BATCH}" \
  --design PRISM-IMAGE-ONLY-2N \
  --nodes 2 \
  --batch \
  --queue "$QUEUE_NAME" \
  --project "$PROJECT_ALLOCATION" \
  --walltime 01:00:00 \
  --webdataset-dir "$WDS_DIR" \
  --finite-webdataset \
  --packed-env "$PRISM_DIR/deepspeed_env.tar.gz" \
  --dist-strategy ddp \
  --wandb-project prism-vlm-demo \
  --max-steps "$DEMO_MAX_STEPS" \
  model=prism_image_only \
  training=projector_only \
  model.backbone_id=allenai/OLMo-1B-0724-hf \
  model.image_encoder_id=google/siglip2-base-patch16-224 \
  model.image_processor_id=google/siglip2-base-patch16-224 \
  training.batch_size="$DDP_LOCAL_BATCH" \
  training.gradient_accumulation_steps=1 \
  training.learning_rate=5e-4 \
  training.lr_connector=5e-4 \
  training.warmup_steps=40 \
  training.save_every_n_steps="$DEMO_MAX_STEPS" \
  training.viz_every_n_steps=100 \
  training.viz_num_examples=10 \
  training.eval_every_n_steps=1000 \
  training.device=xpu \
  wandb.mode=online \
  wandb.entity="$WANDB_ENTITY"
```

This is a two-node run. On Aurora, each node has 12 XPU ranks, so the global
batch is `2 * 12 * DDP_LOCAL_BATCH` samples/update.

### HSDP launch script for a larger batch

HSDP shards optimizer and gradient state within each node and replicates across
nodes. The script below keeps the same two nodes but uses
`HSDP_LOCAL_BATCH`, which should be larger than the validated DDP limit. The
global batch is `2 * 12 * HSDP_LOCAL_BATCH` samples/update.

```bash
python tools/launch_aurora_web.py \
  --file experiments/prism_designs.yaml \
  --id "PRISM-VLM-DEMO-OLMO1B-SIGLIP2-BASE-HSDP-BS${HSDP_LOCAL_BATCH}" \
  --design PRISM-IMAGE-ONLY-2N \
  --nodes 2 \
  --batch \
  --queue "$QUEUE_NAME" \
  --project "$PROJECT_ALLOCATION" \
  --walltime 01:00:00 \
  --webdataset-dir "$WDS_DIR" \
  --finite-webdataset \
  --packed-env "$PRISM_DIR/deepspeed_env.tar.gz" \
  --dist-strategy hsdp \
  --fsdp-sharding shard_grad_op \
  --benchmark-mode \
  --wandb-project prism-vlm-demo \
  --max-steps "$DEMO_MAX_STEPS" \
  model=prism_image_only \
  training=projector_only \
  model.backbone_id=allenai/OLMo-1B-0724-hf \
  model.image_encoder_id=google/siglip2-base-patch16-224 \
  model.image_processor_id=google/siglip2-base-patch16-224 \
  training.batch_size="$HSDP_LOCAL_BATCH" \
  training.gradient_accumulation_steps=1 \
  training.learning_rate=5e-4 \
  training.lr_connector=5e-4 \
  training.warmup_steps=40 \
  training.save_every_n_steps="$DEMO_MAX_STEPS" \
  training.viz_every_n_steps=100 \
  training.viz_num_examples=10 \
  training.eval_every_n_steps=1000 \
  training.device=xpu \
  wandb.mode=online \
  wandb.entity="$WANDB_ENTITY"
```

If the HSDP run is unstable because the global batch changed, either lower
`training.batch_size` or scale the learning rate intentionally for the new
global batch. For an apples-to-apples comparison, keep the global batch fixed
by reducing the local batch or increasing/decreasing the node count together.

## 6. Monitor

Check PBS:

```bash
qstat -u "$USER"
```

Check local logs:

```bash
ls -lt logs/PRISM-IMAGE-ONLY-2N/
```

The run should appear in W&B under `prism-vlm-demo`. The main fields to check are:

- `loss`
- `lr`
- `loss/tokens_global`
- `grad_norm/projector_mean`
- `viz/predictions`

`viz/predictions` logs image-caption examples every 100 training steps for this demo.

# Qwen3 + SigLIP2 Scaling Experiments

This runbook captures the reusable setup for the Qwen3/SigLIP2 VLM scaling
experiments. Local job logs, W&B run IDs, checkpoint paths, and analysis plots
stay outside git; the tracked pieces here are the data split definition,
experiment designs, and launch recipe needed to rerun the studies.

## Tracked Files

| File | Purpose |
|------|---------|
| `experiments/qwen3_siglip_scaling.yaml` | Named experiment designs for language-backbone scaling and SigLIP2 encoder sweeps |
| `src/conf/data/vlm_diverse_global_shuffle_v1.yaml` | Train/cooldown WebDataset split definitions for the globally shuffled VLM mix |
| `scripts/build_global_shuffle_webdataset.py` | Deterministically pool and globally shuffle source WebDataset shards |
| `scripts/materialize_global_shuffle_splits.py` | Create the 25k-step train view and held-out cooldown view |
| `scripts/audit_global_shuffle.py` | Check source composition over step windows after materialization |

## Data

The training mixture is built by pooling PRISM WebDataset exports from the
existing `vlm_diverse_unweighted` preset in `src/conf/data/lustre_datasets.yaml`:

- PixMo: captions, pointing, and counting.
- S1MM-Align style scientific image-text shards: arXiv, bioRxiv, Nature
  Communications, ChemRxiv, medRxiv, engrXiv, psyArXiv, edrXiv, and metaArXiv.
- Nemotron multilingual Wikipedia image-text shards.

The global-shuffle builder assigns each sample a deterministic random key based
on seed, source dataset, source shard, and source sample key. It then performs an
external sort and writes one pooled WebDataset. This avoids source-regime blocks
in the training order.

Build the pooled source dataset:

```bash
python3 scripts/build_global_shuffle_webdataset.py \
  --source-config src/conf/data/lustre_datasets.yaml \
  --groups vlm_diverse_unweighted \
  --output-dir /flare/AuroraGPT/$USER/prism_datasets/vlm_diverse_global_shuffle_v1 \
  --tmp-dir /flare/AuroraGPT/$USER/prism_datasets/_tmp/vlm_diverse_global_shuffle_v1 \
  --seed 20260716 \
  --bucket-bits 13 \
  --workers 8 \
  --maxcount 5000 \
  --maxsize 3000000000 \
  --overwrite
```

Materialize the train and cooldown views:

```bash
python3 scripts/materialize_global_shuffle_splits.py \
  --source-dir /flare/AuroraGPT/$USER/prism_datasets/vlm_diverse_global_shuffle_v1 \
  --global-batch-size 768 \
  --train-steps 25000 \
  --cooldown-steps 1000 \
  --source-maxcount 5000 \
  --cooldown-shards 192 \
  --overwrite
```

The canonical split has:

| Split | Samples | Shards | Intended Use |
|-------|---------|--------|--------------|
| `vlm_diverse_global_shuffle_v1_train25k_gbs768` | 19,200,000 | 3840 | 25k train steps at global batch 768 |
| `vlm_diverse_global_shuffle_v1_cooldown1k_gbs768` | 825,447 | 192 | Held-out 1k-step LR cooldowns |

Audit source balance over training windows:

```bash
python3 scripts/audit_global_shuffle.py \
  --dataset-dir /flare/AuroraGPT/$USER/prism_datasets/vlm_diverse_global_shuffle_v1_train25k_gbs768 \
  --global-batch-size 768 \
  --window-steps 500 \
  --max-steps 25000
```

## Models

Language-backbone scaling uses Qwen3 dense CausalLM backbones through the
existing PRISM image-only model presets:

| PRISM Config | HF Backbone | Hidden Size |
|--------------|-------------|-------------|
| `prism_qwen3_0_6b_image_only` | `Qwen/Qwen3-0.6B` | 1024 |
| `prism_qwen3_1_7b_image_only` | `Qwen/Qwen3-1.7B` | 2048 |
| `prism_qwen3_4b_image_only` | `Qwen/Qwen3-4B` | 2560 |
| `prism_qwen3_8b_image_only` | `Qwen/Qwen3-8B` | 4096 |

The language-backbone scaling runs use the default image encoder in those model
configs: `google/siglip2-base-patch16-224`, with all 14 x 14 image patch tokens
projected into the language context.

The encoder-size sweeps keep the language backbone fixed and use SigLIP2
patch16-256 encoders:

| Encoder | HF ID | Image Hidden Size | Image Patch Tokens |
|---------|-------|-------------------|--------------------|
| Base | `google/siglip2-base-patch16-256` | 768 | 16 x 16 |
| Large | `google/siglip2-large-patch16-256` | 1024 | 16 x 16 |
| SO400M | `google/siglip2-so400m-patch16-256` | 1152 | 16 x 16 |
| Giant | `google/siglip2-giant-opt-patch16-256` | 1536 | 16 x 16 |

Stage the HuggingFace cache before launching on compute nodes:

```bash
export HF_HOME=/flare/AuroraGPT/$USER/hf_home

for model_id in \
  Qwen/Qwen3-0.6B \
  Qwen/Qwen3-1.7B \
  Qwen/Qwen3-4B \
  Qwen/Qwen3-8B \
  google/siglip2-base-patch16-224 \
  google/siglip2-base-patch16-256 \
  google/siglip2-large-patch16-256 \
  google/siglip2-so400m-patch16-256 \
  google/siglip2-giant-opt-patch16-256
do
  huggingface-cli download "$model_id"
done
```

## Training Setup

All main scaling runs are end-to-end VLM training: the Qwen backbone, SigLIP2
encoder, and PRISM image projector are trainable.

| Setting | Value |
|---------|-------|
| Nodes | 16 Aurora nodes |
| Ranks | 12 XPU ranks per node |
| Local batch size | 4 samples per rank |
| Gradient accumulation | 1 |
| Global batch size | 768 samples |
| Steps | 25,000 target steps |
| Scheduler | WSD |
| Warmup | 500 steps |
| WSD decay | Disabled in the main run (`training.wsd_decay_ratio=0.0`) |
| Weight decay | 0 |
| Checkpoint cadence | Every 250 optimizer steps |
| In-training eval | Disabled for these designs; use offline eval grids |
| Distributed strategy | HSDP, full shard |
| Max tokenized text length | 512 in the Aurora launch command |

The default image projector is PRISM's existing two-layer `ModalityProjector`
with hidden width equal to the language hidden size and LayerNorm output
normalization.

Learning-rate transfer rules:

| Run Family | Rule | 0.6B | 1.7B | 4B | 8B |
|------------|------|------|------|----|----|
| muP-style | `lr = 1e-4 * 1024 / d_model` | 1.0e-4 | 5.0e-5 | 4.0e-5 | 2.5e-5 |
| Inverse square root | `lr = 1e-4 * sqrt(1024 / d_model)` | 1.0e-4 | 7.071e-5 | 6.325e-5 | 5.0e-5 |

Each LR is applied consistently to `training.learning_rate`,
`training.lr_connector`, `training.lr_vit`, and `training.lr_llm`.

## Launching

Set the shared environment paths explicitly or place them in the repo-local
`.env` consumed by `tools/launch_aurora_web.py`:

```bash
export VENV_PATH=/flare/ModCon/$USER/prism-envs/qwen3-siglip-py3.12
export HF_HOME=/flare/AuroraGPT/$USER/hf_home
export SHARED_HF_HOME=$HF_HOME/hub
```

Launch one run by selecting any variant ID from
`experiments/qwen3_siglip_scaling.yaml`:

```bash
python3 tools/launch_aurora_web.py \
  --file experiments/qwen3_siglip_scaling.yaml \
  --id QWEN3-4B-SIGLIP2-GLOBALSHUFV1-16N-INVSQRTLR \
  --dataset-config src/conf/data/vlm_diverse_global_shuffle_v1.yaml \
  --dataset-groups vlm_diverse_global_shuffle_v1_train25k_gbs768 \
  --dataset-root /flare/AuroraGPT/$USER/prism_datasets \
  --finite-webdataset \
  --nodes 16 \
  --queue capacity \
  --project AuroraGPT \
  --walltime 06:00:00 \
  --batch \
  --use-shared-venv \
  --hf-home "$HF_HOME" \
  --shared-hf-home "$SHARED_HF_HOME" \
  --dist-strategy hsdp \
  --fsdp-sharding full_shard \
  --benchmark-mode \
  --max-seq-length 512 \
  --webdataset-shuffle-buffer 32768 \
  --use-bucketed-collator \
  --grad-norm-interval 0 \
  wandb.entity=your-wandb-entity
```

Do not pass `--use-bucketing` for these scaling runs. That flag performs
stream-level sorted bucketing and can reintroduce long-range ordering artifacts.

To continue a run, reuse the same Hydra output directory and pass the latest
checkpoint:

```bash
python3 tools/launch_aurora_web.py \
  --file experiments/qwen3_siglip_scaling.yaml \
  --id QWEN3-4B-SIGLIP2-GLOBALSHUFV1-16N-INVSQRTLR-CONT \
  --design QWEN3-4B-SIGLIP2-GLOBALSHUFV1-16N-INVSQRTLR \
  --dataset-config src/conf/data/vlm_diverse_global_shuffle_v1.yaml \
  --dataset-groups vlm_diverse_global_shuffle_v1_train25k_gbs768 \
  --dataset-root /flare/AuroraGPT/$USER/prism_datasets \
  --finite-webdataset \
  --nodes 16 \
  --queue capacity \
  --project AuroraGPT \
  --walltime 06:00:00 \
  --batch \
  --use-shared-venv \
  --hf-home "$HF_HOME" \
  --shared-hf-home "$SHARED_HF_HOME" \
  --dist-strategy hsdp \
  --fsdp-sharding full_shard \
  --benchmark-mode \
  --max-seq-length 512 \
  --webdataset-shuffle-buffer 32768 \
  --use-bucketed-collator \
  --grad-norm-interval 0 \
  --resume-from-checkpoint /path/to/output/checkpoints/step_10000 \
  hydra.run.dir=/path/to/output \
  wandb.entity=your-wandb-entity
```

## Experiment IDs

Language-backbone scaling:

- `QWEN3-0P6B-SIGLIP2-GLOBALSHUFV1-16N-MUPLR`
- `QWEN3-1P7B-SIGLIP2-GLOBALSHUFV1-16N-MUPLR`
- `QWEN3-4B-SIGLIP2-GLOBALSHUFV1-16N-MUPLR`
- `QWEN3-8B-SIGLIP2-GLOBALSHUFV1-16N-MUPLR`
- `QWEN3-0P6B-SIGLIP2-GLOBALSHUFV1-16N-INVSQRTLR`
- `QWEN3-1P7B-SIGLIP2-GLOBALSHUFV1-16N-INVSQRTLR`
- `QWEN3-4B-SIGLIP2-GLOBALSHUFV1-16N-INVSQRTLR`
- `QWEN3-8B-SIGLIP2-GLOBALSHUFV1-16N-INVSQRTLR`

Encoder-size sweeps:

- `QWEN3-0P6B-SIGLIP2-{BASE256,LARGE256,SO400M256,GIANT256}-GLOBALSHUFV1-16N`
- `QWEN3-1P7B-SIGLIP2-{BASE256,LARGE256,SO400M256,GIANT256}-GLOBALSHUFV1-16N-INVSQRTLR`
- `QWEN3-4B-SIGLIP2-{BASE256,LARGE256,SO400M256,GIANT256}-GLOBALSHUFV1-16N-INVSQRTLR`

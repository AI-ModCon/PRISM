# CALVIN VLA encoder-alignment training

This page documents the CALVIN vision-language-action (VLA) training path added to PRISM.

## Scope

- Task: `training.task=vla_calvin`
- Inputs per sample: `head image`, `wrist image`, `language instruction`, `robot pose`
- Target: next-step continuous action vector
- Loss: mean squared error over action dimensions
- Encoder alignment: freeze language model backbone and image encoder; train connector/projector, pose embedder, and action head

## Dataset

Dataset root:

- `/flare/ModCon/sww/vla_training/calvin_dataset`

Important safety constraint:

- `training.calvin_max_chunk` must be `<=20`
- Loader enforces this and raises if exceeded

Split metadata comes from:

- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`

Sampling rule:

- Markov policy training (`obs_t` -> `action_{t+1}`)
- No action history or temporal windowing

## Model Path

VLA mode is enabled by:

- `model.is_vla=true`

Forward path:

1. Encode head and wrist images with existing PRISM image encoder.
2. Project image features with existing image projector.
3. Embed pose with a 2-layer MLP + ReLU.
4. Embed instruction text with the HF LLM token embedding layer.
5. Concatenate `[head tokens, wrist tokens, pose token, text tokens]`.
6. Run LLM backbone with `inputs_embeds`.
7. Read hidden state at the last valid text token.
8. Predict continuous action with a 2-layer MLP + ReLU.

## Trainable Parameters (encoder alignment)

`src/training/trainer_zone_a_vla.py` unfreezes only:

- `projectors.*`
- `pose_embed.*`
- `action_head.*`

Everything else is frozen.

## Logging

W&B logs include:

- `loss` (mean action MSE)
- `mse_dim/0 ... mse_dim/N-1` (per-dimension MSE)
- `lr`

W&B entity/mode is now wired via config:

- `wandb.entity`
- `wandb.mode`

## Key Files

- `src/data/calvin_vla.py`: CALVIN map-style dataset loader (legacy path)
- `src/data/multi_webdataset.py`: composite `vla` modality + `ModalityAwareWebDatasetWrapper` decoders (per-modality path)
- `applications/vla/shard_calvin_vla.py`: convert LeRobot CALVIN parquet → WebDataset shards
- `tools/validate_webdataset.py`: pre-flight shard validator (gates VLA-1)
- `src/data/vla_collate.py`: collator for VLA batches (shared by both paths)
- `src/model.py`: VLA forward path + modules
- `src/training/trainer_zone_a_vla.py`: Zone A VLA trainer (also emits `perf.jsonl`)
- `train.py`: task routing (`vlm` vs `vla_calvin`) + `calvin_loader` switch
- `experiments/prism_designs.yaml`: Aurora VLA experiment entries (`-DEFAULT`, `-SMOKE`, `-WEB`, `-WEB-SMOKE`)
- `experiments/modality_presets.yaml`: `vla` preset (`[text, image]`)

## Aurora Launch Notes

For Aurora runs, use Accelerate path (do not set native DDP/FSDP env flags for VLA).

Example overrides:

- `training.task=vla_calvin`
- `model.is_vla=true`
- `training.calvin_root=/flare/ModCon/sww/vla_training/calvin_dataset`
- `training.calvin_max_chunk=20`
- `model.modalities=[text,image]`

## Choosing a Data Loader: `training.calvin_loader`

Two loader paths are available, selected by `training.calvin_loader`:

- `map` (default during the transition window): the legacy `CalvinVLADataset` map-style loader (`src/data/calvin_vla.py`). Reads parquet directly from `training.calvin_root`. Unchanged from the 2026-05-22 green baseline.
- `webdataset`: routes CALVIN through `ModalityAwareWebDatasetWrapper(modality="vla")` over the WebDataset shards produced by `applications/vla/shard_calvin_vla.py`. Same per-step batch shape; benefits from the per-modality pipeline (bucketing-compatible, sweep harness, `perf.jsonl` action-MSE columns).

To produce the shards once:

```bash
python applications/vla/shard_calvin_vla.py \
    --root /flare/ModCon/sww/vla_training/calvin_dataset \
    --split train \
    --out  /flare/ModCon/sww/vla_training/calvin_webdataset
python tools/validate_webdataset.py \
    /flare/ModCon/sww/vla_training/calvin_webdataset \
    --check pose,action,image,text
```

Then select the webdataset path on a run:

```bash
# via the per-modality sweep harness
python tools/run_sweep.py \
    --preset vla \
    --designs PRISM-AURORA-ZONE-A-VLA-CALVIN-WEB-SMOKE \
    --storage lustre

# or directly via the smoke
bash tools/parity/smoke_vla_web.sh
```

Notes:

- The webdataset path still uses Accelerate. Native DDP/FSDP for VLA is out of scope here.
- Episode-aligned sharding ⇒ the loader sets `shardshuffle=False` so `(obs_t, action_{t+1})` pairs stay intact. Do **not** repoint the calvin group at non-episode-aligned shards.
- `training.calvin_webdataset_root` (optional) overrides the `calvin` group's `path` for one-off smokes against shards in `/tmp`.
- `training.calvin_webdataset_storage` is `lustre` (default) or `daos`; chooses between `src/conf/data/lustre_datasets.yaml` and `daos_datasets.yaml`.

The `map` default will be removed in PR VLA-5 after a one-week clean soak of `smoke_vla_web.sh`. After that, `training.calvin_loader` will become a no-op accepted only for backward compatibility.

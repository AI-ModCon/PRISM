# PRISM Projector Normalization Ablation Study

## Overview

This document describes our systematic ablation study for projector normalization strategies in the PRISM multimodal model. The goal is to find the optimal way to map image embeddings into the LLM's embedding space during projector-only and encoder+projector training.

## Model Architecture

| Component | Model | Hidden Dim | Status |
|-----------|-------|------------|--------|
| LLM Backbone | OLMo-1B-0724-hf | 2048 | Frozen |
| Image Encoder | SigLIP-base | 768 | Frozen (projector-only) / Trainable (encoder+projector) |
| Projector MLP | 2-layer MLP | 768 → 2048 → 2048 | **Trainable** |

## The Problem

During initial projector-only training, we observed:
- **Loss plateau at ~4.0** regardless of hyperparameter tuning
- **Severe norm mismatch**: Projector outputs (~45 with LayerNorm) were ~180x larger than text embedding norms (~0.25)
- Different normalization strategies produced similar plateau behavior

### Text Embedding Statistics (Reference Target)
```
Token Norm Mean:     0.25
Token Norm Std:      0.05
Element-wise Mean:   0.0
Element-wise Std:    0.006
```

---

## Normalization Modes

### Available Modes in `src/modules/projector.py`

| Mode | Description | Behavior |
|------|-------------|----------|
| `none` | No normalization | Raw MLP output; highest variance; norms ~77-88 |
| `layernorm` | Standard LayerNorm | Per-element standardization; norm ~45 (original baseline) |
| `rmsnorm` | RMSNorm (LLaMA-style) | Similar to layernorm but no mean centering |
| `l2_token` | Per-token L2 normalization | All tokens have exact target norm (0.25); may crush variance |
| `l2_sequence` | Sequence-level L2 normalization | Preserves token differences; controls mean norm |
| `scale_only` | Learned scalar multiplier | No normalization; just magnitude control |
| `match_text_stats` | Match token norm distribution | Transforms norms to match text mean/std (0.25 +/- 0.05) |
| `match_text_elemstats` | Match element-wise distribution | Like LayerNorm but scaled to text element stats |

### Modality Embedding Positions

| Position | Description |
|----------|-------------|
| `before_norm` | Added before normalization; affects direction; gets normalized |
| `after_norm` | Added after normalization (original behavior) |
| `none` | No modality embedding added |

---

## Projector-Only Ablation Results

**Training Configuration:**
- Learning Rate: 1e-3
- Warmup Steps: 500
- Max Steps: 2000
- Batch Size: 8
- Gradient Accumulation: 4
- Text Dropout: 0.1

### Loss Comparison

| Variant | Norm Mode | Embed Pos | Final Loss | Final Step | Notes |
|---------|-----------|-----------|------------|------------|-------|
| **SCALE** | scale_only | after_norm | **3.91** | 1150 | Best performer |
| **NONE-NOEMBED** | none | none | **3.95** | 1140 | 2nd best; no normalization or embed |
| L2TOKEN | l2_token | after_norm | 4.09 | 1160 | Exact norm control |
| NONE | none | after_norm | 4.14 | 1150 | Raw output with embed |
| L2TOKEN-NOEMBED | l2_token | none | 4.15 | 950 | - |
| RMSNORM | rmsnorm | after_norm | 4.19 | 1130 | - |
| L2TOKEN-SCALED | match_text_stats | after_norm | 4.21 | 960 | embed_scale=0.0055 |
| LAYERNORM | layernorm | after_norm | 4.21 | 1140 | Original baseline |
| LN-EMBED-BEFORE | layernorm | before_norm | 4.35 | 1150 | - |
| MATCH-ELEMSTATS | match_text_elemstats | after_norm | 4.43 | 1120 | - |
| LAYERNORM-NOEMBED | layernorm | none | 4.50 | 940 | - |
| MATCH-TEXT-NOEMBED | match_text_stats | none | 4.61 | 940 | Perfect norm matching |
| MATCH-TEXT | match_text_stats | after_norm | 4.65 | 1130 | - |
| EMBED-SCALED | l2_token | after_norm | 4.75 | 960 | embed_scale=0.0055 |
| L2SEQ | l2_sequence | after_norm | 4.92 | 1170 | Worst performer |

### Norm Statistics by Mode

| Mode | Img NormMean | Img NormStd | Norm Range | ElemStd | Text Norm |
|------|-------------|-------------|------------|---------|-----------|
| **none** | 77-88 | 22-27 | [20-26, 283-312] | 1.73-2.02 | 0.17-0.34 |
| **layernorm** | 0.244-0.251 | 0.017-0.018 | [0.18, 0.33] | 0.0054-0.0055 | 0.25-0.32 |
| **rmsnorm** | 0.244-0.248 | 0.017-0.018 | [0.18, 0.33] | 0.0054-0.0055 | 0.25-0.32 |
| **l2_token** | 0.2500 | 0.0000 | [0.25, 0.25] | 0.0055-0.0056 | 0.25-0.32 |
| **match_text** | 0.238-0.245 | 0.018 | [0.17, 0.33] | 0.0052-0.0054 | 0.24-0.26 |
| **scale_only** | ~24.0 | - | - | ~0.60 | - |

### Key Observations (Projector-Only)

1. **Best performers avoid heavy normalization**: `scale_only` (3.91) and `none-noembed` (3.95) outperform all normalized variants
2. **L2 normalization crushes semantic variance**: `l2_token` achieves perfect norm (0.25) but ElemStd drops to ~0.006, potentially destroying semantic information
3. **Modality embeddings hurt in some cases**: Removing modality embed often improves results (compare NONE vs NONE-NOEMBED)
4. **LayerNorm/RMSNorm perform identically**: Both produce similar distributions and losses
5. **Matching text statistics doesn't help**: `match_text_stats` modes underperform simpler approaches

---

## Encoder + Projector Ablation Results

**Training Configuration:**
- Connector LR: 2e-4
- Encoder LR: 6e-6 (differential learning rate)
- Scheduler: `molmo_layered` with staged warmup
- Warmup: Connector 200 steps, Main 2000 steps

### Loss Comparison

| Variant | Norm Mode | Embed Pos | Final Loss | Final Step |
|---------|-----------|-----------|------------|------------|
| **LAYERNORM** | layernorm | after_norm | **4.06** | 1760 |
| MATCH-TEXT-NOEMBED | match_text_stats | none | 4.30 | 1790 |
| NONE-NOEMBED | none | none | 4.30 | 1780 |
| NONE-EMBED | none | after_norm | 4.32 | 1780 |
| LAYERNORM-NOEMBED | layernorm | none | 4.64 | 1790 |

### Key Observations (Encoder+Projector)

1. **LayerNorm is optimal with encoder training**: Unlike projector-only, `layernorm` performs best (4.06) when encoder is trainable
2. **Modality embed matters here**: `LAYERNORM` with embed (4.06) outperforms `LAYERNORM-NOEMBED` (4.64)
3. **No clear winner among "none" variants**: All cluster around 4.30-4.32
4. **Lower loss than projector-only**: Encoder training helps but marginal improvement

---

## Experiment Design Files

### `experiments/prism_designs.yaml`

```yaml
# Projector-Only Ablation Group
PRISM-PROJ-ABLATION:
  common:
    xpus: 12
    walltime: "01:00:00"
    training: projector_only
    model: prism_image_only
    webdataset_dir: ${PRISM_DATA_ROOT}/zone_a/pixmo_cap_webdataset
    wandb_project: prism-proj-ablation-v2
  
  variants:
    PRISM-PROJ-ABLATION-LAYERNORM:
      model.projector_norm_mode: layernorm
      model.projector_modality_embed_pos: after_norm
    
    PRISM-PROJ-ABLATION-NONE-NOEMBED:
      model.projector_norm_mode: none
      model.projector_modality_embed_pos: none
    # ... (see full file for all variants)

# Encoder+Projector Ablation Group
PRISM-ENCODER-PROJ-ABLATION:
  common:
    training: encoder_projector
    # Uses differential LR and layered warmup
  
  variants:
    PRISM-EP-ABLATION-LAYERNORM:
      model.projector_norm_mode: layernorm
      model.projector_modality_embed_pos: after_norm
    # ... (see full file for all variants)
```

### `src/conf/model/prism_image_only.yaml`

```yaml
backbone_id: "allenai/OLMo-1B-0724-hf"
freeze_backbone: true
freeze_encoders: true
d_text: 2048
d_img: 768
modalities: ["text", "image"]

# Projector Configuration
projector_norm_mode: "layernorm"
projector_target_norm: 0.25
projector_modality_embed_pos: "after_norm"
projector_modality_embed_scale: 0.02

# Text Statistics Matching Parameters
projector_text_norm_mean: 0.25
projector_text_norm_std: 0.05
projector_text_elem_mean: 0.0
projector_text_elem_std: 0.006
projector_norm_clip_min: 0.1
projector_norm_clip_max: 0.5
```

---

## Training Configuration Files

### `src/conf/training/projector_only.yaml`

```yaml
learning_rate: 0.001
warmup_steps: 500
weight_decay: 0.01
max_steps: 2000
freeze_llm: true
freeze_vit: true
text_dropout: 0.1
```

### `src/conf/training/encoder_projector.yaml`

```yaml
learning_rate: 0.0002  # connector_lr
encoder_lr: 0.000006   # 6e-6 for ViT
warmup_steps: 200      # connector warmup
main_warmup_steps: 2000
freeze_llm: true
freeze_vit: false      # encoder is trainable
scheduler: molmo_layered
```

---

## Launch Commands

### Projector-Only Ablation
```bash
python tools/launch_aurora_web.py \
    --id PRISM-PROJ-ABLATION-NONE-NOEMBED \
    --design PRISM-PROJ-ABLATION-NONE-NOEMBED \
    --nodes 4 \
    --packed-env <your-deepspeed_env.tar.gz> \
    --webdataset-dir ${PRISM_DATA_ROOT}/zone_a/pixmo_cap_webdataset \
    --project AuroraGPT \
    --queue debug-scaling \
    --walltime 01:00:00 \
    --wandb-project prism-proj-ablation-v2 \
    --batch
```

### Encoder+Projector Ablation
```bash
python tools/launch_aurora_web.py \
    --id PRISM-EP-ABLATION-LAYERNORM \
    --design PRISM-EP-ABLATION-LAYERNORM \
    --nodes 4 \
    --packed-env <your-deepspeed_env.tar.gz> \
    --webdataset-dir ${PRISM_DATA_ROOT}/zone_a/pixmo_cap_webdataset \
    --project AuroraGPT \
    --queue debug-scaling \
    --walltime 01:00:00 \
    --wandb-project prism-encoder-proj-ablation \
    --batch
```

---

## WandB Tracking

**Projects:**
- Projector-only: `prism-proj-ablation-v2`
- Encoder+Projector: `prism-encoder-proj-ablation`

**Run URLs (Selected):**
- L2TOKEN: https://wandb.ai/ngetty/prism-proj-ablation-v2/runs/7qsxhac8
- L2SEQ: https://wandb.ai/ngetty/prism-proj-ablation-v2/runs/9fsuj4o0
- LAYERNORM: https://wandb.ai/ngetty/prism-proj-ablation-v2/runs/wf8ia247

---

## Conclusions & Recommendations

### For Projector-Only Training
1. **Use `scale_only` or `none` normalization** - Heavy normalization hurts
2. **Consider removing modality embeddings** - `none` embed position often helps
3. **Avoid L2 normalization** - Crushes semantic variance

### For Encoder+Projector Training
1. **Use `layernorm` with `after_norm` embed position** - Best performer (4.06)
2. **Keep modality embeddings** - They help when encoder is trainable
3. **Use differential learning rates** - Encoder needs much lower LR (6e-6 vs 2e-4)

### Next Steps
1. Run longer training (>2000 steps) to see if plateau breaks
2. Try unfreezing LLM with very low LR (full end-to-end training)
3. Experiment with different projector architectures (deeper MLPs, attention)
4. Evaluate on downstream tasks beyond loss (captioning quality metrics)

---

## Appendix: Validation Metrics (End-to-End Models)

| Model/Checkpoint | Word Overlap | ROUGE-L | BLEU-4 | Char Similarity |
|------------------|--------------|---------|--------|-----------------|
| molmo_tuned_6hr_step9000 | 0.1051 | 0.0705 | 0.0029 | 0.0218 |
| molmo_validation_v2 | 0.1092 | 0.0770 | 0.0020 | 0.0142 |
| molmo_multimetric | 0.1087 | 0.0609 | 0.0023 | 0.0157 |
| molmo_e2e_2node_step4500 | - | - | - | 0.0229 |

---

*Last updated: January 29, 2026*

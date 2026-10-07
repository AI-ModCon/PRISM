# Projector Architecture: Design, Configuration & Ablation Plan

The projector (a.k.a. "vision-language connector") bridges frozen encoder features into the LLM's embedding space. This document covers the current implementation, the Molmo2 reference design, a gap analysis, a concrete implementation plan for scaling up, and a registry of proposed ablation configurations.

---

## 1. Current Architecture

PRISM uses **two** projector types, assigned per-modality:

### 1.1 `ModalityProjector` (Simple MLP)

**Used for**: `text` (when no backbone), `image`
**File**: `src/modules/projector.py:25`

```
Input (B, T, input_dim)
  -> fc1: Linear(input_dim, d_model)     # Xavier uniform, bias=zeros
  -> act: GELU()
  -> fc2: Linear(d_model, d_model)       # Xavier uniform, bias=zeros
  -> [optional] modality_embedding       # (1, 1, d_model), scale=0.02
  -> _apply_normalization()              # 8 configurable modes
Output (B, T, d_model)
```

Key properties:
- 2-layer MLP with **no hidden expansion** (hidden dim = output dim)
- **GELU** activation (not gated)
- Preserves full token count (no spatial pooling)
- Xavier uniform init, zero bias

**Parameter counts** (image projector):

| Backbone | input_dim (d_img) | d_model | fc1 | fc2 | Total |
|----------|------------------|---------|-----|-----|-------|
| OLMo-1B | 768 | 2,048 | 1.6M | 4.2M | **~5.8M** |
| OLMo-7B | 1,152 | 4,096 | 4.7M | 16.8M | **~21.5M** |
| Nemotron-30B | 1,152 | 7,168 | 8.3M | 51.4M | **~59.7M** |

### 1.2 `PerceiverResampler` (Cross-Attention Pooling)

**Used for**: `table`, `time_series`, `geometry`, `graph`
**File**: `src/modules/adapter.py:5`

```
Input (B, T_in, input_dim)
  -> input_proj: Linear(input_dim, d_model)  [or Identity if dims match]
  -> latents: learnable (num_latents, d_model), init randn * 0.02
  -> N x PerceiverLayer:
       Cross-Attention: latents (Q) attend to input (K, V)
       Self-Attention: latents attend to themselves
       FFN: Linear(d_model, 4*d_model) -> GELU -> Linear(4*d_model, d_model)
  -> LayerNorm(d_model)
Output (B, num_latents, d_model)
```

Key properties:
- Compresses variable-length input to **32 fixed latent tokens**
- Pre-norm architecture (LayerNorm before attention/FFN)
- FFN uses 4x expansion with GELU
- Default: `num_latents=32`, `num_layers=2`, `num_heads=8`
- All parameters are **hardcoded** in `model.py` (not exposed in config)

### 1.3 Dimension Flow Summary

```
TEXT (with backbone):
  input_ids -> backbone.get_input_embeddings() -> (B, T, d_model)
  [No external encoder or projector — bypassed entirely]

IMAGE:
  (B, 3, H, W) -> SigLIP2 -> (B, patches, d_img) -> ModalityProjector -> (B, patches, d_model)

TABLE / TIME_SERIES / GEOMETRY / GRAPH:
  encoder_input -> Encoder -> (B, T, d_enc) -> PerceiverResampler -> (B, 32, d_model)

All modalities concatenated: [non-text tokens | text tokens] -> LLM backbone
```

---

## 2. Configuration Reference

### 2.1 Config Dataclass (`src/config.py:96-131`)

All projector config fields live in `ModelConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `projector_norm_mode` | `"layernorm"` | Normalization applied after MLP. See modes below. |
| `projector_target_norm` | `0.25` | Target L2 norm for `l2_token` / `l2_sequence` / `scale_only` modes |
| `projector_modality_embed_pos` | `"after_norm"` | Where to add modality embedding: `before_norm`, `after_norm`, `none` |
| `projector_modality_embed_scale` | `0.02` | Init scale for modality embedding parameter |
| `projector_text_norm_mean` | `0.25` | Target mean of token norms (for `match_text_stats`) |
| `projector_text_norm_std` | `0.05` | Target std of token norms |
| `projector_text_elem_mean` | `0.0` | Target element-wise mean (for `match_text_elemstats`) |
| `projector_text_elem_std` | `0.006` | Target element-wise std |
| `projector_norm_clip_min` | `0.1` | Min norm clamp |
| `projector_norm_clip_max` | `0.5` | Max norm clamp |

### 2.2 Normalization Modes

| Mode | Behavior |
|------|----------|
| `none` | Raw MLP output, no normalization |
| `layernorm` | Standard `nn.LayerNorm(d_model)` **(default)** |
| `rmsnorm` | RMSNorm with learnable weight (no centering) |
| `l2_token` | Per-token L2 normalization to learnable `output_scale` |
| `l2_sequence` | Sequence-level L2 norm, preserves relative token differences |
| `scale_only` | Learned scalar multiplier only |
| `match_text_stats` | Matches token norm distribution (mean/std) to text embeddings |
| `match_text_elemstats` | Matches element-wise distribution to text embeddings |

### 2.3 What is NOT Configurable Today

The following MLP architectural parameters are **hardcoded**:

- **Hidden dimension**: Always equals `d_model` (no expansion factor)
- **Number of layers**: Always 2 (`fc1 -> act -> fc2`)
- **Activation function**: Always GELU (no SwiGLU option)
- **Pooling**: None for image/text — all patch tokens are preserved
- **ViT layer selection**: Only `last_hidden_state` (no multi-layer extraction)
- **PerceiverResampler params**: `num_latents=32`, `num_layers=2`, `num_heads=8` are hardcoded in `model.py:222-281`

### 2.4 YAML Config Examples

**OLMo-1B image-only** (`src/conf/model/prism_image_only.yaml`):
```yaml
d_img: 768                          # SigLIP-base hidden_size
projector_norm_mode: "layernorm"
projector_target_norm: 0.25
projector_text_norm_mean: 0.25
projector_text_norm_std: 0.05
projector_text_elem_std: 0.006
```

**OLMo-7B image-only** (`src/conf/model/prism_7b_image_only.yaml`):
```yaml
d_img: 768
projector_target_norm: 0.35         # Larger for 7B
projector_text_norm_mean: 0.35
projector_text_norm_std: 0.07
projector_text_elem_std: 0.005
```

---

## 3. Molmo2 Reference Architecture

Based on the [Molmo2 Technical Report](https://arxiv.org/abs/2502.11089), the projector (called "Vision-Language Connector") uses a combination of **attention-based pooling** and a **SwiGLU MLP**.

### 3.1 Vision Encoder: SigLIP 2 So400m/14 384px

**Why SigLIP 2?** The Molmo2 authors explicitly chose SigLIP 2 despite it conflicting with their "fully open" goals (open weights, open data, open code). Their rationale (Appendix H):

1. **No competitive open alternatives**: "There are currently no competitive open-data encoders." Restricting to open-data vision encoders would have significantly degraded model performance.
2. **Outperforms video encoders**: Specialized pre-trained video encoders "lag behind using image encoders (such as SigLIP 2)" even on video tasks. SigLIP 2 provided better foundational visual features for both image and video.
3. **Dense features for grounding**: The encoder provides high-quality dense features (from intermediate layers), which are critical for Molmo2's fine-grained pointing and grounding tasks.

**Image preprocessing**: Molmo2 resizes images to 378x378 (distorting aspect ratio if needed) to match SigLIP 2's training methodology. This is a deliberate departure from Molmo1, which padded images to square.

**Relevance to PRISM**: We currently use `google/siglip2-base-patch16-224` (768d). Upgrading to `So400m/14` (1152d, 384px) is a config-only change (`d_img` + model ID), but the increased resolution and hidden dim affect both projector input dimensions and token count.

### 3.2 Multi-Layer ViT Feature Extraction

- Features extracted from **two** ViT layers: the 3rd-to-last and 9th-from-last
- Features are concatenated along the channel dimension, doubling the effective `d_img`
- Rationale: intermediate layers capture different levels of abstraction — earlier layers preserve fine spatial detail needed for grounding, later layers capture higher-level semantics

### 3.3 Attention Pooling (Spatial Token Reduction)

Instead of passing all patch tokens to the LLM, Molmo uses multi-headed attention to **pool** patches within spatial windows:

| Input Type | Window Size | Patches Pooled | Reduction Factor |
|------------|-------------|----------------|-----------------|
| Static Image | 2x2 | 4 patches -> 1 token | 4x |
| Video Frame | 3x3 | 9 patches -> 1 token | 9x |

Mechanism:
- **Query**: Mean of patches within the window
- **Key/Value**: The individual patches in the window
- Multi-headed attention produces one output token per window
- **Shared weights**: Same connector parameters handle both 2x2 and 3x3 pooling

For a 384px image with patch_size=14: `(384/14)^2 = 729 patches -> 729/4 ≈ 182 tokens` after 2x2 pooling.

### 3.4 SwiGLU MLP Projection

After attention pooling, features pass through a SwiGLU MLP:

```
Input (B, T_pooled, d_attn_out)
  -> w1: Linear(d_in, hidden_dim, bias=False)   # Gate
  -> w3: Linear(d_in, hidden_dim, bias=False)   # Up
  -> SiLU(w1(x)) * w3(x)                        # Gated activation
  -> w2: Linear(hidden_dim, d_model, bias=False) # Down projection
Output (B, T_pooled, d_model)
```

### 3.5 Parameter Counts by Backbone

| Variant | LLM Backbone | d_model | MLP Hidden Dim | Hidden/d_model Ratio | Total Projector Params |
|---------|-------------|---------|---------------|---------------------|----------------------|
| Molmo2-4B | Qwen2-4B | 3,584 | 9,728 | 2.71x | **~57M** |
| Molmo2-O-7B | OLMo-7B | 4,096 | 11,008 | 2.69x | **~80M** |
| Molmo2-8B | Qwen2-8B | 4,608 | 12,288 | 2.67x | **~88M** |

The hidden dimension scales at approximately **2.7x d_model**, consistent with the common SwiGLU convention of `(8/3) * d_model` (≈2.67x). Note that SwiGLU uses 3 weight matrices instead of 2, so the effective parameter count per "layer" is 1.5x a standard MLP of the same hidden dim.

### 3.6 Full Pipeline Comparison

```
Molmo2 Pipeline:
  Image (any aspect ratio)
    -> resize to 378x378 (distort to match SigLIP2 training)
    -> SigLIP2-So400m/14 (output_hidden_states=True)
    -> extract layers[-3] + layers[-9]
    -> concat -> (B, 729, 2*1152) = (B, 729, 2304)
    -> Attention Pool (2x2 windows, MHA) -> (B, ~182, d_attn)
    -> SwiGLU MLP (hidden=11008 for 7B) -> (B, ~182, 4096)
    -> LLM backbone

PRISM Pipeline (current):
  Image (224px)
    -> SigLIP2-base/16 -> last_hidden_state
    -> (B, 196, 768)
    -> ModalityProjector: Linear(768,4096)->GELU->Linear(4096,4096)->LayerNorm
    -> (B, 196, 4096)
    -> LLM backbone
```

---

## 4. Gap Analysis

| Feature | PRISM Current | Molmo2 | Gap |
|---------|--------------|--------|-----|
| **ViT model** | SigLIP2-base/16 (768d, 224px) | SigLIP2-So400m/14 (1152d, 384px) | Config change (model ID + `d_img`) |
| **Image preprocessing** | Resize to 224px | Resize to 378px, distort aspect ratio | Preprocessing change |
| **Multi-layer ViT** | `last_hidden_state` only | layers[-3] + layers[-9] concat | Not implemented |
| **MLP hidden dim** | `d_model` (1x, no expansion) | ~2.7x d_model | Hardcoded, needs config field |
| **Activation** | GELU | SwiGLU (3 matrices) | SwiGLU exists in `moe.py`, not wired to projector |
| **MLP param count (7B)** | ~21.5M | ~80M | 3.7x gap |
| **Spatial pooling** | None (all patches kept) | MHA with 2x2/3x3 windows | Not implemented |
| **Number of MLP layers** | 2 (fixed) | 3 matrices (SwiGLU) | Hardcoded |
| **PerceiverResampler config** | Hardcoded (32 latents, 2 layers) | N/A | Needs config exposure |
| **Projector type selection** | Per-modality hardcoded in `model.py` | N/A | Needs `projector_type` config |

---

## 5. Implementation Plan

### 5.1 New Config Fields (`src/config.py`)

Add to `ModelConfig` dataclass after the existing projector fields (~line 131):

```python
# === Projector Architecture Configuration ===
projector_type: str = "mlp"               # "mlp" (current), "swiglu", "molmo"
projector_hidden_dim: Optional[int] = None  # MLP hidden dim. None = d_model (current behavior)
projector_hidden_mult: float = 1.0        # Alternative: hidden_dim = d_model * mult (ignored if hidden_dim is set)
projector_num_layers: int = 2             # Number of MLP layers (2 = current)
projector_activation: str = "gelu"        # "gelu" (current), "silu", "swiglu"
projector_dropout: float = 0.0            # Dropout after projection

# === Attention Pooling (for "molmo" projector_type) ===
projector_use_attn_pool: bool = False     # Enable attention-based spatial pooling
projector_attn_pool_window: int = 2       # Pooling window size (2 = 2x2, 3 = 3x3)
projector_attn_pool_heads: int = 8        # Number of attention heads for pooling

# === Multi-Layer ViT Feature Extraction ===
image_encoder_layers: Optional[List[int]] = None  # ViT layer indices to extract. None = last only.
                                                    # e.g., [-3, -9] for Molmo2-style

# === PerceiverResampler Configuration (currently hardcoded) ===
perceiver_num_latents: int = 32
perceiver_num_layers: int = 2
perceiver_num_heads: int = 8
perceiver_dropout: float = 0.1
```

### 5.2 Projector Module Changes (`src/modules/projector.py`)

**Option A: Extend `ModalityProjector`** (minimal diff, backward compatible)

Add optional `hidden_dim`, `num_layers`, and `activation` parameters. When `hidden_dim` is not specified, fall back to current `d_model` behavior. Add a `"swiglu"` activation path that uses 3 weight matrices per layer (reuse `SwiGLUMLP` from `src/modules/moe.py:5`).

Sketch:
```python
class ModalityProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        hidden_dim: Optional[int] = None,  # NEW: None = d_model (backward compat)
        num_layers: int = 2,               # NEW: default matches current
        activation: str = "gelu",          # NEW: "gelu", "silu", "swiglu"
        dropout: float = 0.0,             # NEW
        # ... existing norm/embed params unchanged ...
    ):
        hidden = hidden_dim or d_model

        if activation == "swiglu":
            # SwiGLU: 3 matrices (gate, up, down) per layer
            # Layer 0: input_dim -> hidden, subsequent: d_model -> hidden
            self.mlp = nn.Sequential(
                SwiGLUMLP(input_dim, hidden, dropout),   # in -> hidden -> d_model? needs adaptation
                *[SwiGLUMLP(d_model, hidden, dropout) for _ in range(num_layers - 1)]
            )
        else:
            # Standard MLP with configurable depth and width
            layers = []
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.GELU() if activation == "gelu" else nn.SiLU())
            for i in range(num_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.GELU() if activation == "gelu" else nn.SiLU())
            layers.append(nn.Linear(hidden, d_model))
            self.mlp = nn.Sequential(*layers)
```

**Option B: New `MolmoProjector` class** (cleaner separation)

A dedicated class that combines attention pooling + SwiGLU MLP. Keeps `ModalityProjector` unchanged for backward compatibility. This is the recommended approach for the full Molmo2-style connector.

### 5.3 Attention Pooling Module (new)

New class in `src/modules/projector.py` or a new file `src/modules/attn_pool.py`:

```python
class WindowedAttentionPool(nn.Module):
    """
    Molmo2-style spatial attention pooling.
    Pools patches within NxN windows using multi-head attention
    where the query is the window mean.
    """
    def __init__(self, d_model: int, num_heads: int = 8, window_size: int = 2):
        ...

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, D) patch features (must know spatial layout)
            H, W: spatial dimensions of the patch grid
        Returns:
            (B, (H//window)*(W//window), D) pooled features
        """
        ...
```

### 5.4 Multi-Layer ViT Extraction (`src/encoders/image.py`)

Currently at line 76: `outputs.last_hidden_state`

Change to:
```python
def forward(self, pixel_values, output_hidden_states=False, extract_layers=None):
    outputs = self.model(pixel_values, output_hidden_states=True)

    if extract_layers:
        # Concatenate features from specified layers
        hidden_states = outputs.hidden_states  # tuple of (B, T, D) per layer
        features = torch.cat([hidden_states[i] for i in extract_layers], dim=-1)
        # features shape: (B, T, D * len(extract_layers))
    else:
        features = outputs.last_hidden_state
    ...
```

This changes the effective `d_img` to `d_vit * num_layers` when multi-layer extraction is enabled. The projector input dimension must account for this — either via an explicit config override or by having the encoder report its actual output dim (already supported via `self.output_dim`).

### 5.5 Model Wiring (`src/model.py`)

Update the image projector instantiation block (lines 242-257) to read the new config fields and select the appropriate projector class:

```python
# Image
if "image" in config.modalities:
    self.encoders["image"] = ImageEncoder(
        d_img=config.d_img,
        extract_layers=config.image_encoder_layers,  # NEW
    )
    img_input_dim = self.encoders["image"].output_dim  # accounts for multi-layer concat

    if config.projector_type == "molmo":
        self.projectors["image"] = MolmoProjector(
            input_dim=img_input_dim,
            d_model=target_dim,
            hidden_dim=config.projector_hidden_dim or int(target_dim * config.projector_hidden_mult),
            attn_pool_window=config.projector_attn_pool_window,
            attn_pool_heads=config.projector_attn_pool_heads,
        )
    elif config.projector_type == "swiglu":
        self.projectors["image"] = ModalityProjector(
            img_input_dim, target_dim,
            hidden_dim=config.projector_hidden_dim or int(target_dim * config.projector_hidden_mult),
            activation="swiglu",
            # ... existing norm/embed params ...
        )
    else:  # "mlp" — current behavior
        self.projectors["image"] = ModalityProjector(
            img_input_dim, target_dim,
            # ... existing params, backward compatible ...
        )
```

Similarly, update PerceiverResampler instantiation to use the new config fields:
```python
self.projectors["table"] = PerceiverResampler(
    input_dim=config.d_table,
    d_model=target_dim,
    num_latents=config.perceiver_num_latents,   # was hardcoded 32
    num_layers=config.perceiver_num_layers,      # was hardcoded 2
    num_heads=config.perceiver_num_heads,         # was hardcoded 8
    dropout=config.perceiver_dropout,             # was hardcoded 0.1
)
```

### 5.6 Files to Modify (Summary)

| File | Change |
|------|--------|
| `src/config.py` | Add ~12 new config fields to `ModelConfig` |
| `src/modules/projector.py` | Parameterize MLP (hidden_dim, activation, num_layers). Optionally add `MolmoProjector` and `WindowedAttentionPool`. |
| `src/modules/__init__.py` | Export new classes |
| `src/encoders/image.py` | Add `extract_layers` param, `output_hidden_states=True`, concat multi-layer features |
| `src/model.py` | Read new config fields, select projector type, pass PerceiverResampler config |
| `src/conf/model/*.yaml` | Add new YAML presets for ablation configs |

### 5.7 Existing Code to Reuse

- **`SwiGLUMLP`** in `src/modules/moe.py:5` — the standard 3-matrix SwiGLU implementation. Can be imported directly into the projector. Its interface (`d_model, hidden_dim, dropout`) already matches what we need.
- **`PerceiverResampler`** in `src/modules/adapter.py` — the cross-attention mechanism can be adapted for windowed attention pooling, or used as a reference for the new `WindowedAttentionPool`.

---

## 6. Proposed Ablation Configurations

### 6.1 Image Projector Ablations (OLMo-7B backbone, d_model=4096, d_img=1152)

| ID | Type | Hidden Dim | Activation | Layers | Attn Pool | Multi-Layer ViT | Est. Params | Notes |
|----|------|-----------|------------|--------|-----------|-----------------|-------------|-------|
| **A0** | mlp | 4,096 (1x) | GELU | 2 | No | No | ~21.5M | **Current baseline** |
| **A1** | mlp | 8,192 (2x) | GELU | 2 | No | No | ~71M | Wider MLP, same activation |
| **A2** | swiglu | 8,192 (2x) | SwiGLU | 1 | No | No | ~67M | SwiGLU, single gated layer |
| **A3** | swiglu | 11,008 (2.7x) | SwiGLU | 1 | No | No | ~90M | Molmo2-scale MLP, no pooling |
| **A4** | swiglu | 11,008 (2.7x) | SwiGLU | 1 | No | [-3, -9] | ~90M | + multi-layer ViT (d_img doubles) |
| **A5** | molmo | 11,008 (2.7x) | SwiGLU | 1 | 2x2 | [-3, -9] | ~95M | Full Molmo2-style connector |
| **A6** | mlp | 4,096 (1x) | GELU | 3 | No | No | ~38M | Deeper MLP (3 layers) |
| **A7** | mlp | 4,096 (1x) | GELU | 2 | No | [-3, -9] | ~30M | Multi-layer ViT only (d_img=2304) |

### 6.2 PerceiverResampler Ablations (OLMo-7B, d_model=4096)

| ID | Modality | Latents | Layers | Heads | Est. Params | Notes |
|----|----------|---------|--------|-------|-------------|-------|
| **P0** | table | 32 | 2 | 8 | ~540M | Current default |
| **P1** | table | 16 | 2 | 8 | ~540M | Fewer latents (tokens halved) |
| **P2** | table | 64 | 2 | 8 | ~540M | More latents |
| **P3** | table | 32 | 4 | 8 | ~1.1B | Deeper resampler |
| **P4** | table | 32 | 1 | 8 | ~270M | Shallower resampler |

### 6.3 Suggested YAML Configs

**Baseline** (`src/conf/model/projector_ablation_a0.yaml`):
```yaml
# A0: Current baseline (no changes needed, use existing config)
projector_type: "mlp"
projector_hidden_dim: null
projector_activation: "gelu"
projector_num_layers: 2
```

**SwiGLU Scaled** (`src/conf/model/projector_ablation_a3.yaml`):
```yaml
# A3: Molmo2-scale SwiGLU MLP, no attention pooling
projector_type: "swiglu"
projector_hidden_dim: 11008
projector_activation: "swiglu"
projector_num_layers: 1
```

**Full Molmo2** (`src/conf/model/projector_ablation_a5.yaml`):
```yaml
# A5: Full Molmo2-style connector
projector_type: "molmo"
projector_hidden_dim: 11008
projector_activation: "swiglu"
projector_use_attn_pool: true
projector_attn_pool_window: 2
projector_attn_pool_heads: 8
image_encoder_layers: [-3, -9]
```

---

## 7. Ablation Results Log

> Record experiment results here as they are completed.

| Date | Config ID | Backbone | Training Zone | Steps | Loss (final) | Eval Metric | Notes |
|------|-----------|----------|--------------|-------|-------------|-------------|-------|
| — | — | — | — | — | — | — | — |

### Template for recording a result:

```
| YYYY-MM-DD | A0 | OLMo-7B | encoder alignment | 5000 | 2.34 | CIDEr: 0.XX | Baseline run, LR=1e-4 |
```

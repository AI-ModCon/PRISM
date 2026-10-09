# Modules API

The reusable building blocks the backbone and the projection layer are made of.

Source: [`src/modules/`](../../src/modules/)

Full signatures and per-argument documentation live in the docstrings. This page
is the orientation layer — what each block is for and where it is used.

## Projection: encoder output → `d_model`

Encoders emit features at their own native width. These two modules bridge that
gap, and which one a modality gets is decided in stage 2 of
`UnifiedTransformer.__init__` (see [model.md](model.md)).

### `ModalityProjector`

`src/modules/projector.py` — an MLP from `input_dim` to `d_model` plus a
learnable per-modality embedding. Used for text, time series and image, where
the token count is already aligned and only the width needs changing.

It carries the knobs for two lines of ablation work:

- **`norm_mode`** — how the projected tokens are normalized. One of `none`,
  `layernorm` (default), `rmsnorm`, `l2_token`, `l2_sequence`, `scale_only`,
  `match_text_stats`, `match_text_elemstats`. The `match_text_*` modes pull the
  projected tokens toward the statistics of the text embeddings.
- **`modality_embed_pos`** — whether the modality embedding is added
  `before_norm`, `after_norm` (default), or `none`.
- **`hidden_mult` / `num_layers`** — IsoFLOP capacity knobs for the per-modality
  scaling study. `VARIANT_MAP` names the studied shapes (`BASE`, `W2X`, `W4X`,
  `D2X`, `D4X`).

One checkpoint constraint to respect: `(hidden_mult, num_layers) == (1, 2)` is
the legacy `fc1`/`fc2` path and keeps `state_dict` keys bit-identical, so
existing checkpoints load. Any other shape switches to a `ModuleList` and is
incompatible with old checkpoints by design.

`RMSNorm` in the same file is the LLaMA/OLMo-style norm used by `rmsnorm` mode.

### `PerceiverResampler`

`src/modules/adapter.py` — cross-attention onto a fixed bank of learnable
latents: `(B, T_in, input_dim)` → `(B, num_latents, d_model)`. Used for table,
geometry and graph, where token count is variable and the fused sequence needs a
predictable length. Inputs with more than 3 dimensions are flattened to
`(B, T, D)` first. `PerceiverLayer` is one cross-attention + self-attention
block of the stack.

## Backbone blocks

These are used by the custom transformer stack — the path taken when no Hugging
Face backbone is configured.

### `MoELayer`

`src/modules/moe.py` — sparse mixture of experts. A linear router picks
`num_experts_per_token` of `num_experts` per token, softmaxes their logits and
sums the selected experts' output.

`forward` returns **`(output, aux_loss)`**. The second element is the
load-balancing term — `num_experts * sum(fraction_of_tokens * average_prob)` —
which `TransformerBlock` passes up for the training loop to add to the main
loss. Dropping it silently disables load balancing and lets the router collapse
onto a few experts.

Two details worth knowing: during training the router logits get Gaussian noise
(σ = 0.1) for exploration, and the expert dispatch loop is the naive
iterate-over-experts implementation, not a grouped GEMM.

The expert MLP is chosen by `mlp_type`, both at hidden width `4 * d_model`:

| `mlp_type` | Class | Form |
|------------|-------|------|
| `swiglu` | `SwiGLUMLP` | `(SiLU(xW1) * xW3) W2`, no biases |
| `relusquared` | `ReLUSquaredMLP` | `ReLU(xW_fc)² W_proj` |

Anything else raises `ValueError`.

### `CausalSelfAttention`

`src/modules/attention.py` — multi-head causal self-attention with a fused
`c_attn` QKV projection and a registered lower-triangular mask buffer sized to
`max_seq_len`. `d_model` must be divisible by `num_heads`.

### `RotaryEmbedding`

`src/modules/attention.py` — standard RoPE, precomputing `cos`/`sin` tables out
to `max_position_embeddings`. `UnifiedTransformer` constructs it at head width
(`d_model // num_heads`) and passes the tables into each block as `freqs_cis`.

### `CausalLMHead`

`src/modules/heads.py` — a bias-free linear projection from `d_model` to
`vocab_size`: `(B, T, d_model)` → `(B, T, vocab_size)`.

## See also

- [model.md](model.md) — where these are assembled
- [encoders.md](encoders.md) — what feeds the projectors

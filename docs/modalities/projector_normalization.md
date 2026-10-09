# Projector Normalization Modes

This document details the math behind the `ModalityProjector` normalization modes
(`src/modules/projector.py`), with emphasis on `rmsnorm`, `scale_only`, and
`l2_sequence`, and on how DNA and text embeddings relate to these modes
differently. For the surrounding MLP architecture and config wiring, see
[projector.md](projector.md).

All modes operate on the projector's output `x` with shape `(B, T, D)`, where
`D = d_model` and `T` is the modality's token count. Configuration is via
`ModelConfig.projector_norm_mode` (`src/config.py`), which is applied uniformly
to every configured `ModalityProjector` instance (one per non-text modality —
DNA, image, table, etc.).

![Projector training phase (encoder alignment) architecture, showing the frozen DNA encoder and LLM embedding table, the trainable ModalityProjector and modality embedding, the normalization step, and the interleaved DNA/text sequence feeding the frozen LLM backbone](../assets/projector_training_arch.png)

*Encoder-alignment architecture: DNA flows through a frozen encoder into the trainable
`ModalityProjector`, where normalization rescales DNA token norms toward the
text embedding distribution before the two modalities are interleaved and
passed through the frozen LLM backbone.*

---

## 1. RMSNorm (`rmsnorm`)

Implementation: `RMSNorm` class, `src/modules/projector.py:10-22`. Same
formulation as LLaMA/OLMo.

For a single token embedding $x \in \mathbb{R}^D$:

$$\mathrm{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon}$$

$$\hat{x} = \frac{x}{\mathrm{RMS}(x)} \odot \gamma$$

- $\epsilon = 10^{-6}$ (numerical floor)
- $\gamma \in \mathbb{R}^D$ is a learnable per-channel weight, initialized to all-ones
- $\odot$ denotes elementwise product
- Applied independently per token, along the last dimension only — shape-preserving over `(B, T, D)`

Unlike `nn.LayerNorm`, there is **no mean-centering** ($x - \mu$) before the
scale — only the second moment (mean square) is used. This is cheaper than
LayerNorm and preserves the sign/direction of the raw activations, which
matters when a downstream embedding table's directionality carries semantic
information.

---

## 2. Scale-Only (`scale_only`)

Implementation: `src/modules/projector.py:207-209`.

$$\hat{x} = x \cdot s$$

- $s \in \mathbb{R}^1$ is a single learnable scalar, shared across every token, channel, and batch element
- Initialized to `target_norm` (default `0.25`, config field `projector_target_norm`)

This is **not a normalization** in the statistical sense — it applies no
per-token or per-channel renormalization at all, just a uniform rescale of
whatever the MLP produced. Relative token-to-token norm differences and
elementwise structure are untouched. It exists as the "control" condition in
the ablation matrix: does the model just need a global scale correction to
align modality embeddings with the LLM's embedding scale, or does it need
true normalization (RMSNorm/LayerNorm/L2)?

---

## 3. Sequence-Level L2 Normalization (`l2_sequence`)

Implementation: `src/modules/projector.py:199-205`.

Per-token L2 norm within a sequence of $T$ tokens:

$$\|x_t\|_2 = \sqrt{\sum_{i=1}^{D} x_{t,i}^2}, \qquad t = 1, \dots, T$$

Mean norm across the sequence:

$$\bar{n} = \frac{1}{T}\sum_{t=1}^{T} \|x_t\|_2$$

Rescale every token by the same shared mean, then by the learnable scalar:

$$\hat{x}_t = \frac{x_t}{\bar{n}} \cdot s$$

- $s$ = `output_scale`, initialized to `target_norm` (same parameter class as `scale_only`)
- $\bar n$ is clamped to a minimum of $10^{-6}$ to avoid division by zero

Because every token in the sequence is divided by the **same** scalar $\bar
n$, relative norm differences between tokens are preserved (a token that was
2x larger than another stays 2x larger after normalization) — only the
sequence's overall scale is pulled toward `target_norm`.

### Contrast: `l2_token`

The sibling mode `l2_token` (`src/modules/projector.py:192-197`) instead
forces **every token individually** to the same norm:

$$\hat{x}_t = \frac{x_t}{\|x_t\|_2} \cdot s \qquad \Rightarrow \qquad \|\hat{x}_t\|_2 = s \ \ \forall t$$

This destroys all token-to-token norm variance (every token has identical
magnitude $s$), whereas `l2_sequence` only equalizes the *average*. The two
modes are direct ablation counterparts for testing whether per-token norm
variance carries useful signal for the LLM backbone.

---

## 4. Text vs. DNA: where the asymmetry actually is

This is a structural asymmetry in the pipeline, not a difference in formula:

- **DNA embeddings** come from `DNAEncoder` (Nucleotide Transformer / Evo2
  hidden states, `d_dna = 1024`) and are **always** routed through a
  `ModalityProjector` (`src/model.py:337-348`), which applies whichever
  `norm_mode` is configured, plus an additive `modality_embedding`.
- **Text embeddings** are the LLM backbone's *native* input embeddings (e.g.
  OLMo's/Llama's own `nn.Embedding` table). Text **never** passes through a
  `ModalityProjector` or any of the norm modes above — it is the reference
  distribution the other modalities are being aligned to, not a modality
  being normalized itself.

This is why `ModelConfig` carries a separate block of **text statistics**,
used only by the two distribution-matching modes
(`src/config.py:198-205`):

```python
projector_text_norm_mean: float = 0.25   # empirical text token-norm mean (OLMo-1B)
projector_text_norm_std:  float = 0.05
projector_text_elem_mean: float = 0.0
projector_text_elem_std:  float = 0.006
projector_norm_clip_min:  float = 0.1
projector_norm_clip_max:  float = 0.5
```

### `match_text_stats`

(`src/modules/projector.py:211-229`) — standardizes DNA's per-token norm
distribution, then rescales to match the observed **text** norm statistics:

$$\hat{n}_t = \frac{n_t - \mu_{\text{DNA}}}{\sigma_{\text{DNA}}} \cdot \sigma_{\text{text}} + \mu_{\text{text}}, \qquad \hat{n}_t \leftarrow \mathrm{clip}(\hat{n}_t,\ 0.1,\ 0.5)$$

$$\hat{x}_t = \frac{x_t}{n_t} \cdot \hat{n}_t$$

where $n_t = \|x_t\|_2$, $\mu_{\text{DNA}}, \sigma_{\text{DNA}}$ are the
current batch's DNA token-norm mean/std, and $\mu_{\text{text}} = 0.25$,
$\sigma_{\text{text}} = 0.05$ are the fixed text targets. This preserves each
token's direction while remapping its norm into the text distribution.

### `match_text_elemstats`

(`src/modules/projector.py:231-266`) — same idea, but matches the
**elementwise** distribution (per-channel mean/std) instead of the per-token
norm:

$$\hat{x} = \frac{x - \mu_{\text{elem}}}{\sigma_{\text{elem}}} \cdot \sigma_{\text{text-elem}} + \mu_{\text{text-elem}}$$

followed by a clip on the resulting token norm into $[0.1, 0.5]$ if it falls
outside range.

### Practical takeaway

DNA embeddings are the ones being actively rescaled to statistically
resemble text embeddings — because the Nucleotide Transformer / Evo2 encoder
produces activations at a very different scale than the LLM's own embedding
table, and a scale mismatch at the fusion point destabilizes early
multimodal training. The `modality_embedding`
(`src/modules/projector.py:124-136`, small-norm additive vector, default
init scale `0.02`) is layered on top of whichever norm mode is active —
purely as a modality-identity signal, not a scale correction.

---

## 5. Summary Table

| Mode | Normalizes | Preserves relative token norms? | Learnable params | Text-aware? |
|------|-----------|----------------------------------|-------------------|-------------|
| `rmsnorm` | Per-token RMS | No (all tokens → unit RMS) | Per-channel weight $\gamma$ | No |
| `scale_only` | Nothing (pure rescale) | Yes (identity up to scalar) | Scalar $s$ | No |
| `l2_sequence` | Sequence-mean L2 | Yes | Scalar $s$ | No |
| `l2_token` | Per-token L2 | No (all tokens → norm $s$) | Scalar $s$ | No |
| `match_text_stats` | Per-token norm distribution | Approximately (direction kept, norm remapped) | None | Yes |
| `match_text_elemstats` | Per-channel distribution | No (elementwise restandardized) | None | Yes |
| `layernorm` | Per-token mean + variance | No | $\gamma, \beta$ | No |
| `none` | — | Yes (raw MLP output) | None | No |

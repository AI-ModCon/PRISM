# Encoders API

The input side: one encoder per modality, each turning a raw scientific input
into tokens the backbone can consume.

Source: [`src/encoders/`](../../src/encoders/)

Full signatures and per-argument documentation live in the docstrings. This page
is the orientation layer — what each encoder wraps and what it emits.

## The contract

Every encoder subclasses `ModalityEncoder` (`src/encoders/base.py`) and
implements `forward(inputs) -> (B, T, output_dim)`. Two other members matter:

- **`output_dim`** — the encoder's native feature width, set in `__init__`.
  It is *not* the model's `d_model`; a projector bridges the two (see
  [model.md](model.md)).
- **`tokens_per_instance()`** — how many tokens this modality contributes per
  instance, used by the interleaving logic. Defaults to `1`; override it for
  encoders that emit several tokens per instance, such as time-series patches.
- **`_last_pad_mask`** — encoders that right-pad variable-length output may set
  this to a bool `(B, T)` mask where `True` marks padding.

## The encoders

| Modality | Class | Wraps | Input → output |
|----------|-------|-------|----------------|
| Text | `TextEncoder` | SmolLM2-360M-Instruct (configurable) | `(B, T_text)` token ids → `(B, T_text, d_text)` |
| Image | `ImageEncoder` | SigLIP2 `google/siglip2-base-patch16-224` | `(B, 3, H, W)` → `(B, T_patches, d_img)` |
| Table | `TableEncoder` | TAPAS `google/tapas-base` | DataFrame or dict of lists → `(B, T_table, d_table)` |
| Time series | `TimeSeriesEncoder` | Moirai / TimeOmni / linear | `(B, T_ts, num_vars)` → `(B, T, d_ts)` |
| Geometry | `GeometryEncoder` | Walrus `polymathic-ai/walrus` | point cloud / physical field → `(B, T, d_geo)` |
| Graph | `GraphEncoder` | GraphMAE2 with a GAT backbone | node features + `edge_index` → `(B, N, d_graph)` |
| Crystal graph | `CrystalGraphTokenEncoder` | Periodic message passing (no pretrained weights) | packed atoms + `edge_index`/`edge_distance` → `(B, num_tokens, hidden_dim)` |
| DNA | `DNAEncoder` | Nucleotide Transformer / Evo2 | `(B, T)` token ids → `(B, T, d_dna)` |

All eight are re-exported from `src.encoders`.

## Time series has three backends

`TimeSeriesEncoder` is the one encoder with a mode switch, selected by
`encoder_type`:

| `encoder_type` | Behavior |
|----------------|----------|
| `moirai` (default) | Salesforce Moirai via Hugging Face. Emits one token per patch. |
| `timeomni` | TimeOmni-style dynamic patch embedding. Accepts a ragged list of `(T_i, num_vars)` tensors as well as a padded tensor, and pads token counts across the batch. |
| `linear` | Plain linear projection of each variate to `d_ts`. No pretrained weights. |

One sharp edge, stated in the source: **choose a patch size that divides the
maximum length.** Otherwise, in interleaved mode a patch can span two series
instances and the model learns artifacts at the boundary.

## Optional dependencies

`GraphEncoder` needs PyTorch Geometric; `GeometryEncoder` needs Walrus. Neither
is in the base install.

Importing `src.encoders` always succeeds — both modules swallow their own
`ImportError` at module scope so the package stays importable without the extras.
The loud failure comes later, from `require_modality_deps()` inside each
`__init__`, which raises with an install hint at the point you actually try to
construct the encoder.

`GeometryEncoder` has one escape hatch: with `WALRUS_FALLBACK=1` set, a missing
Walrus install falls back to `FallbackGeometryEncoder`, a flatten-then-linear
stub. It exists so throughput sweeps stay runnable on broken installs and it
logs a warning when used. **Its features are meaningless** — never use it for a
real run.

On Aurora, never let pip resolve the PyG C++ extensions from PyPI; see
[CONTRIBUTING.md](../../CONTRIBUTING.md) for the install procedure.

## See also

- [model.md](model.md) — how encoder output reaches the backbone
- [modules.md](modules.md) — the projectors that bridge `output_dim` to `d_model`

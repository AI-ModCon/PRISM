---
name: prism-adding-a-modality
description: >
  Add a new modality (encoder + projector) to PRISM end-to-end. Use when writing
  a new ModalityEncoder subclass, registering a projector, extending the Modality
  enum, wiring config fields, handling the token-merge path, or adding a dataset
  group for the modality. Triggers: "add a modality", "new encoder", "subclass
  ModalityEncoder", "ModalityProjector", "Modality enum", "tokens_per_instance",
  "merge modality embeds", "special modality token".
metadata:
  version: "1.0"
  project: prism
---

# PRISM — Adding a Modality

A modality flows: **raw input → `ModalityEncoder` → `ModalityProjector` → merged
into the token sequence at special-token slots → LLM backbone**. Adding one
touches ~6 files in a fixed pattern. Full step-by-step:
[`references/encoder-checklist.md`](references/encoder-checklist.md).

## The pattern (files you'll touch)

| Step | File | What |
|------|------|------|
| 1 | `src/encoders/<mod>.py` | Subclass `ModalityEncoder` (base: `src/encoders/base.py`) |
| 2 | `src/modules/projector.py` | Reuse `ModalityProjector` — usually no change |
| 3 | `src/modalities.py` | Add member to the `Modality` enum + `ALL_MODALITIES` |
| 4 | `src/config.py` | Add `d_<mod>` dim + projector fields to `ModelConfig` |
| 5 | `src/model.py` | Instantiate encoder/projector; handle the merge slot |
| 6 | `src/conf/data/*.yaml` + sharding | Add a dataset group (see prism-data-pipeline) |

## Encoder contract (`src/encoders/base.py`)

```python
class ModalityEncoder(nn.Module):
    def __init__(self, output_dim: int): ...
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...   # required
    def tokens_per_instance(self) -> int: return 1                 # override if >1
```

`tokens_per_instance()` is **load-bearing** for the merge: it tells `model.py`
how many token slots one instance expands to (e.g. time-series expands to 256).
Existing encoders: `image`, `text`, `table`, `time_series`, `geometry`
(+ `FallbackGeometryEncoder`), `graph`.

## Projector (`src/modules/projector.py`)

`ModalityProjector(input_dim, d_model, norm_mode=..., ...)`. Norm modes:
`none, layernorm, rmsnorm, l2_token, l2_sequence, scale_only, match_text_stats,
match_text_elemstats`. Depth/width via `hidden_mult`, `num_layers` (the IsoFLOP
variant ladder BASE/W2X/W4X/D2X/D4X lives here). Configured through
`projector_*` fields on `ModelConfig`.

## Merge path (`src/model.py`, `_merge_text_input_ids_with_modality_embeds`, ~line 533)

Special modality tokens (IDs in `config.modality_start_end_token_indices`) are
replaced by encoder outputs; slot size = `encoders[mod].tokens_per_instance()`.
`UnifiedTransformer.__init__` (~lines 221–354) instantiates encoders/projectors
conditionally per `config.modalities` into `nn.ModuleDict`s.

## Gotchas learned the hard way

- **Tokenizer resize must use the LLM tokenizer + grow-only.** Resizing
  embeddings with the *base* tokenizer (50304→50280) shrinks the table; the
  modality special token (e.g. `<ts>`=50280) then indexes out of bounds → a
  delayed XPU write-fault/segfault (issue #117). Always resize with
  `llm_tokenizer_id` and never shrink.
- **Budget the *total* merged sequence length, not just the modality tokens.**
  An oversized merged sequence OOMs the 64 GB XPU tile (attention ≈ O(T²), cliff
  ~4608 tokens), surfacing as a `drm_neo` page-fault under DDP+oneCCL — not an
  obvious OOM (issues #120/#122/#123). Cap variates at the data path AND reserve
  headroom for text. A model-layer fail-loud guard before the backbone forward is
  the follow-up (#123).
- **`--max-seq-length 4096 // 256 = 16 = all variates`**, so a per-modality
  variate cap can be inert; the real fix budgets merged length end-to-end.

## Verify

Add a smoke config and run a 1-node smoke (see prism-launching-jobs `--dry-run`
first). Confirm the encoder's tokens land at the right slots and loss decreases.

## See also

- Deep references: [`docs/api/encoders.md`](../../api/encoders.md),
  [`docs/modalities/projector.md`](../../modalities/projector.md).
- [prism-data-pipeline](../prism-data-pipeline/SKILL.md), [prism-configuration](../prism-configuration/SKILL.md).

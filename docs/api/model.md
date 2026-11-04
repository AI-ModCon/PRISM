# Model API

The top-level model: how a batch of mixed-modality inputs becomes one token
sequence, and how that sequence is configured.

Source: [`src/model.py`](../../src/model.py), [`src/config.py`](../../src/config.py)

Full signatures and per-argument documentation live in the docstrings. This page
is the orientation layer — what the pieces are and how they fit.

## `UnifiedTransformer`

`src/model.py` — the backbone. One instance owns every encoder, projector and
decoder for the modalities its config enables.

The forward path runs in four stages, marked in the source by `# --- Stage N ---`
comments:

| Stage | What happens |
|-------|--------------|
| 0 | Optionally load a pretrained Hugging Face backbone (`config.llm_backbone_id`). When unset, the custom transformer stack of `TransformerBlock` layers is built instead. |
| 1 | Build `self.encoders` — one `ModalityEncoder` per active modality. |
| 2 | Build `self.projectors` — map every encoder output to `d_model`. |
| 3 | Concatenate the projected embeddings into a single sequence and run the backbone over it. Output decoders in `self.decoders` turn the resulting hidden states back into modality-native predictions. |

Stage 2 uses one of two projectors, chosen per modality:

| Projector | Modalities | Why |
|-----------|-----------|-----|
| `ModalityProjector` | text, time series, image | Token count is already aligned; project each token to `d_model` and tag it with a modality embedding. |
| `PerceiverResampler` | table, geometry, graph | Token count is variable; cross-attend to a fixed set of learnable latents so the fused sequence has a predictable length. |

Text is the only modality with an autoregressive serving path. Every output
decoder is single-forward and AR-safe, so `generate` stays valid regardless of
which decoders are configured — see [the decoder contract](#decoders).

`forward` returns `(logits, loss)`; `loss` is `None` when no labels are passed.

## `TransformerBlock`

`src/model.py` — one layer of the custom (non-HF) stack: pre-norm
`CausalSelfAttention`, then pre-norm `MoELayer`, both residual. `forward`
returns `(hidden_states, aux_loss)`, where `aux_loss` is the MoE
load-balancing term the training loop adds to the main loss. It is only
constructed when no HF backbone is configured.

## `ModelConfig`

`src/config.py` — a dataclass holding model geometry, backbone id, active
modalities and decoder selection. Per-field documentation is in the source as
attribute docstrings.

`ModelConfig.from_preset(name)` resolves one of the entries in `PRISM_CONFIGS`
and raises `ValueError` on an unknown name. The preset keys are **hyphenated**:

```
prism-nano                            prism-micro
prism-mini                            prism-small
prism-base                            prism-granite-2b
prism-phi4-mini                       prism-auroragpt-2b
prism-olmo3-7b                        prism-nemotron-30b
prism-olmo-ts-7b                      prism-olmo-ts-1b-interleaved
prism-olmo-linear-ts-1b-interleaved   prism-olmo-linear-ts-7b-interleaved
```

The underscored names under `src/conf/model/` are Hydra YAML filenames, not
preset keys — `from_preset("prism_olmo3_7b")` raises.

```python
from src.config import ModelConfig
from src.model import UnifiedTransformer

config = ModelConfig.from_preset("prism-olmo3-7b")
model = UnifiedTransformer(config)
```

## Backend metadata contract (version 1)

`src.backends` provides metadata/discovery/resolution only. It imports only the
standard library and never constructs a model, imports torch/transformers,
fetches assets, or contacts a network. There is no automatic execution
integration: callers explicitly pass the selectors to `resolve_backend`.

```python
from src.backends import describe_backend, list_backends, resolve_backend

list_backends()  # ("local_ar", "hf_ar", "u2_reference", "u2_native")
describe_backend("u2_native").status  # "planned" (discovery is allowed)
resolve_backend(required_capabilities=("causal_forward",)).name  # "local_ar"
resolve_backend(llm_backbone_id="allenai/Olmo-3-7B-Instruct").name  # "hf_ar"
```

`BackendDescriptor` is frozen, with `name`, `contract_version=1`, immutable
`frozenset[str]` capabilities, and `status` (`"existing"` or `"planned"`).
The builtin registry is immutable; `list_backends()` returns its names as a
tuple in the order shown above. `describe_backend(name)` returns a descriptor.

| Backend | Status | Capabilities |
|---------|--------|--------------|
| `local_ar` | existing | `causal_forward` |
| `hf_ar` | existing | `causal_forward`, `autoregressive_generate` |
| `u2_reference` | planned | none |
| `u2_native` | planned | none |

`resolve_backend(*, backend=None, llm_backbone_id=None, contract_version=1,
required_capabilities=())` returns a descriptor. Default selection follows
the current `llm_backbone_id` truthiness: `None` or `""` selects local; any
nonempty string selects HF. It does not validate the model ID or availability.
Legacy `hf_model_id` is a weight-initialization field, not a selector; presets
that set only that field still resolve to local. Explicit `hf_ar` requires a
nonempty backbone ID; explicit `local_ar` with a nonempty ID raises `ValueError`.
Malformed/unknown selectors, versions other than integer 1 (including `True`),
and unknown/unsupported capability requests also raise `ValueError`. Capabilities
are an iterable of names, not a bare string or mapping. Planned U2 resolution
raises `NotImplementedError` without fallback; malformed or unknown capability
requests are rejected first.

Local `UnifiedTransformer.generate()` currently returns `None`, so local
generation is not advertised. This contract makes no modality, structured
decoder, diffusion, training, quality, or accelerator/XPU validation claims,
and provides no pretrained U2 functionality. Existing construction, Hydra keys,
`ModelConfig`, checkpoint keys, and APIs are unchanged; no migration is required.

## Decoders

`src/decoders/` — the output side, registered by name in `DECODERS`
(`src/decoders/__init__.py`). `UnifiedTransformer` builds `self.decoders` from
that table, keyed by the names in `config.output_decoders`:

| Key | Class |
|-----|-------|
| `text` | `LMHeadDecoder` |
| `action`, `regression` | `RegressionDecoder` |
| `time_series` | `TimeSeriesDecoder` |
| `geometry` | `GeometryDecoder` |
| `graph` | `GraphDecoder` |
| `image` | `ImageDecoder` |

Every decoder subclasses `OutputDecoder` (`src/decoders/base.py`) and is
**single-forward and AR-safe**: one forward produces the whole output, so the
autoregressive serving path for text is preserved. Work needing iterative or
non-autoregressive decoding — discrete diffusion, de-novo graph structure
generation — is deliberately out of scope. `forward` returns
`(prediction, loss)`, with `loss` `None` when `targets is None`.

The keys are load-bearing: they appear in configs and in checkpoints, so
renaming one breaks existing checkpoints. `remap_legacy_decoder_keys` handles
the one rename that has already happened.

## See also

- [encoders.md](encoders.md) — the input side
- [modules.md](modules.md) — the reusable building blocks

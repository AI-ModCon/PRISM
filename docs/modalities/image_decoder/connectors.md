# Connectors from the PRISM trunk to output decoders

Output decoders separate their conditioning paths into **readout → bridge →
generator**. The OmniGen2 image decoder and simple time-series forecaster are
two instantiations of this API. The public `forward_outputs(...)` and prediction
APIs remain unchanged.

```text
Input encoders → input projectors → PRISM transformer trunk
                                         ├── language head
                                         ├── all valid final-layer states
                                         │   → LayerNorm + Linear bridge
                                         │   → OmniGen2 diffusion transformer
                                         │   → image VAE decoder
                                         └── pooled final-layer state
                                             → identity bridge
                                             → linear quantile forecaster
```

Input projectors remain unchanged. Input and output connectors can reuse a design
or implementation; learned connectors have separate parameters for mapping
encoder features into the trunk and trunk states into a generator. The
time-series identity bridge needs no additional learned transformation.

For a simpler scientific output, the
[time-series forecasting configuration](../time_series_decoder.md) uses the
same typed contracts with masked final-state pooling, an identity bridge and a
linear quantile head. Each decoder validates the choices its generator supports.

## Supported configuration

The complete Qwen3-1.7B/SigLIP2 configuration is
[`qwen3_1_7b_prism_harness_omnigen2_connectors.json`](../../../src/conf/image_generation/qwen3_1_7b_prism_harness_omnigen2_connectors.json).
Its image decoder configuration is:

```json
{
  "readout": {
    "type": "select",
    "layers": "final",
    "positions": "all_valid"
  },
  "bridge": {
    "type": "layernorm_linear",
    "output_dim": 2048
  },
  "generator": {
    "type": "omnigen2",
    "model_id": "/lus/flare/projects/ModCon/sandeep/prism-image-smoke-20260921/assets/OmniGen2",
    "revision": "df5dca8a981d74e6c3af214c145f5c735fe72367",
    "local_files_only": true,
    "conditioning_dim": 2048
  }
}
```

These remain the image decoder's supported choices. The time-series decoder
supports final-layer `pool` readout (`last` or `mean`), an `identity` bridge and
the `linear_quantile` generator:

```yaml
decoder_configs:
  time_series:
    readout: {type: pool, layers: final, pool: last}
    bridge: {type: identity, output_dim: 1024}
    generator:
      type: linear_quantile
      horizon: 96
      num_vars: 1
      quantiles: [0.1, 0.5, 0.9]
```

Its complete Qwen3-0.6B / Intern-S2 input-model preset is
[`prism_qwen3_0_6b_intern_s2_397b_ts_forecast.yaml`](../../../src/conf/model/prism_qwen3_0_6b_intern_s2_397b_ts_forecast.yaml).
The forecasting output is implemented; the referenced Intern-S2 input encoder
still requires integration from its source branch before running that full
model. The identity bridge keeps the pooled context `[B,1,1024]`; the generator
produces quantiles `[B,96,1,3]`. It has no diffusion sampler or VAE.

Unsupported intermediate or mixed layers, query settings, bridge choices and
generator types are rejected. Q-Former, Perceiver IO and learned query
connectors are not implemented by these configurations.

The image readout selects **all valid final-layer positions**, not only text positions.
For caption-only training, these are the templated caption states. If the trunk
input also includes observations, their valid states are selected as well.

With hidden states `[B, L, 2048]`, the readout preserves their sequence layout and
mask. The bridge normalizes and projects each valid state from width 2048 to the
generator's conditioning width 2048. The prepared decoder context packs valid
states into their original order and right-pads each example to the largest valid
length in the batch. Thus its shape is `[B, N_max, 2048]`, where
`N_max = max(attention_mask.sum(1))`. It does not produce a fixed number of query
tokens. Equal widths do not imply identical feature spaces; the bridge still
learns the alignment.

## API contracts

The conditioning contracts live in `src.connectors`:

| Contract | Responsibility |
| --- | --- |
| `BackboneFeatures` | Final trunk hidden states, validity mask, source modality spans, and provenance. |
| `ReadoutResult` | Selected or pooled tokens, validity mask, optional positions in the original trunk sequence, source modality spans, and provenance. |
| `DecoderContext` | Generator-ready tokens and mask, the retained source-position mapping and source metadata, and conditioning provenance. |

`source_modality_spans` describe the **original trunk sequence**. They must not be
interpreted as spans in the packed context. `source_positions` provides the
explicit mapping back to that sequence; masked context slots have no source
position and carry `-1`. A readout without a source-position mapping can use
`None`. This distinction leaves room for future readouts with a different
layout without pretending that old indices still identify output tokens.

`Readout.forward(BackboneFeatures) -> ReadoutResult` and
`ConditioningBridge.connect(ReadoutResult) -> DecoderContext` are the extension
interfaces. The shared factories `build_readout` and `build_bridge` construct
`FinalStateReadout` / `PooledStateReadout` and `LayerNormLinearBridge` /
`IdentityBridge`; each output decoder restricts these to its supported
conditioning route.
The contracts carry conditioning observations and metadata; target values remain
separate arguments to the decoder's training method.

Both decoders' `prepare_condition(condition)` methods accept the existing
`DecoderCondition` and return a `DecoderContext`. Readout and bridge are separate
components. The compatibility method `ImageDecoder.connect(condition)` still
returns `(embeddings, attention_mask)`. Image training and generation continue to
use the same backend entry points and native-context options.

Each decoder's `conditioning_contract()` exposes the normalized readout, bridge,
and generator settings. It describes the selected architecture, rather than a claim
that a particular checkpoint has been trained or evaluated.

The reusable `pack_right_padded` operation preserves valid-token order, compacts
left padding or noncontiguous masks, and moves source-position mappings with the
tokens. Padding is masked before the image bridge's normalization so padded values do
not contaminate connector gradients. An entirely masked example is invalid.

## Loading the configuration

Existing `ModelConfig` construction accepts the nested image configuration:

```python
import json
from pathlib import Path

from src.config import ModelConfig
from src.model import UnifiedTransformer

config_path = Path(
    "src/conf/image_generation/"
    "qwen3_1_7b_prism_harness_omnigen2_connectors.json"
)
config = ModelConfig(**json.loads(config_path.read_text()))
model = UnifiedTransformer(config)
image_decoder = model.decoders["image"]
contract = image_decoder.conditioning_contract()
```

Constructing the model requires access to the configured PRISM assets. The
configuration retains the existing Aurora-local asset paths and pinned generator
revision; it does not download or launch anything itself. Existing tools that
accept `--model-config` can be pointed at this new JSON for a new run.

## Checkpoint and training compatibility

The legacy flat image configuration remains supported. Omitting `readout` and
`bridge` gives the current final/all-valid, LayerNorm/Linear path. The bridge stays
registered under `connector`, preserving state-dict keys
`connector.0.{weight,bias}` and `connector.1.{weight,bias}` and existing optimizer
parameter access. The readout adds no learned weights. The existing reference
mode still uses native OmniGen2 conditioning.

Legacy flat time-series configurations also remain supported. The nested
configuration separates the same pooling and linear head without changing
`head.weight` or `head.bias` keys; its readout and identity bridge are
parameter-free. Last pooling records the selected source position, while mean
pooling returns no single source-position mapping.

The original
[`qwen3_1_7b_prism_harness_omnigen2.json`](../../../src/conf/image_generation/qwen3_1_7b_prism_harness_omnigen2.json)
is unchanged. The explicit connector configuration is a new file with a different
configuration hash. Historical runs and strict checkpoint/report identity checks
must retain their recorded configuration and source snapshots; equivalent
architecture is not permission to substitute bytes in a recorded run protocol.

The generator's diffusion/flow objective, training opt-in, and VAE behavior are
unchanged. The VAE encodes training images into target latents and decodes sampled
latents into images; it is not part of the trunk-to-generator bridge. Whether the
generator is frozen or fine-tuned remains an explicit training choice, separate
from connector selection.

CPU compatibility checks cover this API refactor. They do not establish new
accelerator execution or image-quality results; the completed DOCCI experiment
reports describe the previously executed source snapshots.

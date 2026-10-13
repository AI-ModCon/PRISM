# Simple forecasting head for Qwen3-0.6B

The time-series decoder uses the same **readout → bridge → generator** API as
the OmniGen2 image decoder. Its readout pools final trunk states, its bridge is
an identity, and its generator is a linear quantile head. The head is randomly
initialized and must be trained on observed-history / future-target pairs.

The complete model preset is
[`prism_qwen3_0_6b_intern_s2_397b_ts_forecast.yaml`](../../src/conf/model/prism_qwen3_0_6b_intern_s2_397b_ts_forecast.yaml).
It records the Qwen backbone, referenced Intern-S2 input encoder, input
projector, prefix fusion, freeze settings, and forecasting output. The reusable
[`time_series_direct.yaml`](../../src/conf/decoder/time_series_direct.yaml) overlay
adds just the output configuration to an existing input model.

```text
Observed series → existing Intern-S2 encoder → existing PRISM input projector
                                                   ↓
Prompt tokens ───────────────────────────────→ Qwen3-0.6B trunk
                                                   ├── language head
                                                   └── last valid final state [B,1,1024]
                                                         → identity bridge [B,1,1024]
                                                         → Linear(1024, 96×1×3) generator
                                                         → quantiles [B,96,1,3]
                                                         → median [B,96,1]
```

The readout selects the **last valid token** in the fused sequence, respecting
the attention mask. With prefix fusion and a prompt following the observed
series, that state can attend to both history and prompt. The linear head
predicts all future steps in one forward pass. The identity bridge has no
parameters and preserves the trunk width. The diagram and reusable overlay
below use the `96`-step, univariate defaults; this head has **295,200
parameters** at that setting. The complete model preset overrides these to a
`720`-step, three-variate forecast and has **6,642,000 head parameters**.
Its `ts_variates: 10` setting is for the input encoder and does not determine
the output variate count.
The official
[Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
specifies the 1,024-wide trunk; model assembly reads the actual loaded width.

## Configuration

Both configurations use this decoder block; the complete model preset also
asserts `bridge.output_dim: 1024`. The overlay infers the width from the loaded
trunk so it can be reused with another backbone:

```yaml
output_decoders: [text, time_series]
decoder_loss_weights:
  time_series: 1.0
decoder_configs:
  time_series:
    readout:
      type: pool
      layers: final
      pool: last
    bridge:
      type: identity
    generator:
      type: linear_quantile
      horizon: 96
      num_vars: 1
      quantiles: [0.1, 0.5, 0.9]
```

Inspect the complete preset without loading models or starting training:

```bash
python -m src.train \
  model=prism_qwen3_0_6b_intern_s2_397b_ts_forecast \
  --cfg job
```

Alternatively, add `+decoder=time_series_direct` after selecting an existing
model. The overlay preserves that model's backbone, encoders, projectors and
freeze settings. Override
`model.decoder_configs.time_series.generator.horizon` or
`model.decoder_configs.time_series.generator.num_vars` to match the forecast
targets. Targets may be shorter than the configured horizon or variate count;
unprovided cells are ignored by the loss. These overrides change the linear
head's shape, so an old head checkpoint cannot be reused unchanged. Set
`model.decoder_configs.time_series.readout.pool=mean` for masked mean pooling.
This preset selects text and time-series outputs; explicitly include any other
already configured output heads when combining configurations.

**Input-model requirements:** the full preset wires the `intern_s2_397b`
observed-history encoder through PRISM's time-series projector and prefix
fusion into the Qwen trunk; the forecasting head reads those resulting trunk
states. With the preset defaults, `transformers>=5.2.0` and the extracted
Intern-S2 encoder checkpoint at
`${HF_HOME}/intern-s2-preview-397b-timeseries/model.safetensors` are required.
This is the Intern-S2 **input encoder**, not its native forecasting head: the
output remains the newly initialized PRISM linear quantile head described here
and must be trained on forecast targets.

For a new model, merge the same overlay into a `ModelConfig` dictionary before
construction. For an **already restored trained PRISM model**, attach the new
head afterwards instead. This keeps restoration of the existing weights strict;
the image-specific parent loader does not allow missing time-series head keys.
The following uses the already restored `model`:

```python
from omegaconf import OmegaConf
from src.decoders import TimeSeriesDecoder

overlay = OmegaConf.to_container(
    OmegaConf.load("src/conf/decoder/time_series_direct.yaml"), resolve=True
)
if "time_series" in model.decoders:
    raise ValueError("A time-series head already exists; do not overwrite its weights")
head_config = overlay["decoder_configs"]["time_series"]
head = TimeSeriesDecoder(model.backbone.config.hidden_size, **head_config)
model.decoders["time_series"] = head.to(
    device=model.backbone.get_input_embeddings().weight.device
)  # Keep this new head in FP32; its forward handles the trunk dtype.
model.config.output_decoders = [*model.config.output_decoders, "time_series"]
model.config.decoder_configs["time_series"] = head_config
model.config.decoder_loss_weights.update(overlay["decoder_loss_weights"])
```

Save the updated model configuration together with the new head's trained state.
Do not load the original parent with blanket `strict=False` to hide missing or
unexpected encoder/backbone weights.

`TimeSeriesDecoder.prepare_condition(condition)` returns a `DecoderContext`
with tokens `[B,1,D]` and a valid `[B,1]` mask. Last pooling retains the selected
position in the original trunk sequence; mean pooling has no single source
position. Source modality spans continue to describe that original sequence.
`conditioning_contract()` reports the selected readout, bridge and generator.
The decoder keeps its learned weights under `head.weight` and `head.bias`, and
legacy flat `horizon`, `num_vars`, `quantiles` and `pool` configurations remain
supported. The readout and identity bridge add no checkpoint tensors.

## Training and prediction API

Use PRISM's structured output API with only the observed past and prompt in
`inputs`. Future values are passed separately:

```python
inputs = {"text": prompt_ids, "time_series": observed_history}
inputs["time_series_history_mean"] = history_mean    # [B,V], detached FP32
inputs["time_series_history_scale"] = history_scale  # [B,V], positive FP32
result = model.forward_outputs(
    inputs,
    targets={"time_series": future_values},  # [B,H,V], up to [B,96,1] for this overlay
    requested_outputs=["time_series"],
)
result.loss.backward()  # mean pinball loss at quantiles 0.1, 0.5 and 0.9
quantiles = result.predictions["time_series"]  # [B,96,1,3], original units

model.eval()
forecast = model.predict(inputs, requested_outputs=["time_series"])
median = forecast.predictions["time_series"]  # [B,96,1]
```

For the complete model preset, the corresponding prediction shapes are
`[B,720,3,3]` for quantiles and `[B,720,3]` for the median. An optional
boolean `inputs["time_series_target_mask"]` with the same shape as the target
marks valid target cells; non-finite target values are also ignored. This lets
batches contain variable-length targets or samples without targets, provided
the data pipeline supplies the mask. Without an explicit mask, finite target
cells are treated as valid.

Train the new head on forecasting targets before assessing quality. For a
head-only first stage, freeze the restored input projectors as well as the
backbone and encoders, then build the optimizer from trainable parameters.
`freeze_encoders` alone does not freeze the input projectors. Supplying a head
configuration does not create future targets in a caption/classification dataset
or define a forecasting data loader. Supply raw history and targets in the same
units; the normalization path below handles target scaling. The simple head
does not provide quantile-crossing correction.

## History-based normalization

For WebDataset samples with `time_series_target`, the modality-aware loader
computes a separate mean and scale for each sample and forecast channel from
its observed history, before batch padding. `TimeSeriesEncoder.forecast_statistics`
performs the calculation in FP32 without gradients. Non-finite values and cells
marked unobserved are excluded. Standard deviation uses population variance
(`correction=0`); constant, single-observation, or entirely missing channels use
scale 1 when the standard deviation is at most `1e-6`. An entirely missing
channel also uses mean 0.

The loader derives the value-channel count `V` from the target shape `(H,V)`
(or uses one channel for `(H,)`). History must contain either `V` value channels
or `2*V` channels arranged as values followed by binary observation masks.
Only the value channels contribute statistics, using the corresponding masks.
This assumes history and target channels have the same order; other mappings
require an explicit data adapter. The history tensor passed to Intern-S2 is
unchanged, and its internal per-chunk normalization remains independent.

The loader emits detached `time_series_history_mean` and
`time_series_history_scale` vectors. The collator batches these as `(B,V)`,
padding absent channels with mean 0 and scale 1. The model passes them explicitly
to the time-series decoder as `history_mean` and `history_scale`, in both legacy
auxiliary-loss and structured output paths. There is no mutable statistics cache.
Target padding and missing cells remain excluded by `time_series_target_mask`.

The head predicts standardized quantiles. Targets are transformed with history
statistics only, never future-target statistics:

```text
standardized_target = (raw_target - history_mean) / history_scale
loss = masked_pinball(standardized_quantiles, standardized_target)
reported_quantiles = standardized_quantiles * history_scale + history_mean
```

Normalized loss arithmetic is FP32. Decoder `forward` / `forward_condition`
return quantiles in original units, and `generate` / `generate_condition`
return the inverse-transformed median. The low-level `decoder.predict` method
returns head outputs without applying statistics. Calls that omit both
statistics tensors retain the legacy unscaled behavior; supplying only one,
invalid shapes, non-finite statistics, or non-positive scales raises an error.
Existing heads trained to predict raw values must be retrained for the
standardized-output contract.

Statistics discovery for target-free loading, encoder-bypass inference, and
cached-embedding inference remains future work. Until then, callers must supply
the same history-derived statistics explicitly for normalized inference, as in
the API example above. The WebDataset loader does not compute them for samples
without numeric targets, and no additional shard metadata is required.

The native Intern-S2 forecaster is deferred. It additionally requires native
preprojection encoder features, raw history, two pretrained Q-Formers and a
4,096-wide language-state interface. This initial configuration deliberately
uses the smaller PRISM head selected for the first implementation.

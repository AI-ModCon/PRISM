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
parameters and preserves the trunk width. The new head has **295,200
parameters** at the default 96-step univariate / three-quantile setting.
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
targets. These change the linear head's shape, so an old head checkpoint cannot
be reused unchanged. Set
`model.decoder_configs.time_series.readout.pool=mean` for masked mean pooling.
This preset selects text and time-series outputs; explicitly include any other
already configured output heads when combining configurations.

**Input-model prerequisite:** the current decoder branch does not contain the
[`intern_s2_397b` encoder implementation and loading wiring from the referenced
input-model commit](https://github.com/AI-ModCon/BaseMM_PRISM/blob/9a2f328016fb03edcb603460b39a33857704bdd8/src/conf/model/prism_qwen3_0_6b_intern_s2_397b_ts.yaml).
The full preset preserves those input fields but does not port their runtime.
Its Hydra composition is supported; executing that particular full model still
requires integrating the input encoder and restoring the trained parent
weights. `HF_HOME` must resolve to the prepared local encoder assets. The simple
output head itself is implemented here and works with Qwen trunk states.

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
result = model.forward_outputs(
    inputs,
    targets={"time_series": future_values},  # [B,96,1]
    requested_outputs=["time_series"],
)
result.loss.backward()  # mean pinball loss at quantiles 0.1, 0.5 and 0.9
quantiles = result.predictions["time_series"]  # [B,96,1,3]

model.eval()
forecast = model.predict(inputs, requested_outputs=["time_series"])
median = forecast.predictions["time_series"]  # [B,96,1]
```

Train the new head on forecasting targets before assessing quality. For a
head-only first stage, freeze the restored input projectors as well as the
backbone and encoders, then build the optimizer from trainable parameters.
`freeze_encoders` alone does not freeze the input projectors. Supplying a head
configuration does not create future targets in a caption/classification dataset
or define a forecasting data loader. Keep context and targets in the same units;
if scaling is used, fit it on observed history only and invert it for reported
forecasts. The simple head provides no automatic scaling or quantile-crossing
correction.

The native Intern-S2 forecaster is deferred. It additionally requires native
preprojection encoder features, raw history, two pretrained Q-Formers and a
4,096-wide language-state interface. This initial configuration deliberately
uses the smaller PRISM head selected for the first implementation.

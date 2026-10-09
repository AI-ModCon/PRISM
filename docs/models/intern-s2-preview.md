
# Intern-S2 Preview Time-Series: 35B Baseline and 397B Additions

This note compares the [Intern-S2 Preview 35B reference
implementation](https://huggingface.co/internlm/Intern-S2-Preview/blob/main/modeling_interns2_preview.py)
with [Intern-S2 Preview 397B](https://huggingface.co/internlm/Intern-S2-Preview-397B/blob/main/modeling_interns2_preview.py). It describes the code that is present in those files, rather than undocumented training behavior.

## At a Glance

> **Hugging Face access.** Some models referenced from this encoder's vendored
> upstream code are **gated on Hugging Face** and return `HTTP 401` to an
> anonymous request — `Qwen/Qwen3.5-35B-A3B-Instruct`, cited in
> `src/encoders/intern_s2_preview/configuration_interns2_preview.py`, is one.
> That is not a dead link: you must be signed in to a Hugging Face account
> that has been granted access to the repository, and then authenticate
> locally (`huggingface-cli login`, or `HF_TOKEN` in the environment) before
> the weights will download. PRISM does not redistribute them.
>
> The citing file is vendored upstream code and is kept byte-identical, so
> this note lives here rather than in it.


The 35B model has a time-series **encoder-to-LLM** path: it converts a numeric
signal to embeddings and inserts them at `<TS_CONTEXT>` placeholder locations.
It does not define a forecasting model, forecast flag, forecast outputs, or a
numeric autoregressive decoder.

The 397B implementation preserves that integration role but substantially
changes the input subsampling path and adds a separate, conditioned
`InternS2PreviewTimeSeriesForecaster`. This additional model produces numeric
point and quantile forecasts from raw history. It is a patch-autoregressive
decoder, not a text decoder, and it does not use `<think>...</think>` blocks.

| Area | 35B `Intern-S2-Preview` | 397B `Intern-S2-Preview-397B` |
|---|---|---|
| TS-to-LLM encoder | Yes | Yes, with a new subsampling frontend |
| Pre-projection TS features retained | No | Yes, as `ts_encoder_embedding` |
| Raw history retained after LLM pass | No | Yes, as `ts_history` |
| Separate numerical forecaster | No | Yes, Forecaster 2.5 backbone |
| LLM/TS latent alignment | No | Yes, dual Q-former `Aligner` |
| Learned horizon prediction | No | Yes, optional LLM-state horizon head |
| Numeric cached autoregression | No | Yes, per-layer `DecodeCache` |
| `<think>` protocol inside forecasting | No | No |

## PRISM Extraction and Loading

Run the selective extractor from the repository root. It reads `HF_TOKEN` from
`.env`, consults the safetensors index, and downloads only shards containing
time-series tensors:

```bash
module load frameworks/2025.3.1
python applications/timeseries/extract_intern_s2_timeseries.py
```

For the 35B checkpoint this downloads only `model-00023-of-00023.safetensors`
(about 273 MB), extracts `model.time_series.*`, and strips that prefix. The
vendored implementation and standalone encoder config are written to
`src/encoders/intern_s2_preview/`. The extracted `model.safetensors` and
manifest are written outside the repository under
`$HF_HOME/intern-s2-preview-timeseries/`.

The manifest records that no forecaster is available. On a compatible future
repository, the same script also extracts `time_series_forecaster.*` beneath
the same `$HF_HOME` artifact directory; pass `--encoder-only` to disable that
behavior. Use `--output-dir` only when an explicit non-default checkpoint
location is required.

PRISM loads the extracted encoder with `ts_projector: intern_s2`. The provided
`model=prism_olmo1b_intern_s2_ts` preset uses a 512-step input, sampling rate
1, 64 encoder tokens, and the encoder's 2048-dimensional projected output.
The published custom code declares `transformers==5.2.0`; Aurora's shared
frameworks environment currently provides 4.57.6, so use a Transformers 5.2
environment when constructing this encoder.

## 35B Baseline: Encoder-to-LLM Path

The 35B `InternS2PreviewTimeSeriesModel` accepts batched `(B, T, C)` signals,
valid lengths, sampling rates, and channel counts. The model performs adaptive
subsampling, a Transformer encoder, then a projector to the language-model
hidden size:

```text
raw multichannel signal
  -> adaptive patches selected from sampling rate
  -> per-channel convolution + patch Transformer
  -> average channels + concatenate adjacent patch features
  -> Whisper-like TS Transformer
  -> LLM-width projector
  -> replace <TS_CONTEXT> embedding placeholders
```

`InternS2PreviewTimeSeriesMultiChannelAdaptiveSubsampling` derives a stride
from sampling rate, uses a patch size of twice that stride, and processes each
channel independently through `Conv1d`, fixed sinusoidal positions, and a
one-layer Transformer encoder. It averages channel features, then
`InternS2PreviewTimeSeriesConcatSubsampling` concatenates adjacent token
features to halve the token count.

`InternS2PreviewTimeSeriesEncoder` is the common Whisper-like stage: it adapts
the frontend output to 80 channels, applies two `Conv1d` layers (the second has
stride 2), adds learned absolute positions, and runs pre-norm self-attention and
FFN layers. Its mask is causal by default or chunk-local when configured.
`InternS2PreviewTimeSeriesProjector` maps encoder outputs through
`LayerNorm -> Linear -> activation -> Linear` into LLM width.

The 35B multimodal forward path keeps only the projected valid TS tokens. It
checks that their count matches the number of `<TS_CONTEXT>` placeholders and
replaces those placeholder embeddings before executing the causal LLM. Its
output object and conditional-generation API contain language-model outputs
only; there is no forecast invocation after the LLM forward pass.

## 397B: Encoder Changes

397B replaces the 35B adaptive-subsampling frontend with
`TotalFixlenSingleResChunkQformerSubsampling` and adds chunk orchestration in
`InternS2PreviewTimeSeriesModel`.

| Encoder concern | 35B baseline | 397B change |
|---|---|---|
| Input handling | One batched tensor | Batched tensor or variable-length list; each sample is chunked by `chunk_size` / `chunk_step` |
| Patch scale | Sampling-rate-derived stride and $2 \times$ stride patch | `subrate` derived from sequence length; configured patch/query resolution |
| Local frontend | Conv, positional encoding, per-patch Transformer | Per-channel normalization, Conv stack, appended mean/std features |
| Patch compressor | Mean-pool patch Transformer output | One-layer patch Transformer or learned-query `MRQFormer` |
| Channel fusion | Mean channels before concatenative subsampling | Channel Transformer followed by mean pooling and `fuze_proj` |
| TS data retained | Projected TS tokens only | Raw `ts_history` and pre-projection `ts_encoder_embedding` retained for forecasting |

The 397B frontend normalizes each channel and appends signed-log mean and
log-standard-deviation features. It dynamically forms patches, encodes them
through either a one-layer transformer or `MRQFormer` learned-query
cross-attention, and uses a second Transformer across channels. Chunk outputs
are concatenated after padded regions are removed. The downstream
`InternS2PreviewTimeSeriesEncoder` and projector remain structurally similar to
the 35B path.

For normal multimodal LLM input, 397B still projects valid time-series tokens
into `<TS_CONTEXT>` positions. The important new side channel is that it also
returns the raw series (`ts_history`) and the encoder states before this LLM
projector (`ts_encoder_embedding`), allowing a separate forecaster to consume
numeric and latent information together.

## 397B: New Forecasting Path

The new `InternS2PreviewTimeSeriesForecaster` runs when conditional generation
is called with `ts_forecast=True` (or via `generate_with_time_series_input` with
forecasting enabled). It receives:

- `history`: raw `(T_i, C_i)` values retained from the multimodal input;
- final hidden states from the causal LLM;
- pre-projection time-series encoder states and their validity masks.

`Aligner` runs independent Q-formers over LLM and time-series streams. Each
compresses its variable-length inputs into a fixed number of learned query
tokens using query self-attention, source cross-attention, and an FFN. The
time-series query tokens followed by LLM query tokens are static cross-attention
K/V context for the numerical forecast backbone.

The backbone forecasts one series per channel. The wrapper splits each
`(T_i, C_i)` history into $C_i$ univariate sequences, duplicates the sample's
aligned context for each sequence, and restores point forecasts
`(horizon_i, C_i)` plus quantile forecasts `(horizon_i, C_i, 10)` afterward.

An optional learned horizon head also appears only at 397B. It transforms final
LLM hidden states with a projected pre-norm one-layer `TransformerEncoder`,
masked mean pooling, and `LayerNorm -> Linear -> SiLU -> Linear(1)`. Its
linear-step prediction is rounded and clamped; `forecast_horizon` overrides it.

## 397B Forecasting Decoder

`ForecasterBackbone` is a fixed Forecaster 2.5 configuration:

| Component | Value |
|---|---:|
| Context limit | 16,384 values |
| Input patch size `p` | 32 values |
| Output block `o` | 128 values |
| Feedback patches per step `m = o / p` | 4 |
| Transformer layers | 20 |
| Model width / heads | 1,280 / 16 |
| Output channels | median plus 9 quantile-related channels |

Inputs are split into 32-value patches, concatenated with masks, and tokenized
by a residual MLP. Each of 20 layers applies causal self-attention, optional
cross-attention into the static aligned context, and an FFN. The cross-attention
contribution is multiplied by `tanh(cross_attn_gate)`. Since this gate is
initialized to zero, it initially contributes nothing to the numeric backbone.

Two residual MLP output heads produce a point forecast and a quantile-spread
representation. The point head has 1,280 outputs, which reshape to 128 values
across 10 channels. The spread head has 10,240 outputs and supports the 1,024-
value continuous-quantile span.

### Cached autoregressive rollout

Self-attention is causal over valid patches. Every layer owns a `DecodeCache`
containing preallocated K/V tensors, a write index, a leading-padding count, and
a KV validity mask. The prefill writes the context patches; each later decoding
call writes only newly generated input patches and attends to the cached history.
RoPE includes the cache offset and left-padding adjustment. Q/K receive RMS
normalization and learned per-dimension scaling before unscaled dot-product
attention.

For a requested horizon $H$, decoding proceeds as follows:

1. Left-pad or truncate context to a 32-value multiple and build its mask.
2. Compute running patch statistics, normalize with RevIN, and prefill the
   cached transformer using the cross-attention context.
3. De-normalize the last prefill token's 128-value output; it supplies the first
   forecast block.
4. For every additional block, take the previous block's median channel, reshape
   it into four 32-value patches, update statistics, normalize, and make one
   cached transformer call.
5. Concatenate blocks, slice to $H$, and apply optional quantile reconstruction,
   flip invariance, crossing repair, de-normalization, and non-negativity.

The feedback index is `aridx = 5`, the median quantile in the implementation's
`[median, 0.1, ..., 0.9]` ordering. It is deterministic feedback, not sampling;
uncertainty is carried in the returned quantile channels.

## Does It Use `<think>` Blocks?

Neither implementation uses `<think>`/`</think>` as part of time-series
processing. The 35B model has no numerical forecasting path at all: it encodes
the signal into the LLM input and otherwise follows ordinary causal-LM behavior.

397B adds a numerical forecast path, but it neither creates nor parses literal
`<think>`/`</think>` tokens. When `ts_forecast=True`, the multimodal LLM first
produces hidden states for the supplied prompt. Those hidden states, plus the
pre-projection time-series states and raw numeric history, are passed directly
to `Aligner` and the numeric forecaster.

Either language model may generate reasoning-style text in ordinary chat
generation, depending on its template and prompt. That text behavior is
independent of 397B forecasting: the forecast decoder does not wait for
generated reasoning, use generated text as feedback, or implement a hidden
`<think>` protocol. Its multi-step computation is explicit autoregression over
predicted 32-value input patches and 128-value output blocks.

## Implication for PRISM

PRISM's time-series encoder most closely matches the first role: producing LLM
input tokens. Reproducing Intern-S2 Preview's forecasting behavior requires a
separate numeric patch forecaster, latent cross-attention alignment, RevIN and
running-statistics handling, cached causal rollout, and quantile post-processing.
A forecast prompt or LLM `<think>` text alone cannot reproduce that decoder.
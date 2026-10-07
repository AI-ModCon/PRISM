# Trained PRISM parent: controlled OmniGen2 connector overfit

Date: 2026-09-21. The controlled experiment completed 500 connector-only updates
on Aurora. Numerical repeatability and frozen-weight audits passed; sampled
training images show partial content recovery, but the two sampled held-out
prompts fail. This does not certify P0/P1/P2 or scientific output capability.

## Architecture and fixed protocol

The complete trained PRISM-Harness Qwen3-8B + SigLIP2-base parent is restored from
step 20,000, including its existing input projector. Parent provenance and its
614 checkpoint tensors are recorded in the
[parent integration report](2026-09-21-prism-harness-image-parent.md).
The new output connector is LayerNorm(4096) + Linear(4096,2048), containing
8,398,848 trainable scalar parameters. PRISM, the native OmniGen2 conditioner,
diffusion transformer, and VAE remain frozen.

The authorized experiment targets 500 total AdamW updates, batch size 1,
learning rate 1e-4, weight decay 0, gradient clipping at 1, seed 42, and 256×256
images. Connector weights/updates are FP32; frozen models use BF16 with strict
`torch.use_deterministic_algorithms(True)` and math SDPA. It begins with a
20-update timing pilot, then resumes its optimizer state if numerical checks,
finite gradients, frozen-weight audits and baseline sampling succeed.

Every 100 updates and at the final step:

- Save an explicitly unqualified connector/optimizer/RNG checkpoint before sampling.
- Measure fixed-seed flow losses on all 16 training and all 8 validation examples.
- Repeat each probe with a different same-split caption, retaining the exact target
  and resetting all RNG state, including VAE posterior/noise/timestep sampling.
- Generate two training and two validation examples with 50 denoising steps.
  Native OmniGen2 and untrained-connector baselines use the same prompts and
  initial latent tensors; trained samples replay that bank exactly. Guidance is
  text 5, image 2, negative prompt empty. Generation does not open target pixels.

The new runner is [overfit_prism_image_connector.py](../../tools/overfit_prism_image_connector.py).
It leaves the production acceptance gates and four-update smoke limits intact.
Resume binds source hashes, parent/generator identity, dataset fingerprints,
optimizer configuration and numerical policy. Saved artifacts cannot be loaded
as qualified production connectors.

## Data and limits

[The source protocol](../../examples/image_decoder/overfit_protocol.json) binds
16 original-caption PixMo training pairs and 8 pairs from the existing Aurora
validation shard. Six examples were excluded before training for clear caption
errors, a blank image, ambiguous content, or unsuitable text-heavy supervision.
The original captions were not rewritten or truncated. Selected image and caption
bytes, member offsets, source metadata, URL/video groups and decoded-image hashes
are archived. Cross-split exact and detected perceptual duplicates are excluded.
All optimization records are text-to-image: source/reference images are absent.
The trained vision encoder/projector are loaded and exercised in the separate
parent diagnostic, but this learning run does not train or validate image editing
or image-input-conditioned generation.

Combined manifest SHA256:
`1a0b6e2cdc1545f901ffae604f4334c9d15198549a7ece161ba42a230905fcfb`.
Train manifest:
`e60043f3077d5c0201973959815855368e83b0107f42a4e069a3c6b060d0e5c6`.
Validation manifest:
`546a8c522340b76a1f27858bad6113518526adbc0346cd615e42be7a4ce720a3`.

This holdout applies only to connector training; parent/generator pretraining
membership is unknown. These are diverse diagnostics, not a benchmark. Fine
caption attributes remain noisy; two validation examples contain text/screenshots.
Two training PNGs contain transparency that existing RGB conversion drops to a
black background; one caption describes a white background. The original
captions, images and these limitations remain visible in the archived visual QA.

Runtime data: `/lus/flare/projects/<project>/<user>/prism-connector-overfit-20260921/data/connector-overfit16-reviewed-v2`.
Local data/QA: `outputs/image_decoder_validation/20260921-connector-overfit/data-reviewed`.

## Completed numerical diagnostics

All experiments use one allocated Aurora node, one process and one visible XPU
tile, with pinned frameworks/2025.3.1 and the previously verified OmniGen2 runtime.
The isolated experiment root is
`/lus/flare/projects/<project>/<user>/prism-connector-overfit-20260921`.

| PBS job | Experiment | Observed outcome |
|---|---|---|
| 8846248 | Native N0/N1, adapter A0, native N2 under deterministic BF16/math | Completed, exit 0, 2m14s scheduler walltime |
| 8846266 | Eight-image parent evaluation with same-input/padding controls | Completed, exit 0, 2m55s scheduler walltime |
| 8846308 | Fresh-process numerical comparison, then 20-update connector pilot | Completed, exit 0, 8m05s scheduler walltime; all frozen hashes unchanged |
| 8846391 | Resume optimizer state from step 20 to step 500 | Completed, exit 0, 11m21s scheduler walltime within a 20-minute allocation budget |

Job 8846248 matched **all required captured boundaries exactly** between native
repeats and the adapter for one fixed case/latent. It differs from the archived
run using the previous execution policy; that cross-policy comparison is
explicitly exploratory. No tolerances changed. The combined deterministic/math
policy establishes repeatability for these measurements; it does not isolate
which individual kernel caused the earlier variation. Native runner time was
95.13s including 64.54s loading and 24.37s checkpoint hashing.
The first phase of job 8846308 repeated the experiment on a different node
(`x4407c1s0b0n0` versus `x4518c0s7b0n0`). All within-process comparisons and the
comparison with the first process matched exactly. The worker's numerical guard
passed before starting the connector pilot. The downloaded summary is
`outputs/image_decoder_validation/20260921-connector-overfit/numerical-bf16-math-02-manifest.json`.

Job 8846266 restored all parent weights and retained identical before/after
frozen-state hashes. Across all eight cases, the two repeated forwards, text
padding, and masked-source padding matched exactly, including compiled embeddings,
attention masks, position IDs and all valid final hidden states. Old/new text
routes matched token-for-token in all eight cases. The image generator was not
loaded by this parent-only diagnostic.

Qualitative review recognized each example's main content: cartoon dog, clock,
bow/arrow, cat, shoes, white dog, mathematical text and map screenshot. Counts and
OCR were imperfect: the shoe description overcounted pairs and the mathematical
text was transcribed incorrectly. Responses were capped at 32 generated tokens.
These examples/prompts differ from the previous three-case smoke; this is not a
before/after capability improvement or a held-out benchmark score.

Local numerical traces:
`outputs/image_decoder_validation/20260921-connector-overfit/numerical-bf16-math-01`.
Local parent summary:
`outputs/image_decoder_validation/20260921-connector-overfit/parent-manifest.json`.
Full parent traces remain under `runs/parent-validation-math-01` on Aurora.

The pilot's fresh-process check must match within-process native/adapter boundaries
and the prior process exactly before training starts. A failed check stops the
worker. This bounded experiment check is separate from the formal P0/P1 suite.

## Objective audit

A read-only comparison against the clean, pinned OmniGen2 checkout at
`18e6f9d5271b517fcb32e999f10df943ae9b8f20` found no discrepancy in the audited
flow objective: noisy latents are `t * clean + (1 - t) * noise`, the velocity
target is `clean - noise`, VAE posterior samples use the checkpoint's shift and
scale, and loss is float32 mean squared error. The upstream transformer applies
its own timestep scale. Both routes select final conditioner hidden states.
The apparently reversed upstream time-shift formula includes two `1 - t`
transformations; it equals the PRISM expression. Comparing 49,995 float64
time/resolution combinations gave maximum absolute error 2.22e-16.

The upstream evidence is `omnigen2/transport/path.py:25–31,116–144`,
`omnigen2/transport/transport.py:127–172,221–233`,
`omnigen2/models/transformers/block_lumina2.py:193–215`, and
`train.py:323–334,515–529,557–566`. This audit checks these equations, not
end-to-end training equivalence.

The conditioning distributions differ substantially: native OmniGen2 uses
Qwen2.5-VL final states with a system/user chat template; this experiment uses
the trained PRISM Qwen3 states from raw captions plus LayerNorm/linear projection.
Matching a 2048-wide interface does not establish equivalent semantic features.
The diagnostic also uses 256-square Bicubic target resizing rather than upstream
aspect-ratio-aware Lanczos preprocessing. Classifier-free guidance combines the
new positive connector features with native empty-prompt negative conditioning;
guidance can amplify misaligned positive features. These are experimental
differences to investigate if loss improves without prompt-responsive images.

## Learning results

The pilot completed 20 optimizer updates in 343.21 seconds of runner time,
including loading, hashing, fixed probes and 12 generated samples. Mean optimizer
step time was 0.367 seconds (maximum 0.638). Peak allocated memory was 32.26 GiB.
All 2,305 frozen parameter/buffer hashes matched before and after. Every connector
parameter tensor received finite nonzero gradients; no frozen weights were optimized.

| Step | Train flow loss | Validation flow loss | Train shuffled-minus-correct | Validation shuffled-minus-correct |
|---:|---:|---:|---:|---:|
| 0 | 0.663836 | 0.654472 | +0.059975 | +0.021598 |
| 20 | 0.419422 | 0.417489 | +0.061936 | −0.050936 |
| 100 | 0.402454 | 0.411964 | +0.070529 | −0.054723 |
| 200 | 0.397743 | 0.405444 | +0.073569 | −0.047766 |
| 300 | 0.396469 | 0.419348 | +0.085576 | −0.051622 |
| 400 | 0.394182 | 0.411629 | +0.076377 | −0.057200 |
| 500 | 0.389826 | 0.416688 | +0.087315 | −0.050493 |

At step 500, fixed training loss is 41.28% below initialization and validation
loss is 36.33% lower. Most of the reduction occurs in the first 20 updates.
The negative validation caption gap persists throughout training: at step 500,
the wrong-caption mean loss is 0.366196 versus 0.416688 for the correct captions.
These are matched-noise, single-seed-per-example diagnostics, not confidence
intervals or a benchmark. They do not establish held-out semantic conditioning.

Visual inspection confirms this limitation: native sampling produces a recognizable
lily, refrigerator comparison and clock; the Snoopy prompt produces a generic
white toy. Untrained connector samples are predominantly colored textures.
At step 20 the connector produces lines/text, an almost blank refrigerator image,
and architectural stripes for the clock. These are not useful prompt-faithful outputs.
All 12 downloaded sample files matched their recorded SHA256 values.

The continuation resumed the actual step-20 connector and AdamW state, using the
same code, data, numerical policy and noise bank. It added 480 updates for a
**500-update total**, not 500 additional updates. Runner time was 663.62 seconds;
the 480 updates averaged 0.323 seconds each, with remaining time spent on loading,
hashing, probes and sampling. Peak allocated memory was 32.27 GiB. It ran on
`x4206c5s4b0n0`, with one process and one visible XPU tile on one allocated node.
All 2,305 frozen parameter/buffer hashes again matched exactly. All four connector
parameter tensors had finite, nonzero gradients in every recorded resumed update.

At step 500, the training lily prompt produces a stylized flower-like painting,
although its colors and form differ from the target. The training refrigerator
prompt recovers a paired vertical appliance-like composition. This is partial
training-set learning. The held-out Snoopy prompt produces paired red blocks,
and the held-out clock prompt produces a pale bread/tile-like square; neither
matches the requested subject. Four fixed examples do not measure broad quality,
but these two validation failures agree with the caption-control concern.

All 20 continuation sample files and the final checkpoint were downloaded and
verified against the report's SHA256 values. CPU loading with `weights_only=True`
confirmed step 500 and the expected four connector tensors: LayerNorm weight/bias
`[4096]`, linear weight `[2048,4096]`, and linear bias `[2048]`. The checkpoint
contains optimizer/RNG state and provenance, and remains explicitly unqualified.

Local pilot results: `outputs/image_decoder_validation/20260921-connector-overfit/pilot`.
Gallery: `outputs/image_decoder_validation/20260921-connector-overfit/pilot-gallery.png`.
Checkpoint 20 SHA256:
`9d11999faa0032009db1161c8c70868f6880ac07c248bf2b30352c0b7ac3ed01`.

Local final results and checkpoint:
`outputs/image_decoder_validation/20260921-connector-overfit/overfit-500`.
Checkpoint 500 SHA256:
`0afa7539a45308a4ae1d1442d1ac7e3db75f87c02da10234ba7ff386164b69b8`.
Machine-readable verification: `outputs/image_decoder_validation/20260921-connector-overfit/verified-final-summary.json`.

[Initial generation results](../assets/image_generation/2026-09-21-initial/README.md)
are now committed as a bounded evidence bundle, including the
[matched image gallery](../assets/image_generation/2026-09-21-initial/matched-gallery.png),
[training/validation curves](../assets/image_generation/2026-09-21-initial/train-validation-loss.png),
[caption-sensitivity probes](../assets/image_generation/2026-09-21-initial/fixed-probe-curves.png),
all 32 overfit sample images, original reports, step logs and checksum provenance.
Full data images, model/optimizer checkpoints and numerical traces remain in the
external runtime archives.

## Decision and next bounded experiment

Do not scale this unchanged recipe or begin scientific decoder training based on
the loss reduction. The run demonstrates functional optimization and partial
training-set learning, but does not establish useful held-out image generation.
Representation alignment is a hypothesis to test, not a confirmed root cause.

The next proposed experiment, **not yet executed**, is:

1. Replay captured native positive/negative conditioning and masks through
   `generate_conditioned` for two prompts and two seeds, with identical initial
   latents and 50 denoising steps. This directly checks native-feature injection.
2. Keep the same frozen models, connector architecture and 16/8 data, but run
   200 updates matching the frozen generator's native-conditioned velocity:
   `MSE(generator(PRISM_context, x_t, t), stop_gradient(generator(native_context, x_t, t)))`.
   Both branches must reuse the same VAE posterior sample, noise and timestep.
   This tests a more direct conditioning-alignment target before adding capacity.
3. Compare native-velocity error and matched/shuffled-caption controls at updates
   0/50/100/200, plus fixed-noise samples at guidance 1 and 5. In the caption
   control, keep the teacher velocity conditioned on the correct caption and
   shuffle only the student's caption. Preserve this 500-step checkpoint and
   its images as a baseline.

Native Qwen2.5 chat tokens and raw PRISM Qwen3 caption tokens do not have one-to-one
correspondence, so naive token-position feature MSE is inappropriate. If the
distillation target cannot fit training examples, test a small attention-based
connector next. If training fits but held-out behavior fails, prioritize more
diverse alignment data. Generator adaptation and scientific modalities remain
later decisions with separate validation.

## Local verification

192 focused tests passed, including actual tiny-Qwen routing, strict checkpoint
restoration, frozen/gradient checks, resumed-versus-uninterrupted optimization,
target-free sampling, exact initial-noise replay, numerical guard failure, source
selection and split leakage tests. Changed Python files pass Ruff; `git diff
--check` is clean. Fixture tests establish engineering behavior; the PBS runs above
provide the real-checkpoint execution evidence. The selected generation evidence
is committed separately; full data and runtime archives remain external.

Before committing, the expanded offline CPU suite passed **238 tests** and Ruff
passed all 18 changed Python files. A data-pack collision check now rejects
duplicate selected sample IDs across shards before writing output; regression
cases cover repeated keys and distinct paths sharing a basename. The test run
reported one dependency deprecation warning from `torch_geometric.distributed`.
These checks do not add another model-training or image-quality result.

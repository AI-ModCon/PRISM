# DOCCI connector and diffusion fine-tuning

`tools/train_prism_image_diffusion.py` continues from a completed DOCCI connector
pilot and makes the **entire OmniGen2 diffusion transformer** trainable. It uses
the same caption-conditioned image flow-matching objective and prepared
WebDataset indexes as the [connector-only stage](docci_data.md).

The requested PRISM parent is the Qwen3-1.7B/SigLIP2 checkpoint at
`CODEX_QWEN3_1P7B_SIGLIP_CLEAN_GSHUFV1_INVSQRTLR_TRAIN25K_BS4_HSDP_FULLSHARD_16N_R1/checkpoints/step_19750`.
The connector warm-start is the completed 500-step DOCCI pilot. Starting this
stage resets optimizer state because the trainable scope has changed; it does
not count those earlier 500 connector-only updates as diffusion updates.

| Component | Fine-tuned here? | Numerical representation |
| --- | --- | --- |
| PRISM language backbone, language head, vision encoder, input projector | No | Frozen BF16 |
| PRISM-to-OmniGen2 connector, 4,200,448 parameters | Yes | FP32 |
| OmniGen2 diffusion transformer, 3,967,161,400 parameters | Yes | BF16 compute, FP32 CPU master weights |
| OmniGen2 VAE | No | Frozen BF16 |
| OmniGen2 native language/vision conditioner | No | Frozen BF16; native baseline only |

Captions enter PRISM, whose final hidden states pass through the connector to the
diffusion transformer's text-conditioning input. Images supply VAE latent targets
for the flow loss. Target images never enter PRISM's vision encoder. This stage
does not add a language-head caption-generation objective.

## Precision and resource policy

The original pinned diffusion checkpoint stores FP32 tensors. CPU master weights
are initialized from those exact tensors, and their BF16 casts must equal the
loaded runtime weights. This preserves small accumulated optimizer updates that
could otherwise disappear when repeatedly applied directly to BF16 parameters.
CPU AdamW uses FP32 master weights, gradients, and moments, with separate connector
and diffusion learning rates. Gradients are checked for finiteness and globally
clipped before an update; the updated masters are copied back to runtime weights.
Diffusion gradients originate in BF16 and are upcast for CPU optimization; this
does not claim equivalence to an FP32-gradient training run.
The default learning rates for this bounded stage are `1e-5` for the connector
and `1e-6` for the pretrained diffusion transformer. This is a custom 256-pixel
pilot, rather than a reproduction of the upstream learning-rate schedule,
optimizer hyperparameters, or image-resolution recipe.

The one-process Aurora launcher allocates one node and uses one XPU tile, with 32
CPU threads for the offloaded optimizer. Upstream nonreentrant gradient
checkpointing is enabled for the diffusion transformer's main blocks. This is
full dense fine-tuning; no low-rank adapter replaces the transformer updates.
The unused reference-image branches still belong to the trainable model, but
source-free caption-to-image batches need not provide them with nonzero gradients.
Run records distinguish active, zero-gradient, and missing-gradient parameters.

## Starting and continuing a run

Use `tools/launch_aurora_image_experiment.py` with a reviewed JSON argument list,
first `--dry-run`, then `--submit`. Model execution stays inside the PBS allocation.
The training command extends the connector pilot's parent, data, and numerical
arguments with:

```text
train_prism_image_diffusion.py
--connector-checkpoint /path/to/completed/connector-pilot-step-000500.pt
--expected-connector-step 500
--learning-rate 1e-5
--diffusion-learning-rate 1e-6
--checkpoint-every 0
```

A fresh real-checkpoint repeatability diagnostic must match the current backend
source, checkpoint, device, precision, and attention policy. A diagnostic made
before changing backend code cannot satisfy that source identity requirement.
For a short smoke, use two updates, small fixed probe cohorts, and
`--sample-count 0`. The terminal checkpoint is always saved; `--checkpoint-every 0`
avoids redundant full-model checkpoints between the start and finish.

Joint checkpoints include FP32 master weights for the connector and diffusion
transformer, optimizer state, RNG state, sampler position, and the exact
parent/data/source/runtime protocol. They are substantially larger than
connector-only artifacts. `--resume` requires an identical protocol and a new
output directory; `--steps` is the total number of joint optimizer updates.

Fixed train and validation probes retain their example IDs, noise seeds, and
wrong-caption controls across steps. Final full validation is reported separately
from the fixed cohort. Sampling uses captions and replayed initial noise, without
target pixels. A native OmniGen2 baseline is generated only before diffusion
updates, so it still represents the original pretrained generator.

```bash
python tools/plot_prism_image_connector.py \
  --run-dir /path/to/joint/run --output-dir /path/to/joint/run/plots
```

This produces `connector-diffusion-losses.png`, a PDF, and an auditable JSON series.
Loss improvement alone does not establish caption fidelity or scientific decoder
readiness. DOCCI development examples are held out from these updates; DOCCI was
an original OmniGen2 training source, so they are not guaranteed unseen by the
pretrained generator.

The [completed 100-update execution report](../../reports/2026-09-21-docci-joint-diffusion-pilot.md)
records successful dense optimization, saved checkpoints and images, small loss
changes, and remaining caption-fidelity failures.

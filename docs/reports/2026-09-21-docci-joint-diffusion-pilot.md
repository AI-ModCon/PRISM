# DOCCI: joint connector and full diffusion training

**Completed:** the 100-update Aurora pilot fine-tuned both the PRISM output connector and full OmniGen2 diffusion transformer. Both groups changed; all 1,634 audited frozen PRISM/VAE/native-conditioner parameters and buffers stayed unchanged. Job **8847769** finished with exit 0 in **16m41s**. Loss improved only slightly, and the four reviewed generations still miss their captions’ main objects.

## Starting point and training scope

This experiment continues from the [500-update DOCCI connector pilot](2026-09-21-docci-connector-pilot.md), whose loss improved but whose sampled caption fidelity remained poor. The trained PRISM Qwen3-1.7B + SigLIP2 parent is unchanged: checkpoint `step_19750`, including its vision encoder, input projector/alignment, language backbone, and text head. The connector starts from the prior trained checkpoint:

```text
/lus/flare/projects/ModCon/sandeep/prism-docci-qwen3-1p7b-20260921/runs/pilot-500-01/connector-pilot-step-000500.pt
```

Warm-start SHA256: `e84b3550304f4cebe56b394e1e1f65dfc333bd9c6ed8e9adbb240c6694c4387f`. A completed prior report, frozen-state audit, parent/generator identity, dataset fingerprint, expected step, and complete FP32 connector state are required. The previous connector optimizer is not reused.

```text
Caption → frozen PRISM → trainable 2048→2048 connector → trainable OmniGen2 DiT
Image   → frozen VAE → latent flow-matching target
```

The connector contains **4,200,448 parameters**; the diffusion transformer contains **3,967,161,400 parameters**. The complete dense transformer is trainable; this does not use LoRA. PRISM, the VAE, and OmniGen2's native conditioner remain frozen and in evaluation mode. The target image is not supplied to PRISM's vision encoder, and there is no captioning/text-head loss.

BF16 transformer computation uses **FP32 CPU master weights initialized from the original FP32 safetensors**, without a BF16 round trip. BF16 gradients are copied to FP32 CPU buffers for global clipping and AdamW updates, then updated masters are cast back to runtime weights. FP32 connector weights form a separate learning-rate group. Gradient checkpointing covers 32 main transformer blocks; optimizer-boundary gradient cleanup frees device memory before evaluation and sampling.

## Data and configured protocol

The experiment reuses the prepared DOCCI WebDataset: **9,647 training pairs** and **100 official `qual_dev` validation pairs**. The 5,000 test and 100 qualitative-test pairs remain unused. Original captions are preserved; targets are resized to 256 × 256. Data fingerprint: `f45daff5995f78f2409c3d9a3efc9839aae9e860f08b06e1a59781e180f9a7f0`.

The [data and parent report](2026-09-21-docci-connector-pilot.md) records conversion checks, strict restoration of all 526 parent tensors, and exact text/hidden-state checks. Related subjects can cross official DOCCI splits, and pretrained OmniGen2 may already have seen DOCCI. This validation split is held out from the current optimizer; unseen-data generalization is not established.

| Setting | Joint pilot |
| --- | --- |
| Optimizer | CPU FP32-master AdamW; betas 0.9/0.999; weight decay 0; global gradient clip 1 |
| Learning rates | Connector `1e-5`; diffusion transformer `1e-6` |
| Batch / accumulation | 1 / 1 |
| Precision / kernels | BF16 compute; deterministic algorithms; math SDPA |
| Resources | One Aurora node, one process, one visible XPU tile; 32 CPU threads |
| Budget | 100 optimizer updates, starting fresh from the 500-step connector checkpoint |
| Loss probes | Fixed 16 train / 16 validation examples at steps 0, 25, 50, 75, 100; final all-100 validation |
| Gallery | Two train and two validation captions, 50 denoising steps, fixed initial latent replay |
| Native control | Two validation captions before diffusion training only |

The pilot does **not** include the two smoke updates. Its 100 updates consumed 100 distinct pairs, approximately 1.04% of one shuffled training epoch. This is a conservative custom fine-tuning pilot, not an exact replication of OmniGen2's original training recipe.

Each loss probe resets the complete VAE-posterior/timestep/noise RNG state. A different same-split caption is evaluated against the same target and randomness. The wrong-minus-matched loss gap is positive when the matched caption helps. Fixed-cohort curves remain separate from the larger final validation result. Train probes follow the initial shuffled optimizer examples and record whether each has actually been optimized.

Sampling uses caption metadata only and never opens target pixels. Samples replay the same saved initial latents before/after adaptation. The native baseline uses the original pretrained diffusion weights; sampling the native conditioner after DiT training would be an adapted control, not the original baseline.

## Completed smoke and numerical evidence

| Stage | PBS job | Observed result |
| --- | --- | --- |
| Fresh numerical repeatability | 8847724 | Exit 0; all three within-process BF16 comparisons exact |
| Two-update dense-training smoke | 8847735 | Terminal F, exit 0; scheduler walltime 6m44s |
| Separate 100-update pilot | 8847769 | Terminal F, exit 0; scheduler walltime 16m41s |

The numerical check compares native repeat, adapter/native, and native-after-adapter calls under the same BF16 policy. Comparison to the archived FP16 reference is not exact; that cross-precision comparison does not establish parity. These diagnostics do not certify P0/P1/P2 acceptance.

The smoke runner completed in **392.56 seconds**, including loading, hashing, evaluation, and checkpoint output. Its two optimizer steps took **7.962** and **6.612 seconds**. Peak allocated XPU memory was **27,758,145,536 bytes (25.85 GiB)**.

| First optimizer update | Connector | Diffusion transformer |
| --- | ---: | ---: |
| Gradient norm before global clipping | 0.11273698 | 0.52871931 |
| Tensors with nonzero gradients | 4 | 549 |
| Runtime tensors changed | 4 | 400 |

The second update changed four connector and 398 diffusion runtime tensors. The entire diffusion parameter set is in the optimizer; text-to-image inputs need not activate reference-image-only branches. BF16 rounding can also leave a runtime tensor unchanged while its FP32 master accumulates an update. Both parameter groups' aggregate master/runtime hashes changed, and every audited frozen PRISM/VAE/native-conditioner parameter or buffer remained unchanged.

The terminal smoke checkpoint contains full diffusion FP32 masters and Adam state, connector weights/state, diffusion buffers, RNG state, sampler order/cursor, and the bound protocol. It is **47,674,267,569 bytes** with SHA256 `d4cbfa164f07fd48cef7149637213b64b74426164d26904d6ff5bbc7850a7f69`. Large checkpoints remain on Aurora under the task root:

```text
/lus/flare/projects/ModCon/sandeep/prism-docci-joint-20260921
```

Local verification passed **204 tests**: 84 focused tests and 120 prior regression tests. Fixtures establish exact four-step versus two-plus-two resumed master weights, Adam state, RNG, sampler, diffusion buffers, and fixed evaluations. A real multi-billion-parameter resume has **not** yet been exercised.

The pilot's initial 16 train and 16 validation probes exactly reproduce the prior
connector checkpoint's corresponding final probes. Its starting master/runtime
hashes and first training loss also match the independent smoke. However, the
first backward pass differs slightly between these processes: connector gradient
norm is 0.11293755 versus 0.11273698, and diffusion norm is 0.52915627 versus
0.52871931. The second-step loss differs by 0.000116885. Read-only inspection found
no concrete RNG, mode-restoration, or cache bug; the numerical cause remains
unresolved. These runs do not establish bitwise cross-process gradient replay.

## Completed 100-update results

The runner completed in **990.16 seconds**. Median optimizer time was **5.779
seconds per update**; peak allocated XPU memory was **27,761,802,752 bytes
(25.86 GiB)**. The final checkpoint passed an independent file checksum audit.

| Fixed cohort | Initial flow MSE | Step-100 flow MSE | Reduction |
| --- | ---: | ---: | ---: |
| 16 training examples | 0.440569 | 0.439509 | 0.241% |
| 16 validation examples | 0.451623 | 0.451275 | 0.077% |

All-100 final validation MSE is **0.464942**. Its wrong-minus-matched caption gap
is **−0.003797**, and matched captions win on **53/100** examples. The fixed-16
validation gap is −0.034206, compared with −0.032670 initially. These small loss
changes do not establish improved caption conditioning.

The saved [loss curves](../assets/image_generation/2026-09-21-docci-joint/runs/pilot-100-01/plots/connector-diffusion-losses.png)
and [machine-readable series](../assets/image_generation/2026-09-21-docci-joint/runs/pilot-100-01/plots/connector-diffusion-loss-series.json)
keep stochastic training losses, fixed-cohort losses, and the full-validation
marker separate. The 100 validation examples never enter optimization.

### Generated images

All ten planned images completed: four warm-connector/original-DiT images, two
original native OmniGen2 controls, and four post-training images. Every comparison
uses 50 diffusion steps and an exactly replayed initial latent. Sampling is
verified target-free; displayed DOCCI targets are comparison material only.

The [training comparison](../assets/image_generation/2026-09-21-docci-joint/runs/pilot-100-01/train-comparison.png)
and [validation comparison](../assets/image_generation/2026-09-21-docci-joint/runs/pilot-100-01/validation-comparison.png)
show sharper scene structure in some cases, but no clear improvement in the
captions’ requested objects:

- Genie/sign: the trained output still contains a garbled sign and omits the genie.
- Mosaic pyramids: a clearer paved scene still lacks both pyramids and their mosaic.
- Stuffed dog/penguin: the trained output shows an unrelated road/wall scene; the
  animals and blue hats are absent. The native baseline renders recognizable animals.
- Two airplanes: the PRISM-conditioned images still omit the airplanes; the native
  baseline includes two airplanes, including a yellow one.

Three tuned images depict road-like scenes despite different captions. This is a
reason to investigate conditioning; four examples do not establish a general mode
collapse or an image-quality benchmark. Caption fidelity, rather than exact pixel
reconstruction, is the relevant qualitative criterion for this text-to-image run.

### Saved checkpoint and evidence

```text
/lus/flare/projects/ModCon/sandeep/prism-docci-joint-20260921/runs/pilot-100-01/connector-diffusion-pilot-step-000100.pt
```

Size: **47,674,409,749 bytes**. SHA256:
`ed02656d8601e4602ee4980bc8764a44018abb973f5cc5b392a08f7b03777066`.
This is a full joint-training resume artifact, not a connector-only export.

The [artifact bundle](../assets/image_generation/2026-09-21-docci-joint/README.md)
contains reports, step/evaluation logs, images, plots, audited target bytes,
checkpoint metadata, launch inputs, terminal PBS status, and the warm-start replay
audit. Large checkpoints remain on Aurora.

Implementation: [joint runner](../../tools/train_prism_image_diffusion.py) and
[CPU-master AdamW](../../src/training/cpu_master_adamw.py). This pilot verifies real
full-diffusion optimization and frozen-weight preservation. It does not yet
establish useful caption fidelity or readiness for scientific decoders.

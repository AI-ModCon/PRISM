# DOCCI connector pilot with the trained PRISM Qwen3-1.7B parent

**Completed:** DOCCI conversion, exact parent restoration, a two-update smoke, and a 500-update Aurora connector pilot. Fixed validation loss fell by 20.7%, and all frozen weights remained unchanged. Caption fidelity is still poor in the saved sample gallery; this is a working training pipeline, not a qualified image generator.

## Experiment

The user selected connector-only training with pretrained OmniGen2 weights. The input caption passes through the restored PRISM Qwen3-1.7B language backbone, then a new LayerNorm + linear connector maps the final 2,048-dimensional hidden states to OmniGen2's 2,048-dimensional conditioning space. Equal widths do not imply the representations are already aligned.

The image supplies a VAE latent target for the diffusion flow-matching objective. It is not sent to the PRISM vision encoder for this task. There is no text-head loss or alternating image-captioning task. The parent checkpoint includes the original SigLIP2 encoder, image projector/alignment, language backbone, and text head, all restored and frozen. OmniGen2's diffusion transformer, VAE, and native conditioner remain frozen. Only the four tensors of the new connector are optimized: **4,200,448 parameters**.

Parent checkpoint:

```text
/lus/flare/projects/AuroraGPT/sww/prism_outputs/CODEX_QWEN3_1P7B_SIGLIP_CLEAN_GSHUFV1_INVSQRTLR_TRAIN25K_BS4_HSDP_FULLSHARD_16N_R1/checkpoints/step_19750/model.safetensors
```

Aurora task root:

```text
/lus/flare/projects/ModCon/sandeep/prism-docci-qwen3-1p7b-20260921
```

The isolated source snapshot is based on `2c0795e640e95d94cfb69a4832281ca5bafbbb2d`; file hashes and exact launch arguments are saved under `provenance/`. Pretrained OmniGen2 weights are revision `df5dca8a981d74e6c3af214c145f5c735fe72367`, with upstream code revision `18e6f9d5271b517fcb32e999f10df943ae9b8f20` and the existing image-decoder environment.

## Data actually prepared

Official [Google DOCCI](https://huggingface.co/datasets/google/docci) images, descriptions, and grouping metadata were downloaded from version-pinned publisher objects. Original image bytes and full captions were preserved in WebDataset `.jpg`, `.txt`, and `.json` members. See the [operational data guide](../modalities/image_decoder/docci_data.md) for commands and schema.

| Split | Records | Shards | Use in this experiment |
| --- | ---: | ---: | --- |
| Train | 9,647 | 38 | Shuffled optimizer stream |
| Validation (`qual_dev`) | 100 | 1 | Loss probes and final evaluation |
| Test | 5,000 | 20 | Unused by model training/evaluation |
| Qualitative test | 100 | 1 | Unused by model training/evaluation |
| Total | 14,847 | 60 | Integrity-checked conversion |

The source files total **7,766,706,359 bytes**; the tar shards total **7,694,991,360 bytes**, plus indexes/audit. Every image was decoded and joined to its caption. No images were missing or corrupt, and no exact decoded-pixel duplicates crossed splits. The independent PRISM WebDataset validator passed all 9,647 train and 100 validation examples, with zero problem shards.

The grouping audit reports **149 related clusters and 13 entity tags crossing official splits**. Original splits are retained; near-duplicate and subject-disjoint separation are not established. Validation is held out from this connector's optimization. DOCCI is listed in [OmniGen2's dataset construction section](https://arxiv.org/html/2506.18871v1#S3), so these images are not guaranteed unseen by the pretrained generator.

Every training caption was checked with the exact Qwen3-1.7B tokenizer: median 130, 95th percentile 237, maximum 450 tokens. All fit the 1,024-token PRISM limit without truncation. The pilot resizes target images to 256 × 256; this is not the production aspect-ratio or resolution recipe.

Data fingerprint: `f45daff5995f78f2409c3d9a3efc9839aae9e860f08b06e1a59781e180f9a7f0`.

## Completed parent validation

Aurora job **8847607** completed on one node, one process, and one visible XPU tile. This is not a 16-node experiment; the 16-node label belongs to the parent training run.

All **526 parent tensors** restored strictly: 311 backbone, 208 encoder, and 7 projector tensors. The language vocabulary resized from 151,936 base rows to the checkpoint's 151,669 rows, preserving the tied embedding/text-head alias. No parent keys were missing or unexpected. All parent tensors retained BF16 storage before device transfer.

Three cases exercised a short DOCCI caption, the longest training caption, and image-conditioned text. The new text route matched the existing route exactly in all three. Repeated hidden states, text padding, and masked image padding matched exactly; changed inputs changed the representations. All frozen parent parameters/buffers were unchanged. This check captured the connector boundary without loading the image generator; it is not a generation-quality result.

## Training protocol

A separate two-update smoke used two loss-probe examples per split and two diffusion sampling steps. After it passed, the 500-update pilot started from a fresh connector initialization; the smoke checkpoint was not resumed with a changed protocol.

The pilot uses batch size 1, accumulation 1, AdamW at 1e-4, FP32 connector weights, BF16 frozen models, deterministic algorithms, and math SDPA. A 500-update run visits **500 distinct training examples (5.18% of one epoch)** from the 9,647-example shuffled stream. Preparing the complete corpus does not mean it has all been optimized on.

Losses are written each optimizer step. Fixed loss probes use 32 initial shuffled training examples and 32 validation examples every 100 updates. The final evaluation covers all 100 validation examples. Each record reuses its VAE/noise/timestep RNG seed, and the wrong-caption control uses the same target and randomness. Training probes are selected from the optimizer's first examples and report whether each has actually been optimized.

The 50-step sample gallery compares the same caption and saved initial diffusion latent before/after training. Two training and two validation examples are included, plus native OmniGen2 samples for the validation captions. Sampling constructs inputs from caption metadata only and does not load target pixels. These caption-conditioned samples are not image-conditioned reconstructions.

Connector checkpoints persist optimizer state, all RNG states, explicit shuffled order/next cursor, source/data/runtime hashes, and fixed sampling latents. Resume requires a new output directory and identical protocol settings except total steps and output/resume paths.

## Execution ledger

| Stage | PBS job | Status | Evidence |
| --- | --- | --- | --- |
| Parent restoration/routing | 8847607 | Completed; all explicit checks passed | `runs/parent-01/manifest.json`, `provenance/parent-acceptance-checks.json` |
| Fresh numerical check + 2-step smoke | 8847638 | Completed; all explicit checks passed | `jobs/smoke-01/`, `provenance/smoke-command.json` |
| 500-update pilot | 8847660 | Completed, exit 0, 10m40s PBS walltime | `provenance/pilot-command.json` |

Local focused validation: **62 tests passed** across DOCCI conversion/loading, pilot training/resume, launcher checks, and plot-cohort integrity; Ruff and `git diff --check` passed. The real Aurora runs remain separate evidence from these fixture tests.

Compact local evidence: [data conversion](../assets/image_generation/2026-09-21-docci/webdataset/conversion.json), [parent report](../assets/image_generation/2026-09-21-docci/runs/parent-01/manifest.json), [explicit parent checks](../assets/image_generation/2026-09-21-docci/provenance/parent-acceptance-checks.json), and [tokenizer audit](../assets/image_generation/2026-09-21-docci/provenance/tokenizer-lengths.json).

## Completed two-update smoke

Job **8847638** exited successfully in 5m07s, including the fresh numerical precheck. Native-repeat, adapter/native, native-after-adapter, and cross-process comparisons all passed exactly under BF16 and math SDPA. The comparison to the archived reference with a different precision policy did not pass; this run establishes same-policy repeatability, not cross-precision parity or formal image-capability acceptance.

Both optimizer updates completed, all four FP32 connector tensors received nonzero gradients, and every frozen parameter/buffer remained unchanged. Peak allocated XPU memory was **22,396,540,928 bytes (20.86 GiB)**.

| Fixed two-example probe | Step 0 | Step 2 |
| --- | ---: | ---: |
| Train flow loss | 0.586879 | 0.476386 |
| Validation flow loss | 0.714323 | 0.577704 |

These tiny probes establish executable training only. The final validation wrong-caption-minus-correct gap was **−0.123246**, which does not support useful caption alignment. The smoke samples used only two diffusion steps and are not visual-quality evidence.

Smoke connector checkpoint: `runs/smoke-01/connector-pilot-step-000002.pt` under the Aurora task root. The 500-step pilot starts fresh with its own evaluation/sampling protocol.

Smoke [loss curves](../assets/image_generation/2026-09-21-docci/runs/smoke-01/plots/connector-losses.png) and [loss-series JSON](../assets/image_generation/2026-09-21-docci/runs/smoke-01/plots/connector-loss-series.json) are saved locally.

## Completed 500-update pilot

Aurora job **8847660** reached terminal state F with exit 0. The training runner took 628.14 seconds; scheduler walltime was 10m40s on one node, one process, and one visible XPU tile. Peak allocated memory was **23,237,801,472 bytes (21.64 GiB)**.

All 500 optimizer updates had finite losses and gradients. The optimizer consumed **500 distinct training IDs**, none from validation; all four connector tensors received nonzero gradients. The before/after hashes of every frozen PRISM and OmniGen2 parameter/buffer matched. All 10 generated samples were built without target-image inputs, and the saved initial latent hash matched across each case's before/after/native comparisons.

| Step | Fixed train loss (32) | Fixed validation loss (32) | Validation wrong-minus-correct loss (32) |
| ---: | ---: | ---: | ---: |
| 0 | 0.573274 | 0.576600 | -0.021067 |
| 100 | 0.460593 | 0.459662 | -0.010433 |
| 200 | 0.463296 | 0.461352 | -0.011889 |
| 300 | 0.460108 | 0.457739 | -0.008441 |
| 400 | 0.461264 | 0.459100 | -0.010259 |
| 500 | 0.459710 | 0.457064 | -0.009888 |

Fixed train loss fell **19.8%**, and fixed validation loss fell **20.7%**. Most of that reduction happened in the first 100 updates, followed by a plateau. The final **all-100 validation loss was 0.465382**; it is shown separately from the fixed-32 curve. Its mean wrong-caption-minus-correct gap was **−0.003008**, with correct captions winning on 54/100 individual probes. This control does not demonstrate a consistent aggregate advantage for matched captions.

The 50-step images improved from saturated artifacts to some scene/text-like structure, but the inspected examples still miss key caption content: the stuffed-animal scene remains blurred, the runway scene lacks the requested planes, and the training examples do not recover the genie/sign or mosaic pyramids. Native OmniGen2 generates recognizable requested subjects for the same two validation captions under matched starting noise and sampling settings, although details differ from the reference images. The image target is only a training reference; caption-conditioned generation is not expected to reproduce its exact pixels.

The infrastructure and data route are verified. These results do not establish learned semantic alignment or justify moving to scientific decoders yet. Any longer training should continue to track caption controls and visual fidelity; the current loss plateau makes loss-only extrapolation unreliable.

Artifacts:

- [Loss curves (PNG)](../assets/image_generation/2026-09-21-docci/runs/pilot-500-01/plots/connector-losses.png), [PDF](../assets/image_generation/2026-09-21-docci/runs/pilot-500-01/plots/connector-losses.pdf), and [auditable series](../assets/image_generation/2026-09-21-docci/runs/pilot-500-01/plots/connector-loss-series.json).
- [Validation comparison](../assets/image_generation/2026-09-21-docci/runs/pilot-500-01/validation-comparison.png) and [training comparison](../assets/image_generation/2026-09-21-docci/runs/pilot-500-01/train-comparison.png).
- [Run report](../assets/image_generation/2026-09-21-docci/runs/pilot-500-01/report.json), [per-step log](../assets/image_generation/2026-09-21-docci/runs/pilot-500-01/steps.jsonl), [evaluations](../assets/image_generation/2026-09-21-docci/runs/pilot-500-01/evaluations.jsonl), and [explicit final checks](../assets/image_generation/2026-09-21-docci/provenance/pilot-acceptance-checks.json).

Final connector/optimizer/RNG checkpoint, retained on Aurora:

```text
/lus/flare/projects/ModCon/sandeep/prism-docci-qwen3-1p7b-20260921/runs/pilot-500-01/connector-pilot-step-000500.pt
```

The 50,936,209-byte file was rehashed after the run: SHA256 `e84b3550304f4cebe56b394e1e1f65dfc333bd9c6ed8e9adbb240c6694c4387f`. It contains connector weights plus resume state, not a duplicated full PRISM/OmniGen2 model. Checkpoints at steps 0, 100, 200, 300, and 400 remain alongside it.

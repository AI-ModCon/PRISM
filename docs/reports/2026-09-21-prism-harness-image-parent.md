# Trained PRISM parent for OmniGen2

The selected parent is the complete Qwen3-8B vision-language model referenced by
`PRISM-Harness` at `92fcb3c8955c84034d5949db175a684876f2809a`, in
`tools/harness_acceptance/RESULTS.md`. It replaces the untrained Qwen3-0.6B
configuration template as the starting point for this experiment.

## Parent and architecture

Checkpoint on Aurora:

```text
/lus/flare/projects/AuroraGPT/sww/prism_outputs/CODEX_QWEN3_8B_SIGLIP_CLEAN_GSHUFV1_INVSQRTLR_TRAIN25K_BS4_HSDP_FULLSHARD_16N_R1/checkpoints/step_20000/model.safetensors
```

The saved `.hydra/config.yaml`, `training_state.json`, checkpoint header, and
training log were inspected. The checkpoint contains 614 tensors: 399 backbone,
208 vision encoder, and seven image projector tensors. Its file size is
16,602,829,888 bytes. Qwen embeddings and output head have shape `[151669,4096]`
and BF16 dtype. Both the vision encoder and language model were unfrozen during
training; projector, encoder, and language-model optimizer groups all used
`5e-5`. The saved state records 20,000 completed steps. The training log ends at
step 20,000 with token-weighted loss 0.6443; this is a training metric.

```text
image -> trained SigLIP2-base/224 -> trained PRISM projector (768 -> 4096)
                                                                         \
text  -> trained Qwen3-8B token embeddings -------------------------------> trained Qwen3-8B
                                                                            |
                                                              final hidden states [B,L,4096]
                                                               /                          \
                                                   existing text head              NEW LayerNorm + Linear
                                                                                         4096 -> 2048
                                                                                              |
                                                                                 frozen OmniGen2 generator
                                                                                              |
                                                                                         native VAE -> image
```

Prefix fusion, the two-layer image projector, source processor, tokenizer and
all trained parent weights are retained. The new image output connector is the
only component selected for optimization. Training targets enter the generator's
VAE/flow loss separately from source inputs.
The backend still loads OmniGen2's native Qwen2.5-VL component for reference and
negative-prompt support. PRISM replaces the **positive** conditioning path; the
flow-training forward uses the new PRISM connector's states directly.

The [Aurora config](../../src/conf/image_generation/qwen3_8b_prism_harness_omnigen2.json)
uses immutable, already staged Qwen and SigLIP snapshots and the pinned OmniGen2
checkpoint. These base snapshots provide architecture/tokenizer/processor assets;
the complete trained PRISM checkpoint overwrites the parent model weights.

## Restoration and diagnostics

The image loader now restores the saved vocabulary size safely, preserves tensor
dtypes and tied weights, and rejects missing parent tensors or normalization-key
collisions. Only new output-connector tensors may be absent. A restoration report
records the loaded tensor inventory, saved dtypes, strict assignment policy and
tied-weight consistency checks. Full value-preservation assertions are covered
by local tests, not a separate runtime comparison of every loaded tensor. Backbone
construction uses `torch_dtype`, compatible with the pinned Transformers 4.51.3.

`tools/validate_prism_image_parent.py` measures real parent encoder/projector/LM
and output-connector routing, legacy/new text-route agreement, explicit padding,
source-image and instruction changes, and frozen-state integrity. Its image
transport is intercepted to capture conditioning tensors; it does not run the
image generator or claim full P1 acceptance.

`tools/smoke_prism_image_training.py` is a separate bounded optimization diagnostic:
at most four examples/steps, images up to 256 pixels, and one to four sampling
steps. It exercises the real frozen generator's gradient path, keeps the new
connector in FP32, audits frozen parameters/buffers, and samples without targets.
It writes an **unqualified** artifact that the production connector loader rejects.
The full `train_image_decoder.py` evidence gates remain intact.

The fixed local cases use two existing PixMo UI images (pen and chicken) plus one
text prompt. Two manually written captions pair with those same images for the
optimization smoke. These are engineering examples with unknown overlap with the
parent's training corpus, **not held-out alignment or image-quality benchmarks**.
Both optimization rows are text-to-image: they exercise the trained LM through
the new connector and generator. The parent diagnostic exercises the trained
vision encoder/projector separately; image-conditioned image generation still
requires an editing or reference-image training slice.
The parent benchmark and optimization manifest deliberately overlap and must
never be used as independent train/validation splits.

The optional single-word answer labels in the caption cases are not a caption
quality metric: a correct descriptive sentence can fail exact matching. The pen
counterfactual changes both source image and prompt, so that comparison is joint
sensitivity, not an isolated instruction effect. The separate zero-image-input
ablation measures dependence on the encoder path. None of these establishes
semantic alignment on held-out examples.

Use `tools/launch_aurora_prism_image.py --mode parent` first and
`--mode connector-smoke` for the bounded generator run. Both require `--dry-run`
review before `--submit`, allocate one node/process/XPU tile, and default to ten
minutes. The integration is staged separately from the source parent and existing
environments under `/lus/flare/projects/ModCon/sandeep/prism-parent-omnigen2-20260921`.

## Executed parent diagnostic

Aurora PBS job **8845790** completed with exit 0 in **3 minutes** on
`x4518c6s3b0n0`, one process and one visible XPU tile. Full checkpoint SHA256:
`446f2cca812df52d7b178a417186690a6f05aade0e54f5fe3844e30e784d2abf`.

- All 614 BF16 parent tensors restored, with no missing parent or unexpected
  keys; four output-connector tensors were newly initialized. Vocabulary resized
  from 151936 base-config rows to the saved 151669 rows.
- Both image cases produced encoder `[1,196,768]`, projector `[1,196,4096]`,
  final LM `[1,202,4096]`, and connector `[1,202,2048]` tensors, all finite.
  Text-only conditioning produced seven valid tokens.
- Legacy and new text routes generated **identical token IDs on all three cases**.
  The frozen parent parameters and buffers had identical before/after hashes.
- Zeroing the encoder input changed last-token states by relative L2 0.270 and
  0.272 for the two image cases. This demonstrates input dependence, not accuracy.
- Padding comparisons were **not exact**: last-token relative L2 differences
  were 0.78–1.06%. The diagnostic has no same-input repeated-forward control,
  so it cannot distinguish padding effects from native BF16 repeatability.
  Full padding/batch invariance acceptance remains open.
- The pen caption began with a correct black-pen description. The chicken caption
  instead described a person's head and shoulders; the arithmetic prompt elicited
  a continuation rather than an answer within the 16-token limit. These are
  qualitative limitations of this parent/protocol, not a held-out score. The
  report's single-word exact-match aggregate is not a meaningful caption metric.

The diagnostic took 37.09 seconds after loading; the scheduler's three-minute
duration includes environment setup and full checkpoint loading/hashing. OmniGen2
was not loaded by this parent-only run. The local summary is
`outputs/image_decoder_validation/20260921-prism-parent/parent-manifest.json`.
Full traces and reports remain under `runs/parent-01` in the isolated Aurora directory.

Local validation: **181 targeted tests passed**; changed Python files pass Ruff
and `git diff --check`. These include offline optimization fixtures and the real
tiny-Qwen routing tests; only the PBS experiment establishes the real 8B parent
execution above.

## Executed real-generator optimization smoke

Aurora PBS job **8845819** completed with exit 0 in **4 minutes 44 seconds** on
`x4206c2s3b0n0`, one process and one XPU tile. The runner measured 276.41 seconds
including loading. This used the full trained PRISM parent and real pinned
OmniGen2 generator, not a fixture or the native positive conditioner.

- **8,398,848 FP32 connector parameters** were optimized with AdamW, learning rate
  `1e-4`, gradient clipping at 1.0, seed 42, two 256×256 T2I examples and two steps.
- Every connector parameter tensor received finite, nonzero gradients on both
  steps. The respective training losses were **0.852840** and **0.562994**; these
  use different examples/noise draws and are not a controlled learning curve.
- Resetting the seed for the first-example training probe gave loss
  **0.852477 → 0.226447**. This is a single training probe after two updates, with
  known XPU/BF16 repeatability limitations; it does not pass P2 or establish
  generalization despite the apparent 73.4% decrease.
- All **2,305 frozen parameter/buffer hashes** matched before and after. The PRISM
  parent checkpoint hash matched the parent diagnostic. Generator manifest SHA256
  `46ad5c24af04baef2c5b6d365ee0ccfdc24eb626d98d56c017470b1f19f37d09`
  matched the earlier native reference smoke.
- Peak allocated XPU memory: **31.76 GiB**; peak reserved: **31.99 GiB**.
- Two-step target-free sampling produced a 256×256 PNG. Its SHA256 was verified
  locally and the image inspected: it is predominantly pale texture, without a
  recognizable pen. This is sampling execution evidence, not useful image quality.

The saved `connector-smoke.pt` SHA256 is
`e7b335fdd8def48603357d3a2f7b4c7dcca569c1f7587456ae8d5a9b0cdd20cd`.
It remains explicitly **unqualified**, and the production loader rejects it.
The [initial generation results bundle](../assets/image_generation/2026-09-21-initial/README.md)
includes the byte-identical [smoke image](../assets/image_generation/2026-09-21-initial/smoke/after-smoke.png)
and [original report](../assets/image_generation/2026-09-21-initial/smoke/report.json).
The local summary is
`outputs/image_decoder_validation/20260921-prism-parent/connector-report.json`;
the inspected image and verified connector artifact are adjacent as
`after-smoke.png` and `connector-smoke.pt`. The isolated Aurora directory retains
the original source/config, reports, traces, scheduler logs and weights. Both
jobs are finished; no further job was submitted in that diagnostic stage.

## Remaining qualification

The parent training run had held-out evaluation disabled. PRISM-Harness reports
text-grid reasoning benchmarks, but those do not establish image-input alignment
or image-generation quality. Fresh visual evaluation is still required.

The original full 8B text-grid summaries were copied from Aurora and their
aggregates recomputed. These are **historical parent baselines**, not reruns of
the extended model:

| Harness strategy | Successful routes / scored tasks | Success rate |
|---|---:|---:|
| Direct | 255 / 1200 | 21.25% |
| Format repair | 311 / 1200 | 25.92% |
| Online environment/tool repair, one repair | 324 / 1200 | 27.00% |

The exact saved training metadata, header and these summaries are archived under
`outputs/image_decoder_validation/20260921-prism-parent/provenance/`, with file
SHA256 records. Raw runtime artifacts remain outside source commits.

Native OmniGen2 XPU/BF16 repeatability remains unresolved from the earlier smoke.
Neither parent restoration nor a successful optimization diagnostic closes that
gate. Continue with the registered native/reference acceptance suite, held-out
parent VLM evaluation, connector overfit/pilot training, then sealed image
qualification as specified in the 2026-09-21 image-first validation plan, since
retired; see `git log`.
Scientific output experiments follow their own domain baselines and validity checks.

Subsequent execution is recorded in the [controlled connector overfit report](2026-09-21-prism-connector-overfit.md): deterministic BF16/math attention gave exact native/adapter repeats across two processes on different nodes, and eight parent cases passed repeat/padding/text-route checks. The separately authorized 16-pair learning experiment extends beyond the two-step smoke described here; consult that report for its actual status and qualification limits.

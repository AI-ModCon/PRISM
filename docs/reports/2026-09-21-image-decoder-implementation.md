# Image decoder implementation and validation record

Date: 2026-09-21. Branch: `feat/decoder-extension-plan`. Rebased parent:
`234e8aa5433e261237746ea0de6ed8b95ca71dd7`, incorporating `origin/main` at `0e267e3`.

## Implemented

- Typed decoder conditions/results; explicit output requests, separate supervision,
  named weighted losses, and native output generation on the eager HF backbone.
- Ordered multiple-image conditioning with padding excluded before encoding and
  normalization; prompt-only prefix/interleaved compilation and sequence guards.
- Lazy, pinned official OmniGen2 adapter, reference bypass, sequence connector,
  frozen-generator flow-matching loss, and upstream PyTorch kernel selection on XPU.
- Distinct source/target preprocessing, group/content split checks, bounded connector
  runner, frozen-state/gradient audits, identity-bound artifacts and strict restoration.
- Reference/parity traces, complete attempt ledgers, registered qualification/ablation
  checks, and a one-node/one-XPU-tile Aurora smoke launcher.
- Masked scientific baseline pooling and explicit graph node correspondence. These
  changes do not implement or qualify de novo scientific structure generation.

Usage and exact scope: [image decoder guide](../modalities/image_decoder.md).

## Local verification

233 selected tests passed; 7 tests requiring broader model assets/integration were
deselected. Coverage includes new contracts and runners, synthetic optimization,
strict checkpoint restoration, actual tiny random Qwen3 routing, kernel-selection behavior, old decoder behavior,
tokenizer resizing, interleaving, and native loss aggregation. These are offline
engineering tests, not pretrained model quality measurements. Ruff and
`git diff --check` passed. The locally installed base environment lacks the pinned
image stack; its metadata preflight correctly reports blocked.

Review found and fixed masked images contaminating projector statistics, unbound
tokenizer/processor replacements, and cached negative conditions retaining a CPU
device when the generator runs on XPU. Training now requires a separate P1
acceptance report with measured, hashed regression artifacts; reference-adapter
parity alone cannot authorize training. Regression tests cover these boundaries.

## Aurora staging and smoke

Isolated directory:
`/lus/flare/projects/ModCon/sandeep/prism-image-smoke-20260921`.
The original remote repository on `main` is unchanged. The staged code is a tracked
Git archive plus an explicit source overlay; all 33 final overlay file hashes were checked.

- Upstream source: `18e6f9d5271b517fcb32e999f10df943ae9b8f20`.
- Released model: `df5dca8a981d74e6c3af214c145f5c735fe72367`.
- Download: 33 required component files, 31,254,573,901 bytes.
- Integrity: all 33 files match the pinned Hub revision (nine LFS SHA256 and
  24 Git blob SHA1 identities). The resulting SHA256 inventory is
  `2b50ee34c5b19490c20d7dcba78df0543d90a3294c42c0bb1ec76c7bc3bc77ea`.
- Environment: `frameworks/2025.3.1`, framework XPU PyTorch
  `2.10.0a0+git449b176`, isolated Transformers 4.51.3, Diffusers 0.33.1,
  Tokenizers 0.21.4. Existing PRISM environments were preserved.
- Initial import exposed upstream's package-presence check selecting CUDA Triton
  normalization on XPU. The adapter now explicitly selects upstream's existing
  PyTorch RMSNorm/native SwiGLU/SDPA paths. No downloaded source or weights changed;
  cross-kernel CUDA/XPU numerical equivalence remains unmeasured.
- Smoke budget: one node, one process on one XPU tile, ten minutes; one 256×256
  T2I case with seed 42 and two denoising steps, followed by the PRISM reference
  adapter comparison in the same allocation.

Fresh-process imports passed on Aurora with the selected PyTorch kernels and no
CUDA build. The first PBS job, `8845492`, stopped in preflight after five seconds:
the local-dir checkpoint lacked its verified provenance inventory. No model was
loaded. Its failure reports and job logs are retained. The downloaded files were
then verified against pinned Hub metadata and the inventory written. The launcher
now runs file-integrity preflight before allocating a node.

The second job, `8845586`, passed file-integrity preflight but stopped after 42
seconds because XPU enumeration returned zero devices. The launcher had selected
`ZE_AFFINITY_MASK=0.0` under Aurora's FLAT device hierarchy. Existing PRISM launchers
and operational notes confirm that this mode requires integer tile IDs. The smoke
launcher now sets `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ZE_AFFINITY_MASK=0`, and
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, and checks exactly one visible tile before
loading weights. Neither failed job loaded the model or generated images.

The corrected preflight passed, and PBS job `8845620` ran on `x4407c3s4b0n0`.
The early probe confirmed exactly one visible Intel Data Center GPU Max 1550 tile.
Native reference generation succeeded: one 256×256 image, seed 42, two denoising
steps, 2.007 seconds sampling, 120.373 seconds for the complete reference command.
Peak allocated memory was 15,984,962,560 bytes (14.887 GiB); peak reserved memory
was 16,494,100,480 bytes (15.361 GiB). The two-step image was visually inspected:
it is very blurry and is execution evidence, not a quality result.

The adapter also generated an image (2.195 seconds sampling, 83.634 seconds for
its command), but **exact parity failed**. Tokens, masks, positions, initial
noise, and the timestep schedule matched. The first divergence was in native
conditioner hidden states (maximum absolute difference 2.0 for positive and 1.0
for negative conditioning); final pixels differed by up to 3 integer levels.
The job ended with exit 2 after 3 minutes 30 seconds. Both images and all 52
captured tensor arrays are retained. This does not establish whether the cause
is the adapter or numerical repeatability across separate XPU model loads.
A bounded native-repeat/adapter/fresh-process diagnostic was submitted as PBS
job `8845651`, using the same original latent bank and unchanged exact tolerances.
Each of two fresh processes loads once and runs native N0, native N1, adapter A0,
then native N2. The allocation remains one node, one tile and ten minutes; the
diagnostic does not train weights or pass P0/P1. It completed with exit 0 in
3 minutes 25 seconds on `x4204c1s0b0n0`; exit 0 means the diagnostic executed,
not that any numerical comparison passed.

Both processes completed all four calls. **Every exact comparison differed**,
including native N0 versus native N1, adapter A0 versus native N1, native N2
versus native N1, and fresh-process N0 versus prior-process N0. In the first
process, consecutive native hidden states differed by up to 1.0 (positive) and
2.0 (negative); in the second, they differed by up to 13.1875 and 1.625.
Across diagnostic comparisons, final pixel differences were 2–4 integer levels.
The original initial-latent hash and all recorded execution settings stayed
unchanged, including eval mode and disabled autocast. Deterministic algorithms
were not enabled; multiple SDPA backends were available. No tolerances changed.

This establishes native-path variation under the tested XPU/BF16 setup. It does
not isolate the responsible kernel/state or exclude an additional adapter effect.
The next numerical investigation needs a deterministic execution policy and/or
an FP32 native baseline, followed by repeated native and adapter comparisons.
Exact P1 parity remains unproven, and no connector or scientific training began.

Evidence (local archive paths relative to the repository root; these files are
outside source commits and are not GitHub links):

- Original reference and adapter comparison:
  `outputs/image_decoder_validation/20260921-aurora-smoke/reference-smoke-03/runs/reference-smoke-03-parity/report.md`
- First repeatability process:
  `outputs/image_decoder_validation/20260921-aurora-smoke/repeatability-04/runs/repeatability-04/report.md`
- Fresh repeatability process:
  `outputs/image_decoder_validation/20260921-aurora-smoke/repeatability-04/runs/repeatability-04-fresh/report.md`

The corresponding directories contain images, raw boundary tensors, per-call
settings, source snapshots, and scheduler logs. A completed diagnostic must not
be used as a passed P0/P1 training prerequisite.

Its staged code is the verified 33-file inventory plus
the four-file overlay archive with SHA256
`0c97df3118fb0e7e7141bb6228d39de22430f57d8fe197f7b7f07146c955c426`.
The subsequently tightened training-acceptance gate is covered by local tests;
it is not part of this generation-only runtime experiment.

The diagnostic uses the complete updated 35-file overlay, SHA256
`88ca3c09fe7b89969cb34e2ae1978eb1ad80b09c62ff77ecda64bbf25c49d5f5`, with
manifest SHA256 `15e049d5d608cdcbf131bd3cf87e7f03fd7b9c50c67e0843d51de3a44ca813f5`.
The original smoke source snapshot and both failed comparison traces are retained.

## Remaining gates

The short smoke cannot pass full P0/P1 (30 cases × three seeds), P2 optimization,
P3 held-out pilot, or P4 qualification. Real PRISM connector training additionally
requires an input-aligned parent with matching tokenizer/processor evidence. Image
qualification and independent scientific evaluations precede scientific training.
No learned image or scientific capability is claimed by this implementation record.

# DOCCI conditioning diagnosis and staged adaptation

This continues the completed 500-step connector and 100-step joint pilots. The
goal is to establish useful caption conditioning with the PRISM parent frozen,
then train on the complete prepared DOCCI pool. Execution results belong in the
run report; settings below are planned until backed by a completed run.

The current execution choice, updated September 22, is to train the connector and
diffusion transformer jointly on all 9,647 prepared pairs. The user explicitly
superseded the earlier requirement to recover small-set image quality before
scaling. Those diagnostics remain useful evidence; full-data flow training may
learn a different connector/generator interface. PRISM and VAE remain frozen.

## Fixed scope and data

- Parent: PRISM Qwen3-1.7B/SigLIP2 `step_19750`; vision encoder, input alignment,
  language model and text head stay frozen.
- Trainable: the PRISM output connector and all OmniGen2 diffusion transformer
  parameters, using the existing BF16 runtime / FP32 CPU-master optimizer.
- Data: 9,647 official training pairs and 100 `qual_dev` pairs. No validation
  target enters optimization. Original test splits remain unused.
- Caption-to-image only: target pixels enter the VAE, never PRISM's vision encoder.
- Initial resolution is 256 square pixels, with 50-step generation for comparisons.

## 1. Repeated small-set training

Select 32 training examples deterministically from the seeded full-data order.
Record the selected IDs, every optimizer exposure, unique count, and epochs over
that subset. Use repeated updates rather than one pass through distinct examples.
A six-update batch-four smoke measures memory and timing before choosing the
larger allocation. Start from the audited 500-step connector and original DiT;
keep the previous 100-step result as a comparison, not an unrecorded initializer.

Evaluate a fixed training cohort and held-out validation cohort periodically,
save optimizer checkpoints, and generate from caption metadata and fixed initial
noise. Judge the requested principal objects and attributes as well as flow MSE.
Successful execution alone does not establish successful small-set fitting.

## 2. Conditioning controls

Compare native OmniGen2 and PRISM conditioning using the same targets, VAE
posterior seeds, diffusion noise and timesteps. Repeat matched/wrong-caption
controls across multiple seeds. Record token lengths, masks, finite values and
feature scales. Compare raw PRISM prompts with explicit chat formatting as an
ablation, without silently changing the training format.

For fixed-noise generations, compare the original native path, the current PRISM
positive/native-negative path, guidance disabled, and a PRISM negative branch.
An empty raw prompt may need an explicit EOS anchor; record this choice and do
not describe that untrained anchor as a learned unconditional representation.

Use native controls to calibrate the caption-loss diagnostic. Inspect whether
the small-set model produces the requested objects and distinguishes captions.
If it fails, investigate the interface or a teacher-supervised connector before
spending a full-data run on the same failing setup. Any stronger connector or
teacher objective requires its own recorded experiment and validation.

An optional bounded implementation is available in
[`align_prism_image_conditioning.py`](../../../tools/align_prism_image_conditioning.py).
It keeps both parents and the DiT frozen, uses exact token-aligned native
features as teachers, and trains only the connector on caption-content positions
after the actual frozen caption RMSNorm. It reads caption metadata only, saves
its own checkpoint evidence kind, and does not establish generation quality.
Use a separate source snapshot if this fallback is needed. Its explicit
validation/restore contract is implemented in
[`prism_image_alignment_checkpoint.py`](../../../tools/prism_image_alignment_checkpoint.py)
and the diagnostic's `--alignment-checkpoint` route. Run paired flow controls
plus caption-only generation after restoration; existing flow-training loaders
do not accept this distinct checkpoint implicitly. Do not change a running
experiment's source.

If content alignment fails to recover generation, the diagnostic supports
`--template-swap-controls` with a strict aligned checkpoint and chat prompts.
Compare native and aligned PRISM at CFG1, then native-prefix, native-suffix and
both-template replacements in the aligned sequence before DiT RMSNorm. Require
observed teacher token/mask identity, unchanged content partitions, actual DiT
conditioning hashes and identical noisy inputs/initial latents. These are
teacher-assisted localization controls; causal suffix states can carry caption
semantics, so a rescue does not by itself identify a formatting-only defect.

The region-alignment revision uses `--objective regions` with explicit verified
content-alignment initialization. It equally weights prefix, content and suffix
MSE after frozen DiT RMSNorm and retains the same 32/eight caption cohorts. Its
schema-2 checkpoint kind and objective are distinct from the content-only schema-1
contract. The strict helper validates the complete warm-start lineage, all 40
token/feature audits, fresh optimizer/sampler state and exact exposure counts.
Collect with the versioned `provenance/regions` helpers, verify both checkpoint
digests, and evaluate content retention plus caption-only images. The ordinary
diagnostic's `--native-cfg1-control` supplies matched-guidance native comparison
alongside the native CFG5 reference. Feature loss alone is not a scaling gate.

If its generation diagnostic supports proceeding, the joint runner's explicit
`--init-alignment-checkpoint` route restores through that same strict helper.
It requires the audited step-500 warm start, original DiT and chat prompts, and
copies FP32 connector weights before constructing fresh CPU-master optimizer
state. Run a short joint smoke in a new source snapshot and measure batch four
with accumulation four before budgeting full-data training. The resulting audited
joint checkpoint can initialize the full-data stage; the feature checkpoint is
never relabeled as a joint flow-training checkpoint.

The training runner's global-RNG-reset wrong-caption probes must be checked
against the separate diagnostic's recorded actual noisy-input/timestep hashes
before treating their gaps as verified caption sensitivity.

If joint updates worsen matched-seed images, use the separate frozen
[`diagnose_prism_joint_components.py`](../../../tools/diagnose_prism_joint_components.py)
control to cross the verified region and joint-six connectors with the original
and joint-six DiTs. Native conditioning on each DiT supplies two additional
references. This has its own evidence kind and fixed two-training/two-held-out,
three-timestep protocol; it performs no optimization. Validate the region
checkpoint against the original DiT before admitting the joint checkpoint, then
verify each complete runtime state swap and restore the original baseline at exit.

Measure prefix/content/suffix feature drift under the captured original RMSNorm.
Compare actual noisy inputs and conditioning hashes across all six flow routes,
and generate both held-out cases at CFG5 with the same native negative condition,
initial noise and 50-step schedule. Record the actual CFG branches and final
latent hashes. The versioned `provenance/components` collector and renderer check
both checkpoint digests, source reports, all phase audits and route coverage.
Use the results to distinguish connector drift, DiT drift and their interaction;
successful artifact acceptance alone still does not establish image quality.

## 3. Full prepared-data training

With numerical execution and checkpoint restoration audited, initialize a
new full-data stage from a completed joint checkpoint with fresh optimizer and sampler.
This is explicitly different from exact resume, which must retain the same
source, data selection, conditioning, optimizer and runtime protocol.

Train through the complete 9,647-pair pool, record exact coverage and repeated
exposures, save periodic checkpoints, and keep train/validation loss curves and
caption-control results. Budget the capacity allocation from measured throughput.
This larger-data experiment is now authorized to proceed without a further
tiny-set semantic-quality prerequisite. Evaluate whether joint learning improves
caption following over the full pool; native-feature preservation is not the
optimization objective. Keep periodic losses and images to assess that question.
An adapted DiT with native conditioning is not the original pretrained baseline.
Further data expansion is a later decision based on training versus held-out
caption fidelity, not on denoising loss alone.

## Artifact locations

Aurora experiment root:
`/lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922`.

Keep model checkpoints on Aurora. Bring back compact reports, loss series,
plots, sample comparisons, launch manifests, and source hashes. Preserve the
earlier pilot directories and all unrelated workspace files.

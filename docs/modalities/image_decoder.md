# Image output through PRISM

This implements the image-first decoder plan on the eager Hugging Face backbone.
The implementation and its offline contract tests are distinct from the P0–P4
checkpoint, optimization, and image-capability gates. Those gates require the
recorded real runs described below; an engineering smoke does not pass them.

The [DOCCI data workflow](image_decoder/docci_data.md) adds verified WebDataset
conversion and a resumable connector-only pilot using the trained PRISM
Qwen3-1.7B/SigLIP2 parent. Its [run report](../reports/2026-09-21-docci-connector-pilot.md)
separates completed data/parent checks from actual optimization and quality evidence.

The subsequent [joint-training workflow](image_decoder/docci_diffusion.md)
trains the connector and diffusion transformer together while PRISM, the VAE
and native text conditioner remain frozen. The completed [DOCCI report](../reports/2026-09-22-docci-alignment-stages.md#completed-caption-conditioning-evaluation)
records 1,809 full-data updates, held-out caption controls and the
[final penguin/airplane comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/full-docci-joint-conditioning-01/matched-seed-before-after.png).
The architecture below shows the default frozen-generator route; joint training
explicitly enables gradients for the connector and diffusion transformer.

The [connector API](image_decoder/connectors.md) makes the current path explicit
as a final-state readout, a LayerNorm/linear bridge, and the OmniGen2 generator.
Its [configuration](../../src/conf/image_generation/qwen3_1_7b_prism_harness_omnigen2_connectors.json)
preserves the existing image route and checkpoint keys. Alternate readers are
not implemented.

## Architecture and APIs

```text
source images -> PRISM image encoder/projector --+
instruction tokens -> pretrained Qwen embeddings+-> causal LM hidden states
                                                      |
                                             masked sequence connector
                                                      |
ordered source images -> frozen native VAE ------------+-> frozen OmniGen2 -> image

target images -> separate VAE / noise / flow-matching loss (training only)
```

`DecoderCondition` carries `[B,L,D]` hidden states, `[B,L]` validity, per-example
modality spans, native context, output specification, and provenance. `DecoderResult`
returns `predictions`, unweighted named `losses`, their weighted sum `loss`, and
provenance. `ModelConfig.decoder_loss_weights` defaults to weight 1 per decoder.

```python
result = model.forward_outputs(
    inputs={"text": token_ids, "text_attention_mask": text_mask,
            "image": source_pixels, "image_mask": source_mask},
    targets={"image": target_rgb_minus1_to1},
    requested_outputs=["image"],
    native_context={"image": {"reference_images": ordered_pil_images}},
    output_specs={"image": {"height": 512, "width": 512}},
)
result.loss.backward()

generated = model.predict(
    inputs=inputs, requested_outputs=["image"],
    native_context={"image": {"reference_images": ordered_pil_images}},
    decoder_kwargs={"image": {"height": 512, "width": 512,
                              "num_inference_steps": 50}},
).predictions["image"]
```

Source pixels are `[B,N,3,H,W]` with a binary `[B,N]` reference mask; valid references
must precede padding. Their order matches `ordered_pil_images: list[list[PIL.Image]]`.
The input encoder processor and generator VAE use separate preprocessing. A batch
with no source images omits `image` entirely. Image sampling currently accepts one
prompt per call; training accepts batches with equal target dimensions.
The released generator supports at most five source references per example.

Prefix fusion places source tokens before text. Interleaved fusion replaces each
adjacent configured start/end marker pair with one reference, consumes every
reference, and needs no answer-length metadata. For interleaved manifests, prompts
must already contain the appropriate markers in the intended positions. The
merged-length guard runs before the backbone. Always supply an explicit text mask;
it is mandatory when PAD and EOS share an ID.

`forward(inputs, labels=...)` retains the legacy text tuple. An explicit
`requested_outputs=...` selects the new result, including through DDP's public
forward hook. Text targets must be supplied separately, aligned with input text
IDs, with `-100` for instructions and ignored positions. Image-only training does
not add a text language-model loss. Native trainer batch dispatch supports nested
targets and named losses; the dedicated runner below owns the image evidence and
freeze checks. Custom backbones and VLA fail explicitly for this new route.

Time-series and geometry baselines now pool only valid positions. Structured graph
output requires explicit `native_context['graph']['node_indices']`; a resampled
graph token is not automatically a node. These heads are not de novo graph or 3D
structure generators. Scientific training remains gated on image qualification
and subsequent independent domain evaluations.

## Supported reference

The adapter uses the official `omnigen2` package, not a stock Diffusers pipeline:

- Source: `VectorSpaceLab/OmniGen2`, commit `18e6f9d5271b517fcb32e999f10df943ae9b8f20`.
- Weights: `OmniGen2/OmniGen2`, revision `df5dca8a981d74e6c3af214c145f5c735fe72367`.
- Adapter API pins: `diffusers==0.33.1`, `transformers==4.51.3`.
- Native VAE, Qwen2.5-VL conditioner, flow scheduler, and diffusion transformer are
  loaded explicitly. Optional approximate inference caches are disabled.

Stage the clean upstream checkout on `PYTHONPATH` and its checkpoint in a separate
image environment. Keep Aurora's framework PyTorch installation intact. Importing
PRISM or constructing its image decoder does not download weights or import the
optional generator. Runtime loading is offline by default and rejects an unsupported
source/dependency revision. `backend.ensure_loaded()` precedes optimizer construction
and frozen-state auditing; `checkpoint_manifest()` hashes actual component files.

Prefer the `snapshots/<immutable revision>` directory returned by Hugging Face's
`snapshot_download(..., revision=<SHA>)`. A separately copied/local-dir checkpoint
also needs `prism_checkpoint_provenance.json` with `revision` and a `files` mapping
from relative paths to verified SHA256 values. Verify downloaded bytes against
the pinned Hub revision's LFS hashes or Git blob IDs before creating that inventory.
Do not establish provenance by renaming an arbitrary directory.

On XPU/CPU the adapter selects upstream's existing `torch.nn.RMSNorm`, native
SwiGLU and PyTorch SDPA paths before importing model classes. Package presence alone
would otherwise select CUDA Triton kernels on Aurora. This kernel policy is recorded
in provenance; it does not establish parity with CUDA fused kernels.

## Data

Training JSONL records use this schema (paths are relative to the manifest):

```json
{"id":"edit-001","task":"edit","prompt":"Change the mug to blue.","source_images":["images/red-mug.png"],"target_image":"targets/blue-mug.png","split":"train","group_ids":["mug-001"],"source_ids":["red-mug"]}
```

Tasks are `t2i`, `edit`, and `in_context`. Source and target paths/content must be
distinct. `validate_manifest_splits()` checks shared groups, paths, and decoded
image content across splits. The dataset fingerprints sources and targets and
normalizes only targets to RGB `[-1,1]`. No automatic text truncation is allowed.

Validation JSONL contains no target fields:

```json
{"case_id":"t2i-smoke","task":"text_to_image","prompt":"A red ceramic mug on a white table.","reference_paths":[],"group":"smoke-mug","split":"smoke","seeds":[42],"height":256,"width":256}
```

For full P0/P1 use 10 cases per task and three seeds per case. P4 requires 200 per
task, the registered training seeds, independent subject groups, and the sealed
protocol. Partial suites remain incomplete; they cannot authorize training.

## Reference and parity runs

Metadata preflight performs no model loading and records blockers:

```bash
python tools/validate_image_decoder.py --preflight \
  --upstream /path/to/OmniGen2 --checkpoint /path/to/pinned-snapshot \
  --output-dir /new/run/preflight
```

Run real generation on an allocated compute node:

```bash
python tools/validate_image_decoder.py --reference \
  --upstream /path/to/OmniGen2 --checkpoint /path/to/pinned-snapshot \
  --cases /path/to/reference-30.jsonl --device xpu --dtype bfloat16 \
  --output-dir /new/run/reference

python tools/validate_image_decoder.py --parity --parity-mode full_pipeline \
  --upstream /path/to/OmniGen2 --checkpoint /path/to/pinned-snapshot \
  --cases /path/to/reference-30.jsonl --reference-dir /new/run/reference \
  --device xpu --dtype bfloat16 --output-dir /new/run/parity
```

`--smoke --steps 2` permits a small manifest. Success records a completed smoke,
while the P0/P1 gate remains blocked. `--parity-mode replay` tests saved conditioning
only. Full-pipeline mode passes original native inputs through PRISM's typed image
adapter and unchanged reference conditioner. Neither mode claims that a newly
trained PRISM/Qwen conditioner must numerically equal the original conditioner.
Full-pipeline reference mode validates the typed `ImageDecoder` boundary; it does
not exercise PRISM's encoders/Qwen/new connector or establish the separate
checkpoint-reload and batch-invariance acceptance checks.

Generation runs keep resolved configuration/cases, checkpoint hashes, available
tensor traces and images, failures, a report, and a gate manifest. Preflight and
blocked runs retain configuration, reports and errors without inventing outputs.
Comparison is exact by default;
any tolerances must be supplied in a preregistered file. Global and per-call RNGs
are reset, since the native source-image VAE also draws from global RNG state.
P0/P1 comparisons require matching device identity, PyTorch version, dtype and
kernel policy. Runs record peak allocated and reserved accelerator memory;
cross-device or CUDA/XPU kernel equivalence requires a separate experiment.

For Aurora use `tools/launch_aurora_image_smoke.py`: first `--dry-run`, inspect its
PBS/worker scripts, then `--submit`. Supply `--repo`, `--venv`, `--upstream`,
`--checkpoint`, `--cases`, `--output-dir`, and `--job-dir`. It requests one node,
one process on one XPU tile, two denoising steps and ten minutes by default. It
loads `frameworks/2025.3.1` before activating the selected environment, and never
uses another allocation's nodefile. It does not install or download dependencies.
`--with-parity` adds a second smoke through PRISM's reference adapter in the same
allocation and walltime, writing to `<output-dir>-parity`.
Submission first runs metadata/file-integrity preflight on the login node; a failed
preflight does not submit a PBS job.

If exact comparison fails, `--repeatability-reference /path/to/completed/reference`
selects a bounded diagnostic instead of `--with-parity`. Within each process it
loads once and runs native N0, native N1, adapter A0, then native N2 using the
original archived initial latents and seed. A second fresh process repeats that
sequence. Exact within-process and cross-process comparisons distinguish native
repeatability from adapter-specific or process-dependent effects; they do not
automatically establish a root cause, choose looser tolerances, or pass P0/P1.

## Connector training and restoration

The selected Aurora parent is now PRISM-Harness's fully trained **Qwen3-8B +
SigLIP2-base/224 + PRISM image projector**, checkpoint `step_20000`, not the 0.6B
template below. Use
[`qwen3_8b_prism_harness_omnigen2.json`](../../src/conf/image_generation/qwen3_8b_prism_harness_omnigen2.json)
with its [exact checkpoint and provenance](../reports/2026-09-21-prism-harness-image-parent.md).
Its new connector maps 4096-dimensional PRISM states to OmniGen2's 2048-dimensional
conditioner interface. The loader strictly restores the saved parent tensor
dtypes, vocabulary, and tied weights; no encoder/projector/backbone weight may be
missing. New connector weights alone may start randomly.

`tools/validate_prism_image_parent.py` benchmarks parent restoration, text routes,
padding, and conditioning sensitivity without running the image sampler.
`tools/smoke_prism_image_training.py` permits at most four real-generator
optimization steps for engineering diagnostics and saves an unqualified artifact.
Launch either through `tools/launch_aurora_prism_image.py`, review `--dry-run`, then
use `--submit`. Neither diagnostic replaces the full training evidence below.

Export the aligned parent's exact `ModelConfig` as JSON, adding image decoder
configuration and its immutable generator revision. Do not silently change the
parent's input projector or tokenizer settings. The loader permits only the new
image connector weights to be missing from that parent checkpoint.
[`qwen3_0_6b_connector.json`](../../src/conf/image_generation/qwen3_0_6b_connector.json)
is a configuration template, not evidence of an aligned parent. The small
[`smoke.jsonl`](../../examples/image_decoder/smoke.jsonl) can exercise T2I loading and
sampling; it does not cover editing or multiple references.

```bash
python tools/train_image_decoder.py \
  --model-config /path/to/model-config.json --checkpoint /path/to/aligned-parent.pt \
  --tokenizer /path/to/tokenizer --source-processor /path/to/processor \
  --manifest /path/to/train.jsonl --reference-report /run/reference/manifest.json \
  --gate-report /run/parity/manifest.json --alignment-report /run/alignment.json \
  --acceptance-report /run/p1-acceptance.json \
  --steps 20 --batch-size 1 --device xpu --decoder-dtype bfloat16 \
  --output-dir /new/run/connector-smoke
```

The alignment report binds checkpoint/config/tokenizer/processor hashes and measured
metrics to the selected parent. The runner verifies the actual generator hash
against passed full P0/reference-parity artifacts and the separate P1 acceptance
evidence, freezes every module except the image connector,
checks finite nonzero gradients, fingerprints frozen parameters/buffers before and
after training, and saves optimizer/RNG/provenance plus connector weights. It never
marks P2 passed merely because a few optimization steps completed. Tiny-set learning,
held-out improvement, and ablations still need measured experiments.

Reference-adapter numerical parity leaves full P1 acceptance unproven. The companion
acceptance report uses `schema_version: 1`, `stage: "P1_acceptance"`,
`evidence_kind: "real_checkpoint"`, and `status: "passed"`, with
`parity_manifest_sha256`, `parent_checkpoint_sha256`, `model_config_sha256`, and
`reference_checkpoint_sha256` binding it to the exact experiment. Its `checks`
must include `checkpoint_reload`, `padding_batch_invariance`,
`text_checkpoint_regression`, and `target_free_unified_transformer`. Each check
needs passed real-checkpoint evidence, finite numeric `metrics`, and existing
`artifacts: [{"path": "relative/to/report", "sha256": "..."}]`. These are measured
regression results to supply after running the checks, not flags to fill in to
unlock training. The runner verifies identities and artifact bytes before loading
the training model; fixtures, smokes, boolean-only and missing evidence are rejected.

`src.decoders.loading.load_image_connector()` restores a connector after verifying
its PRISM-parent and generator identities. Its state must match exactly the image
connector; external frozen generator tensors are not duplicated in the artifact.
Qualification scoring uses grouped paired bootstrap comparisons:

```bash
python tools/validate_image_decoder.py --evaluate /path/to/scores.json \
  --protocol /path/to/protocol.json --cases /path/to/sealed-600.jsonl \
  --output-dir /new/run/qualification
```

The protocol needs `protocol_id`, `registered_at`, `evaluator_revision`,
`max_statistical_looks: 1`, at least two `training_seeds`, per-task
`necessary_ablations`, and `auxiliary_gates` containing editing `preservation` and
multi-reference `identity` thresholds. Cases need both `natural` and `procedural`
strata in every task. The score export has matching `evaluator_revision` and
`records` with `case_id`, sampling `seed`, `training_seed`, `variant`, and `scores`
(finite values in `[0,1]`, including `success` and the applicable auxiliary scores).
Every case/seed/training-seed combination needs `full`, `reference`, `native_only`
and all registered ablation variants. Failed generation gets zero success;
missing rows leave the gate unproven. Externally supplied scores need their own
provenance audit. Scoring is not an automatic claim of image quality.

The full staged criteria were set out in the 2026-09-21 image-first validation
plan, since retired; see `git log`. The execution record is in
[the implementation and validation report](../reports/2026-09-21-image-decoder-implementation.md).

## Bounded connector overfit experiment

`tools/overfit_prism_image_connector.py` is a separately authorized diagnostic for
up to 16 training pairs and 8 validation pairs, at most 1,000 total optimizer
updates and 256-pixel outputs. The initial protocol is 500 updates, AdamW `1e-4`,
FP32 connector weights, frozen BF16 parent/generator, and 50 denoising steps.
It does not modify `train_image_decoder.py` acceptance requirements or the
four-step smoke budget. Its checkpoints carry
`real_checkpoint_connector_overfit_diagnostic` and `qualification=unqualified`;
the production connector loader rejects them.

Prepare original caption-image pairs from existing WebDataset shards with
`tools/prepare_prism_image_overfit_data.py`. Selection preserves original captions
and selected member bytes, records provenance, and rejects shared source groups,
identical decoded images, and detected near-duplicates across train/validation.
This is a connector holdout, not a claim about the parent's training membership.

The runner computes per-example fixed-seed flow losses over both splits. With
`--conditioning-ablation`, it also replaces each T2I caption with a distinct
same-split caption while retaining the target and resetting the complete RNG;
the shuffled-minus-correct loss gap measures conditioning sensitivity. It
compares native, untrained-connector, and trained-connector images using replayed
native initial latents. Default galleries contain two examples per split. Only
caption/source inputs reach generation; target pixels are supervision only.
Checkpoints include optimizer/RNG state and strictly bind data, parent, generator,
code, and numerical policy before resume. A short timing run can resume to a
500-step total in a new output directory.

Use `tools/launch_aurora_image_experiment.py` with an explicit JSON argument list,
review `--dry-run`, then `--submit`. It allocates one node, one process and one XPU
tile for 5–60 minutes; model execution occurs only inside PBS. Numerical checks
use `--deterministic --attention-backend math`. Changes from an archived precision
or attention policy require explicit exploratory labeling, never relaxed parity
tolerances. Record actual execution/results separately from these capabilities.

The [2026-09-21 execution report](../reports/2026-09-21-prism-connector-overfit.md)
records 500 completed connector-only updates, unchanged frozen weights, partial
training-image content recovery, and failures on the sampled held-out prompts.
Its lower flow loss does not qualify the connector for production or scientific
decoding.
The [DOCCI joint fine-tuning workflow](image_decoder/docci_diffusion.md) adds an
explicit full-diffusion training option from a completed connector checkpoint,
with FP32 CPU master weights and optimizer state. Its runner preserves the frozen
PRISM/VAE/native-conditioner boundary and records both trainable groups separately.
The committed [initial generation results](../assets/image_generation/2026-09-21-initial/README.md)
include matched sample galleries, raw generated images, training/validation plots,
measured losses and provenance for the smoke and 500-update diagnostic.

## Public data for expanded caption conditioning

The [Public image-caption data collection for PRISM survey](../reports/2026-09-21-public-image-caption-data-survey.md) records a
candidate pool of DOCCI train, BLIP3-o 60K and DenseFusion-4V: approximately
186,400 pairs before cross-source deduplication, with 211.990 GB of published
image and annotation downloads. Only metadata/manifests have been collected and
verified on Aurora; no images or expanded training run are included in that step.
Caption-only conditioning and target-image supervision remain separate. The
audit includes immutable download URLs, byte totals, count provenance, access
limitations, and the verified delivery receipt. Validate image-caption joins
and held-out splits before using this pool for further connector training.

# DOCCI data for PRISM image generation

This workflow prepares the public [Google DOCCI dataset](https://huggingface.co/datasets/google/docci)
for caption-conditioned image generation with the trained PRISM Qwen3-1.7B/SigLIP2
parent and pretrained OmniGen2. The initial runner trains the new connector only.
A separate [joint fine-tuning runner](docci_diffusion.md) updates the connector and
full diffusion transformer from that saved starting point. PRISM, the native
OmniGen2 conditioner, and VAE remain frozen in both stages. Dataset
preparation and successful loader tests do not establish completed training or
image-generation quality; those require the recorded compute runs.

The caption conditions PRISM's language-model hidden states. The corresponding
image supplies the VAE/flow-matching training target. The target is never supplied
to PRISM's vision encoder, and no language-head captioning loss is added.

## Download and attribution

`tools/download_docci.py` downloads the three official publisher-hosted objects
referenced by the Hugging Face loader. It does not execute a remote dataset script.
GCS object generations and expected byte sizes are pinned in the tool; resumable
transfers use `.partial` files and validate the returned generation/range before
appending. `download.json` records object URLs, generations, byte sizes, and SHA256
digests. Existing complete files must have the expected size and are hashed again.

Run CPU-only data preparation from the PRISM checkout on an Aurora login node,
using Python 3.10 or later with Pillow. For example:

```bash
export PRISM_DOCCI_ROOT=/lus/flare/projects/ModCon/sandeep/prism-docci-qwen3-1p7b-20260921
export PRISM_PYTHON=/lus/flare/projects/ModCon/sandeep/prism-image-smoke-20260921/.venv-image/bin/python

"$PRISM_PYTHON" tools/download_docci.py \
  --output-dir "$PRISM_DOCCI_ROOT/source"
```

The three source objects total 7,766,706,359 bytes. Retain room for the source
archive, the converted uncompressed tar shards, and subsequent model outputs.
The image archive is 7,592,938,768 bytes. Its pinned generation is
`1714249456173037`; the downloaded archive used for this preparation has SHA256
`c1b1aee00856757d71cfe2cc6ab641089284276a791bfb82146b4742c59a0a1d`.

DOCCI is distributed under **CC BY 4.0**. Preserve attribution to Google DOCCI,
the [dataset project](https://google.github.io/docci/), and the
[license](https://creativecommons.org/licenses/by/4.0/) when redistributing the
data or derivatives. The generated sample metadata and conversion audit retain
the dataset identity, license, and source URLs. Image bytes and caption strings
are unchanged in the shards; training-time resize/normalization is a separate
transformation. This bundle uses DOCCI, not the separate DOCCI-AAR image collection.

## Convert and validate

Use a **new output directory**. Conversion refuses to overwrite an existing one.

```bash
"$PRISM_PYTHON" tools/prepare_docci_webdataset.py \
  --descriptions "$PRISM_DOCCI_ROOT/source/docci_descriptions.jsonlines" \
  --images-archive "$PRISM_DOCCI_ROOT/source/docci_images.tar.gz" \
  --metadata "$PRISM_DOCCI_ROOT/source/docci_metadata.jsonlines" \
  --output "$PRISM_DOCCI_ROOT/webdataset" \
  --shard-mb 256 \
  --shard-records 256
```

The converter streams the archive without extracting its paths. It verifies the
complete description/image join, decodes every JPEG, preserves full captions,
checks exact decoded-pixel duplicates across splits, and computes provenance
hashes. Unsafe paths, unsupported member types, duplicate IDs, unexpected images,
missing described images, corrupt images, or cross-split exact-pixel duplicates
fail conversion. There is no silent exclusion or short-caption filter.

All shards are written to a temporary sibling directory. The final directory is
published only when the entire conversion passes. An in-progress shard directory
is not a usable training bundle. Full CPU image decoding can take several minutes;
the source download completing does not mean conversion has completed.

| Official split | Output index | Records | Role |
| --- | --- | ---: | --- |
| `train` | `train.jsonl` | 9,647 | Optimizer updates |
| `qual_dev` | `validation.jsonl` | 100 | Validation and qualitative development |
| `test` | `test.jsonl` | 5,000 | Sealed for this training workflow |
| `qual_test` | `qual_test.jsonl` | 100 | Sealed for this training workflow |

The CLI enforces these exact official counts. Conversion decodes all splits for
integrity and leakage checks, but model training/evaluation uses only the two
indexes explicitly selected by the training runner.

`--metadata` is optional but recommended. The converter joins every metadata row
to its description and retains `cluster_id`, `entity_tags`, `image_width`, and
`image_height`. Large auxiliary annotation fields are not copied to every shard.
Related clusters and entity tags crossing official splits are reported in
`conversion.json`; the official splits are preserved. These related-image groups
are distinct from the strict identity/pixel `group_ids`. Related or contrasting
images are not automatically discarded, and the audit does **not** certify
subject-disjoint or near-duplicate-free validation. Within-split exact duplicates
are retained and listed; cross-split exact duplicates fail conversion.

## Output format and indexed loading

```text
webdataset/
  conversion.json
  train.jsonl
  validation.jsonl
  test.jsonl
  qual_test.jsonl
  shards/
    train/docci-000000.tar
    validation/docci-000000.tar
    test/docci-000000.tar
    qual_test/docci-000000.tar
    ...
```

Each tar uses ordinary WebDataset members:

```text
train_00000.jpg   original image bytes
train_00000.txt   original complete description, UTF-8
train_00000.json  source, split, task, hashes, dimensions, grouping metadata
```

Each JSONL index record includes `id`, `task: "t2i"`, `prompt`, `source_images: []`,
`split`, `source_split`, `group_ids`, a relative `shard` path, `member`,
`header_offset`, `data_offset`, `size`, `image_sha256`, `pixel_sha256`, `width`, and
`height`. Source provenance and optional related-group metadata are also retained.
The shard offset locates the target image; a standalone `target_image` file path
is not required. `conversion.json` records all shard/index hashes, counts,
validation results, source hashes, and a bundle `data_fingerprint`.

```python
from src.data.image_generation_webdataset import ImageGenerationWebDataset

train = ImageGenerationWebDataset(
    "/path/to/webdataset/train.jsonl",
    target_size=(256, 256),
    split="train",
)
validation = ImageGenerationWebDataset(
    "/path/to/webdataset/validation.jsonl",
    target_size=(256, 256),
    split="validation",
)
assert train.data_fingerprint == validation.data_fingerprint
sample = train[0]
```

The map-style loader supports deterministic shuffled sampling and resumable
training. It checks the completed audit, all index hashes, split ownership, shard
sizes, and locator bounds during construction. On access it checks the actual tar
header and image-byte hash before decoding. It does not reread and hash every
entire shard at startup. Targets receive EXIF orientation correction, RGB
conversion, explicit square resizing, and CHW float32 normalization to `[-1, 1]`.
The items use the existing `ImageGenerationCollator` contract; source/reference
image lists remain empty. Full captions are passed to the collator, which refuses
silent truncation. Square resizing is the initial pilot preprocessing, not an
aspect-ratio-bucketed production pipeline.

## Connector training on Aurora

Run model loading, sampling, and optimization only inside an allocated Aurora
compute job with the pinned image-decoder environment and OmniGen2 source/weights
configured as described in [Image output through PRISM](../image_decoder.md).
Data preparation above requires no GPU allocation. The indexed runner is a
single-device pilot; it does not imply 16-node training just because the parent
checkpoint was produced by a 16-node run.

The selected parent checkpoint is:

```text
/lus/flare/projects/AuroraGPT/sww/prism_outputs/CODEX_QWEN3_1P7B_SIGLIP_CLEAN_GSHUFV1_INVSQRTLR_TRAIN25K_BS4_HSDP_FULLSHARD_16N_R1/checkpoints/step_19750/model.safetensors
```

After the native repeatability check and short real-checkpoint smoke pass, an
example 500-step pilot command is:

```bash
"$PRISM_PYTHON" tools/train_prism_image_connector.py \
  --model-config src/conf/image_generation/qwen3_1_7b_prism_harness_omnigen2.json \
  --checkpoint "$PRISM_PARENT_CHECKPOINT" \
  --tokenizer "$PRISM_TOKENIZER" \
  --source-processor "$PRISM_SOURCE_PROCESSOR" \
  --train-index "$PRISM_DOCCI_ROOT/webdataset/train.jsonl" \
  --validation-index "$PRISM_DOCCI_ROOT/webdataset/validation.jsonl" \
  --repeatability-report "$PRISM_REPEATABILITY_REPORT" \
  --output-dir "$PRISM_DOCCI_ROOT/runs/pilot-500-01" \
  --steps 500 --batch-size 1 --gradient-accumulation 1 \
  --height 256 --width 256 --learning-rate 1e-4 \
  --checkpoint-every 100 --eval-every 100 --sample-every 500 \
  --probe-count 32 --final-validation-count 0 \
  --sample-count 2 --sampling-steps 50 --native-baseline \
  --expected-parent-tensors 526 \
  --device xpu --dtype bfloat16 --deterministic --attention-backend math
```

Set the checkpoint/tokenizer/processor variables to the verified local parent
assets, and `PRISM_REPEATABILITY_REPORT` to the matching passed diagnostic
manifest. The output directory must be new, including on resume. `--steps` is the
total optimizer-step target, not an additional-step count. Resume with
`--resume /path/to/connector-checkpoint.pt`, preserving the recorded run protocol
and selecting a new output directory.

Only `train.jsonl` feeds optimizer updates. Periodic loss probes and generated
samples use designated train/validation cases; `--final-validation-count 0`
evaluates all 100 validation records. Native OmniGen2 and wrong-caption controls
help distinguish connector conditioning from the frozen generator's image prior.
Training loss alone is insufficient evidence of useful caption-conditioned
generation. Record the source revision, `download.json`, `conversion.json`, run
arguments, restored parent tensor count, connector checkpoints, losses, sample
galleries, and frozen-weight checks alongside each run.

Generate loss plots after steps and evaluations have been written:

```bash
python tools/plot_prism_image_connector.py \
  --run-dir "$PRISM_DOCCI_ROOT/runs/pilot-500-01" \
  --output-dir "$PRISM_DOCCI_ROOT/runs/pilot-500-01/plots"
```

The plotter needs Matplotlib and no model weights or accelerator. It writes PNG,
PDF, and auditable JSON series. Stochastic optimizer losses have a separate panel
from fixed-noise train/validation losses. The final all-validation result gets a
separate marker, while the original 32-example validation curve keeps the same
IDs and seeds. For an exact resume, pass each run directory in chronological order
with repeated `--run-dir` options. Reports from different protocols cannot be
combined into one curve.

# Public image-caption data collection for PRISM

Companion [manifests and provenance](../data/public_image_sources/20260921/README.md).

Audited 2026-09-21. User scope: **metadata/manifests only**. No image payloads or
image archives were downloaded, no training jobs were executed, and no dataset
access agreements were accepted. This collection surveys the nine public source families
named in OmniGen2's original paper; it does not reproduce its undisclosed 15M
mixture or resolve the paper's 15M/150M discrepancy.

## Selected pool and measured download sizes

| Source | Candidate training records | Published image archives + annotations | Count evidence |
|---|---:|---:|---|
| [DOCCI, train](https://huggingface.co/datasets/google/docci) | **9,647** | **7.604 GB** | Parsed all 14,847 original caption records; train image IDs/filenames unique; excluded 5,200 held-out records |
| [BLIP3-o 60K](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) | **approximately 60,000** | **108.169 GB** | Publisher-advertised image count; 11 hosted tar files plus standalone prompt text |
| [DenseFusion-4V-100K](https://huggingface.co/datasets/BAAI/DenseFusion-1M) | **116,761** | **96.218 GB** | Downloaded caption metadata: unique image IDs, URLs and paths; zero blank captions |
| **Selected pool** | **approximately 186,408 before deduplication** | **211.990 GB** | A planning sum, not a verified cross-source unique-image count |

The smaller **DOCCI + BLIP3-o starter** contains approximately **69,647** candidate
training pairs and requires **115.772 GB** of published downloads. DOCCI's image
archive contains every official split; only its 9,647 training records enter the
selected manifest. Its held-out images must not be used for optimization.

Sizes are decimal GB (1 GB = 1,000,000,000 bytes), measured from upstream file
inventories or HTTP Content-Length, not inferred from dataset names. These are
source download sizes, not extracted storage. DenseFusion's figure includes its
original 173.641 MB JSONL; the 96.356 MB metadata-only Parquet downloaded for this
audit is an alternative representation, not another set of training samples.

For capacity planning only, encoding retained training images as 512-pixel JPEGs
at **100–300 kB per image** would give approximately **7–21 GB** for the starter
or **19–56 GB** for the extended pool. This is an explicit assumption, not a
measured processed-data size or a guarantee of retained quality. Originals,
temporary extraction, shards, captions and cached features add further storage.

## What has actually been collected

- Original DOCCI captions and image filenames; `docci_train_records.jsonl` has
  exactly 9,647 records with source archive pointers. Images are not materialized.
- All 11 BLIP3-o standalone prompt files. They contain 57,561 nonempty lines and
  57,121 distinct strings, without a verified image-to-prompt mapping. The public
  viewer's 7,103 text rows describe one prompt file, **not** the image collection.
- DenseFusion caption/URL metadata: all 116,761 rows were read, with unique image
  IDs, URLs and paths, and no blank fields. All paths map to the 48 published image
  archives. Captions have 22 repeated strings; image contents were not downloaded
  or deduplicated. Full verification and a compact training-record index are in
  the full bundle under `dense_sources/`; the verification summary is also tracked
  as [the DenseFusion verification summary](../data/public_image_sources/20260921/densefusion4v_content_verification.json). Across DOCCI train and DenseFusion, metadata describes
  **126,408 records with unique per-source image identifiers**.
- Revision-pinned file inventories, byte sizes, available upstream hashes,
  dataset cards, split metadata, access findings and primary-source links across
  all surveyed public source families.
- `download_manifest.jsonl`: **73 source objects** for the selected future image
  collection, including 60 image archives and 13 annotation/prompt files. This is
  an inventory, not authorization to download images or a runnable training set.
- `selected_subset.json`: machine-readable counts, byte totals, assumptions and
  scope. `source_bundle_checksums.json` records hashes for the complete external metadata bundle, not just these tracked summaries.
- The complete metadata bundle contains `collection.json` and `verify_bundle.py`.
  Run `python3 verify_bundle.py` inside that bundle to verify every hashed file.
- Git contains compact reports, the selected download manifest, scope, checksum
  inventory and delivery receipt. Caption rows, source snapshots, images, model
  weights and generated experiment artifacts remain outside Git.

Local directory: `outputs/public_image_data_audit/20260921/`.
Aurora metadata destination:
`/lus/flare/projects/ModCon/sandeep/prism-public-image-data-20260921/metadata/`.
The transfer completed on 2026-09-21: 134 metadata files, 149,693,911 bytes.
The compressed transfer archive and all 133 file hashes were verified on Aurora.
See [the delivery receipt](../data/public_image_sources/20260921/delivery-receipt.json).

## Wider public-source inventory

These are alternatives or later extensions. **Do not add the rows in this table
to obtain unique images:** source pools and recaptioned variants overlap.

| Source/subset | Published or measured record count | Payload size available from metadata | Practical note |
|---|---:|---:|---|
| DOCCI train | 9,647 verified unique image filenames | 7.604 GB including all-split image archive and captions | Selected; retain official held-out splits |
| BLIP3-o 60K | approximately 60K advertised | 108.169 GB image/text archives + standalone prompts | Selected synthetic instruction-generation data; image count awaits archive inspection |
| DenseFusion 4V | 116,761 caption records | 96.218 GB images + original captions | Selected; caption schema is image ID, caption, URL and image path |
| DenseFusion 1M | 1,058,790 caption records | 799.284 GB images + original captions | Larger alternative; 4V is drawn from the parent pool, so counts are not additive |
| ALLaVA Caption-LAION | 468,670 reported on current Hub card | 93.373 GB images + captions | Additional natural-image candidate; unique IDs and overlap not yet audited |
| ALLaVA Caption-VFLAN | 194,976 reported on current Hub card | 36.695 GB images + captions | Requires separate Vision-FLAN image archive; exclude text-only/instruction variants |
| ShareGPT4V / ShareGPT4V-PT | 102,025 / 1,246,901 viewer rows | 0.134 / 1.492 GB **captions only** | Images are separate; known COCO + LLaVA source archives total 46.693 GB, incomplete image total |
| JourneyDB original train | 4,189,737 labeled images; 4,453,193 total train images, reported | 3.112 TB images + one annotation version | Gated, custom usage terms; multiple images per prompt; defer |
| BLIP3-o JourneyDB repack | approximately 4M advertised; 4.277M viewer estimate is partial | 3.132 TB image/text tar shards | Same source family, not additional images; original terms and split lineage require review |
| BLIP3-o long / short pretraining | 27M / 5M advertised | 1.374 / 0.828 TB image/text tar shards | Larger alternatives; counts and overlap not independently verified |
| Recap-DataComp | 940,890,257 current train rows | 527.440 GB **caption/URL Parquet**, including 1,000-row preview | Image bytes absent; web retrieval required. Duplicate Hub configs reference the same data |
| SAM-LLaVA | 11,529,794 current caption rows | 2.236 GB **compressed captions only** | Separate SA-1B images/access and filename join required; image size unknown |
| LAION-Aesthetic | Historical approximately 120M; current exact count unavailable | 20.683 GB **metadata only** across three language subsets | Current repositories gated; image bytes absent |

All approximate/count-provenance/access details and download URLs are retained in:

- [Small sources: DOCCI, ShareGPT4V and BLIP3-o](../data/public_image_sources/20260921/small_sources.summary.json)
- [Dense sources: DenseFusion, ALLaVA and JourneyDB](../data/public_image_sources/20260921/dense_sources.summary.json)
- [Large sources: Recap, SAM-LLaVA and LAION](../data/public_image_sources/20260921/large_sources.summary.json)

DOCCI's card declares CC BY 4.0, BLIP3-o declares Apache-2.0, and DenseFusion's
Hub tag is CC BY 4.0 with separate source-image conditions. These declarations
are recorded as source metadata; they do not establish rights to every underlying
image. ShareGPT4V and ALLaVA declare noncommercial datasets. No gate was bypassed.

## Validation before future training

Before treating the selected pool as training-ready, verify image/annotation
joins, decode files, remove exact/perceptual duplicates, preserve source groups
across train/validation, and sample-check caption accuracy. DOCCI split filtering
is already performed in metadata. BLIP3-o image counts and mappings remain
publisher-reported until its tar contents are inspected. No visual quality audit
or image-level cross-source deduplication was possible in this metadata-only task.

The source choices favor bounded downloads with hosted images and usable
descriptive captions. They are a proposed PRISM alignment pool, not a measured
claim that this data volume will solve the connector's current generation errors.

## How these pairs train the connector

For text-to-image alignment, each caption is the conditioning input and its image
is the training target. The caption passes through the frozen trained PRISM
language model, the trainable 4096-to-2048 connector, and the frozen OmniGen2
generator. The target image enters OmniGen2's VAE and flow-matching loss path;
it does not enter PRISM's vision encoder or the sampling input. At inference,
only the caption and sampled initial noise are needed.

This collection prepares a candidate pool; it does not authorize or launch an
expanded training run. Image-caption joins and held-out splits must be validated
before training. The [connector experiment report](2026-09-21-prism-connector-overfit.md)
records the existing 16/8-pair baseline and its unresolved conditioning limitations.

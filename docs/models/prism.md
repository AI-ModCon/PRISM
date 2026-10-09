# PRISM model card

A model card for PRISM itself, rather than for one of the third-party backbones
the other pages in this folder describe. It is written against the code in this
repository, and it is deliberately explicit about what does **not** exist:
PRISM publishes no trained weights, and no benchmark score in this repository is
attributable to a PRISM model.

Source: [`src/model.py`](../../src/model.py),
[`src/config.py`](../../src/config.py),
[`src/conf/model/`](../../src/conf/model),
[`src/encoders/`](../../src/encoders), [`src/eval/`](../../src/eval).

## At a glance

| Property | Value |
|---|---|
| What it is | A training and evaluation framework for multimodal transformers over scientific data |
| Released weights | **None.** No PRISM checkpoint is published anywhere |
| Version | 0.1.0 in [`pyproject.toml`](../../pyproject.toml); no git tag, no release |
| Distribution name | `prism-mm` (the PyPI name `prism` is taken); import package and CLI stay `prism` |
| Software license | Apache-2.0. It covers the code only — no weights or data are licensed by this project |
| Copyright | ModCon, 2026 — **temporary**, pending confirmation across the contributing institutions (see [`NOTICE`](../../NOTICE)). Portions were produced under U.S. Government contract DE-AC02-06CH11357 |
| Maturity | `Development Status :: 3 - Alpha` |
| Most exercised platform | Intel Max Series GPUs on Aurora. Polaris, Perlmutter and single-node CUDA are also supported; see [platforms/](../platforms) |
| Shipped model configs | 26, under [`src/conf/model/`](../../src/conf/model) |
| Citation | [`CITATION.cff`](../../CITATION.cff) |

## What this card describes, and what it cannot

A conventional model card describes one trained artifact: its weights, the data
it saw, and the scores it reached. PRISM has no such artifact. Nothing in this
repository produces a distributable checkpoint, no workflow uploads one, and
[`CHANGELOG.md`](../../CHANGELOG.md) records that nothing has shipped in a
tagged release. Every checkpoint path in the documentation is one the reader is
expected to produce by training.

So this is a **framework card**. It describes the model PRISM *builds* — what
architecture a shipped config instantiates, what upstream weights it loads, what
data it is pointed at, and what can and cannot be said about quality. Anyone who
trains a PRISM model and releases it needs to write a card for that checkpoint;
this page is the template and the starting set of inherited caveats.

### Intended use

Research. PRISM exists to study how a pretrained language backbone can be
extended with per-modality encoders and trainable projectors so that scientific
data — text, images, time series, DNA sequences, and, experimentally, graphs,
tables, and geometry — reaches one token stream, and to measure how that
training scales on DOE leadership-class hardware.

### Out-of-scope use

- **Any production or decision-making use.** Alpha-stage research code with no
  released weights and no evaluated model.
- **Any use implying a validated PRISM model exists.** None does.
- **Any scientific claim resting on PRISM outputs** without the user's own
  evaluation. See [Evaluation](#evaluation) — the repository contains no
  benchmark score for any PRISM configuration.
- **Clinical, diagnostic, or genomic interpretation.** The DNA path loads a
  third-party nucleotide model and has been exercised for training throughput
  only.

## Architecture

PRISM assembles a model in four stages: per-modality encoders produce features,
projectors map those features to the backbone's hidden dimension, a transformer
trunk consumes the merged token stream, and output decoders read the trunk's
hidden states. [`api/model.md`](../api/model.md) documents the stages in detail.

The trunk has two mutually exclusive implementations, and which one is built
depends on a single config field:

| | Trunk | Built when |
|---|---|---|
| **Path A** | A Hugging Face `AutoModelForCausalLM` | `backbone_id` is set |
| **Path B** | PRISM's own `TransformerBlock` stack, whose feed-forward is a top-2-of-8 `MoELayer` | `backbone_id` is unset |

**All 26 shipped configs set `backbone_id`, so Path B is exercised by none of
them.** `src/model.py` builds the MoE stack only under `if self.backbone is
None`. MoE upcycling from a dense checkpoint — which
[`index.md`](../index.md) still advertises as a feature — is a retired stub that
writes no parameter and has no remaining caller under `src/`. Treat the MoE
trunk as an unexercised alternate path, not as PRISM's architecture.

23 of the 26 configs also set `freeze_backbone: true`, and both
`freeze_backbone` and `freeze_encoders` default to `True` in `ModelConfig`. What
PRISM trains, in shipped practice, is **projectors and connectors over a frozen
pretrained backbone and frozen pretrained encoders**.

### Modalities and encoders

[`src/modalities.py`](../../src/modalities.py) declares seven modalities;
[`src/encoders/__init__.py`](../../src/encoders/__init__.py) exports eight
encoder classes. The counts differ because `CrystalGraphTokenEncoder` is not
built by `UnifiedTransformer` — the materials regressor installs it into the
`graph` slot itself.

| Modality | Encoder | Default upstream model | Shipped configs enabling it |
|---|---|---|---|
| Text | `TextEncoder` | `HuggingFaceTB/SmolLM2-360M-Instruct` | 26 of 26 |
| Time series | `TimeSeriesEncoder` | backend-dependent; `ts_projector` defaults to `linear` (no weights) | 14 of 26 |
| Image | `ImageEncoder` | `google/siglip2-base-patch16-224` | 9 of 26 |
| DNA | `DNAEncoder` | `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` | 2 of 26 |
| Graph | `GraphEncoder` | none — GraphMAE2-style, trained from scratch | 0 of 26 |
| Table | `TableEncoder` | `google/tapas-base` | 0 of 26 |
| Geometry | `GeometryEncoder` | `polymathic-ai/walrus` | 0 of 26 |
| (materials) | `CrystalGraphTokenEncoder` | none — periodic message passing from scratch | not built by the model |

Two notes that catch readers out. `TextEncoder` is constructed **only** on the
backbone-less path; with a backbone, the backbone's own embedding layer embeds
text and SmolLM2 is never loaded. And the time-series default disagrees between
layers — the encoder class defaults to `moirai`, but `UnifiedTransformer` always
passes `config.ts_projector`, so the effective default backend is `linear`.

Graph, table, and geometry are **Experimental** in the sense
[`README.md`](../../README.md) defines: the code path exists and has been run,
but no shipped config enables it, no test instantiates the real encoder, and
its datasets are marked `skip`. The geometry sweep numbers were measured on
synthetic data with no real corpus staged.

### Output decoders

[`src/decoders/`](../../src/decoders) registers seven keys — `text`, `action`,
`regression`, `time_series`, `geometry`, `graph`, `image` — and the default is
text only. There is no `table` decoder, so table is an input-only modality.

### Backbones actually exercised

The 26 shipped configs draw on two public families plus one internal
checkpoint: AllenAI OLMo (12 configs: OLMo-1B-0724-hf, OLMo-7B-0724-hf,
Olmo-3-7B-Instruct, Olmo-3-7B-Think), Qwen3 (13 configs, 0.6B through 32B), and
AuroraGPT-2B (1 config, which needs ALCF-internal access and ships a
placeholder path). Backbone loading is offline-only — `local_files_only=True`,
rank-0-first under `torch.distributed`, with a grow-only
`resize_token_embeddings`.

## Training data

PRISM trains on third-party corpora that it does not redistribute. The honest
summary is that **data provenance is recorded far less completely than code
provenance**, and a released model would need that gap closed first.

- [`src/data/datasets_config.json`](../../src/data/datasets_config.json)
  configures 57 datasets across four zones. **Not one carries a license,
  redistribution term, or terms-of-use field** — the schema has no such field.
  The same is true of every YAML under `src/conf/data/`.
- Only four of the 26 entries in `zone_a` are enabled; the rest, and all 29
  `zone_catalog` entries, are `skip: true`. `zone_catalog` is an aspirational
  inventory, and several of its entries are explicitly restricted or unavailable.
- The corpora behind the headline scaling runs are **not** the ones in
  `datasets_config.json` — they are WebDataset exports configured in
  `src/conf/data/daos_datasets.yaml`. For the largest of those there is no
  upstream identifier recorded anywhere: no Hub id, URL, or citation.
- Exactly one training dataset has its license stated in-tree: DOCCI
  (CC BY 4.0), used for the image-generation connector, with its attribution
  obligation written out and machine-recorded in the produced shards.
- Two enabled entries are labelled synthetic or proxy in their own name fields
  and should not be presented as real corpora.
- The one thorough licensing record in the repository —
  `docs/data/public_image_sources/` — covers image-caption sources that were
  **surveyed and not downloaded**. It is worth reading as the model of what a
  data statement should look like, and for its central warning: a permissive
  mirror tag does not clear the upstream terms, and a metadata license is not a
  license over the underlying images.

No PII, consent, copyright-clearance, or opt-out statement exists for any
corpus.

## Evaluation

**No benchmark score in this repository is attributable to a PRISM model
configuration.** No file pairs a config name with a task metric value. What
exists is evaluation *capability*, plus systems measurements.

[`src/eval/`](../../src/eval) registers 16 active tasks through
`EvaluatorRegistry` (a 17th, `vision_mme`, is deliberately commented out),
spanning text (MMLU, MMLU-Pro, GPQA, AIME-2025, IFEval), vision (VQAv2,
MathVista, MathVision, MMStar, MMMU), graph (ChEBI-20), time series (Time-MMD,
SciTS, Monash), table (Spider), and geometry (MatBench). Before citing any
number these produce, note:

- Every metric is an in-tree reimplementation. No standard harness
  (lm-eval-harness, HELM, VLMEvalKit, official scorers) is used, and several
  scorers are self-described proxies — VQAv2 is scored by substring
  containment, MMMU is restricted to one subject subset, and the Spider
  evaluator feeds a hardcoded dummy table and compares SQL by string equality.
- Two scorers fail toward flattering numbers: MatBench substitutes `0.0` for an
  unparsable prediction rather than dropping it, and the Monash evaluator
  returns `mae 0.0` when the dataset cannot be fetched.
- `tools/universal_evaluator.py` wires only 5 of the 16 tasks into its
  end-to-end path.

The numbers the repository *does* report are systems results — throughput,
memory, and scaling efficiency, under [`results/`](../results) and in README's
tested-configuration table. They describe how fast PRISM trains, not how well
any model performs. The one table of generation-quality metrics in
[`results/vlm_ablations.md`](../results/vlm_ablations.md) names informal
checkpoint nicknames with no config, no benchmark, and no released artifact
behind them; the image-generation pilot is machine-readably marked
`qualification: unqualified` with both quality gates unestablished.

## Limitations

An earlier revision of this card listed six stale claims elsewhere in the
documentation — a modality count, a config-count denominator, two claims about
MoE upcycling, a time-series encoder that no longer matched the code, and a
per-architecture backbone adapter. **All six have since been reconciled** in
the documentation pass that followed (#234, #235), and
`tests/test_docs_consistency.py` now derives the counts from `Modality` and the
`src/conf/model/` glob so they cannot drift again silently. The table is
removed rather than kept as history: a card that lists fixed problems as
current is itself a drift source — and the encoder row would now trip that very
test, which asserts no document names an encoder absent from `src/`.

Two gaps remain, and they are real. The repository contains **no measurement of
compute or energy cost** for any training run — throughput and memory are
reported, node-hours and kWh are not. And there is **no bias, fairness,
toxicity, dual-use, or societal-impact assessment**.
[`SECURITY.md`](../../SECURITY.md) routes model-behaviour concerns such as
hallucination and bias out of the vulnerability process and into ordinary
issues; that is a triage decision, not an assessment.

## Provenance and third-party components

[`NOTICE`](../../NOTICE) records the components PRISM **redistributes**: the
OLMo-derived interleaved tokenizers (Apache-2.0, Allen Institute for AI) and the
Walrus geometry submodule (MIT, Polymathic AI, consumed at a pinned commit and
not redistributed). Upstream weights that PRISM only *loads* — SigLIP2, TAPAS,
SmolLM2, Moirai, Nucleotide Transformer, Evo2, and the OLMo and Qwen3 backbones
— are governed by their own licenses on their own model pages, and this
repository asserts nothing about them.

## Citing PRISM

Use [`CITATION.cff`](../../CITATION.cff), which GitHub renders as a
"Cite this repository" entry. It deliberately carries no DOI, release date, or
`preferred-citation`, because none exists — see the comments in the file.

## See also

- [`index.md`](../index.md) — documentation index
- [`api/model.md`](../api/model.md) — `UnifiedTransformer` and `ModelConfig` in detail
- [`api/encoders.md`](../api/encoders.md) — the `ModalityEncoder` contract and all eight encoders
- [`evaluation/evaluation.md`](../evaluation/evaluation.md) — the evaluator registry and how to run it
- [`results/scaling_study.md`](../results/scaling_study.md) — the systems measurements this card refers to
- [`../CONTRIBUTING.md`](../../CONTRIBUTING.md) — including the policy on AI/LLM-assisted contributions

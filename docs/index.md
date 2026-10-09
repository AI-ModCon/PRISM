# PRISM: Poly-Reasoning Integrated Scientific Multimodal Modeling Framework

PRISM is a modular recipe for multimodal scientific modeling, organized around
representation, alignment, and fusion. Explicit **encoder**, **adapter**,
**language-backbone**, **readout**, and **decoder** interfaces separate domain
representation from shared reasoning and native generation.

Feature-level mid-fusion reuses specialist encoders while letting their
representations interact through a language backbone. Output readouts and
conditioning bridges connect the resulting states to task-specific generators
or prediction heads. Compatible encoders and backbones can be changed
independently, subject to preprocessing, feature, attention, and checkpoint
contracts.

The design hypothesis is that these stable boundaries permit domain
customization and controlled model improvement while retaining a common
training and evaluation workflow.

In the shipped configurations a pretrained causal LM is the trunk, and each
non-text modality — images, time series, DNA, geometry, graphs, tables — has
its own encoder and a projector that maps its features into the backbone's
embedding space, so one set of transformer weights sees every modality as
tokens.

For what each modality's support actually amounts to — which are exercised by
shipped configs and CI, and which are experimental — see the **modality
support status** table in the [README](../README.md#modality-support-status).
It is the single source for that; this page does not keep a second copy.

## Architecture overview

The five interfaces, in the order a sample moves through them:

| Stage | What it does | Where |
|---|---|---|
| **Encoder** | Converts a raw modality input into a feature sequence | [`src/encoders/`](../src/encoders/) |
| **Adapter** (projector) | Maps those features to the shared `d_model` | [`src/modules/projector.py`](../src/modules/projector.py) |
| **Language backbone** | Processes the merged sequence. Token harmonization concatenates the spans, applies RoPE, and builds the attention mask — non-text spans attend bidirectionally, text stays causal | [`src/model.py`](../src/model.py) |
| **Readout** | Selects the backbone states a decoder conditions on (`FinalStateReadout`, `PooledStateReadout`) | [`src/connectors/`](../src/connectors/) |
| **Conditioning bridge** | Maps those states into a generator's conditioning space (`IdentityBridge`, `LayerNormLinearBridge`) | [`src/connectors/`](../src/connectors/) |
| **Decoder** | Generates the output — text, image, time series, geometry, graph, or a regression head | [`src/decoders/`](../src/decoders/) |

Readout and bridge are independently parameterized per route, so a decoder can be
attached to a different backbone without changing either side. See
[api/model.md](api/model.md) for the decoder registry and
[api/modules.md](api/modules.md) for the reusable blocks.

## Training stages

Stages are selected with Hydra `training=...`. See
[training/training.md](training/training.md) for the full table and example
invocations.

| Stage | What it does |
|---|---|
| Encoder alignment | Trains the projector against a frozen encoder and frozen backbone, so modality features land in the backbone's embedding space. Captioning-style data. |
| Encoder alignment, unfrozen | The same data with encoder and/or backbone weights unfrozen, for end-to-end training with differential learning rates. |
| SFT | Supervised fine-tuning on instruction data, typically with LoRA adapters on the backbone. |
| RL | Reinforcement learning with GRPO: samples several completions per prompt, scores them with reward functions, and updates on group-relative advantages. |

Older documents call these Zone A / B / C / D. That vocabulary is retired —
the Hydra config keys keep their original names for compatibility, but the
stages are referred to by what they do.

---

## Documentation Index

Documentation is grouped by task. If you are new to PRISM, start with
[getting-started.md](getting-started.md), then pick the platform page that
matches your machine.

### Skills — task-oriented playbooks

Distilled *do-this / not-that* guidance on the framework's hard-won lessons,
loadable as Claude agent skills or read as docs. See the
[skills catalogue](skills/README.md).

| Skill | Use when… |
|-------|-----------|
| [prism-launching-jobs](skills/prism-launching-jobs/SKILL.md) | Submitting a training job on Aurora (launcher/storage/flags) |
| [prism-configuration](skills/prism-configuration/SKILL.md) | Composing Hydra configs / experiment designs |
| [prism-data-pipeline](skills/prism-data-pipeline/SKILL.md) | WebDataset conversion, staging, bucketing, validation |
| [prism-daos-storage](skills/prism-daos-storage/SKILL.md) | DAOS containers, staging, FSDP/DAOS hangs |
| [prism-adding-a-modality](skills/prism-adding-a-modality/SKILL.md) | Adding a new encoder/projector/modality end-to-end |
| [prism-vllm-inference](skills/prism-vllm-inference/SKILL.md) | Serving/evaluating a checkpoint through vLLM |
| [prism-platforms](skills/prism-platforms/SKILL.md) | Running on Polaris / Perlmutter / baremetal |
| [prism-distributed-strategy](skills/prism-distributed-strategy/SKILL.md) | Choosing DDP / FSDP / HSDP / DeepSpeed |
| [prism-scaling-and-isoflop](skills/prism-scaling-and-isoflop/SKILL.md) | Throughput sweeps, IsoFLOP, MFU/scaling |
| [prism-evaluation](skills/prism-evaluation/SKILL.md) | Universal evaluator, benchmarks, vLLM parity |
| [prism-env-build](skills/prism-env-build/SKILL.md) | Building/packing the compute-node venv |
| [prism-multi-agent-workflow](skills/prism-multi-agent-workflow/SKILL.md) | Committing/branching safely in this shared clone |

### Getting started

| Document | Description |
|----------|-------------|
| [getting-started.md](getting-started.md) | Clone, pick a platform, build an environment, run a first job |
| [training/cli.md](training/cli.md) | Unified CLI (`prism`) reference: installation, commands, options, examples |

### Platforms — `platforms/`

Environment build and job submission, one page per machine, plus the
machine-independent path configuration every one of them needs.

| Document | Description |
|----------|-------------|
| [platforms/site_paths.md](platforms/site_paths.md) | The `PRISM_*` site variables: where model, data, tokenizer, asset, and output roots come from, for one user or a shared allocation |
| [platforms/aurora_operations.md](platforms/aurora_operations.md) | Aurora (Intel Max GPU): environment setup, launcher reference, session debugging, troubleshooting, env var reference |
| [platforms/daos_setup.md](platforms/daos_setup.md) | DAOS storage on Aurora: containers, data prep, models, troubleshooting |
| [platforms/running_on_polaris.md](platforms/running_on_polaris.md) | Polaris (NVIDIA A100, CUDA, PBS) |
| [platforms/running_on_perlmutter.md](platforms/running_on_perlmutter.md) | Perlmutter (NVIDIA A100, Slurm) |
| [platforms/running_on_rbdgx3.md](platforms/running_on_rbdgx3.md) | RBDGX3 single-node cluster: topology and launch scripts |

### Training — `training/`

| Document | Description |
|----------|-------------|
| [training/training.md](training/training.md) | Training entry points and stage selection (Hydra-based) |
| [training/data.md](training/data.md) | Data pipeline: datasets, WebDataset format, bucketing, straggler analysis, multi-dataset loading |
| [training/cli.md](training/cli.md) | Unified CLI (`prism`) reference |
| [training/deepspeed.md](training/deepspeed.md) | DeepSpeed integration notes (see `platforms/aurora_operations.md` for current guidance) |
| [training/training_calvin_vla.md](training/training_calvin_vla.md) | CALVIN vision-language-action (VLA) training path |

### Modalities — `modalities/`

Encoders, projectors, and output decoders for the non-text modalities.

| Document | Description |
|----------|-------------|
| [modalities/timeseries.md](modalities/timeseries.md) | Time-series modality: datasets, encoders (linear / Moirai / TimeOmni dynamic patching), training configs, evaluation |
| [modalities/projector.md](modalities/projector.md) | Projector architecture, Molmo2 reference implementation, gap analysis, ablation plans |
| [modalities/projector_normalization.md](modalities/projector_normalization.md) | Math behind `ModalityProjector`'s normalization modes (`rmsnorm`, `scale_only`, `l2_sequence`), including how DNA and text embeddings relate to them |
| [modalities/image_decoder.md](modalities/image_decoder.md) | Image output: the image-first decoder on the eager Hugging Face backbone |
| [modalities/image_decoder/](modalities/image_decoder/) | Image decoder deep dives: connectors, DOCCI data, diffusion, alignment stages |
| [modalities/time_series_decoder.md](modalities/time_series_decoder.md) | Linear quantile forecasting head sharing the readout → bridge → generator API |

### Models — `models/`

| Document | Description |
|----------|-------------|
| [models/prism.md](models/prism.md) | PRISM's own model card: what the shipped configs build, what data they point at, and why no benchmark score is claimed |
| [models/auroragpt_vlm.md](models/auroragpt_vlm.md) | AuroraGPT-2B backbone integration: 256K vocab, BF16 cross-entropy, configuration |
| [models/intern-s2-preview.md](models/intern-s2-preview.md) | Intern-S2 Preview 35B vs 397B time-series handling, as a reference implementation |

### Evaluation and inference — `evaluation/`

Two lanes: the universal evaluator reads a training checkpoint directly, while the vLLM
lane runs an exported copy for batched throughput and serving. `vllm_parity.py` is what
keeps the two agreeing.

| Document | Description |
|----------|-------------|
| [evaluation/evaluation.md](evaluation/evaluation.md) | Universal evaluator: methodology and benchmarks |
| [evaluation/inference_vllm.md](evaluation/inference_vllm.md) | vLLM-backed inference: the in-tree plugin, checkpoint export, serving, batched generation, parity. vLLM ships inside the `frameworks/2025.3.1` module — no container image or `pip install` |

### Results — `results/`

Recorded experiment outcomes. These are dated reports, not usage guides.

| Document | Description |
|----------|-------------|
| [results/scaling_study.md](results/scaling_study.md) | Scaling experiments, distributed strategies (DDP, FSDP, HSDP, COMPOSITE), throughput results, production configs |
| [results/per_modality_sweep.md](results/per_modality_sweep.md) | Per-modality throughput sweep harness (PRISM-MODALITY-SMOKE-1N): how to run it, baseline results, the `WEBDATASET_LOCAL_PATH` leak finding |
| [results/qwen3_siglip_scaling_experiments.md](results/qwen3_siglip_scaling_experiments.md) | Reproducible Qwen3/SigLIP2 VLM scaling experiments: data build, model choices, LR transfer, launch commands |
| [results/vlm_ablations.md](results/vlm_ablations.md) | Projector normalization ablation study: architectures, training schedules, evaluation |
| [results/patrick_tsqa_ab.md](results/patrick_tsqa_ab.md) | TSQA A/B: DDP vs HSDP on interleaved OLMo-1B (Aurora, 2026-07-03) |
| [results/xpu_flash_attention_gate0.md](results/xpu_flash_attention_gate0.md) | Aurora XPU flash-attention micro-benchmark (Gate 0, 2026-07-09) |
| [reports/](reports/) | Dated pilot and implementation reports, mostly image-decoder work |

### Scientific applications — `applications/`

| Document | Description |
|----------|-------------|
| [applications/bioreason_grpo.md](applications/bioreason_grpo.md) | BioReason: the three-stage training pipeline, with emphasis on Stage 3 GRPO |
| [applications/bioreason-eval.md](applications/bioreason-eval.md) | BioReason KEGG evaluation invocation |
| [applications/kegg_curation_pipeline.md](applications/kegg_curation_pipeline.md) | KEGG dataset curation pipeline — design doc |
| [applications/kegg_curation_implementation_status.md](applications/kegg_curation_implementation_status.md) | KEGG curation — what is built and live-tested, stage by stage |
| [applications/materials_multimodal.md](applications/materials_multimodal.md) | Materials property prediction from text + crystal graphs (`src/materials`, `CrystalGraphTokenEncoder`) |

### API reference — `api/`

| Document | Description |
|----------|-------------|
| [api/model.md](api/model.md) | `UnifiedTransformer` and `ModelConfig`: the four forward stages, presets, decoder registry |
| [api/encoders.md](api/encoders.md) | The `ModalityEncoder` contract, the eight encoders, optional-dependency gating |
| [api/modules.md](api/modules.md) | Projectors, MoE, attention, and LM head: the reusable blocks |

### Development — `development/`

| Document | Description |
|----------|-------------|
| [development/ci_multiplatform.md](development/ci_multiplatform.md) | Tiered CI strategy: required PR checks, HPC smoke contracts |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Development setup, checks CI runs, compatibility surfaces, AI/LLM policy |

### Superseded material

`docs/plans/` and `docs/historical/` held dated design plans and bring-up debug
journals whose work has all landed. Plans are tracking artifacts and belong in
the issue tracker, not in the repository, so they were removed rather than
carried into the public tree; `git log` retains them.

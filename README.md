# PRISM: Poly-Reasoning Integrated Scientific Multimodal Model

[![CI](https://github.com/AI-ModCon/BaseMM_PRISM/actions/workflows/pr_quality.yml/badge.svg?branch=main)](https://github.com/AI-ModCon/BaseMM_PRISM/actions/workflows/pr_quality.yml)
[![codecov](https://codecov.io/gh/AI-ModCon/BaseMM_PRISM/branch/main/graph/badge.svg)](https://codecov.io/gh/AI-ModCon/BaseMM_PRISM)

PRISM is a modular recipe for multimodal scientific modeling, organized around
representation, alignment, and fusion. Explicit **encoder**, **adapter**,
**language-backbone**, **readout**, and **decoder** interfaces separate domain
representation from shared reasoning and native generation.

Feature-level mid-fusion reuses specialist encoders while letting their
representations interact through a language backbone. Output readouts and conditioning
bridges connect the resulting states to task-specific generators or prediction heads.
Compatible encoders and backbones can be changed independently, subject to
preprocessing, feature, attention, and checkpoint contracts.

The design hypothesis is that these stable boundaries permit domain customization and
controlled model improvement while retaining a common training and evaluation workflow.

In the shipped configurations this means: a pretrained causal LM is the trunk, and each
non-text modality — images, time series, DNA, geometry, graphs, tables — has its own
encoder plus a projector that maps its features into the backbone's embedding space, so
one set of transformer weights sees every modality as tokens. The interfaces live in
[`src/encoders/`](src/encoders/), [`src/modules/projector.py`](src/modules/projector.py),
[`src/connectors/`](src/connectors/) (`Readout`, `ConditioningBridge`), and
[`src/decoders/`](src/decoders/).

**Hardware platforms**: Aurora (Intel Max Series GPUs, primary) and Polaris (NVIDIA A100)
at ALCF. Perlmutter and single-node CUDA workstations are also supported. Python 3.10 or
newer; CI covers 3.10 and 3.12.

**Full documentation: [docs/index.md](docs/index.md)** — start at
[docs/getting-started.md](docs/getting-started.md) if you are new.

---

## Quick start — laptop, workstation, or CI

No HPC allocation, DAOS container, or accelerator needed.

> **On an HPC login node, skip this block.** A plain `pip install -r
> requirements/base.txt` resolves PyTorch and the CUDA wheel stack from PyPI,
> shadowing the system-provided accelerator-aware build — on Aurora this is the
> `undefined symbol: __kmpc_fork_call` failure. Use the platform paths under
> *Installation* instead.

```bash
git clone --recursive https://github.com/AI-ModCon/BaseMM_PRISM.git
cd BaseMM_PRISM

python -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt
pip install -e . --no-deps

prism --help          # train / launch / data / eval / analyze / serve
```

To see PRISM actually train — no download, no account, no network at all:

```bash
python examples/portable_smoke/train_tiny.py
```

It synthesises a ~220 KB random-weight backbone, trains a ~90K-parameter model for 40
steps in a few seconds, and verifies it never touched the network. It trains on random
noise, so it proves the plumbing works on your machine and nothing about model quality —
see [examples/portable_smoke/README.md](examples/portable_smoke/README.md).

To run the test suite the way CI does, install the CI requirements first —
`requirements/base.txt` alone is missing packages that `pytest` needs at collection time:

```bash
pip install -r requirements/ci.txt
make test
```

Some tests in that selection construct real backbones and **download model configs from
Hugging Face on first run**. They are not marked `network`, so budget for the download or
pre-populate `HF_HOME`.

## Quick start — Aurora

```bash
# DAOS-backed (fastest; requires DAOS container access)
python tools/launch_aurora_unified.py --storage daos \
    --id MY-RUN --design PRISM-IMAGE-ONLY-2N --nodes 2 --batch

# WebDataset staged from Lustre to /tmp (no DAOS required)
python tools/launch_aurora_unified.py --storage webdataset-staged \
    --id MY-RUN --nodes 2 --batch --webdataset-dir <shard-dir>

# Generic Lustre (no DAOS, no staging)
python tools/launch_aurora_unified.py --storage lustre \
    --id MY-RUN --design PRISM-IMAGE-ONLY-1N --nodes 1
```

The dispatcher `exec`s the per-backend launcher (`tools/launch_aurora_{daos,web,}.py`)
with remaining flags forwarded verbatim. **The flag surface differs per backend** — for
example `--dataset-groups` and `--no-pil4dfs` are DAOS-only — so check `--help` on the
launcher for the storage path you picked rather than assuming one list covers all three.

DAOS is **not** required; it is the fastest path for vision/VLM workloads with container
access. Full launcher reference, environment setup, CCL tuning and troubleshooting:
[docs/platforms/aurora_operations.md](docs/platforms/aurora_operations.md).

## Configuration

PRISM resolves machine-specific paths through five `PRISM_*` variables rather than
hardcoding them, so the same configs work for one user or a shared allocation:

| Variable | Points at |
|---|---|
| `PRISM_DATA_ROOT` | Prepared WebDataset / Arrow shards |
| `PRISM_OUTPUT_ROOT` | Run directories and checkpoints |
| `PRISM_HF_HUB` | A Hugging Face hub directory |
| `PRISM_TOKENIZERS` | Interleaved tokenizers |
| `PRISM_ASSETS` | Staged third-party generator assets |

```bash
cp .env.template .env     # then fill in the five values
```

Values resolve from the process environment first, then a file named by
`PRISM_SITE_ENV`, then the repo's `.env`. On Aurora, copy the assignments from
[`.env.aurora.example`](.env.aurora.example). Full model, including how a shared
allocation points everyone at one file:
[docs/platforms/site_paths.md](docs/platforms/site_paths.md).

## The `prism` CLI

```bash
prism launch --platform aurora-daos --id MY-RUN --nodes 1 --dry-run   # inspect the PBS script
prism launch --platform aurora-daos --id MY-RUN --nodes 1 --batch     # submit it
```

| Command | Description |
|---------|-------------|
| `prism train` | Hydra-based training with config composition |
| `prism launch` | Generate and submit HPC jobs |
| `prism data` | Data download, conversion, validation, staging |
| `prism eval` | Model evaluation and throughput benchmarks |
| `prism analyze` | Model and dataset analysis |
| `prism serve` | Gradio UI (`serve ui`) or FastAPI server (`serve api`) |

`--platform` accepts `aurora`, `aurora-daos`, `aurora-web`, `perlmutter`, and
`baremetal`. Polaris has a launcher (`tools/launch_polaris.py`) but is not yet wired into
the CLI; invoke it directly. vLLM serving is likewise reached through
`tools/vllm_serve.py`, not `prism serve`.

Full reference: [docs/training/cli.md](docs/training/cli.md).

## Installation

| Platform | How |
|---|---|
| **Aurora** (canonical) | `bash tools/build_aurora_env.sh` builds a `uv`-managed venv on `/flare` from a checked-in lockfile; pass `--use-shared-venv` to any Aurora launcher. A legacy packed-tarball path (`tools/setup_deepspeed_env.sh`) also exists; both require `uv`. |
| **Polaris** | `bash tools/setup_polaris_env.sh` — builds `.venv-polaris` atop `conda/2025-09-28` |
| **Perlmutter** | `requirements/perlmutter.txt`. There is no setup script; see [docs/getting-started.md](docs/getting-started.md) for the known gaps |
| **Workstation / CI** | `requirements/base.txt`, as in the quick start. PyG C++ extension wheels are commented out — uncomment if your platform has matching wheels |

On any HPC platform, keep `--no-deps` when installing PRISM itself: a plain
`pip install` resolves a fresh PyTorch wheel that shadows the system-provided,
accelerator-aware build. Module-load order matters too — load the frameworks module
*then* activate the venv; the reverse breaks `torch_geometric`.

Per-platform detail, including lockfile regeneration:
[docs/platforms/](docs/platforms/).

## Architecture

### Modality encoders

| Modality | Encoder | Dimension |
|----------|---------|------------|
| **Text** | [OLMo-1B / OLMo-3 7B](https://huggingface.co/allenai/OLMo-7B-0724-hf) | 2048 / 4096 |
| **Image** | [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224) | 768 |
| **Time series** | [Moirai 2.0](https://huggingface.co/Salesforce/moirai-2.0-R-small) / [Intern-S2 preview](https://huggingface.co/internlm/Intern-S2-Preview-397B) / [TimeOmni](https://arxiv.org/abs/2510.03255) | Variable by backend |
| **DNA** | [Nucleotide Transformer v2 500M](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) | 1024 |
| **Geometry** | [Walrus](https://huggingface.co/polymathic-ai/walrus) | 512 |
| **Graph** | GraphMAE2 (PyG) | 768 |
| **Table** | [TAPAS](https://huggingface.co/google/tapas-base) | 768 |

Dimensions are the `d_*` defaults in `src/config.py`; `d_img` defaults to 1152
(SigLIP-large native) and is set to 768 by the SigLIP2-base configs.

### Modality support status

All seven modalities have an encoder, projector, and collator; five of the seven also
have an output decoder (table and DNA are input-only — `src/decoders/` registers no
`table` or `dna` key). Maturity beyond that varies widely. This table reflects what is
actually exercised by shipped configs and CI, not what the architecture permits.

| Modality | Status | Shipped model configs | Notes |
|---|---|---|---|
| **Text** | Supported | 26 of 26 | Every config enables it. With an HF backbone, the backbone's own embeddings are used and `TextEncoder` is bypassed. |
| **Image** | Supported | 9 of 26 | Best-validated multimodal path; the basis of the scaling results below. |
| **Time series** | Supported | 14 of 26 | Five encoder backends (`linear`, `moirai`, `intern_s2`, `intern_s2_397b`, `timeomni`). One of two modalities with a vLLM inference processor (the other is image). |
| **DNA** | Supported | 2 of 26 | Nucleotide Transformer encoder (`src/encoders/dna.py`), used by the BioReason/KEGG path — see [docs/applications/bioreason_grpo.md](docs/applications/bioreason_grpo.md). Input-only. |
| **Graph** | Experimental | 0 | Requires `torch_geometric`. Runs in the per-modality sweep (172.7 samp/s) but no shipped model config enables it, and the handler is labelled untested in its own smoke config. |
| **Table** | Experimental | 0 | TAPAS-based; runs in the sweep (65.5 samp/s). Input-only — no output decoder. No shipped model config enables it. |
| **Geometry** | Experimental | 0 | Requires `walrus` and `the_well`. Runs in the sweep (69.4 samp/s) **on synthetic data** — no real PDEBench corpus is staged, and `GeometryEncoder` hardcodes `input_dim=6`. A `WALRUS_FALLBACK=1` stub exists for smoke runs and produces meaningless features. |

"Experimental" means: the code path exists and has been run, but it is not enabled by any
shipped model config, has no dedicated test that instantiates the real encoder, and its
datasets are marked `skip` in `src/data/datasets_config.json`. Expect to do integration
work. See [docs/results/per_modality_sweep.md](docs/results/per_modality_sweep.md) for
the measured throughput and the caveats behind each number.

### Token harmonization

Modality-specific MLP projectors map encoder features to the backbone dimension
(`src/modules/projector.py`). A central `Modality` enum (`src/modalities.py`) is the
source of truth for the modality list. Rotary positional embeddings are applied over the
concatenated sequence; non-text spans attend bidirectionally while text stays causal.

### Backbones

| Model | Parameters | Status | Notes |
|-------|-----------|--------|-------|
| **OLMo-3 7B** | 7B | Primary | Production E2E target; FSDP/HSDP-validated on Aurora |
| **OLMo-1B** | 1B | Primary | Fast iteration, smoke gates, modality sweeps |
| AuroraGPT-2B | 2B | Active | `PRISM-AGPT2B-PROJ`; Llama architecture, XPU OOM fix in `src/model.py` |

There is no per-architecture adapter: `src/model.py` loads the backbone with a generic
`AutoModelForCausalLM.from_pretrained`, so any causal-LM architecture `transformers` can
load is a candidate. Only the three above are exercised by training runs and smoke gates;
everything else is best-effort — if you stand one up, please PR a config under
`src/conf/model/` and a smoke entry.

One architecture-keyed table does exist, `DECODER_LAYER_MAP` in
`src/training/distributed.py`, but it is an FSDP/HSDP wrap-policy lookup into upstream
`transformers` decoder-layer classes, not a weight-mapping adapter. A backbone missing
from it still loads; it falls back to scanning for a class named `*DecoderLayer`.

## Training

Stage is selected with Hydra `training=...`: encoder alignment (projector warmup against
a frozen encoder and backbone), the same with weights unfrozen for end-to-end runs, SFT,
and RL with GRPO. See [docs/training/training.md](docs/training/training.md) for the full
table and example invocations, and [docs/training/data.md](docs/training/data.md) for the
WebDataset pipeline, bucketing, and multi-dataset loading.

Named experiment designs live in `experiments/prism_designs.yaml` and modality presets in
`experiments/modality_presets.yaml`; both are consumed by `--design` and
`tools/run_sweep.py`.

### Validated configurations

Measured on Aurora. Full scaling table, methodology, and per-step timing breakdowns:
[docs/results/scaling_study.md](docs/results/scaling_study.md).

| Stage | Model | Strategy | Nodes | BS / GAS | Throughput | Notes |
|-------|-------|----------|-------|----------|------------|-------|
| Projector-only | OLMo-1B | DDP | 2 | 8 / 1 | ~415 samp/s | `PRISM-IMAGE-ONLY-2N`, pixmo |
| Projector-only | OLMo-7B | DDP | 2 | 3 / 16 | ~131 samp/s | `PRISM-IMAGE-ONLY-7B`, pixmo + `--use-bucketing` |
| E2E | OLMo-3 7B | FSDP `full_shard` | 1 | 24 / 1 | **77.7 samp/s** | `PRISM-OLMO3-E2E-PROD` + `--torch-compile` |
| E2E | OLMo-3 7B | HSDP `full_shard` | 2 | 20 / 1 | **122.1 samp/s** | best 2N config, + `--torch-compile` |

`--torch-compile` is what produced the two E2E numbers above. It is not universally safe
on XPU — see the troubleshooting section of
[docs/platforms/aurora_operations.md](docs/platforms/aurora_operations.md) before
enabling it on a new configuration.

## Testing

```bash
make test          # unit suite, as CI's `unit` job runs it
make lint          # ruff, same scope as CI
make type-check    # mypy, same scope as CI
make docs          # relative-link check, as CI's `doc_links` job runs it
make site-paths    # hardcoded-path ratchet
```

`make help` lists every target. These wrap the exact invocations CI uses — prefer them
over hand-written `pytest`/`mypy` commands, which drift. Tests that require XPU devices
must run on compute nodes via an interactive allocation, not the login node.

`tools/parity/` holds the smoke harness used during merges to `main` (1-node DDP
projector, 2-node FSDP E2E, CalvinVLA, and a launcher dry-run diff). Pass criterion and
per-script detail: [tools/parity/README.md](tools/parity/README.md).

## Evaluation and serving

There are two evaluation lanes, and they answer different questions.

### Universal evaluator — correctness against a training checkpoint

Reads a PRISM checkpoint directly through `UnifiedTransformer`, so it needs no export
step and sees the model exactly as training left it.

```bash
python tools/universal_evaluator.py --checkpoint <ckpt> --mode inspect_train --limit 10
python tools/universal_evaluator.py --checkpoint <ckpt> --mode inspect_eval
python tools/universal_evaluator.py --checkpoint <ckpt> --mode run_eval
```

`--checkpoint` is required in every mode. `bash tools/run_evaluator.sh` wraps it for a
compute node. See [docs/evaluation/evaluation.md](docs/evaluation/evaluation.md).

### vLLM lane — batched throughput and serving

Runs an **exported** checkpoint on vLLM's PagedAttention engine, through the in-tree
plugin at [`src/vllm_plugin/`](src/vllm_plugin/). This is the path for batch generation
at volume and for standing up a server.

It does not use a container image: vLLM `0.15.0+xpu` ships **inside the
`frameworks/2025.3.1` module** on Aurora, and the `_*_runner.sh` wrappers load that
module and call its interpreter directly. Do not `pip install vllm` — see
[docs/platforms/aurora_operations.md](docs/platforms/aurora_operations.md) on the module
pin.

```bash
# 1. Export a training checkpoint into a vLLM-loadable directory
python -m src.vllm_plugin.checkpoint_export --checkpoint <ckpt> --out exported/my-model

# 2. Evaluate it on a compute node (the runner loads frameworks/2025.3.1 for you)
bash tools/_vllm_eval_runner.sh --model exported/my-model --image test_images/ --limit 5

# 3. Or serve it
python tools/vllm_serve.py --model exported/my-model --port 8000 --enforce-eager
```

`vllm_serve.py` wraps vLLM's own OpenAI server — flags mirror `vllm serve` — and adds a
`/v1/prism/ts` route for modality payloads that do not fit the OpenAI schema.

| Tool | Purpose |
|---|---|
| `src/vllm_plugin/checkpoint_export.py` | Training checkpoint → vLLM-loadable directory |
| `tools/vllm_eval.py` | Image+text batched evaluation |
| `tools/vllm_eval_time_series.py` | Time-series evaluation, vLLM vs HF baseline with a speedup ratio |
| `tools/vllm_serve.py` | Serve a checkpoint |
| `tools/vllm_throughput.py` | Throughput benchmark |
| `tools/vllm_parity.py`, `vllm_parity_assert.py` | Parity between the two lanes — the gate that keeps them honest |

Only **image** and **time series** have vLLM inference processors; other modalities are
universal-evaluator only. Because the vLLM lane runs an exported copy, its numbers are
meaningful only if parity holds — run `vllm_parity_assert.py` before trusting a result
from it.

Full reference, including the export key-renaming and the serving flags:
[docs/evaluation/inference_vllm.md](docs/evaluation/inference_vllm.md).

## Project structure

```
BaseMM_PRISM/
├── src/
│   ├── train.py                # Hydra entry point
│   ├── model.py                # UnifiedTransformer
│   ├── config.py               # ModelConfig, TrainingConfig
│   ├── site_paths.py           # PRISM_* path resolution
│   ├── modalities.py           # Central Modality enum
│   ├── conf/                   # Hydra YAML (model, training, data, decoder)
│   ├── data/                   # WebDataset streaming, bucketing, per-modality dispatch
│   ├── encoders/               # SigLIP2, TAPAS, Moirai, Walrus, GraphMAE2, DNA, crystal graph
│   ├── decoders/               # Output decoders + registry
│   ├── modules/                # Projectors, MoE, attention, LM head
│   ├── training/               # Native DDP and Accelerate trainers, GRPO
│   ├── eval/                   # Evaluator tasks
│   ├── materials/              # Materials property prediction
│   ├── vllm_plugin/            # In-tree vLLM plugin
│   ├── ui/                     # Gradio UI (`prism serve ui`)
│   └── api/                    # FastAPI server (`prism serve api`)
├── tools/                      # Launchers, env builders, evaluator, parity gates, CI scripts
├── experiments/                # Named designs, modality presets, ablation catalog
├── requirements/               # Per-platform dependency sets and lockfiles
├── examples/portable_smoke/    # Offline train-in-seconds demo
├── docs/                       # See docs/index.md
└── tests/
```

## Documentation

[docs/index.md](docs/index.md) indexes every page. Most-used entry points:

- [docs/getting-started.md](docs/getting-started.md) — clone → environment → first run
- [docs/platforms/site_paths.md](docs/platforms/site_paths.md) — configuring the `PRISM_*` roots
- [docs/platforms/aurora_operations.md](docs/platforms/aurora_operations.md) — Aurora ops, env, CCL tuning, troubleshooting
- [docs/training/training.md](docs/training/training.md) — training stages
- [docs/results/scaling_study.md](docs/results/scaling_study.md) — scaling results
- [docs/skills/README.md](docs/skills/README.md) — task-oriented playbooks (launching jobs, DAOS, adding a modality, …)

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](./CONTRIBUTING.md) before
opening a pull request — it covers the development setup, the checks CI runs, the
compatibility surfaces that must be preserved, and the project's policy on
**AI/LLM-assisted contributions**.

- [CONTRIBUTING.md](./CONTRIBUTING.md) — how to contribute
- [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) — expected conduct
- [SECURITY.md](./SECURITY.md) — reporting a vulnerability (do not open a public issue)
- [CHANGELOG.md](./CHANGELOG.md) — notable changes

## License

PRISM is licensed under the Apache License, Version 2.0 — see [LICENSE](./LICENSE) for the
full terms.

Copyright (c) 2026, UChicago Argonne, LLC. Attribution for redistributed third-party
components, including the OLMo-derived tokenizer assets and the Walrus submodule, is
recorded in [NOTICE](./NOTICE).

## Acknowledgments

This project acknowledges support from the U.S. Department of Energy's Genesis Mission.

This software was produced under U.S. Government contract DE-AC02-06CH11357 for Argonne
National Laboratory, which is operated by UChicago Argonne, LLC for the U.S. Department of
Energy.

---
name: prism-configuration
description: >
  Configure PRISM training via Hydra and experiment designs. Use when composing
  configs (model/training/data groups), overriding on the CLI, choosing an
  experiment --design, picking a PRISM_CONFIGS preset, or wiring an env-var
  override that a launcher injects. Triggers: "Hydra config", "src/conf",
  "prism_designs.yaml", "--design", "config override", "PRISM_CONFIGS",
  "DIST_STRATEGY / MAX_SEQ_LENGTH env var", "which model config".
metadata:
  version: "1.0"
  project: prism
---

# PRISM — Configuration (Hydra + Experiment Designs)

PRISM configuration has three layers, from lowest to highest precedence:
**Hydra config groups → experiment `--design` → env-var overrides the launcher
injects.** Know which layer a value comes from before you change it.

## 1. Hydra config groups (`src/conf/`)

```
src/conf/
├── config.yaml        # root defaults (model, training, data, exp) + seed/workers/wandb
├── model/             # prism_olmo3_7b.yaml (default), prism_auroragpt_2b.yaml, prism_7b_image_only.yaml, …
├── training/          # zone_a.yaml (default), molmo_stage1.yaml (E2E), projector_only.yaml, zone_a_ts.yaml
├── data/              # daos_datasets.yaml, lustre_datasets.yaml, per-modality smoke configs
└── exp/               # experiment metadata (defaults.yaml)
```

Override groups and values on the CLI (last-wins):

```bash
python train.py model=prism_7b_image_only training=molmo_stage1 training.batch_size=2
```

## 2. Training stages (`training=`)

| Config | Encoder | LLM | Use case |
|--------|---------|-----|----------|
| `projector_only` | Frozen | Frozen | Initial alignment |
| `zone_a` (default) | Frozen | Frozen | Standard DDP training |
| `encoder_projector` | Trainable | Frozen | Fine-tune vision encoder |
| `molmo_stage1` | Trainable | Trainable | Full E2E, differential LRs |

## 3. Code-level presets: `PRISM_CONFIGS` (`src/config.py`)

Named `ModelConfig` + `TrainingConfig` presets (not Hydra files). Examples:
`prism-nano`, `prism-micro`, `prism-mini`, `prism-small`, `prism-base`,
`prism-granite-2b`, `prism-phi4-mini`, `prism-auroragpt-2b`, `prism-olmo3-7b`,
`prism-nemotron-30b`, and time-series variants (`prism-olmo-ts-7b`,
`prism-olmo-ts-1b-interleaved`, …). Use these for programmatic model construction.

## 4. Experiment designs (`experiments/prism_designs.yaml`)

`--design` (on the launcher) selects a named experiment. Structure:

```yaml
experiments:
  - id: PRISM-...
    name: ...
    resources: { ngpus: ..., topology: ..., walltime: ... }
    common_overrides: { ... }          # applied to every variant
    variants:
      - id: ...
        overrides: { ... }             # variant-specific Hydra overrides
```

Key production designs: `PRISM-OLMO3-E2E-PROD` (primary E2E), `PRISM-IMAGE-ONLY-7B`
(projector-only DDP), `PRISM-IMAGE-ONLY-2N` (fast OLMo-1B), `PRISM-AGPT2B-PROJ`.
See [`docs/results/scaling_study.md`](../../results/scaling_study.md) and the design catalogue for
the full list and their tested throughput.

## 5. Env-var overrides (launcher → train.py)

Launchers translate CLI flags into env vars that `src/train.py` reads via
`os.environ`. The main ones:

| Env var | Meaning |
|---------|---------|
| `DIST_STRATEGY` / `USE_NATIVE_FSDP` / `USE_NATIVE_DDP` | Distributed strategy selection |
| `MAX_SEQ_LENGTH` (default 2048) | Sequence cap; set to 1024 for E2E |
| `USE_BUCKETING`, `USE_BUCKETED_COLLATOR`, `BUCKET_BUFFER_SIZE` | Length bucketing |
| `GRAD_CKPT_FREQ` (default 1) | Gradient-checkpoint every N layers |
| `DATASET_GROUPS`, `DATASET_CONFIG`, `DATASET_PROPORTIONS`, `USE_MULTI_DATASET` | Data selection |
| `DAOS_MOUNT`, `LOCAL_SHARDS_DIR` | Storage paths |
| `ENABLE_ALL_MODALITIES` | Force all encoders on |

## 6. The config-drift trap (read this before "fixing" a config)

The single most common silent bug: you fix a value in a YAML, but the launcher
still injects the old value via `${VAR:-default}`. **A value can be set in three
places** (Hydra YAML, launcher flag default, env-var fallback) and the highest
layer wins. When changing a default:

1. Grep the launcher for the corresponding `${VAR:-...}` default.
2. Grep `src/train.py` for the `os.environ.get("VAR", ...)` fallback.
3. Make the YAML, launcher default, and env fallback agree — or delete the
   redundant layer.

See the installed `launcher-config-drift` skill for the full pattern.

## Gotchas

- `ModelConfig.modalities` is typed `list[Modality]` with `__post_init__`
  string-coercion — pass modality names as strings from YAML/CLI and they parse.
- Env-var overrides beat Hydra CLI overrides when both are present, because the
  launcher exports them last. If a CLI override "doesn't take," check the env.

## See also

- Deep reference: [`docs/training/training.md`](../../training/training.md).
- [prism-launching-jobs](../prism-launching-jobs/SKILL.md), [prism-distributed-strategy](../prism-distributed-strategy/SKILL.md).
- Generic installed skill: `launcher-config-drift`.

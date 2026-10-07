# Training Pipeline

> The standalone `training/train_zone_{a,b,c}.py` scripts have been removed.
> All training now goes through the Hydra-configured `train.py` entry point,
> dispatched by the Aurora launcher.

## Entry points

- **Trainer**: `train.py` (Hydra; selects model/training/data configs)
- **Launcher**: `tools/launch_aurora_daos.py` (Aurora w/ DAOS),
  `tools/launch_aurora_web.py` (Aurora w/ WebDataset staging from Lustre),
  `tools/launch_baremetal.py` (local / single-node dev),
  `tools/launch_perlmutter.py` (Perlmutter)

See [aurora_operations.md](../platforms/aurora_operations.md) for the full launcher
reference and [scaling_study.md](../results/scaling_study.md) for production configs.

## Stage selection

Training stage is selected via Hydra `training=...`:

| Hydra config | Encoders | LLM | Use case |
|---|---|---|---|
| `training=projector_only` | frozen | frozen | encoder alignment: projector warmup against a frozen encoder and backbone |
| `training=zone_a` | frozen | frozen | standard frozen-backbone DDP training |
| `training=encoder_projector` | trainable | frozen | fine-tune the vision encoder |
| `training=molmo_stage1` | trainable | trainable | encoder alignment with weights unfrozen; full E2E with differential LRs |
| `training=bioreason_sft` | frozen | frozen + LoRA | SFT on instruction data (BioReason/KEGG); projector loaded from the alignment stage and frozen |
| `training=bioreason_grpo` | frozen | frozen + LoRA | RL with GRPO (`ZoneDTrainer` in `src/training/trainer_grpo.py`) |

The last two set `freeze_llm: true` with `lora_enabled: true` — the base
backbone is frozen and the LoRA adapters are the only trainable part.

### A note on the stage names

Older documents label these **Zone A / B / C / D**. That vocabulary is retired:
it was ambiguous about what each stage trains, and the labels in circulation
disagreed with the code. The stages are now named for what they do — *encoder
alignment*, *encoder alignment with weights unfrozen*, *SFT*, and *RL*.

The Hydra keys and class names keep their original spelling for compatibility
(`training=zone_a`, `ZoneATrainer`, `ZoneDTrainer`, `trainer_zone_a.py`), so
those still appear verbatim wherever a command or symbol is quoted. Renaming
them is a breaking change and is tracked separately.

DPO is **not** implemented in the PRISM trainer. It is sometimes described as
"Zone C", but Zone C was SFT, which is implemented; DPO was never wired up at
all.

## Example invocations

```bash
# Aurora, DAOS-backed, projector warmup
python tools/launch_aurora_daos.py \
    --id MY-RUN --design PRISM-OLMO3-E2E-PROD \
    --nodes 2 --batch --queue debug-scaling \
    --dataset-groups projector --use-bucketing \
    --no-pil4dfs --fsdp-production-mode

# Local / baremetal dev
python tools/launch_baremetal.py --id local-test
```

CLI override of any Hydra key is supported, e.g.
`training=molmo_stage1 training.batch_size=2 training.max_steps=200`.

See `experiments/prism_designs.yaml` for the full set of named designs.

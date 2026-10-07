# PRISM Skills Catalogue

Task-oriented **playbooks** that distill PRISM's hard-won operational lessons —
across Aurora, Polaris, and Perlmutter — into fast *do-this / not-that* guidance
so new contributors and coding agents can adopt the framework without
re-deriving every gotcha from scratch.

Each skill is a folder containing a `SKILL.md` written in the
[Claude skills](https://docs.claude.com/en/docs/claude-code/skills) convention
(YAML frontmatter + progressive disclosure via `references/`). That means every
skill doubles as:

1. **A human-readable playbook** — read it top-to-bottom like any doc.
2. **An agent-loadable skill** — symlink the folder into `~/.claude/skills/`
   (or a repo `.claude/skills/`) and a coding agent will auto-discover it by the
   `description` triggers.

These skills are **distilled + linked**: they capture the decisions, defaults,
and traps, then point into the long-form `docs/*.md` for full detail. When a
number or flag changes, fix it in the deep doc and keep the skill pointing there.

## Catalogue

| Skill | Use when… | Deep docs |
|-------|-----------|-----------|
| [prism-launching-jobs](prism-launching-jobs/SKILL.md) | Submitting a PRISM training job on Aurora — picking a launcher/storage backend, key flags, interactive vs batch | [aurora_operations](../platforms/aurora_operations.md) |
| [prism-configuration](prism-configuration/SKILL.md) | Composing Hydra configs, choosing an experiment `--design`, or wiring an env-var override | [training](../training/training.md) |
| [prism-data-pipeline](prism-data-pipeline/SKILL.md) | Converting data to WebDataset, staging shards, enabling bucketing, or validating a dataset | [data](../training/data.md) |
| [prism-daos-storage](prism-daos-storage/SKILL.md) | Creating/mounting DAOS containers, staging data+models, or hitting a DAOS/FSDP hang | [daos_setup](../platforms/daos_setup.md) |
| [prism-adding-a-modality](prism-adding-a-modality/SKILL.md) | Adding a new encoder/projector/modality end-to-end | [api/encoders](../api/encoders.md), [projector](../modalities/projector.md) |
| [prism-vllm-inference](prism-vllm-inference/SKILL.md) | Serving/evaluating a PRISM checkpoint through vLLM, or exporting a checkpoint | [inference_vllm](../evaluation/inference_vllm.md) |
| [prism-platforms](prism-platforms/SKILL.md) | Running on Polaris / Perlmutter / baremetal instead of Aurora — what transfers, what doesn't | [running_on_polaris](../platforms/running_on_polaris.md), [running_on_perlmutter](../platforms/running_on_perlmutter.md) |
| [prism-distributed-strategy](prism-distributed-strategy/SKILL.md) | Choosing DDP vs FSDP vs HSDP vs DeepSpeed, setting batch-size ceilings, or debugging a collective hang | [scaling_study](../results/scaling_study.md), [DeepSpeed](../training/deepspeed.md) |
| [prism-scaling-and-isoflop](prism-scaling-and-isoflop/SKILL.md) | Measuring throughput, running the IsoFLOP sweep, or reasoning about scaling/MFU | [scaling_study](../results/scaling_study.md), [per_modality_sweep](../results/per_modality_sweep.md) |
| [prism-evaluation](prism-evaluation/SKILL.md) | Running the universal evaluator, a benchmark suite, or a vLLM parity check | [evaluation](../evaluation/evaluation.md) |
| [prism-env-build](prism-env-build/SKILL.md) | Building/packing the venv tarball shipped to compute nodes, or debugging an import shadow | [aurora_operations](../platforms/aurora_operations.md) |
| [prism-multi-agent-workflow](prism-multi-agent-workflow/SKILL.md) | Committing/branching/pruning in this repo where multiple agents share the clone | [CONTRIBUTING](../../CONTRIBUTING.md) |

## Companion generic skills (already installed)

These PRISM skills are **project-specific** and deliberately do *not* restate
general knowledge. Where a generic installed skill covers the background, the
PRISM skill cross-references it rather than duplicating it:

| Generic skill | Covers | PRISM skill that builds on it |
|---------------|--------|-------------------------------|
| `pbs`, `aurora` | Generic PBS syntax, Aurora system layout | prism-launching-jobs, prism-platforms |
| `hpc-iteration-discipline` | Not burning queue time on shared clusters | prism-launching-jobs |
| `distributed-training-debugging` | Generic FSDP/DDP/NCCL/XCCL hang & OOM debugging | prism-distributed-strategy |
| `python-env-shadowing-hpc` | `module load` vs venv vs `~/.local` import precedence | prism-env-build, prism-platforms |
| `ml-data-pipeline-correctness` | Verifying a dataloader feeds what you think | prism-data-pipeline |
| `launcher-config-drift` | YAML fixed but `${VAR:-default}` still injects old value | prism-configuration |
| `vllm` | Generic vLLM engine + Aurora XPU internals | prism-vllm-inference |
| `multi-agent-git-hygiene`, `shell-quoting-traps` | Worktree/branch safety, launcher heredoc quoting | prism-multi-agent-workflow, prism-launching-jobs |

## Using a skill as an agent skill

```bash
# Symlink one PRISM skill so your coding agent discovers it
ln -s "$PWD/docs/skills/prism-launching-jobs" ~/.claude/skills/prism-launching-jobs

# …or symlink the whole catalogue
for d in docs/skills/prism-*; do
  ln -s "$PWD/$d" "$HOME/.claude/skills/$(basename "$d")"
done
```

The agent loads a skill when the task matches its `description` triggers. Reading
the `SKILL.md` directly always works too.

## Adding a new skill

1. Create `docs/skills/prism-<topic>/SKILL.md` with this frontmatter:

   ```yaml
   ---
   name: prism-<topic>
   description: >
     One line on WHAT this covers, then explicit "Use when…" triggers
     (verbs and concrete nouns the agent will match on).
   metadata:
     version: "1.0"
     project: prism
   ---
   ```

2. Keep the body **scannable** (~100–200 lines): decisions, defaults, gotchas,
   and a copy-pasteable command or two. Push long tables / walkthroughs into
   `references/*.md` inside the skill folder and link them from a "When to load
   which reference" table.
3. **Distill, don't duplicate** — link into the relevant `docs/*.md` for depth,
   and cross-reference any generic installed skill instead of restating it.
4. Verify every path/flag/number against the actual repo before asserting it
   (agent memory and old docs drift; the code is the source of truth).
5. End with a **See also** footer, then add a row to the catalogue table above.

---
name: prism-platforms
description: >
  Run PRISM on platforms other than Aurora — Polaris, Perlmutter, or a baremetal
  multi-GPU node. Use when picking the right launcher and env setup for a
  non-Aurora machine, understanding storage/module/backend differences, or
  reasoning about what performance findings transfer across platforms. Triggers:
  "run on Polaris", "run on Perlmutter", "baremetal / rbdgx3", "NCCL vs XCCL",
  "setup_polaris_env", "launch_polaris / launch_perlmutter / launch_baremetal".
metadata:
  version: "1.0"
  project: prism
---

# PRISM — Platforms (beyond Aurora)

PRISM's primary target is Aurora (Intel XPU); Polaris (NVIDIA A100) is the
secondary, with Perlmutter and single-node CUDA also supported. The training
code is platform-agnostic — **the launcher, env build, storage layout, and
distributed backend differ per machine.** Load the matching reference:

| Platform | Reference |
|----------|-----------|
| Polaris (A100 / CUDA / PBS) | [`references/polaris.md`](references/polaris.md) |
| Perlmutter (A100 / CUDA / Slurm-style modules) | [`references/perlmutter.md`](references/perlmutter.md) |
| Baremetal multi-GPU (rbdgx3) | [`references/baremetal.md`](references/baremetal.md) |

For Aurora itself, see prism-launching-jobs / prism-daos-storage.

## The distributed backend is auto-selected by device

DDP/FSDP backend follows the accelerator: **XPU → `xccl`, CUDA → `nccl`, else
`gloo`.** The Polaris launcher exports `DIST_BACKEND=nccl` explicitly. Don't
hardcode a backend in portable code.

## What transfers across platforms — and what doesn't

- **Transfers:** the model, Hydra configs, experiment designs, the data pipeline.
- **Does NOT transfer:**
  - *FSDP `full_shard` efficiency.* AGPT's ~94% `full_shard` efficiency (2B,
    seq 8192, compute-bound) does **not** carry to PRISM's comm-bound
    0.6B/seq-512 regime, where HSDP `shard_grad_op` wins by ~42–47%. Re-measure
    per platform; don't assume the number.
  - *`torch.compile` viability.* Single-node compile can help; multi-node compile
    is blocked by an oneCCL deadlock on XPU. CUDA platforms behave differently —
    re-check.
  - *Env-shadowing behavior.* Each platform's `module load` + venv precedence
    differs — see the installed `python-env-shadowing-hpc` skill and
    prism-env-build.

## Keeping a port branch clean

Platform-port branches (`polaris-multinode`, `rocm`, …) should be minimal diffs
against `main`. Don't commit `.claude/`, `CLAUDE.md`, checkpoints, logs, or
experiment artifacts onto them. See the installed `platform-port-branch-hygiene`
skill.

## See also

- Deep references: [`docs/platforms/running_on_polaris.md`](../../platforms/running_on_polaris.md),
  [`docs/platforms/running_on_perlmutter.md`](../../platforms/running_on_perlmutter.md),
  [`docs/platforms/running_on_rbdgx3.md`](../../platforms/running_on_rbdgx3.md).
- [prism-env-build](../prism-env-build/SKILL.md), [prism-distributed-strategy](../prism-distributed-strategy/SKILL.md).
- Generic installed skills: `python-env-shadowing-hpc`, `platform-port-branch-hygiene`, `pbs`.

---
name: prism-distributed-strategy
description: >
  Choose and configure PRISM's distributed training strategy on Aurora XPU. Use
  when picking DDP vs FSDP vs HSDP vs DeepSpeed ZeRO, setting per-strategy batch-
  size ceilings, re-exporting CCL/XCCL env after module load, or debugging a
  collective hang / throughput collapse. Triggers: "DDP vs FSDP vs HSDP",
  "full_shard vs shard_grad_op", "ZeRO-2 / ZeRO-3", "CCL env vars", "xccl",
  "static_graph", "find-unused-params", "backward collective slow", "AllGather hang".
compatibility: Aurora XPU. See distributed-training-debugging for generic hang/OOM debugging.
metadata:
  version: "1.0"
  project: prism
---

# PRISM — Distributed Strategy (Aurora XPU)

Pick the strategy by model size and how comm-bound you are. For generic
FSDP/DDP/NCCL/XCCL hang & OOM debugging, use the installed
**`distributed-training-debugging`** skill; this skill is PRISM's *choices and
validated numbers* on XPU.

## Which strategy

| Strategy | Use when | Notes |
|----------|----------|-------|
| **DDP** (default) | Projector-only / frozen-backbone; model fits per-GPU | `xccl` backend; `static_graph=True` requires fixed modalities |
| **FSDP** `full_shard` | 7B+ E2E; single node or high compute intensity | bf16 mixed precision; needs `--no-pil4dfs` over DAOS |
| **HSDP** `shard_grad_op` | Comm-bound small models across nodes | Often **beats** FSDP full_shard by ~42–47% for 0.6B/seq-512 |
| **DeepSpeed ZeRO-2** | Alt to DDP for proj-only | ~92% DDP parity proj-only; ~9.8 samp/s 1N E2E |
| **DeepSpeed ZeRO-3** | Max sharding | ~14.9 samp/s @ BS=8; auto-enables `--no-pil4dfs` |

> **`full_shard` efficiency does not transfer between regimes.** AGPT's ~94%
> `full_shard` efficiency is a compute-bound (2B, seq 8192) result; PRISM at
> 0.6B/seq-512 is comm-bound, so keep HSDP `shard_grad_op` there. Re-measure —
> don't inherit the number.

## Batch-size ceilings (OLMo-7B, measured)

| Config | Max BS | Note |
|--------|--------|------|
| DDP projector | 3 | higher OOMs |
| FSDP 2-node | 16 | `PRISM-OLMO3-E2E-PROD` design BS is sized for **8N** |
| FSDP 2-node smoke | 4 | design BS=16 OOMs on 2N — smoke scripts drop to 4 |

Also: `--max-seq-length 1024` for E2E (2048 OOMs); `--grad-ckpt-freq 1` required
at seq=1024 (do NOT set 2).

## XPU / CCL rules (each has cost a run)

- **Re-export CCL env vars after `module load frameworks`** — the module
  overrides them. Validated config: `CCL_WORKER_COUNT=1`, no `RS=ring`, no
  `TORCH_XPU_ALLOC_CONF`, no `empty_cache()` inside FSDP.
- **Do NOT pass `device_id` to `init_process_group` on XPU** — causes DataLoader
  worker hangs.
- **`--find-unused-params` disables `static_graph` and costs ~30%.** Avoid unless
  a multi-modality 2N run actually crashes without it.
- Current CCL/XCCL: `frameworks/2025.3.1` (XCCL 2025.3.1) is clean — the April
  2026 `ccl_check_usm_pointers` bug is gone; launchers moved off the 2025.2.0 pin.
- After a hung job: `pkill -9 python3; sleep 10` to clear stale XCCL state.

## Not viable (don't re-litigate)

- **Multi-node `torch.compile`:** oneCCL deadlock. Single-node compile is fine.
  The compile "storm" is scale-dependent (appears at 10N+, not 2N).
- **IPEX varlen for training:** no OLMo dispatch, no training-mode/autograd
  kernel (IPEX 2.10.10). Use bucketing for padding reduction instead. Varlen is
  fine for inference/eval.
- **DeepSpeed + compile:** NaN / AttributeError.

## Watch out: intermittent backward-collective collapse

2N HSDP eager sometimes holds ~252 samp/s and sometimes collapses to 25–75
(backward grows 1s→12s after ~step 20). Mechanism unknown; it predates torch 2.13.
It masquerades as a "bad node" or "eager hang." If a scaling curve looks noisy,
suspect this before blaming a config change.

## See also

- Deep references: [`docs/results/scaling_study.md`](../../results/scaling_study.md),
  [`docs/training/deepspeed.md`](../../training/deepspeed.md).
- [prism-scaling-and-isoflop](../prism-scaling-and-isoflop/SKILL.md), [prism-daos-storage](../prism-daos-storage/SKILL.md).
- Generic installed skill: `distributed-training-debugging`.

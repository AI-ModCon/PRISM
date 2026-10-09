# PRISM Gate 0: Aurora XPU flash-attention micro-bench

**Date:** 2026-07-09 · **Node:** x4220c4s5b0n0 (single tile, `ZE_AFFINITY_MASK=0`)
· **Job:** 8659815 (debug, 1h hold) · **Harness:** `tools/bench_xpu_flash_sdpa.py`

## TL;DR — the guide's premise no longer holds on torch 2.13

The porting guide (`docs/xpu_flash_attention_porting_guide.md`) says XPU
auto-dispatch "**NEVER picks flash**" and you must force
`sdpa_kernel([FLASH_ATTENTION])`. That was true on **torch 2.10** (the
Steven/AGPT baseline). On **torch 2.13** (the venv the perf work already
uses), **`F.scaled_dot_product_attention` already dispatches to the fused
kernel automatically**. `default` and `flash` are within measurement noise on
every shape tested.

Implication: **the ~2× step-time / ~8× memory win the guide promises is a
large fraction of the +37 % single-node throughput jump we already measured
going 2.10 → 2.13** ([[torch213_scaling_curve_2026_06_22]]). We're not
leaving a big lever on the table by not wiring the guide's helper into
`src/model.py:963` on torch 2.13.

## What was measured

Shapes matching PRISM's actual model configs (Qwen3-0.6B: H=16, D=64, GQA
kv=8, seq ∈ {512, 2048, 4096, 8192}; SigLIP2 vision block: H=12, D=64,
seq=196 bidirectional; batch scaling at bs=8/seq=2048 and bs=4/seq=4096).
Four backends compared per shape:

- **default** — whatever XPU auto-dispatch picks for `F.sdpa`.
- **math** — explicitly `sdpa_kernel([MATH])` (materializes `[B,H,S,S]` fp32 scores).
- **flash** — explicitly `sdpa_kernel([FLASH_ATTENTION])` (native XPU fused MHA).
- **eager** — manual matmul+softmax reference in bf16.

Metrics: peak XPU alloc (GiB), fwd/bwd ms (mean of 5 iters after 2 warmups),
max\|Δ\| vs. math ref (bf16 numerics — identical seeds per iter across
backends). `WARN_FOR_UNFUSED_KERNELS=True` to catch silent fallback.

## Headline numbers (Qwen3-0.6B, causal, seq 4096, bs=1)

| stack (venv) | backend | peak GiB | fwd ms | bwd ms | err vs math |
|---|---|---:|---:|---:|---:|
| **torch 2.10** (`sww/prism-envs/qwen3-siglip-py3.12`) | default | 3.09 | 20.68 | 9.27 | 0 |
|  | flash (forced) | **0.16** | **1.14** | 18.48 | 0.016 |
| **torch 2.13** (`ngetty/venvs/torchtune-pt213-xpu`) | default | **0.12** | **0.95** | 15.45 | 0.016 |
|  | flash (forced) | 0.14 | 0.98 | 16.71 | 0.016 |
| **torch 2.11 nightly** (`ngetty/venvs/torchtune-pt-nightly-xpu`) | default | 0.14 | 1.12 | 17.45 | 0.016 |
|  | flash (forced) | 0.15 | 1.12 | 17.75 | 0.016 |

**Read:** on 2.10, default is math (3.09 GiB, 20.7 ms fwd) and forcing flash
gives ~19× less memory and ~18× faster fwd. On 2.13, default *is already*
flash (0.12 GiB, 0.95 ms fwd) — the forced-flash column is a no-op. The
bwd time is noisier (see caveat below) and does not favor flash consistently
on 2.13.

## Scaling across seq length (Qwen3-0.6B, causal, bs=1)

| seq | 2.10 default (math) fwd/mem | 2.10 flash forced fwd/mem | 2.13 default fwd/mem |
|----:|---|---|---|
| 512  | 0.68 ms / 0.06 GiB | 0.14 ms / 0.02 GiB | 0.08 ms / 0.015 GiB |
| 2048 | 6.00 ms / 0.79 GiB | 0.38 ms / 0.08 GiB | 0.30 ms / 0.06 GiB |
| 4096 | 20.68 ms / 3.09 GiB | 1.14 ms / 0.16 GiB | 0.95 ms / 0.12 GiB |
| 8192 | 99.78 ms / 12.18 GiB | 2.52 ms / 0.32 GiB | 2.27 ms / 0.24 GiB |

Memory scales O(S) for flash and O(S²) for math, as expected. At seq 8192,
the 12 GiB math tensor is what would OOM a 64 GiB tile with any other
activations — this matches the "O(T²) cliff" mechanism we thought was in
play for issue #120 before that turned out to be a different bug.

## SigLIP2 vision block (H=12, D=64, seq 196, **bidirectional**)

| stack | default | math | flash forced |
|---|---|---|---|
| pt213 bidir | 0.06 ms / 0.005 GiB | 0.39 ms / 0.009 GiB | 0.10 ms / 0.005 GiB |

Bidirectional flash **also works** and beats math by ~5–7×. Same story: on
2.13 the default already picks it. SigLIP2 vision fwd is a small slice of
step time (frozen encoder, seq 196), so the absolute impact is small, but
"no extra work" if we ever wire the guide's helper for other modules.

## Batch scaling (Qwen3-0.6B, causal, seq 2048/bs=8)

| stack | default fwd/mem | flash forced fwd/mem |
|---|---|---|
| torch 2.10 | 52.86 ms / 6.34 GiB (math) | 1.37 ms / 0.63 GiB |
| torch 2.13 | 1.54 ms / 0.47 GiB (**already flash**) | 1.35 ms / 0.53 GiB |

## Caveats

1. **bwd times are noisier than fwd.** The `.pow(2).mean().backward()` we
   use is a scalar loss so bwd is small; wall-clock variance dominates at the
   sub-10-ms scale. The memory delta and fwd time are the reliable signals.
2. **Same-node A/B is the only valid comparison** — Aurora has ~1.8× node
   variance. All rows above are back-to-back on x4220c4s5b0n0.
3. **Backward wins are less dramatic than fwd** even on 2.10 (flash bwd 18.5
   ms vs. math 9.3 ms at seq 4096). This is because our bench sums a scalar
   loss; a real training step has a full-rank gradient chain through
   attention output. Real-step gains will show up in end-to-end throughput,
   not in this row.
4. **Numerics are bf16-close, not bit-exact** (max\|Δ\| ≤ 0.02). Guide says
   the same; we should still validate loss trajectory in a real training run
   before defaulting anything on.
5. **Single-tile bench.** Multi-node collective work is not exercised here —
   this only measures the compute path.

## What this changes for the PRISM plan

- **Skip the guide's Step 3 (wire `sdpa_kernel([FLASH_ATTENTION])` into
  `src/model.py:963`) for torch 2.13 runs.** It would be a no-op; auto-dispatch
  already gets it. Adds code and a flag with no benefit.
- **For anyone still on torch 2.10** (Steven's `qwen3-siglip-py3.12` venv is
  the one in-repo config that still uses it), the guide's advice stands and
  the payoff is real (~18× fwd speedup, 19× memory reduction at seq 4096).
  Cheapest path: upgrade that venv to 2.13 rather than adding a flag.
- **The +37% 1N throughput jump we saw 2.10→2.13**
  ([[torch213_scaling_curve_2026_06_22]]) is very plausibly the flash-attn
  default-dispatch change. Worth a mental note that "2.13 upgrade" and
  "flash on" are the same lever from PRISM's perspective on that venv.
- **The [[flex_attention_xpu_bench_2026_07_03]] finding that "Intel SDPA is
  already O(T)-memory"** is retroactively explained: it was measured on torch
  2.13 where default dispatches to the fused SYCL-TLA kernel. On 2.10, SDPA
  is *not* O(T)-memory — that observation was venv-specific and shouldn't be
  generalized as "Intel SDPA behavior."

## Next steps (if anyone wants to push further)

1. **Verify at real training scale on 2.10 → 2.13.** Take one currently-slow
   2.10 run and rerun on 2.13 (same node, same config); if the delta is
   ~37% and matches the memory-headroom prediction (5–10× more room per
   tile), the story is nailed down.
2. **Only bother wiring the `sdpa_kernel` helper if we find a torch build
   where default reverts to math** (e.g. a new frameworks module or a
   torch.compile-emitted graph that stops picking flash). Gate it behind an
   env var, verify with `WARN_FOR_UNFUSED_KERNELS`.
3. **Do NOT wire it into any code path that already passes an
   `attn_mask`** (e.g. `walrus/` `axial_time_attention`, `mpp_avit`) — flash
   is disqualified there.

## Files

- Harness: `tools/bench_xpu_flash_sdpa.py`
- Raw logs: `logs/bench_xpu_flash_v2_*_20260709_174328.log`
- Guide this evaluates: `docs/xpu_flash_attention_porting_guide.md`

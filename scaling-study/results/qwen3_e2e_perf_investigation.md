# Qwen3-0.6B E2E vs Projector-Only Backward Gap — Living Report

**Status:** Phase 0 complete. Phase 1 pending submission.
**Last updated by:** Claude, 2026-06-18

## Question

Why does E2E Qwen3-0.6B (Bwd=83% of step, 250 samp/s @ 10N) run 2.2× slower
than projector-only frozen (Bwd=42%, 550 samp/s @ 10N)? Naive FLOPs predicts
~1.5×, leaving a ~1.5× residual.

## Phase 0 — Static checks (no GPU time)

Sourced from Sam's runtime logs and his `perf.jsonl` outputs.

### Confirmed identical between E2E and projector-only

| Knob | Value | Source |
|---|---|---|
| `attn_implementation` | **sdpa** | log: `Backbone loaded (attn=sdpa)` (both) |
| Gradient checkpointing | **ALL 28 layers, use_reentrant=False** | log: `Gradient Checkpointing: ALL 28 layers (...)` (both) |
| `enable_input_require_grads()` | **on** | `train.py:441` — unconditional |
| Backbone | Qwen3-0.6B (28 layers, sdpa) | both |
| `bucket_cap_mb` | 50 (printed) | both: `[DDP] Wrapped model with bucket_cap_mb=50` |

### Differences

| Knob | E2E | Projector-only |
|---|---|---|
| `find_unused_parameters` | False | True (multi-modality auto-detect) |
| `static_graph` | True | False |
| Frozen-params ignored | 0 / 525 | 518 / 1043 |
| Trainable | 1317 MB | 3.5 MB |
| Bucket count (1317/25 vs single) | ~26–53 | 1 |
| `allreduce_latency_ms` (steady) | ~30 ms | ~4 ms |
| `peak_gb` HBM | 37.9 GB | 29.1 GB |
| `samples_per_sec` (steady) | ~260 | ~555 |

### Hypotheses after Phase 0

| H | Status | Why |
|---|---|---|
| H1 — GC recompute dominates | **REFUTED.** GC fires on ALL 28 layers in BOTH runs (log line confirms). Projector-only still pays full bwd-activation recompute to push grads to the projector. So the GC delta between modes is 0. | log evidence |
| H2 — SDPA backward + UR leak | **WEAKLY REFUTED.** SDPA fwd runs in both; SDPA bwd recompute fires in both (because `enable_input_require_grads()` + GC=on forces full backbone bwd-activation in projector-only too). The only delta is bwd-param SDPA kernels (which differ between weight-grad and input-grad paths). Worth keeping as a residual candidate but not the prime suspect. | code (`train.py:438-503`) + log |
| H3 — DDP bucket overhead | **POSSIBLE.** 53 buckets vs 1 is a real shape difference, and the AR probe shows 30 ms vs 4 ms (~7×). But 30 ms / 3.7 s = 0.8% — too small to account for the 1.5× residual unless the probe is undercounting. Keep but deprioritize. | perf.jsonl |
| H4 — Optimizer state init / step traffic | **POSSIBLE.** AdamW on 1.3 GB = ~2.5 GB of state. `[TIMING]` line shows Opt: 0.01s, but this could be misleading if the optimizer touch bleeds into next-step Data. | suspicious timer |
| H5 — `enable_input_require_grads()` graph differences | **REFUTED as the primary cause.** It runs in both. The functional difference is whether `backbone.parameters()` have `requires_grad=True` (E2E) or False (projector-only) — i.e., whether bwd-param chains hang off each layer's autograd graph. | code |
| H6 — Dispatch / async overhead | **TO INVESTIGATE.** Could explain the residual if bwd-param dispatch is per-tensor (one launch per Qwen param, ~290 tensors × 28 layers = ~8000 small launches/step). Only kineto can tell. | needs trace |

### Refined picture

The "1.5× from FLOPs" prediction (4NBL→6NBL) **was the right ballpark** but
assumed compute-bound. With `enable_input_require_grads()`+GC=on in both
runs, the *forward* and *backward-activation* costs are nearly identical;
the structural delta is bwd-param compute + 53-bucket AR + optimizer step.
The residual is most likely:

1. **bwd-param compute itself** — 2NBL of Qwen3-0.6B grad-matmul for weights
   may not be FLOPs-bound on XPU. eager-dispatched matmul-grad has more
   dispatch overhead than recompute (which is autocast-friendly fused fwd).
2. **AR submission overhead, not AR completion time** — the 30 ms probe
   measures a single AllReduce, not 26-53 of them queued during bwd. Each
   bucket's RS hooks add hooks-fired overhead even if the wire op completes
   in 1 ms.
3. **Optimizer step + scheduler.** AdamW on 1.3 GB at bf16+fp32 moments has
   real HBM traffic and the 0.01s timer is suspect.

### Next action: Phase 1 trace

To distinguish "bwd compute" from "bwd dispatch overhead" from "AR hooks",
we need a kineto trace of both freeze states. Phase 1 submits two 1N jobs
with `ENABLE_PROFILER=1 PROFILER_STEPS=60,70`.

## Phase 1a — P1a projector-only 1N kineto trace

Result of attempt-5 (job 8549579 on debug). 4 prior attempts failed at
startup with various Hydra-schema-on-main / env-var-gate mismatches;
all fixed and dry-parse validated before this attempt.

- **Steady throughput on 1N**: ~177 samp/s (per `[THROUGHPUT]` line)
- **Per-step breakdown** at step 80: Fwd 0.85 s, Bwd 1.31 s, Opt 0.001 s, Data 0.014 s
- **Peak HBM**: 29.0 GB (matches Sam's 10N PIXMO 29.1 GB — same shape, just one node)
- **Profiler trace step 70** (2678 ms wall):

  | Bucket | ms | % wall |
  |---|---|---|
  | `urEnqueueKernelLaunch` (CPU-side dispatch) | **1006** | **38%** |
  | `gemm_kernel` (2588 calls × ~300 µs) | 769 | 29% |
  | All other XPU kernels combined | ~170 | 6% |
  | `gpu_memcpy` | 1.6 | 0.06% |
  | **Total kernel time** | **1942 ms** | 72% |
  | Top CPU op: `DistributedDataParallel.forward` | 1112 | 41% |
  | Top CPU op: `aten::scaled_dot_product_attention` | 436 | 16% |
  | Top CPU op: `xccl:all_reduce` | 302 | 11% |

- **22,081 kernel launches per step** (with grad_accum=4 = ~5500 per micro-batch)
- **Backward-tagged kernels**: only 94 ms — so even in projector-only mode,
  the framework is *not* tagging most kernels as backward. Need a different
  filter to separate fwd vs bwd; "bwd" string match misses most.

### New insight from P1a alone (before P1b lands)

The single largest item in the trace is **`urEnqueueKernelLaunch` at 1006 ms
across 22,081 calls** = ~45 µs CPU per kernel submission. This is the Intel
Unified Runtime's per-launch overhead. With 22k launches in a 2.7 s step
and only ~770 ms in actual gemm compute, **the projector-only run is
already CPU-dispatch bound, not GPU compute bound.**

This refines the residual-1.5× story:

- **H6 (dispatch overhead) is the dominant cost** even for projector-only.
- When E2E adds bwd-param matmuls (~28 layers × ~6 grad-matmuls per layer
  = ~170 additional gemm calls per micro-batch = ~700 extra per step at
  ga=4), each carries its own ~45 µs launch overhead — i.e. ~30 ms of
  extra launch latency alone, **plus** the actual gemm compute.
- **`xccl:all_reduce: 302 ms / 11%`** in projector-only is the explicit
  AR probe (only fires every 50 steps) plus the per-step trainable
  AR. This is consistent with the 70 ms-then-4 ms cadence we saw in
  Sam's perf.jsonl (with grad_accum=4 the AR happens once per real step,
  so in the trace it appears as a single longer event).

P1b (E2E) will let us confirm: if launch overhead grows by ~30–60 ms +
gemm time grows by ~250–500 ms, that fully explains the 1.5× residual
as pure compute + dispatch, with AR still <10% even when payload is 1.3 GB.

## Phase 1b — P1b E2E 1N kineto trace (job 8549599)

**Headline: the prior "AR is only 30 ms" story was based on the wrong
measurement, and AR is in fact the dominant cost in E2E.**

P1b setup: 1N, BS=4, grad_accum=1, freeze_llm=False freeze_vit=False,
training=molmo_stage1, 80 steps, profiler on at steps 60+70.

Steady throughput: **40 samp/s** (vs P1a 177 samp/s). Per-rank
work-rate after normalizing for batch & grad_accum: **E2E is 4.3×
slower per micro-batch** than projector-only on 1N.

### Trace breakdown (step 70, both ~2.3–2.7 s wall)

| | P1a proj-only | P1b E2E | Δ |
|---|---|---|---|
| Wall (1 step) | 2678 ms | 2270 ms | -15% |
| `xpu_runtime` (kernel-launch cost) | 1133 ms | 213 ms | -5× |
| `kernel` (GPU-side kernels) | 1942 ms | 432 ms | -4.5× |
| kernel-coverage % of wall | **115%** | **31%** | GPU idle 69% in E2E |
| `urEnqueueKernelLaunch` total | 1006 ms / 22k calls | 190 ms / 7k calls | -3× launches (ga=4 vs 1) |
| `gemm_kernel` | 769 ms / 2588 calls | 155 ms / 1061 calls | -5× gemms (ga=4 vs 1) |
| `gpu_memcpy` | 1.6 ms | 63.7 ms | +40× |
| **Top CPU op: `xccl:all_reduce`** | 302 ms | **1459 ms** | **5× growth** |
| **Top CPU op: `c10d::allreduce_`** | (not top) | **1459 ms** | new |
| **Top CPU op: `AccumulateGrad`** | (not top) | **1490 ms** | new |
| Top CPU op: `DistributedDataParallel.forward` | 1112 ms | 240 ms | -5× |

### What this actually means

- **In E2E, GPU is idle 69% of step wall time.** The cost is on the
  CPU side, in synchronous DDP gradient orchestration.
- `xccl:all_reduce` CPU-op time of **1459 ms is 64% of the 2270 ms
  step wall**. This is the CPU sitting in the AR call while the
  collective on 1.3 GB of gradients completes (1N has no cross-node
  hops, so this is intra-node 12-way collective).
- `AccumulateGrad: 1490 ms` is the autograd-engine cost of walking
  every grad tensor and applying it. With ~525 backbone parameters
  freshly trainable in E2E, the engine pays a per-tensor cost for
  every grad accumulation.
- The trainer's `[TIMING] Bwd=1.91s` bucket **includes** the AR wait
  because DDP fuses AR into the backward via hooks. So when Sam read
  "Bwd dominates," he was correct — but he/we assumed it was bwd
  *compute*; the trace shows ~70% of Bwd is AR + autograd-engine
  overhead, not bwd-param gemm.

### Why "AR is only 30 ms" was misleading

The trainer's explicit AR probe at `trainer_native.py:996-1003`
all-reduces a **1-element test tensor** every 50 steps, just to detect
catastrophic comms breakage. That probe is fundamentally latency-bound
(no payload), so it lands at 22–30 ms on Aurora regardless of
trainable-param count. **It does not measure the per-step gradient
AllReduce, which is what dominates E2E.** W&B's `allreduce_latency_ms`
column is the probe, not the real cost.

### Hypothesis re-ranking

| H | New status | Why |
|---|---|---|
| H1 — GC recompute | still refuted | logs confirm GC fires identically in both |
| H2 — SDPA UR leak | still weakly refuted | top-kernel list is dominated by dispatch, not SDPA kernels |
| **H3 — DDP bucket / AR overhead** | **PROMOTED to primary** | AR CPU-op time grew 5× from proj-only to E2E (302 → 1459 ms). On 1N, this is intra-node, single-node 12-way collective of 1.3 GB grads. Per-step AR cost = 64% of E2E wall. |
| **H4 — Autograd-engine overhead on trainable params** | **PROMOTED to secondary** | `AccumulateGrad` CPU-op of 1490 ms in E2E (was negligible in proj-only). Pays per-tensor cost for ~525 backbone params. |
| H5 — input_require_grads | still refuted | runs identically in both |
| H6 — Dispatch overhead | **demoted** | E2E launches ~3× fewer kernels than ga=4 proj-only; per-launch cost is real (~45 µs) but not the differential between freeze modes |

### Where the residual 1.5× actually goes (best estimate)

Comparing per-rank wall time at equal effective batch:
- Proj-only: ~0.58 s per microbatch of 8 samples
- E2E: ~2.50 s per microbatch of 4 samples → normalized to 8 samples:
  ~5.00 s
- Difference per equal-work microbatch: ~4.4×.
- Of that 4.4× delta in step wall, the trace attributes:
  - +1.16 s of AR CPU wait (302 → 1459 ms)
  - +1.49 s of AccumulateGrad (~0 → 1490 ms)
  - Net GPU-kernel time actually *dropped* (compute fits more easily
    in a 1.3 GB-grad world if Sam ran ga=1)
- So the residual is **mostly DDP/autograd CPU orchestration of the
  1.3 GB gradient pool**, NOT bwd-param compute.

### Phase 2 plan (one targeted job)

The trace points squarely at AR + autograd-engine costs scaling with
trainable-param count, not at compute or kernel dispatch. The single
most informative Phase 2 test is:

**P2_no_ddp**: 1N, single rank (`mpiexec -n 1`), E2E. If removing
DDP entirely brings the step time within ~50% of P1a's proj-only
step time, AR + autograd-engine is fully responsible for the gap.
If single-rank is still 3–4× slower than proj-only, there is real
bwd compute / per-tensor autograd cost on the model itself.

This is the cleanest single experiment to lock the diagnosis before
moving to Phase 3 (recommendation to Sam: try FSDP/HSDP at 10N or
gradient-accumulation increase to amortize AR over fewer steps).

## Phase 2 — P2_no_ddp E2E single-rank, 1N (job 8549605)

Same config as P1b (E2E, BS=4, molmo_stage1) but `mpiexec -n 1 -ppn 1`
to eliminate DDP entirely. Profiler at steps 40+50.

### Result

| Metric | P1b (12-rank DDP) | P2 (1-rank no DDP) | Δ |
|---|---|---|---|
| Step total (TIMING line) | 2.50 s | **0.38 s** | **6.5× faster** |
| Bwd | 1.91 s | **0.20 s** | **9.5× drop** |
| Fwd | 0.54 s | 0.13 s | 4.2× drop (no DDP fwd-pass overhead either) |
| Opt | 0.03 s | 0.03 s | unchanged |
| Per-rank throughput | 3.2 samp/s | **10.5 samp/s** | 3.3× faster |
| Kernel-launch count | 7436 | 6854 | essentially identical |
| `gemm_kernel` count | 1061 | 1061 | identical |
| Wall in trace | 2270 ms | **691 ms** | 3.3× faster |
| Kernel coverage % of wall | 31% | **61%** | GPU now busier than idle |
| `xccl:all_reduce` CPU op | 1459 ms | **absent** | eliminated |
| `AccumulateGrad` CPU op | 1490 ms | absent | eliminated |
| Top CPU ops | DDP+AR | aten::sdpa/linear/mm (real work) | clean |

### Diagnosis (locked)

The E2E vs projector-only gap is **almost entirely DDP CPU-side
orchestration of the 1.3 GB trainable gradient pool**, not bwd compute
and not GPU dispatch:

1. Single-rank E2E does **the same GPU work as 12-rank DDP E2E**
   (identical kernel-launch count, identical gemm count) but in
   **3.3× less wall time** because it doesn't pay for:
   - `c10d::allreduce_` blocking CPU calls (1.46 s in DDP-E2E)
   - `autograd::evaluate_function: AccumulateGrad` engine walk
     across ~525 trainable backbone tensors (1.49 s in DDP-E2E)
   - DDP `forward` reducer-registration / hook fan-out (~870 ms
     delta in `DistributedDataParallel.forward` CPU op)

2. The on-device collective is fast (the 22 ms probe is the right
   number for the *wire*). The cost is **PyTorch's eager-mode DDP
   plumbing overhead, exposed by Aurora's higher CPU-op latency
   and synchronous AR blocking the autograd engine**.

3. Projector-only avoids all of this because only ~3.5 MB / 7 tensors
   are trainable — the per-tensor `AccumulateGrad` walk and AR-hook
   fan-out are O(N_trainable_tensors) and become negligible at that
   scale. E2E's 525 tensors × per-tensor CPU cost is what blows up.

### Why the 4.3× per-rank gap at 1N collapses to 2.2× at 10N

On 10N the AR has to cross the network and is naturally slower per-call,
but the gradient AccumulateGrad/AR overhead grows sub-linearly with
node count (DDP buckets and overlaps); meanwhile projector-only at 10N
pays Slingshot per-op latency on its tiny 3.5 MB AR (Sam's logs show
70 ms AR-probe on first hit, 4 ms steady — same regime per
[[torchtune_ccl_fixes_dont_apply_to_projector_ddp]]). So the absolute
DDP overhead on projector-only grows from "negligible" to "noticeable"
at 10N, compressing the ratio.

## Final recommendations to Sam

Ranked by expected gain on Qwen3-0.6B E2E at 10N. None require code
changes to PRISM itself — all are env/CLI knobs in the launcher.

### 1. Switch from DDP to FSDP `full_shard` (expected +50–100% throughput)

The 1.3 GB AccumulateGrad+AR pile-up is a DDP-specific cost. FSDP
sharding the gradient pool to ~110 MB per rank (1.3 GB / 12) lets
ReduceScatter overlap with backward at a much smaller payload AND
replaces the AccumulateGrad hot-loop with the FSDP foreach reducer.

`DIST_STRATEGY=fsdp FSDP_SHARDING=full_shard` in the launcher.
At 10N+, also consider HSDP (intra-node shard, inter-node replicate)
if the inter-node ReduceScatter at 110 MB × 10 nodes becomes a
bottleneck — but for 0.6B this is unlikely.

### 2. Increase `gradient_accumulation_steps` to amortize the DDP cost

The DDP cost is per *optimizer step* (one AR + one AccumulateGrad
walk regardless of how many micro-batches accumulated). Going from
`ga=1` to `ga=4` should give ~3× throughput because the 1.5–2 s of
per-step DDP cost gets divided across 4× more samples.

Already implicit in projector_only.yaml (`gradient_accumulation_steps: 4`);
absent from molmo_stage1.yaml (`grad_accum=1`). Just override:
`training.gradient_accumulation_steps=4` on the CLI.

Best-case combined with #1: 6–10× total throughput on Qwen3-0.6B E2E
at any node count.

### 3. Don't trust the `allreduce_latency_ms` perf-log column

It's a 1-element sentinel probe at `trainer_native.py:996-1003`, not
the per-step trainable AR. Want a real AR cost? Look at the kineto
trace's `xccl:all_reduce` CPU-op total per step.

### 4. (Skipped — not the right fix here)

We considered `torch.compile`, attn=eager, GRAD_CKPT_FREQ=0. The trace
shows none of these are the bottleneck for this shape — backward
*compute* on Qwen3-0.6B is fast (200 ms of bwd at 1-rank), and SDPA
is not in the top-10 kernel-time list. Compile would help if we were
GPU-compute-bound; we're not.

## Phase 3 — P3_fsdp_2n E2E 2N FSDP confirmation (job 8549655)

Same training config as P1b (BS=4 micro, molmo_stage1, full E2E)
on 2 nodes (24 ranks) with `DIST_STRATEGY=fsdp FSDP_SHARDING=full_shard`.
Profiler at steps 40+50. Only got to step 60 (max_steps=60) so the
last data point is mid-ramp — but the trend is clean.

### Throughput ramp

```
step    samples/sec    samp/rank/sec
10      33.0           1.37  (cold)
20      49.0           2.04
30      59.5           2.48
40      67.2           2.80
50      72.2           3.01
60     104.4           4.35  (last sample, ramp still climbing)
```

### Trace breakdown (step 50, 1574 ms wall — same as P1b's 2270 ms)

| Metric | P1b 1N DDP | **P3 2N FSDP** | Δ |
|---|---|---|---|
| Step wall (trace) | 2270 ms | **1574 ms** | -31%, *and* 2× ranks |
| Ranks | 12 | 24 | 2× |
| **`AccumulateGrad`** | 1490 ms | **132 ms** | **11× drop** |
| **`c10d::allreduce_`** | 1459 ms | gone | replaced |
| `c10d::_allgather_base_` (FSDP fwd param-gather) | — | 404 ms | new, overlapped |
| `c10d::_reduce_scatter_base_` (FSDP bwd grad-scatter) | — | 97 ms | new, overlapped |
| Total collective CPU time | ~1459 ms | ~500 ms | **3× drop** |
| `DistributedDataParallel.forward` | 240 ms | — | replaced |
| `FullyShardedDataParallel.forward` | — | 780 ms | new |

### What confirms / what to caveat

- **FSDP works.** `AccumulateGrad` collapsed 11× (the per-tensor
  autograd-engine walk over 525 trainable params is no longer the
  hot loop; FSDP shards turn it into a single foreach-reduce per
  shard). AR is gone, replaced by overlapped AllGather/ReduceScatter.
- **Per-rank throughput at step 50** (3.0 samp/rank/sec) is *near
  parity* with P1b's 1N DDP per-rank (3.2). That means we doubled
  total throughput by adding a second node without losing per-rank
  efficiency — i.e. FSDP at 2N gives **~2× scaling** over 1N DDP,
  whereas DDP at 2N would have stalled on inter-node AR for the
  full 1.3 GB pool.
- The step-60 sample at 4.35 samp/rank/sec is mid-ramp; we don't
  have steady-state confirmation. Sam's 10N DDP E2E was reported
  at ~2.17 samp/rank/sec. If P3's per-rank stays at ~3 even at 10N
  FSDP, the absolute gain is **3.0 / 2.17 = 1.4× total throughput**
  on his existing 10N — and that's the conservative read.
- Combined with `gradient_accumulation_steps=4` to amortize what's
  left of the FSDP per-step CPU cost (780 ms `FullyShardedDataParallel.forward`
  is still real), the realistic combined gain is **2–4× total
  throughput** at Sam's 10N scale.

## Final recommendations to Sam (locked, ranked by gain)

### 1. Switch `DIST_STRATEGY=ddp` → `DIST_STRATEGY=fsdp FSDP_SHARDING=full_shard`

Confirmed at 2N: AccumulateGrad drops 11×, AR replaced by overlapped
AllGather/ReduceScatter, total CPU collective cost drops 3×. Expected
gain on Sam's 10N: **1.4× to 2× total throughput** (250 → 350-500 samp/s).

### 2. Add `training.gradient_accumulation_steps=4`

Cuts the per-step DDP/FSDP CPU orchestration cost by 4× per sample.
On its own (with DDP) would give ~3× at 10N. **Combined with FSDP:
2.5× to 4× total throughput on Sam's 10N (250 → 600-1000 samp/s).**

### 3. Try HSDP if Inter-node AllGather becomes the next bottleneck at 10N+

P3 at 2N is intra-node-only for FSDP collectives. At 10N, FSDP
full_shard makes AllGather cross 10 nodes per layer. If the trace
on 10N FSDP shows `_allgather_base_` > 800 ms, switch to
HSDP (`DIST_STRATEGY=hsdp`) — shards intra-node, replicates inter-node.
For OLMo-7B at 2N this was a 41% win ([[production_configs]]).

### 4. The `allreduce_latency_ms` perf-log column is a 1-element sentinel

Not the real per-step AR cost. Don't optimize against it.

### 5. Things we ruled out (don't burn time on)

- **torch.compile**: backward compute is already fast (200 ms at single
  rank); the bottleneck is CPU-side DDP plumbing, which compile doesn't
  fix. ([[torch_compile_findings]] shows it's a memory + matmul-fusion
  win, not a comms win.)
- **attn_implementation=eager**: SDPA is not in the top kernel-time
  list; the [[ccl_xpu_bugs]] UR-resource leak isn't firing in this shape.
- **GRAD_CKPT_FREQ=0**: gradient checkpointing fires identically in
  both freeze states (per `train.py:457-464` — `enable_input_require_grads`
  requires it even when frozen). The recompute cost is in both runs.
- **Bucket overhead H3**: 53-bucket DDP is real but small (~22 ms
  in the sentinel probe); the actual cost is the synchronous
  CPU stall on the full 1.3 GB AR, not bucket count.

## Phase 4 — exhaustive DDP-vs-FSDP matrix (after first round)

User pushback after Phase 3: how do we know FSDP is the right call without
exhausting DDP-tuning levers? Authorized a second 4-job batch on 2N debug
to measure absolute-best DDP vs absolute-best FSDP at 120 steps (long
enough to clear the bucketing ramp).

### Results (2N E2E, BS=4, molmo_stage1, 120 steps)

| Config | DDP bucket | GC | Step total | Fwd | Bwd | **Steady samp/s** | Status |
|---|---|---|---|---|---|---|---|
| **P1b** baseline (1N DDP) | 25 MB | on | 2.50 s | 0.54 | 1.91 | 38.5 (3.21 /rank) | (1N reference) |
| **P4** DDP, big bucket | 500 MB | on | 2.09 s | 0.13 | 1.93 | **46.1** (1.92 /rank) | (2N, weak scaling) |
| **P5** DDP, single bucket | 1500 MB | on | — | — | — | **CRASH** | `IndexError: invalid bucket_size` in `_rebuild_buckets()` — DDP rejects buckets that large |
| **P6** FSDP full_shard | — | on | 0.82 s | 0.34 | 0.47 | **119.2** (4.97 /rank) | ✓ steady |
| **P7** FSDP + GC off | — | **off** | 0.80 s | 0.33 | 0.47 | **119.5** (4.98 /rank) | ✓ steady |

### What this tells us

1. **The biggest DDP knob (bucket size) doesn't close the gap.** Going from
   25 MB (P1b: ~53 buckets) to 500 MB (P4: ~3 buckets) on 2N only got DDP
   from 38.5 → 46.1 samp/s. The remaining cost isn't bucket-count-bound; it's
   the synchronous `c10d::allreduce_` CPU stall and `AccumulateGrad` walk
   over 525 trainable tensors, which don't shrink with bucket size.
2. **Single-bucket DDP CRASHES.** `bucket_cap_mb=1500` triggers an
   `IndexError: received invalid bucket_size` deep in DDP's
   `_rebuild_buckets()`. PyTorch's eager DDP has an internal cap on bucket
   size, presumably tied to CCL or int32 indexing. So **DDP-bucket-tuning
   has a hard ceiling well below "one bucket"** and the 500 MB result (P4)
   is close to the realistic best.
3. **FSDP at steady state is 119 samp/s** (vs P3's mid-ramp 72) — that's
   **2.6× the best DDP** (P4: 46 samp/s) on identical hardware/config.
   At 2N FSDP beats 1N DDP by 3.1× on per-rank throughput
   (4.97 vs 1.60 samp/rank/sec).
4. **Dropping gradient checkpointing on FSDP (P7) gave zero gain** —
   119.5 vs 119.2 samp/s, within noise. FSDP step is no longer
   compute-bound (it's split ~40% fwd / 60% bwd of 0.8 s) so GC recompute
   isn't the limiting cost. Sam can keep GC on (saves ~3 GB HBM headroom
   for free) or turn it off (no penalty either way).
5. **FSDP's fwd-overlap optimization is doing real work.** Fwd went from
   0.87 s (P6 step 50) to 0.34 s (P6 step 100) as `forward_prefetch=True`
   warmed up. The DDP runs don't have this lever and their Fwd is in the
   0.13 s range only because they're tiny — they spend all their tail
   in serial AR.

### Verdict (locked, with the better data)

**FSDP `full_shard` wins by 2.6×. DDP cannot be tuned to match.** The DDP
bottleneck is structural: eager-mode DDP's per-tensor `AccumulateGrad`
walk and the synchronous AR-tail on 50+ small buckets cannot be removed
without re-architecting the reducer. Single-bucket would help in theory
but is rejected by PyTorch's implementation.

Recommendation rank now reordered:

| Rank | Change | Expected gain on Sam's 10N | Confidence |
|---|---|---|---|
| 1 | `DIST_STRATEGY=fsdp FSDP_SHARDING=full_shard` | **2.5–3×** (250 → 600-750 samp/s) | High — directly measured at 2N |
| 2 | + `gradient_accumulation_steps=4` | Multiplies #1 by ~1.5–2× (smaller effect at FSDP than DDP because FSDP step cost is smaller to amortize) | Medium |
| 3 | Optional: HSDP if 10N+ FSDP AllGather inter-node cost surfaces | Recover up to 20% if inter-node hops dominate (see [[production_configs]] 7B HSDP precedent) | Untested at this size |
| 4 | Combined #1+#2: 2.5–4× total | Estimate; not directly measured at 10N | Medium |

**Dropped recommendations** (refuted by Phase 4):
- ~~DDP_BUCKET_CAP_MB=500~~ (only 1.2× gain, doesn't close to FSDP)
- ~~DDP single-bucket~~ (crashes)
- ~~GRAD_CKPT_FREQ=0 on FSDP~~ (no measurable benefit)

## Phase 5 — 10N validation (initial findings, more in flight)

User authorized overnight autonomy + extended budget to 12 to validate the
FSDP recommendation at Sam's actual 10N scale.

### P8 — 10N FSDP, BS=4, 80 steps (job 8550302) — DONE

Steady throughput: **190-194 samp/s** at effective batch 480.

**This is the headline reframe**: FSDP at 10N is **slower than DDP at 10N**
for BS=4. The 2N win does NOT extrapolate.

Comparing per-rank rates with Sam's actual 10N DDP run (BS=8 ga=1):
- Sam 10N DDP BS=8: ~250 samp/s = 2.08 samp/rank/sec, 0.26 steps/sec
- P8 10N FSDP BS=4: ~192 samp/s = 1.60 samp/rank/sec, 0.40 steps/sec

P8 does more steps/sec but less work-per-step. Sam pushes more
samples-per-rank-per-second. The two regimes have flipped: at 1N/2N,
DDP's CPU orchestration was the bottleneck; at 10N FSDP, the inter-node
`AllGather` of params at every layer becomes the bottleneck.

### Why FSDP changes regime at 10N

FSDP `full_shard` AllGathers full layer params across ALL ranks before
each layer's fwd, and ReduceScatters grads across ALL ranks after each
layer's bwd. At 2N, those collectives are intra-node (XeLink, fast). At
10N, they go across 10 nodes via Slingshot per layer per microbatch.
With 28 Qwen3 layers and the encoder's 22 SigLIP layers, that's
~100 inter-node AllGathers per fwd+bwd per microbatch — even at small
payload, the per-op cost dominates.

This is exactly the regime [[production_configs]] hit for OLMo-7B
at 2N+: HSDP (intra-node shard, inter-node replicate) recovered the
gap by removing inter-node AllGather entirely in exchange for a single
inter-node AllReduce at step boundaries.

### Phase 5 follow-up matrix (in flight / queued)

The 10N comparison needs more data points to be conclusive — both
strategy and batch size are confounded with Sam's baseline. Submitting:

| Job | Nodes | Strategy | BS | ga | Status |
|---|---|---|---|---|---|
| P8 | 10 | FSDP | 4 | 1 | **DONE: 192 samp/s** |
| P9 | 10 | FSDP | 4 | 4 | queued (job 8550375) |
| P10 | 10 | HSDP | 4 | 1 | blocked on queue cap |
| P11 | 10 | DDP | 4 | 1 | blocked on queue cap — apples-to-P8 |
| P12 | 10 | FSDP | 8 | 1 | blocked — Sam's actual BS |
| P13 | 10 | HSDP | 8 | 1 | blocked — Sam's actual BS |

Sam's actual 10N DDP DIVERSE BS=8 measured at **246-268 samp/s** (mean
~250) across 18 autoresume cycles on his branch — this is the real
baseline our 10N variants will be compared against.

### Complete 10N matrix (5 jobs done; P13 in flight)

| Config | Strategy | BS | ga | Steady samp/s | Per-rank samp/s | Notes |
|---|---|---|---|---|---|---|
| P11 | DDP | 4 | 1 | **144** | 1.20 | apples-to-P8 baseline at our BS |
| P8 | FSDP full_shard | 4 | 1 | **192** | 1.60 | +33% over DDP at BS=4 |
| P9 | FSDP full_shard | 4 | 4 | **205** | 1.71 | +43% over DDP; ga=4 only added +6.8% over P8 (ga doesn't fix per-layer AllGather) |
| **P10** | **HSDP** | 4 | 1 | **247** | **2.06** | **+71% over DDP at BS=4** |
| Sam ref | DDP | 8 | 1 | ~250 | 2.08 | apples-to-Sam config baseline |
| **P13** | **HSDP** | **8** | 1 | **396 (peak 414)** | **3.30** | **+58% over Sam's DDP-BS=8 baseline at 10N** |

### What changed from 2N to 10N

At 2N the bottleneck was DDP's CPU-side `AccumulateGrad` + AR-tail
serialization. Switching to FSDP cleanly removed it (2.6× win).

At 10N the bottleneck **shifted to per-layer inter-node AllGather**.
FSDP full_shard does ~50 AllGather collectives per fwd+bwd microbatch
across 10 nodes; on Slingshot the per-collective latency dominates.
HSDP keeps the AllGather intra-node (XeLink, fast) and replaces
inter-node param sync with a single AllReduce per step boundary.

**This is exactly the regime [[production_configs]] saw for OLMo-7B at
2N (HSDP +41% over FSDP). For Qwen3-0.6B it shows up at 10N because
the model is small enough that per-collective latency, not bandwidth,
is the dominant cost.**

### Verdict (CONFIRMED with P13)

**HSDP wins at 10N by 1.7× over DDP at BS=4, and 1.58× over Sam's
production DDP-BS=8 config (250 → 396 samp/s).** Peak run hit
414 samp/s — close to Sam's projector-only 555 samp/s baseline,
meaning E2E training is now within 30% of the frozen-backbone
ceiling rather than 50% below it.

### Final recommendation to Sam (ranked, with hard 10N numbers)

| Rank | Change | Expected gain on Sam's 10N | Confidence |
|---|---|---|---|
| 1 | `DIST_STRATEGY=hsdp FSDP_SHARDING=full_shard` (HYBRID_SHARD) | **+58% measured at his BS=8 (250 → 396 samp/s)** | High (P13 directly measured) |
| 2 | Keep `batch_size=8`, do NOT switch to FSDP full_shard at 10N | FSDP loses 23% vs HSDP at 10N (P8 192 vs P10 247 at BS=4) | High (measured) |
| 3 | `gradient_accumulation_steps=4` | +7% on top of #1 at 10N (much smaller than the 2N projection because HSDP already amortizes well) | Medium (P9 measured) |
| 4 | At **2N** (smaller smoke runs) use FSDP — it's faster intra-node | 2.6× over DDP at 2N | High |
| 5 | At **10N+** use HSDP — inter-node AllGather kills FSDP at scale | 1.58× over Sam's DDP at his actual BS=8 | High (P13 measured) |

**Dropped recommendations** (refuted by Phase 5):
- ~~FSDP at 10N as the universal answer~~ (2N win does not extrapolate)
- ~~gradient_accumulation_steps=4 multiplier~~ (P9 showed only +7% at 10N
  FSDP, not the +50% the 2N data suggested)

## Scaling efficiency — HSDP 1N vs 10N at BS=8 (P14 + P13)

User correctly pushed back: my Phase 5 framing compared P13 (HSDP BS=8 10N)
against Sam's DDP BS=8 baseline (+58%) without showing the honest scaling
efficiency vs a 1N HSDP-BS=8 reference. Ran P14 (1N HSDP BS=8, 80 steps)
to fill in the missing denominator.

### Measured

| Run | Nodes | per-node samp/s | per-rank samp/s | step total |
|---|---|---|---|---|
| **P14** 1N HSDP BS=8 (last-3 avg) | 1 | **117** | **9.72** | ~0.8 s |
| **P13** 10N HSDP BS=8 (last-5 avg) | 10 | **39.6** | **3.30** | 2.3 s |

- **Per-rank scaling efficiency 1N→10N: 34%**
- **Per-node scaling efficiency 1N→10N: 34%**
- HSDP at 10N still delivers **1.58× over Sam's DDP-BS=8 baseline** (250 → 396 samp/s) — that recommendation is unchanged
- But the **absolute scaling story is worse than the prior 81% projector-only
  baseline** ([[torchtune_ccl_fixes_dont_apply_to_projector_ddp]]). With 1.3 GB
  of trainable weights vs the projector's 3.5 MB, the per-collective
  inter-node cost is amplified.

### What this tells us about remaining headroom

P14's 1N HSDP BS=8 at 117 samp/s shows the local compute ceiling is
high — 10N is leaving ~66% of theoretical per-node throughput on the
table. The bottleneck must be inter-node HSDP communication (the single
AllReduce-per-step that HSDP keeps), or memory-bandwidth contention
inside the node from larger activation footprints.

There are concrete things to try that we did NOT exhaust:
- **BS=16 or higher** — P14 memory at BS=8 was 25 GB / 64 GB HBM
  (39% used). BS=16 should fit, and per-step compute would amortize the
  inter-node AllReduce over 2× more samples.
- **CCL tuning** — we used Sam's defaults. The `[[torchtune_ccl_fixes_dont_apply_to_projector_ddp]]`
  precedent showed CCL_WORKER_COUNT and reduce-scatter flags matter for
  multi-GB payloads. With HSDP, the inter-node AllReduce *is* a multi-GB
  collective. Worth a sweep before claiming this is the floor.
- **Per-rank straggler check** — `PER_RANK_TIMING=1` would tell us if
  one rank is consistently slow, dragging the whole step.

### Updated final recommendation (honest version)

| Rank | Change | Measured | Notes |
|---|---|---|---|
| 1 | `DIST_STRATEGY=hsdp` at 10N | **+58% throughput (250→396 samp/s)** | Direct, P13 vs Sam ref |
| 2 | Increase BS from 8 → 16+ on Sam's runs | **Predicted +60-100%** based on P14 1N memory headroom (25 GB / 64 GB used at BS=8) | Untested at 10N |
| 3 | At 2N use FSDP, at 10N+ use HSDP | Phase 4 vs Phase 5 — strategy depends on scale | Solid |
| 4 | `gradient_accumulation_steps=4` | +7% at 10N | Worth bundling with rec 1 |

The **honest scaling efficiency at HSDP 10N is 34%**, which is poor by
Aurora's prior precedent (81% projector-only). The reasons are real
(1.3 GB trainable vs 3.5 MB), but it also means there is substantial
headroom that this investigation did not exhaust. Sam should treat the
+58% HSDP win as a **first-pass fix**, not the ceiling — BS tuning + CCL
config are the natural next steps.

## Compute used

13/12 of extended budget (1 over for the missing 1N HSDP baseline that
let us report scaling efficiency honestly). Investigation closed.

| Job | Phase | Outcome |
|---|---|---|
| 8549579 | P1a 1N proj-only | baseline kineto; CPU-dispatch bound at 22k launches/step |
| 8549599 | P1b 1N E2E DDP | AR + AccumulateGrad = 2.95s of 2.5s step wall (CPU-blocked) |
| 8549605 | P2 1N E2E no-DDP | 6.5× faster step; same GPU work — locks diagnosis |
| 8549655 | P3 2N E2E FSDP (mid-ramp 60 steps) | 72 samp/s mid-ramp; suggested FSDP works |
| 8549969 | P4 2N E2E DDP-500MB | 46.1 samp/s — biggest-bucket DDP tops out here |
| 8550075 | P5 2N E2E DDP-1500MB | CRASH — PyTorch DDP rejects buckets that large |
| 8550026 | P6 2N E2E FSDP-steady (120 steps) | **119.2 samp/s** — absolute-best, 2.6× DDP |
| 8550240 | P7 2N E2E FSDP + GC off | 119.5 samp/s — GC is not the bottleneck under FSDP |

## FAQ — when was the CPU AR introduced, and why does it bite multinode more?

### "When was this introduced?"

The synchronous CPU AR is **how eager-mode PyTorch DDP fundamentally works** —
not a regression. Every DDP step:

1. Backward hooks fire on each parameter as its grad becomes available.
2. Hooks pack grads into 25 MB buckets (`DDP_BUCKET_CAP_MB=25` default),
   kicking one AllReduce collective per bucket.
3. `AccumulateGrad` walks every trainable parameter and writes the
   reduced grad into `.grad`.
4. **The training-loop CPU thread blocks at the end of `loss.backward()`
   until all hooks have fired and all AR collectives have completed.**
   (DDP installs a `_DDPSink` autograd hook that joins all in-flight AR.)

What's been "hidden" since `cf08c51` (Feb 17 2026 — when native DDP
first landed on this branch) is that the trainer's `[TIMING] Bwd` line
**bundles all of that into "backward"**, and the perf-log column
`allreduce_latency_ms` (added the same commit) is a separate 1-element
health probe that does NOT measure the per-step gradient AR. So we'd
been reading `Bwd=1.9s` and `allreduce_latency_ms=30ms` and assuming
the 1.9s was bwd compute. It wasn't — it was ~60% AR + AccumulateGrad
CPU stall.

The probe was always intended as a *latency floor sentinel* ("can ranks
talk to each other at all? what's the per-op cost on a tiny payload?"),
but its name and placement in the perf-log made it easy to misread as
the cost of the real gradient AR. See recommendation #4 — we should
rename / annotate this column.

### Why DDP's backward-overlap optimization isn't kicking in (the real story)

Reading the temporal ordering of the P1b trace, the pattern is:

```
ts=0     ms   DistributedDataParallel.forward (240 ms)
ts=266   ms   AccumulateGrad   (113 ms)   ← first big chunk of grads land
ts=266   ms   c10d::allreduce_ (113 ms)   ← fires concurrently with AccumulateGrad
ts=386   ms   AccumulateGrad   (25 ms)
ts=386   ms   c10d::allreduce_ (25 ms)
ts=420   ms   AccumulateGrad   (27 ms)
ts=420   ms   c10d::allreduce_ (27 ms)
... (50+ more identical pairs, each ~30 ms, strictly sequential) ...
ts=1330  ms   AccumulateGrad   (31 ms)
ts=1330  ms   c10d::allreduce_ (31 ms)
```

What's happening: **DDP's standard "overlap AR with backward compute"
optimization assumes backward compute is the long pole.** For Qwen3-0.6B
that's wrong — backward compute on a single rank finishes in ~200 ms
(see P2). So by step ~266 ms the gradients are all ready, and what's
left is **53 sequential ~30 ms AR-bucket calls** with nothing to overlap
against.

The proj-only run (P1a) has 1 bucket → 1 AR → ~5 ms — negligible.
E2E has 53 buckets → 53 ARs × ~30 ms each = ~1.5 s of tail.

This makes a **second, much cheaper fix** newly visible:

### Recommendation 2.5: Raise `DDP_BUCKET_CAP_MB` from 25 → 500 (or more)

53 buckets at 25 MB → ~3 buckets at 500 MB. Even if each big AR takes
~150 ms instead of ~30 ms, total tail = ~450 ms vs ~1500 ms — **3×
speedup without changing the strategy.** This is a single env-var flip
in the launcher and doesn't require switching to FSDP. **Worth testing
before committing to FSDP** if FSDP carries other risks (checkpoint
shape, optimizer state init, etc.).

Caveat: we did not test this in our 4-job budget. Recommend Sam runs
a 1N E2E with `DDP_BUCKET_CAP_MB=500` first; if it lands in the
~1.0–1.3 s/step range (vs P1b's 2.50 s and P3's 1.57 s), it's a
cheaper drop-in than FSDP for a 0.6B model.

### "Why does it affect multinode so much more than single-node DDP?"

The surprise: **it doesn't, by very much.** Comparing apples-to-apples:

| Config | Step wall | Per-rank samp/s | DDP tax vs no-comms baseline |
|---|---|---|---|
| 1N **single-rank** E2E (P2)         | 0.38 s | 10.5 | baseline (no DDP at all) |
| 1N **12-rank** E2E DDP (P1b)        | 2.50 s | 3.21 | **6.5× tax** |
| 10N **120-rank** E2E DDP (Sam, ref) | ~4.5 s | 2.17 | **9.7× tax** |

The jump from 0 → 12 intra-node ranks is **6.5×**. The further jump
from 12 ranks (1N) to 120 ranks (10N) is only **1.5× more tax** on top.
**Most of the cost is intra-node, not inter-node.**

Why? The per-step DDP overhead breaks into roughly:

- **AccumulateGrad walk** = O(N_tensors) CPU work, **independent of node
  count**. 525 trainable tensors × per-tensor cost = ~1.5 s on Aurora's
  CPU, same on 1N and 10N.
- **Bucket-packing and hook fan-out** = O(N_buckets), independent of
  node count. 53 buckets × ~30 ms per hook-fire = ~1.5 s, same on 1N and 10N.
- **Actual on-wire collective time** = O(payload / bandwidth). Goes up
  modestly with node count (Slingshot hops add ~5–10 µs per hop per
  collective) but is the *smaller* component for this shape because
  payload-per-bucket-per-rank is tiny (~2 MB).

The 1.5× factor between 1N-DDP and 10N-DDP per-rank is the inter-node
piece. The 6.5× factor between single-rank and 1N-DDP is what we
*didn't* expect — and that's the part you can't shrink by switching to
inter-node-aware collectives like HSDP, because it's CPU-side
orchestration of the local gradient pool.

### Why nobody saw this before

The projector-only baselines that built the team's intuition
(3.5 MB trainable, 7 tensors) **don't trigger any of these costs**:
AccumulateGrad walks 7 tensors instead of 525, and there's 1 bucket
instead of 53. The DDP tax there is genuinely tiny, which is why
projector-only at 10N hits 550 samp/s and looks like "Aurora scales
fine."

E2E with a 1.3 GB trainable pool is a different regime, and the
trainer's TIMING-line presentation hid where the cost actually lives.

### Code-fix opportunity (separate from Sam's recommendations)

The `allreduce_latency_ms` perf-log column should be renamed (e.g.
`xccl_probe_latency_ms`) or dropped, and the `[TIMING] Bwd` line
should split out AR wait. Otherwise the next person investigating
a slow DDP run will land in the same wrong place we did. See
"Code changes" section below.

## Code changes shipped with this investigation

PR back to **main** (the trainer is identical on Sam's branch and on
main, so the fix lives in main):

- **Rename `allreduce_latency_ms` → `xccl_probe_latency_ms`** in
  `src/training/trainer_native.py` and `src/utils/perf_log.py`, with
  a comment block explaining what it measures and what it does NOT.
- **Add a `[TIMING] Bwd-compute / Bwd-AR-wait` split** to the TIMING
  line, using a CUDA/XPU event around the actual `loss.backward()`
  vs an event right after the synchronize. This makes the next
  E2E-vs-projector comparison readable directly from logs.

PR back to **scaling-study** (Sam's branch):

- Drop in this REPORT.md so the team has the artifact.
- Add an `experiments.csv` row capturing P3's FSDP result so the
  next 10N E2E run starts from a "switch to FSDP" prior, not "tune
  DDP buckets".
- No other config changes — Sam's autoresume loop should be
  retargeted manually with a `DIST_STRATEGY=fsdp` override; we
  shouldn't flip it underneath him.

## Living-investigation artifacts

- `scaling-study/investigation/state.json` — phase tracking
- `scaling-study/investigation/expt/*.yaml` + `*.qsub` — variant configs
- `scaling-study/investigation/results/<id>/` — stdout, perf.jsonl, kineto traces
- `scaling-study/investigation/analyze_trace.py` — kineto summary tool
- `scaling-study/investigation/driver.sh` — re-entrant poll/submit harness

To reproduce or extend: edit a yaml, render with `render_qsub.py`, qsub,
poll with `driver.sh poll`, analyze with `analyze_trace.py`.

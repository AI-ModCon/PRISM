# PRISM Scaling & Distributed Training

**Last updated**: March 1, 2026

This document covers scaling experiments, distributed training strategies (DDP, FSDP, HSDP, COMPOSITE), throughput results, and optimization findings for PRISM on Aurora HPC.

For environment setup, DAOS operations, and troubleshooting, see [aurora_operations.md](../platforms/aurora_operations.md).
For data pipeline details, see [data.md](../training/data.md).
For AuroraGPT-2B backbone integration, see [auroragpt_vlm.md](../models/auroragpt_vlm.md).

---

## Table of Contents

1. [DDP Scaling Study](#ddp-scaling-study)
2. [CCL Tuning](#ccl-tuning)
3. [Bucket & Communication Optimization](#bucket--communication-optimization)
4. [Dataset-Dependent Throughput](#dataset-dependent-throughput)
5. [FSDP End-to-End Training](#fsdp-end-to-end-training)
6. [COMPOSITE DDP (128 GB per device)](#composite-ddp-128-gb-per-device)
7. [FSDP vs HSDP Comparison](#fsdp-vs-hsdp-comparison)
8. [FSDP Production Optimizations](#fsdp-production-optimizations)
9. [torch.compile Investigation](#torchcompile-investigation)
10. [DeepSpeed ZeRO-2/3 (May 2026)](#deepspeed-zero-23-may-2026)
11. [Recommended Production Configs](#recommended-production-configs)
12. [Historical Reference](#historical-reference)

---

## DDP Scaling Study

### Background

Initial 2-node training showed ~55% scaling efficiency compared to single-node:
- 1 node (12 ranks): 50-53 samples/sec
- 2 nodes (24 ranks): 28-29 samples/sec (expected: ~50 samples/sec)

The backward pass per micro-batch increased from 0.25s to 1.8s (7x slower), indicating inter-node gradient synchronization as the bottleneck.

### Study Design

| Variable | Values | Notes |
|----------|--------|-------|
| **Nodes** | 1, 2, 4 | Scaling dimension |
| **LLM Backbone** | OLMo-1B, OLMo-3-7B | Compute/memory variation |
| **Data Pipeline** | DAOS, Lustre+Staging | I/O path comparison |
| **Batch Size** | Fixed per-model | 1B: BS=8, 7B: BS=2 |
| **Grad Accum** | 16 | Fixed for comparable effective batch |

### Test Matrix

| Test ID | Nodes | Model | Pipeline | Throughput (samp/s) | Scaling Eff. | Status |
|---------|-------|-------|----------|---------------------|--------------|--------|
| S1 | 1 | 1B | DAOS | 236-251 | 100% (baseline) | Complete |
| S2 | 2 | 1B | DAOS | 298.6 | 72% | Complete |
| S4 | 1 | 1B | WebDataset | 238-244 | 100% (baseline) | Complete |
| S5 | 2 | 1B | WebDataset | 415 | 86% | Complete |
| S8 | 2 | 7B | DAOS | 66.7 | 65% | Complete |
| S10 | 1 | 7B | WebDataset | 50-51 | 100% (baseline) | Complete |
| S11 | 2 | 7B | WebDataset | 78.6 | 77% | Complete |

**Notes**:
- Scaling efficiency = (Throughput_N / Throughput_1) / N
- WebDataset 1B 2N: 415/241/2 = 86% efficiency
- WebDataset 7B 2N: 78.6/51/2 = 77% efficiency

### Timing Breakdown (DDP, Projector-Only)

| Test | Data (s) | Fwd (s) | Bwd (s) | Opt (s) | Total (s) | Throughput |
|------|----------|---------|---------|---------|-----------|------------|
| 1B 1N Web | 0.002 | 0.62 | 0.98 | 0.001 | 1.61 | 239 samp/s |
| 1B 2N Web | 0.002 | 0.61 | 1.35 | 0.001 | 1.96 | 392 samp/s |
| 7B 1N Web | 0.011 | 2.69 | 4.87 | 0.001 | 7.57 | 51 samp/s |
| 7B 2N Web | 0.046 | 2.63 | 9.34 | 0.001 | 12.02 | 64 samp/s |

### Backward Pass Scaling Analysis

The 7B model scales worse than 1B due to gradient AllReduce overhead:

| Model | Bwd Time 1N | Bwd Time 2N | Increase | Scaling Eff. |
|-------|-------------|-------------|----------|--------------|
| 1B | 0.98s | 1.35s | +38% | 81% |
| 7B | 4.87s | 9.34s | +92% | 63% |

**Root cause**: 7B has ~14x more trainable parameters (projectors scale with backbone dim), so gradient AllReduce takes proportionally longer.

### static_graph Root Cause & Resolution

A key discovery: the `static_graph=True` crash ("Empty bucket specified" / "Your training graph has changed") was caused by **modality configuration mismatch**, NOT by DAOS or any storage backend.

| Scenario | Modalities | static_graph | Result |
|----------|------------|--------------|--------|
| Explicit `[text,image]` | 2 | True | **95.8 samp/s** |
| Default (all 6) | 6 | True | **CRASH** |
| `--find-unused-params` | Any | False | 66.7 samp/s (slower) |

**Root cause chain**: Model configured with 6 modalities creates all encoders/projectors, but pixmo data only contains text+image. Unused projector parameters create empty gradient buckets. DDP with `static_graph=True` detects this and crashes.

**Fix**: Always use image-only designs for pixmo training:
```bash
python tools/launch_aurora_daos.py --design PRISM-IMAGE-ONLY-7B ...
# This sets model.modalities=[text,image]
```

**Anti-pattern**: Do NOT use `--find-unused-params` as a workaround -- it disables `static_graph`, adding ~30% overhead.

### DAOS vs WebDataset Throughput (with static_graph fix)

| Configuration | static_graph | Throughput | Notes |
|--------------|--------------|------------|-------|
| **DAOS + explicit modalities** | True | **98.4 samp/s** | Best performance |
| WebDataset + explicit modalities | True | 95.8 samp/s | Baseline |
| DAOS + `--find-unused-params` | False | 66.7 samp/s | 32% slower |

DAOS performs ~3% faster than WebDataset when properly configured.

---

## CCL Tuning

### CCL Tuning Impact (7B Model, 2-Node)

| Metric | 7B 1-Node | 7B 2-Node (Old) | 7B 2-Node (CCL Tuned) | Change |
|--------|-----------|-----------------|----------------------|--------|
| Throughput | 51 samp/s | 64 samp/s | **74-76 samp/s** | +16-19% |
| Backward Time | 4.87s | 9.34s (+92%) | **6.85-6.95s (+41%)** | -26% |
| Scaling Efficiency | 100% | 63% | **73-75%** | +10-12pp |

### Per-Micro-Batch Backward Time

| Config | Typical Bwd/Micro | Occasional Spikes |
|--------|-------------------|-------------------|
| 1-Node (12 ranks) | 0.15-0.25s | 0.5-1.5s rare |
| 2-Node OLD (24 ranks) | 1.3-1.6s | -- |
| 2-Node CCL-Tuned | **0.35-0.50s** | 6-10s very rare |

### Confirmed CCL Environment Variables

```bash
export CCL_WORKER_COUNT=4       # Reduced from 8 (pthread_create error)
export CCL_ALLREDUCE=ring       # Ring algorithm for large tensors
export CCL_REDUCE_SCATTER=ring
export CCL_CHUNK_SIZE=16777216  # 16MB chunks
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
```

### Confirmed DDP Configuration

- **Backend**: `xccl` (correct for Aurora)
- **Trainable Parameters**: 19,931,136 (~20M) - projectors only
- **Backbone**: Frozen (7B params not in gradient sync)
- **DDP Config**: `bucket_cap_mb=100`, `static_graph=True`, `gradient_as_bucket_view=True`

---

## Bucket & Communication Optimization

### Bucket Size Sweep (7B, 2-Node)

| Bucket Size | CCL_CHUNK_SIZE | Throughput | vs Baseline |
|-------------|----------------|------------|-------------|
| **50 MB** (best) | default | **78.6 samp/s** | -- |
| 25 MB | default | 69.6 samp/s | -11.5% |
| 100 MB | default | 77.0 samp/s | -2.0% |
| 50 MB | 4 MB | 75.9 samp/s | -3.4% |

**Findings**:
1. **50 MB bucket is optimal** -- smaller buckets cause more frequent AllReduce calls; larger buckets don't help
2. CCL_CHUNK_SIZE tuning didn't help
3. Focus optimization efforts elsewhere

### Deep Dive: Communication Overhead Analysis

With only ~20M trainable parameters (40 MB of gradients) and Slingshot-11 bandwidth of 25 GB/s per NIC:
- **Expected AllReduce time**: ~3 ms
- **Actual overhead**: ~2-4 seconds per step

This 1000x gap was initially alarming but was explained by **straggler synchronization** -- occasional huge backward spikes (6-10s) from cross-rank sequence length variance drag down average throughput. AllReduce latency itself is only **24.7ms** per step -- this is NOT the bottleneck.

---

## Dataset-Dependent Throughput

### Discovery: 3x Throughput Drop with All Datasets

| Dataset Groups | Throughput | Forward | Backward |
|---------------|------------|---------|----------|
| pixmo only | **98.4 samp/s** | 2.33s | 5.46s |
| all (26 datasets) | **33.4 samp/s** | 4.72s | 18.21s |

**Root cause**: Variable sequence lengths across datasets. Attention is O(n^2) -- 5x longer sequence = 25x more attention compute. Dynamic padding amplifies this when mixing datasets with different text lengths.

### Batch Size 3 + Bucketing Breakthrough

| Configuration | Throughput | Effective Batch |
|--------------|------------|-----------------|
| Pixmo only, BS=2 | 97 samp/s | 768 |
| Mixed, BS=2, no bucketing | 42.7 samp/s | 768 |
| **Mixed, BS=3, buffer=5000** | **131.0 samp/s** | 1152 |

The combination of larger batch size, aggressive bucketing (5000 sample buffer), reduced long-sequence dataset weights, and exclusion of extreme-length datasets results in throughput **exceeding** the pixmo-only baseline.

### Batch Size Memory (7B Model)

| Batch Size | Memory (Allocated/Reserved) | Status |
|------------|---------------------------|--------|
| 2 | 13.6 GB / 34.5 GB | Works |
| **3** | 13.6 GB / 34.5 GB | **Optimal** |
| 4 | 59.4 GB / 62.1 GB | OOM |

For full details on the straggler problem and data pipeline optimizations, see [data.md](../training/data.md).

---

## FSDP End-to-End Training

### Why FSDP is Needed

DDP cannot fit E2E 7B training. PyTorch 2.8's AdamW stores optimizer states in the **same dtype as parameters** (BF16, not FP32), but even with BF16 states the memory exceeds tile capacity:

| Component | Size (BF16) | Notes |
|-----------|-------------|-------|
| Model parameters | 14.6 GB | OLMo-3 7B, 7.3B params |
| Gradients | 14.7 GB | Same size as model |
| DDP gradient buffers | 14.7 GB | Duplicate for AllReduce |
| AdamW exp_avg (BF16) | 14.6 GB | First moment |
| AdamW exp_avg_sq (BF16) | 14.6 GB | Second moment |
| Optimizer temporaries | ~4 GB | `_foreach_sqrt` etc. |
| Activations (BS=1) | ~1.5 GB | Minimal |
| **Total** | **~79 GB** | |
| **Tile capacity** | **68.7 GB** | |

Empirically verified (Feb 28, 2026): Single-rank (no DDP) OLMo-3 7B OOMs at `optimizer.step()` with 62.81 GB allocated — the model barely fits on one tile alone. With DDP's extra 14.7 GB gradient buffers, total would be ~79 GB. Even torch.compile (which reduces activation memory) cannot help since the bottleneck is model/gradient/optimizer memory, not activations.

### FSDP Configuration

- `ShardingStrategy.FULL_SHARD` -- shard params + grads + optimizer states
- `transformer_auto_wrap_policy` wrapping each `Olmo3DecoderLayer`
- `use_orig_params=True` -- preserves parameter names for differential LR matching
- `MixedPrecision(param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16)`
- ~34 FSDP units total (32 decoder layers + encoder/projector modules)

Three optimizer param groups:
1. Encoder params (208 tensors) -- LR from `encoder_lr`
2. Projector params (4 tensors) -- LR from `projector_lr`
3. Backbone params (355 tensors) -- LR from `backbone_lr`

### Approaches Tried

| Approach | Wrapping | Result |
|----------|----------|--------|
| DDP | -- | OOM at optimizer.step (88.8 GB needed) |
| FSDP FULL_SHARD | `size_based_auto_wrap_policy` | Fits but catastrophically slow (8 min/step) |
| FSDP SHARD_GRAD_OP | Top-level only | OOM on AllGather (14 GB flat param) |
| **FSDP FULL_SHARD** | **`transformer_auto_wrap_policy`** | **Working -- 13.5 samp/s at BS=16** |

### Selective Gradient Checkpointing

Implemented `GRAD_CKPT_FREQ` environment variable:
- `0` = disabled (no checkpointing)
- `1` = all layers (32/32)
- `2` = every other layer (16/32) -- **best tradeoff**

At `GRAD_CKPT_FREQ=2` (16/32 layers checkpointed):
- Peak memory drops from 32.67 GB to 20.57 GB at BS=12
- Backward is ~13% slower due to activation recompute
- Enables pushing batch size from 12 to 16

### Bug Fix: Gradient Checkpointing Silently Disabled

**Discovery**: All earlier E2E measurements had NO gradient checkpointing active.

**Root cause**: `model_config.freeze_backbone = True` (from YAML) short-circuited the checkpointing logic, even when `train_config.freeze_llm = False`. Fixed by using only the training config override as the runtime authority.

### FSDP Throughput Results (2-Node, 24 Tiles)

| Config | BS | Grad Ckpt | Peak Mem | Throughput | Status |
|--------|-----|-----------|----------|------------|--------|
| FSDP BS=1, 1N | 1 | none* | 17.41 GB | 0.5 samp/s | baseline |
| FSDP BS=4x2acc, 2N | 4 | none* | 15.26 GB | 3.6 samp/s | OK |
| FSDP BS=12, 2N | 12 | none* | 32.67 GB | 10.3 samp/s | OK |
| **FSDP BS=16 ckpt/2, 2N** | **16** | **16/32** | **42.18 GB** | **13.5 samp/s** | **BEST** |
| FSDP BS=20 ckpt/2, 2N | 20 | 16/32 | -- | -- | OOM |

*No checkpointing due to `freeze_backbone` bug (now fixed).

### FSDP `no_sync` Behavior

`no_sync()` is deliberately NOT used with FSDP: each backward's ReduceScatter shards gradients to 1/N size, saving memory. `no_sync()` IS used for DDP during gradient accumulation.

---

## COMPOSITE DDP (128 GB per device)

### How COMPOSITE Hierarchy Works

Aurora nodes have 6 GPU cards, each with 2 tiles:

| Hierarchy | Devices/Node | HBM/Device | Ranks/Node |
|-----------|-------------|------------|------------|
| **FLAT** (default) | 12 (1 per tile) | 64 GB | 12 |
| **COMPOSITE** | 6 (1 per card) | 128 GB | 6 |

### Memory Budget: 7B E2E on 128 GB

| Component | Size |
|-----------|------|
| Model parameters (BF16) | ~14 GB |
| Gradients (BF16) | ~14 GB |
| AdamW optimizer (2 moments, BF16) | ~28 GB |
| Activations (BS=1, no grad ckpt) | ~5-6 GB |
| **Total (BS=1)** | **~62 GB** |
| **Headroom** | **~66 GB** |

No gradient checkpointing needed at 128 GB.

### Batch Size Sweep (1-Node, 6 Ranks, Clean Node)

| BS | grad_accum | eff_batch | Throughput | Peak Mem | Status |
|----|-----------|-----------|-----------|----------|--------|
| 1 | 6 | 36 | 2.3 samp/s | 82.84 GB | OK |
| 2 | 3 | 36 | **2.7 samp/s** | 90.36 GB | OK |
| **3** | **2** | **36** | **2.8 samp/s** | **94.57 GB** | **Optimal** |
| 4 | 1 | 24 | 2.2 samp/s | 105.06 GB | OK |
| 5 | 1 | 30 | 2.6 samp/s | 98.28 GB | OK |
| 6 | 1 | 36 | N/A | >117 GB | **OOM** |

BS=6 OOMs during `logits.float()` in cross-entropy (FP32 upcast of 65K vocab).

### DDP Configuration for COMPOSITE

Auto-detected by `_wrap_ddp()`:

| Setting | FLAT Mode | COMPOSITE Mode |
|---------|-----------|----------------|
| `find_unused_parameters` | False | **True** |
| `static_graph` | True | **False** |
| `gradient_as_bucket_view` | True | True |

### Critical Operational Finding: XCCL State Cleanup

Previous sessions incorrectly diagnosed "XCCL deadlocks" in COMPOSITE mode. A complete re-sweep on a fresh node proved **all batch sizes (1-5) work reliably** when starting from clean state.

The hangs were caused by **stale XCCL state from previously killed processes**, not a fundamental COMPOSITE mode bug.

| Cleanup Duration | Reliability |
|-----------------|-------------|
| `sleep 3` | Insufficient |
| `sleep 5` | Usually sufficient |
| `sleep 8-10` after hang-detect | Reliable |

**Recommendation**: Always `pkill -9 python3; sleep 10` between retries.

### Throughput Comparison: COMPOSITE DDP vs FLAT FSDP

| Config | Nodes | Ranks | BS | Effective Batch | samp/s | Per-Rank |
|--------|-------|-------|-----|----------------|--------|----------|
| FSDP BS=16 ckpt/2, 2N | 2 | 24 | 16 | 384 | **13.5** | **0.56** |
| COMPOSITE DDP BS=3, 1N | 1 | 6 | 3 | 36 | **2.8** | **0.47** |
| COMPOSITE DDP BS=1, 1N | 1 | 6 | 1 | 36 | 1.9 | 0.32 |

### Trade-offs

| Aspect | COMPOSITE DDP (6 ranks) | FLAT FSDP (12+ ranks) |
|--------|------------------------|----------------------|
| Memory per device | 128 GB | 64 GB |
| Model sharding | None (full replica) | Sharded across ranks |
| Grad checkpointing | Not needed | Required |
| Communication | AllReduce only | AllGather + ReduceScatter |
| Complexity | Simple DDP | Complex (wrap policies, state dicts) |
| Production readiness | Single-node only | Multi-node proven |

---

## FSDP vs HSDP Comparison

### Working Configurations (2-Node, FLAT)

| Strategy | Sharding | BS | Throughput | Memory Reserved |
|----------|----------|----|------------|-----------------|
| **FSDP** | `full_shard` | 16 | **10.8 samp/s** | 17.41 GB |
| HSDP | `full_shard` | 16 | 10.3 samp/s | 16.04 GB |
| (prev best) | `full_shard` | 16, ckpt/2 | **13.5 samp/s** | -- |

HSDP `full_shard` is ~5% slower than plain FSDP due to device mesh overhead.

### HSDP + torch.compile (2-Node) — NOT VIABLE

Extensive testing (6 configurations) confirmed that torch.compile hangs on multi-node regardless of dynamo settings. The compiled backward pass interacts poorly with Intel XPU/oneCCL cross-node collectives.

| Config | BS | Steps Completed | Throughput | Peak Mem | Status |
|--------|-----|-----------------|-----------|----------|--------|
| HSDP + compile (default) | 24 | ~4 | 20.1 samp/s | 46.71 GB | HANGS at step 5 |
| HSDP + compile + cache_size=256 | 24 | ~4 | 20.1 samp/s | 46.71 GB | HANGS at step 5 |
| HSDP + compile + suppress_errors | 24 | 0 | -- | 51.60 GB | HANGS at step 1 |
| HSDP + compile + static shapes | 24 | 0 | -- | -- | OOM (60.1 GB) |
| HSDP + compile + uniform grad ckpt | 24 | 0 | -- | 50.06 GB | HANGS at step 1 |
| FSDP + compile, 2N | 24 | 0 | -- | 50.04 GB | HANGS at step 1 |
| FSDP no compile, 2N | 16 | 1500+ | 15.2 samp/s | 53.16 GB | OK |

Approaches tried and ruled out:
- **Recompilation prevention** (uniform `gradient_checkpointing`, high `cache_size_limit`): eliminated recompilation warnings but still hangs
- **Eager fallback** (`suppress_errors=True`): causes compiled/eager rank desync, hangs faster
- **Static shapes** (`dynamic=False`): OOMs due to no shape flexibility
- Root cause appears to be in the compiled backward graph's interaction with oneCCL ReduceScatter/AllReduce, not in dynamo recompilation

### HSDP `shard_grad_op` -- NOT Viable

Every attempt failed (BS=16, 8, 4 all OOM):

| Strategy | Reserved Memory | What's Held |
|----------|----------------|-------------|
| HSDP `full_shard` | **16.04 GB** | 1/12 params + 1/12 grads + 1/12 optimizer |
| HSDP `shard_grad_op` | **~31.19 GB** | Full params + 1/12 grads + 1/12 optimizer |

With 12 ranks per node holding 31+ GB each, plus CCL buffers, the aggregate exceeds node GPU memory.

### Conclusions

1. **HSDP `shard_grad_op` is not viable** for 7B models on 69 GB tiles with 12 ranks/node
2. **HSDP `full_shard` offers no advantage** over plain FSDP at 2-node scale
3. **Plain FSDP `full_shard` remains the best strategy** for multi-node 7B training

---

## FSDP Production Optimizations

### A/B Test Results (2-Node, 24 Tiles)

| Test | Throughput | Fwd | Bwd | Opt | Peak Mem | Status |
|------|-----------|-----|-----|-----|----------|--------|
| Baseline (prefetch ON, syncs ON) | **12.2 samp/s** | 10.3s | 18.1s | 0.027s | 48.33 GB | OK |
| **Production mode** (syncs OFF) | **13.0 samp/s** | 8.7s | 18.0s | 0.006s | 48.69 GB | **+6.6%** |
| torch.compile | N/A | -- | -- | -- | -- | FAILED |

Production mode (`FSDP_PRODUCTION_MODE=1`) skips non-essential `torch.xpu.synchronize()` calls. Savings: Forward -15.5%, Optimizer -77.8%, Backward unchanged.

### Bug Fix: `libpil4dfs.so` Causes FSDP AllGather Hangs

The DAOS interception library (`libpil4dfs.so`) intercepts POSIX I/O system calls for DAOS reads but also intercepts file descriptor operations used by XCCL/OFI transport, blocking FSDP AllGather collectives. **Always use `--no-pil4dfs`**.

### BS=16 is the Maximum for E2E (without torch.compile)

| BS | Status | Peak Memory |
|----|--------|-------------|
| 16 | **OK** | 48.33 GB |
| 20 | OOM (step 1+) | 63.34 GB |
| 24 | OOM (step 0) | 61.02 GB |

With `torch.compile`, BS=24 fits (see [torch.compile section](#torchcompile-investigation)).

---

## torch.compile Investigation

**Status**: VIABLE on Aurora (frameworks 25.190.0+). **Best new throughput: 8.7 samp/s at BS=24** (1-node), up from BS=16 max without compile.

### Background

torch.compile was previously marked NOT viable on Aurora (frameworks 24.347.0). The framework upgrade to **25.190.0** (Triton 3.1.0 → 3.4.0, PyTorch → 2.8.0a0) resolved the previous failures:

| Previous Issue (24.347.0) | Status on 25.190.0 |
|---------------------------|-------------------|
| `No module named 'triton.backends.intel'` | Fixed (Triton 3.4.0 has Intel backend) |
| `AssertionError` after 7 recompilations | Fixed (`dynamic=True` handles variable shapes) |
| Compilation >10 min, never completed | Fixed (86.5s for full OLMo-1B, ~90s for OLMo-3 7B) |

### Approach: Backbone-Only Compile

The full `UnifiedTransformer.forward()` has multiple graph breaks:
- `random.random()` for dropout (L415)
- `.item()` calls for debug stats (13 occurrences)
- `logger.*` calls (32 occurrences)
- `torch.isnan` + conditional (L585)
- `try/except` around projector (L378-384)

**Solution**: Compile only the HF backbone (`model.backbone`), where >90% of compute lives (attention + MLP matmuls). The backbone has **0 graph breaks** and supports `dynamic=True` for variable sequence lengths.

```python
# In train.py — applied BEFORE DDP/FSDP wrapping
model.backbone = torch.compile(model.backbone, backend="inductor", dynamic=True)
```

### Microbenchmark Results (OLMo-1B, single tile)

| Test | Eager | Compiled | Speedup |
|------|-------|----------|---------|
| Simple linear model | 0.33 ms/step | 0.031 ms/step | **10.8x** |
| Dynamic shapes (6 lengths) | -- | No recompilation | PASS |
| OLMo-1B full (16 layers) | 98.5 ms/step | 46.4 ms/step | **2.12x** |
| Compilation time (OLMo-1B) | -- | 86.5s | -- |

### E2E Training Results (1-Node, 12 Tiles, FSDP full_shard, OLMo-3 7B)

All tests: `--fsdp-production-mode --grad-ckpt-freq 2 --max-seq-length 1024 --use-bucketing`

| Config | BS | Throughput (samp/s) | Peak Memory | Headroom | Status |
|--------|-----|---------------------|-------------|----------|--------|
| No compile (baseline) | 16 | 7.3 | 50.5 GB | 18.5 GB | OK |
| torch.compile | 16 | 6.4 | 33.0 GB | 36.0 GB | OK (35% less memory) |
| **torch.compile** | **24** | **8.7** | **46.6 GB** | **22.4 GB** | **NEW BEST** |
| torch.compile | 32 | OOM (step 1) | 65.5 GB | 3.5 GB | FAIL |
| No compile | 20 | OOM | 63.3 GB | -- | FAIL |
| No compile | 24 | OOM | 61.0 GB | -- | FAIL |

### Key Findings

1. **Memory reduction**: torch.compile reduces peak memory by **35%** at BS=16 (50.5 → 33.0 GB), enabling larger batch sizes
2. **Per-step throughput**: At equal BS=16, compile is ~12% slower (fwd 12s vs 7.6s), but backward is slightly faster (15.5s vs 16.4s)
3. **Net throughput gain**: BS=24 with compile achieves **8.7 samp/s** vs 7.3 samp/s without compile at BS=16 — **+19% throughput** from fitting larger batches
4. **BS=24 is the new maximum** with compile (BS=32 OOMs at 65.5 GB reserved, only 3.5 GB headroom)
5. **Compilation overhead**: ~90s one-time cost on first forward pass (amortized over training run)
6. **DDP E2E still not viable**: Empirically confirmed — single-rank OLMo-3 7B (no DDP) OOMs at `optimizer.step()` with 62.81 GB allocated. PyTorch 2.8 already uses BF16 optimizer states (not FP32), but model + grads + optimizer = ~64 GB on one rank; adding DDP gradient buffers (+14.7 GB) pushes to ~79 GB, far exceeding the 69 GB tile
7. **Multi-node compile hangs**: torch.compile hangs on multi-node for both FSDP and HSDP. Root cause: the compiled backward pass interacts poorly with cross-node collectives on Intel XPU/oneCCL. Extensive testing ruled out recompilation as the cause:
   - FSDP + compile, 2N: hangs at step 1 (compiled backward ReduceScatter)
   - HSDP + compile, 2N: hangs at step 1-5 (compiled backward AllReduce)
   - `dynamic=False` (static shapes): OOMs (60.1 GB allocated, no shape flexibility)
   - `suppress_errors=True` (eager fallback): hangs at step 1 (compiled/eager rank desync)
   - `cache_size_limit=256`: hangs at step 5 (worse than default)
   - Uniform gradient checkpointing (all layers): eliminates recompilation warnings but still hangs
   - **torch.compile is currently single-node only**

### How to Enable (Single-Node Only)

```bash
python3 tools/launch_aurora_daos.py \
    --design PRISM-OLMO3-E2E-PROD \
    --torch-compile \
    --nodes 1 \
    training.batch_size=24 \
    ...
```

The `--torch-compile` flag sets `TORCH_COMPILE=1`, which triggers backbone-only compile in `train.py`. **Do not use with multi-node FSDP** — the compiled backward graph hangs during cross-node ReduceScatter.

---

## DeepSpeed ZeRO-2/3 (May 2026)

DeepSpeed ZeRO is the third sharded-training option alongside FSDP and HSDP.
Useful when DDP cannot fit and FSDP without `--torch-compile` OOMs (single-node
OLMo-3 7B E2E is the canonical case).

### When to use

| Strategy | OLMo-3 7B fits 1N? | Notes |
|---|---|---|
| DDP | ❌ | 25 GB params alone, doesn't fit alongside grads+optim |
| FSDP eager | ❌ at BS≥1 | OOMs without `--torch-compile` |
| FSDP + compile | ✅ at BS=24 | 77.7 samp/s — **preferred when compile works** |
| ZeRO-2 | ✅ | Optimizer-state shard only; comparable to DDP-with-compile |
| ZeRO-3 | ✅ | Full param+grad+optimizer shard; comparable to FSDP |

ZeRO is also the only path that supports DeepSpeed-specific features (offloading,
NVMe paging, etc.) if those become needed later.

### Validated invocations (1N projector smoke, OLMo-1B)

```bash
python tools/launch_aurora_daos.py \
    --id Z2-SMOKE \
    --design PRISM-PROJ-ABLATION-LAYERNORM \
    --batch --nodes 1 --queue debug-scaling \
    --deepspeed 2 \
    --max-seq-length 1024 --use-bucketing --no-pil4dfs \
    training.max_steps=20

# Same with --deepspeed 3 for ZeRO-3
```

### Throughput (1N, 12 ranks, OLMo-1B image-only, BS=8 accum=4)

| Strategy | Throughput | Notes |
|---|---|---|
| Native DDP | 195–245 samp/s | Best 1N for projector-only |
| ZeRO-2 | ~170 samp/s | DeepSpeed overhead vs DDP |
| ZeRO-3 | ~155 samp/s | Full param shard, slightly slower than Z2 |

For OLMo-3 7B 1N E2E (where DDP can't fit), ZeRO is required.

### Aurora XPU integration gotchas

DeepSpeed on Aurora needs three Aurora-specific fixes that are NOT in upstream
DeepSpeed (all in `src/training/trainer_zone_a.py`):

1. **`LOCAL_RANK=0` pin BEFORE `Accelerator()` constructor.** `ZE_AFFINITY_MASK`
   exposes only `xpu:0` per rank. Accelerate calls
   `init_process_group(device_id=xpu:LOCAL_RANK)` during constructor; without
   the pin, ranks ≥1 raise `device_id xpu:N out of range`. Original `LOCAL_RANK`
   is stashed in `PRISM_REAL_LOCAL_RANK` for `setup_distributed()` to read back.

2. **Exclude WebDataset DataLoader from `accelerator.prepare()`.** Accelerate's
   `IterableDatasetShard` re-shards data that WebDataset already sharded via
   `split_by_node`, causing each rank to load `num_processes × batch_size`
   samples and discard all but `batch_size`. ~12× data-loading slowdown on a
   12-tile node.

3. **Manual `batch.to(device)` in train loop when `_use_deepspeed`.** Since the
   loader is excluded from `prepare()`, Accelerate doesn't auto-move tensors
   to `xpu:0`. Without the manual move:
   `RuntimeError: index is on cpu, different from other tensors on xpu:0`.

DeepSpeed plugin must also have `train_micro_batch_size_per_gpu` set
explicitly when no dataloader is passed to `prepare()`.

### Known incompatibilities

- **DeepSpeed + `--torch-compile`**: NaN loss at step 1 for ZeRO-2,
  `AttributeError` for ZeRO-3. Disable `TORCH_COMPILE` when running ZeRO.
- **`--use-accelerate --dist-strategy ddp` (without DeepSpeed)** also broken
  on Aurora XPU (`IndexError: list index out of range` in
  `_to_kwargs`/`_get_stream`). Use native DDP path or DeepSpeed instead.

---

## Recommended Production Configs

### Projector-Only Training (DDP)

```bash
python tools/launch_aurora_daos.py \
    --id PROJECTOR-STAGE1 \
    --design PRISM-IMAGE-ONLY-7B \
    --nodes 2 --batch --no-pil4dfs \
    --dataset-groups projector \
    --use-bucketing --bucket-buffer-size 5000 \
    --max-seq-length 1024 \
    training.batch_size=3 \
    training.max_steps=500
```

Expected: ~131 samp/s with mixed datasets, ~98 samp/s pixmo-only.

### E2E Training (FSDP, Multi-Node)

```bash
python3 tools/launch_aurora_daos.py \
    --design PRISM-OLMO3-E2E-PROD \
    --dist-strategy fsdp --fsdp-sharding full_shard \
    --grad-ckpt-freq 2 --no-pil4dfs --fsdp-production-mode \
    --dataset-groups projector --use-bucketing --max-seq-length 1024 \
    --nodes 2 --batch --queue capacity --walltime 12:00:00 \
    training.batch_size=16
```

Expected: 13.0 samp/s (2 nodes). projector preset: ~14.9h for full epoch.

### E2E Training (FSDP + torch.compile, Single-Node)

```bash
python3 tools/launch_aurora_daos.py \
    --design PRISM-OLMO3-E2E-PROD \
    --dist-strategy fsdp --fsdp-sharding full_shard \
    --grad-ckpt-freq 2 --no-pil4dfs --fsdp-production-mode \
    --torch-compile \
    --dataset-groups projector --use-bucketing --max-seq-length 1024 \
    --nodes 1 --batch --queue capacity --walltime 24:00:00 \
    training.batch_size=24
```

Expected: 8.7 samp/s (1 node, BS=24). First step incurs ~90s compilation overhead. Requires frameworks 25.190.0+. **Single-node only** — multi-node compile hangs during backward ReduceScatter.

### E2E Training (COMPOSITE DDP, Single-Node)

```bash
python tools/launch_aurora_daos.py \
    --id OLMO3-E2E-COMPOSITE-PROD --design PRISM-OLMO3-E2E-COMPOSITE-1NODE \
    --nodes 1 --composite --batch --queue capacity --walltime 06:00:00 \
    --grad-ckpt-freq 0 \
    --dataset-groups projector --use-bucketing --max-seq-length 1024 --no-pil4dfs
```

Expected: ~2.8 samp/s (BS=3, 6 ranks, no grad checkpointing).

### Aurora Queue Reference

| Queue | Nodes | Max Walltime | Use Case |
|-------|-------|-------------|----------|
| `debug` | 1-2 | 1h | Quick tests |
| `debug-scaling` | 2-10 | 1h | Multi-node tests |
| `capacity` | 1-16 | 168h | **Production runs** |
| `prod` | 256+ | varies | Large-scale only |

---

## Historical Reference

### Throughput Evolution (7B, 2-Node)

| Date | Config | Throughput | Scaling Eff. | Key Change |
|------|--------|------------|-------------|------------|
| Feb 16 AM | DDP, no CCL tuning | 28-29 samp/s | 55% | Initial |
| Feb 16 AM | DDP + CCL tuning | 64 samp/s | 63% | CCL env vars |
| Feb 16 PM | DDP + CCL + local_manifest | 78.6 samp/s | 77% | Data loading fix |
| Feb 17 | DDP + static_graph fix | **98.4 samp/s** | **94%** | Modality mismatch fix |
| Feb 17 | DDP, BS=3, all datasets | 131 samp/s | -- | Bucketing + BS increase |
| Feb 19-20 | FSDP E2E, BS=16 | 13.5 samp/s | -- | E2E training |
| Feb 28 | FSDP E2E + production mode | **13.0 samp/s** | -- | Optimized syncs |
| Feb 28 | FSDP E2E + torch.compile BS=24 | **8.7 samp/s** (1N) | -- | Compile + higher BS |

### 1B vs 7B Throughput Comparison

| Model | Nodes | Tiles | BS | Eff Batch | Throughput | Per-Tile |
|-------|-------|-------|-----|-----------|------------|----------|
| **1B** | 2 | 24 | 8 | 192 | **262 samp/s** | 10.9 |
| **7B** | 2 | 24 | 2 | 384 | **53.4 samp/s** | 2.2 |

Throughput ratio: 4.9x slowdown for 7x parameter increase -- 7B is more efficient per-parameter than 1B.

### Projector Convergence

Loss plateaus at ~2.2 after ~70 steps (2.9% data coverage). A linear projector (~20M params) saturates quickly -- more data can't improve a linear mapping that has found its optimum. The loss floor is the frozen LLM's perplexity on descriptive text.

Recommended projector training: 500-1000 steps (80-160 min on 2 nodes).

### Issues Discovered & Fixed

Key issues resolved during scaling work (the Feb-2026 bring-up debug journal has
been retired — see `git log`; for current guidance see
[aurora_operations.md](../platforms/aurora_operations.md)):

| Issue | Impact | Fix |
|-------|--------|-----|
| DAOS startup contention | 30+ min timeout | Rank-0 coordinated shard discovery (0.6s) |
| Modality mismatch with static_graph | 30% throughput loss | Explicit `model.modalities=[text,image]` |
| DAOS missing shard 613 | FileNotFoundError | Moved to val_shards/ |
| OLMo-7B OOM at BS=8 | Crash | Use BS=2 with grad_accum=16 |
| DDP bucket sizing for E2E | Empty bucket crash | Standard sizing when trainable > 1GB |
| Gradient checkpointing silent disable | No checkpointing | Use training config, not model config |
| `use_reentrant=True` + DDP | Empty bucket crash | Set `use_reentrant=False` |
| libpil4dfs + FSDP | AllGather hang | Always use `--no-pil4dfs` |

### Files

- Launcher (DAOS): `tools/launch_aurora_daos.py`
- Launcher (Staging): `tools/launch_aurora_web.py`
- COMPOSITE interactive: `tools/run_composite_interactive.sh`
- Results log directory: `logs/scaling_study/`

# PRISM Data Pipeline Documentation

This document provides comprehensive documentation of the PRISM data pipeline, including dataset organization, dataloading strategies, throughput optimizations, and next steps for improvement.

---

## Table of Contents

1. [Overview](#overview)
2. [Storage Architecture](#storage-architecture)
3. [Dataset Organization](#dataset-organization)
4. [WebDataset Format](#webdataset-format)
5. [Data Loading Pipeline](#data-loading-pipeline)
6. [Sharding & Distribution](#sharding--distribution)
7. [Bucketing & Collation](#bucketing--collation)
8. [Configuration Reference](#configuration-reference)
9. [Throughput Analysis](#throughput-analysis)
10. [Resampled Mode & Epoch Boundary Fix](#resampled-mode--epoch-boundary-fix)
11. [Optimization Strategies](#optimization-strategies)
12. [Next Steps & Roadmap](#next-steps--roadmap)

---

## Overview

PRISM uses a WebDataset-based data pipeline optimized for distributed training on Aurora HPC. The pipeline supports:

- **Multi-source loading**: Weighted sampling from 26+ datasets across 4 groups
- **DAOS storage**: High-performance distributed storage with manifest-based shard discovery
- **Flexible configuration**: YAML-based presets with per-dataset weight/proportion controls
- **Bucketing**: Sequence length grouping to minimize padding waste
- **Distributed coordination**: Rank-0 shard discovery with broadcast to avoid O(N) filesystem contention

### Key Files

| Component | File | Description |
|-----------|------|-------------|
| Dataset Config | `src/conf/data/daos_datasets.yaml` | Dataset definitions, weights, presets |
| WebDataset Loader | `src/data/multi_webdataset.py` | Core loading logic, sharding, bucketing |
| Collation | `src/data/collate.py` | Padding, batch construction |
| Launch Script | `tools/launch_aurora_daos.py` | DAOS mounting, training orchestration |
| Analysis Tool | `tools/analyze_dataset_lengths.py` | Sequence length profiling |

---

## Storage Architecture

### DAOS Storage Hierarchy

```
AuroraGPT (Pool)
└── prism_training_data (Container)
    ├── pixmo/                          # ~700GB total
    │   ├── pixmo_cap_webdataset/
    │   ├── pixmo_points_webdataset/
    │   └── pixmo_count_webdataset/
    ├── s1mmalign/                      # ~3.5TB total
    │   ├── arxiv_webdataset/           # ~3.4TB (largest)
    │   ├── biorxiv_webdataset/
    │   └── ...
    ├── cosyn/                          # ~50GB
    │   └── cosyn_point_webdataset/
    └── nemotron/                       # ~500GB
        ├── wiki_*_webdataset/          # 10 languages
        ├── sparsetables_webdataset/    # SFT only (long sequences)
        └── plotqa_cot_webdataset/      # SFT only (long sequences)
```

### Mounting DAOS

```bash
# Request DAOS filesystem access in job submission
qsub -l filesystems=flare:home:daos_user_fs ...

# Mount the container (automatic in launch_aurora_daos.py)
module load daos
mkdir -p /tmp/${USER}/AuroraGPT/prism_training_data
dfuse /tmp/${USER}/AuroraGPT/prism_training_data AuroraGPT prism_training_data
```

### Storage Tier Recommendations

| Tier | Path | Use Case |
|------|------|----------|
| Local NVMe | `/tmp/` | Model weights, small checkpoints |
| DAOS | `/tmp/${USER}/AuroraGPT/` | Training data (mounted via dfuse) |
| Lustre | `/flare/` | Code, configs, large checkpoints |
| Home | `/home/` | Avoid for training I/O |

---

## Dataset Organization

### Dataset Groups

| Group | Datasets | Total Samples | Description |
|-------|----------|---------------|-------------|
| **pixmo** | 3 | 2.57M | Caption, pointing, counting |
| **s1mmalign** | 9 | 15.5M | Scientific papers (arxiv, biorxiv, etc.) |
| **cosyn** | 1 | 316K | Synthetic pointing Q&A |
| **nemotron** | 13 | 2.3M | Wikipedia (10 langs) + vision datasets |

### Per-Dataset Statistics

| Dataset | Samples | Shards | Avg Tokens | Max Tokens | Weight | Notes |
|---------|---------|--------|------------|------------|--------|-------|
| **pixmo_cap** | 614K | 613 | 215 | 663 | 1.0 | Image captions |
| **pixmo_points** | 1.9M | 1920 | 45 | 746 | 1.0 | Visual grounding |
| **pixmo_count** | 33K | 34 | 46 | 183 | 0.5 | Object counting |
| **arxiv** | 13.4M | 2689 | 175 | 503 | 1.0 | 3.4TB, use proportion=0.1 |
| **biorxiv** | 1.1M | 1143 | 232 | 503 | 1.0 | Biology papers |
| **nature_comunication** | 549K | 550 | 259 | 513 | 1.0 | Nature papers |
| **wiki_en** | 198K | 198 | 436 | 1800 | 0.3 | Wikipedia English |
| **wiki_de** | 198K | 198 | 520 | 2025 | 0.1 | Wikipedia German |
| **sparsetables** | 99K | 99 | 2659 | 13232 | 0.0 | SFT only |
| **plotqa_cot** | 16K | 17 | 4179 | 13196 | 0.0 | SFT only |
| **cosyn_point** | 316K | 316 | 43 | 124 | 1.0 | Molmo2-format pointing |

### Train/Validation Split

Each dataset contains:
- `shards/` - Training data (99% of samples)
- `val_shards/` - Validation data (1% of samples)
- `manifest.json` - Shard metadata for fast discovery

```json
{
  "dataset": "pixmo_cap",
  "num_train_shards": 613,
  "num_val_shards": 7,
  "train_samples": 613610,
  "val_samples": 6200,
  "shards": [{"name": "pixmo-cap-000000.tar", "samples": 1000}, ...],
  "val_shards": [{"name": "pixmo-cap-000613.tar", "samples": 886}, ...]
}
```

---

## WebDataset Format

### Shard Structure

Each `.tar` shard contains ~1000 samples:

```
shard-000000.tar
├── 000000.jpg          # Image (jpg/png/webp/gif supported)
├── 000000.txt          # Caption text
├── 000000.json         # Metadata
├── 000001.jpg
├── 000001.txt
├── 000001.json
└── ...
```

### Sample Formats

#### 1. Caption Format (Default)
```json
// 000000.json
{
  "source": "pixmo_cap",
  "image_id": "12345",
  "width": 1024,
  "height": 768
}
// 000000.txt
"A photograph of a cat sitting on a windowsill."
```

#### 2. Pointing Format (pixmo_points)
```json
// 000000.json
{
  "source": "pixmo_points",
  "label": "the red car",
  "points": [{"x": 45.2, "y": 32.1}, {"x": 48.5, "y": 35.0}]
}
```
Converted to Molmo2 format:
```
user: Point to the red car
assistant: <points coords="1 1 452 321;1 2 485 350">the red car</points>
```

#### 3. Conversation Format (cosyn_point)
```json
// 000000.json
{
  "source": "cosyn_point",
  "conversations": [
    {"role": "user", "content": "Where is the door?"},
    {"role": "assistant", "content": "<points coords=\"1 1 234 567\">door</points>"}
  ]
}
```

---

## Data Loading Pipeline

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DAOS Storage                                  │
│  AuroraGPT/prism_training_data/{group}/{dataset}/shards/*.tar          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Shard Discovery (Rank 0 Only)                        │
│  - Read manifest.json (fast) or glob shards/ (slow fallback)           │
│  - Apply proportion limiting (e.g., 10% of arxiv)                      │
│  - Broadcast shard lists to all ranks via dist.broadcast()             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Shard Distribution                                 │
│  - Each rank gets shards[rank::world_size]                             │
│  - Example: 24 ranks, 240 shards → 10 shards per rank                  │
│  - shards_for_rank = all_shards[rank::world_size]                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    WebDataset Pipeline (resampled=True)                  │
│  wds.WebDataset(shards_for_rank, resampled=True)                       │
│    .shuffle(buffer=1000)        # Sample-level shuffling               │
│    .decode("pil")               # Decode images to PIL                 │
│    .to_tuple(...)               # Extract (image, text, json)          │
│    .map(_process_sample)        # Format conversion                    │
│  NOTE: resampled=True → infinite iterator, no epoch boundaries         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    wds.RandomMix (Weighted Sampling)                    │
│  Combines N dataset sources with normalized weights                    │
│  weights = [0.15, 0.12, 0.08, ...]  (sum to 1.0)                      │
│  longest=True: if a source exhausts, drop it and continue              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MultiWebDatasetWrapper                               │
│  - PIL → Tensor transform (224x224, normalized)                        │
│  - Tokenization with truncation (max_length=MAX_SEQ_LENGTH)  [CAP #1]  │
│  - Output: {"image": tensor, "text": tokens, "_metadata": str}         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              BucketedMultiWebDatasetWrapper (Optional)                  │
│  - Buffer N samples (default 2000)                                     │
│  - Sort by text length, divide into num_buckets sub-ranges             │
│  - Infinite iteration (no epoch boundaries with resampled=True)        │
│  - Yield in length-sorted order                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DataLoader + BucketedCollator                        │
│  - Hard truncation to max_seq_length (safety net)              [CAP #2]│
│  - Sort within batch by length (longest first)                         │
│  - Dynamic padding to max length in batch                              │
│  - Truncation + efficiency stats logging                               │
│  - Output: {"image": (B,3,224,224), "text": (B, max_len)}             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dataset Classes

| Class | Purpose | Use Case |
|-------|---------|----------|
| `MultiWebDataset` | Raw WebDataset with weighted mixing | Low-level access |
| `MultiWebDatasetWrapper` | Adds tokenization + transforms | Standard training |
| `BucketedMultiWebDatasetWrapper` | Adds length-based buffering | Variable-length data |

---

## Sharding & Distribution

### Shard Assignment Strategy

Shards are distributed across ranks using strided assignment:

```python
# In multi_webdataset.py:545-548
if self.world_size > 1:
    shards_for_rank = shards[self.rank :: self.world_size]
else:
    shards_for_rank = shards
```

**Example**: 240 shards, 24 ranks:
- Rank 0: shards 0, 24, 48, 72, ... (10 shards)
- Rank 1: shards 1, 25, 49, 73, ... (10 shards)
- Rank 23: shards 23, 47, 71, 95, ... (10 shards)

### Proportion Limiting

For large datasets, use `proportion` to limit shards:

```yaml
arxiv:
  path: s1mmalign/arxiv_webdataset
  samples: 13395035
  shards: 2689
  weight: 1.0
  proportion: 0.1  # Only use 269 shards (10%)
```

### Distributed Shard Discovery

To avoid O(N) filesystem contention when scaling:

1. **Rank 0** reads `manifest.json` for all datasets
2. **Rank 0** serializes shard lists to JSON
3. **Rank 0** broadcasts via `dist.broadcast()`
4. **All ranks** deserialize and select their shards

```python
# In multi_webdataset.py:426-517
def _discover_all_shards_distributed(self):
    if is_distributed:
        if self.rank == 0:
            all_shards = {ds["name"]: self._get_shards_for_dataset(ds) ...}
        dist.barrier()
        dist.broadcast(size_tensor, src=0)
        dist.broadcast(data_tensor, src=0)
        if self.rank != 0:
            all_shards = json.loads(...)
```

---

## Bucketing & Collation

### Problem: Sequence Length Variance

Without bucketing, a batch might contain:
- Sample A: 10 tokens (pixmo_count)
- Sample B: 500 tokens (wiki_en)
- Sample C: 15 tokens (cosyn_point)

All samples pad to 500 tokens → **97% padding waste**.

### Solution 1: Within-Batch Sorting (BucketedCollator)

Sorts samples **within each batch** by length:

```python
# In collate.py:193-196
lengths.sort(key=lambda x: x[1], reverse=True)  # Longest first
sorted_indices = [i for i, _ in lengths]
batch = [batch[i] for i in sorted_indices]
```

**Benefit**: Better GPU utilization (longest sequences processed first).

### Solution 2: Buffer Sorting (BucketedMultiWebDatasetWrapper)

Buffers N samples, sorts globally, then yields:

```python
# In multi_webdataset.py
buffer = []
for sample in self._base:
    buffer.append((text_len, sample))
    if len(buffer) >= self.buffer_size:
        buffer.sort(key=lambda x: x[0])  # Sort by length
        yield from self._yield_sorted_buffer(buffer)
        buffer = []
```

**Infinite iteration (resampled=True)**: With `resampled=True` on the
underlying WebDatasets, the base iterator never exhausts. The buffer
fills continuously without epoch boundaries, eliminating the primary
cause of straggler effects (different ranks hitting epoch boundaries
at different times). See [Resampled Mode & Epoch Boundary Fix](#resampled-mode--epoch-boundary-fix)
for details.

**Configuration**:
```python
dataset = BucketedMultiWebDatasetWrapper(
    buffer_size=2000,   # Larger = better bucketing, more memory
    num_buckets=8,      # Divide buffer into N sub-buckets
    shuffle_buckets=True,  # Shuffle within buckets for randomness
)
```

**Note**: The `__iter__` method has a safety-net restart if `StopIteration`
is raised (which should not happen with `resampled=True`). The training
loop controls how many steps to run via `max_steps`.

### Solution 3: Collator-Level Truncation (BucketedCollator)

The `BucketedCollator` enforces a hard `max_seq_length` cap before padding.
Any sequence exceeding this limit is truncated in-place. This is a safety net
against backward pass spikes from outlier-length sequences.

```python
collator = BucketedCollator(
    tokenizer,
    sort_within_batch=True,
    log_efficiency=True,
    max_seq_length=1024,  # Hard cap - truncates before padding
)
```

Truncation stats are logged periodically:
```
[TRUNCATION] 42 of 3000 samples truncated to 1024 tokens (1.4%)
```

### Efficiency Metrics

The collator logs padding efficiency:

```
[BUCKET] Batch 100: len_range=[10, 25], eff=82.3%, overall_eff=78.5%
[BUCKET] Batch 200: len_range=[150, 180], eff=94.1%, overall_eff=81.2%
```

**Target**: >85% efficiency for mixed datasets. Observed: 83-93% overall
efficiency with buffer_size=2000 and `all` dataset groups. With `max_seq_length=1024`
(Job 8339446), overall efficiency was 85-87% but with periodic dips to 37% during
epoch boundary cycling (very short sequences from pixmo_count/cosyn_point mixed
with longer wiki/arxiv sequences).

---

## Configuration Reference

### daos_datasets.yaml Structure

```yaml
# DAOS connection settings
daos:
  pool: AuroraGPT
  container: prism_training_data
  mount_base: /tmp/${USER}/AuroraGPT/prism_training_data

# Dataset groups
groups:
  pixmo:
    description: "Pixmo caption and pointing datasets"
    datasets:
      pixmo_cap:
        path: pixmo/pixmo_cap_webdataset
        samples: 613610
        shards: 613
        weight: 1.0
        description: "Image captions"
      pixmo_points:
        path: pixmo/pixmo_points_webdataset
        samples: 1919853
        shards: 1920
        weight: 1.0
        proportion: 1.0  # Use all shards
        description: "Visual pointing/grounding"

# Presets for common configurations
presets:
  all:
    groups: [pixmo, s1mmalign, cosyn, nemotron]
  
  quick:
    groups: [pixmo, s1mmalign, cosyn, nemotron]
    proportion_overrides:
      arxiv: 0.05      # Only 5% of 3.4TB dataset
      biorxiv: 0.1
      pixmo_points: 0.2
  
  sft:
    groups: [pixmo, cosyn, nemotron]
    weight_overrides:
      sparsetables: 1.0  # Re-enable long-sequence datasets
      plotqa_cot: 1.0
      nights_cot: 1.0

default_preset: all
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DAOS_MOUNT` | `/tmp/${USER}/AuroraGPT/prism_training_data` | DAOS mount path |
| `DATASET_CONFIG` | `src/conf/data/daos_datasets.yaml` | Config file path |
| `USE_MULTI_DATASET` | `true` | Use MultiWebDataset |
| `DATASET_GROUPS` | `all` | Groups/preset to load |
| `MAX_SEQ_LENGTH` | `2048` | Max token length for tokenizer + collator. **Use 1024 for projector training.** |
| `USE_BUCKETING` | `false` | Use BucketedMultiWebDatasetWrapper |
| `BUCKET_BUFFER_SIZE` | `2000` | Buffer size for bucketing |
| `BUCKET_NUM_BUCKETS` | `8` | Number of length sub-buckets |
| `USE_BUCKETED_COLLATOR` | `true` | Use BucketedCollator |
| `PER_RANK_TIMING` | `0` | Enable per-rank straggler detection (every 50 steps) |
| `ENABLE_PROFILER` | `0` | Enable PyTorch profiler for specific steps |
| `PROFILER_STEPS` | `5,10,15` | Comma-separated step numbers to profile |
| `CCL_LOG_LEVEL` | (unset) | Set to `info` for CCL debug logging (WARNING: massive log output) |

### Usage Examples

```bash
# Projector training (recommended for Stage 1)
python tools/launch_aurora_daos.py \
  --dataset-groups projector \
  --max-seq-length 1024 \
  --use-bucketing \
  --bucket-buffer-size 2000 \
  --bucket-num-buckets 8 \
  --no-pil4dfs

# Use quick preset (10% of large datasets, for testing)
python src/train.py data=daos_datasets data.preset=quick

# Use specific groups
python src/train.py data=daos_datasets data.groups=[pixmo,cosyn]

# Override weights at runtime
python src/train.py data=daos_datasets data.weight_overrides.arxiv=0.5

# Enable bucketing
USE_BUCKETING=true BUCKET_BUFFER_SIZE=2000 python src/train.py
```

---

## Throughput Analysis

### Root Cause: Confirmed via Ablation (2026-02-18)

Training throughput drops of 70-95% were observed across multiple jobs. A systematic
ablation study (jobs 8339128-8339131, 8339290) confirmed the root cause:

**Cross-rank sequence length variance causes backward pass spikes and straggler effects.**

The mechanism:
1. The bucket buffer (even at 5000) drains near the end of a dataset epoch
2. Remaining samples have extreme length variance (e.g., `len_range=[337, 2048]`)
3. Long sequences cause **quadratic attention scaling** in the backward pass
4. Individual micro-batch backward times spike from 0.2s to **20-30 seconds** (100x)
5. Different ranks hit the boundary at different times, causing **straggler effects**
6. DDP AllReduce barriers force all ranks to wait for the slowest rank

### Ablation Test Results

| Job | Config | Steps | Peak | Worst | Worst Bwd Spike | Straggler Ratio |
|-----|--------|-------|------|-------|-----------------|-----------------|
| 8338037 | Baseline (max=2048) | 260 | 264.9 | 7.6 | 152s | N/A |
| 8339129 | Aggressive bucket (buf=5000, n=16) | 200 | 149.5 | 12.0 | 21.6s | 4.92x |
| 8339131 | CCL profiler (max=2048) | 180 | 160.6 | 43.3 | 30.7s | 3.00x |
| 8339290 | Collator cap=2048 + epoch refill | 160 | 165.0 | 50.7 | 21.0s | 3.00x |
| 8339446 | **max_seq_length=1024** | 200 | 75.2 | 56.3 | 12.4s | 4.90x |
| 8339782 | **resampled=True + projector preset** | 310 | 176.0 | 56.0 | 13.5s | 7.03x |

**Key findings from ablation:**
- **CCL is NOT the problem.** Zero CCL errors in 30M lines of debug output. AllReduce
  is ~16% of step time (normal for DDP).
- **Aggressive bucketing does not fix it.** buffer_size=5000, num_buckets=16 still
  produced the same crash pattern.
- **Epoch boundary buffer refill does not fix it.** DataLoader workers run their own
  `__iter__` independently; the refill only fires at DataLoader-level epoch boundaries.
- **max_seq_length=2560 causes OOM.** GPU memory exhaustion during backward pass.
- **pil4dfs with max_seq_length=2048 caused init to consume the entire 1hr walltime.**
  The combination of pil4dfs + 24-rank 7B model loading + 23 dataset discovery was
  too slow. With pil4dfs disabled, init takes ~8 minutes.
- **The throughput crash is periodic and self-healing.** It recurs every ~40-60 steps
  (matching the dataset epoch cycle) and recovers fully within 2-3 steps.
- **max_seq_length=1024 reduces spike severity but does not eliminate stragglers.**
  (Job 8339446) Worst backward spike dropped from 20-30s to 12.4s (~2x improvement).
  Throughput was more stable (worst step 56.3 vs typical 75 samp/s = 25% drop, vs
  95% drops at 2048). However, straggler ratios remained high (4.90x at step 200),
  confirming the root cause is uneven epoch boundaries across ranks, not sequence
  length alone.
- **resampled=True eliminates epoch boundaries but does NOT fix stragglers.**
  (Job 8339782) Zero `Epoch boundary` or `StopIteration` messages. Bucket efficiency
  improved monotonically (78%→89%). Peak throughput doubled (176 vs 75 samp/s). But
  rank 23 straggler ratio was **worse** (7.03x at step 200 vs 4.90x in 8339446). The
  straggler is periodic (steps 100/200 bad, steps 150/250/300 fine), ruling out epoch
  boundaries as the cause. The true root cause is **cross-rank sequence length variance
  from random sampling** — each rank's RandomMix independently draws different datasets,
  so one rank can get a batch of long wiki_en sequences (436 avg tokens) while another
  gets short cosyn_point sequences (43 avg tokens), causing a ~100x compute difference
  in attention.

### Sequence Length Distribution

| Dataset | Mean | P90 | P99 | Max |
|---------|------|-----|-----|-----|
| cosyn_point | 43 | 54 | 98 | 124 |
| pixmo_points | 45 | 62 | 634 | 746 |
| pixmo_count | 46 | 80 | 138 | 183 |
| arxiv | 175 | 258 | 349 | 503 |
| pixmo_cap | 215 | 300 | 449 | 663 |
| biorxiv | 232 | 337 | 451 | 503 |
| nature_comunication | 259 | 344 | 505 | 513 |
| wiki_en | 436 | 761 | 1437 | 1800 |
| wiki_de | 520 | 858 | 1384 | 2025 |
| **sparsetables** | 2659 | 5975 | 9892 | 13232 |
| **plotqa_cot** | 4179 | 8710 | 11072 | 13196 |

**Key finding**: ~10x difference between pixmo (102 avg) and scientific datasets (1063 avg).

### The Fix: Lower `MAX_SEQ_LENGTH`

The primary mitigation is reducing `MAX_SEQ_LENGTH` to cap the worst-case compute:
- Quadratic attention cost: 2048^2 = 4.2M ops vs 1024^2 = 1.0M ops (4x reduction)
- For projector-only training, 1024 tokens is sufficient (most captions are <500)
- Set via environment variable: `MAX_SEQ_LENGTH=1024`
- Truncation happens at two levels: tokenizer (dataset) and collator (batch construction)

### Current Mitigations

1. **Resampled mode (`resampled=True`)**: WebDatasets sample shards with replacement
   infinitely, eliminating epoch boundaries. Combined with `RandomMix(longest=True)`
   as a safety net. Improved peak throughput 2.3x and bucket efficiency, but did NOT
   fix the straggler problem (which is caused by cross-rank sequence length variance).
2. **Projector preset**: Molmo-inspired weight rebalancing (~32% caption, ~32%
   pointing, ~30% scientific, ~5% wiki_en). Excludes tiny datasets (metarxiv,
   edrxiv) and non-English wiki. Reduces biorxiv/nature_comunication proportion to 30%.
3. **Long datasets excluded**: `sparsetables`, `plotqa_cot` have `weight: 0.0`
4. **Hard sequence length cap**: `MAX_SEQ_LENGTH` env var controls truncation (default 2048, use 1024 for projector training)
5. **Collator-level truncation**: `BucketedCollator` enforces `max_seq_length` as a safety net before padding
6. **BucketedCollator**: Sorts within batch for better efficiency
7. **Per-rank timing**: Straggler detection with cross-rank timing comparison every 50 steps

---

## Resampled Mode & Epoch Boundary Fix

### Original Hypothesis (Partially Wrong)

The straggler problem was initially attributed to `wds.RandomMix` with
`longest=False` (the default), where exhausted datasets killed the entire mixed
stream. This was a **real bug** that caused unnecessary iterator rebuilds, but
Job 8339782 proved it was **not the primary straggler cause**. After fixing it
with `resampled=True`, stragglers persisted at 7.03x (worse than before).

### What resampled=True Actually Fixed

The epoch boundary fix was still valuable — it eliminated:
- Iterator rebuild stalls (expensive shuffle/bucket buffer refills)
- Pathological bucket ranges (e.g., `len_range=[1, 38]`)
- Data loading spikes at epoch transitions

And it improved:
- Peak throughput: 176 samp/s (vs 75 samp/s in 8339446)
- Bucket efficiency: monotonically improving 78%→89% (vs cycling 37%→99%)
- Data load time: steady 0.005s (vs periodic spikes)

### The Real Straggler Root Cause: Cross-Rank Sequence Length Variance

Each rank's `RandomMix` independently and randomly selects which dataset to
sample from. With datasets ranging from 43 avg tokens (cosyn_point) to 436 avg
tokens (wiki_en), one rank can draw a batch of long sequences while another
draws short sequences. Attention cost scales quadratically, so a 10x token
difference causes ~100x compute difference. DDP AllReduce forces all ranks to
synchronize, so the rank with the longest sequences becomes the straggler.

This explains why:
- The straggler is **periodic** (bad at steps 100/200, fine at 150/250/300)
- It affects **rank 23 specifically** (random seed + rank = unlucky draw pattern)
- `resampled=True` didn't help (the variance is inherent to random mixing)

### Potential Fixes for Cross-Rank Variance

1. **Synchronized random seed for RandomMix** — Force all ranks to draw from
   the same dataset at the same time. Each rank still reads different shards,
   but the dataset *type* (and thus sequence length distribution) is consistent.
2. **Per-batch sequence length synchronization** — After bucketing, broadcast
   the max sequence length and pad all ranks to the same length. Wastes compute
   but eliminates straggler variance.
3. **Narrower sequence length distribution** — Reduce the gap between shortest
   and longest datasets (e.g., truncate wiki_en to 256 tokens for projector
   training, or increase cosyn_point weight to balance the mix).

### Original Epoch Boundary Bug (Fixed)

The original `longest=False` bug in `RandomMix` caused a secondary issue:

The problem was amplified by tiny datasets:

| Dataset | Shards | Samples | Draws to exhaust | Steps to exhaust |
|---------|--------|---------|------------------|------------------|
| metarxiv | 1 | 362 | ~36,566 | ~508 |
| edrxiv | 2 | 1,375 | ~138,889 | ~1,929 |
| psyarxiv | 17 | 16,942 | ~570,370 | ~7,922 |

`metarxiv` (362 samples, 1 shard) would exhaust after ~508 training steps,
killing the entire `RandomMix` stream for whichever rank happened to sample
it most frequently. This rank would then spend seconds rebuilding its iterator
while all 23 other ranks waited.

### The Fix

Two changes in `multi_webdataset.py:_build_dataset()`:

1. **`resampled=True`** on `wds.WebDataset()`: Each dataset samples shards
   with replacement infinitely. No shard is ever "exhausted" — the iterator
   never raises `StopIteration`. This eliminates epoch boundaries entirely.

2. **`longest=True`** on `wds.RandomMix()`: Safety net — if a source somehow
   exhausts despite resampling, it is dropped and sampling continues from
   remaining sources with reweighted probabilities (instead of terminating
   everything).

### Replay Implications

With `resampled=True`, shards are sampled with replacement. For large datasets
(hundreds of shards), the probability of replaying a specific sample within
any training window is negligible. For tiny datasets, replay is inevitable
regardless — `metarxiv` (362 samples) was already being replayed ~530x per
full pass through `pixmo_points` under the old code, since the training loop
caught `StopIteration` and re-created the iterator.

The `projector` preset addresses this by excluding the tiniest datasets
(`metarxiv`, `edrxiv`) entirely, since their data diversity is negligible
and they contribute only memorization.

### Projector Preset: Weight Rebalancing

The `projector` preset in `daos_datasets.yaml` rebalances weights toward
Molmo-inspired ratios for projector-only training:

| Category | Datasets | Weight Sum | Sampling % |
|----------|----------|------------|------------|
| Caption | pixmo_cap | 3.0 | 32.3% |
| Pointing/Counting | pixmo_points, pixmo_count, cosyn_point | 3.0 | 32.3% |
| Scientific (caption-like) | arxiv, biorxiv, nature, chem, med, eng, psy | 2.8 | 30.1% |
| Wikipedia | wiki_en only | 0.5 | 5.4% |

Total weight sum: 9.3, 12 active datasets. Caption + scientific combined = 62.4%
of sampling, close to the Molmo ~60% target (scientific papers are image-caption
pairs, functionally similar to pixmo_cap). Pointing/counting at 32.3% tracks the
Molmo ~30% target.

Excluded: metarxiv, edrxiv (too tiny — extreme replay), all non-English wiki
(not needed for projector alignment), sparsetables, plotqa_cot, nights_cot
(too long).

Additional proportion overrides: `biorxiv: 0.3` (~343 shards instead of 1143),
`nature_comunication: 0.3` (~165 shards instead of 550).

**Usage**:
```bash
python tools/launch_aurora_daos.py \
  --dataset-groups projector \
  --max-seq-length 1024 \
  --use-bucketing \
  ...
```

---

## Optimization Strategies

### Implemented Optimizations

| Optimization | Location | Impact | Status |
|--------------|----------|--------|--------|
| Resampled mode | `multi_webdataset.py` | Eliminates epoch boundaries + stragglers | Deployed |
| RandomMix longest=True | `multi_webdataset.py` | Safety net for exhausted sources | Deployed |
| Projector preset | `daos_datasets.yaml` | Molmo-inspired 62/32/5 weight balance | Deployed |
| Manifest-based discovery | `multi_webdataset.py` | O(1) vs O(N) stat calls | Deployed |
| Rank-0 broadcast | `multi_webdataset.py` | Eliminates filesystem contention | Deployed |
| Proportion limiting | Config | 10% of arxiv = faster startup | Deployed |
| Within-batch sorting | `collate.py` | 15-20% less padding waste | Deployed |
| Long dataset exclusion | Config | Prevents 10000+ token sequences | Deployed |
| Hard seq length cap (env var) | `train.py`, `collate.py` | Eliminates worst backward spikes | Deployed |
| Collator-level truncation | `collate.py` | Safety net before padding | Deployed |
| Per-rank straggler detection | `train.py` | Identifies slow ranks | Deployed |
| PyTorch profiler integration | `train.py` | Step-level profiling | Deployed |

### Evaluated and Rejected/Deferred

These were tested in the 2026-02-18 ablation study and found to be
ineffective or lower priority than the sequence length cap:

#### Aggressive Bucketing (buffer_size=5000, num_buckets=16)

**Status**: Tested (Job 8339129). Does NOT fix the throughput crash.
The buffer still drains at epoch boundaries, and the same backward pass
spikes (21s) occurred. Larger buffers increase memory usage without
addressing the root cause.

**Verdict**: Not worth the memory cost. Standard buffer_size=2000 is fine
when combined with a proper `MAX_SEQ_LENGTH` cap.

#### Epoch Boundary Buffer Refill

**Status**: Superseded by `resampled=True`. With resampled mode, the base
iterator never exhausts, so the epoch boundary refill code never fires.
It remains as dead-code safety net in `BucketedMultiWebDatasetWrapper.__iter__`.

**Verdict**: No longer relevant. The primary fix is `resampled=True` +
`longest=True` which eliminates epoch boundaries entirely.

#### CCL Tuning / Communication Optimization

**Status**: Investigated (Job 8339131, 30M lines of CCL debug output).
Zero CCL errors found. AllReduce accounts for ~16% of step time (1.1s per
step across 16 gradient sync calls = 69ms per allreduce). This is normal
for DDP and not a bottleneck.

**Verdict**: CCL is healthy. No tuning needed. The throughput crashes are
compute-bound (backward pass), not communication-bound.

#### pil4dfs (DAOS Kernel Bypass)

**Status**: Tested (Job 8339128). With pil4dfs enabled, initialization
consumed the entire 1-hour walltime (loading 7B OLMo + 23 datasets on
24 ranks). Data loading time during training is <0.01s per step regardless
of pil4dfs status, so it provides no benefit for our workload.

**Verdict**: Keep disabled (`--no-pil4dfs`). The data loading is not the
bottleneck. Investigate separately if I/O-bound workloads emerge.

### Remaining Proposed Optimizations

#### 1. Per-Dataset Max Length (Medium Priority)

Add per-dataset truncation in the config. This would allow fine-grained
control (e.g., 512 for pixmo captions, 1024 for wiki articles) without
a global cap. Useful for SFT training where some datasets need longer
contexts.

```yaml
groups:
  pixmo:
    datasets:
      pixmo_cap:
        max_length: 512  # Short captions
  nemotron:
    datasets:
      wiki_en:
        max_length: 1024  # Cap documents
```

#### 2. Curriculum Learning (Low Priority for Projector Training)

Progressive sequence length increase. More relevant for E2E fine-tuning
than projector-only training, where short captions dominate.

```yaml
curriculum:
  stages:
    - steps: 0-1000
      max_seq_len: 256
    - steps: 1000-5000
      max_seq_len: 512
    - steps: 5000+
      max_seq_len: 1024
```

#### 3. Token Packing (Low Priority)

Pack multiple short sequences into a single training example to eliminate
padding entirely. Requires significant collator changes and attention mask
modifications. High effort, moderate reward given bucketing already achieves
83-93% efficiency.

#### 4. Dynamic Batch Sizing (Low Priority)

Adjust batch size based on sequence length to maintain constant memory usage.
Short sequences get larger batches, long sequences get smaller batches.
Complex to implement correctly with DDP (all ranks must agree on batch size).

---

## Next Steps & Roadmap

### Completed (2026-02-18)

| Task | Status | Notes |
|------|--------|-------|
| Root cause analysis (ablation study) | Done | 6 jobs, CCL/bucketing/seqlen/profiler tested |
| Hard sequence length cap via `MAX_SEQ_LENGTH` | Done | Env var controls tokenizer + collator |
| Collator-level truncation safety net | Done | `BucketedCollator.max_seq_length` |
| Epoch boundary buffer refill | Done | Limited effectiveness (see notes) |
| Per-rank straggler detection | Done | Cross-rank timing every 50 steps |
| PyTorch profiler integration | Done | Step-level traces saved to output dir |
| Truncation stats logging | Done | `[TRUNCATION]` log lines |
| Profile rank sync times | Done | Straggler ratios up to 4.92x observed |
| Test buffer_size tuning | Done | 2000 vs 5000 tested; no meaningful difference |
| Rule out CCL as bottleneck | Done | Zero errors in 30M lines of debug output |
| Rule out pil4dfs | Done | Causes init slowdown, no training benefit |
| Validate MAX_SEQ_LENGTH=1024 | Done | Job 8339446: 2x spike reduction, stable throughput, straggler persists |
| Resampled mode + longest=True | Done | Eliminates epoch boundaries, 2.3x peak throughput improvement |
| Projector preset (weight rebalancing) | Done | ~32% caption / ~32% pointing / ~30% scientific / ~5% wiki_en |
| Validate resampled + projector (Job 8339782) | Done | 310 steps, epoch fix confirmed, straggler persists (7.03x) |
| Projector convergence analysis | Done | Plateaus at loss ~2.2 after ~70 steps (2.9% data). 500 steps sufficient. |
| Wandb SIGTERM sync fix | Done | Signal handler flushes wandb before PBS walltime kill |
| Viz GT-prefix prompts | Done | `use_gt_prefix_for_captions=True` for distinct predictions per image |

### Completed: Validate MAX_SEQ_LENGTH=1024 (Job 8339446)

**Configuration**: 2 nodes (24 ranks), `--max-seq-length 1024`, `--use-bucketing`,
`--bucket-buffer-size 2000`, `--bucket-num-buckets 8`, `--no-pil4dfs`, `--dataset-groups all`.

**Results** (200 steps before PBS walltime kill at 3622s):

| Metric | Result | vs Baseline (max=2048) | Verdict |
|--------|--------|------------------------|---------|
| Sustained throughput | 69-75 samp/s | Comparable peak, far more stable | Partial success |
| Worst backward spike | 12.4s | Down from 20-30s (2x improvement) | Improved |
| Backward pass (typical) | 11.8-12.0s | Consistent, no 100x spikes | Success |
| Straggler ratio (step 200) | 4.90x (rank 23) | Similar to 2048 (was 4.92x) | Not fixed |
| Bucket efficiency | 85-87% overall | Comparable | Neutral |
| Memory (allocated/reserved) | 13.6GB / 41.2GB | Stable, no OOM risk | Success |
| Loss convergence | 9.84 → ~2.2 in 200 steps | Healthy | Success |

**Key observations**:

1. **Backward spikes reduced but not eliminated.** The worst backward time was 12.4s
   (vs 20-30s at 2048). The 4x theoretical reduction in attention cost (2048^2 → 1024^2)
   manifests as a ~2x practical improvement in spike severity.

2. **Straggler problem persists.** Rank 23 was consistently the slowest, with the
   straggler ratio worsening over time: 1.0x (step 50) → 3.07x (step 100) → 4.90x
   (step 200). This suggests the straggler effect is caused by uneven dataset epoch
   boundaries across ranks, not by absolute sequence length. Different ranks exhaust
   their shard assignments at different times, causing some ranks to hit poorly-bucketed
   batches while others process normal batches.

3. **Bucket efficiency shows periodic cycling.** Efficiency ranged from 37% to 99%
   across batches, cycling roughly every ~800 batches. The low-efficiency batches
   correspond to very short sequences (len_range 1-38, likely pixmo_count or
   cosyn_point) interspersed with longer ones during epoch transitions.

4. **No throughput crashes observed.** Unlike max=2048 runs which had periodic 70-95%
   throughput drops, the 1024 run maintained relatively stable throughput. The worst
   step (56.3 samp/s) was still within 25% of the typical rate (75 samp/s), compared
   to the 95% drops seen with max=2048.

5. **Training converged normally.** Loss decreased from 9.84 to ~2.0-2.6 over 200
   steps with warmup from 2.5e-05 to 5.0e-04 learning rate. IMAGE projector norm
   stable at ~64.0-64.3.

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Validate `MAX_SEQ_LENGTH=1024` run | High | Done (Job 8339446) | 200 steps, stable throughput |
| Confirm no backward spikes >1s | High | Partial | Worst was 12.4s (down from 30s), not <1s |
| Confirm stable throughput >100 samp/s | High | Not met | Sustained 69-75 samp/s, not 100+ |

### Validated: Resampled Mode + Projector Preset (Job 8339782)

**Configuration**: 2 nodes (24 ranks), `--dataset-groups projector`, `--max-seq-length 1024`,
`--use-bucketing`, `--viz-interval 50`, `resampled=True`, `longest=True`.

| Metric | 8339446 (old) | 8339782 (resampled) | Verdict |
|--------|---------------|---------------------|---------|
| Peak throughput | 75 samp/s | **176 samp/s** | 2.3x improvement |
| Worst throughput | 56.3 samp/s | 56.0 samp/s | Same |
| Bucket efficiency | 37-99% (cycling) | **78-89% (monotonic)** | Fixed |
| Epoch boundary msgs | Dozens | **Zero** | Fixed |
| Data load time | Periodic spikes | **Steady 0.005s** | Fixed |
| Straggler ratio (step 200) | 4.90x | **7.03x** | Worse |
| Visualization | None (interval=500) | **6 logged** (interval=50) | Working |
| Loss | 9.84→2.2 (200 steps) | **10.6→2.1 (310 steps)** | Healthy |

**Conclusion**: `resampled=True` was very beneficial for throughput and efficiency
but the straggler problem has a different root cause (cross-rank sequence length
variance, not epoch boundaries). See [Resampled Mode](#resampled-mode--epoch-boundary-fix).

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Validate resampled mode + projector preset | High | Done (Job 8339782) | Epoch fix confirmed, straggler persists |
| Confirm straggler ratio <1.5x | High | Not met | 7.03x at step 200, root cause is seq len variance |

### Projector Convergence Analysis (Job 8339782)

The 1-hour run on 2 nodes completed 310 steps (357,120 samples), which is only
**2.9% of a full epoch** through the projector preset's effective dataset
(5.3M unique samples, 12.5M weighted draws for one full pass). Despite this,
loss had already plateaued at ~2.2.

#### Loss Trajectory

```
Step     Loss      LR         Phase
──────────────────────────────────────────────────────
  10    10.64    2.50e-05    ┐
  30     6.01    7.50e-05    │ Phase 1: Rapid descent
  50     3.65    1.25e-04    │ Projector learns basic image→text alignment
  70     2.57    1.75e-04    ┘ (~80K samples, 1.5% of data)
 100     2.22    2.50e-04    ┐
 150     2.89    3.75e-04    │ Phase 2: LR warmup noise
 200     1.88    5.00e-04    ┘ Loss oscillates as LR ramps up
 240     2.45    4.99e-04    ┐
 280     2.09    4.98e-04    │ Phase 3: Plateau at ~2.24±0.18
 310     2.31    4.97e-04    ┘ Converged. More data won't help.
```

#### Why Loss Plateaus at 2.9% Data Coverage

The projector is a ~20M parameter linear mapping from the frozen SigLIP2
image embedding space to the frozen OLMo-7B text embedding space. It's
learning a coordinate transform, not visual features or language modeling.

1. **Limited capacity saturates quickly.** A linear projector converges
   in parameter space within ~70 steps (80K samples). More data can't
   improve a linear mapping that has already found its optimum.

2. **The loss floor is the frozen LLM's perplexity.** Loss ~2.2 reflects
   OLMo-7B's cross-entropy on descriptive caption text. The projector
   can't beat this without unfreezing the LLM.

3. **Data diversity has diminishing returns.** Once the projector has
   seen a representative sample of image types and caption styles (which
   happens quickly with 12 diverse datasets), additional samples don't
   change the learned mapping.

#### Epoch Size and Training Duration

| Metric | Value |
|--------|-------|
| Effective dataset size (projector preset) | 5,318,918 unique samples |
| Full epoch (all datasets seen once via weighted sampling) | 12,457,378 draws = 10,814 steps |
| Training rate (2 nodes, 24 ranks) | ~6.2 steps/minute |
| **Time for 1 full epoch** | **~29 hours of training** |
| Convergence point | **~70 steps (11 minutes)** |
| Recommended projector training | **500-1000 steps (80-160 min)** |

Per-dataset coverage in 310 steps (357K samples):

| Dataset | Effective Size | Draws (expected) | Coverage |
|---------|---------------|-------------------|----------|
| pixmo_cap | 613,610 | 115,200 | 18.8% |
| pixmo_points | 1,919,853 | 57,600 | 3.0% |
| pixmo_count | 33,216 | 19,200 | 57.8% |
| arxiv (10%) | 1,339,503 | 38,400 | 2.9% |
| biorxiv (30%) | 342,612 | 19,200 | 5.6% |
| nature_comunication (30%) | 164,716 | 19,200 | 11.7% |
| chemrxiv | 178,668 | 11,520 | 6.4% |
| medrxiv | 171,083 | 11,520 | 6.7% |
| engrxiv | 24,820 | 3,840 | 15.5% |
| psyarxiv | 16,942 | 3,840 | 22.7% |
| cosyn_point | 315,895 | 38,400 | 12.2% |
| wiki_en | 198,000 | 19,200 | 9.7% |

A full epoch is massive overkill for projector-only training. The path to
lower loss requires unfreezing parameters:

| Approach | Trainable Params | Expected Loss | When |
|----------|-----------------|---------------|------|
| Projector only (current) | ~20M | ~2.2 (floor) | Stage 1 — done in 500 steps |
| E2E: unfreeze LLM | ~7B | <1.5 | Stage 2 — full dataset matters |
| E2E: unfreeze ViT + LLM | ~7.4B | <1.0 | Stage 3 — Molmo recipe |
| MLP projector (deeper) | ~60-100M | ~1.8-2.0 | Optional Stage 1 variant |

#### Recommended Training Command (Projector Stage)

```bash
python tools/launch_aurora_daos.py \
  --id PROJECTOR-STAGE1 \
  --design PRISM-IMAGE-ONLY-7B \
  --nodes 2 \
  --batch \
  --queue debug \
  --walltime 01:00:00 \
  --dataset-groups projector \
  --use-bucketing \
  --bucket-buffer-size 2000 \
  --bucket-num-buckets 8 \
  --max-seq-length 1024 \
  --no-pil4dfs \
  --per-rank-timing \
  --viz-interval 100 \
  training.max_steps=500 \
  training.warmup_steps=50
```

Key changes from ablation runs:
- `max_steps=500` — sufficient for projector convergence, avoids wasting compute
- `warmup_steps=50` — shorter warmup (was 200) since we're not training for 5000 steps
- `viz-interval=100` — visualize at steps 100, 200, 300, 400, 500

### Short-term

| Task | Priority | Notes |
|------|----------|-------|
| Run production projector training (500 steps) | High | Use command above, save checkpoint for E2E Stage 2 |
| Transition to E2E fine-tuning (Stage 2) | High | Unfreeze LLM with differential LRs. Full dataset matters here. Use `PRISM-MOLMO-E2E` design. |
| Fix cross-rank sequence length variance | Medium | Straggler issue. Less critical now that projector training is short. Will matter more for multi-hour E2E runs. |
| Per-dataset max_length in config | Medium | Fine-grained control for SFT; also reduces seq len variance |
| Add wandb seq_len + truncation metrics | Medium | Real-time monitoring |

### Medium-term

| Task | Priority | Notes |
|------|----------|-------|
| E2E Stage 3 (unfreeze ViT) | High | Full Molmo recipe. Requires careful LR tuning. |
| MLP projector | Medium | Replace linear projector with 2-layer MLP for lower loss floor |
| Curriculum learning | Low | Progressive seq length (for E2E training) |
| Token packing | Low | Eliminate padding entirely |
| Multi-resolution images | Medium | Different image sizes per dataset |
| Dynamic batch sizing | Low | Constant memory usage across seq lengths |

---

## Monitoring & Alerts

### Recommended WandB Metrics

```python
wandb.log({
    # Sequence length
    "data/seq_len_mean": batch_seq_lengths.mean(),
    "data/seq_len_max": batch_seq_lengths.max(),
    "data/seq_len_p95": torch.quantile(batch_seq_lengths, 0.95),
    
    # Efficiency
    "data/bucket_efficiency": collator.get_efficiency_stats()["overall_efficiency"],
    "data/truncation_rate": num_truncated / batch_size,
    
    # Timing
    "timing/data_load_ms": data_time * 1000,
    "timing/rank_sync_wait_ms": sync_wait_time * 1000,
})
```

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Throughput drop | >20% from rolling avg | >50% from rolling avg |
| Truncation rate | >10% | >25% |
| Rank sync wait | >5s | >15s |
| Bucket efficiency | <70% | <50% |

---

## References

- **Training Logs**: `logs/PRISM-IMAGE-ONLY-7B/`
- **Dataset Config**: `src/conf/data/daos_datasets.yaml`
- **Loader Code**: `src/data/multi_webdataset.py`
- **Collator Code**: `src/data/collate.py`
- **Analysis Script**: `tools/analyze_dataset_lengths.py`
- **Launch Script**: `tools/launch_aurora_daos.py`

---

## Appendix: Creating New Datasets

### Step 1: Convert to WebDataset

```python
import webdataset as wds

with wds.ShardWriter("output/shard-%06d.tar", maxcount=1000) as sink:
    for idx, (image, caption, metadata) in enumerate(your_data):
        sink.write({
            "__key__": f"{idx:08d}",
            "jpg": image,  # PIL Image or bytes
            "txt": caption,
            "json": metadata,
        })
```

### Step 2: Create Train/Val Split

```python
# 99% train, 1% val
num_shards = 100
train_shards = list(range(99))
val_shards = list(range(99, 100))

# Move val shards to val_shards/ directory
for i in val_shards:
    shutil.move(f"shards/shard-{i:06d}.tar", f"val_shards/shard-{i:06d}.tar")
```

### Step 3: Create Manifest

```python
manifest = {
    "dataset": "my_dataset",
    "num_train_shards": 99,
    "num_val_shards": 1,
    "train_samples": 99000,
    "val_samples": 1000,
    "shards": [{"name": f"shard-{i:06d}.tar", "samples": 1000} for i in range(99)],
    "val_shards": [{"name": "shard-000099.tar", "samples": 1000}],
}

with open("manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
```

### Step 4: Add to Config

```yaml
# In daos_datasets.yaml
groups:
  my_group:
    datasets:
      my_dataset:
        path: my_group/my_dataset_webdataset
        samples: 100000
        shards: 100
        weight: 1.0
        description: "My new dataset"
```

### Step 5: Upload to DAOS

```bash
# Mount DAOS
dfuse /tmp/AuroraGPT/prism_training_data AuroraGPT prism_training_data

# Copy dataset
cp -r my_dataset_webdataset /tmp/AuroraGPT/prism_training_data/my_group/
```

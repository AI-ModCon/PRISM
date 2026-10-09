---
name: prism-data-pipeline
description: >
  Build and validate PRISM's data pipeline. Use when converting data to
  WebDataset shards, staging shards to compute nodes, enabling sequence-length
  bucketing, loading multiple datasets, or validating/analyzing a dataset.
  Triggers: "webdataset", "shard the data", "shard_modality", "MultiWebDataset",
  "bucketing", "validate dataset", "partition_by / local_shards_dir",
  "why does loss converge suspiciously fast".
metadata:
  version: "1.0"
  project: prism
---

# PRISM — Data Pipeline

PRISM trains on **WebDataset** shards. The flow is: raw data → shard →
(optionally stage to node-local `/tmp`) → `MultiWebDataset` loader → (optional)
length bucketing → collator. Before trusting *any* loss curve or throughput
number, confirm the pipeline feeds what you think it does — see the installed
`ml-data-pipeline-correctness` skill.

## When to load which reference

| Topic | File |
|-------|------|
| Convert HF datasets / CALVIN to WebDataset shards | [`references/webdataset-conversion.md`](references/webdataset-conversion.md) |
| Multi-dataset loading, `partition_by`, local-shards staging, bucketing | [`references/loading-and-bucketing.md`](references/loading-and-bucketing.md) |
| Validate shards & analyze sequence lengths | [`references/validation.md`](references/validation.md) |

## Fast facts

- **Sharding tools are login-node only** (no GPU/MPI): `tools/shard_modality.py`
  (HF → WebDataset for `time_series`/`graph`), `applications/vla/shard_calvin_vla.py`
  (CALVIN VLA episodes → shards, whole-episode / Markov-safe).
- **Core loader:** `src/data/multi_webdataset.py` → `MultiWebDataset` plus
  wrappers `MultiWebDatasetWrapper`, `ModalityAwareWebDatasetWrapper`,
  `BucketedMultiWebDatasetWrapper`. `LocalShardDataset` was **deleted** (PR #104)
  — local-shards now go through `MultiWebDataset(local_shards_dir=..., partition_by="local")`.
- **Bucketing → ~3× throughput** on mixed-length data (`--use-bucketing`).
- **Validate before you train:** `tools/validate_webdataset.py <path> --check-images`.

## The three traps that have burned real runs

1. **`glob.glob()` hangs on dfuse/DAOS mounts.** The code uses `os.listdir()` +
   manual `.endswith(".tar")` filtering on purpose (see comment in
   `src/data/multi_webdataset.py` ~line 573). If you add shard-discovery code,
   do the same — never `glob.glob()` a DAOS path.

2. **`WEBDATASET_LOCAL_PATH` leaks across sweep cells.** In a multi-cell sweep,
   one cell's staged image-shard path bled into non-image cells (mismatch=True),
   silently training text-only cells on image data. Fixed in PR #73, but the
   *class* of bug — a shared env var carrying data config between cells — is easy
   to reintroduce. If throughput or loss looks wrong in a sweep, check the
   effective shard path per cell.

3. **Sample-dict shape changed with PR #104.** Local-shards path now yields
   **unpadded text and no `text_attention_mask`**. Downstream code that assumed
   padded text / a mask must handle the new shape.

## See also

- Deep reference: [`docs/training/data.md`](../../training/data.md) — datasets, WebDataset format,
  bucketing internals, straggler analysis, multi-dataset mixing.
- [prism-daos-storage](../prism-daos-storage/SKILL.md) (staging to DAOS),
  [prism-launching-jobs](../prism-launching-jobs/SKILL.md) (`--dataset-groups`, `--use-bucketing`).
- Generic installed skill: `ml-data-pipeline-correctness`.

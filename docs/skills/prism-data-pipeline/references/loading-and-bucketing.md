# Loading, Partitioning & Bucketing

## Classes in `src/data/multi_webdataset.py`

| Class | Role |
|-------|------|
| `MultiWebDataset` | Core multi-source loader; distributes shards across ranks |
| `MultiWebDatasetWrapper` | Training wrapper (image→tensor, caption→tokens) |
| `ModalityAwareWebDatasetWrapper` | Per-modality tuple specs (`image`/`time_series`/`graph`/`vla`) |
| `BucketedMultiWebDatasetWrapper` | Adds sequence-length bucketing |
| `_DeterministicRandomMix` | Finite weighted mixer (no resampling) |

`LocalShardDataset` no longer exists — deleted in PR #104.

## `partition_by` and `local_shards_dir`

`MultiWebDataset.__init__` has two shard-distribution modes:

- **`partition_by="global"` (default):** shards divided across *all* ranks in
  `world_size`.
- **`partition_by="local"`:** shards divided across a node's *local* ranks
  (`LOCAL_RANK` / `LOCAL_WORLD_SIZE`). Use when each node holds a node-private
  subset in tmpfs.
- **`local_shards_dir=<dir>`:** read from a single flat local dir (staged `.tar`
  files + `local_manifest.json`). **Bypasses DAOS config entirely** and *forces*
  `partition_by="local"`.

The launcher wires these via `LOCAL_SHARDS_DIR` (see the `webdataset-staged`
backend, `--local-shards-dir`, default `/tmp/webdataset`).

## Bucketing (~3× throughput)

Enable with `--use-bucketing` (launcher) → `USE_BUCKETING=1`. Related knobs:
`--bucket-buffer-size` (default 2000), `--bucket-num-buckets` (default 8),
`--use-bucketed-collator` / `--no-bucketed-collator` (on by default).

Bucketing groups similar-length sequences to cut padding waste. It is
**orthogonal to IPEX varlen** (kernel swap vs padding reduction) — but varlen is
blocked for PRISM *training* (no autograd kernel); bucketing is the training-time
lever. See prism-distributed-strategy / `docs/results/scaling_study.md`.

## Multi-dataset mixing

`--dataset-groups` (comma-separated names or a preset like `projector`/`pixmo`),
`--dataset-config` (`src/conf/data/daos_datasets.yaml` or `lustre_datasets.yaml`),
`--dataset-proportions dataset1:0.1,dataset2:0.2` (fraction of shards per
dataset), `--finite-webdataset` (disable resampling). See `docs/training/data.md`.

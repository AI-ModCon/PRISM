# Validating & Analyzing Datasets

## `tools/validate_webdataset.py` — shard integrity (login node)

```bash
python tools/validate_webdataset.py /flare/<project>/prism_data/shards \
    --check-images --check text,image,pose,action
```

| Arg | Notes |
|-----|-------|
| `dataset_path` | positional; dataset root or shards dir |
| `--max-shards N` | only check the first N shards (fast smoke) |
| `--check-images` | run `PIL.verify` on images |
| `--check k1,k2,…` | keys to probe; `pose`→`pose.npy`, `action`→`action.npy` auto-expand; arbitrary `.npy` names allowed |

Exits non-zero if any required key (text/image + extras) is missing or fails to
decode. Run this after every conversion and before any training run.

## `tools/analyze_dataset_lengths.py` — token-length stats (compute node)

Diagnoses throughput differences that come from sequence-length distribution.
**Requires DAOS mounted** (run on a compute node).

```bash
python tools/analyze_dataset_lengths.py \
    --daos-mount /tmp/$USER/AuroraGPT/prism_training_data \
    --dataset-groups all \
    --samples-per-dataset 100 \
    --tokenizer allenai/OLMo-2-1124-7B-Instruct
```

Reports min/max/mean/median/p90/p99 token counts per dataset plus sample texts.
Use it to decide bucketing parameters and to explain why one dataset is slower
than another (longer sequences → more compute, more padding without bucketing).

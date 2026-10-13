# 2. Data

Use the numeric forecast corpus, not the text-only `SciTS-processed` split.
PRISM's converter writes `SciTS-forecast/` with `shards/`, `val_shards/`, and
`manifest.json`. Forecast examples contain a history array, a numeric target,
an observation mask, prompt text, and metadata.

## Convert SciTS

The source tree must contain `meta_data.jsonl` and the referenced series files.
Use an empty output directory on shared Lustre; keep it readable from the
Aurora compute nodes.

```bash
export PRISM_DATA_ROOT=/flare/<project>/$USER/data
export SCITS_ROOT=/path/to/SciTS

python applications/timeseries/convert_scits_to_webdataset.py \
    --scits-dir "$SCITS_ROOT" \
    --output-dir "$PRISM_DATA_ROOT/SciTS-forecast" \
    --dataset-name scits \
    --samples-per-shard 1000 \
    --val-ratio 0.10
```

The split is deterministic by sample ID. The converter keeps forecast targets
in `.gt_ts.npy` and their validity in `.gt_ts_mask.npy`; missing observations
are not treated as target values. The model preset supports up to ten target
channels. Do not use `--max-samples` for the production conversion.

## Check the converted dataset

```bash
test -s "$PRISM_DATA_ROOT/SciTS-forecast/manifest.json"
test -d "$PRISM_DATA_ROOT/SciTS-forecast/shards"
test -d "$PRISM_DATA_ROOT/SciTS-forecast/val_shards"
```

Pass the `SciTS-forecast` root to the Aurora WebDataset launcher. It validates
the manifest and stages training shards from Lustre to the compute nodes; the
held-out shard directory is made available to the trainer as well.

Next: [train the forecast model](training.md).
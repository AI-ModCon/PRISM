# WebDataset Conversion

Both sharding tools run on the **login node** (no GPU/MPI/Aurora deps) and write
256–512 MB WebDataset shards plus a manifest.

## `tools/shard_modality.py` — HF datasets → shards (time_series, graph)

```bash
python tools/shard_modality.py \
    --modality time_series \
    --hf-id ChatTSRepo/ChatTS-Training-Dataset \
    --hf-config align_256 \
    --hf-split train \
    --out-uri /flare/<project>/prism_data/chatts_ts \
    --shard-mb 384
```

| Arg | Notes |
|-----|-------|
| `--modality {time_series,graph}` | required |
| `--source {hf}` | only `hf` supported today |
| `--hf-id` | required; HF dataset repo id |
| `--hf-config`, `--hf-split` | optional config; split defaults to `train` |
| `--out-uri` | required; local/Lustre path (login node can't write `daos://`) |
| `--max-samples` | cap (smoke runs) |
| `--shard-mb` | target shard size, default 384 |

Output entries per sample: `<base>.text`, `<base>.<ext>` (`ts.npy` / `graph.pt`),
`<base>.meta.json`, plus a `shards.json` manifest consumed by `MultiWebDataset`.

## `tools/shard_calvin_vla.py` — CALVIN VLA episodes → shards

```bash
python tools/shard_calvin_vla.py \
    --root /flare/ModCon/sww/vla_training/calvin_dataset \
    --split train \
    --out /flare/<project>/prism_data/calvin_shards \
    --shard-mb 384 --jpeg-quality 92
```

- Reads a LeRobot root (`meta/` + `data/` parquet + JSONL).
- **Whole episodes only** — never split an episode across shards (Markov-safe);
  shard rollover happens at episode boundaries.
- Per-sample keys: `head.jpg`, `wrist.jpg`, `pose.npy`, `action.npy`,
  `instruction.txt`, `meta.json`.
- Output: `<out>/manifest.json` + `<out>/shards/calvin-NNNNNN.tar`.
- `--max-episodes` caps for smokes.

## After sharding

Always validate before training — see [`validation.md`](validation.md).

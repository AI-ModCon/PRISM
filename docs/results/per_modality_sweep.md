# Per-Modality Throughput Sweep

**Last updated**: May 26, 2026

The PRISM-MODALITY-SMOKE-1N design + `experiments/modality_presets.yaml`
text_{only,image,ts,graph,table,geometry} cells + per-cell
`dataset_overrides` form a one-command sweep over which modality the model
is asking for. Result is a CSV that, for each preset, shows samples/sec,
tokens/sec, what modalities the dataloader actually emitted, and whether
that matched what the model expected. The startup divergence check from
PR #69 + the per-modality columns from PR #71's
`tools/perf_aggregate.py --per-modality` mode are what make every row
self-validating.

VLA (CALVIN) is a separate `vla` preset against the
`PRISM-AURORA-ZONE-A-VLA-CALVIN-WEB-SMOKE` design, gated by a
bidirectional skip rule in `tools/run_sweep.py` so VLA-only data never
lands on the non-VLA trainer and vice versa. See the
[VLA section](#vla-cell) below for results.

## Running the sweep

### Prereqs

The `text_geometry` cell builds `GeometryEncoder`, which imports walrus +
hydra + the_well. Install once per venv before running the sweep on a
fresh environment:

```bash
pip install -e src/libs/walrus
pip install hydra-core the_well
```

If walrus install is broken (e.g. transitive C-extension failure on the
compute node) but you still want a throughput number for the sweep, set
`WALRUS_FALLBACK=1` — `GeometryEncoder.__new__` swaps in a flatten ->
linear -> d_geo `FallbackGeometryEncoder`. Output features are
meaningless; the variant exists so the sweep can produce a row instead
of a `FAIL` cell. Production runs MUST have walrus installed.

```bash
# 1. Hold a node interactively (debug queue, 1 node, Lustre)
qsub tools/hold_nodes.sh
# Wait for it to land. The script writes the assigned hostname(s) to
# logs/hold_nodefile.txt.

# 2. Drive the sweep through the unified runner (5 non-VLA cells)
python tools/run_sweep.py \
    --preset text_image,text_ts,text_graph,text_table,text_geometry \
    --designs PRISM-MODALITY-SMOKE-1N \
    --storage lustre \
    --sweep-id phase2-smoke-$(date +%Y%m%d-%H%M%S)

# 3. Add the VLA cell (separate design, requires shard prereq — see VLA section)
python tools/run_sweep.py \
    --preset vla \
    --designs PRISM-AURORA-ZONE-A-VLA-CALVIN-WEB-SMOKE \
    --storage lustre \
    --sweep-id <same id as above>

# 4. After all cells finish
python tools/perf_aggregate.py outputs/ --per-modality --sweep-id <id> > sweep.csv
```

`text_only` routes to a sibling variant — `PRISM-MODALITY-SMOKE-TEXTONLY-1N`
unfreezes the backbone so the cell has trainable parameters. The
bidirectional skip rule in `tools/run_sweep.py` routes `text_only` to
the TEXTONLY variant and routes the TEXTONLY variant only to `text_only`.

## Sweep results

### Phase 2 baseline (sweep_id `phase2-smoke-20260523-213110`, pre-leak-fix)

OLMo-1B, 1 node x4219c2s3b0n0, native DDP, BS=8, GAS=1, 50 steps,
`--no-pil4dfs`. All cells used `--webdataset-dir` pointing at
`pixmo_cap_webdataset` for the staging block. **Three of four
non-image cells silently consumed image shards (`mismatch=True`)** —
that's the WEBDATASET_LOCAL_PATH leak documented below. The reported
~349 samp/s for those rows is image throughput, not per-modality.

| preset        | samples/sec | tokens/sec | mismatch | notes                                       |
| ------------- | ----------- | ---------- | -------- | ------------------------------------------- |
| text_image    | 211.9       | 108K       | False    | honest baseline; loader and model agree     |
| text_ts       | 349.1       | 178K       | True     | loader silently fed image shards (leak)     |
| text_graph    | 351.2       | 180K       | True     | (same)                                      |
| text_table    | 345.0       | 176K       | True     | (same)                                      |
| text_geometry | FAIL        | —          | —        | Walrus encoder import error (pre-existing)  |
| text_only     | FAIL        | —          | —        | no trainable params under freeze-everything |

### Phase 3 verification (sweep_id `phase3-postfix-20260525-232521`, post-leak-gate)

OLMo-1B, 1 node x4117c4s1b0n0, same configuration, PBS job 8507247.
Both leak gates active: PR #73's gate in `src/data/multimodal.py` and
the symmetric gate at `src/train.py` legacy fast path (PR #90). With
the gates in place, non-image cells correctly refuse to load image
shards; they then surface a separate dataloader-dispatch bug (CSV
fallback unreachable, `preferred_source: local` not consistently
honored) and fail honestly instead of producing fake numbers.

| preset        | samples/sec | tokens/sec | mismatch | notes                                                 |
| ------------- | ----------- | ---------- | -------- | ----------------------------------------------------- |
| text_image    | 197.4       | 101K       | False    | post-fix baseline (within ~7% of Phase 2)             |
| text_ts       | FAIL        | —          | —        | leak gate fires correctly; loader-dispatch follow-up  |
| text_graph    | FAIL        | —          | —        | (same)                                                |
| text_table    | FAIL        | —          | —        | (same)                                                |
| text_geometry | FAIL        | —          | —        | Walrus encoder import error (pre-existing)            |

### Phase 4 implementation (post per-modality-sweep-dispatch fix)

Five primary fixes land together (post-PR #90):

1. **CSV/JSONL dispatch** — `src/data/multimodal.py:698`'s `elif os.path.isdir(...)`
   block was structurally unreachable. Folded into the outer `if not loaded_local:`
   chain. Datasets whose `local_path` is a directory of CSV/JSONL files
   (graph_captioning, table_reasoning, ts_qa, ts_instruction) now load
   locally instead of falling through to the offline-fatal HF remote loader.
2. **`_process_ts_instruction` handler** — the local TS reasoning JSONL
   (`/flare/ModCon/sandeep/PRISM/data/zone_a/ts_instruction`) has
   description/characteristics/series schema, not the q/a fields the legacy
   handler expected. The new handler synthesizes a (prompt, target) pair
   and delegates to `_process_ts_qa` for normalization + start/end-token
   envelope.
3. **`ts_qa` local-path** — `text_ts` preset now points `ts_qa` at
   `/flare/ModCon/ngetty/data/zone_a/ts_qa/align_256/train.jsonl` so the
   cell has two real local sources.
4. **`FallbackGeometryEncoder`** — gated by `WALRUS_FALLBACK=1`; only swaps
   in when `require_modality_deps(GEOMETRY)` raises. Keeps the text_geometry
   sweep cell runnable on environments where walrus is broken without
   silently downgrading production builds. The real fix is `pip install -e
   src/libs/walrus hydra-core the_well` — documented in
   [Running the sweep](#running-the-sweep).
5. **`PRISM-MODALITY-SMOKE-TEXTONLY-1N` variant** — sibling of
   `PRISM-MODALITY-SMOKE-1N` with `model.freeze_backbone: false`. The
   text_only cell routes to this variant instead of being skipped; a
   bidirectional route rule in `tools/run_sweep.py` mirrors the VLA pattern.

End-to-end validation surfaced four additional blocking bugs that also
land in this PR:

6. **`launch_aurora.py --design` flag** — sweep cells pass `--id <run-id>
   --design <design-id>` so the run id can vary per cell while the design
   id stays stable for experiment-yaml lookup. The DAOS launcher accepted
   `--design` since PR #71; the Lustre launcher didn't. Patched to parity.
7. **`run_sweep.py` data-group wiring** — the launcher invocation set
   `exp.preset=<name>` (a label) and `model.modalities=[...]`, but
   `+data=per_modality_smoke/<name>` (the Hydra config-group override that
   makes the per-cell `dataset_overrides` reachable to
   `_resolve_dataset_overrides`) was missing. Phase 2 masked the gap
   because the WEBDATASET_LOCAL_PATH leak fed image shards to every cell;
   PR #73 + #90 closed the leak and exposed the missing wiring.
8. **manifest.json JSONL false-positive** — once the dispatch fix made the
   CSV/JSONL branch reachable, WebDataset directories whose tar loader was
   disabled (e.g. `webdataset` not pip-installed in the active venv) fell
   into the JSONL branch and loaded `manifest.json` / `local_manifest.json`
   as training data. Filter both names from the JSONL candidate list.
9. **trainer ZeroDivisionError / final-step perf record** — the design
   sets `viz_every_n_steps=0` to disable visualization, but the trainer's
   `step % viz_interval == 0` then divided by zero. Three one-line guards.
   Separately, `if step % 50 == 0 and step > 0` never fired with
   `max_steps=50` because the loop exits at step 49 — added an
   `is_final_step` branch so perf records emit at least once per run.
10. **`HF_DATASETS_CACHE` propagation** — the launcher forwards `HF_HOME`
    to compute nodes but ignored `HF_DATASETS_CACHE`. When `load_dataset()`
    tries to write a lockfile to the default cache (often a read-only
    shared hub directory), the cell aborts with EACCES. Now optional
    via `.env`.

### Phase 4 validation sweep — all 6 cells honest

OLMo-1B, 1 node, native DDP, world_size=12, BS=8 (BS=2 for text_ts due
to ts_qa's 16×256 2D tensors), max_steps=50, `--use-shared-venv`,
`wandb.mode=offline`. Non-image cells use staged data on `/tmp/local_data`
via `scripts/stage_local_data.py`. Multiple sweep_ids
(`phase4-v9-125008` image; `phase4-v17-staged-160523` non-image;
`phase4-v24-geo-165128` geometry retry).

**The 5 "connector-warmup" cells** all share frozen backbone+encoders,
training only the per-modality projector. This is apples-to-apples for
the dataloader+collator+encoder code under test. Two text_image rows
to expose the I/O cost: staged uses `launch_aurora_web.py
--webdataset-dir <pixmo_cap_webdataset>` (tar shards staged to
`/tmp/webdataset` via `scripts/stage_shards.py`); Lustre-direct uses
`launch_aurora.py` (reads from `/flare/...` shards):

| preset                | samples/sec  | tokens/sec | data_s | fwd_s | bwd_s | notes                                            |
| --------------------- | ------------ | ---------- | ------ | ----- | ----- | ------------------------------------------------ |
| text_image (staged)   | **192.7**    | 98,676     | 0.6%   | 44%   | 55%   | baseline; tar shards on `/tmp/webdataset`        |
| text_image (Lustre)   | 26.1         | 5,898      | 57%    | 7%    | 34%   | data-bound; same training, shards on `/flare/...`|
| text_ts (staged)      | **9.15**     | 2,916      | 22%    | 11%   | 66%   | BS=2, 16×256 2D series, per-sample normalization |
| text_graph (staged)   | **172.7**    | 25,794     | 21%    | 20%   | 58%   | tiny graphs (128 nodes × 32), BS=8               |
| text_table (staged)   | **65.5**     | 47,862     | 24%    | 24%   | 51%   | staged JSONLs, Tapas tokenizer in collate        |
| text_geometry (staged)| **69.4**     | 1,939      | 9%     | 84%   | 7%    | walrus encoder dominates fwd; short PDE prompts  |

**text_only is a separate training mode** — under the same
frozen-backbone+encoders design it has 0 trainable params, so it
routes to the `PRISM-MODALITY-SMOKE-TEXTONLY-1N` variant which sets
`freeze_llm: false` and trains the full 1B-param LLM. That makes it
an E2E SFT smoke, not a connector-warmup smoke:

| preset    | samples/sec | tokens/sec | data_s | fwd_s | bwd_s | trainable params         |
| --------- | ----------- | ---------- | ------ | ----- | ----- | ------------------------ |
| text_only | **54.3**    | 21,341     | 8%     | 15%   | 74%   | LLM=114 (1B params) only |

`mismatch=false` and `batch_modality_counts={<modality>: BS}` for all
6 cells confirms dispatch + handlers succeeded.

#### Why text_image is 192.7 staged vs 26.1 Lustre-direct (7.4× gap)

Staging tar shards to `/tmp/webdataset` via `stage_shards.py` reduces
`data_s` from 57% to 0.6% of step time, recovering the Phase 3 baseline.
The Lustre-direct row is included as a control showing how much of the
"~200 samp/s historical baseline" is the staging path, not the training
mode. The leak gates from PR #73/#90 do NOT block staging — the gates
require `WEBDATASET_LOCAL_MODALITY` (default `"image"`) to match the
cell's modality, which is the correct behavior for image cells.

#### Why text_ts is 9.15 samp/s (vs text_graph's 172.7 or text_image staged at 192.7)

Three compounding causes:
1. **BS=2 not 8** — ts_qa records are 16-variate × 256-timestep 2D
   tensors. At BS=8 the XPU OOMs (`UR_RESULT_ERROR_OUT_OF_RESOURCES`).
   4× fewer samples per step than other cells.
2. **Per-sample 2D normalization in Python** — `_process_ts_qa`
   computes per-variate mean/std and injects them into the prompt
   template (`"... {ts_start_token}{ts_end_token} Mean: {m:.2f}, "
   "Std: {s:.2f} ..."` for each of 16 variates). This is pure
   Python overhead per sample.
3. **Backward dominates** — `bwd_s=66%` even though only the
   projector is trainable, because the underlying 1B-param LLM still
   needs gradient-enabled forward for the projector's contribution
   (gradient flows through the LLM to the projector input).

text_graph's tiny graphs (128 nodes × 32 features = 4096 floats per
graph, same total as ts_qa but no per-sample Python normalization)
runs at full BS=8 and skips the prompt-template overhead.

#### Why text_geometry has low tokens/sec (1,939 vs text_table's 47k)

The PDE prompt is just `"PDE Problem Statement: Predict evolution for
{params}. Initial Condition provided."` — ~28 tokens. Tokens/sec =
samples/sec × tokens/sample. text_geometry is the second-fastest cell
by samples/sec (69.4); the low tok/s reflects short prompt text, not
slow compute. fwd_s=84% is the walrus encoder doing real geometry
work (voxelization, GAT-style attention, decoding).

#### Staging workflow

The launcher's `launch_aurora_web.py --webdataset-dir <dir>` handles
WebDataset tar staging via `scripts/stage_shards.py`. For non-WebDataset
cells (JSONL/CSV/Arrow), use the new `scripts/stage_local_data.py`:

```bash
ssh <compute-node> 'python scripts/stage_local_data.py \
    --sources /flare/.../ts_qa/align_256 /flare/.../ts_instruction ... \
    --local-dir /tmp/local_data'
```

Then pass per-cell `++data.dataset_overrides.<dataset>.local_path=
/tmp/local_data/<basename>` Hydra overrides to the launcher. Each cell's
preset yaml already declares the on-Lustre `local_path`; the staged
override is per-run.

#### Open follow-ups

- **Pre-shard local data**: ts_qa (1 file), ts_instruction (3), 
  graph_captioning (1), geo_pde_synthetic (1) all have n_shards <
  world_size=12. The fix is HF's native `.shard()` works when
  n_shards ≥ world_size — split the JSONL/CSV/Arrow into ≥12 files
  on Lustre and drop the filter-strided fallback in
  `src/data/multimodal.py:2440`.
- **text_geometry uses synthetic data**: `/flare/.../geo_pde` is empty
  on the shared filesystem. The 200-record synthetic IC at
  `/flare/ModCon/ngetty/data/zone_a/geo_pde_synthetic_arrow` produces
  a real perf number but isn't physically meaningful. Populate the
  shared geo_pde dir with real PDEBench data when sourced.
- **GeometryEncoder input_dim hardcoded to 6**: synthetic IC was
  reshaped from (256,) to (256, 6) to match. Real PDEBench data
  with different feature count would need encoder generalization.

## Leak history: WEBDATASET_LOCAL_PATH (PR #73) + LOCAL_SHARDS_DIR (follow-up)

PR #71's startup divergence check surfaced the symptom (`mismatch=True`
on 3 of 4 non-image cells). The Aurora WebDataset launcher
(`tools/launch_aurora_web.py`) exports two staging paths when
`--webdataset-dir` is passed: `WEBDATASET_LOCAL_PATH` (consumed by
`StreamingMultimodalDataset` / `MultiWebDatasetWrapper`) and
`LOCAL_SHARDS_DIR` (consumed by `train.py`'s legacy fast path).
Pre-fix, both honored the override for ANY dataset hitting the code
path. PR #73 gated `WEBDATASET_LOCAL_PATH` on
`WEBDATASET_LOCAL_MODALITY` (default `"image"`); a follow-up adds the
symmetric gate at `train.py:776`. Both are required because the sweep
harness exercises the `train.py` fast path, not the
`StreamingMultimodalDataset` path PR #73 gated.

### Honest non-image numbers (Phase 4 fixes)

All four follow-ups in the prior revision of this section are now in
the codebase — see [Phase 4 implementation](#phase-4-implementation-post-per-modality-sweep-dispatch-fix)
above. `tools/shard_modality.py` is no longer required for the four
non-image cells; the dispatch fix loads CSV/JSONL local data directly.

## VLA cell

VLA (CALVIN) integrated into the per-modality data path in PR #76. The
trainer split between `trainer_zone_a.py` and `trainer_zone_a_vla.py` is
preserved; only the dataloader changes. Two loader paths are selectable
via `training.calvin_loader`:

- `map` (legacy `CalvinVLADataset` — pre-#76 baseline; to be removed in
  the VLA-5 follow-up after a 1-week clean soak)
- `webdataset` (new path through
  `ModalityAwareWebDatasetWrapper(modalities=["vla"])` over the shards
  written by `tools/shard_calvin_vla.py`)

### Live results (May 24, 2026 — 1 node x12 XPU, OLMo-1B, BS=1, GAS=1)

Both runs used Lustre, `--max-steps 20`, `--dist-strategy ddp`, wandb
offline. Throughput is `samples_per_sec` at step 10 (window mean —
step 0 is dominated by first-batch warmup and not comparable).

| preset / loader   | samples/sec | loss → step 10 | action_mse step 10 | calvin_loader | source PBS |
| ----------------- | ----------- | -------------- | ------------------ | ------------- | ---------- |
| vla / map         | 50.20       | 0.190 → 0.241  | 0.241              | map           | 8505834    |
| vla / webdataset  | 47.01       | 0.258 → 0.152  | 0.152              | webdataset    | 8505875    |

`loss == action_mse` for the VLA cell — the loss is the action regression
MSE; both columns are surfaced so the row schema matches the rest of the
per-modality sweep. Source artifacts:
`outputs/VLA-{MAP-BASELINE,WEB-SMOKE}-<HHMMSS>/.../checkpoints/perf.jsonl`
(the `<HHMMSS>` directory suffix is the launcher's local-time stamp, not
the PBS ID).

**Throughput delta**: webdataset path is ~6.4% slower than map at this
scale. Within the 10% acceptance gate from the integration plan, but
worth re-measuring once the shard set scales beyond 50 episodes / 15
shards (the smoke shards used here cover only a fraction of an epoch).

The per-step `action_mse_per_dim` column (7-dim CALVIN action) is
populated on every row; example from `8505875` step 10:

```
[0.092, 0.037, 0.660, 0.003, 0.104, 0.144, 0.022]
```

Dimensions 0/1/3/6 are the gripper Cartesian deltas (small magnitudes,
fast to fit); dim 2 is the gripper Z delta which dominates the loss
during early training. The aggregator surfaces `action_mse_per_dim` as
columns when the cell is `vla`.

### Known false positive: `mismatch=true` on VLA composite cell

The `startup_modality_check` record on the VLA webdataset cell emits
`mismatch=true` because:

- `dataloader_modalities` = `["vla"]` (the composite modality name)
- `model_modalities` = `["text", "image"]` (per the design override)

The composite `vla` modality expands at sample-emission time into
{image_head, image_wrist, pose, action, text}; pose and action are head
outputs (regression target + observation), not input modalities. The
mismatch is therefore expected, not a real data leak.

Follow-up: extend the startup check to special-case composite modalities
so this row reports `mismatch=false`. Tracked separately, not blocking
the sweep doc.

### Sharding prerequisite

The webdataset cell requires shards on disk before the sweep runs. From
a login node:

```bash
python tools/shard_calvin_vla.py \
    --root <calvin-dataset-root> \
    --split train \
    --out  /flare/ModCon/$USER/data/zone_a/vla_training/calvin_webdataset

python tools/validate_webdataset.py \
    /flare/ModCon/$USER/data/zone_a/vla_training/calvin_webdataset \
    --check pose,action,image,text
```

Episode-aligned sharding ⇒ the loader sets `shardshuffle=False` to keep
`(obs_t, action_{t+1})` Markov pairs intact. Don't repoint the calvin
group at non-episode-aligned shards.

## What the instrumentation covers (PRs #69, #71)

Each `perf.jsonl` record now carries:

- `tokens_per_sec` — non-pad text tokens × world_size / window time
- `tokens_per_batch` — non-pad text tokens × world_size / step (window mean)
- `batch_modality_counts` — per-step `{modality: n_samples}`; distinguishes
  "modality in model config" from "modality actually emitted in the batch"
- `sweep_id`, `preset` — propagated from `exp.sweep_id` / `exp.preset`
  Hydra overrides, used by `perf_aggregate.py --per-modality` to group
- `event: startup_modality_check` — one extra record at startup with
  `dataloader_modalities` vs `model_modalities` and a `mismatch` boolean

The `--per-modality` aggregator joins the per-step throughput records
against the startup-check record on `(run_dir, preset)`, so every CSV row
self-validates whether the dataloader and model agreed on what to feed.

## Files

- `experiments/modality_presets.yaml` — preset → modality list (`vla` added in PR #76)
- `experiments/prism_designs.yaml` — `PRISM-MODALITY-SMOKE-1N` (non-VLA), `PRISM-AURORA-ZONE-A-VLA-CALVIN-WEB-SMOKE` (VLA)
- `src/conf/data/per_modality_smoke/*.yaml` — per-preset `dataset_overrides` (skip ts_qa, enable the modality's source)
- `src/conf/data/{daos,lustre}_datasets.yaml` — `calvin` group declares the VLA shard location
- `tools/run_sweep.py` — unified sweep driver; bidirectional VLA skip rule (vla preset ↔ VLA designs)
- `tools/hold_nodes.sh` — PBS hold-node helper (debug-scaling queue)
- `tools/perf_aggregate.py --per-modality` — CSV rollup
- `tools/shard_modality.py` (PR #73) — shard creator for non-image modalities (text_ts/text_graph data)
- `tools/shard_calvin_vla.py` (PR #76) — shard creator for the VLA cell's CALVIN composite shards
- `tools/validate_webdataset.py` (PR #76) — pre-flight shard validator (gates the VLA sharder)

## Open follow-ups

Single source of truth for outstanding work tied to this doc:

- HF `IterableDataset` sharding fix for non-image cells (see Phase 4
  "Next steps" above). Without this, the dispatch fix can prove the
  modality-gate is right (mismatch=false, correct dataloader_modalities)
  but per-cell samples/sec is unmeasurable.
- Restage `pixmo_cap` to /tmp to recover text_image's pre-leak-gate
  ~197 samp/s baseline (the current 25.4 samp/s is honest but I/O-bound).
- Extend `event: startup_modality_check` to special-case composite
  modalities so the VLA cell reports `mismatch=false` (referenced from
  the [VLA cell](#vla-cell) section).
- Delete the legacy `calvin_loader=map` branch after a 1-week clean
  soak of `tools/parity/smoke_vla_web.sh` (VLA-5 in the 2026-05-24 VLA
  integration plan, since retired).
